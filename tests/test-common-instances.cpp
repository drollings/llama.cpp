#include "common.h"
#include "llama.h"

#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

#include <cstdio>
#include <cstdlib>

static void test_instances_parse_round_trip() {
    const std::vector<common_instance> instances = common_instances_parse(
        "swarm0:group=swarm:ctx=16384:parallel=1,"
        "swarm1:group=swarm:ctx=16384,"
        "ledger:ctx=65536:pinned:default,"
        "scratch:ctx=65536:pinned");

    assert(instances.size() == 4);

    const common_instance & swarm0 = instances[0];
    assert(swarm0.name == "swarm0");
    assert(swarm0.group == "swarm");
    assert(swarm0.ctx_size == 16384);
    assert(swarm0.parallel == 1);
    assert(!swarm0.pinned);
    assert(!swarm0.is_default);

    const common_instance & swarm1 = instances[1];
    assert(swarm1.name == "swarm1");
    assert(swarm1.group == "swarm");
    assert(swarm1.ctx_size == 16384);
    assert(swarm1.parallel == 0);
    assert(!swarm1.pinned);
    assert(!swarm1.is_default);

    const common_instance & ledger = instances[2];
    assert(ledger.name == "ledger");
    assert(ledger.group == "ledger"); // default group == name
    assert(ledger.ctx_size == 65536);
    assert(ledger.pinned);
    assert(ledger.is_default);

    const common_instance & scratch = instances[3];
    assert(scratch.name == "scratch");
    assert(scratch.group == "scratch");
    assert(scratch.ctx_size == 65536);
    assert(scratch.pinned);
    assert(!scratch.is_default);

    // round-trip: to_string then parse again must yield identical instances
    const std::string s = common_instances_to_string(instances);
    const auto reparsed = common_instances_parse(s);
    assert(reparsed.size() == instances.size());
    for (size_t i = 0; i < instances.size(); ++i) {
        assert(reparsed[i].name == instances[i].name);
        assert(reparsed[i].group == instances[i].group);
        assert(reparsed[i].ctx_size == instances[i].ctx_size);
        assert(reparsed[i].parallel == instances[i].parallel);
        assert(reparsed[i].is_default == instances[i].is_default);
        assert(reparsed[i].pinned == instances[i].pinned);
    }

    // empty spec yields no instances
    assert(common_instances_parse("").empty());
}

static void expect_parse_error(const std::string & spec) {
    bool threw = false;
    try {
        common_instances_parse(spec);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    if (!threw) {
        fprintf(stderr, "expected parse error for '%s'\n", spec.c_str());
        assert(false);
    }
}

static void test_instances_parse_errors() {
    expect_parse_error(":ctx=8");                // empty name
    expect_parse_error("foo:ctx=abc");           // non-integer ctx
    expect_parse_error("foo:ctx=-5");            // negative ctx
    expect_parse_error("foo:parallel=-1");       // negative parallel
    expect_parse_error("foo:unknown");           // unknown option
    expect_parse_error("foo::pinned");           // empty option
    expect_parse_error("foo:pinned=x");          // pinned takes no value
    expect_parse_error("foo:sleep=0");           // sleep is not supported in this branch
    expect_parse_error("foo,foo");               // duplicate instance name
    expect_parse_error("foo:group=bar,bar");     // group collides with instance name
    // name/group character class: rejected by common_instance_validate
    expect_parse_error("a=b:ctx=8");             // '=' in name
    expect_parse_error("a/b:ctx=8");             // '/' in name
    expect_parse_error("a b:ctx=8");             // space in name
    expect_parse_error("a:b:ctx=8");             // ':' in name (also a grammar separator)
    expect_parse_error("foo:group=a/b");         // '/' in group
    expect_parse_error("foo:group=a=b");         // '=' in group
}

static void test_instance_params() {
    common_params base;
    base.n_ctx = 1024;
    base.n_parallel = 4;

    common_instance inst;
    inst.name = "worker";
    inst.group = "swarm";

    // no overrides: inherit ctx, but parallel defaults to 1 (never the base value)
    common_params p = common_instance_params(base, inst);
    assert(p.n_ctx == 1024);
    assert(p.n_parallel == 1);

    // explicit overrides
    inst.ctx_size = 8192;
    inst.parallel = 2;
    p = common_instance_params(base, inst);
    assert(p.n_ctx == 8192);
    assert(p.n_parallel == 2);

    // zero ctx_size keeps the base value; zero parallel keeps 1
    common_params base2;
    base2.n_ctx = 2048;
    base2.n_parallel = 8;
    common_instance empty_inst;
    empty_inst.name = "bare";
    p = common_instance_params(base2, empty_inst);
    assert(p.n_ctx == 2048);
    assert(p.n_parallel == 1); // divergence from _swarm_api: never inherits base.n_parallel
}

static void test_borrowed_model(const common_params & base) {
    common_params params = base;
    params.n_ctx = 256;
    params.n_parallel = 1;
    params.warmup = false;

    auto model_init = common_init_from_params(params, true);
    llama_model * model = model_init->model();
    if (model == nullptr) {
        fprintf(stderr, "failed to load model\n");
        exit(1);
    }
    assert(model_init->context() == nullptr);

    // build two contexts from the SAME model with different ctx sizes
    // note: the KV cache pads n_ctx_seq to a multiple of 256
    common_instance inst_a;
    inst_a.name = "a";
    inst_a.ctx_size = 512;
    common_instance inst_b;
    inst_b.name = "b";
    inst_b.ctx_size = 1024;

    common_params p_a = common_instance_params(params, inst_a);
    common_params p_b = common_instance_params(params, inst_b);

    std::vector<common_init_result_ptr> ctxs;
    ctxs.push_back(common_init_from_model_params(p_a, model));
    ctxs.push_back(common_init_from_model_params(p_b, model));

    assert(ctxs[0]->context() != nullptr);
    assert(ctxs[1]->context() != nullptr);
    assert(ctxs[0]->model() == model); // borrowed: model() returns the shared model
    assert(ctxs[1]->model() == model);
    assert(llama_n_ctx(ctxs[0]->context()) == 512);
    assert(llama_n_ctx(ctxs[1]->context()) == 1024);

    // samplers are initialized on the borrowed path too
    assert(ctxs[0]->sampler(0) != nullptr);
    assert(ctxs[1]->sampler(0) != nullptr);

    // destroy the contexts; the model must survive
    ctxs.clear();

    const llama_vocab * vocab = llama_model_get_vocab(model);
    assert(llama_vocab_n_tokens(vocab) > 0);

    // a fresh context can still be built from the model
    auto ctx3 = common_init_from_model_params(p_a, model);
    assert(ctx3->context() != nullptr);
    assert(ctx3->model() == model);

    // model_init is destroyed last; the model is freed there
}

int main(int argc, char ** argv) {
    test_instances_parse_round_trip();
    test_instances_parse_errors();
    test_instance_params();

    common_params params;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "-m") {
            params.model.path = argv[i + 1];
        }
    }

    if (params.model.path.empty()) {
        fprintf(stderr, "WARNING: no model file provided. Set LLAMACPP_TEST_MODELFILE=<gguf_model_path> to run the borrowed-model test.\n");
        return 0;
    }

    ggml_backend_load_all();
    test_borrowed_model(params);

    fprintf(stdout, "%s: all tests passed\n", __func__);
    return 0;
}
