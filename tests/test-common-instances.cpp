#include "common.h"
#include "llama.h"
#include "server-instances.h"

#include <functional>
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
    // 'latest' is a reserved routing token, never a legal name or group
    expect_parse_error("latest");                // reserved name
    expect_parse_error("latest:group=g");        // reserved name with a group
    expect_parse_error("foo:group=latest");      // reserved group
}

static void test_instances_parse_valid_names() {
    // control group: previously valid names stay valid, including near-misses
    // of the reserved token (the reservation is the exact string "latest")
    for (const char * spec : {
            "a",
            "swarm0:group=swarm:ctx=512",
            "ledger:ctx=512:pinned:default",
            "Latest",
            "latest1",
            "my-latest-thing",
        }) {
        const auto insts = common_instances_parse(spec);
        assert(insts.size() == 1);
    }
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

static void test_instances_lora_grammar() {
    // repeatable lora= with explicit and default scales
    auto v = common_instances_parse("a:lora=./x.gguf:0.5:lora=./y.gguf,b:lora=./z.gguf");
    assert(v.size() == 2);
    assert(v[0].lora.size() == 2);
    assert(v[0].lora[0].first == "./x.gguf" && v[0].lora[0].second == 0.5f);
    assert(v[0].lora[1].first == "./y.gguf" && v[0].lora[1].second == 1.0f);
    assert(v[1].lora.size() == 1);
    assert(v[1].lora[0].first == "./z.gguf" && v[1].lora[0].second == 1.0f);

    // a non-float component after lora= is not a scale: default applies, comp parses on
    auto w = common_instances_parse("a:lora=./x.gguf:pinned");
    assert(w.size() == 1 && w[0].lora.size() == 1);
    assert(w[0].lora[0].second == 1.0f && w[0].pinned);

    // no lora= means an empty list (inherit the base --lora set)
    auto p = common_instances_parse("plain:ctx=512");
    assert(p.size() == 1 && p[0].lora.empty());

    // invalid scales and paths are parse errors
    expect_parse_error("a:lora=");            // empty path
    expect_parse_error("a:lora=./x.gguf:0");  // zero scale
    expect_parse_error("a:lora=./x.gguf:-2"); // negative scale
    expect_parse_error("a:lora=./x.gguf:lora=./x.gguf"); // duplicate path
}

static void test_instance_params_lora() {
    common_params base;
    base.lora_adapters.push_back({ "base.gguf", 1.0f, "", "", nullptr });

    // empty inst.lora inherits the base set untouched
    common_instance inherit;
    inherit.name = "inherit";
    common_params p = common_instance_params(base, inherit);
    assert(p.lora_adapters.size() == 1);
    assert(p.lora_adapters[0].path == "base.gguf" && p.lora_adapters[0].scale == 1.0f);

    // non-empty inst.lora REPLACES the base set; ptrs stay null for the pool
    common_instance over;
    over.name = "over";
    over.lora = { { "./a.gguf", 0.5f }, { "./b.gguf", 2.0f } };
    p = common_instance_params(base, over);
    assert(p.lora_adapters.size() == 2);
    assert(p.lora_adapters[0].path == "./a.gguf" && p.lora_adapters[0].scale == 0.5f);
    assert(p.lora_adapters[1].path == "./b.gguf" && p.lora_adapters[1].scale == 2.0f);
    assert(p.lora_adapters[0].ptr == nullptr && p.lora_adapters[1].ptr == nullptr);
}

static void test_lora_fingerprint() {
    std::vector<common_adapter_lora_info> empty;
    assert(common_lora_fingerprint(empty).empty());

    // stable and order-independent; ptr is never an input
    std::vector<common_adapter_lora_info> a = {
        { "b.gguf", 1.0f, "", "", nullptr },
        { "a.gguf", 0.5f, "", "", (llama_adapter_lora *) 0x1234 },
    };
    std::vector<common_adapter_lora_info> b = {
        { "a.gguf", 0.5f, "", "", nullptr },
        { "b.gguf", 1.0f, "", "", nullptr },
    };
    assert(!common_lora_fingerprint(a).empty());
    assert(common_lora_fingerprint(a) == common_lora_fingerprint(b));

    // scale is part of the identity
    std::vector<common_adapter_lora_info> c = {
        { "a.gguf", 1.0f, "", "", nullptr },
    };
    std::vector<common_adapter_lora_info> d = {
        { "a.gguf", 0.5f, "", "", nullptr },
    };
    assert(common_lora_fingerprint(c) != common_lora_fingerprint(d));
}

static void test_adapter_buf_size_null() {
    assert(llama_adapter_lora_buf_size(nullptr) == 0);
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

// a demand-driven build followed by repeated start_loops() must leave exactly one
// scheduler thread: a second start would assign over a joinable thread and abort.
static void test_demand_build_starts_one_loop(const common_params & base) {
    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfg;
    cfg.name     = "solo";
    cfg.group    = "solo";
    cfg.ctx_size = 256;
    cfg.parallel = 1;
    params.instances.push_back(cfg);

    int builds = 0;
    server_instances mgr;
    mgr.set_context_builder([&builds, &mgr](server_instance & inst) {
        ++builds;
        return mgr.build_context_default(inst);
    });

    assert(mgr.load(params));
    assert(builds == 0);
    assert(!mgr.instances.front()->built);

    // a no-model handler forces the demand-driven build of the default instance
    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req req { {}, {}, "/props", "", "", {}, no_stop };
    auto res = mgr.handle_get_props(req);
    assert(res->status == 200);
    assert(builds == 1);
    assert(mgr.instances.front()->built);
    assert(mgr.instances.front()->loop_started);

    // repeated starts are no-ops: no second thread, no abort, no extra build
    mgr.start_loops();
    mgr.start_loops();
    assert(builds == 1);
    assert(mgr.instances.front()->loop_started);

    mgr.terminate();
}

// control: start_loops() on a pool of only unbuilt windows starts nothing.
static void test_start_loops_skips_unbuilt(const common_params & base) {
    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfg;
    cfg.name     = "cold";
    cfg.group    = "cold";
    cfg.ctx_size = 256;
    cfg.parallel = 1;
    params.instances.push_back(cfg);

    server_instances mgr;
    assert(mgr.load(params));
    assert(!mgr.instances.front()->built);

    mgr.start_loops();
    assert(!mgr.instances.front()->built);
    assert(!mgr.instances.front()->loop_started);
    assert(!mgr.instances.front()->ctx_server);
    assert(!mgr.instances.front()->loop_thread.joinable());

    mgr.terminate();
}

// resize is a manager-owned teardown + rebuild: success leaves a fresh window
// with exactly one scheduler, failure leaves a well-defined unbuilt window
// that a later demand retries at the new size.
static void test_resize_teardown_rebuild(const common_params & base) {
    common_params params = base;
    params.n_ctx      = 256;
    params.n_parallel = 1;
    params.warmup     = false;

    common_instance cfg;
    cfg.name     = "r";
    cfg.group    = "r";
    cfg.ctx_size = 256;
    cfg.parallel = 1;
    params.instances.push_back(cfg);

    int builds = 0;
    server_instances mgr;
    mgr.set_context_builder([&builds, &mgr](server_instance & inst) {
        ++builds;
        return mgr.build_context_default(inst);
    });
    assert(mgr.load(params));

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req props_req { {}, {}, "/props", "", "", {}, no_stop };
    auto resize_req = [&](int32_t n_ctx) {
        return server_http_req { { { "name", "r" } }, {}, "/instances/r/resize", "",
                                 safe_json_to_str({ { "ctx_size", n_ctx } }), {}, no_stop };
    };

    // demand-build the 256 window first
    assert(mgr.handle_get_props(props_req)->status == 200);
    assert(builds == 1);

    // successful resize: fresh 512 window, exactly one scheduler
    auto res = mgr.handle_post_instance_resize(resize_req(512));
    assert(res->status == 200);
    assert(json::parse(res->data)["n_ctx"].get<int>() == 512);
    assert(builds == 2);
    const auto inst = mgr.instances.front();
    assert(inst->built && inst->loop_started && inst->loop_thread.joinable());
    assert(inst->effective.n_ctx == 512);
    mgr.start_loops();
    mgr.start_loops();
    assert(builds == 2);
    assert(mgr.handle_get_props(props_req)->status == 200);

    // program the builder to fail: the resize 507s and leaves a clean unbuilt
    // window (null context/routes, no joinable thread) at the new size
    mgr.set_context_builder([](server_instance &) { return false; });
    res = mgr.handle_post_instance_resize(resize_req(1024));
    assert(res->status == 507);
    assert(!inst->built && !inst->ctx_server && !inst->routes);
    assert(!inst->loop_thread.joinable());
    assert(inst->cfg.ctx_size == 1024);
    assert(inst->effective.n_ctx == 1024);

    // a later demand retries at the new size and serves again
    mgr.set_context_builder(nullptr);
    assert(mgr.handle_get_props(props_req)->status == 200);
    assert(builds == 2);
    assert(inst->built && inst->loop_started && inst->loop_thread.joinable());
    assert(inst->effective.n_ctx == 1024);

    mgr.terminate();
}

// pool simulation: the pool loads the adapter once against the shared model, then
// hands the ptr to init_from_model. the bogus path proves the skip guard: if init
// tried to load-and-own again it would fail on the bogus file and return no context.
static void test_borrowed_model_adapter_skip(const common_params & base, const std::string & adapter_path) {
    common_params params = base;
    params.n_ctx = 256;
    params.n_parallel = 1;
    params.warmup = false;

    auto model_init = common_init_from_params(params, true);
    llama_model * model = model_init->model();
    assert(model != nullptr);

    // the pool's single load (outside init_from_model)
    llama_adapter_lora * pool_adapter = llama_adapter_lora_init(model, adapter_path.c_str());
    if (pool_adapter == nullptr) {
        fprintf(stderr, "WARNING: cannot load adapter '%s' on this model, skipping.\n", adapter_path.c_str());
        return;
    }
    assert(llama_adapter_lora_buf_size(pool_adapter) > 0);

    common_params p = params;
    p.lora_adapters = { { "/nonexistent/bogus.gguf", 0.5f, "", "", pool_adapter } };
    auto ctx = common_init_from_model_params(p, model);
    assert(ctx->context() != nullptr); // the bogus path was never touched
    assert(p.lora_adapters[0].ptr == pool_adapter);

    // teardown: the context never owned the adapter (no double-free below), the
    // pool frees its single load here
    ctx.reset();
    llama_adapter_lora_free(pool_adapter);
}

int main(int argc, char ** argv) {
    test_instances_parse_round_trip();
    test_instances_parse_errors();
    test_instances_parse_valid_names();
    test_instance_params();
    test_instances_lora_grammar();
    test_instance_params_lora();
    test_lora_fingerprint();
    test_adapter_buf_size_null();

    common_params params;
    std::string   adapter_path;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "-m") {
            params.model.path = argv[i + 1];
        }
        if (std::string(argv[i]) == "--lora") {
            adapter_path = argv[i + 1];
        }
    }
    // hermetic thread count: the -1 default resolves through the host backend
    // registry, which segfaults in some container toolchains
    params.cpuparams.n_threads       = 4;
    params.cpuparams_batch.n_threads = 4;

    if (params.model.path.empty()) {
        fprintf(stderr, "WARNING: no model file provided. Set LLAMACPP_TEST_MODELFILE=<gguf_model_path> to run the borrowed-model test.\n");
        return 0;
    }

    // params are built by hand here (no CLI parse), so resolve the thread
    // counts the same way parsing would; leaving -1 crashes context init
    if (params.cpuparams.n_threads < 0) {
        params.cpuparams.n_threads = common_cpu_get_num_math();
    }
    if (params.cpuparams_batch.n_threads < 0) {
        params.cpuparams_batch.n_threads = common_cpu_get_num_math();
    }

    ggml_backend_load_all();
    test_borrowed_model(params);
    test_demand_build_starts_one_loop(params);
    test_start_loops_skips_unbuilt(params);
    test_resize_teardown_rebuild(params);
    if (!adapter_path.empty()) {
        test_borrowed_model_adapter_skip(params, adapter_path);
    } else {
        fprintf(stderr, "WARNING: no adapter file provided. Pass --lora <adapter_gguf> to run the adapter skip test.\n");
    }

    fprintf(stdout, "%s: all tests passed\n", __func__);
    return 0;
}
