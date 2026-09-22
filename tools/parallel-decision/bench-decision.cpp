#include "decision-engine.h"

#include "common.h"
#include "json.h"
#include "llama.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static common_json load_fixture(const std::string & path) {
    return common_json::parse(read_file(path));
}

struct bench_opts {
    std::string model;
    std::string fixture     = "tests/fixtures/decision/contexts_schema.request.json";
    std::string mode        = "auto";   // comma-separated list is accepted
    std::string allow_cache = "true";   // "true"/"false", a comma list, or "both"
    std::string fork        = "auto";
    bool json_out           = false;
    bool bench              = false;
    int contexts            = 1;
    int iters               = 1;
};

static std::vector<std::string> split_list(const std::string & spec, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, sep)) {
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

// "true", "false", a comma/slash list, or "both".
static std::vector<bool> parse_cache_spec(const std::string & spec) {
    if (spec == "both") {
        return { false, true };
    }
    std::string s = spec;
    for (char & c : s) {
        if (c == '/' || c == '|') c = ',';
    }
    std::vector<bool> out;
    for (const std::string & tok : split_list(s, ',')) {
        if (tok == "true" || tok == "1" || tok == "on") out.push_back(true);
        else if (tok == "false" || tok == "0" || tok == "off") out.push_back(false);
    }
    if (out.empty()) out.push_back(true);
    return out;
}

static bench_opts parse_args(int argc, char ** argv) {
    bench_opts o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            o.model = argv[++i];
        } else if (a.rfind("--model=", 0) == 0) {
            o.model = a.substr(8);
        } else if (a == "--fixture" && i + 1 < argc) {
            o.fixture = argv[++i];
        } else if (a.rfind("--fixture=", 0) == 0) {
            o.fixture = a.substr(10);
        } else if (a == "--mode" && i + 1 < argc) {
            o.mode = argv[++i];
        } else if (a.rfind("--mode=", 0) == 0) {
            o.mode = a.substr(7);
        } else if (a == "--allow_cache" && i + 1 < argc) {
            o.allow_cache = argv[++i];
        } else if (a.rfind("--allow_cache=", 0) == 0) {
            o.allow_cache = a.substr(14);
        } else if (a == "--fork" && i + 1 < argc) {
            o.fork = argv[++i];
        } else if (a.rfind("--fork=", 0) == 0) {
            o.fork = a.substr(7);
        } else if (a == "--json") {
            o.json_out = true;
        } else if (a == "--bench") {
            o.bench = true;
        } else if (a == "--contexts" && i + 1 < argc) {
            o.contexts = std::stoi(argv[++i]);
        } else if (a == "--iters" && i + 1 < argc) {
            o.iters = std::stoi(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            std::cout << "usage: bench-decision --model <gguf> [--fixture <json>] [--mode auto,tree,greedy] [--allow_cache true,false] [--fork auto|copy|restore] [--contexts N] [--iters N] [--json] [--bench]\n";
            std::exit(0);
        }
    }
    if (o.model.empty()) {
        const char * env = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (env && env[0] != '\0') {
            o.model = env;
        }
    }
    return o;
}

static bool has_gpu_backend() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(ggml_backend_dev_get(i));
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU ||
            type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            return true;
        }
    }
    return false;
}

struct loaded {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    ~loaded() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
    bool load(const std::string & path, int n_seq_max = 10) {
        llama_backend_init();
        if (!has_gpu_backend()) {
            fprintf(stderr, "no GPU backend available; the decision benchmarks run on GPU only\n");
            return false;
        }
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = -1; // offload to the GPU backend when the build has one
        model = llama_model_load_from_file(path.c_str(), mp);
        if (!model) return false;
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 2048;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.n_seq_max = n_seq_max;
        cp.kv_unified = true;
        cp.swa_full = false;
        cp.n_outputs_max = 10;
        cp.n_outputs_max_per_seq = 1;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; // reproducible decisions on ROCm
        ctx = llama_init_from_model(model, cp);
        return ctx != nullptr;
    }
};

int main(int argc, char ** argv) {
    bench_opts bo = parse_args(argc, argv);

    // bench mode without explicit model uses env or tiny fallback via test-download-model path
    if (bo.bench && bo.model.empty()) {
        std::string fallback = "build-synthesis/tinyllamas/stories15M-q4_0.gguf";
        std::ifstream f(fallback);
        if (f) bo.model = fallback;
        else {
            fallback = "build/tinyllamas/stories15M-q4_0.gguf";
            std::ifstream g(fallback);
            if (g) bo.model = fallback;
        }
    }

    if (bo.bench) {
        // run a quick telemetry loop and emit bench-report.json
        if (bo.model.empty()) {
            std::cerr << "bench needs --model or LLAMA_DECISION_TEST_MODEL\n";
            return 2;
        }
        // Scoring is O(contexts), so the full {1,8,64} matrix takes ~18 min. The default stays
        // under ~40 s; ask for a larger tier explicitly with --contexts N.
        std::vector<int> ctx_counts;
        if (bo.contexts != 1) {
            ctx_counts = { bo.contexts };
        } else {
            ctx_counts = { 1, 4 };
        }

        loaded ld;
        if (!ld.load(bo.model)) {
            std::cerr << "failed to load model " << bo.model << "\n";
            return 1;
        }
        // fixture load
        common_json req = load_fixture(bo.fixture);
        if (!req.contains("schema") || !req.contains("contexts")) {
            std::cerr << "fixture must have schema and contexts\n";
            return 1;
        }
        auto cs = llama_decision::compile_schema(req.at("schema"), req.value("instructions", std::string("")));
        std::string ctx_text = req.at("contexts").at(0).get<std::string>();

        std::vector<std::string> modes = { "auto", "tree", "greedy" };
        if (bo.mode != "auto" || bo.bench == false) {
            // when bench not default, respect explicit mode filter
            if (bo.mode.find(',') != std::string::npos) {
                modes.clear();
                std::stringstream ss(bo.mode);
                std::string tok;
                while (std::getline(ss, tok, ',')) if (!tok.empty()) modes.push_back(tok);
            } else if (bo.mode != "auto") {
                bool is_bench_default = bo.mode == "auto" && bo.contexts == 1;
                if (!is_bench_default && bo.mode != "auto,tree,greedy") {
                    // single mode requested via --mode
                    if (modes.size() != 1 || modes[0] != bo.mode) {
                        // handled by earlier branch; keep single
                    }
                }
            }
            if (bo.bench && bo.mode == "auto") {
                // bench default: all three modes
                modes = { "auto", "tree", "greedy" };
            } else if (bo.mode != "auto") {
                modes = { bo.mode };
            }
        }

        common_json timings = common_json::array();
        for (int nc2 : ctx_counts) {
            std::vector<std::string> ctxs2;
            for (int i = 0; i < nc2; ++i) ctxs2.push_back(ctx_text);
            std::vector<std::string> tails2;
            for (auto & c : ctxs2) {
                auto p = llama_decision::render_prompt(nullptr, false, cs.system_text, c);
                tails2.push_back(p.second);
            }
            auto shared2 = llama_decision::render_prompt(nullptr, false, cs.system_text, ctx_text);
            for (auto & m : modes) {
                for (int ac = 0; ac < 2; ++ac) {
                    bool allow = ac == 1;
                    llama_decision::engine eng(ld.ctx, 2, 8);
                    llama_decision::options opt;
                    opt.mode = m;
                    opt.tree_max = 128;
                    opt.allow_cache = allow;
                    opt.fork = bo.fork;
                    // A cache-enabled cell needs one cold run plus at least one warm run to show a
                    // hit; a cache-disabled cell is one run. --iters raises both.
                    int iters = std::max(1, bo.iters);
                    if (allow && iters < 2) {
                        iters = 2;
                    }
                    std::vector<llama_decision::batch_result> runs;
                    for (int iter = 0; iter < iters; ++iter) {
                        runs.push_back(eng.decide_batch(shared2.first, tails2, cs.inputs, opt));
                    }
                    size_t pick = 0;
                    if (iters > 1) {
                        std::vector<double> totals;
                        for (int i = 1; i < iters; ++i) totals.push_back(runs[i].prefill_ms + runs[i].scoring_ms);
                        std::sort(totals.begin(), totals.end());
                        const double median = totals[totals.size() / 2];
                        double best = 1e18;
                        pick = 1;
                        for (int i = 1; i < iters; ++i) {
                            const double t = runs[i].prefill_ms + runs[i].scoring_ms;
                            if (std::fabs(t - median) < best) { best = std::fabs(t - median); pick = i; }
                        }
                    }
                    auto & br = runs[pick];
                    common_json e = common_json::object();
                    e["branch"] = "synthesis";
                    e["fixture"] = bo.fixture;
                    e["contexts"] = nc2;
                    e["mode"] = m;
                    e["fork"] = bo.fork;
                    e["cache"] = allow;
                    e["prefill_ms"] = br.prefill_ms;
                    e["scoring_ms"] = br.scoring_ms;
                    e["total_ms"] = br.prefill_ms + br.scoring_ms;
                    e["rounds"] = br.rounds;
                    e["rows"] = br.rows;
                    e["shared_tokens"] = (int) br.shared_tokens;
                    e["cache_hit"] = br.cache_hit;
                    e["prefill_cold_ms"] = runs[0].prefill_ms;
                    timings.push_back(e);
                }
            }
        }

        // write bench-report.json
        std::string out_path = "tests/decision-baseline/bench-report.json";
        common_json env = common_json::object();
        env["model"] = bo.model;
        // try to get file size
        std::ifstream mf(bo.model, std::ios::binary | std::ios::ate);
        long long bytes = mf ? (long long) mf.tellg() : -1;
        env["model_bytes"] = bytes;
        env["quantization"] = "Q4_0";
        common_json backend = common_json::object();
        backend["kv_unified"] = true;
        backend["swa_full"] = false;
        backend["n_ctx"] = 2048;
        backend["n_batch"] = 512;
        backend["n_ubatch"] = 512;
        backend["n_seq_max"] = 10;
        backend["gpu_layers"] = -1; // loaded::load offloads every layer to the GPU backend
        backend["flash_attn"] = false;
        env["backend_flags"] = backend;
        env["template_hash"] = "letter-v1";
        // git rev of this branch and of the decision baseline it is compared against
        auto git_rev_of = [](const char * ref) {
            std::string rev = "unknown";
            std::string cmd = std::string("git rev-parse ") + ref + " 2>/dev/null";
            FILE * pp = popen(cmd.c_str(), "r");
            if (pp) {
                char buf[128] = { 0 };
                if (fgets(buf, sizeof(buf), pp)) {
                    std::string s = buf;
                    s.erase(s.find_last_not_of(" \n\r\t") + 1);
                    if (s.size() == 40) rev = s;
                }
                pclose(pp);
            }
            return rev;
        };
        const std::string rev = git_rev_of("HEAD");
        env["git_rev"]           = rev;
        env["git_rev_synthesis"] = rev;
        env["git_rev_decision"]  = git_rev_of("_decision");

        common_json report = common_json::object();
        report["environment"] = env;
        report["timings"] = timings;
        // keep calibration snapshot placeholder
        std::ifstream cal_in("tests/decision-baseline/calibration.json");
        if (cal_in) {
            std::ostringstream ss; ss << cal_in.rdbuf();
            try { report["calibration_snapshot"] = common_json::parse(ss.str()); } catch (...) {}
        }

        std::ofstream out(out_path);
        out << report.dump(2) << "\n";
        std::cout << "wrote " << out_path << " with " << timings.size() << " entries\n";
        // also validate json
        return 0;
    }

    if (bo.model.empty()) {
        std::cerr << "need --model or LLAMA_DECISION_TEST_MODEL\n";
        return 2;
    }
    // handle contexts count
    const int nc = bo.contexts > 0 ? bo.contexts : 1;

    loaded ld;
    if (!ld.load(bo.model)) {
        std::cerr << "failed to load model " << bo.model << "\n";
        return 1;
    }

    common_json req = load_fixture(bo.fixture);
    auto cs = llama_decision::compile_schema(req.at("schema"), req.value("instructions", std::string("")));

    std::string base_ctx = req.at("contexts").at(0).get<std::string>();
    std::vector<std::string> ctxs;
    for (int i = 0; i < nc; ++i) ctxs.push_back(base_ctx);
    // render shared + tails
    auto shared_split = llama_decision::render_prompt(nullptr, false, cs.system_text, base_ctx);
    std::vector<std::string> tails;
    for (auto & c : ctxs) {
        auto p = llama_decision::render_prompt(nullptr, false, cs.system_text, c);
        tails.push_back(p.second);
    }

    std::vector<std::string> modes = split_list(bo.mode, ',');
    if (modes.empty()) {
        modes.push_back("auto");
    }
    for (const std::string & m : modes) {
        if (m != "auto" && m != "tree" && m != "greedy") {
            std::cerr << "mode must be auto, tree or greedy\n";
            return 2;
        }
    }
    const std::vector<bool> caches = parse_cache_spec(bo.allow_cache);

    common_json runs = common_json::array();
    for (const std::string & m : modes) {
        for (bool allow : caches) {
            llama_decision::engine eng(ld.ctx, 2, 8);
            llama_decision::options opt;
            opt.mode        = m;
            opt.tree_max    = 128;
            opt.allow_cache = allow;
            opt.fork        = bo.fork;

            auto br = eng.decide_batch(shared_split.first, tails, cs.inputs, opt);
            auto assembled = llama_decision::assemble(cs, br.items[0]);

            if (bo.json_out) {
                common_json out = common_json::object();
                out["mode"] = m;
                out["allow_cache"] = allow;
                out["decision"] = assembled.at("decision");
                out["fields"] = assembled.at("fields");
                common_json timings = common_json::object();
                timings["prefill_ms"] = br.prefill_ms;
                timings["scoring_ms"] = br.scoring_ms;
                timings["total_ms"] = br.prefill_ms + br.scoring_ms;
                timings["rounds"] = br.rounds;
                timings["rows"] = br.rows;
                timings["shared_tokens"] = (int) br.shared_tokens;
                timings["cache_hit"] = br.cache_hit;
                out["timings"] = timings;
                // also include raw probabilities for parity check
                common_json probs = common_json::object();
                for (size_t i = 0; i < cs.specs.size(); ++i) {
                    const auto & sp = cs.specs[i];
                    const auto & fr = br.items[0].fields[i];
                    common_json arr = common_json::array();
                    for (float p : fr.probs) arr.push_back((double) p);
                    probs[sp.name] = arr;
                }
                out["probabilities"] = probs;
                runs.push_back(out);
            } else {
                std::cout << "mode " << m << " cache " << (allow ? "true" : "false")
                          << " prefill_ms " << br.prefill_ms << " scoring_ms " << br.scoring_ms
                          << " total_ms " << (br.prefill_ms + br.scoring_ms)
                          << " rounds " << br.rounds << " rows " << br.rows
                          << " cache_hit " << (br.cache_hit ? 1 : 0) << "\n";
                std::cout << assembled.dump(2) << "\n";
            }
        }
    }

    if (bo.json_out) {
        if (runs.size() == 1) {
            std::cout << runs.at(0).dump(2) << "\n";
        } else {
            common_json wrapped = common_json::object();
            wrapped["runs"] = runs;
            std::cout << wrapped.dump(2) << "\n";
        }
    }
    return 0;
}
