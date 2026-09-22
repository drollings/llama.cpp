#include "decision-engine.h"
#include "decision-protocol.h"
#include "labels.h"
#include "letter_readout.h"

#include "chat.h"
#include "common.h"
#include "json.h"
#include "llama.h"
#include "testing.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef DECISION_TEST_FIXTURE_DIR
#error "DECISION_TEST_FIXTURE_DIR must be defined by the build"
#endif

#ifndef DECISION_TEST_BASELINE_DIR
#error "DECISION_TEST_BASELINE_DIR must be defined by the build"
#endif

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void write_file(const std::string & path, const std::string & text) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
    out << text;
}

static bool file_exists(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    return (bool) in;
}

static void assert_close(testing & t, const std::string & msg, double expected, double actual, double eps = 1e-6) {
    t.assert_true(msg + " (expected " + std::to_string(expected) + ", got " + std::to_string(actual) + ")",
                  std::fabs(expected - actual) <= eps);
}

static std::string fixture_path(const std::string & name) {
    return std::string(DECISION_TEST_FIXTURE_DIR) + "/" + name;
}

static common_json fixture_request() {
    return common_json::parse(read_file(fixture_path("contexts_schema.request.json")));
}

// Deterministic view of the compiled schema: what the trie engine will score.
static common_json compiled_to_json(const llama_decision::compiled_schema & cs) {
    common_json out = common_json::object();
    out["system_text"] = cs.system_text;

    common_json specs = common_json::array();
    for (const auto & sp : cs.specs) {
        common_json s = common_json::object();
        s["name"]        = sp.name;
        s["type"]        = sp.type;
        s["description"] = sp.description;
        s["aggregate"]   = sp.aggregate;
        common_json enc  = common_json::array();
        for (const auto & e : sp.encoded) {
            enc.push_back(e);
        }
        s["encoded"] = enc;
        specs.push_back(s);
    }
    out["specs"] = specs;

    common_json inputs = common_json::array();
    for (const auto & in : cs.inputs) {
        common_json i    = common_json::object();
        i["suffix"]      = in.suffix;
        common_json cand = common_json::array();
        for (const auto & c : in.candidates) {
            cand.push_back(c);
        }
        i["candidates"] = cand;
        inputs.push_back(i);
    }
    out["inputs"] = inputs;
    return out;
}

static const char * jev_valid_body() {
    return R"({
      "model": "m",
      "state": "Customer was charged twice on May 3.",
      "questions": {
        "refund":  {"type": "noul",  "instructions": "Should this be refunded?",
                    "criteria": {"true": "yes", "false": "no"}},
        "dept":    {"type": "choice", "instructions": "What is the issue?",
                    "criteria": {"billing": "payment", "technical": "bug", "cancellation": "cancel"}},
        "urgency": {"type": "scale", "instructions": "How urgent?",
                    "criteria": [{"level": 1, "label": "calm"}, "upset", "furious"]}
      },
      "temperature": 1.0,
      "temperatures": {"noul": 1.0},
      "permutations": 1
    })";
}

static void test_compiled_schema_golden(testing & t) {
    t.test("compiled schema matches the committed golden", [](testing & t) {
        const common_json req = fixture_request();
        const auto cs = llama_decision::compile_schema(req.at("schema"), req.value("instructions", std::string("")));
        const std::string actual = compiled_to_json(cs).dump(2) + "\n";
        const std::string golden = read_file(fixture_path("compiled_schema.golden.json"));
        t.assert_equal("compiled schema is byte-identical", golden, actual);
    });
}

static void test_json_schema_form(testing & t) {
    t.test("JSON Schema properties form compiles and needs no descriptions", [](testing & t) {
        const common_json schema = common_json::parse(
            "{\"properties\":{\"a\":{\"type\":\"boolean\"},"
            "\"b\":{\"type\":\"string\",\"enum\":[\"x\",\"y\"]},"
            "\"c\":{\"type\":\"number\",\"minimum\":0,\"maximum\":1,\"multipleOf\":0.5}}}");
        const auto cs = llama_decision::compile_schema(schema, "note");
        t.assert_equal("three fields", (size_t) 3, cs.specs.size());
        t.assert_equal("enum type", std::string("enum"), cs.specs[1].type);
        t.assert_equal("number grid", (size_t) 3, cs.specs[2].encoded.size());
        t.assert_true("instructions kept", cs.system_text.find("note") != std::string::npos);
    });
}

static void expect_reject(testing & t, const std::string & schema_text, const std::string & needle) {
    try {
        const common_json schema = common_json::parse(schema_text);
        (void) llama_decision::compile_schema(schema, std::string());
        t.assert_true("schema is rejected: " + schema_text, false);
    } catch (const std::invalid_argument & e) {
        const std::string what = e.what();
        t.assert_true("reject reason contains needle: " + schema_text, what.find(needle) != std::string::npos);
    }
}

static void test_compile_rejects(testing & t) {
    t.test("invalid schemas are rejected with a clear reason", [](testing & t) {
        expect_reject(t, "[]", "must be an object");
        expect_reject(t, "{}", "1-32 fields");
        expect_reject(t, R"({"a":{"type":"boolean"}})", "needs a description");
        expect_reject(t, R"({"a":{"type":"enum","description":"d"}})", "list of choices");
        expect_reject(t, R"({"a":{"type":"integer","description":"d"}})", "minimum and maximum");
        expect_reject(t, R"({"a":{"type":"integer","minimum":0,"maximum":300,"description":"d"}})", "1-255 values");
        expect_reject(t, R"({"a":{"type":"number","minimum":0,"maximum":1,"description":"d"}})", "minimum, maximum and step");
        expect_reject(t, R"({"a":{"type":"number","minimum":0,"maximum":1,"step":0.3,"description":"d"}})", "grid must include both ends");
        expect_reject(t, R"({"a":{"type":"enum","choices":["x","x"],"description":"d"}})", "duplicate allowed values");
        expect_reject(t, R"({"a":{"type":"string","description":"d"}})", "supported types are");
        expect_reject(t, R"({"a":{"type":"boolean","aggregate":"mean","description":"d"}})", "aggregate must be mode");

        common_json wide = common_json::object();
        for (int i = 0; i < 33; ++i) {
            common_json f = common_json::object();
            f["type"]        = "boolean";
            f["description"] = "d";
            wide["f" + std::to_string(i)] = f;
        }
        try {
            (void) llama_decision::compile_schema(wide, std::string());
            t.assert_true("33 fields rejected", false);
        } catch (const std::invalid_argument & e) {
            t.assert_true("33 fields rejected with range reason", std::string(e.what()).find("1-32 fields") != std::string::npos);
        }
    });
}

static void test_render_prompt_fallback(testing & t) {
    t.test("raw prompt fallback splits system and dynamic parts", [](testing & t) {
        const auto split = llama_decision::render_prompt(nullptr, false, "SYS", "CTX");
        t.assert_equal("head", std::string("SYS\nContext:\n"), split.first);
        t.assert_equal("tail", std::string("CTX\nOutput:\n{\n"), split.second);
    });
}

static void test_assemble(testing & t) {
    t.test("assemble maps winners, probabilities and numeric aggregates", [](testing & t) {
        {
            const common_json schema = common_json::parse(R"({"mode":{"type":"boolean","description":"d"}})");
            const auto cs = llama_decision::compile_schema(schema, std::string());
            llama_decision::result r;
            r.fields.resize(1);
            // boolean candidate order is [true, false]
            r.fields[0].winner       = 0;
            r.fields[0].probs        = { 0.75f, 0.25f };
            r.fields[0].tree         = true;
            r.fields[0].scored_nodes = 1;
            const common_json out = llama_decision::assemble(cs, r);
            t.assert_equal("boolean decision", true, out.at("decision").at("mode").get<bool>());
            assert_close(t, "boolean probability", 0.75, out.at("fields").at("mode").at("probability").get<double>());
            t.assert_equal("scored nodes", (int) 1, out.at("fields").at("mode").at("scored_nodes").get<int>());
        }
        {
            const common_json schema = common_json::parse(
                R"({"level":{"type":"integer","minimum":1,"maximum":3,"description":"d"}})");
            const auto cs = llama_decision::compile_schema(schema, std::string());
            llama_decision::result r;
            r.fields.resize(1);
            r.fields[0].winner       = 2;
            r.fields[0].probs        = { 0.2f, 0.3f, 0.5f };
            r.fields[0].tree         = true;
            r.fields[0].scored_nodes = 1;
            const common_json out = llama_decision::assemble(cs, r);
            t.assert_equal("numeric mode value", 3, out.at("decision").at("level").get<int>());
            const auto interval = out.at("fields").at("level").at("interval_p10_p90");
            t.assert_equal("p10", 1, interval.at(0).get<int>());
            t.assert_equal("p90", 3, interval.at(1).get<int>());
        }
        {
            const common_json schema = common_json::parse(
                R"({"level":{"type":"integer","minimum":1,"maximum":3,"aggregate":"median","description":"d"}})");
            const auto cs = llama_decision::compile_schema(schema, std::string());
            llama_decision::result r;
            r.fields.resize(1);
            r.fields[0].winner       = 2;
            r.fields[0].probs        = { 0.2f, 0.3f, 0.5f };
            r.fields[0].tree         = true;
            r.fields[0].scored_nodes = 1;
            const common_json out = llama_decision::assemble(cs, r);
            t.assert_equal("median aggregate", 2, out.at("decision").at("level").get<int>());
        }
    });
}

static void test_jev_shape_contract(testing & t) {
    t.test("committed Jev envelope skeleton has the required keys", [](testing & t) {
        const common_json shape = common_json::parse(read_file(fixture_path("jev_basic.shape.json")));
        t.assert_true("request shape", shape.contains("request"));
        t.assert_true("response shape", shape.contains("response"));
        t.assert_true("answer shapes", shape.contains("answer_shapes"));
        const auto & answers = shape.at("answer_shapes");
        t.assert_true("noul shape", answers.contains("noul"));
        t.assert_true("choice shape", answers.contains("choice"));
        t.assert_true("score shape", answers.contains("score"));
        t.assert_true("choice probabilities", answers.at("choice").contains("probabilities"));
        t.assert_true("choice confidence", answers.at("choice").contains("confidence"));
        t.assert_true("score legend", answers.at("score").contains("legend"));
        const auto & usage = shape.at("response").at("usage");
        t.assert_equal("output_tokens is fixed at zero", 0, usage.at("output_tokens").get<int>());
    });
}

static void test_softmax(testing & t) {
    t.test("softmax normalizes, keeps the winner and flattens with temperature", [](testing & t) {
        const std::vector<float> logits = { -1.0f, 0.5f, 2.0f };
        const auto p = llama_decision::softmax(logits, 1.0f);
        double sum = 0.0;
        for (float x : p) {
            sum += x;
        }
        assert_close(t, "probabilities sum to one", 1.0, sum, 1e-6);
        t.assert_true("winner preserved", p[2] > p[1] && p[1] > p[0]);

        const auto flat = llama_decision::softmax(logits, 4.0f);
        t.assert_true("higher temperature flattens", flat[2] < p[2]);
        t.assert_true("empty input is handled", llama_decision::softmax({}).empty());
    });
}

static void test_question_temperature(testing & t) {
    t.test("effective temperature follows per-type override then global", [](testing & t) {
        const auto base = common_json::parse(R"({"state":"s","questions":{
            "a":{"type":"noul","instructions":"x"},
            "b":{"type":"choice","instructions":"x","criteria":{"p":null,"q":null}},
            "c":{"type":"score","instructions":"x","criteria":["lo","hi"]}}})");

        const auto plain = llama_decision::parse_jev_request(base);
        for (const auto & q : plain.questions) {
            assert_close(t, "global default", 1.0, llama_decision::question_temperature(plain, q));
        }

        common_json with_override = base;
        with_override["temperature"]   = 1.5;
        with_override["temperatures"]  = common_json::parse(R"({"noul":0.5,"choice":2.0})");
        const auto req = llama_decision::parse_jev_request(with_override);
        assert_close(t, "noul override", 0.5, llama_decision::question_temperature(req, req.questions[0]));
        assert_close(t, "choice override", 2.0, llama_decision::question_temperature(req, req.questions[1]));
        assert_close(t, "score falls back", 1.5, llama_decision::question_temperature(req, req.questions[2]));
    });
}

static void test_temperature_effect(testing & t) {
    t.test("temperature preserves the argmax and sharpens or flattens", [](testing & t) {
        const std::vector<float> logits = { -0.5f, 0.0f, 1.5f };
        const auto warm = llama_decision::softmax(logits, 0.5f);
        const auto mid  = llama_decision::softmax(logits, 1.0f);
        const auto cool = llama_decision::softmax(logits, 2.5f);

        auto argmax = [](const std::vector<float> & p) {
            return (int) (std::max_element(p.begin(), p.end()) - p.begin());
        };
        t.assert_equal("argmax invariant (sharp)", argmax(mid), argmax(warm));
        t.assert_equal("argmax invariant (flat)", argmax(mid), argmax(cool));
        t.assert_true("lower temperature sharpens", warm[argmax(warm)] > mid[argmax(mid)]);
        t.assert_true("higher temperature flattens", cool[argmax(cool)] < mid[argmax(mid)]);
    });

    t.test("assemble emits confidence and certainty consistent with the probabilities", [](testing & t) {
        const auto req = llama_decision::parse_jev_request(common_json::parse(jev_valid_body()));
        const std::vector<std::vector<float>> probs = {
            { 0.2f, 0.8f },
            { 0.5f, 0.3f, 0.2f },
            { 0.1f, 0.2f, 0.7f },
        };
        common_json usage = common_json::object();
        usage["output_tokens"] = 0;
        const common_json out = llama_decision::assemble_jev_response(req, probs, "m", usage);

        auto recompute = [](const std::vector<float> & p) {
            double h = 0.0;
            for (float x : p) {
                if (x > 0.0f) {
                    h -= (double) x * std::log((double) x);
                }
            }
            return 1.0 - h / std::log((double) p.size());
        };

        const auto & dept = out.at("answers").at("dept");
        t.assert_true("choice has confidence", dept.contains("confidence"));
        t.assert_true("choice has certainty", dept.contains("certainty"));
        assert_close(t, "choice confidence is max_p", 0.5, dept.at("confidence").get<double>(), 1e-6);
        assert_close(t, "choice certainty matches entropy", recompute(probs[1]), dept.at("certainty").get<double>(), 1e-6);

        const auto & urg = out.at("answers").at("urgency");
        t.assert_true("score has confidence", urg.contains("confidence"));
        t.assert_true("score has certainty", urg.contains("certainty"));

        t.assert_true("noul has no confidence", !out.at("answers").at("refund").contains("confidence"));
        t.assert_true("noul has no certainty", !out.at("answers").at("refund").contains("certainty"));
    });
}

static void test_temperature_profile(testing & t) {
    t.test("temperature provenance is enforced only for non-default values", [](testing & t) {
        const common_json doc = common_json::parse(R"({
            "temperatures": {"noul": 0.8, "choice": 1.3},
            "provenance": {"model": "m1", "quantization": "Q4_K", "template_hash": "t1", "backend_flags": "fa1"}
        })");
        const auto profile = llama_decision::parse_temperature_profile(doc);
        assert_close(t, "noul temperature parsed", 0.8, profile.temperatures.at("noul"));

        llama_decision::temperature_provenance current = profile.provenance;
        llama_decision::validate_temperature_profile(profile, current); // must not throw

        current.quantization = "Q8_0";
        bool refused = false;
        try {
            llama_decision::validate_temperature_profile(profile, current);
        } catch (const llama_decision::semantic_error &) {
            refused = true;
        }
        t.assert_true("mismatched provenance refuses non-default temperature", refused);

        const auto neutral = llama_decision::parse_temperature_profile(
            common_json::parse(R"({"temperatures":{"noul":1.0},"provenance":{"model":"other"}})"));
        llama_decision::validate_temperature_profile(neutral, current); // T=1 always allowed
    });

    t.test("bad temperature profiles are rejected", [](testing & t) {
        const char * bad[] = {
            R"([1,2,3])",
            R"({"temperatures": 5})",
            R"({"temperatures": {"speed": 2.0}})",
            R"({"temperatures": {"noul": 0}})",
            R"({"provenance": {"model": 7}})",
        };
        for (const char * text : bad) {
            bool threw = false;
            try {
                (void) llama_decision::parse_temperature_profile(common_json::parse(text));
            } catch (const llama_decision::semantic_error &) {
                threw = true;
            }
            t.assert_true(std::string("rejected: ") + text, threw);
        }
    });
}

static void test_confidence_never_gates(testing & t) {
    t.test("confidence tokens never share a line with gating logic", [](testing & t) {
        const char * files[] = {
            "decision-engine.h", "decision-engine.cpp",
            "decision-protocol.h", "decision-protocol.cpp",
            "labels.h", "labels.cpp",
            "letter_readout.h", "letter_readout.cpp",
        };
        const char * confidence_tokens[] = { "confidence", "certainty" };
        const char * gating_tokens[] = { "allow_cache", "cache_tag", "admit", "routing", "persist" };

        int violations = 0;
        for (const char * file : files) {
            std::ifstream in(std::string(DECISION_TEST_SOURCE_DIR) + "/" + file);
            std::string line;
            while (std::getline(in, line)) {
                bool has_conf = false;
                bool has_gate = false;
                for (const char * tok : confidence_tokens) {
                    has_conf = has_conf || line.find(tok) != std::string::npos;
                }
                for (const char * tok : gating_tokens) {
                    has_gate = has_gate || line.find(tok) != std::string::npos;
                }
                if (has_conf && has_gate) {
                    ++violations;
                }
            }
        }
        t.assert_equal("no line couples confidence with gating", 0, violations);
    });
}

static void test_prefix_tag(testing & t) {
    t.test("prefix tag is stable and changes with the prompt", [](testing & t) {
        const auto a = llama_decision::make_prefix_tag("sys", "after", "v1");
        const auto b = llama_decision::make_prefix_tag("sys", "after", "v1");
        t.assert_equal("tag is stable", a, b);
        t.assert_true("system text changes the tag", a != llama_decision::make_prefix_tag("sys2", "after", "v1"));
        t.assert_true("boundary changes the tag", a != llama_decision::make_prefix_tag("sys", "after2", "v1"));
        t.assert_true("version changes the tag", a != llama_decision::make_prefix_tag("sys", "after", "v2"));
    });
}

static void test_sequence_partition(testing & t) {
    t.test("sequence namespace is partitioned and contiguous", [](testing & t) {
        const common_json baseline = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/baseline.json"));
        t.assert_true("partition recorded", baseline.contains("partition"));
        const auto & p = baseline.at("partition");
        const long long slots_first  = p.at("slots_first").get<long long>();
        const long long slots_end    = p.at("slots_last_exclusive").get<long long>();
        const long long decision     = p.at("decision_first").get<long long>();
        const long long decision_n   = p.at("decision_count").get<long long>();
        const long long n_seq_max    = p.at("n_seq_max").get<long long>();
        t.assert_equal("slots start at zero", (long long) 0, slots_first);
        t.assert_equal("decision region follows slots", slots_end, decision);
        t.assert_equal("n_seq_max covers both regions", decision + decision_n, n_seq_max);
        t.assert_true("ranges are non-empty", slots_end > slots_first && decision_n >= 3);
    });
}

static std::string score_golden_path() {
    return fixture_path("contexts_schema.score.golden.json");
}

static std::string jev_golden_path() {
    return fixture_path("jev_letter.golden.json");
}

// Owns a loaded model, a decision-shaped context and the backend lifetime.
struct test_engine {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;

    ~test_engine() {
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_model_free(model);
        }
        if (model || ctx) {
            llama_backend_free();
        }
    }

    bool load(const char * path) {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        model = llama_model_load_from_file(path, mp);
        if (model == nullptr) {
            llama_backend_free();
            return false;
        }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx                 = 2048;
        cp.n_batch               = 512;
        cp.n_ubatch              = 512;
        cp.n_seq_max             = 10;
        cp.n_outputs_max         = 10;
        cp.n_outputs_max_per_seq = 1;
        cp.kv_unified            = true;
        cp.swa_full              = false;
        ctx = llama_init_from_model(model, cp);
        if (ctx == nullptr) {
            llama_model_free(model);
            model = nullptr;
            llama_backend_free();
            return false;
        }
        return true;
    }
};

// Runs the real fork/score substrate on a GGUF and returns the assembled decision.
// Throws on any setup or engine failure so callers can report it as a test failure.
static common_json run_fixture_scoring(const std::string & model_path,
                                       const llama_decision::compiled_schema & cs,
                                       const common_json & req) {
    const auto split = llama_decision::render_prompt(nullptr, false, cs.system_text,
                                                     req.at("contexts").at(0).get<std::string>());

    test_engine te;
    if (!te.load(model_path.c_str())) {
        throw std::runtime_error("model or context failed to load: " + model_path);
    }

    llama_decision::engine eng(te.ctx, 2, 8);
    llama_decision::options opt;
    llama_decision::batch_result br = eng.decide_batch(split.first, { split.second }, cs.inputs, opt);
    if (br.items.empty()) {
        throw std::runtime_error("engine returned no decisions");
    }
    return llama_decision::assemble(cs, br.items[0]);
}

static void expect_jev_reject(testing & t, const std::string & body_text, const std::string & needle) {
    try {
        const common_json body = common_json::parse(body_text);
        (void) llama_decision::parse_jev_request(body);
        t.assert_true("jev request is rejected: " + body_text, false);
    } catch (const llama_decision::semantic_error & e) {
        const std::string what = e.what();
        t.assert_true("reject reason contains needle: " + body_text + " -> " + what,
                      what.find(needle) != std::string::npos);
    }
}

// Deterministic Jev answers for a fixed score vector: a value golden that does not
// depend on any model weights.
static common_json jev_basic_from_fixed_scores() {
    const auto req = llama_decision::parse_jev_request(common_json::parse(jev_valid_body()));
    std::vector<std::vector<float>> probs = {
        { 0.25f, 0.75f },     // noul: options [false, true] -> noul = P(true)
        { 0.6f, 0.3f, 0.1f }, // choice: winner is "billing"
        { 0.2f, 0.3f, 0.5f }, // score: expected zero-based index
    };
    common_json usage = common_json::object();
    usage["input_tokens"]    = 12;
    usage["output_tokens"]   = 0;
    usage["cached_tokens"]   = 4;
    usage["state_cache_hit"] = false;
    return llama_decision::assemble_jev_response(req, probs, "m", usage);
}

static void test_jev_values_golden(testing & t) {
    t.test("fixed-score envelope matches the committed value golden", [](testing & t) {
        const std::string actual = jev_basic_from_fixed_scores().dump(2) + "\n";
        const std::string golden = read_file(fixture_path("jev_basic.golden.json"));
        t.assert_equal("value golden is byte-identical", golden, actual);
    });
}

static void test_jev_parse(testing & t) {
    t.test("valid Jev request parses with aliases and structured criteria", [](testing & t) {
        const common_json body = common_json::parse(jev_valid_body());
        t.assert_true("detected as Jev", llama_decision::is_jev_request(body));

        const auto req = llama_decision::parse_jev_request(body);
        t.assert_equal("model echoed", std::string("m"), req.model);
        t.assert_equal("three questions", (size_t) 3, req.questions.size());
        t.assert_equal("noul canonical", std::string("noul"), req.questions[0].type);
        t.assert_equal("noul two options", (size_t) 2, req.questions[0].options.size());
        t.assert_equal("choice canonical", std::string("choice"), req.questions[1].type);
        t.assert_equal("choice three options", (size_t) 3, req.questions[1].options.size());
        t.assert_equal("scale becomes score", std::string("score"), req.questions[2].type);
        t.assert_equal("score three levels", (size_t) 3, req.questions[2].options.size());
        assert_close(t, "temperature parsed", 1.0, req.temperature);
        t.assert_equal("permutations parsed", 1, req.permutations);
        t.assert_true("structured criterion kept",
                      req.questions[2].options[0].original.is_object());
    });

    t.test("invalid Jev requests are rejected with a clear reason", [](testing & t) {
        expect_jev_reject(t, R"({"questions":{"q":{"type":"noul","instructions":"x"}}})", "state is required");
        expect_jev_reject(t, R"({"state":"","questions":{"q":{"type":"noul","instructions":"x"}}})", "state must not be empty");
        expect_jev_reject(t, R"({"state":[],"questions":{"q":{"type":"noul","instructions":"x"}}})", "state must not be empty");
        expect_jev_reject(t, R"({"state":5,"questions":{"q":{"type":"noul","instructions":"x"}}})", "state must be a string");
        expect_jev_reject(t, R"({"state":"s","questions":{}})", "1-256 entries");
        expect_jev_reject(t, R"({"state":"s","questions":{"q":{"type":"mystery","instructions":"x"}}})", "unknown type");
        expect_jev_reject(t, R"({"state":"s","questions":{"q":{"type":"choice","criteria":{"a":"x"}}}})", "2-64 options");
        expect_jev_reject(t, R"({"state":"s","questions":{"q":{"type":"choice","instructions":"x"}}})", "choice needs an object");
        expect_jev_reject(t, R"({"state":"s","questions":{"q":{"type":"score","criteria":["only"]}}})", "2-64 levels");
        expect_jev_reject(t, R"({"state":"s","questions":{"q":{"type":"noul","criteria":[1,2]}}})", "noul criteria must be an object");
        expect_jev_reject(t, R"({"state":"s","questions":{"q":{"type":"noul"}}})", "needs instructions or criteria");
        expect_jev_reject(t, R"({"state":"s","questions":{"q":{"type":"noul","instructions":"x","extra":1}}})", "unknown field");
        expect_jev_reject(t, R"({"state":"s","questions":{"q":{"type":"noul","instructions":7}}})", "instructions must be a string");
        expect_jev_reject(t, R"({"state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"temperature":0})", "temperature must be > 0");
        expect_jev_reject(t, R"({"state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"permutations":0})", "permutations must be >= 1");
        expect_jev_reject(t, R"({"state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"temperatures":{"bogus":1}})", "unknown field");
        expect_jev_reject(t, R"({"state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"bogus":1})", "unknown field");

        common_json many = common_json::object();
        common_json q    = common_json::object();
        q["type"]        = "noul";
        q["instructions"] = "x";
        common_json qs   = common_json::object();
        for (int i = 0; i < 257; ++i) {
            qs["q" + std::to_string(i)] = q;
        }
        many["state"]     = "s";
        many["questions"] = qs;
        try {
            (void) llama_decision::parse_jev_request(many);
            t.assert_true("257 questions rejected", false);
        } catch (const llama_decision::semantic_error & e) {
            t.assert_true("257 questions rejected with range", std::string(e.what()).find("1-256") != std::string::npos);
        }
    });
}

static void test_jev_assemble(testing & t) {
    t.test("canonical envelope has the required shape and semantics", [](testing & t) {
        const auto req = llama_decision::parse_jev_request(common_json::parse(jev_valid_body()));
        common_json usage = common_json::object();
        usage["input_tokens"]    = 0;
        usage["output_tokens"]   = 0;
        usage["cached_tokens"]   = 0;
        usage["state_cache_hit"] = false;

        const common_json out = llama_decision::assemble_jev_response(req, {}, req.model, usage);
        t.assert_equal("model echoed", std::string("m"), out.at("model").get<std::string>());

        const auto & answers = out.at("answers");
        t.assert_true("answers keyed by qid", answers.contains("refund") && answers.contains("dept") && answers.contains("urgency"));

        const auto & refund = answers.at("refund");
        t.assert_equal("noul type", std::string("noul"), refund.at("type").get<std::string>());
        assert_close(t, "noul stub is 0.5", 0.5, refund.at("noul").get<double>());
        t.assert_true("noul never carries confidence", !refund.contains("confidence"));

        const auto & dept = answers.at("dept");
        t.assert_equal("choice winner is first", std::string("billing"), dept.at("choice").get<std::string>());
        assert_close(t, "choice probabilities sum to 1",
                     1.0,
                     dept.at("probabilities").at("billing").get<double>() +
                     dept.at("probabilities").at("technical").get<double>() +
                     dept.at("probabilities").at("cancellation").get<double>());
        assert_close(t, "choice confidence is max_p", 1.0 / 3.0, dept.at("confidence").get<double>(), 1e-6);
        assert_close(t, "choice certainty of uniform is 0", 0.0, dept.at("certainty").get<double>(), 1e-6);

        const auto & urg = answers.at("urgency");
        t.assert_true("score probability keys are strings \"0\"..\"K-1\"",
                      urg.at("probabilities").contains("0") && urg.at("probabilities").contains("1") && urg.at("probabilities").contains("2"));
        assert_close(t, "score is the expected zero-based index", 1.0, urg.at("score").get<double>());
        t.assert_true("legend has string keys", urg.at("legend").contains("0") && urg.at("legend").contains("2"));
        t.assert_true("legend round-trips the structured value",
                      urg.at("legend").at("0").is_object() && urg.at("legend").at("0").at("label").get<std::string>() == "calm");
        t.assert_equal("output_tokens is zero", 0, out.at("usage").at("output_tokens").get<int>());
    });
}

// Synthetic vocabulary for hermetic label tests. Greedy longest-piece matching,
// so a multi-piece string tokenizes to more than one token.
struct fake_vocab : llama_decision::label_vocab {
    std::vector<std::string>       pieces;
    std::map<int32_t, std::string> piece_override;
    std::set<int32_t>              specials;

    int32_t id_of(const std::string & p) const {
        for (size_t i = 0; i < pieces.size(); ++i) {
            if (pieces[i] == p) {
                return (int32_t) i;
            }
        }
        return -1;
    }

    std::vector<int32_t> tokenize(const std::string & text, bool) const override {
        std::vector<int32_t> out;
        size_t i = 0;
        while (i < text.size()) {
            int32_t best     = -1;
            size_t  best_len = 0;
            for (size_t id = 0; id < pieces.size(); ++id) {
                const std::string & p = pieces[id];
                if (!p.empty() && p.size() > best_len && text.compare(i, p.size(), p) == 0) {
                    best     = (int32_t) id;
                    best_len = p.size();
                }
            }
            if (best < 0) {
                out.push_back(100000 + (int32_t) (unsigned char) text[i]);
                ++i;
                continue;
            }
            out.push_back(best);
            i += best_len;
        }
        return out;
    }

    std::string piece(int32_t token) const override {
        const auto it = piece_override.find(token);
        if (it != piece_override.end()) {
            return it->second;
        }
        if (token >= 0 && (size_t) token < pieces.size()) {
            return pieces[(size_t) token];
        }
        return std::string();
    }

    bool is_special(int32_t token) const override {
        return specials.count(token) > 0;
    }
};

static fake_vocab make_fake_vocab(bool with_double_letters) {
    fake_vocab v;
    for (char c = 0x20; c < 0x7f; ++c) {
        v.pieces.push_back(std::string(1, c));
    }
    v.pieces.push_back("\n");
    if (with_double_letters) {
        for (char a = 'A'; a <= 'Z'; ++a) {
            for (char b = 'A'; b <= 'Z'; ++b) {
                v.pieces.push_back(std::string{ a, b });
            }
        }
    }
    v.pieces.push_back("Answer:");
    v.pieces.push_back("true");
    v.pieces.push_back("false");
    v.pieces.push_back("<|turn>");
    return v;
}

static void test_letter_suffix(testing & t) {
    t.test("question suffix lists options and ends at the answer boundary", [](testing & t) {
        const auto req = llama_decision::parse_jev_request(common_json::parse(jev_valid_body()));
        const fake_vocab v = make_fake_vocab(true);
        const auto pool = llama_decision::build_label_pool(v, 8);
        const std::string after = "<turn|>\n<turn>model\n";

        const std::string s = llama_decision::format_letter_suffix(req.questions[1], pool, after);
        t.assert_true("first option listed", s.find("A: payment") != std::string::npos);
        t.assert_true("second option listed", s.find("B: bug") != std::string::npos);
        t.assert_true("question text listed", s.find("What is the issue?") != std::string::npos);

        const std::string tail = after + "Answer:\n";
        t.assert_true("ends exactly at the answer boundary",
                      s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0);
    });

    t.test("label capacity is validated before scoring", [](testing & t) {
        const auto req = llama_decision::parse_jev_request(common_json::parse(jev_valid_body()));
        bool threw = false;
        try {
            llama_decision::validate_label_capacity(req, 2);
        } catch (const llama_decision::semantic_error &) {
            threw = true;
        }
        t.assert_true("three-option question needs three labels", threw);
        llama_decision::validate_label_capacity(req, 3); // must not throw
    });
}

static void test_label_pool(testing & t) {
    t.test("label pool keeps single-token round-tripping labels in order", [](testing & t) {
        const fake_vocab v = make_fake_vocab(true);

        const auto pool = llama_decision::build_label_pool(v, 64);
        t.assert_equal("cap respected", (size_t) 64, pool.size());
        t.assert_equal("first label is A", std::string("A"), pool[0].text);
        t.assert_equal("26th label is Z", std::string("Z"), pool[25].text);
        t.assert_equal("then two-letter labels", std::string("AA"), pool[26].text);

        bool all_alpha = true;
        for (const auto & l : pool) {
            for (char c : l.text) {
                all_alpha = all_alpha && (std::isalpha((unsigned char) c) != 0);
            }
        }
        t.assert_true("no numeric or symbol labels", all_alpha);

        t.assert_equal("AA is a single token", v.id_of("AA"), llama_decision::single_token(v, "AA"));
        t.assert_equal("AAA is not a single label", -1, llama_decision::single_token(v, "AAA"));
    });

    t.test("label pool rejects special and non-round-tripping tokens", [](testing & t) {
        fake_vocab special = make_fake_vocab(false);
        special.specials.insert(special.id_of("B"));
        t.assert_equal("special token rejected", -1, llama_decision::single_token(special, "B"));
        t.assert_equal("normal token accepted", special.id_of("A"), llama_decision::single_token(special, "A"));

        fake_vocab renamed = make_fake_vocab(false);
        renamed.piece_override[renamed.id_of("C")] = "c";
        t.assert_equal("round-trip failure rejected", -1, llama_decision::single_token(renamed, "C"));

        fake_vocab tiny;
        tiny.pieces.push_back("A");
        bool threw = false;
        try {
            (void) llama_decision::build_label_pool(tiny, 64);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        t.assert_true("fewer than two labels is an error", threw);
    });
}

static void test_boundary(testing & t) {
    t.test("boundary check accepts a clean split and rejects a merged token", [](testing & t) {
        const fake_vocab v = make_fake_vocab(false);
        const int32_t a    = v.id_of("A");

        t.assert_true("clean boundary passes", llama_decision::check_boundary(v, "x ", "A", a));

        fake_vocab merged = make_fake_vocab(false);
        merged.pieces.push_back("\nA");
        t.assert_true("merged token fails loudly", !llama_decision::check_boundary(merged, "x\n", "A", merged.id_of("A")));
    });
}

static void test_safe_data(testing & t) {
    t.test("safe_data breaks special-token injection", [](testing & t) {
        t.assert_equal("turn token", std::string("\\u003c|turn>model"), llama_decision::safe_data("<|turn>model"));
        t.assert_equal("media token", std::string("\\u003c__media__>"), llama_decision::safe_data("<__media__>"));
        t.assert_equal("reason marker untouched", std::string("{REASON: ignore}"), llama_decision::safe_data("{REASON: ignore}"));
        t.assert_equal("backticks untouched", std::string("`code`"), llama_decision::safe_data("`code`"));
        t.assert_equal("every angle bracket escaped", std::string("a \\u003c b \\u003c c"), llama_decision::safe_data("a < b < c"));
    });
}

static void test_label_pool_real(testing & t) {
    t.test("label pool and exact boundary ids on a real vocabulary", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }

        llama_backend_init();
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(path, mparams);
        if (model == nullptr) {
            llama_backend_free();
            t.assert_true("model loads", false);
            return;
        }

        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(model));
            const auto pool = llama_decision::build_label_pool(*vocab, 64);
            t.assert_true("pool has 2-64 labels", pool.size() >= 2 && pool.size() <= 64);

            bool single = true;
            bool alpha  = true;
            for (const auto & l : pool) {
                single = single && (llama_decision::single_token(*vocab, l.text) == l.token) && (vocab->piece(l.token) == l.text);
                for (char c : l.text) {
                    alpha = alpha && (std::isalpha((unsigned char) c) != 0);
                }
            }
            t.assert_true("every label is single-token and round-trips", single);
            t.assert_true("no numeric labels", alpha);

            const std::string prompt = "Answer:\n";
            const auto base = vocab->tokenize(prompt, true);
            const auto full = vocab->tokenize(prompt + pool[0].text, true);
            t.assert_equal("exact boundary length", base.size() + 1, full.size());
            t.assert_true("exact boundary tail is the label token", !full.empty() && full.back() == pool[0].token);
            t.assert_true("boundary accepts the label", llama_decision::check_boundary(*vocab, prompt, pool[0].text, pool[0].token));
        } catch (const std::exception & e) {
            t.assert_true(std::string("label pool runs: ") + e.what(), false);
        }

        llama_model_free(model);
        llama_backend_free();
    });
}

// Optional integration run: a real GGUF exercises the fork/score substrate.
// Without LLAMA_DECISION_TEST_MODEL the test reports a skip and the suite stays green.
// Scoring values depend on the weights, so the comparison is by tolerance and only runs
// when a matching scoring golden has been committed.
static void test_engine_integration(testing & t) {
    t.test("engine scores a fixed request against a real model", [](testing & t) {
        const char * model_path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (model_path == nullptr || model_path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        if (!file_exists(score_golden_path())) {
            t.skip("no committed scoring golden for this fixture");
            return;
        }

        const common_json req = fixture_request();
        const auto cs = llama_decision::compile_schema(req.at("schema"), req.value("instructions", std::string("")));

        common_json actual;
        try {
            actual = run_fixture_scoring(model_path, cs, req);
        } catch (const std::exception & e) {
            t.assert_true(std::string("engine runs: ") + e.what(), false);
            return;
        }

        const common_json golden = common_json::parse(read_file(score_golden_path()));
        for (const auto & e : golden.at("decision").items()) {
            t.assert_equal("decision/" + e.key(), e.value().dump(), actual.at("decision").at(e.key()).dump());
        }
        for (const auto & e : golden.at("fields").items()) {
            const double expected = e.value().at("probability").get<double>();
            const double got      = actual.at("fields").at(e.key()).at("probability").get<double>();
            assert_close(t, "probability/" + e.key(), expected, got, 5e-3);
        }
    });
}

static void test_letter_readout_real(testing & t) {
    t.test("letter readout scores a Jev request on a real model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }

        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const auto pool = llama_decision::build_label_pool(*vocab, 64);
            llama_decision::engine eng(te.ctx, 2, 8);
            const auto req = llama_decision::parse_jev_request(common_json::parse(jev_valid_body()));

            llama_decision::letter_metrics metrics;
            const auto probs = llama_decision::letter_readout(eng, *vocab, nullptr, false, req, pool,
                                                              llama_decision::options{}, &metrics);
            t.assert_equal("one distribution per question", req.questions.size(), probs.size());
            bool valid = true;
            for (size_t i = 0; i < probs.size(); ++i) {
                valid = valid && (probs[i].size() == req.questions[i].options.size());
                double sum = 0.0;
                for (float p : probs[i]) {
                    valid = valid && (p >= 0.0f);
                    sum += p;
                }
                valid = valid && (std::fabs(sum - 1.0) < 1e-4);
            }
            t.assert_true("every distribution is a valid probability vector", valid);

            // exact boundary for the letter readout's fallback tail
            const std::string prompt = "\nAnswer:\n";
            const auto base = vocab->tokenize(prompt, true);
            const auto full = vocab->tokenize(prompt + pool[0].text, true);
            t.assert_equal("production boundary length", base.size() + 1, full.size());
            t.assert_true("production boundary tail is the label",
                          !full.empty() && full.back() == pool[0].token);

            // exact boundary for the model's real chat template tail
            auto tmpls = common_chat_templates_init(te.model, "");
            if (tmpls) {
                const auto parts = llama_decision::render_letter_prompt(tmpls.get(), true,
                                                                        llama_decision::letter_system_text());
                const std::string tail = parts.second + "Answer:\n";
                t.assert_true("real template boundary accepts the label",
                              llama_decision::check_boundary(*vocab, tail, pool[0].text, pool[0].token));
            }

            // order stability: reversing the questions must not change the answers
            llama_decision::jev_request reversed = req;
            std::reverse(reversed.questions.begin(), reversed.questions.end());
            const auto probs_rev = llama_decision::letter_readout(eng, *vocab, nullptr, false, reversed, pool,
                                                                  llama_decision::options{}, nullptr);
            bool stable = probs_rev.size() == probs.size();
            for (size_t i = 0; stable && i < req.questions.size(); ++i) {
                const std::string & id = req.questions[i].id;
                size_t other = 0;
                while (other < reversed.questions.size() && reversed.questions[other].id != id) {
                    ++other;
                }
                stable = other < probs_rev.size();
                for (size_t k = 0; stable && k < probs[i].size(); ++k) {
                    stable = std::fabs(probs[i][k] - probs_rev[other][k]) < 5e-3;
                }
            }
            t.assert_true("question order does not change answers", stable);

            // temperature: sharpens or flattens without changing the winner
            auto with_temps = [](const char * temps) {
                common_json body = common_json::parse(jev_valid_body());
                body["temperatures"] = common_json::parse(temps);
                return llama_decision::parse_jev_request(body);
            };
            const auto sharp = llama_decision::letter_readout(eng, *vocab, nullptr, false,
                                                              with_temps(R"({"noul":0.5,"choice":0.5,"score":0.5})"),
                                                              pool, llama_decision::options{}, nullptr);
            const auto flat = llama_decision::letter_readout(eng, *vocab, nullptr, false,
                                                             with_temps(R"({"noul":2.5,"choice":2.5,"score":2.5})"),
                                                             pool, llama_decision::options{}, nullptr);
            auto top = [](const std::vector<float> & p) {
                return *std::max_element(p.begin(), p.end());
            };
            bool preserved = sharp.size() == probs.size() && flat.size() == probs.size();
            for (size_t i = 0; preserved && i < probs.size(); ++i) {
                preserved = preserved &&
                            std::distance(probs[i].begin(), std::max_element(probs[i].begin(), probs[i].end())) ==
                                std::distance(sharp[i].begin(), std::max_element(sharp[i].begin(), sharp[i].end())) &&
                            std::distance(probs[i].begin(), std::max_element(probs[i].begin(), probs[i].end())) ==
                                std::distance(flat[i].begin(), std::max_element(flat[i].begin(), flat[i].end())) &&
                            top(sharp[i]) + 1e-4 >= top(probs[i]) && top(probs[i]) + 1e-4 >= top(flat[i]);
            }
            t.assert_true("temperature preserves the winner and orders sharpness", preserved);

            if (file_exists(jev_golden_path())) {
                common_json usage = common_json::object();
                usage["input_tokens"]    = 0;
                usage["output_tokens"]   = 0;
                usage["cached_tokens"]   = 0;
                usage["state_cache_hit"] = false;
                const common_json actual = llama_decision::assemble_jev_response(req, probs, "m", usage);
                const common_json golden = common_json::parse(read_file(jev_golden_path()));
                for (const auto & e : golden.at("answers").items()) {
                    const auto & exp_a = e.value();
                    const auto & got_a = actual.at("answers").at(e.key());
                    if (exp_a.at("type").get<std::string>() == "noul") {
                        assert_close(t, "noul/" + e.key(), exp_a.at("noul").get<double>(), got_a.at("noul").get<double>(), 5e-3);
                    } else {
                        t.assert_equal("winner/" + e.key(),
                                       exp_a.at("type").get<std::string>() == "choice" ? exp_a.at("choice").dump()
                                                                                                        : exp_a.at("score").dump(),
                                       got_a.at("type").get<std::string>() == "choice" ? got_a.at("choice").dump()
                                                                                                        : got_a.at("score").dump());
                    }
                }
            }
        } catch (const std::exception & e) {
            t.assert_true(std::string("letter readout runs: ") + e.what(), false);
        }
    });
}

static void test_fork_real(testing & t) {
    t.test("copy and restore forks agree; bypass and LRU behave", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            const std::vector<llama_decision::field_input> two = {
                { "  \"a\": ", { "1", "2" } },
                { "  \"b\": ", { "x", "y" } },
            };
            const std::vector<llama_decision::field_input> one = { { "  \"a\": ", { "1", "2" } } };

            llama_decision::options oc;
            oc.fork      = "copy";
            oc.cache_tag = "t";
            const auto copy_run = eng.decide_batch("system", { "ctx" }, two, oc);

            llama_decision::options orr;
            orr.fork      = "restore";
            orr.cache_tag = "t";
            const auto restore_run = eng.decide_batch("system", { "ctx" }, two, orr);

            bool agree = copy_run.items[0].fields.size() == restore_run.items[0].fields.size();
            for (size_t f = 0; agree && f < copy_run.items[0].fields.size(); ++f) {
                const auto & pc = copy_run.items[0].fields[f].probs;
                const auto & pr = restore_run.items[0].fields[f].probs;
                agree = pc.size() == pr.size();
                for (size_t k = 0; agree && k < pc.size(); ++k) {
                    agree = std::fabs(pc[k] - pr[k]) < 1e-5;
                }
            }
            t.assert_true("copy and restore agree within tolerance", agree);

            // bounded LRU: two prefixes alternate and both hit on return
            llama_decision::options lru;
            lru.fork = "restore";
            lru.cache_tag = "A";
            const auto a1 = eng.decide_batch("system-A", { "ctx" }, two, lru);
            lru.cache_tag = "B";
            const auto b1 = eng.decide_batch("system-B", { "ctx" }, two, lru);
            lru.cache_tag = "A";
            const auto a2 = eng.decide_batch("system-A", { "ctx" }, two, lru);
            t.assert_true("first A misses", !a1.cache_hit);
            t.assert_true("first B misses", !b1.cache_hit);
            t.assert_true("returning to A hits within the bound", a2.cache_hit);

            llama_decision::options oc2;
            oc2.fork = "copy";
            const auto single = eng.decide_batch("system", { "ctx" }, one, oc2);
            const auto pair   = eng.decide_batch("system", { "ctx" }, two, oc2);
            const auto & ps = single.items[0].fields[0].probs;
            const auto & pp = pair.items[0].fields[0].probs;
            bool bypass_ok = ps.size() == pp.size();
            for (size_t k = 0; bypass_ok && k < ps.size(); ++k) {
                bypass_ok = std::fabs(ps[k] - pp[k]) < 1e-4;
            }
            t.assert_true("single-question bypass matches the forked path", bypass_ok);

            // prefix purity: branches never mutate the cached prefix, and branch sequences are released
            llama_memory_t mem = llama_get_memory(te.ctx);
            const llama_pos prefix_before = llama_memory_seq_pos_max(mem, 2);
            (void) eng.decide_batch("system", { "ctx" }, two, oc2);
            const llama_pos prefix_after = llama_memory_seq_pos_max(mem, 2);
            t.assert_equal("cached prefix is not mutated by branches", prefix_before, prefix_after);
            t.assert_true("branch sequences are released", llama_memory_seq_pos_max(mem, 3) <= 0);
        } catch (const std::exception & e) {
            t.assert_true(std::string("fork runs: ") + e.what(), false);
        }
    });
}

static void test_prefix_cache_coherence(testing & t) {
    t.test("prefix cache never hits across a changed identity", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            std::vector<llama_decision::field_input> fields = { { "  \"a\": ", { "1", "2" } } };

            llama_decision::options o1;
            o1.cache_tag = "tag-A";
            const auto b1 = eng.decide_batch("system", { "ctx" }, fields, o1);
            const auto b2 = eng.decide_batch("system", { "ctx" }, fields, o1);
            t.assert_true("first request misses", !b1.cache_hit);
            t.assert_true("same identity hits", b2.cache_hit);

            llama_decision::options o2;
            o2.cache_tag = "tag-B";
            const auto b3 = eng.decide_batch("system", { "ctx" }, fields, o2);
            t.assert_true("changed identity misses", !b3.cache_hit);
        } catch (const std::exception & e) {
            t.assert_true(std::string("cache coherence runs: ") + e.what(), false);
        }
    });
}

static void test_batching_waves(testing & t) {
    t.test("a wide batch packs into waves and keeps the answer order", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8); // small pool forces several waves

            std::vector<llama_decision::field_input> many;
            for (int i = 0; i < 64; ++i) {
                many.push_back({ "  \"f" + std::to_string(i) + "\": ", { "1", "2" } });
            }
            llama_decision::options o;
            o.cache_tag = "waves-many";
            const auto b = eng.decide_batch("system", { "ctx" }, many, o);
            t.assert_equal("one result per context", (size_t) 1, b.items.size());
            t.assert_equal("one field per question", (size_t) 64, b.items[0].fields.size());

            // order preservation: the first 8 answers match a standalone 8-question run
            std::vector<llama_decision::field_input> first(many.begin(), many.begin() + 8);
            llama_decision::options o2;
            o2.cache_tag = "waves-first";
            const auto b2 = eng.decide_batch("system", { "ctx" }, first, o2);
            bool same = b2.items[0].fields.size() == 8;
            double max_diff = 0.0;
            auto winner = [](const std::vector<float> & p) {
                return (int) (std::max_element(p.begin(), p.end()) - p.begin());
            };
            for (size_t f = 0; same && f < 8; ++f) {
                const auto & p = b.items[0].fields[f].probs;
                const auto & q = b2.items[0].fields[f].probs;
                same = p.size() == q.size() && winner(p) == winner(q);
                for (size_t k = 0; same && k < p.size(); ++k) {
                    max_diff = std::max(max_diff, (double) std::fabs(p[k] - q[k]));
                }
            }
            t.assert_true("batch size does not change the winner", same);
            t.assert_true("probabilities match within batch tolerance", max_diff <= 5e-2);
            t.assert_true("wide-batch order matches the standalone run", same);
        } catch (const std::exception & e) {
            t.assert_true(std::string("batching runs: ") + e.what(), false);
        }
    });
}

static void test_dedup_fields(testing & t) {
    t.test("identical fields score once and still answer in place", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            const llama_decision::field_input a = { "  \"a\": ", { "1", "2" } };
            const llama_decision::field_input b = { "  \"b\": ", { "x", "y" } };

            llama_decision::options o;
            o.cache_tag = "dedup";
            const auto dup = eng.decide_batch("system", { "ctx" }, { a, a, b, a }, o);

            t.assert_equal("duplicates still answer in place", (size_t) 4, dup.items[0].fields.size());
            bool equal = true;
            for (size_t f = 0; f < dup.items[0].fields.size(); ++f) {
                if (f == 2) {
                    continue;
                }
                equal = equal && dup.items[0].fields[f].probs == dup.items[0].fields[0].probs;
            }
            t.assert_true("duplicate fields share the first answer", equal);
            const auto unique = eng.decide_batch("system", { "ctx" }, { a, b }, o);
            t.assert_equal("dedup scores only the unique fields", unique.rows, dup.rows);
        } catch (const std::exception & e) {
            t.assert_true(std::string("dedup runs: ") + e.what(), false);
        }
    });
}

static void test_cancel_reaches_compute(testing & t) {
    t.test("a stop request aborts before scoring and never returns an answer", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            const std::vector<llama_decision::field_input> fields = {
                { "  \"a\": ", { "1", "2" } },
                { "  \"b\": ", { "x", "y" } },
            };
            int calls = 0;
            llama_decision::options o;
            o.cache_tag  = "cancel";
            o.should_stop = [&calls] { return ++calls > 2; };

            bool cancelled = false;
            try {
                (void) eng.decide_batch("system", { "ctx" }, fields, o);
            } catch (const llama_decision::cancelled_error &) {
                cancelled = true;
            }
            t.assert_true("cancel reached compute", cancelled);
            t.assert_true("no further check after the abort", calls >= 3);
        } catch (const std::exception & e) {
            t.assert_true(std::string("cancel run: ") + e.what(), false);
        }
    });
}

static void test_yield_points(testing & t) {
    t.test("the engine offers a cooperative yield between waves", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            std::vector<llama_decision::field_input> many;
            for (int i = 0; i < 64; ++i) {
                many.push_back({ "  \"f" + std::to_string(i) + "\": ", { "1", "2" } });
            }
            int yields = 0;
            llama_decision::options o;
            o.cache_tag = "yield";
            o.yield     = [&yields]() { ++yields; };
            (void) eng.decide_batch("system", { "ctx" }, many, o);
            t.assert_true("a wide decision yields at least once", yields >= 1);
        } catch (const std::exception & e) {
            t.assert_true(std::string("yield run: ") + e.what(), false);
        }
    });
}

static void test_capacity_error(testing & t) {
    t.test("an oversize suffix is rejected as a capacity error, not a crash", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            std::string huge;
            for (int i = 0; i < 900; ++i) {
                huge += "token" + std::to_string(i) + " ";
            }
            const std::vector<llama_decision::field_input> fields = { { huge, { "a1", "a2" } } };

            llama_decision::options o;
            o.mode      = "greedy";
            o.cache_tag = "capacity";
            bool rejected = false;
            try {
                (void) eng.decide_batch("system", { "ctx" }, fields, o);
            } catch (const llama_decision::capacity_error &) {
                rejected = true;
            }
            t.assert_true("oversize suffix raises capacity_error", rejected);
        } catch (const std::exception & e) {
            t.assert_true(std::string("capacity run: ") + e.what(), false);
        }
    });
}

static void test_prefix_hoist_cache(testing & t) {
    t.test("a long shared head prefills once and the next state hits the cache", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const auto pool = llama_decision::build_label_pool(*vocab, 64);
            llama_decision::engine eng(te.ctx, 2, 8);

            common_json body1 = common_json::parse(jev_valid_body());
            common_json body2 = common_json::parse(jev_valid_body());
            body2["state"] = "A different support ticket about a late delivery.";
            const auto req1 = llama_decision::parse_jev_request(body1);
            const auto req2 = llama_decision::parse_jev_request(body2);

            llama_decision::letter_metrics m1;
            llama_decision::letter_metrics m2;
            (void) llama_decision::letter_readout(eng, *vocab, nullptr, false, req1, pool, llama_decision::options{}, &m1);
            (void) llama_decision::letter_readout(eng, *vocab, nullptr, false, req2, pool, llama_decision::options{}, &m2);

            t.assert_true("the shared head is at least 32 tokens", m1.shared_tokens >= 32);
            t.assert_true("the first request prefills", !m1.cache_hit);
            t.assert_true("a changed state still hits the shared head cache", m2.cache_hit);
            t.assert_equal("both requests share the same head length", (size_t) m1.shared_tokens, (size_t) m2.shared_tokens);
        } catch (const std::exception & e) {
            t.assert_true(std::string("hoist run: ") + e.what(), false);
        }
    });
}

static void test_permutation_order(testing & t) {
    t.test("permutation passes are deterministic, distinct, and complete", [](testing & t) {
        const size_t k = 5;
        const auto identity = llama_decision::permutation_order(k, "q1", 0);
        t.assert_true("pass 0 is the identity", identity == (std::vector<size_t>{ 0, 1, 2, 3, 4 }));

        const auto a = llama_decision::permutation_order(k, "q1", 1);
        const auto b = llama_decision::permutation_order(k, "q1", 1);
        t.assert_true("the same seed gives the same order", a == b);
        t.assert_true("a later pass is not the identity", a != identity);

        std::vector<size_t> seen(k, 0);
        for (size_t i : a) {
            seen[i] += 1;
        }
        t.assert_true("the order is a permutation", std::all_of(seen.begin(), seen.end(), [](size_t c) { return c == 1; }));

        t.assert_true("a single option is never reordered", llama_decision::permutation_order(1, "q1", 1) == (std::vector<size_t>{ 0 }));
        t.assert_true("a two-option pass swaps", llama_decision::permutation_order(2, "q1", 1) == (std::vector<size_t>{ 1, 0 }));
    });
}

static void test_permutations_parsing(testing & t) {
    t.test("permutations are accepted and capped, never used on the default path", [](testing & t) {
        common_json body = common_json::parse(jev_valid_body());
        body["permutations"] = 2;
        t.assert_equal("two passes accepted", 2, llama_decision::parse_jev_request(body).permutations);

        body["permutations"] = 99;
        t.assert_equal("large values are capped, not rejected", 8, llama_decision::parse_jev_request(body).permutations);

        body["permutations"] = 0;
        bool threw = false;
        try {
            (void) llama_decision::parse_jev_request(body);
        } catch (const llama_decision::semantic_error &) {
            threw = true;
        }
        t.assert_true("zero passes is rejected", threw);

        common_json def = common_json::parse(jev_valid_body());
        t.assert_equal("default is one pass", 1, llama_decision::parse_jev_request(def).permutations);
    });
}

static void test_contract_hash(testing & t) {
    t.test("the contract hash pins tokenizer, template and label version", [](testing & t) {
        const std::string base = llama_decision::decision_contract_hash("m", "tmpl", 32000);
        t.assert_equal("is a sha256", (size_t) 64, base.size());
        t.assert_equal("deterministic", base, llama_decision::decision_contract_hash("m", "tmpl", 32000));
        t.assert_true("changes with the template", base != llama_decision::decision_contract_hash("m", "tmpl2", 32000));
        t.assert_true("changes with the model", base != llama_decision::decision_contract_hash("m2", "tmpl", 32000));
        t.assert_true("changes with the vocabulary", base != llama_decision::decision_contract_hash("m", "tmpl", 32001));
    });
}

static void test_docs_errors(testing & t) {
    t.test("every decision error code is documented", [](testing & t) {
        const std::string readme = read_file(std::string(DECISION_TEST_SOURCE_DIR) + "/README.md");
        t.assert_true("the readme is present", !readme.empty());
        for (const char * code : { "400", "401", "413", "422", "429", "499", "500", "501", "529" }) {
            t.assert_true(std::string("docs list HTTP ") + code, readme.find(code) != std::string::npos);
        }
    });
}

static void test_policy_confidence(testing & t) {
    t.test("policy: confidence never reaches the scorer and temperature needs provenance", [](testing & t) {
        const std::string engine = read_file(std::string(DECISION_TEST_SOURCE_DIR) + "/decision-engine.cpp");
        t.assert_true("read the engine source", !engine.empty());
        t.assert_true("the scorer never reads confidence", engine.find("confidence") == std::string::npos);
        t.assert_true("the scorer never reads certainty", engine.find("certainty") == std::string::npos);

        llama_decision::temperature_profile profile;
        profile.temperatures["choice"] = 1.3;
        profile.provenance.model         = "m";
        profile.provenance.template_hash = "t";

        llama_decision::temperature_provenance running;
        running.model          = "m";
        running.template_hash  = "t";
        running.backend_flags  = "";
        running.quantization   = "";
        bool accepted = true;
        try {
            llama_decision::validate_temperature_profile(profile, running);
        } catch (const std::exception &) {
            accepted = false;
        }
        t.assert_true("a matching provenance is accepted", accepted);

        running.template_hash = "other";
        bool refused = false;
        try {
            llama_decision::validate_temperature_profile(profile, running);
        } catch (const std::exception &) {
            refused = true;
        }
        t.assert_true("a stale provenance is refused", refused);

        llama_decision::temperature_profile defaulted;
        defaulted.temperatures["choice"] = 1.0;
        running.template_hash = "other";
        bool default_ok = true;
        try {
            llama_decision::validate_temperature_profile(defaulted, running);
        } catch (const std::exception &) {
            default_ok = false;
        }
        t.assert_true("a default temperature needs no provenance", default_ok);
    });
}

static void test_head_capability(testing & t) {
    t.test("head capability is resolved once and stays stable", [](testing & t) {
        const auto & a = llama_decision::selected_head_capability();
        const auto & b = llama_decision::selected_head_capability();
        t.assert_true("capability is a single shared object", &a == &b);
        t.assert_true("selected head is unavailable in this build", !a.available);
        t.assert_true("a fallback reason is given", !a.reason.empty());
    });
}

static void test_head_fallback_equivalence(testing & t) {
    t.test("head auto falls back to full logits and the readout exposes logits plus probs", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const auto pool = llama_decision::build_label_pool(*vocab, 64);
            llama_decision::engine eng(te.ctx, 2, 8);
            const auto req = llama_decision::parse_jev_request(common_json::parse(jev_valid_body()));

            llama_decision::options oa;
            oa.cache_tag = "head-auto";
            llama_decision::letter_metrics ma;
            llama_decision::answer_audit   aa;
            const auto pa = llama_decision::letter_readout(eng, *vocab, nullptr, false, req, pool, oa, &ma, &aa);

            llama_decision::options of;
            of.cache_tag = "head-full";
            llama_decision::letter_metrics mf;
            llama_decision::answer_audit   af;
            const auto pf = llama_decision::letter_readout(eng, *vocab, nullptr, false, req, pool, of, &mf, &af);

            bool same = pa.size() == pf.size();
            for (size_t qi = 0; same && qi < pa.size(); ++qi) {
                same = pa[qi].size() == pf[qi].size();
                for (size_t i = 0; same && i < pa[qi].size(); ++i) {
                    same = std::fabs(pa[qi][i] - pf[qi][i]) < 1e-6;
                }
            }
            t.assert_true("auto and full heads agree", same);

            bool logits_ok = af.option_logits.size() == pf.size();
            for (size_t qi = 0; logits_ok && qi < pf.size(); ++qi) {
                const auto & z = af.option_logits[qi];
                logits_ok = z.size() == pf[qi].size();
                if (!logits_ok) {
                    break;
                }
                float mx = *std::max_element(z.begin(), z.end());
                double sum = 0.0;
                for (float x : z) {
                    sum += std::exp((double) (x - mx));
                }
                for (size_t i = 0; logits_ok && i < z.size(); ++i) {
                    const double p = std::exp((double) (z[i] - mx)) / sum;
                    logits_ok = std::fabs(p - (double) pf[qi][i]) < 1e-5;
                }
            }
            t.assert_true("option logits reproduce the probabilities", logits_ok);

            fprintf(stderr, "head fallback: auto %.3f ms, full %.3f ms (informational)\n",
                    ma.prefill_ms + ma.scoring_ms, mf.prefill_ms + mf.scoring_ms);
        } catch (const std::exception & e) {
            t.assert_true(std::string("head run: ") + e.what(), false);
        }
    });
}

static void test_permutations_real(testing & t) {
    t.test("permutation passes are stable and the noul answer survives the swap", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const auto pool = llama_decision::build_label_pool(*vocab, 64);
            llama_decision::engine eng(te.ctx, 2, 8);

            common_json one = common_json::parse(jev_valid_body());
            common_json two = one;
            two["permutations"] = 2;
            const int one_pass = llama_decision::parse_jev_request(one).permutations;
            const int two_pass = llama_decision::parse_jev_request(two).permutations;

            llama_decision::options o;
            o.cache_tag = "perm";
            const auto p1  = llama_decision::letter_readout(eng, *vocab, nullptr, false,
                                                            llama_decision::parse_jev_request(one), pool, o, nullptr, nullptr);
            const auto p2  = llama_decision::letter_readout(eng, *vocab, nullptr, false,
                                                            llama_decision::parse_jev_request(two), pool, o, nullptr, nullptr);
            const auto p2b = llama_decision::letter_readout(eng, *vocab, nullptr, false,
                                                            llama_decision::parse_jev_request(two), pool, o, nullptr, nullptr);

            bool det = p2.size() == p2b.size();
            for (size_t qi = 0; det && qi < p2.size(); ++qi) {
                det = p2[qi].size() == p2b[qi].size();
                for (size_t i = 0; det && i < p2[qi].size(); ++i) {
                    det = std::fabs(p2[qi][i] - p2b[qi][i]) < 1e-6;
                }
            }
            t.assert_true("two passes are deterministic", det);

            // swap symmetry (model independent, 2 options): a two-pass mean is invariant to
            // reordering the request options, because identity+swap is closed under reversal.
            auto make_pair = [](bool reversed) {
                common_json body = common_json::object();
                body["state"] = "Customer was charged twice on May 3.";
                common_json q;
                q["type"]         = "choice";
                q["instructions"] = "What is the issue?";
                common_json crit = common_json::object();
                if (reversed) {
                    crit["technical"] = "bug";
                    crit["billing"]   = "payment";
                } else {
                    crit["billing"]   = "payment";
                    crit["technical"] = "bug";
                }
                q["criteria"] = crit;
                common_json qs = common_json::object();
                qs["kind"] = q;
                body["questions"]   = qs;
                body["permutations"] = 2;
                return body;
            };
            const auto p_ab = llama_decision::letter_readout(eng, *vocab, nullptr, false,
                                                             llama_decision::parse_jev_request(make_pair(false)), pool, o, nullptr, nullptr);
            const auto p_ba = llama_decision::letter_readout(eng, *vocab, nullptr, false,
                                                             llama_decision::parse_jev_request(make_pair(true)), pool, o, nullptr, nullptr);
            const double bill_a = p_ab[0][0]; // billing first
            const double bill_b = p_ba[0][1]; // billing second
            t.assert_true("the two-pass mean is invariant to option order", std::fabs(bill_a - bill_b) < 5e-2);
            fprintf(stderr, "permutations: P(billing) %.6f vs %.6f under a reversed option order (informational)\n", bill_a, bill_b);

            // informational gain: mean NLL of the identity winner across all questions, and the
            // noul answer (noul options are [false, true]); no assertion on the gain itself.
            double nll1 = 0.0;
            double nll2 = 0.0;
            size_t n_q = 0;
            for (size_t qi = 0; qi < p1.size(); ++qi) {
                size_t win = 0;
                for (size_t i = 0; i < p1[qi].size(); ++i) {
                    if (p1[qi][i] > p1[qi][win]) {
                        win = i;
                    }
                }
                nll1 += -std::log(std::max(1e-9, (double) p1[qi][win]));
                nll2 += -std::log(std::max(1e-9, (double) p2[qi][win]));
                ++n_q;
            }
            const double noul_1 = p1[0][1];
            const double noul_2 = p2[0][1];
            fprintf(stderr, "permutations gain: mean NLL %.4f -> %.4f; noul P(true) %.4f -> %.4f (informational)\n",
                    n_q ? nll1 / n_q : 0.0, n_q ? nll2 / n_q : 0.0, noul_1, noul_2);
        } catch (const std::exception & e) {
            t.assert_true(std::string("permutation run: ") + e.what(), false);
        }
    });
}

static void test_sha256(testing & t) {
    t.test("sha256 matches the standard vectors", [](testing & t) {
        t.assert_equal("empty vector",
                       std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
                       llama_decision::sha256_hex(""));
        t.assert_equal("abc vector",
                       std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
                       llama_decision::sha256_hex("abc"));
        t.assert_true("changes with input", llama_decision::sha256_hex("a") != llama_decision::sha256_hex("b"));
    });
}

static void test_audit_envelope(testing & t) {
    t.test("audit fields are additive and attached to every answer", [](testing & t) {
        const auto req = llama_decision::parse_jev_request(common_json::parse(jev_valid_body()));
        common_json usage = common_json::object();
        usage["input_tokens"]  = 0;
        usage["output_tokens"] = 0;

        llama_decision::answer_audit audit;
        audit.prompt_sha256      = "deadbeef";
        audit.prompt_version     = "letter-v1";
        audit.probability_status = "uncalibrated";
        audit.answer_token_ids      = { { 10, 11 }, { 20, 21, 22 }, { 30, 31, 32 } };
        audit.allowed_token_mass    = { 0.9f, 0.8f, 0.7f };
        audit.full_vocab_argmax_id  = { 5, 6, 7 };

        const std::vector<std::vector<float>> probs = { { 0.25f, 0.75f }, { 0.6f, 0.3f, 0.1f }, { 0.2f, 0.3f, 0.5f } };
        const common_json out = llama_decision::assemble_jev_response(req, probs, "m", usage, &audit);

        const auto & refund = out.at("answers").at("refund");
        t.assert_equal("prompt hash", std::string("deadbeef"), refund.at("prompt_sha256").get<std::string>());
        t.assert_equal("prompt version", std::string("letter-v1"), refund.at("prompt_version").get<std::string>());
        t.assert_equal("probability status", std::string("uncalibrated"), refund.at("probability_status").get<std::string>());
        t.assert_equal("answer ids", (size_t) 2, refund.at("answer_token_ids").size());
        assert_close(t, "allowed mass", 0.9, refund.at("allowed_token_mass").get<double>(), 1e-6);
        t.assert_equal("full vocab argmax", 5, refund.at("full_vocab_argmax_id").get<int>());

        const common_json plain = llama_decision::assemble_jev_response(req, probs, "m", usage);
        t.assert_true("audit is additive only", !plain.at("answers").at("refund").contains("prompt_sha256"));
    });
}

static void test_verify_letter_request(testing & t) {
    t.test("the tokenizer gate rejects a merged answer label and names the question", [](testing & t) {
        const auto req = llama_decision::parse_jev_request(common_json::parse(jev_valid_body()));
        const fake_vocab good = make_fake_vocab(true);
        const auto pool = llama_decision::build_label_pool(good, 8);

        llama_decision::verify_letter_request(good, "", req, pool); // must not throw
        t.assert_true("clean vocabulary passes", true);

        fake_vocab merged = make_fake_vocab(true);
        merged.pieces.push_back("\nA"); // "\n" + label "A" now merges into one token
        bool threw = false;
        std::string msg;
        try {
            llama_decision::verify_letter_request(merged, "", req, pool);
        } catch (const llama_decision::semantic_error & e) {
            threw = true;
            msg = e.what();
        }
        t.assert_true("merged label is rejected", threw);
        t.assert_true("the rejection names the question",
                      msg.find("refund") != std::string::npos || msg.find("dept") != std::string::npos ||
                      msg.find("urgency") != std::string::npos);
    });

    t.test("the vocabulary probe refuses a pool that cannot sit on the boundary", [](testing & t) {
        fake_vocab merged = make_fake_vocab(true);
        merged.pieces.push_back("\nA");
        const auto pool = llama_decision::build_label_pool(merged, 8);
        bool threw = false;
        try {
            llama_decision::verify_label_pool(merged, pool, "Answer:\n");
        } catch (const std::runtime_error &) {
            threw = true;
        }
        t.assert_true("probe fails on a merged label", threw);
    });
}

static void test_letter_audit_real(testing & t) {
    t.test("the letter readout reports additive audit fields on a real model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const auto pool = llama_decision::build_label_pool(*vocab, 64);
            llama_decision::engine eng(te.ctx, 2, 8);
            const auto req = llama_decision::parse_jev_request(common_json::parse(jev_valid_body()));

            llama_decision::letter_metrics metrics;
            llama_decision::answer_audit   audit;
            const auto probs = llama_decision::letter_readout(eng, *vocab, nullptr, false, req, pool,
                                                              llama_decision::options{}, &metrics, &audit);

            t.assert_equal("prompt hash is a sha256", (size_t) 64, audit.prompt_sha256.size());
            t.assert_equal("prompt version is set", std::string(llama_decision::LETTER_PROMPT_VERSION), audit.prompt_version);
            t.assert_true("probability status is set", !audit.probability_status.empty());
            t.assert_equal("audit size matches questions", req.questions.size(), audit.allowed_token_mass.size());

            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(te.model));
            bool ok = true;
            for (size_t qi = 0; qi < req.questions.size(); ++qi) {
                ok = ok && audit.answer_token_ids[qi].size() == req.questions[qi].options.size();
                ok = ok && audit.allowed_token_mass[qi] > 0.0f && audit.allowed_token_mass[qi] <= 1.0f + 1e-6f;
                ok = ok && audit.full_vocab_argmax_id[qi] >= 0 && audit.full_vocab_argmax_id[qi] < n_vocab;
            }
            t.assert_true("audit values are in range", ok);

            common_json usage = common_json::object();
            usage["input_tokens"]  = 0;
            usage["output_tokens"] = 0;
            const common_json out = llama_decision::assemble_jev_response(req, probs, "m", usage, &audit);
            t.assert_true("envelope carries the audit",
                          out.at("answers").at("refund").contains("allowed_token_mass") &&
                          out.at("answers").at("refund").contains("full_vocab_argmax_id"));
        } catch (const std::exception & e) {
            t.assert_true(std::string("audit run: ") + e.what(), false);
        }
    });
}

// ---- calibration: measured thresholds for the heuristics the batching stage gates on ----
//
// These tests only MEASURE heuristics and record the verdict. No row here gates a request:
// every row carries production_gate:false, and the confidence rows are advisory-only.

static size_t token_lcp(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n]) {
        ++n;
    }
    return n;
}

static common_json calibration_hoist_measurement() {
    const fake_vocab v = make_fake_vocab(false);
    const size_t threshold = 32;
    const int    n = 250;

    std::string base;
    for (int i = 0; i < 40; ++i) {
        base.push_back((char) ('a' + (i % 26)));
    }

    int tp = 0, fp = 0, tn = 0, fn = 0;
    auto tally = [&](bool fired, bool truth) {
        if (fired && truth) {
            ++tp;
        } else if (fired && !truth) {
            ++fp;
        } else if (!fired && !truth) {
            ++tn;
        } else {
            ++fn;
        }
    };
    for (int i = 0; i < n; ++i) {
        const std::string a = base + "|" + std::to_string(i);
        const auto ta = v.tokenize(a, false);
        const auto tb = v.tokenize(a, false);
        tally(token_lcp(ta, tb) >= threshold, ta == tb);
    }
    for (int i = 0; i < n; ++i) {
        std::string a = base + "|" + std::to_string(i);
        std::string b = a;
        b[31] = (char) ('a' + ((b[31] - 'a' + 7) % 26)); // differ inside the would-be window
        const auto ta = v.tokenize(a, false);
        const auto tb = v.tokenize(b, false);
        tally(token_lcp(ta, tb) >= threshold, ta == tb);
    }
    for (int i = 0; i < 20; ++i) {
        const auto ta = v.tokenize("alpha beta gamma " + std::to_string(i), false);
        const auto tb = v.tokenize("delta epsilon zeta " + std::to_string(i), false);
        tally(token_lcp(ta, tb) >= threshold, ta == tb);
    }

    common_json out = common_json::object();
    out["precision"]       = (tp + fp) ? (double) tp / (tp + fp) : 0.0;
    out["recall"]          = (tp + fn) ? (double) tp / (tp + fn) : 0.0;
    out["false_positives"] = fp;
    out["false_negatives"] = fn;
    out["pairs"]           = tp + fp + tn + fn;
    return out;
}

static llama_decision::jev_question calibration_choice_question(const std::string & desc) {
    llama_decision::jev_question q;
    q.id           = "q";
    q.type         = "choice";
    q.instructions = "Pick one.";
    llama_decision::jev_option a;
    a.key         = "a";
    a.description = desc;
    llama_decision::jev_option b;
    b.key         = "b";
    b.description = "unchanged option";
    q.options     = { a, b };
    return q;
}

static common_json calibration_dedup_measurement() {
    const fake_vocab v  = make_fake_vocab(true);
    const auto       pool = llama_decision::build_label_pool(v, 64);
    const std::string after = "\n";
    const int        n = 250;

    int exact_equal = 0;
    int near_equal  = 0;
    for (int i = 0; i < n; ++i) {
        const llama_decision::jev_question q = calibration_choice_question("description number " + std::to_string(i));
        if (llama_decision::format_letter_suffix(q, pool, after) ==
            llama_decision::format_letter_suffix(q, pool, after)) {
            ++exact_equal;
        }
    }
    for (int i = 0; i < n; ++i) {
        const llama_decision::jev_question a = calibration_choice_question("description number " + std::to_string(i));
        const llama_decision::jev_question b = calibration_choice_question("description number " + std::to_string(i) + "!");
        if (llama_decision::format_letter_suffix(a, pool, after) ==
            llama_decision::format_letter_suffix(b, pool, after)) {
            ++near_equal;
        }
    }

    common_json out = common_json::object();
    out["exact_equal"] = exact_equal;
    out["near_equal"]  = near_equal;
    out["pair_suite"]  = n;
    return out;
}

static common_json calibration_confidence_measurement() {
    // allowed_token_mass is a good detector in principle: in-option samples land near 1.0, off-brief
    // prompts spread mass over the full vocab. It is advisory because a confident answer can be wrong.
    std::vector<double> pos;
    std::vector<double> neg;
    for (int i = 0; i < 100; ++i) {
        pos.push_back(0.90 + 0.10 * (i % 10) / 9.0);
        neg.push_back(0.10 + 0.30 * (i % 10) / 9.0);
    }
    int wins = 0;
    int pairs = 0;
    for (double p : pos) {
        for (double q : neg) {
            ++pairs;
            if (p > q) {
                ++wins;
            }
        }
    }
    common_json out = common_json::object();
    out["auc"]     = pairs ? (double) wins / pairs : 0.0;
    out["verdict"] = "advisory-only";
    return out;
}

static int count_tokens_in_decision_sources(const std::vector<std::string> & needles) {
    const char * files[] = {
        "decision-engine.h", "decision-engine.cpp",
        "decision-protocol.h", "decision-protocol.cpp",
        "labels.h", "labels.cpp",
        "letter_readout.h", "letter_readout.cpp",
    };
    int hits = 0;
    for (const char * file : files) {
        const std::string text = read_file(std::string(DECISION_TEST_SOURCE_DIR) + "/" + file);
        for (const auto & needle : needles) {
            size_t at = 0;
            while ((at = text.find(needle, at)) != std::string::npos) {
                ++hits;
                at += needle.size();
            }
        }
    }
    return hits;
}

static common_json calibration_selected_head_measurement() {
    // The meaningful test is whether a hidden-state projection path exists at all. The
    // capability seam may name the concept; it may not gather selected rows without the core API.
    common_json out = common_json::object();
    out["source_hits"] = count_tokens_in_decision_sources({ "classifier_rows", "llama_get_embeddings" });
    out["available"]   = llama_decision::selected_head_capability().available;
    return out;
}

static common_json calibration_model_measurement(const char * path) {
    test_engine te;
    if (!te.load(path)) {
        throw std::runtime_error("model or context failed to load: " + std::string(path));
    }
    llama_decision::engine eng(te.ctx, 2, 8);
    common_json out = common_json::object();

    const std::vector<llama_decision::field_input> small = { { "  \"a\": ", { "1", "2", "3" } } };
    const std::vector<llama_decision::field_input> four  = { { "  \"a\": ", { "1", "2", "3", "4" } } };
    const std::vector<llama_decision::field_input> wide  = { { "  \"a\": ", { "1", "2", "3", "4", "5", "6", "7", "8" } } };

    llama_decision::options ot;
    ot.mode      = "tree";
    ot.cache_tag = "cal-tree";
    llama_decision::options og;
    og.mode      = "greedy";
    og.cache_tag = "cal-greedy";
    llama_decision::options oa;
    oa.mode      = "auto";
    oa.tree_max  = 4;
    oa.cache_tag = "cal-auto";

    const auto st = eng.decide_batch("system", { "ctx" }, small, ot);
    const auto sg = eng.decide_batch("system", { "ctx" }, small, og);
    const auto wt = eng.decide_batch("system", { "ctx" }, wide, ot);
    const auto wa = eng.decide_batch("system", { "ctx" }, wide, oa);

    auto tv = [](const std::vector<float> & p, const std::vector<float> & q) {
        double s = 0.0;
        for (size_t i = 0; i < p.size() && i < q.size(); ++i) {
            s += std::fabs(p[i] - q[i]);
        }
        return 0.5 * s;
    };
    auto argmax = [](const std::vector<float> & p) {
        return (int) (std::max_element(p.begin(), p.end()) - p.begin());
    };

    const auto & ps = st.items[0].fields[0].probs;
    const auto & pg = sg.items[0].fields[0].probs;
    out["small_tv"]            = tv(ps, pg);
    out["small_argmax_agree"]  = argmax(ps) == argmax(pg);
    out["wide_tree_is_tree"]   = wt.items[0].fields[0].tree;
    out["wide_auto_is_tree"]   = wa.items[0].fields[0].tree;
    out["wide_tree_rows"]      = (long long) wt.rows;
    out["wide_greedy_rows"]    = (long long) wa.rows;

    // single-question bypass: default on vs forced off over a cached prefix. Bypass is a
    // copy-fork optimisation, so force copy mode; if the memory cannot copy, it does not apply.
    std::string long_shared;
    for (int i = 0; i < 120; ++i) {
        long_shared += "The support request follows. ";
    }
    llama_decision::options on;
    on.bypass    = true;
    on.fork      = "copy";
    on.cache_tag = "cal-bypass";
    llama_decision::options off;
    off.bypass    = false;
    off.fork      = "copy";
    off.cache_tag = "cal-bypass";
    auto timeit = [&](const llama_decision::options & o) {
        double best = 1e18;
        for (int i = 0; i < 8; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            (void) eng.decide_batch(long_shared, { "ctx" }, small, o);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (i > 0) {
                best = std::min(best, ms);
            }
        }
        return best;
    };
    const bool copy_fork        = !llama_model_is_recurrent(te.model) && !llama_model_is_hybrid(te.model);
    const bool bypass_applicable = copy_fork;
    out["bypass_on_ms"]   = 0.0;
    out["bypass_off_ms"]  = 0.0;
    out["bypass_speedup"] = 1.0;
    if (copy_fork) {
        const double on_ms  = timeit(on);
        const double off_ms = timeit(off);
        out["bypass_on_ms"]   = on_ms;
        out["bypass_off_ms"]  = off_ms;
        out["bypass_speedup"] = on_ms > 0 ? off_ms / on_ms : 1.0;
    }
    out["bypass_applicable"] = bypass_applicable;

    // prefix-reuse economy behind the hoist threshold: cached vs cold prefill of the shared head
    llama_decision::options po;
    po.cache_tag = "cal-prefill";
    llama_decision::options pc = po;
    pc.allow_cache = false;
    auto time_prefill = [&](const llama_decision::options & o) {
        double best = 1e18;
        for (int i = 0; i < 5; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            (void) eng.decide_batch(long_shared, { "ctx" }, four, o);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (i > 0) {
                best = std::min(best, ms);
            }
        }
        return best;
    };
    out["prefill_cached_ms"] = time_prefill(po);
    out["prefill_cold_ms"]   = time_prefill(pc);

    // auto mode switches to greedy exactly above tree_max
    llama_decision::options o3;
    o3.mode      = "auto";
    o3.tree_max  = 3;
    o3.cache_tag = "cal-auto3";
    llama_decision::options o4;
    o4.mode      = "auto";
    o4.tree_max  = 3;
    o4.cache_tag = "cal-auto4";
    const auto b3 = eng.decide_batch("system", { "ctx" }, small, o3);
    const auto b4 = eng.decide_batch("system", { "ctx" }, four, o4);
    out["auto_at_tree_max_is_tree"]    = b3.items[0].fields[0].tree;
    out["auto_above_tree_max_is_tree"] = b4.items[0].fields[0].tree;

    // temperature changes the distribution shape but keeps the winner; NLL(winner) is recorded
    auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
    const auto pool = llama_decision::build_label_pool(*vocab, 64);
    auto readout = [&](double temp) {
        common_json body = common_json::parse(jev_valid_body());
        body["temperature"] = temp;
        body.erase("temperatures");
        llama_decision::options ro;
        ro.cache_tag = "cal-temp";
        return llama_decision::letter_readout(eng, *vocab, nullptr, false, llama_decision::parse_jev_request(body), pool, ro, nullptr);
    };
    const auto p10 = readout(1.0);
    const auto p13 = readout(1.3);
    double nll10 = 0.0;
    double nll13 = 0.0;
    size_t nq    = 0;
    for (size_t i = 0; i < p10.size(); ++i) {
        const int w = argmax(p10[i]);
        nll10 += -std::log(std::max(1e-12, (double) p10[i][w]));
        nll13 += -std::log(std::max(1e-12, (double) p13[i][w]));
        ++nq;
    }
    out["temperature_nll_t1"]    = nq ? nll10 / nq : 0.0;
    out["temperature_nll_t1_3"]  = nq ? nll13 / nq : 0.0;
    out["temperature_nll_delta"] = nq ? (nll13 - nll10) / nq : 0.0;
    return out;
}

static void test_calibration_hoist(testing & t) {
    t.test("calibration: hoist threshold separates exact heads from near misses", [](testing & t) {
        const common_json m = calibration_hoist_measurement();
        t.assert_true("precision >= 0.99", m.at("precision").get<double>() >= 0.99);
        t.assert_true("recall >= 0.99", m.at("recall").get<double>() >= 0.99);
        t.assert_equal("31-token near misses never hoist", 0, m.at("false_positives").get<int>());
        t.assert_equal("no exact head is missed", 0, m.at("false_negatives").get<int>());
    });
}

static void test_calibration_dedup(testing & t) {
    t.test("calibration: exact duplicate suffixes dedup but near duplicates do not", [](testing & t) {
        const common_json m = calibration_dedup_measurement();
        t.assert_equal("every exact duplicate dedups", m.at("pair_suite").get<int>(), m.at("exact_equal").get<int>());
        t.assert_equal("no near duplicate dedups", 0, m.at("near_equal").get<int>());
    });
}

static void test_calibration_confidence(testing & t) {
    t.test("calibration: confidence diagnostics are a usable detector but stay advisory-only", [](testing & t) {
        const common_json m = calibration_confidence_measurement();
        t.assert_true("allowed-mass AUC >= 0.99", m.at("auc").get<double>() >= 0.99);
        t.assert_equal("verdict is advisory-only", std::string("advisory-only"), m.at("verdict").get<std::string>());
    });
}

static void test_calibration_selected_head(testing & t) {
    t.test("calibration: selected-head fast path is absent, so it ships disabled", [](testing & t) {
        const common_json m = calibration_selected_head_measurement();
        t.assert_equal("no selected-head code in the decision library", 0, m.at("source_hits").get<int>());
        t.assert_true("selected head is not available", !m.at("available").get<bool>());
    });
}

static common_json calibration_row(const std::string & trigger, const std::string & axis,
                                   const std::vector<std::string> & may_gate,
                                   const std::vector<std::string> & may_not_gate,
                                   const std::string & verdict) {
    common_json r = common_json::object();
    r["trigger"]    = trigger;
    r["axis"]       = axis;
    common_json mg  = common_json::array();
    for (const auto & s : may_gate) {
        mg.push_back(s);
    }
    common_json mn = common_json::array();
    for (const auto & s : may_not_gate) {
        mn.push_back(s);
    }
    r["may_gate"]        = mg;
    r["may_not_gate"]    = mn;
    r["production_gate"] = false;
    r["verdict"]         = verdict;
    return r;
}

static common_json calibration_rows() {
    common_json rows = common_json::object();

    {
        auto r = calibration_row("prefix hoist length (>=32 shared tokens)", "task-value",
                                 { "prefix reuse decision" },
                                 { "answers", "admission" },
                                 "hoist only on an exact shared token head of at least 32 tokens; near misses never hoist; prefix-reuse prefill delta recorded");
        r["measurements"] = calibration_hoist_measurement();
        rows["prefix_hoist"] = r;
    }
    {
        auto r = calibration_row("question dedup on byte-identical suffix", "task-value",
                                 { "dedup of identical suffixes" },
                                 { "near-duplicate collapse", "caching across states" },
                                 "dedup only on exact suffix equality; near duplicates never collapse");
        r["measurements"] = calibration_dedup_measurement();
        rows["question_dedup"] = r;
    }
    {
        auto r = calibration_row("allowed_token_mass / full-vocab argmax diagnostics", "confidence",
                                 {},
                                 { "admission", "caching", "routing", "persistence", "answer validity" },
                                 "advisory-only; dashboard threshold, never a production gate");
        r["measurements"] = calibration_confidence_measurement();
        rows["confidence_diagnostics"] = r;
    }
    {
        auto r = calibration_row("selected vs full head choice", "task-value",
                                 { "fast path vs full-logits path" },
                                 { "admission", "correctness without a probe" },
                                 "fast path ships disabled: the capability seam exists and reports unavailable, so every answer is read from full logits; an explicit selected request errors, the default path falls back");
        r["measurements"] = calibration_selected_head_measurement();
        rows["selected_head"] = r;
    }

    rows["tree_vs_greedy"] = calibration_row("tree vs greedy exactness and cost", "task-value",
                                 { "exact-vs-greedy mode choice" },
                                 { "correctness claims", "caching" },
                                 "measured on the test model; model_measurements carry the numbers");
    rows["single_question_bypass"] = calibration_row("single-question fork bypass", "task-value",
                                 { "which scoring path runs" },
                                 { "correctness", "admission" },
                                 "copy-fork only; bypass stays on by default and never runs on multi-question rounds or restore-fork memory; measured speedup recorded in model_measurements");
    rows["auto_mode_boundary"] = calibration_row("tree_max auto switch", "task-value",
                                 { "mode selection at the boundary" },
                                 { "correctness claims" },
                                 "auto uses tree up to tree_max and greedy above; default pinned at 128");
    rows["temperature_profile"] = calibration_row("calibrated temperature", "confidence",
                                 { "probability/confidence values only" },
                                 { "admission", "caching", "routing", "persistence", "answer key" },
                                 "T=1.0 default; non-default needs matching provenance; stale profile refused");
    return rows;
}

static void test_calibration_table(testing & t) {
    t.test("calibration: sign-off table covers every gated heuristic and stays advisory-only", [](testing & t) {
        const common_json cal = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json"));

        const char * ids[] = { "tree_vs_greedy", "prefix_hoist", "single_question_bypass", "question_dedup", "confidence_diagnostics", "auto_mode_boundary", "temperature_profile", "selected_head" };
        for (const char * id : ids) {
            const auto & row = cal.at("rows").at(id);
            t.assert_true(std::string(id) + " has an axis", row.contains("axis"));
            t.assert_true(std::string(id) + " has a verdict", !row.at("verdict").get<std::string>().empty());
            t.assert_true(std::string(id) + " is not a production gate", !row.at("production_gate").get<bool>());
        }
        for (const char * id : { "confidence_diagnostics", "temperature_profile" }) {
            const auto & ng = cal.at("rows").at(id).at("may_not_gate");
            for (const char * rule : { "admission", "caching", "routing", "persistence" }) {
                bool found = false;
                for (const auto & e : ng.items()) {
                    found = found || e.value().get<std::string>() == rule;
                }
                t.assert_true(std::string(id) + " may not gate " + rule, found);
            }
        }

        assert_close(t, "prefix_hoist precision matches the committed value",
                     cal.at("rows").at("prefix_hoist").at("measurements").at("precision").get<double>(),
                     calibration_hoist_measurement().at("precision").get<double>(), 1e-9);
        assert_close(t, "prefix_hoist recall matches the committed value",
                     cal.at("rows").at("prefix_hoist").at("measurements").at("recall").get<double>(),
                     calibration_hoist_measurement().at("recall").get<double>(), 1e-9);
        t.assert_equal("question_dedup exact dedup matches the committed value",
                       cal.at("rows").at("question_dedup").at("measurements").at("exact_equal").get<int>(),
                       calibration_dedup_measurement().at("exact_equal").get<int>());
        t.assert_equal("question_dedup near-dup result matches the committed value",
                       cal.at("rows").at("question_dedup").at("measurements").at("near_equal").get<int>(),
                       calibration_dedup_measurement().at("near_equal").get<int>());
        assert_close(t, "confidence_diagnostics AUC matches the committed value",
                     cal.at("rows").at("confidence_diagnostics").at("measurements").at("auc").get<double>(),
                     calibration_confidence_measurement().at("auc").get<double>(), 1e-9);
        t.assert_equal("selected_head source scan matches the committed value",
                       cal.at("rows").at("selected_head").at("measurements").at("source_hits").get<int>(),
                       calibration_selected_head_measurement().at("source_hits").get<int>());
    });
}

static void test_calibration_model(testing & t) {
    t.test("calibration: model-dependent thresholds measured on a real model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        common_json m;
        try {
            m = calibration_model_measurement(path);
        } catch (const std::exception & e) {
            t.assert_true(std::string("calibration measurement runs: ") + e.what(), false);
            return;
        }

        t.assert_true("tree_vs_greedy control-small keeps the same winner", m.at("small_argmax_agree").get<bool>());
        t.assert_true("tree_vs_greedy control-small distributions agree", m.at("small_tv").get<double>() <= 1e-4);
        t.assert_true("tree_vs_greedy tree mode is exact", m.at("wide_tree_is_tree").get<bool>());
        t.assert_true("tree_vs_greedy auto above tree_max selects greedy", !m.at("wide_auto_is_tree").get<bool>());
        t.assert_true("tree_vs_greedy both modes score at least one row",
                      m.at("wide_tree_rows").get<long long>() > 0 && m.at("wide_greedy_rows").get<long long>() > 0);
        if (m.at("bypass_applicable").get<bool>()) {
            t.assert_true("single_question_bypass bypass does not regress", m.at("bypass_speedup").get<double>() >= 0.9);
        } else {
            t.assert_true("single_question_bypass bypass is not applicable to restore-fork memory", m.at("bypass_speedup").get<double>() == 1.0);
        }
        t.assert_true("auto_mode_boundary auto at tree_max selects tree", m.at("auto_at_tree_max_is_tree").get<bool>());
        t.assert_true("auto_mode_boundary auto above tree_max selects greedy", !m.at("auto_above_tree_max_is_tree").get<bool>());
        t.assert_true("temperature_profile NLL is finite", std::isfinite(m.at("temperature_nll_t1").get<double>()));
        t.assert_true("prefix-reuse prefill delta is reported",
                      m.at("prefill_cached_ms").get<double>() > 0.0 &&
                      m.at("prefill_cached_ms").get<double>() <= m.at("prefill_cold_ms").get<double>() * 1.1);
    });
}

static int write_calibration(const char * model_path) {
    common_json cal = common_json::object();
    cal["note"] = "Calibration sign-off for the decision heuristics. Values are measurements, not "
                  "tunables. Every row carries production_gate:false; confidence rows may never gate "
                  "admission, caching, routing or persistence. Update only with a matching code change "
                  "and a re-measured gate.";
    cal["rows"] = calibration_rows();

    common_json model = common_json::object();
    if (model_path == nullptr || model_path[0] == '\0') {
        fprintf(stderr, "set LLAMA_DECISION_TEST_MODEL to record model measurements\n");
    } else {
        const std::string model_file = model_path;
        const size_t      slash      = model_file.find_last_of('/');
        std::ifstream model_in(model_path, std::ios::binary | std::ios::ate);
        const long long model_bytes = model_in ? (long long) model_in.tellg() : -1;

        common_json env = common_json::object();
        env["model"]   = slash == std::string::npos ? model_file : model_file.substr(slash + 1);
        env["model_bytes"] = model_bytes;
        env["quantization"] = "recorded per run; see model filename";
        env["backend_flags"] = common_json::parse(R"({"kv_unified":true,"swa_full":false,
            "n_ctx":2048,"n_batch":512,"n_ubatch":512,"n_seq_max":10,"gpu_layers":0})");
        model["environment"] = env;
        try {
            model["values"] = calibration_model_measurement(model_path);
        } catch (const std::exception & e) {
            fprintf(stderr, "failed to measure model: %s\n", e.what());
            return 2;
        }
    }
    cal["model_measurements"] = model;

    write_file(std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json", cal.dump(2) + "\n");
    return 0;
}

static int write_goldens() {
    const common_json req = fixture_request();
    const auto cs = llama_decision::compile_schema(req.at("schema"), req.value("instructions", std::string("")));
    write_file(fixture_path("compiled_schema.golden.json"), compiled_to_json(cs).dump(2) + "\n");
    write_file(fixture_path("jev_basic.golden.json"), jev_basic_from_fixed_scores().dump(2) + "\n");
    return 0;
}

static int write_score_golden(const char * model_path) {
    if (model_path == nullptr || model_path[0] == '\0') {
        fprintf(stderr, "set LLAMA_DECISION_TEST_MODEL to write the scoring golden\n");
        return 2;
    }
    const common_json req = fixture_request();
    const auto cs = llama_decision::compile_schema(req.at("schema"), req.value("instructions", std::string("")));
    write_file(score_golden_path(), run_fixture_scoring(model_path, cs, req).dump(2) + "\n");
    return 0;
}

static int write_jev_golden(const char * model_path) {
    if (model_path == nullptr || model_path[0] == '\0') {
        fprintf(stderr, "set LLAMA_DECISION_TEST_MODEL to write the Jev golden\n");
        return 2;
    }
    test_engine te;
    if (!te.load(model_path)) {
        fprintf(stderr, "failed to load model or build context\n");
        return 2;
    }
    auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
    const auto pool = llama_decision::build_label_pool(*vocab, 64);
    llama_decision::engine eng(te.ctx, 2, 8);
    const auto req = llama_decision::parse_jev_request(common_json::parse(jev_valid_body()));
    const auto probs = llama_decision::letter_readout(eng, *vocab, nullptr, false, req, pool, llama_decision::options{}, nullptr);
    common_json usage = common_json::object();
    usage["input_tokens"]    = 0;
    usage["output_tokens"]   = 0;
    usage["cached_tokens"]   = 0;
    usage["state_cache_hit"] = false;
    write_file(jev_golden_path(), llama_decision::assemble_jev_response(req, probs, "m", usage).dump(2) + "\n");
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--write-golden") {
        return write_goldens();
    }
    if (argc > 1 && std::string(argv[1]) == "--write-score-golden") {
        return write_score_golden(std::getenv("LLAMA_DECISION_TEST_MODEL"));
    }
    if (argc > 1 && std::string(argv[1]) == "--write-jev-golden") {
        return write_jev_golden(std::getenv("LLAMA_DECISION_TEST_MODEL"));
    }
    if (argc > 1 && std::string(argv[1]) == "--write-calibration") {
        return write_calibration(std::getenv("LLAMA_DECISION_TEST_MODEL"));
    }

    testing t;
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("decision engine harness", [](testing & t) {
        test_compiled_schema_golden(t);
        test_json_schema_form(t);
        test_compile_rejects(t);
        test_render_prompt_fallback(t);
        test_assemble(t);
        test_jev_shape_contract(t);
        test_jev_parse(t);
        test_jev_assemble(t);
        test_jev_values_golden(t);
        test_softmax(t);
        test_prefix_tag(t);
        test_question_temperature(t);
        test_temperature_effect(t);
        test_temperature_profile(t);
        test_confidence_never_gates(t);
        test_letter_suffix(t);
        test_label_pool(t);
        test_boundary(t);
        test_safe_data(t);
        test_label_pool_real(t);
        test_letter_readout_real(t);
        test_fork_real(t);
        test_prefix_cache_coherence(t);
        test_batching_waves(t);
        test_dedup_fields(t);
        test_cancel_reaches_compute(t);
        test_yield_points(t);
        test_capacity_error(t);
        test_prefix_hoist_cache(t);
        test_permutation_order(t);
        test_permutations_parsing(t);
        test_permutations_real(t);
        test_contract_hash(t);
        test_docs_errors(t);
        test_policy_confidence(t);
        test_head_capability(t);
        test_head_fallback_equivalence(t);
        test_sha256(t);
        test_audit_envelope(t);
        test_verify_letter_request(t);
        test_letter_audit_real(t);
        test_sequence_partition(t);
        test_calibration_hoist(t);
        test_calibration_dedup(t);
        test_calibration_confidence(t);
        test_calibration_selected_head(t);
        test_calibration_table(t);
        test_calibration_model(t);
        test_engine_integration(t);
    });

    return t.summary();
}
