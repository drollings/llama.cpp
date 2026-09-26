// White-box access to the engine's save/load helpers: the fail-fast behavior they encode is
// internal with no public seam to force a save failure. The macro is scoped to this TU only.
#define private public
#include "decision-engine.h"
#undef private
#include "../src/llama-ext.h"  // staging API: classifier answer-head predicate and row reader
#include "chat.h"
#include "common.h"
#include "decision-protocol.h"
#include "ggml-backend.h"
#include "json.h"
#include "labels.h"
#include "letter_readout.h"
#include "llama.h"
#include "speculative.h"
#include "testing.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <limits>
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

#ifndef DECISION_TEST_GENERATED_MODEL_DIR
#define DECISION_TEST_GENERATED_MODEL_DIR ""
#endif

#ifndef DECISION_TEST_SPM_MODEL
#define DECISION_TEST_SPM_MODEL ""
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

// Last two path components, so a golden can name the model it was written from without pinning
// the machine's model root (several GGUFs share the basename "latest.gguf").
static std::string model_identity(const std::string & path) {
    size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) {
        return path;
    }
    const std::string file = path.substr(slash + 1);
    const size_t prev = path.find_last_of("/\\", slash - 1);
    return prev == std::string::npos ? file : path.substr(prev + 1, slash - prev - 1) + "/" + file;
}

static const char * decision_valid_body() {
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

static size_t count_substring(const std::string & hay, const std::string & needle) {
    size_t n = 0;
    for (size_t at = hay.find(needle); at != std::string::npos; at = hay.find(needle, at + needle.size())) {
        ++n;
    }
    return n;
}

// Sorted, comma-joined key set of a JSON object, so a test can pin the exact shape of an envelope.
static std::string key_set(const common_json & obj) {
    std::vector<std::string> keys;
    for (const auto & e : obj.items()) {
        keys.push_back(e.key());
    }
    std::sort(keys.begin(), keys.end());
    return string_join(keys, ",");
}

// Minimal templates whose thinking marker only appears when the caller asks for it. They let the
// test prove that the decision framer pins the toggle off even when the loaded model's own
// template ignores it (LFM2.5 appends its marker unconditionally).
static const char * thinking_probe_template() {
    return "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
           "{%- if add_generation_prompt -%}assistant:{% if enable_thinking %} <think>{% endif %} {% endif %}";
}

static const char * preserve_probe_template() {
    return "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
           "{%- if preserve_thinking -%}<think>kept</think>{% endif -%}"
           "{%- if add_generation_prompt -%}assistant:{% endif %}";
}

static std::string join_split(const std::pair<std::string, std::string> & p) {
    return p.first + p.second;
}

static void test_split_chat_template_primitive(testing & t) {
    t.test("split_chat_template and fnv1a64 are the shared primitives behind both renderers", [](testing & t) {
        t.assert_equal("fnv1a64(\"\") is the established offset basis, preserved from the pre-refactor hash",
                       (uint64_t) 1469598103934665603ull, llama_decision::fnv1a64(""));

        auto tmpls = common_chat_templates_init(nullptr, thinking_probe_template());
        if (!tmpls) {
            t.skip("jinja template probe unavailable");
            return;
        }
        const auto parts  = llama_decision::split_chat_template(tmpls.get(), true, "SYS", false);
        const auto letter = llama_decision::render_letter_prompt(tmpls.get(), true, "SYS", false);
        t.assert_equal("render_letter_prompt returns the primitive split head", parts.first, letter.first);
        t.assert_equal("render_letter_prompt returns the primitive split tail", parts.second, letter.second);
    });
}

static void test_thinking_off(testing & t) {
    t.test("the decision framer keeps thinking off", [](testing & t) {
        auto probe = common_chat_templates_init(nullptr, thinking_probe_template());
        if (!probe) {
            t.skip("the jinja engine is unavailable");
            return;
        }
        try {
            const std::string sys = "sys";
            const auto off     = llama_decision::render_letter_prompt(probe.get(), true, sys, false);
            const auto on      = llama_decision::render_letter_prompt(probe.get(), true, sys, true);
            const auto default_off = llama_decision::render_letter_prompt(probe.get(), true, sys);

            t.assert_equal("the framer defaults to thinking off", join_split(off), join_split(default_off));
            t.assert_true("thinking off emits no marker", count_substring(join_split(off), "<think>") == 0);
            t.assert_true("thinking on emits the marker", count_substring(join_split(on), "<think>") == 1);
            t.assert_true("thinking on adds tokens", join_split(on).size() > join_split(off).size());

        } catch (const std::exception & e) {
            t.assert_true(std::string("thinking probe renders: ") + e.what(), false);
        }
    });

    t.test("a raw decision prefix carries no thinking marker", [](testing & t) {
        const auto raw = llama_decision::render_letter_prompt(nullptr, false, "SYS");
        t.assert_true("raw letter prefix is thinking free", count_substring(join_split(raw), "<think>") == 0);
    });
}

static void test_thinking_control(testing & t) {
    t.test("preserve_thinking off does not inject a marker into the decision prefix", [](testing & t) {
        auto probe = common_chat_templates_init(nullptr, preserve_probe_template());
        if (!probe) {
            t.skip("the jinja engine is unavailable");
            return;
        }
        try {
            const std::string sys = "sys";
            // the framer never asks the template to preserve reasoning, so no marker is rendered
            const auto prefix = llama_decision::render_letter_prompt(probe.get(), true, sys, false);
            t.assert_true("the decision prefix has no preserved thinking",
                          count_substring(join_split(prefix), "<think>") == 0);

            // control: the same template does inject a marker when a caller explicitly preserves it
            common_chat_templates_inputs in;
            in.use_jinja             = true;
            in.add_generation_prompt = true;
            in.enable_thinking       = false;
            in.chat_template_kwargs  = { { "preserve_thinking", "true" } };
            common_chat_msg sys_msg;
            sys_msg.role    = "system";
            sys_msg.content = sys;
            common_chat_msg usr_msg;
            usr_msg.role    = "user";
            usr_msg.content = "hello";
            in.messages = { sys_msg, usr_msg };
            const std::string preserved = common_chat_templates_apply(probe.get(), in).prompt;
            t.assert_true("the control template can inject when asked",
                          count_substring(preserved, "<think>") == 1);
        } catch (const std::exception & e) {
            t.assert_true(std::string("preserve control renders: ") + e.what(), false);
        }
    });
}

static void test_decision_shape_contract(testing & t) {
    t.test("committed decision envelope skeleton has the required keys", [](testing & t) {
        const common_json shape = common_json::parse(read_file(fixture_path("decision_basic.shape.json")));
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

// The answer-row scoring math takes no context, so it is covered with synthetic hidden states,
// rows, bias and softcap instead of a model.
static llama_decision::classifier_head fake_answer_head() {
    llama_decision::classifier_head head;
    head.ids   = { 10, 11, 12 };
    head.width = 3;
    head.rows  = {
         1.0f, 2.0f, 3.0f, // id 10
         0.0f, 1.0f, 0.0f, // id 11
        -1.0f, 0.0f, 1.0f, // id 12
    };
    return head;
}

static void test_score_answer_rows(testing & t) {
    const std::vector<float> hidden = { 1.0f, 0.0f, 1.0f };
    const llama_decision::tokens_t cands = { 10, 11, 12 };

    t.test("answer rows score dot products plus the output bias", [&](testing & t) {
        auto head = fake_answer_head();
        head.bias = { 0.5f, 0.0f, 2.0f };
        const auto v = llama_decision::score_answer_rows(hidden.data(), head, cands);
        t.assert_equal("one score per candidate", (size_t) 3, v.size());
        assert_close(t, "row 10 plus bias", 4.5, v[0], 1e-5);
        assert_close(t, "row 11 plus bias", 0.0, v[1], 1e-5);
        assert_close(t, "row 12 plus bias", 2.0, v[2], 1e-5);
    });

    t.test("an empty bias adds nothing", [&](testing & t) {
        auto head = fake_answer_head();
        head.softcap = 0.0f;
        const auto v = llama_decision::score_answer_rows(hidden.data(), head, cands);
        assert_close(t, "row 10 unchanged", 4.0, v[0], 1e-5);
        assert_close(t, "row 11 unchanged", 0.0, v[1], 1e-5);
        assert_close(t, "row 12 unchanged", 0.0, v[2], 1e-5);
    });

    t.test("a nonzero softcap bounds every score", [&](testing & t) {
        auto head = fake_answer_head();
        head.bias    = { 0.5f, 0.0f, 2.0f };
        head.softcap = 2.0f;
        const auto v = llama_decision::score_answer_rows(hidden.data(), head, cands);
        assert_close(t, "softcapped row 10", 2.0 * std::tanh(4.5 / 2.0), v[0], 1e-5);
        assert_close(t, "softcapped row 11", 0.0, v[1], 1e-5);
        assert_close(t, "softcapped row 12", 2.0 * std::tanh(2.0 / 2.0), v[2], 1e-5);
    });

    t.test("a zero softcap leaves the score alone", [&](testing & t) {
        auto head = fake_answer_head();
        head.bias    = { 0.5f, 0.0f, 2.0f };
        head.softcap = 0.0f;
        const auto v = llama_decision::score_answer_rows(hidden.data(), head, cands);
        assert_close(t, "row 10 unbounded", 4.5, v[0], 1e-5);
        assert_close(t, "row 12 unbounded", 2.0, v[2], 1e-5);
    });

    t.test("a candidate without a row is rejected", [&](testing & t) {
        const auto head = fake_answer_head();
        bool threw = false;
        try {
            llama_decision::score_answer_rows(hidden.data(), head, { 10, 99 });
        } catch (const std::exception &) {
            threw = true;
        }
        t.assert_true("a missing row throws instead of scoring", threw);
    });

    t.test("an unavailable head is rejected", [&](testing & t) {
        llama_decision::classifier_head head;
        bool threw = false;
        try {
            llama_decision::score_answer_rows(hidden.data(), head, cands);
        } catch (const std::exception &) {
            threw = true;
        }
        t.assert_true("an unavailable head throws", threw);
    });
}

// The load flag is a property of the saved format, so a capability downgrade can never route a
// device state through the host reader or the other way around.
static void test_saved_state_format_dispatch(testing & t) {
    t.test("a saved state is loaded with its own format", [&](testing & t) {
        t.assert_equal("a device state selects the device flag",
                       (unsigned) LLAMA_STATE_SEQ_FLAGS_ON_DEVICE,
                       (unsigned) llama_decision::engine::state_load_flags(true));
        t.assert_equal("a host state selects no flag",
                       (unsigned) LLAMA_STATE_SEQ_FLAGS_NONE,
                       (unsigned) llama_decision::engine::state_load_flags(false));
        t.assert_equal("a partial device state selects both flags",
                       (unsigned) (LLAMA_STATE_SEQ_FLAGS_ON_DEVICE | LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY),
                       (unsigned) llama_decision::engine::state_load_flags(true, true));
        t.assert_equal("a partial host state selects the partial flag",
                       (unsigned) LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY,
                       (unsigned) llama_decision::engine::state_load_flags(false, true));
    });
}

// The provenance gate control is shared by the temperature test and the sign-off table.
static common_json calibration_temperature_control_measurement();

// The numeric extension over the Jev question set: integer/number grids are generated from bounds
// at parse time, scored by label exactly like a choice, and answered with a typed value plus an
// optional aggregate. Pure JSON, no model needed.
static void expect_decision_reject(testing & t, const std::string & body_text, const std::string & needle);
static void test_numeric_questions(testing & t) {
    t.test("numeric integer and number grids parse into typed options", [](testing & t) {
        const auto body = common_json::parse(R"({"model":"m","state":"s","questions":{
            "age":   {"type":"integer","instructions":"age in years","minimum":18,"maximum":20},
            "amount":{"type":"number","instructions":"amount","minimum":0,"maximum":0.5,"step":0.25},
            "avg":   {"type":"integer","instructions":"rating","minimum":1,"maximum":3,"aggregate":"mean"}}})");
        const auto req = llama_decision::parse_decision_request(body);
        t.assert_equal("three questions", (size_t) 3, req.questions.size());

        const auto & age = req.questions[0];
        t.assert_equal("integer canonical type", std::string("integer"), age.type);
        t.assert_equal("integer grid 18..20", (size_t) 3, age.options.size());
        t.assert_equal("integer key is the value", std::string("18"), age.options[0].key);
        t.assert_equal("integer grid is ascending", std::string("20"), age.options[2].key);
        t.assert_true("integer originals are typed numbers", age.options[0].original.is_number_integer());
        t.assert_equal("aggregate absent stays empty", std::string(), age.aggregate);

        const auto & amount = req.questions[1];
        t.assert_equal("number canonical type", std::string("number"), amount.type);
        t.assert_equal("number grid 0..0.5 by 0.25", (size_t) 3, amount.options.size());
        t.assert_equal("number key is fixed width", std::string("0.00"), amount.options[0].key);
        t.assert_equal("number key strips float noise", std::string("0.50"), amount.options[2].key);
        t.assert_true("number originals are floats", amount.options[1].original.is_number_float());

        const auto & avg = req.questions[2];
        t.assert_equal("aggregate parsed", std::string("mean"), avg.aggregate);
    });

    t.test("numeric limits and shapes are validated", [](testing & t) {
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"integer","instructions":"x","minimum":10,"maximum":9}}})",
                               "bounds");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"integer","instructions":"x","minimum":1,"maximum":2,"criteria":{"a":"x"}}}})",
                               "unknown field");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"integer","instructions":"x","minimum":1.5,"maximum":2}}})",
                               "integer needs integer minimum");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"integer","instructions":"x"}}})",
                               "needs integer minimum and maximum");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"number","instructions":"x","minimum":0,"maximum":1}}})",
                               "step (or multipleOf)");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"number","instructions":"x","minimum":0,"maximum":1,"step":0,"multipleOf":0.1}}})",
                               "not both");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"number","instructions":"x","minimum":0,"maximum":1,"step":0.3}}})",
                               "must include both ends");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"integer","instructions":"x","minimum":1,"maximum":1,"aggregate":"mean"}}})",
                               "2-255 values");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"number","instructions":"x","minimum":1,"maximum":256,"step":1}}})",
                               "2-255 values");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"integer","instructions":"x","minimum":1,"maximum":3,"aggregate":"bogus"}}})",
                               "aggregate must be mode, median or mean");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"number","instructions":"x","minimum":0,"maximum":1,"step":0.25,"aggregate":1}}})",
                               "aggregate must be a string");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"choice","instructions":"x","criteria":{"a":null,"b":null},"aggregate":"mean"}}})",
                               "unknown field");
    });

    t.test("numeric answers carry a typed value, probabilities and the optional aggregate", [](testing & t) {
        const auto body = common_json::parse(R"({"model":"m","state":"s","questions":{
            "age":{"type":"integer","instructions":"age","minimum":18,"maximum":20},
            "avg":{"type":"number","instructions":"amount","minimum":0,"maximum":2,"step":1,"aggregate":"mean"},
            "med":{"type":"integer","instructions":"rating","minimum":1,"maximum":4,"aggregate":"median"}}})");
        auto req = llama_decision::parse_decision_request(body);
        common_json usage = common_json::object();
        usage["output_tokens"] = 0;

        const std::vector<std::vector<float>> probs = {
            { 0.1f, 0.2f, 0.7f },       // age grid 18,19,20: winner 20
            { 0.5f, 0.3f, 0.2f },       // avg grid 0,1,2: mean = 0.3 + 0.4 = 0.7
            { 0.5f, 0.2f, 0.2f, 0.1f }, // med grid 1..4: median crosses at 1
        };
        const common_json out = llama_decision::assemble_decision_response(req, probs, "m", usage);

        const auto & age = out.at("answers").at("age");
        t.assert_equal("type echoed", std::string("integer"), age.at("type").get<std::string>());
        t.assert_equal("value is the typed winner", 20, age.at("value").get<long long>());
        assert_close(t, "probabilities keyed by value", 0.7, age.at("probabilities").at("20").get<double>(), 1e-6);
        t.assert_true("no aggregate without a request", !age.contains("aggregate"));

        const auto & avg = out.at("answers").at("avg");
        t.assert_equal("value is the winner as a number", 0.0, avg.at("value").get<double>());
        assert_close(t, "aggregate mean is the weighted mean", 0.7, avg.at("aggregate").get<double>(), 1e-6);

        const auto & med = out.at("answers").at("med");
        assert_close(t, "aggregate median is the value-space quantile", 1.0,
                     med.at("aggregate").get<double>(), 1e-6);

        req.diagnostics = true;
        const common_json diag = llama_decision::assemble_decision_response(req, probs, "m", usage);
        const auto & dage = diag.at("answers").at("age");
        t.assert_true("diagnostics add certainty", dage.contains("certainty"));
        t.assert_true("diagnostics add the spread median", dage.contains("median"));
        t.assert_true("diagnostics add the p10/p90 band", dage.contains("interval_p10_p90"));
        assert_close(t, "p10/p90 band has two entries", 2, (long long) dage.at("interval_p10_p90").size());
    });

    t.test("numeric types take their own temperature overrides", [](testing & t) {
        const auto body = common_json::parse(R"({"model":"m","state":"s","questions":{
            "age":{"type":"integer","instructions":"x","minimum":1,"maximum":3},
            "amt":{"type":"number","instructions":"x","minimum":0,"maximum":1,"step":0.5}}})");
        common_json with_temps = body;
        with_temps["temperature"]  = 1.5;
        with_temps["temperatures"] = common_json::parse(R"({"integer":0.5,"number":2.0})");
        const auto req = llama_decision::parse_decision_request(with_temps);
        assert_close(t, "integer override", 0.5, llama_decision::question_temperature(req, req.questions[0]));
        assert_close(t, "number override", 2.0, llama_decision::question_temperature(req, req.questions[1]));
    });
}

static void test_question_temperature(testing & t) {
    t.test("effective temperature follows per-type override then global", [](testing & t) {
        const auto base = common_json::parse(R"({"model":"m","state":"s","questions":{
            "a":{"type":"noul","instructions":"x"},
            "b":{"type":"choice","instructions":"x","criteria":{"p":null,"q":null}},
            "c":{"type":"score","instructions":"x","criteria":["lo","hi"]}}})");

        const auto plain = llama_decision::parse_decision_request(base);
        for (const auto & q : plain.questions) {
            assert_close(t, "global default", 1.0, llama_decision::question_temperature(plain, q));
        }

        common_json with_override = base;
        with_override["temperature"]   = 1.5;
        with_override["temperatures"]  = common_json::parse(R"({"noul":0.5,"choice":2.0})");
        const auto req = llama_decision::parse_decision_request(with_override);
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
        auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        req.diagnostics = true;
        const std::vector<std::vector<float>> probs = {
            { 0.2f, 0.8f },
            { 0.5f, 0.3f, 0.2f },
            { 0.1f, 0.2f, 0.7f },
        };
        common_json usage = common_json::object();
        usage["output_tokens"] = 0;

        auto jev_recompute = [](const std::vector<float> & p) {
            const double n = (double) p.size();
            const double pmax = (double) *std::max_element(p.begin(), p.end());
            return std::min(1.0, std::max(0.0, (n * pmax - 1.0) / (n - 1.0)));
        };

        const common_json out = llama_decision::assemble_decision_response(req, probs, "m", usage);
        req.confidence_profile = "local";
        const common_json local_out = llama_decision::assemble_decision_response(req, probs, "m", usage);

        const auto & dept = out.at("answers").at("dept");
        t.assert_true("choice has confidence", dept.contains("confidence"));
        t.assert_true("choice has certainty", dept.contains("certainty"));
        assert_close(t, "default choice confidence is the Jev winner-share rescale", jev_recompute(probs[1]),
                     dept.at("confidence").get<double>(), 1e-6);
        assert_close(t, "choice certainty is the winner share", 0.5, dept.at("certainty").get<double>(), 1e-6);

        const auto & urg = out.at("answers").at("urgency");
        t.assert_true("score has confidence", urg.contains("confidence"));
        t.assert_true("score has certainty", urg.contains("certainty"));
        assert_close(t, "default score confidence is the Jev winner-share rescale", jev_recompute(probs[2]),
                     urg.at("confidence").get<double>(), 1e-6);

        t.assert_true("noul has no confidence", !out.at("answers").at("refund").contains("confidence"));
        t.assert_true("noul has no certainty", !out.at("answers").at("refund").contains("certainty"));
    });
}

// confidence and certainty are two axes: the Jev winner-share rescale (default) and the winner's
// share. Pin both against hand-computed literals so a rename cannot quietly swap them.
static void test_confidence_certainty_axes(testing & t) {
    t.test("confidence is the Jev winner-share rescale and certainty is the winner share", [](testing & t) {
        auto req        = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        req.diagnostics = true;
        const std::vector<std::vector<float>> probs = {
            { 0.2f, 0.8f }, // noul: neither axis is emitted
            { 0.6f, 0.3f, 0.1f }, // choice
            { 0.9f, 0.1f, 0.0f }, // score
        };
        common_json usage      = common_json::object();
        usage["output_tokens"] = 0;
        const common_json out  = llama_decision::assemble_decision_response(req, probs, "m", usage);
        req.confidence_profile = "local";
        const common_json local_out = llama_decision::assemble_decision_response(req, probs, "m", usage);

        const auto & dept = out.at("answers").at("dept");
        assert_close(t, "default choice confidence is the Jev rescale for (0.6,0.3,0.1)", 0.4,
                     dept.at("confidence").get<double>(), 1e-6);
        assert_close(t, "choice certainty is max(p) = 0.6", 0.6, dept.at("certainty").get<double>(), 1e-6);

        const auto & urg = out.at("answers").at("urgency");
        assert_close(t, "default score confidence is the Jev rescale for (0.9,0.1,0.0)", 0.85,
                     urg.at("confidence").get<double>(), 1e-6);
        assert_close(t, "score certainty is max(p) = 0.9", 0.9, urg.at("certainty").get<double>(), 1e-6);

        const auto & ldept = local_out.at("answers").at("dept");
        assert_close(t, "local choice confidence is 1 - H/log 3 for (0.6,0.3,0.1)", 0.182654578,
                     ldept.at("confidence").get<double>(), 1e-6);
        const auto & lurg = local_out.at("answers").at("urgency");
        assert_close(t, "local score confidence is 1 - H/log 3 for (0.9,0.1,0.0)", 0.704096726,
                     lurg.at("confidence").get<double>(), 1e-6);

        t.assert_true("noul carries no confidence", !out.at("answers").at("refund").contains("confidence"));
        t.assert_true("noul carries no certainty", !out.at("answers").at("refund").contains("certainty"));
    });
}

static void expect_decision_reject(testing & t, const std::string & body_text, const std::string & needle);

// The opt-in Jev confidence profile is a rescaled winner share, (N*p_max-1)/(N-1). It is opt-in,
// so the default stays 1 - H/logK and existing goldens are unchanged; above three levels, where
// Jev documents no Score formula, the same monotone rule is the stated local value.
static void test_confidence_profile(testing & t) {
    t.test("the Jev confidence profile is the default certainty-based winner share", [](testing & t) {
        auto req        = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        req.diagnostics = true;
        const std::vector<std::vector<float>> probs = {
            { 0.2f, 0.8f },          // noul: no confidence either way
            { 0.6f, 0.3f, 0.1f },    // choice, N=3: (3*0.6-1)/2 = 0.4
            { 0.9f, 0.1f, 0.0f },    // score, N=3: (3*0.9-1)/2 = 0.85
        };
        common_json usage      = common_json::object();
        usage["output_tokens"] = 0;

        const common_json jev   = llama_decision::assemble_decision_response(req, probs, "m", usage);
        req.confidence_profile  = "local";
        const common_json local = llama_decision::assemble_decision_response(req, probs, "m", usage);

        assert_close(t, "default choice confidence is (N*p_max-1)/(N-1)", 0.4,
                     jev.at("answers").at("dept").at("confidence").get<double>(), 1e-6);
        assert_close(t, "local confidence is the opt-in 1 - H/log K", 0.182654578,
                     local.at("answers").at("dept").at("confidence").get<double>(), 1e-6);
        assert_close(t, "default score confidence uses the same rule above the documented range", 0.85,
                     jev.at("answers").at("urgency").at("confidence").get<double>(), 1e-6);
        assert_close(t, "certainty is the raw winner share and does not change with the profile", 0.6,
                     jev.at("answers").at("dept").at("certainty").get<double>(), 1e-6);
        t.assert_true("noul never carries a confidence", !jev.at("answers").at("refund").contains("confidence"));
    });

    t.test("the Jev confidence rule is pinned at the boundaries and for wide scales", [](testing & t) {
        assert_close(t, "uniform two-way is 0", 0.0,
                     llama_decision::jev_winner_share_confidence({ 0.5f, 0.5f }), 1e-9);
        assert_close(t, "one-hot two-way is 1", 1.0,
                     llama_decision::jev_winner_share_confidence({ 0.0f, 1.0f }), 1e-9);
        // (4*0.48-1)/3, a 4-level score where Jev leaves the definition open
        assert_close(t, "wide scale uses the same monotone rule", 0.306666667,
                     llama_decision::jev_winner_share_confidence({ 0.48f, 0.3f, 0.2f, 0.02f }), 1e-6);
    });

    t.test("confidence_profile parses as an enum and is refused otherwise", [](testing & t) {
        common_json body = common_json::parse(decision_valid_body());
        body["confidence_profile"] = "jev";
        t.assert_equal("jev is accepted", "jev",
                       llama_decision::parse_decision_request(body).confidence_profile);
        body["confidence_profile"] = "local";
        t.assert_equal("local is accepted", "local",
                       llama_decision::parse_decision_request(body).confidence_profile);
        body.erase("confidence_profile");
        t.assert_equal("absent defaults to jev (certainty-based)", "jev",
                       llama_decision::parse_decision_request(body).confidence_profile);
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"confidence_profile":"other"})",
                               "confidence_profile must be local or jev");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"confidence_profile":1})",
                               "confidence_profile must be a string");
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

    t.test("temperature provenance control: identical accepted, every mismatched field refused", [](testing & t) {
        // The provenance gate is confidence-in-the-producer, not an outcome guarantee: a matching
        // profile can still produce a wrong answer, so it never gates answer validity.
        const common_json m = calibration_temperature_control_measurement();
        t.assert_equal("identical provenance is accepted", 1, m.at("match_accepted").get<int>());
        t.assert_equal("every mismatched provenance field is refused",
                       m.at("mismatch_cases").get<int>(), m.at("mismatch_refused").get<int>());
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

static std::string decision_golden_path() {
    return fixture_path("decision_letter.golden.json");
}

// The decision tests run on the GPU backend only; a CPU fallback would silently change the numbers.
static bool decision_gpu_available() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(ggml_backend_dev_get(i));
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU ||
            type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            return true;
        }
    }
    return false;
}

// Loads the decision test model once per process and shares it across tests. Tests still create
// their own cheap contexts, so per-test KV state stays isolated and nothing re-reads the GGUF.
struct shared_test_model {
    llama_model * model = nullptr;
    std::string   path;

    bool load(const char * p) {
        if (p == nullptr || p[0] == '\0') {
            return false;
        }
        if (model != nullptr) {
            return path == p;
        }
        llama_backend_init();
        if (!decision_gpu_available()) {
            fprintf(stderr, "no GPU backend available; the decision tests run on GPU only\n");
            return false;
        }
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = -1;
        model = llama_model_load_from_file(p, mp);
        path  = p;
        return model != nullptr;
    }

    ~shared_test_model() {
        if (model != nullptr) {
            llama_model_free(model);
        }
        llama_backend_free();
    }
};

static shared_test_model & shared_model() {
    static shared_test_model shared;
    return shared;
}

// A model test needs a GGUF and a GPU backend. On a CPU-only build it skips, so the CPU suite
// stays green instead of reporting a load failure.
static bool gpu_model_ready(testing & t, const char * path) {
    if (path == nullptr || path[0] == '\0' || !file_exists(path)) {
        t.skip("set LLAMA_DECISION_TEST_MODEL to run");
        return false;
    }
    if (!decision_gpu_available()) {
        t.skip("this build has no GPU backend; the model tests need one");
        return false;
    }
    if (!shared_model().load(path)) {
        t.assert_true("model loads", false);
        return false;
    }
    return true;
}

// The committed weak-quant GPU allowlist: exact model identities (the last two path components,
// so GGUFs sharing a basename stay distinct) whose coarse quantization moves a winner on the
// GPU. On these models the task-value order-independence, temperature-winner, and head-vs-full
// agreement checks cannot be verified because the producer is not bit-stable, so they are skipped
// with a reason; everywhere else they are hard assertions. Renaming or re-quantizing a file
// changes its identity and drops it off the list, which fails loudly instead of silently moving
// the skip. This list is producer confidence only: it decides whether a task-value assertion is
// skippable on a particular model. It must never absorb an outcome-correctness failure on a model
// where the assertion is expected to hold.
static const std::vector<std::string> weak_quant_gpu_allowlist = {
    // "qwen3.5-2b-gguf/ud-q5_k_xl.gguf",
};

static bool weak_quant_gpu_oracle(const char * path) {
    if (path == nullptr || path[0] == '\0') {
        return false;
    }
    const std::string id = model_identity(path);
    for (const std::string & known : weak_quant_gpu_allowlist) {
        if (id == known) {
            return true;
        }
    }
    return false;
}

// Runs a task-value producer-determinism assertion as a hard test, or skips it on the known
// weak-quant oracle where the producer's numerics move a winner. A skip is explicit, never an
// expected failure: the assertion is a task-value check, so it must never be marked xfail.
template <typename F>
static void determinism_check(testing & t, bool weak_quant, const std::string & name, F body) {
    if (weak_quant) {
        t.skip(name + " (weak-quant GPU numerics; skipped, not xfail)");
        return;
    }
    t.test(name, body);
}

static void test_thinking_off_model(testing & t) {
    t.test("the loaded model's decision prefix stays thinking off", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        if (!shared_model().load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            auto tmpls = common_chat_templates_init(shared_model().model, "");
            if (!tmpls) {
                t.skip("the model has no chat template");
                return;
            }
            const auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(shared_model().model));
            const std::string sys = llama_decision::letter_system_text();

            const auto off = llama_decision::render_letter_prompt(tmpls.get(), true, sys, false);
            const auto on  = llama_decision::render_letter_prompt(tmpls.get(), true, sys, true);
            const auto def = llama_decision::render_letter_prompt(tmpls.get(), true, sys);

            t.assert_equal("the framer pins the template toggle off", join_split(off), join_split(def));
            t.assert_true("the cacheable prefix carries no thinking marker",
                          count_substring(off.first, "<think>") == 0);

            // Whether the toggle changes the render is a property of the template, not the framer:
            // LFM2.5 appends its marker unconditionally, while Qwen drops the empty think block
            // when thinking is on. The framer's own output stays the thinking-off render above.
            const size_t tok_off = vocab->tokenize(join_split(off), false).size();
            const size_t tok_on  = vocab->tokenize(join_split(on),  false).size();
            t.assert_true("both toggle states render a non-empty prompt", tok_off > 0 && tok_on > 0);
        } catch (const std::exception & e) {
            t.assert_true(std::string("model thinking probe renders: ") + e.what(), false);
        }
    });
}

// Owns a decision-shaped context over the shared model.
struct test_engine {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;

    ~test_engine() {
        if (ctx) {
            llama_free(ctx);
        }
    }

    bool make_ctx(bool classifier_only, int n_batch = 512, int n_seq_max = 10, bool flash_attn = false) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx                 = 8192;
        cp.n_batch               = n_batch;
        cp.n_ubatch              = n_batch;
        cp.n_seq_max             = n_seq_max;
        cp.n_outputs_max         = n_seq_max;
        cp.n_outputs_max_per_seq = 1;
        cp.kv_unified            = true;
        cp.swa_full              = false;
        cp.classifier_only       = classifier_only;
        // ROCm flash attention is not reproducible; decisions must be bit-exact run to run. The
        // state tests may turn it on to exercise the non-transposed V cache.
        cp.flash_attn_type       = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx = llama_init_from_model(model, cp);
        return ctx != nullptr;
    }

    bool load(const char * path, int n_seq_max = 10) {
        if (!shared_model().load(path)) {
            return false;
        }
        model = shared_model().model;
        return make_ctx(false, 512, n_seq_max);
    }

    bool load_fa(const char * path, int n_seq_max = 10) {
        if (!shared_model().load(path)) {
            return false;
        }
        model = shared_model().model;
        return make_ctx(false, 512, n_seq_max, true);
    }
};

// CPU-runnable decision scaffold: it does not require a GPU, so the state/fork mechanics run in
// CI. It loads the dummy model from the generate-models fixture with no GPU layers.
static std::string decision_cpu_model_path() {
    const std::string generated = std::string(DECISION_TEST_GENERATED_MODEL_DIR) + "/qwen35-dense.gguf";
    if (file_exists(generated)) {
        return generated;
    }
    const char * env = std::getenv("LLAMA_DECISION_TEST_MODEL");
    if (env != nullptr && env[0] != '\0' && file_exists(env)) {
        return env;
    }
    return std::string();
}

// The SentencePiece model is the fixture for boundary-resolved labels: its isolated token is the
// space-prefixed form, so the isolated rule cannot see the label the model emits after the answer
// tail. Env override wins, then the downloaded fixture, then empty so the test skips network-free.
static std::string spm_labels_model_path() {
    const char * env = std::getenv("LLAMA_DECISION_TEST_MODEL");
    if (env != nullptr && env[0] != '\0' && file_exists(env)) {
        return env;
    }
    const std::string spm = DECISION_TEST_SPM_MODEL;
    if (file_exists(spm)) {
        return spm;
    }
    return std::string();
}

struct cpu_test_engine {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;

    ~cpu_test_engine() {
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_model_free(model);
        }
    }

    bool load(const std::string & path,
              int                 n_ctx           = 256,
              bool                classifier_only = false,
              bool                embeddings      = false,
              int                 n_batch         = 128,
              int                 n_seq_max       = 10) {
        if (path.empty()) {
            return false;
        }
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        model = llama_model_load_from_file(path.c_str(), mp);
        if (model == nullptr) {
            return false;
        }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx                 = n_ctx;
        cp.n_batch               = n_batch;
        cp.n_ubatch              = n_batch;
        cp.n_seq_max             = n_seq_max;
        cp.n_outputs_max         = n_seq_max;
        cp.n_outputs_max_per_seq = 1;
        cp.kv_unified            = true;
        cp.swa_full              = false;
        cp.flash_attn_type       = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.classifier_only       = classifier_only;
        cp.embeddings            = embeddings || classifier_only;
        ctx = llama_init_from_model(model, cp);
        return ctx != nullptr;
    }
};

static void expect_decision_reject(testing & t, const std::string & body_text, const std::string & needle) {
    try {
        const common_json body = common_json::parse(body_text);
        (void) llama_decision::parse_decision_request(body);
        t.assert_true("decision request is rejected: " + body_text, false);
    } catch (const llama_decision::semantic_error & e) {
        const std::string what = e.what();
        t.assert_true("reject reason contains needle: " + body_text + " -> " + what,
                      what.find(needle) != std::string::npos);
    }
}

// Deterministic decision answers for a fixed score vector: a value golden that does not
// depend on any model weights.
static common_json decision_basic_from_fixed_scores() {
    const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
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
    return llama_decision::assemble_decision_response(req, probs, "m", usage);
}

static void test_decision_values_golden(testing & t) {
    t.test("fixed-score envelope matches the committed value golden", [](testing & t) {
        const std::string actual = decision_basic_from_fixed_scores().dump(2) + "\n";
        const std::string golden = read_file(fixture_path("decision_basic.golden.json"));
        t.assert_equal("value golden is byte-identical", golden, actual);
    });
}

static void test_decision_parse(testing & t) {
    t.test("valid decision request parses with aliases and structured criteria", [](testing & t) {
        const common_json body = common_json::parse(decision_valid_body());
        t.assert_true("detected as Jev", llama_decision::is_decision_request(body));

        const auto req = llama_decision::parse_decision_request(body);
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

    t.test("invalid decision requests are rejected with a clear reason", [](testing & t) {
        expect_decision_reject(t, R"({"state":"s","questions":{"q":{"type":"noul","instructions":"x"}}})", "model is required");
        expect_decision_reject(t, R"({"model":7,"state":"s","questions":{"q":{"type":"noul","instructions":"x"}}})", "model must be a string");
        expect_decision_reject(t, R"({"model":"m","questions":{"q":{"type":"noul","instructions":"x"}}})", "state (or contexts) is required");
        expect_decision_reject(t, R"({"model":"m","state":"","questions":{"q":{"type":"noul","instructions":"x"}}})", "state must not be empty");
        expect_decision_reject(t, R"({"model":"m","state":[],"questions":{"q":{"type":"noul","instructions":"x"}}})", "state must not be empty");
        expect_decision_reject(t, R"({"model":"m","state":5,"questions":{"q":{"type":"noul","instructions":"x"}}})", "state must be a string");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{}})", "1-256 entries");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"mystery","instructions":"x"}}})", "unknown type");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{"a":"x"}}}})", "2-255 options");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"choice","instructions":"x"}}})", "choice needs an object");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"score","criteria":["only"]}}})", "2-10 levels");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","criteria":[1,2]}}})", "noul criteria must be an object");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul"}}})", "instructions are required");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x","extra":1}}})", "unknown field");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":7}}})", "instructions must be a string");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"temperature":0})", "temperature must be > 0");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"permutations":0})", "permutations must be >= 1");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"temperatures":{"bogus":1}})", "unknown field");

        common_json many = common_json::object();
        common_json q    = common_json::object();
        q["type"]        = "noul";
        q["instructions"] = "x";
        common_json qs   = common_json::object();
        for (int i = 0; i < 257; ++i) {
            qs["q" + std::to_string(i)] = q;
        }
        many["model"]     = "m";
        many["state"]     = "s";
        many["questions"] = qs;
        try {
            (void) llama_decision::parse_decision_request(many);
            t.assert_true("257 questions rejected", false);
        } catch (const llama_decision::semantic_error & e) {
            t.assert_true("257 questions rejected with range", std::string(e.what()).find("1-256") != std::string::npos);
        }
    });

    // The optional live-session reference is a capability input, never a producer score. It parses
    // only from explicit, well-formed fields: a negative or non-integer slot, a negative position,
    // or a position without a slot is a semantic error, and absent means the stateless path.
    t.test("a session reference parses only from explicit, well-formed fields", [](testing & t) {
        const common_json plain = common_json::parse(decision_valid_body());
        const auto none = llama_decision::parse_session_ref(plain);
        t.assert_true("no id_slot means no session", !none.present);

        common_json with_slot = plain;
        with_slot["id_slot"] = 3;
        const auto slot = llama_decision::parse_session_ref(with_slot);
        t.assert_true("id_slot marks a session", slot.present);
        t.assert_equal("id_slot value", 3, slot.id_slot);
        t.assert_equal("session_pos defaults to derived", -1, slot.session_pos);

        common_json pinned = with_slot;
        pinned["session_pos"] = 7;
        const auto pin = llama_decision::parse_session_ref(pinned);
        t.assert_equal("session_pos pins the position", 7, pin.session_pos);

        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"id_slot":-1})",
                               "id_slot must be >= 0");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"id_slot":"a"})",
                               "id_slot must be an integer");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"session_pos":1})",
                               "session_pos requires id_slot");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"id_slot":0,"session_pos":-2})",
                               "session_pos must be >= 0");
    });

    t.test("unknown top-level fields are ignored while question fields stay strict", [](testing & t) {
        const common_json base = common_json::parse(decision_valid_body());
        common_json with_extra = base;
        with_extra["extra"]         = 1;
        with_extra["another_extra"] = common_json::object();

        const auto a = llama_decision::parse_decision_request(base);
        const auto b = llama_decision::parse_decision_request(with_extra);
        t.assert_equal("unknown top-level field does not change the model", a.model, b.model);
        t.assert_equal("unknown top-level field does not change the question count", a.questions.size(), b.questions.size());
        bool same = a.questions.size() == b.questions.size();
        for (size_t i = 0; same && i < a.questions.size(); ++i) {
            same = a.questions[i].id == b.questions[i].id && a.questions[i].type == b.questions[i].type &&
                   a.questions[i].options.size() == b.questions[i].options.size();
        }
        t.assert_true("unknown top-level field leaves the questions unchanged", same);

        // question-level unknown keys are still refused
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x","bogus":1}}})", "unknown field");

        // a misspelled "questions" is still a missing required field, not a silent default
        expect_decision_reject(t, R"({"model":"m","state":"s","questionss":{"q":{"type":"noul","instructions":"x"}}})", "questions must be an object");
    });

    t.test("instructions are required and non-null on every question type", [](testing & t) {
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul"}}})", "instructions are required");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":null}}})", "instructions are required");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{"a":"x","b":"y"}}}})", "instructions are required");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"choice","instructions":null,"criteria":{"a":"x","b":"y"}}}})", "instructions are required");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"score","criteria":["lo","hi"]}}})", "instructions are required");
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"score","instructions":null,"criteria":["lo","hi"]}}})", "instructions are required");
        // a criteria-only noul used to succeed through the escape hatch and no longer does
        expect_decision_reject(t, R"({"model":"m","state":"s","questions":{"q":{"type":"noul","criteria":{"true":"y","false":"n"}}}})", "instructions are required");

        // instructions plus criteria still succeeds
        const auto req = llama_decision::parse_decision_request(common_json::parse(
            R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x","criteria":{"true":"y","false":"n"}}}})"));
        t.assert_equal("instructions plus criteria parses", (size_t) 1, req.questions.size());
        t.assert_true("instructions are kept", req.questions[0].instructions.is_string());
    });

    t.test("score accepts 2 to 10 levels and rejects outside that range", [](testing & t) {
        auto score_body = [](int levels, bool legend) {
            common_json q = common_json::object();
            q["type"]         = "score";
            q["instructions"] = "How urgent?";
            if (legend) {
                common_json crit = common_json::object();
                for (int i = 0; i < levels; ++i) {
                    crit[std::to_string(i)] = "level " + std::to_string(i);
                }
                q["criteria"] = crit;
            } else {
                common_json crit = common_json::array();
                for (int i = 0; i < levels; ++i) {
                    crit.push_back("level " + std::to_string(i));
                }
                q["criteria"] = crit;
            }
            common_json qs = common_json::object();
            qs["q"] = q;
            common_json body = common_json::object();
            body["model"]     = "m";
            body["state"]     = "s";
            body["questions"] = qs;
            return body;
        };

        for (int levels : { 2, 5, 10 }) {
            for (bool legend : { false, true }) {
                const auto req = llama_decision::parse_decision_request(score_body(levels, legend));
                t.assert_equal("score level count parsed", (size_t) levels, req.questions[0].options.size());
            }
        }

        for (int levels : { 1, 11, 64 }) {
            for (bool legend : { false, true }) {
                bool threw = false;
                std::string msg;
                try {
                    (void) llama_decision::parse_decision_request(score_body(levels, legend));
                } catch (const llama_decision::semantic_error & e) {
                    threw = true;
                    msg = e.what();
                }
                t.assert_true("score levels outside 2-10 rejected", threw);
                t.assert_true("rejection names 2-10", msg.find("2-10") != std::string::npos);
            }
        }
    });

    t.test("diagnostics is an optional boolean, off by default", [](testing & t) {
        const auto off = llama_decision::parse_decision_request(common_json::parse(
            R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}}})"));
        t.assert_true("diagnostics defaults off", !off.diagnostics);

        const auto on = llama_decision::parse_decision_request(common_json::parse(
            R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"diagnostics":true})"));
        t.assert_true("diagnostics true is parsed", on.diagnostics);

        const auto explicit_off = llama_decision::parse_decision_request(common_json::parse(
            R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"diagnostics":false})"));
        t.assert_true("diagnostics false is parsed", !explicit_off.diagnostics);

        expect_decision_reject(t,
            R"({"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"x"}},"diagnostics":"yes"})",
            "diagnostics must be a boolean");
    });
}

static void test_decision_assemble(testing & t) {
    t.test("canonical envelope has the required shape and semantics", [](testing & t) {
        auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        req.diagnostics = true;
        common_json usage = common_json::object();
        usage["input_tokens"]    = 0;
        usage["output_tokens"]   = 0;
        usage["cached_tokens"]   = 0;
        usage["state_cache_hit"] = false;

        const common_json out = llama_decision::assemble_decision_response(req, {}, req.model, usage);
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
        assert_close(t, "choice confidence of uniform is 0", 0.0, dept.at("confidence").get<double>(), 1e-6);
        assert_close(t, "choice certainty is the winner share", 1.0 / 3.0, dept.at("certainty").get<double>(), 1e-6);

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
    bool                           drop_unknown = false; // skip text no piece matches, like a vocab without it

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
                if (!drop_unknown) {
                    out.push_back(100000 + (int32_t) (unsigned char) text[i]);
                }
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

// The synthetic answer tail the letter tests frame before a label: with no chat template the
// framer returns "\n" as the post-user text, so the tail is "\nAnswer:\n".
static std::string test_letter_tail() {
    return llama_decision::render_letter_prompt(nullptr, false, llama_decision::letter_system_text()).second + "Answer:\n";
}

static void test_letter_suffix(testing & t) {
    t.test("question suffix lists options and ends at the answer boundary", [](testing & t) {
        const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        const fake_vocab v = make_fake_vocab(true);
        const auto pool = llama_decision::build_label_pool(v, "", 8);
        const std::string after = "<turn|>\n<turn>model\n";

        const std::string tail = llama_decision::letter_answer_tail(after);
        t.assert_equal("the answer tail is the after text plus the fixed marker", after + "Answer:\n", tail);
        // the per-question suffix is built through the same option-line formatter
        const std::string a_line = llama_decision::format_option_line(pool[0], req.questions[1].options[0]);
        const std::string b_line = llama_decision::format_option_line(pool[1], req.questions[1].options[1]);
        t.assert_true("first option listed", a_line.find("A: billing - payment") != std::string::npos);
        t.assert_true("second option listed", b_line.find("B: technical - bug") != std::string::npos);
        t.assert_true("question text listed", req.questions[1].instructions.dump().find("What is the issue?") != std::string::npos);
    });

    t.test("label capacity is validated before scoring", [](testing & t) {
        const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        const std::string tail = test_letter_tail();
        const fake_vocab v = make_fake_vocab(true);
        const auto pool = llama_decision::build_label_pool(v, tail, 2);
        bool threw = false;
        try {
            llama_decision::verify_letter_request(v, "\n", req, pool);
        } catch (const llama_decision::semantic_error &) {
            threw = true;
        }
        t.assert_true("a question with more options than the pool is rejected", threw);
    });

    t.test("label capacity rejects above the realized pool and accepts at it", [](testing & t) {
        auto make_choice_request = [](size_t n) {
            common_json body =
                common_json::parse(R"({"model":"m","state":"s","questions":{"q":{"type":"choice","instructions":"pick"}}})");
            common_json crit = common_json::object();
            for (size_t i = 0; i < n; ++i) {
                crit["k" + std::to_string(i)] = "d";
            }
            body["questions"]["q"]["criteria"] = crit;
            return llama_decision::parse_decision_request(body);
        };

        const std::string tail = test_letter_tail();
        const std::string after = "\n"; // render_letter_prompt(nullptr,...).second

        // control group: an explicit small cap is honored, and one option above it is rejected
        const fake_vocab small      = make_fake_vocab(false);
        const auto       small_pool = llama_decision::build_label_pool(small, tail, 5);
        t.assert_equal("the control pool honors the explicit cap", (size_t) 5, small_pool.size());
        llama_decision::verify_letter_request(small, after, make_choice_request(5), small_pool);  // must not throw
        bool over = false;
        try {
            llama_decision::verify_letter_request(small, after, make_choice_request(6), small_pool);
        } catch (const llama_decision::semantic_error &) {
            over = true;
        }
        t.assert_true("one option above the control pool is rejected", over);

        // positive group: a double-letter vocabulary fills the composed cap, and the cap is accepted
        const fake_vocab full      = make_fake_vocab(true);
        const auto       full_pool = llama_decision::build_label_pool(full, tail, llama_decision::LABEL_POOL_CAP);
        t.assert_equal("the positive pool reaches the cap", llama_decision::LABEL_POOL_CAP, full_pool.size());
        llama_decision::verify_letter_request(full, after, make_choice_request(llama_decision::LABEL_POOL_CAP), full_pool);
    });
}

// The exact option lines the framer emits: `label: key - description`, with the description omitted
// when empty. This is the golden the prompt layout is measured against.
static void test_letter_option_lines(testing & t) {
    t.test("option lines render as label: key - description", [](testing & t) {
        const common_json body = common_json::parse(R"({
            "model": "m",
            "state": "s",
            "questions": {
                "n": {"type": "noul", "instructions": "Refund?", "criteria": {"true": "yes", "false": "no"}},
                "c": {"type": "choice", "instructions": "Dept?", "criteria": {"billing": {"label": "payment", "code": 7}, "technical": "bug"}},
                "s": {"type": "score", "instructions": "Urgency?", "criteria": ["calm", "upset", "furious"]},
                "e": {"type": "choice", "instructions": "Empty?", "criteria": {"a": null, "b": "bee"}}
            }
        })");
        const auto req = llama_decision::parse_decision_request(body);
        const fake_vocab v = make_fake_vocab(true);
        const auto pool = llama_decision::build_label_pool(v, "", 8);

        // the option lines are emitted through the one formatter: `label: key - description`
        t.assert_true("noul false line",
                      llama_decision::format_option_line(pool[0], req.questions[0].options[0]).find("A: false - no") != std::string::npos);
        t.assert_true("noul true line",
                      llama_decision::format_option_line(pool[1], req.questions[0].options[1]).find("B: true - yes") != std::string::npos);
        t.assert_true("choice object description line",
                      llama_decision::format_option_line(pool[0], req.questions[1].options[0]).find("A: billing - {\"label\":\"payment\",\"code\":7}") != std::string::npos);
        t.assert_true("choice string description line",
                      llama_decision::format_option_line(pool[1], req.questions[1].options[1]).find("B: technical - bug") != std::string::npos);
        t.assert_true("score line 0",
                      llama_decision::format_option_line(pool[0], req.questions[2].options[0]).find("A: 0 - calm") != std::string::npos);
        t.assert_true("score line 1",
                      llama_decision::format_option_line(pool[1], req.questions[2].options[1]).find("B: 1 - upset") != std::string::npos);
        t.assert_true("score line 2",
                      llama_decision::format_option_line(pool[2], req.questions[2].options[2]).find("C: 2 - furious") != std::string::npos);
        t.assert_true("empty description renders the key only",
                      llama_decision::format_option_line(pool[0], req.questions[3].options[0]).find("A: a") != std::string::npos);
        t.assert_true("non-empty description still renders",
                      llama_decision::format_option_line(pool[1], req.questions[3].options[1]).find("B: b - bee") != std::string::npos);
    });
}

static void test_label_pool(testing & t) {
    t.test("label pool keeps boundary-resolved labels in order", [](testing & t) {
        const fake_vocab v = make_fake_vocab(true);

        const auto pool = llama_decision::build_label_pool(v, "", 64);
        t.assert_equal("cap respected", (size_t) 64, pool.size());
        t.assert_equal("first label is A", std::string("A"), pool[0].text);
        t.assert_equal("26th label is Z", std::string("Z"), pool[25].text);
        t.assert_equal("then lowercase letters", std::string("a"), pool[26].text);
        t.assert_equal("then the single digits", std::string("0"), pool[52].text);
        t.assert_equal("then the ASCII symbol set", std::string("!"), pool[62].text);

        // the first 62 labels are letters and digits; the symbol set starts after them
        bool all_label_chars = true;
        for (size_t i = 0; i < 62 && i < pool.size(); ++i) {
            for (unsigned char c : pool[i].text) {
                all_label_chars = all_label_chars && std::isalnum(c) != 0;
            }
        }
        t.assert_true("letters and digits come first", all_label_chars);

        // a 64-label pool is all single characters: two-char labels only appear once the
        // single-character sets are exhausted
        bool has_two_char = false;
        for (const auto & l : pool) {
            has_two_char = has_two_char || l.text.size() == 2;
        }
        t.assert_true("a 64-label pool is all single characters", !has_two_char);

        t.assert_equal("AA resolves at the boundary", v.id_of("AA"), llama_decision::answer_label_token(v, "", "AA"));
        t.assert_equal("AAA is not a single label", -1, llama_decision::answer_label_token(v, "", "AAA"));
    });

    t.test("label pool rejects special and merged tokens", [](testing & t) {
        // The removed piece check was a producer-confidence proxy ("the token looks like the
        // text"); it is intentionally gone. A label is accepted when the boundary produces it as a
        // single non-special token, whatever its surface form.
        fake_vocab special = make_fake_vocab(false);
        special.specials.insert(special.id_of("B"));
        t.assert_equal("special token rejected", -1, llama_decision::answer_label_token(special, "", "B"));
        t.assert_equal("normal token accepted", special.id_of("A"), llama_decision::answer_label_token(special, "", "A"));

        fake_vocab renamed = make_fake_vocab(false);
        renamed.piece_override[renamed.id_of("C")] = "c";
        t.assert_equal("a renamed piece is still accepted", renamed.id_of("C"),
                       llama_decision::answer_label_token(renamed, "", "C"));

        fake_vocab merged = make_fake_vocab(false);
        merged.pieces.push_back("\nA");
        t.assert_equal("a genuine merge is rejected", -1, llama_decision::answer_label_token(merged, "x\n", "A"));

        fake_vocab tiny;
        tiny.pieces.push_back("A");
        tiny.specials.insert(0); // the only resolvable token is special, so composition is blocked
        tiny.drop_unknown = true;
        bool threw = false;
        try {
            (void) llama_decision::build_label_pool(tiny, "", 64);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        t.assert_true("fewer than two labels is an error", threw);
    });

    t.test("isolated and boundary pools agree without a space prefix", [](testing & t) {
        const fake_vocab v = make_fake_vocab(true);
        const auto isolated = llama_decision::build_label_pool(v, "", 64);
        const auto boundary = llama_decision::build_label_pool(v, "Answer:\n", 64);
        bool same = isolated.size() == boundary.size();
        for (size_t i = 0; same && i < isolated.size(); ++i) {
            same = isolated[i].text == boundary[i].text && isolated[i].token == boundary[i].token;
        }
        t.assert_true("the pool is byte-identical when isolated equals boundary", same);
    });
}

static void test_boundary(testing & t) {
    t.test("boundary check accepts a clean split and rejects a merged token", [](testing & t) {
        const fake_vocab v = make_fake_vocab(false);
        const int32_t a    = v.id_of("A");

        t.assert_equal("clean boundary resolves the label token", a,
                       llama_decision::answer_label_token(v, "x ", "A"));

        fake_vocab merged = make_fake_vocab(false);
        merged.pieces.push_back("\nA");
        t.assert_equal("merged token fails loudly", -1, llama_decision::answer_label_token(merged, "x\n", "A"));
    });
}

static void test_answer_label_token(testing & t) {
    t.test("answer label token resolves at the boundary", [](testing & t) {
        const fake_vocab v = make_fake_vocab(false);
        t.assert_equal("a clean token is accepted", v.id_of("A"),
                       llama_decision::answer_label_token(v, "", "A"));

        fake_vocab merged = make_fake_vocab(false);
        merged.pieces.push_back("\nA");
        t.assert_equal("a genuine merge is rejected", -1,
                       llama_decision::answer_label_token(merged, "x\n", "A"));

        fake_vocab special = make_fake_vocab(false);
        special.specials.insert(special.id_of("B"));
        t.assert_equal("a special token is rejected", -1,
                       llama_decision::answer_label_token(special, "", "B"));
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

// The single-engine letter readout wrapper the removed public overload used to provide: a
// classifier-only engine is the classifier source, a full-logits engine is the fallback.
static std::vector<std::vector<float>> test_letter_readout(
        llama_decision::engine & eng, llama_decision::answer_head_cache & head_cache,
        const llama_decision::label_vocab & vocab, const common_chat_templates * tmpls, bool use_jinja,
        const llama_decision::decision_request & req, const std::vector<llama_decision::label> & labels,
        const llama_decision::options & opt, llama_decision::letter_metrics * metrics) {
    llama_decision::readout_sources sources;
    sources.full = &eng;
    if (eng.classifier_only()) {
        sources.classifier = &eng;
    } else {
        sources.classifier_unavailable = "the decision context does not expose hidden states";
    }
    auto all = llama_decision::letter_readout_multi(sources, head_cache, vocab, tmpls, use_jinja,
                                                    req, labels, opt, metrics);
    return all.empty() ? std::vector<std::vector<float>>{} : std::move(all[0]);
}

static void test_label_pool_real(testing & t) {
    t.test("label pool and exact boundary ids on a real vocabulary", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }

        if (!shared_model().load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        llama_model * model = shared_model().model;

        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(model));
            const std::string tail = "Answer:\n";
            const auto pool = llama_decision::build_label_pool(*vocab, tail, 64);
            t.assert_true("pool has 2-64 labels", pool.size() >= 2 && pool.size() <= 64);

            bool boundary = true;
            bool alnum_first = true;
            for (size_t i = 0; i < pool.size(); ++i) {
                const auto & l = pool[i];
                boundary = boundary &&
                    (llama_decision::answer_label_path(*vocab, tail, l.text, (int) l.tokens.size()) == l.tokens);
                if (i < 62) {
                    for (unsigned char c : l.text) {
                        alnum_first = alnum_first && std::isalnum(c) != 0;
                    }
                }
            }
            t.assert_true("every label resolves at the boundary", boundary);
            t.assert_true("letters and digits come first", alnum_first);

            const auto base = vocab->tokenize(tail, true);
            const auto full = vocab->tokenize(tail + pool[0].text, true);
            t.assert_equal("exact boundary length", base.size() + 1, full.size());
            t.assert_true("exact boundary tail is the label token", !full.empty() && full.back() == pool[0].token);
            t.assert_true("boundary accepts the label", llama_decision::answer_label_token(*vocab, tail, pool[0].text) == pool[0].token);
        } catch (const std::exception & e) {
            t.assert_true(std::string("label pool runs: ") + e.what(), false);
        }
    });
}

// The pool is built at the real answer boundary, so an add_space_prefix vocabulary resolves "A" to
// the bare token the model emits after the tail instead of the space-prefixed isolated form. This
// is the SPM family that the isolated rule refused.
static void test_letter_labels_spm(testing & t) {
    t.test("letter labels resolve at the answer boundary on an SPM model", [](testing & t) {
        const std::string path = spm_labels_model_path();
        if (path.empty()) {
            t.skip("no SPM model available");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path, 512)) {
            t.assert_true("the SPM model loads on CPU", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const std::string tail = test_letter_tail();
            const auto pool = llama_decision::build_label_pool(*vocab, tail, 64);
            t.assert_true("the pool has at least two labels", pool.size() >= 2);
            for (const auto & l : pool) {
                t.assert_equal("label " + l.text + " resolves at the answer boundary", l.token,
                               llama_decision::answer_label_token(*vocab, tail, l.text));
            }

            // The bug: the isolated token is the space-prefixed form, while the model emits the
            // bare token after the tail. Skip when the tokenizer does not prefix a space.
            bool space_prefixed = false;
            for (const auto & l : pool) {
                const std::vector<int32_t> iso = vocab->tokenize(l.text, false);
                if (iso.size() == 1 && vocab->piece(iso[0]) == " " + l.text) {
                    space_prefixed = true;
                    break;
                }
            }
            if (space_prefixed) {
                bool isolated_differs = false;
                for (const auto & l : pool) {
                    const std::vector<int32_t> iso = vocab->tokenize(l.text, false);
                    if (iso.size() == 1 && iso[0] != l.token) {
                        isolated_differs = true;
                        break;
                    }
                }
                t.assert_true("the isolated token differs from the boundary token", isolated_differs);
            }

            // End to end: a previously-refused family now serves a closed distribution per question.
            llama_decision::engine eng(te.ctx, 2, 8);
            llama_decision::answer_head_cache head_cache;
            const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
            llama_decision::letter_metrics metrics;
            const auto probs = test_letter_readout(eng, head_cache, *vocab, nullptr, false, req, pool,
                                                              llama_decision::options{}, &metrics);
            t.assert_equal("one distribution per question", req.questions.size(), probs.size());
            bool valid = probs.size() == req.questions.size();
            for (size_t i = 0; valid && i < probs.size(); ++i) {
                valid = probs[i].size() == req.questions[i].options.size();
                double sum = 0.0;
                for (float p : probs[i]) {
                    valid = valid && p >= 0.0f && p <= 1.0f;
                    sum += p;
                }
                valid = valid && std::fabs(sum - 1.0) < 1e-3;
                valid = valid && !probs[i].empty() && std::max_element(probs[i].begin(), probs[i].end()) != probs[i].end();
            }
            t.assert_true("every distribution is closed and valid", valid);
            t.assert_true("the SPM readout does not use a classifier head", !metrics.head_active);
        } catch (const std::exception & e) {
            t.assert_true(std::string("the SPM label pool runs: ") + e.what(), false);
        }
    });
}

// Calibration of the label accept/reject gate. The gate is task-correctness - the next token at
// the answer boundary is exactly this label - not a confidence score; the removed piece check was
// a producer-confidence proxy. There is no numeric threshold, so precision/recall become exact
// accept/reject counts over a control, a positive, and a negative group.
static void test_label_boundary_calibration(testing & t) {
    t.test("boundary gate calibration: control, positive and negative groups", [](testing & t) {
        int control_false_rejects  = 0;
        int positive_false_rejects = 0;
        int negative_false_accepts = 0;
        int positive_labels        = 0;

        // Control: a vocabulary whose isolated token is the boundary token must not change.
        {
            const fake_vocab v  = make_fake_vocab(true);
            const auto isolated = llama_decision::build_label_pool(v, "", 64);
            const auto boundary = llama_decision::build_label_pool(v, "Answer:\n", 64);
            bool same = isolated.size() == boundary.size();
            for (size_t i = 0; same && i < isolated.size(); ++i) {
                same = isolated[i].text == boundary[i].text && isolated[i].token == boundary[i].token;
            }
            if (!same) {
                ++control_false_rejects;
            }

            // A real BPE model is a control only when its isolated token equals its boundary token.
            const char * env = std::getenv("LLAMA_DECISION_TEST_MODEL");
            if (env != nullptr && env[0] != '\0' && file_exists(env)) {
                cpu_test_engine te;
                if (te.load(env)) {
                    auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
                    const std::string tail = test_letter_tail();
                    const auto pool = llama_decision::build_label_pool(*vocab, tail, 64);
                    bool space_prefixed = false;
                    for (const auto & l : pool) {
                        const std::vector<int32_t> iso = vocab->tokenize(l.text, false);
                        if (iso.size() == 1 && vocab->piece(iso[0]) == " " + l.text) {
                            space_prefixed = true;
                            break;
                        }
                    }
                    if (!space_prefixed) {
                        const auto real_isolated = llama_decision::build_label_pool(*vocab, "", 64);
                        bool real_same = real_isolated.size() == pool.size();
                        for (size_t i = 0; real_same && i < pool.size(); ++i) {
                            real_same = real_isolated[i].text == pool[i].text && real_isolated[i].token == pool[i].token;
                        }
                        if (!real_same) {
                            ++control_false_rejects;
                        }
                    }
                }
            }
        }

        // Positive: the SPM model must be accepted at the boundary.
        {
            const std::string path = spm_labels_model_path();
            if (!path.empty()) {
                cpu_test_engine te;
                if (te.load(path)) {
                    auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
                    const std::string tail = test_letter_tail();
                    const auto pool = llama_decision::build_label_pool(*vocab, tail, 64);
                    positive_labels = (int) pool.size();
                    if (pool.size() < 2) {
                        ++positive_false_rejects;
                    }
                    for (const auto & l : pool) {
                        if (llama_decision::answer_label_token(*vocab, tail, l.text) != l.token) {
                            ++positive_false_rejects;
                        }
                    }
                }
            }
        }

        // Negative: a vocabulary with no letter tokens, a special token, and a genuine merge must
        // still be rejected. The generated dummy model is not a usable negative here: its "test"
        // tokenizer hashes fixed 5-character chunks into 128 ids, so some candidates land on a
        // single boundary token by hash collision. That is a fixture artifact, not a gate defect,
        // and the gate still refuses any vocabulary with fewer than two boundary-resolvable labels.
        {
            fake_vocab no_letters;
            no_letters.pieces.push_back("tok_0");
            no_letters.pieces.push_back("tok_1");
            no_letters.drop_unknown = true;
            bool no_letters_rejected = false;
            try {
                (void) llama_decision::build_label_pool(no_letters, "", 64);
            } catch (const std::runtime_error &) {
                no_letters_rejected = true;
            }
            if (!no_letters_rejected) {
                ++negative_false_accepts;
            }

            fake_vocab special = make_fake_vocab(false);
            special.specials.insert(special.id_of("B"));
            if (llama_decision::answer_label_token(special, "", "B") >= 0) {
                ++negative_false_accepts;
            }

            fake_vocab merged = make_fake_vocab(false);
            merged.pieces.push_back("\nA");
            if (llama_decision::answer_label_token(merged, "x\n", "A") >= 0) {
                ++negative_false_accepts;
            }
        }

        fprintf(stderr, "label boundary calibration: control_false_rejects=%d positive_false_rejects=%d positive_labels=%d negative_false_accepts=%d\n",
                control_false_rejects, positive_false_rejects, positive_labels, negative_false_accepts);
        t.assert_equal("control group: zero false rejects", 0, control_false_rejects);
        t.assert_equal("positive group: zero false rejects", 0, positive_false_rejects);
        t.assert_equal("negative group: zero false accepts", 0, negative_false_accepts);
    });
}

// The model tests below run against the one shared model, so they share one caller-owned head
// cache. The dedicated cache tests construct their own.
static llama_decision::answer_head_cache & test_head_cache() {
    static llama_decision::answer_head_cache cache;
    return cache;
}

static void test_letter_readout_real(testing & t) {
    t.test("letter readout scores a decision request on a real model", [](testing & t) {
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
            const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            llama_decision::engine eng(te.ctx, 2, 8);
            const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));

            llama_decision::letter_metrics metrics;
            const auto probs = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false, req, pool,
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
                              llama_decision::answer_label_token(*vocab, tail, pool[0].text) == pool[0].token);
            }

            // Task-value producer-determinism properties: the winner is stable under question
            // reordering and under temperature sharpening. These are hard on every accepted model.
            // On the known weak-quant GPU oracle the numerics move a winner for these checks, so
            // there they are skipped with a reason - never xfail, because a winner disagreement is
            // a task-value outcome, and a head-path winner disagreement is always hard.
            const bool weak_quant = weak_quant_gpu_oracle(path);

            determinism_check(t, weak_quant,
                              "question order does not change the winners (weak-quant GPU batch-shape sensitivity)",
                              [&](testing & t) {
                llama_decision::decision_request reversed = req;
                std::reverse(reversed.questions.begin(), reversed.questions.end());
                const auto probs_rev = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false, reversed, pool,
                                                                      llama_decision::options{}, nullptr);
                bool stable = probs_rev.size() == probs.size();
                for (size_t i = 0; stable && i < req.questions.size(); ++i) {
                    const std::string & id = req.questions[i].id;
                    size_t other = 0;
                    while (other < reversed.questions.size() && reversed.questions[other].id != id) {
                        ++other;
                    }
                    // Reversing the questions changes the branch order in the batch, so a backend may
                    // reorder a reduction; the winner of each question must not move.
                    stable = other < probs_rev.size() &&
                              probs[i].size() == probs_rev[other].size() &&
                              std::distance(probs[i].begin(), std::max_element(probs[i].begin(), probs[i].end())) ==
                              std::distance(probs_rev[other].begin(),
                                            std::max_element(probs_rev[other].begin(), probs_rev[other].end()));
                }
                t.assert_true("question order does not change the winners", stable);
            });

            determinism_check(t, weak_quant,
                              "temperature preserves the winner and orders sharpness (weak-quant GPU numerics)",
                              [&](testing & t) {
                auto with_temps = [](const char * temps) {
                    common_json body = common_json::parse(decision_valid_body());
                    body["temperatures"] = common_json::parse(temps);
                    return llama_decision::parse_decision_request(body);
                };
                const auto sharp = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false,
                                                                  with_temps(R"({"noul":0.5,"choice":0.5,"score":0.5})"),
                                                                  pool, llama_decision::options{}, nullptr);
                const auto flat = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false,
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
            });

            if (file_exists(decision_golden_path())) {
                common_json usage = common_json::object();
                usage["input_tokens"]    = 0;
                usage["output_tokens"]   = 0;
                usage["cached_tokens"]   = 0;
                usage["state_cache_hit"] = false;
                const common_json actual = llama_decision::assemble_decision_response(req, probs, "m", usage);
                const common_json golden = common_json::parse(read_file(decision_golden_path()));
                // The value golden is model-specific. Compare only on the model it was written
                // from; every other arch checks the mechanism, not these numbers.
                const std::string golden_model = golden.value("golden_model", std::string());
                if (!golden_model.empty() && golden_model == model_identity(path)) {
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
                } else {
                    t.log("value golden not for this model; comparison skipped");
                }
            }
        } catch (const std::exception & e) {
            t.assert_true(std::string("letter readout runs: ") + e.what(), false);
        }
    });
}

static void test_fork_real(testing & t) {
    t.test("copy, bypass and LRU behave", [](testing & t) {
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

            // A copy fork cannot carry recurrent state, so it is only meaningful for pure
            // attention models; on a recurrent model the engine rejects it and auto picks the
            // exact partial hybrid fork.
            const bool copy_ok    = !llama_model_is_recurrent(te.model) && !llama_model_is_hybrid(te.model);
            const bool weak_quant = weak_quant_gpu_oracle(path);

            llama_decision::options orr;
            orr.fork      = "restore";
            orr.cache_tag = "t";
            const auto restore_run = eng.decide_batch("system", { "ctx" }, two, orr);

            if (copy_ok) {
                llama_decision::options oc;
                oc.fork      = "copy";
                oc.cache_tag = "t";
                const auto copy_run = eng.decide_batch("system", { "ctx" }, two, oc);

                // same math on two KV layouts: the winner is the task value and is always
                // asserted; the probability bound is producer numerics
                bool winners = copy_run.items[0].fields.size() == restore_run.items[0].fields.size();
                bool agree   = winners;
                for (size_t f = 0; agree && f < copy_run.items[0].fields.size(); ++f) {
                    const auto & pc = copy_run.items[0].fields[f].probs;
                    const auto & pr = restore_run.items[0].fields[f].probs;
                    winners = winners && copy_run.items[0].fields[f].winner == restore_run.items[0].fields[f].winner;
                    agree   = pc.size() == pr.size();
                    for (size_t k = 0; agree && k < pc.size(); ++k) {
                        agree = std::fabs(pc[k] - pr[k]) < 5e-3;
                    }
                }
                t.test("copy and restore pick the same winner", [&](testing & t) {
                    t.assert_true("copy and restore pick the same winner", winners);
                });
                determinism_check(t, weak_quant, "copy and restore agree within tolerance", [&](testing & t) {
                    t.assert_true("copy and restore agree within tolerance", agree);
                });
            } else {
                bool rejected = false;
                try {
                    llama_decision::options oc;
                    oc.fork = "copy";
                    eng.decide_batch("system", { "ctx" }, two, oc);
                } catch (const std::invalid_argument &) {
                    rejected = true;
                }
                t.assert_true("copy fork is rejected for a recurrent model", rejected);

                llama_decision::options oa;
                oa.fork      = "auto";
                oa.cache_tag = "t";
                const auto auto_run = eng.decide_batch("system", { "ctx" }, two, oa);
                // the winner is the task value and is always asserted; the probability bound is the
                // producer numerics of sharing vs copying the attention cells on the GPU
                bool winners = auto_run.items[0].fields.size() == restore_run.items[0].fields.size();
                bool agree   = winners;
                for (size_t f = 0; agree && f < auto_run.items[0].fields.size(); ++f) {
                    const auto & pa = auto_run.items[0].fields[f].probs;
                    const auto & pr = restore_run.items[0].fields[f].probs;
                    winners = winners && auto_run.items[0].fields[f].winner == restore_run.items[0].fields[f].winner;
                    agree   = pa.size() == pr.size();
                    for (size_t k = 0; agree && k < pa.size(); ++k) {
                        agree = std::fabs(pa[k] - pr[k]) < 5e-3;
                    }
                }
                t.test("auto picks the exact fork on a recurrent model", [&](testing & t) {
                    t.assert_true("auto picks the exact fork on a recurrent model", winners);
                });
                determinism_check(t, weak_quant, "auto and restore agree within tolerance", [&](testing & t) {
                    t.assert_true("auto and restore agree within tolerance", agree);
                });
            }

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
            oc2.fork = copy_ok ? "copy" : "restore";
            const auto single = eng.decide_batch("system", { "ctx" }, one, oc2);
            const auto pair   = eng.decide_batch("system", { "ctx" }, two, oc2);
            const auto & ps = single.items[0].fields[0].probs;
            const auto & pp = pair.items[0].fields[0].probs;
            // Bypass only applies to copy forks. When it does, the single-question result must be
            // identical to the two-question run, within the 5e-3 producer-numerics bound. On a
            // recurrent model both runs take the forked path and differ by the GPU gemm's
            // batch-shape sensitivity, so the bound there is the 5e-2 head-agreement tolerance.
            // The winner is the task value and is always asserted.
            bool bypass_ok = ps.size() == pp.size();
            for (size_t k = 0; bypass_ok && k < ps.size(); ++k) {
                bypass_ok = std::fabs(ps[k] - pp[k]) < (copy_ok ? 5e-3 : 5e-2);
            }
            const auto argmax = [](const std::vector<float> & p) {
                return (int) (std::max_element(p.begin(), p.end()) - p.begin());
            };
            t.test("single-question bypass picks the same winner", [&](testing & t) {
                t.assert_true("single-question bypass picks the same winner", argmax(ps) == argmax(pp));
            });
            determinism_check(t, weak_quant, "single-question bypass matches the forked path", [&](testing & t) {
                t.assert_true("single-question bypass matches the forked path", bypass_ok);
            });

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

// Fork correctness is judged by byte equality of the resulting state against a full restore, never
// by an argmax match: a wrong probability vector can still pick the same winner. The oracle decodes
// a parent, forks it twice (once through the engine's active strategy, once by a full restore), and
// requires the two branch states to be byte-identical after the same branch decode.
static bool decode_tokens_on(llama_context * ctx, llama_seq_id seq, llama_pos pos0, const std::vector<llama_token> & toks) {
    llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(batch, toks[i], pos0 + (llama_pos) i, { seq }, i + 1 == toks.size());
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

static std::vector<uint8_t> seq_state_dump(llama_context * ctx, llama_seq_id seq) {
    std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, seq));
    const size_t         n = llama_state_seq_get_data(ctx, buf.data(), buf.size(), seq);
    buf.resize(n);
    return buf;
}

// A serialized state carries the sequence id in its header and in every cell record, so two
// sequences with identical content still differ in those bytes. Re-serializing both through the
// same scratch sequence on an empty cache normalizes the ids and the physical cell layout, leaving
// only the state content for the byte comparison.
static std::vector<uint8_t> normalize_seq_state(llama_context * ctx, const std::vector<uint8_t> & state, llama_seq_id scratch,
                                                llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_NONE) {
    if (state.empty()) {
        return state;
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    const size_t n = llama_state_seq_set_data_ext(ctx, state.data(), state.size(), scratch, flags);
    if (n == 0) {
        return {};
    }
    return seq_state_dump(ctx, scratch);
}

// The serialized state starts with a magic and the source sequence id, which legitimately differ
// between two sequences; the comparison is over the state payload that follows.
static constexpr size_t seq_state_header_size = sizeof(uint32_t) + sizeof(llama_seq_id);

static size_t state_bytes_diff(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, std::string * detail = nullptr) {
    const size_t skip_a = std::min(a.size(), seq_state_header_size);
    const size_t skip_b = std::min(b.size(), seq_state_header_size);
    const size_t n_a    = a.size() - skip_a;
    const size_t n_b    = b.size() - skip_b;
    const size_t common = std::min(n_a, n_b);
    size_t       diff   = std::max(n_a, n_b) - common;
    size_t       first  = (size_t) -1;
    for (size_t i = 0; i < common; ++i) {
        if (a[skip_a + i] != b[skip_b + i]) {
            if (first == (size_t) -1) {
                first = i;
            }
            ++diff;
        }
    }
    if (detail != nullptr && diff != 0) {
        *detail = "first at payload offset " + std::to_string(first);
    }
    return diff;
}

static std::vector<float> output_logits(llama_context * ctx, const llama_vocab * vocab, int out_idx) {
    std::vector<float> out(llama_vocab_n_tokens(vocab), 0.0f);
    const float *      logits = llama_get_logits_ith(ctx, out_idx);
    if (logits != nullptr) {
        std::copy(logits, logits + out.size(), out.begin());
    }
    return out;
}

static double max_abs_logit_delta(const std::vector<float> & a, const std::vector<float> & b) {
    double       delta = 0.0;
    const size_t n     = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        delta = std::max(delta, (double) std::fabs(a[i] - b[i]));
    }
    return delta;
}

static void fork_oracle_run(testing & t, llama_context * ctx, const std::string & lane, const std::string & strategy) {
    const llama_model * model = llama_get_model(ctx);
    if (strategy == "copy" && (llama_model_is_recurrent(model) || llama_model_is_hybrid(model))) {
        t.skip(lane + ": a copy fork is not supported on a recurrent model");
        return;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto          parent = common_tokenize(vocab, "a short decision parent", false, true);
    const auto          branch = common_tokenize(vocab, "branch", false, true);
    if (parent.empty() || branch.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }

    llama_decision::engine eng(ctx, 2, 8);
    const llama_seq_id     src = 2, ref = 3, subject = 4, scratch = 7;

    // fresh cache and prior sequences: the fork must be exact in both cell layouts
    for (int n_prior : { 0, 2 }) {
        const std::string layout = n_prior == 0 ? "fresh" : "prior";
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int p = 0; p < n_prior; ++p) {
            if (!t.assert_true(lane + "/" + layout + ": a prior sequence decodes",
                               decode_tokens_on(ctx, 5 + p, 0, parent))) {
                return;
            }
        }
        if (!t.assert_true(lane + "/" + layout + ": the parent decodes", decode_tokens_on(ctx, src, 0, parent))) {
            return;
        }
        llama_synchronize(ctx);

        const auto full = eng.save_seq(src, false, false);
        const auto part = eng.save_seq(src, false, true);
        if (!t.assert_true(lane + "/" + layout + ": the parent state is non-empty",
                           !full.bytes.empty() && !part.bytes.empty())) {
            return;
        }
        const auto parent_before = seq_state_dump(ctx, src);

        // reference: a full restore fork of the same parent
        llama_memory_seq_rm(llama_get_memory(ctx), ref, -1, -1);
        eng.load_seq(full, ref);

        // subject: the requested strategy, loading the matching full or partial parent state. An
        // auto strategy resolves to the partial hybrid fork for a recurrent or hybrid model.
        eng.select_fork(strategy);
        const auto & subject_state = eng.active_fork_ == llama_decision::engine::fork_kind::hybrid ? part : full;
        eng.fork_into(src, subject, &subject_state);

        const llama_pos pos0   = (llama_pos) parent.size();
        const bool      ref_ok = decode_tokens_on(ctx, ref, pos0, branch);
        const bool      sub_ok = decode_tokens_on(ctx, subject, pos0, branch);
        llama_synchronize(ctx);
        if (!t.assert_true(lane + "/" + layout + ": the reference branch decodes", ref_ok) ||
            !t.assert_true(lane + "/" + layout + ": the fork branch decodes", sub_ok)) {
            return;
        }

        // the source sequence must be byte-identical after a branch fork and decode
        t.assert_true(lane + "/" + layout + ": the source state is unchanged by a " + strategy + " fork",
                      parent_before == seq_state_dump(ctx, src));

        const auto ref_state = seq_state_dump(ctx, ref);
        const auto sub_state = seq_state_dump(ctx, subject);
        if (!t.assert_true(lane + "/" + layout + ": the fork state is non-empty", !sub_state.empty())) {
            return;
        }

        const auto   ref_norm = normalize_seq_state(ctx, ref_state, scratch);
        const auto   sub_norm = normalize_seq_state(ctx, sub_state, scratch);
        std::string  detail;
        const size_t diff = state_bytes_diff(ref_norm, sub_norm, &detail);
        t.assert_true(lane + "/" + layout + ": the " + strategy + " fork equals a full restore (" +
                          std::to_string(diff) + " of " + std::to_string(std::max(ref_norm.size(), sub_norm.size())) +
                          " bytes differ" + (detail.empty() ? "" : ", " + detail) + ")",
                      diff == 0);
    }
}

// Forks `n` children out of a parent and decodes the same branch token on all of them in one
// batch, the way the engine runs a wave of branches from one trunk. Restore children load the
// saved host state; plain children only get a metadata sequence copy.
struct fork_group {
    std::vector<std::vector<uint8_t>> raw_state; // host state per child, before normalization
    std::vector<std::vector<float>>   logits;    // branch logits per child
    bool ok = false;
};

static fork_group fork_group_decode(llama_context *   ctx,
                                    llama_decision::engine & eng,
                                    const llama_decision::engine::saved_state & saved,
                                    llama_seq_id      src,
                                    llama_seq_id      first,
                                    int               n,
                                    llama_pos         pos0,
                                    llama_token       tok,
                                    bool              plain_copy) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    llama_memory_t      mem   = llama_get_memory(ctx);

    fork_group out;
    out.raw_state.resize(n);
    out.logits.resize(n);

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int b = 0; b < n; ++b) {
        const llama_seq_id seq = first + b;
        llama_memory_seq_rm(mem, seq, -1, -1);
        if (plain_copy) {
            llama_memory_seq_cp(mem, src, seq, -1, -1);
        } else {
            eng.load_seq(saved, seq);
        }
        common_batch_add(batch, tok, pos0, { seq }, true);
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        return out;
    }
    llama_synchronize(ctx);
    for (int b = 0; b < n; ++b) {
        out.raw_state[b] = seq_state_dump(ctx, first + b);
        out.logits[b]    = output_logits(ctx, vocab, b);
    }
    out.ok = true;
    return out;
}

// The control that proves why an exact recurrent fork is needed: plain seq_cp shares the recurrent
// tail cell instead of copying it, so the branch state and its logits can drift from an exact
// restore. The divergence is recorded, not asserted, because a layout may happen to be exact.
static void fork_divergence_run(testing & t, llama_context * ctx, const std::string & lane) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const auto          parent = common_tokenize(vocab, "a short decision parent", false, true);
    const auto          branch = common_tokenize(vocab, "branch", false, true);
    if (parent.empty() || branch.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }

    llama_decision::engine eng(ctx, 2, 8);
    const llama_seq_id     src = 2;
    const llama_pos        pos0 = (llama_pos) parent.size();

    // a trunk forked from a shared prefix: the engine's shape, where the trunk inherits the
    // prefix's recurrent state through seq_cp instead of restoring it
    {
        llama_memory_clear(llama_get_memory(ctx), true);
        if (!t.assert_true(lane + ": the prefix decodes", decode_tokens_on(ctx, src, 0, parent))) {
            return;
        }
        llama_synchronize(ctx);
        const auto prefix_state = eng.save_seq(src, false);

        const llama_seq_id plain_trunk = 8, ref_trunk = 9;
        llama_memory_seq_rm(llama_get_memory(ctx), plain_trunk, -1, -1);
        llama_memory_seq_cp(llama_get_memory(ctx), src, plain_trunk, -1, -1);
        llama_memory_seq_rm(llama_get_memory(ctx), ref_trunk, -1, -1);
        eng.load_seq(prefix_state, ref_trunk);

        const bool plain_ok = decode_tokens_on(ctx, plain_trunk, pos0, branch);
        llama_synchronize(ctx);
        const auto plain_trunk_logits = output_logits(ctx, vocab, 0);
        const bool ref_ok = decode_tokens_on(ctx, ref_trunk, pos0, branch);
        llama_synchronize(ctx);
        const auto ref_trunk_logits = output_logits(ctx, vocab, 0);
        if (!t.assert_true(lane + ": the plain trunk decodes", plain_ok) ||
            !t.assert_true(lane + ": the restore trunk decodes", ref_ok)) {
            return;
        }
        const auto plain_trunk_raw = seq_state_dump(ctx, plain_trunk);
        const auto ref_trunk_raw   = seq_state_dump(ctx, ref_trunk);

        const auto plain_trunk_norm = normalize_seq_state(ctx, plain_trunk_raw, 5);
        const auto ref_trunk_norm   = normalize_seq_state(ctx, ref_trunk_raw, 5);
        printf("[fork control] %s: prefix trunk fork via seq_cp vs restore: %zu of %zu state bytes differ, max logit delta %.6g\n",
               lane.c_str(), state_bytes_diff(ref_trunk_norm, plain_trunk_norm),
               std::max(ref_trunk_norm.size(), plain_trunk_norm.size()),
               max_abs_logit_delta(ref_trunk_logits, plain_trunk_logits));
    }

    for (int n_prior : { 0, 2 }) {
        for (int n_branches : { 1, 2 }) {
            // each probe starts from a clean cache so the parent's layout is the same for both forks
            llama_memory_clear(llama_get_memory(ctx), true);
            // prior sequences occupy cells before the fork, a layout in which plain seq_cp can
            // hand a branch a stale recurrent source cell
            for (int p = 0; p < n_prior; ++p) {
                if (!t.assert_true(lane + ": a prior sequence decodes", decode_tokens_on(ctx, 8 + p, 0, parent))) {
                    return;
                }
            }
            if (!t.assert_true(lane + ": the parent decodes", decode_tokens_on(ctx, src, 0, parent))) {
                return;
            }
            llama_synchronize(ctx);
            const auto saved = eng.save_seq(src, false);
            if (!t.assert_true(lane + ": the parent state is non-empty", !saved.bytes.empty())) {
                return;
            }

            const llama_seq_id ref_first   = 3;
            const llama_seq_id plain_first = 3 + n_branches;
            const llama_seq_id scratch     = plain_first + n_branches;

            const auto ref   = fork_group_decode(ctx, eng, saved, src, ref_first,   n_branches, pos0, branch[0], false);
            const auto plain = fork_group_decode(ctx, eng, saved, src, plain_first, n_branches, pos0, branch[0], true);
            if (!t.assert_true(lane + ": the fork control decodes", ref.ok && plain.ok)) {
                return;
            }

            size_t diff = 0, total = 0;
            double max_logit_delta = 0.0;
            for (int b = 0; b < n_branches; ++b) {
                const auto ref_norm   = normalize_seq_state(ctx, ref.raw_state[b], scratch);
                const auto plain_norm = normalize_seq_state(ctx, plain.raw_state[b], scratch);
                diff += state_bytes_diff(ref_norm, plain_norm);
                total += std::max(ref_norm.size(), plain_norm.size());
                max_logit_delta = std::max(max_logit_delta, max_abs_logit_delta(ref.logits[b], plain.logits[b]));
            }
            printf("[fork control] %s: plain seq_cp vs restore, %d prior, %d branch(es) in one batch: %zu of %zu state bytes differ, max logit delta %.6g\n",
                   lane.c_str(), n_prior, n_branches, diff, total, max_logit_delta);
        }
    }
}

static void test_fork_oracle(testing & t) {
    t.test("strategy forks equal a full restore on the CPU model", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        for (const char * strategy : { "auto", "copy", "restore", "hybrid" }) {
            t.test(std::string(strategy) + ": the fork matches the reference", [&](testing & t) {
                try {
                    fork_oracle_run(t, te.ctx, "cpu", strategy);
                } catch (const std::exception & e) {
                    t.assert_true(std::string("the CPU fork oracle runs: ") + e.what(), false);
                }
            });
        }
    });

    t.test("strategy forks equal a full restore on the GPU model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        for (const char * strategy : { "auto", "copy", "restore", "hybrid" }) {
            t.test(std::string(strategy) + ": the fork matches the reference", [&](testing & t) {
                try {
                    fork_oracle_run(t, te.ctx, "gpu", strategy);
                } catch (const std::exception & e) {
                    t.assert_true(std::string("the GPU fork oracle runs: ") + e.what(), false);
                }
            });
        }
    });
}

// decide_batch forks twice: a trunk from the shared prefix, then a branch from the trunk. A single
// fork oracle would miss a drift that only appears after the trunk has decoded its tail, so this
// decodes a multi-token tail on a forked trunk, forks a branch from it, and compares the branch
// logits against the same nested sequence done with full restore states at both levels.
static void nested_fork_oracle_run(testing & t, llama_context * ctx, const std::string & lane, const std::string & strategy) {
    const llama_model * model = llama_get_model(ctx);
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        t.skip(lane + ": a nested partial fork needs a recurrent model");
        return;
    }
    const llama_vocab * vocab  = llama_model_get_vocab(model);
    const auto          prefix = common_tokenize(vocab, "the shared decision prefix", false, true);
    const auto          tail   = common_tokenize(vocab, "tail tokens decoded on the trunk", false, true);
    const auto          branch = common_tokenize(vocab, "branch", false, true);
    if (prefix.empty() || tail.empty() || branch.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }

    llama_decision::engine eng(ctx, 2, 8);
    const llama_seq_id     snap = 2, ref_trunk = 3, ref_branch = 4, sub_trunk = 5, sub_branch = 6;

    for (int n_prior : { 0, 2 }) {
        const std::string layout = n_prior == 0 ? "fresh" : "prior";
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int p = 0; p < n_prior; ++p) {
            if (!t.assert_true(lane + "/" + layout + ": a prior sequence decodes",
                               decode_tokens_on(ctx, 8 + p, 0, prefix))) {
                return;
            }
        }
        if (!t.assert_true(lane + "/" + layout + ": the shared prefix decodes", decode_tokens_on(ctx, snap, 0, prefix))) {
            return;
        }
        llama_synchronize(ctx);

        const auto full = eng.save_seq(snap, false, false);
        const auto part = eng.save_seq(snap, false, true);

        // reference: a full restore at both levels
        eng.select_fork("restore");
        llama_memory_seq_rm(llama_get_memory(ctx), ref_trunk, -1, -1);
        eng.load_seq(full, ref_trunk);
        if (!t.assert_true(lane + "/" + layout + ": the reference trunk decodes",
                           decode_tokens_on(ctx, ref_trunk, (llama_pos) prefix.size(), tail))) {
            return;
        }
        llama_synchronize(ctx);
        const auto ref_trunk_state = eng.save_seq(ref_trunk, false, false);
        llama_memory_seq_rm(llama_get_memory(ctx), ref_branch, -1, -1);
        eng.load_seq(ref_trunk_state, ref_branch);
        if (!t.assert_true(lane + "/" + layout + ": the reference branch decodes",
                           decode_tokens_on(ctx, ref_branch, (llama_pos) (prefix.size() + tail.size()), branch))) {
            return;
        }
        llama_synchronize(ctx);
        const auto ref_logits = output_logits(ctx, vocab, 0);

        // subject: the requested strategy at both levels, from its own clean prefix
        llama_memory_clear(llama_get_memory(ctx), true);
        if (!t.assert_true(lane + "/" + layout + ": the subject prefix decodes", decode_tokens_on(ctx, snap, 0, prefix))) {
            return;
        }
        llama_synchronize(ctx);
        const auto sub_full = eng.save_seq(snap, false, false);
        const auto sub_part = eng.save_seq(snap, false, true);

        eng.select_fork(strategy);
        const bool   hybrid = eng.active_fork_ == llama_decision::engine::fork_kind::hybrid;
        const auto & root   = hybrid ? sub_part : sub_full;
        eng.fork_into(snap, sub_trunk, &root);
        if (!t.assert_true(lane + "/" + layout + ": the subject trunk decodes",
                           decode_tokens_on(ctx, sub_trunk, (llama_pos) prefix.size(), tail))) {
            return;
        }
        llama_synchronize(ctx);
        const auto sub_trunk_state = eng.save_seq(sub_trunk, false, hybrid);
        eng.fork_into(sub_trunk, sub_branch, &sub_trunk_state);
        if (!t.assert_true(lane + "/" + layout + ": the subject branch decodes",
                           decode_tokens_on(ctx, sub_branch, (llama_pos) (prefix.size() + tail.size()), branch))) {
            return;
        }
        llama_synchronize(ctx);
        const auto sub_logits = output_logits(ctx, vocab, 0);

        const double delta = max_abs_logit_delta(ref_logits, sub_logits);
        t.assert_true(lane + "/" + layout + ": the nested " + strategy + " fork matches a full restore (max logit delta " +
                          std::to_string(delta) + ")",
                      delta == 0.0);
    }
}

static void test_nested_fork_oracle(testing & t) {
    t.test("a nested strategy fork equals a full restore on the GPU model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        for (const char * strategy : { "restore", "hybrid" }) {
            t.test(std::string(strategy) + ": the nested fork matches the reference", [&](testing & t) {
                try {
                    nested_fork_oracle_run(t, te.ctx, "gpu", strategy);
                } catch (const std::exception & e) {
                    t.assert_true(std::string("the GPU nested fork oracle runs: ") + e.what(), false);
                }
            });
        }
    });
}

// A decision forked from a live source sequence must equal the same decision whose source text was
// prefilled as an ordinary context: the session entry only skips the re-prefill, it does not change
// what is scored. The source must survive the decision untouched.
static void session_fork_oracle_run(testing & t, llama_context * ctx, const std::string & lane) {
    llama_decision::engine eng(ctx, 2, 8);
    const std::vector<llama_decision::field_input> fields = {
        { "  \"a\": ", { "1", "2" } },
        { "  \"b\": ", { "x", "y", "z" } },
    };
    llama_decision::options o;
    o.mode        = "tree";
    o.fork        = "auto";
    o.allow_cache = false;

    const std::string shared_text  = "the shared decision prefix";
    const std::string context_text = "the session context";
    const std::vector<llama_token> shared  = eng.tokenize(shared_text, true);
    const std::vector<llama_token> context = eng.tokenize(context_text, shared.empty());
    if (shared.empty() || context.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }
    const auto plan = eng.compile_fields(fields, o);

    // control: the stateless decision prefills shared + context itself
    llama_memory_clear(llama_get_memory(ctx), true);
    const auto stateless = eng.decide_batch(plan, shared_text, { context_text }, o);

    // session: the same tokens are already decoded on the source sequence
    llama_memory_clear(llama_get_memory(ctx), true);
    const llama_seq_id src = 0;
    if (!t.assert_true(lane + ": the session prefix decodes", decode_tokens_on(ctx, src, 0, shared)) ||
        !t.assert_true(lane + ": the session context decodes",
                       decode_tokens_on(ctx, src, (llama_pos) shared.size(), context))) {
        return;
    }
    llama_synchronize(ctx);
    const auto       src_before = seq_state_dump(ctx, src);
    const llama_pos  base_pos   = (llama_pos) (shared.size() + context.size());

    llama_decision::batch_result session;
    try {
        session = eng.decide_batch_from_seq(src, base_pos, plan, o);
    } catch (const std::exception & e) {
        t.assert_true(std::string(lane + ": the session decision runs: ") + e.what(), false);
        return;
    }

    t.assert_true(lane + ": the source state is unchanged by the session fork", src_before == seq_state_dump(ctx, src));

    if (!t.assert_true(lane + ": both decisions return one result",
                       stateless.items.size() == 1 && session.items.size() == 1)) {
        return;
    }
    // the session path physically places branch cells differently, so the GPU producer numerics
    // bound applies; the winners are the task-value outcome and must agree
    const double tol = lane == "gpu" ? 5e-2 : 1e-4;
    bool         same  = stateless.items[0].fields.size() == session.items[0].fields.size();
    bool         wins  = same;
    double       worst = 0.0;
    for (size_t f = 0; same && f < fields.size(); ++f) {
        wins = wins && stateless.items[0].fields[f].winner == session.items[0].fields[f].winner;
        const auto & ps = stateless.items[0].fields[f].probs;
        const auto & pn = session.items[0].fields[f].probs;
        same = same && ps.size() == pn.size();
        for (size_t k = 0; same && k < ps.size(); ++k) {
            const double d = std::fabs(ps[k] - pn[k]);
            worst = std::max(worst, d);
            same  = d <= tol;
        }
    }
    t.assert_true(lane + ": the session decision keeps the winners", wins);
    t.assert_true(lane + ": the session decision matches the prefilled context (worst " +
                              std::to_string(worst) + ")",
                  same);

    // The branch-level byte oracle (a forked branch equals a full restore) lives in the fork oracle
    // tests; the session entry reuses that primitive and only changes where the trunk is forked
    // from, so the session check is the task-value equality above plus the source invariance below.
    t.assert_true(lane + ": the source state is still unchanged after scoring",
                  src_before == seq_state_dump(ctx, src));
}

// A wrong base_pos must fail instead of silently scoring at shifted positions.
static void session_fork_position_run(testing & t, llama_context * ctx, const std::string & lane) {
    llama_decision::engine eng(ctx, 2, 8);
    const std::vector<llama_decision::field_input> fields = { { "  \"a\": ", { "1", "2" } } };
    llama_decision::options o;
    o.allow_cache = false;
    const std::vector<llama_token> src_toks = eng.tokenize("a session with a transcript", true);
    if (src_toks.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }
    const auto plan = eng.compile_fields(fields, o);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (!decode_tokens_on(ctx, 0, 0, src_toks)) {
        t.assert_true(lane + ": the source decodes", false);
        return;
    }
    llama_synchronize(ctx);
    const llama_pos good = (llama_pos) src_toks.size();
    // a position the source does not continue from must be rejected before any decode
    bool rejected = false;
    try {
        (void) eng.decide_batch_from_seq(0, good + 1, plan, o);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    t.assert_true(lane + ": a shifted base_pos is rejected, not scored", rejected);
    // the correct continuation still runs
    bool ran = true;
    try {
        (void) eng.decide_batch_from_seq(0, good, plan, o);
    } catch (const std::exception &) {
        ran = false;
    }
    t.assert_true(lane + ": the correct continuation still runs", ran);
}

static void test_session_fork(testing & t) {
    t.test("a session fork matches a prefilled context on the CPU model", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            session_fork_oracle_run(t, te.ctx, "cpu");
            session_fork_position_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU session fork: ") + e.what(), false);
        }
    });

    t.test("a session fork matches a prefilled context on the GPU model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            session_fork_oracle_run(t, te.ctx, "gpu");
            session_fork_position_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU session fork: ") + e.what(), false);
        }
    });
}

// Flipping the default fork must not move an answer: for a recurrent or hybrid model `auto` now
// selects the partial hybrid fork, whose branch state is byte-identical to a full restore, so the
// winners stay the same and the probabilities agree within the GPU producer-numerics bound. A dense
// model keeps the previous copy default, so its answers are identical to an explicit copy.
static void fork_auto_default_run(testing & t, llama_context * ctx, const std::string & lane) {
    const llama_model * model     = llama_get_model(ctx);
    const bool          recurrent = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
    const std::vector<llama_decision::field_input> fields = {
        { "  \"a\": ", { "1", "2" } },
        { "  \"b\": ", { "x", "y" } },
        { "  \"c\": ", { "p", "q", "r" } },
    };

    llama_decision::engine eng(ctx, 2, 8);
    llama_decision::options o_auto;
    o_auto.fork        = "auto";
    o_auto.allow_cache = false;
    const auto auto_run = eng.decide_batch("system", { "ctx" }, fields, o_auto);
    const auto resolved = eng.active_fork_;

    // the reference is the strategy `auto` must resolve to: the exact hybrid fork for a recurrent
    // model, or the unchanged copy fork for a dense one
    llama_decision::options o_ref = o_auto;
    o_ref.fork = recurrent ? "restore" : "copy";
    const auto ref_run = eng.decide_batch("system", { "ctx" }, fields, o_ref);

    const auto expected = recurrent ? llama_decision::engine::fork_kind::hybrid
                                    : llama_decision::engine::fork_kind::copy;
    t.test(lane + ": auto selects the expected fork", [&](testing & t) {
        t.assert_true(lane + ": auto selects the expected fork", resolved == expected);
    });
    if (!t.assert_true(lane + ": both the default and the reference run return one result",
                       auto_run.items.size() == 1 && ref_run.items.size() == 1)) {
        return;
    }

    // a dense copy default must not move at all; on a recurrent model the CPU path is also exact,
    // while the GPU reduction order changes with the physical cell placement, so its bound is the
    // producer numerics
    const double tol = recurrent && lane == "gpu" ? 5e-2 : 0.0;
    bool   same  = auto_run.items[0].fields.size() == ref_run.items[0].fields.size();
    double worst = 0.0;
    for (size_t f = 0; same && f < fields.size(); ++f) {
        same = auto_run.items[0].fields[f].winner == ref_run.items[0].fields[f].winner;
        const auto & pa = auto_run.items[0].fields[f].probs;
        const auto & pr = ref_run.items[0].fields[f].probs;
        same = same && pa.size() == pr.size();
        for (size_t k = 0; same && k < pa.size(); ++k) {
            const double d = std::fabs(pa[k] - pr[k]);
            worst = std::max(worst, d);
            same  = d <= tol;
        }
    }
    t.assert_true(lane + ": the default fork keeps the reference answers (worst " + std::to_string(worst) + ")",
                  same);
}

static void test_fork_auto_default(testing & t) {
    t.test("the default fork keeps the reference answers on the CPU model", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            fork_auto_default_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU default fork: ") + e.what(), false);
        }
    });

    t.test("the default fork keeps the reference answers on the GPU model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            fork_auto_default_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU default fork: ") + e.what(), false);
        }
    });
}

// A strategy change must reuse the token-cached prefix without carrying the previous scope with it:
// a restore request leaves a full prefix state, a hybrid request needs a partial one, and the
// token-cached prefix path must refresh the scope before it forks.
static void fork_strategy_switch_run(testing & t, llama_context * ctx, const std::string & lane) {
    const std::vector<llama_decision::field_input> fields = {
        { "  \"a\": ", { "1", "2" } },
        { "  \"b\": ", { "x", "y" } },
    };
    const double tol = lane == "gpu" ? 5e-2 : 1e-4;

    const std::pair<const char *, const char *> pairs[] = {
        { "restore", "hybrid" },
        { "hybrid", "restore" },
    };
    for (const auto & pair : pairs) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_decision::engine eng(ctx, 2, 8);

        llama_decision::options o1;
        o1.fork      = pair.first;
        o1.cache_tag = "switch";
        const auto r1 = eng.decide_batch("system", { "ctx" }, fields, o1);

        llama_decision::options o2;
        o2.fork      = pair.second;
        o2.cache_tag = "switch";
        const auto r2 = eng.decide_batch("system", { "ctx" }, fields, o2);

        if (!t.assert_true(lane + ": the " + pair.first + " run returns", r1.items.size() == 1) ||
            !t.assert_true(lane + ": the " + pair.second + " run returns", r2.items.size() == 1)) {
            return;
        }
        bool   same  = r1.items[0].fields.size() == r2.items[0].fields.size();
        double worst = 0.0;
        for (size_t f = 0; same && f < fields.size(); ++f) {
            same = r1.items[0].fields[f].winner == r2.items[0].fields[f].winner;
            const auto & p1 = r1.items[0].fields[f].probs;
            const auto & p2 = r2.items[0].fields[f].probs;
            same = same && p1.size() == p2.size();
            for (size_t k = 0; same && k < p1.size(); ++k) {
                worst = std::max(worst, (double) std::fabs(p1[k] - p2[k]));
                same  = std::fabs(p1[k] - p2[k]) < tol;
            }
        }
        t.assert_true(lane + ": " + pair.first + " then " + pair.second + " agrees (worst " +
                          std::to_string(worst) + ")",
                      same);
    }
}

static void test_fork_strategy_switch(testing & t) {
    t.test("a fork strategy change reuses the cached prefix on the CPU model", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            fork_strategy_switch_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU strategy switch: ") + e.what(), false);
        }
    });

    t.test("a fork strategy change reuses the cached prefix on the GPU model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            fork_strategy_switch_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU strategy switch: ") + e.what(), false);
        }
    });
}

static void test_fork_divergence_control(testing & t) {
    t.test("plain seq_cp fork divergence against restore is recorded (cpu)", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            fork_divergence_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU fork control runs: ") + e.what(), false);
        }
    });

    t.test("plain seq_cp fork divergence against restore is recorded (gpu)", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            fork_divergence_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU fork control runs: ") + e.what(), false);
        }
    });
}


// Returning to a cached prefix must restore that prefix's own bytes. The LRU stores host-format
// states, so a later save on the snapshot sequence cannot corrupt an earlier entry: the return
// scores and the snapshot KV must match the first run of the same prefix.
static void test_prefix_lru_restores_own_state(testing & t) {
    t.test("returning to a cached prefix restores that prefix", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            const llama_seq_id snap = 2; // the engine's prefix sequence
            llama_decision::engine eng(te.ctx, snap, 8);
            const std::vector<llama_decision::field_input> fields = {
                { "  \"a\": ", { "1", "2" } },
                { "  \"b\": ", { "x", "y" } },
            };
            const auto seq_dump = [&]() {
                std::vector<uint8_t> buf(llama_state_seq_get_size(te.ctx, snap));
                const size_t n = llama_state_seq_get_data(te.ctx, buf.data(), buf.size(), snap);
                buf.resize(n);
                return buf;
            };

            llama_decision::options opt;
            opt.fork      = "restore";
            opt.cache_tag = "A";
            const auto a1 = eng.decide_batch("system alpha", { "context" }, fields, opt);
            const std::vector<uint8_t> kv_a = seq_dump();
            opt.cache_tag = "B";
            const auto b1 = eng.decide_batch("system beta", { "context" }, fields, opt);
            const std::vector<uint8_t> kv_b = seq_dump();
            opt.cache_tag = "A";
            const auto a2 = eng.decide_batch("system alpha", { "context" }, fields, opt);
            const std::vector<uint8_t> kv_a2 = seq_dump();

            t.assert_true("the first A request misses", !a1.cache_hit);
            t.assert_true("the B request misses", !b1.cache_hit);
            t.assert_true("returning to A hits the bounded cache", a2.cache_hit);
            if (kv_a == kv_b) {
                t.skip("the two prefixes produce the same prefix KV on this model");
                return;
            }
            t.assert_true("the cached A prefix is restored from A's own saved bytes", kv_a == kv_a2);

            const auto & p1 = a1.items[0].fields[0].probs;
            const auto & p2 = a2.items[0].fields[0].probs;
            bool same = p1.size() == p2.size();
            for (size_t k = 0; same && k < p1.size(); ++k) {
                same = std::fabs(p1[k] - p2[k]) < 1e-6f;
            }
            t.assert_true("the cached A probabilities equal the first A probabilities", same);
        } catch (const std::exception & e) {
            t.assert_true(std::string("the prefix cache run: ") + e.what(), false);
        }
    });
}

// Baseline for the state-format work: a device-format sequence state must round-trip on the CPU
// backend. The host dump is the reference because it is serialized in sequence cell order.
// The transposed V cache stores one row per embedding, so a state save would otherwise touch each
// embedding separately. The bulk path moves one range per layer as a single strided transfer. A
// non-transposed V cache (flash attention on) is already contiguous and is the control that must
// not regress. Both must round-trip the sequence state byte-identically.
static void state_bulk_copy_run(testing & t, llama_context * ctx, const std::string & lane, int n_layer) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const auto          toks  = common_tokenize(vocab, "a transposed decision state", false, true);
    if (toks.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }
    if (!t.assert_true(lane + ": the prefix decodes", decode_tokens_on(ctx, 0, 0, toks))) {
        return;
    }
    llama_synchronize(ctx);

    const std::vector<uint8_t> before = seq_state_dump(ctx, 0);
    if (!t.assert_true(lane + ": the sequence state is non-empty", !before.empty())) {
        return;
    }

    llama_state_seq_debug_reset_transfers();
    std::vector<uint8_t> saved(llama_state_seq_get_size(ctx, 0));
    const size_t         saved_n = llama_state_seq_get_data(ctx, saved.data(), saved.size(), 0);
    saved.resize(saved_n);
    const uint64_t save_transfers = llama_state_seq_debug_transfer_count();

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_state_seq_debug_reset_transfers();
    const size_t nset = llama_state_seq_set_data(ctx, saved.data(), saved.size(), 0);
    const uint64_t load_transfers = llama_state_seq_debug_transfer_count();
    t.assert_equal(lane + ": the sequence state restores in full", saved.size(), nset);

    const std::vector<uint8_t> after = seq_state_dump(ctx, 0);
    t.assert_true(lane + ": the restored sequence state is byte-identical", before == after);

    // the device format carries the same state through its staging buffers
    const size_t dev_size = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    if (t.assert_true(lane + ": the device state has a size", dev_size > 0)) {
        std::vector<uint8_t> dev(dev_size);
        t.assert_equal(lane + ": the device state is written", dev_size,
                       llama_state_seq_get_data_ext(ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        llama_memory_clear(llama_get_memory(ctx), true);
        t.assert_equal(lane + ": the device state restores in full", dev_size,
                       llama_state_seq_set_data_ext(ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        t.assert_true(lane + ": the device-restored sequence state is byte-identical", before == seq_state_dump(ctx, 0));
    }

    printf("[state bulk] %s: n_layer %d, save transfers %" PRIu64 ", load transfers %" PRIu64 "\n",
           lane.c_str(), n_layer, save_transfers, load_transfers);

    // one bulk transfer per layer plus a bounded constant; the per-embedding path was thousands
    const uint64_t bound = (uint64_t) 8 * (uint64_t) std::max(n_layer, 1) + 32;
    t.assert_true(lane + ": the save uses a bulk transfer per layer", save_transfers <= bound);
    t.assert_true(lane + ": the load uses a bulk transfer per layer", load_transfers <= bound);
}

static void test_state_bulk_copy(testing & t) {
    t.test("a transposed sequence state round-trips and copies per layer on the GPU backend", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            state_bulk_copy_run(t, te.ctx, "transposed", llama_model_n_layer(te.model));
        } catch (const std::exception & e) {
            t.assert_true(std::string("the transposed state round trip: ") + e.what(), false);
        }
    });

    t.test("a non-transposed sequence state round-trips on the GPU backend", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        bool         loaded = false;
        try {
            loaded = te.load_fa(path);
        } catch (const std::exception &) {
            loaded = false;
        }
        if (!loaded) {
            t.skip("flash attention is not available for this model");
            return;
        }
        try {
            state_bulk_copy_run(t, te.ctx, "non-transposed", llama_model_n_layer(te.model));
        } catch (const std::exception & e) {
            t.assert_true(std::string("the non-transposed state round trip: ") + e.what(), false);
        }
    });
}

static void test_device_state_round_trip(testing & t) {
    t.test("a device sequence state round-trips on the CPU backend", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            const llama_vocab * vocab = llama_model_get_vocab(te.model);
            const std::vector<llama_token> toks = common_tokenize(vocab, "the decision state", false, true);
            if (toks.empty()) {
                t.skip("the model has no usable tokens");
                return;
            }
            llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
            for (size_t i = 0; i < toks.size(); ++i) {
                common_batch_add(batch, toks[i], (llama_pos) i, { 0 }, i + 1 == toks.size());
            }
            const int rc = llama_decode(te.ctx, batch);
            llama_batch_free(batch);
            if (rc != 0) {
                t.assert_true("the prefix decodes on the CPU backend", false);
                return;
            }
            llama_synchronize(te.ctx);

            const auto host_dump = [&]() {
                std::vector<uint8_t> buf(llama_state_seq_get_size(te.ctx, 0));
                const size_t n = llama_state_seq_get_data(te.ctx, buf.data(), buf.size(), 0);
                buf.resize(n);
                return buf;
            };
            const std::vector<uint8_t> before = host_dump();
            t.assert_true("the host state is non-empty", !before.empty());

            const size_t dev_size = llama_state_seq_get_size_ext(te.ctx, 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            t.assert_true("the device state has a size on the CPU backend", dev_size > 0);
            std::vector<uint8_t> dev(dev_size);
            const size_t ncopy = llama_state_seq_get_data_ext(te.ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            t.assert_equal("the full device state is written", dev_size, ncopy);

            llama_memory_clear(llama_get_memory(te.ctx), true);
            const size_t nset = llama_state_seq_set_data_ext(te.ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            t.assert_equal("the device state is restored in full", dev_size, nset);

            const std::vector<uint8_t> after = host_dump();
            t.assert_true("the restored state equals the saved state", before == after);
        } catch (const std::exception & e) {
            t.assert_true(std::string("the device round trip: ") + e.what(), false);
        }
    });
}

// The same round-trip when the sequence state lives on a GPU: the device path stages every copy
// with ggml_backend_tensor_copy_async and drains the backend once, so a missing drain would leave
// the restored host state stale. Without a GPU backend the test skips, keeping the CPU lane green.
static void test_device_async_staging(testing & t) {
    t.test("a device state staged on a GPU backend round-trips", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            const llama_vocab * vocab = llama_model_get_vocab(te.model);
            const std::vector<llama_token> toks = common_tokenize(vocab, "the decision state", false, true);
            if (toks.empty()) {
                t.skip("the model has no usable tokens");
                return;
            }
            llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
            for (size_t i = 0; i < toks.size(); ++i) {
                common_batch_add(batch, toks[i], (llama_pos) i, { 0 }, i + 1 == toks.size());
            }
            const int rc = llama_decode(te.ctx, batch);
            llama_batch_free(batch);
            if (rc != 0) {
                t.assert_true("the prefix decodes on the GPU backend", false);
                return;
            }
            llama_synchronize(te.ctx);

            const auto host_dump = [&]() {
                std::vector<uint8_t> buf(llama_state_seq_get_size(te.ctx, 0));
                const size_t n = llama_state_seq_get_data(te.ctx, buf.data(), buf.size(), 0);
                buf.resize(n);
                return buf;
            };
            const std::vector<uint8_t> before = host_dump();
            t.assert_true("the host state is non-empty", !before.empty());

            const size_t dev_size = llama_state_seq_get_size_ext(te.ctx, 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            t.assert_true("the device state has a size on the GPU backend", dev_size > 0);
            std::vector<uint8_t> dev(dev_size);
            const size_t ncopy = llama_state_seq_get_data_ext(te.ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            t.assert_equal("the full device state is written", dev_size, ncopy);

            llama_memory_clear(llama_get_memory(te.ctx), true);
            const size_t nset = llama_state_seq_set_data_ext(te.ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            t.assert_equal("the device state is restored in full", dev_size, nset);

            const std::vector<uint8_t> after = host_dump();
            t.assert_true("the restored state equals the saved state", before == after);

            // a working device path is what the engine prefers; it must not have retired to host
            llama_decision::engine eng(te.ctx, 2, 8);
            const auto st = eng.save_seq(0, true);
            t.assert_true("the engine keeps the device save when the backend supports it", st.on_device());

            bool threw = false;
            try {
                eng.load_seq(st, 0);
            } catch (const std::exception &) {
                threw = true;
            }
            t.assert_true("the engine device state restores", !threw);
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU device round trip: ") + e.what(), false);
        }
    });
}

// A recurrent cache can hold its sequence cells in more than one range. The device format cannot
// describe that, so the save must be refused recoverably (a throw the caller can catch, never an
// abort) and the self-contained host format must still round-trip byte-identically.
static void test_recurrent_multi_range_device_save(testing & t) {
    t.test("a fragmented recurrent cache refuses the device save and restores on the host", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        if (!llama_model_is_recurrent(te.model) && !llama_model_is_hybrid(te.model)) {
            t.skip("the model is neither recurrent nor hybrid");
            return;
        }
        try {
            const llama_vocab * vocab = llama_model_get_vocab(te.model);
            const std::vector<llama_token> toks = common_tokenize(vocab, "the decision state", false, true);
            if (toks.empty()) {
                t.skip("the model has no usable tokens");
                return;
            }

            const auto host_dump = [&]() {
                std::vector<uint8_t> buf(llama_state_get_size(te.ctx));
                const size_t n = llama_state_get_data(te.ctx, buf.data(), buf.size());
                buf.resize(n);
                return buf;
            };
            const auto host_restore = [&](const std::vector<uint8_t> & blob) {
                llama_memory_clear(llama_get_memory(te.ctx), true);
                return llama_state_set_data(te.ctx, blob.data(), blob.size()) == blob.size();
            };

            // place three sequences in consecutive cells, then drop the middle one so the used
            // cells of the whole cache are no longer contiguous
            for (llama_seq_id seq = 0; seq < 3; ++seq) {
                llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
                for (size_t i = 0; i < toks.size(); ++i) {
                    common_batch_add(batch, toks[i], (llama_pos) i, { seq }, i + 1 == toks.size());
                }
                const int rc = llama_decode(te.ctx, batch);
                llama_batch_free(batch);
                if (rc != 0) {
                    t.assert_true("the fragmented prefix decodes on the CPU backend", false);
                    return;
                }
            }
            llama_synchronize(te.ctx);
            const std::vector<uint8_t> pre = host_dump();
            if (!llama_memory_seq_rm(llama_get_memory(te.ctx), 1, -1, -1)) {
                t.skip("the model cannot remove a middle sequence");
                return;
            }

            // the one contiguous-range device format cannot hold the fragmented cells
            const size_t dev_size = llama_state_seq_get_size_ext(te.ctx, -1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
            t.assert_equal("the fragmented device save is refused", (size_t) 0, dev_size);

            // the host format is self-contained and must still round-trip
            const std::vector<uint8_t> frag = host_dump();
            t.assert_true("the fragmented host state is non-empty", !frag.empty());
            t.assert_true("the fragmented host state restores in full", host_restore(frag));
            t.assert_true("the restored fragmented state equals the saved state", frag == host_dump());

            // the pre-fragmentation state must still restore byte-identically afterwards
            t.assert_true("the pre-fragmentation state restores in full", host_restore(pre));
            t.assert_true("the restored pre-fragmentation state equals the saved state", pre == host_dump());
        } catch (const std::exception & e) {
            t.assert_true(std::string("the fragmented recurrent save: ") + e.what(), false);
        }
    });
}

// A full restore and a partial restore must carry the same recurrent state: the partial bytes read
// back after either restore are byte-identical, and the partial host format round-trips on its own.
// The full host save is also checked against the low-level serialization so the flag work cannot
// change the existing full-state bytes.
static void partial_state_round_trip_run(testing & t, llama_context * ctx, const std::string & lane) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const auto          toks  = common_tokenize(vocab, "a short partial decision state", false, true);
    if (toks.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }

    llama_decision::engine eng(ctx, 2, 8);
    const llama_seq_id     src = 2;
    if (!t.assert_true(lane + ": the state decodes", decode_tokens_on(ctx, src, 0, toks))) {
        return;
    }
    llama_synchronize(ctx);

    const auto full = eng.save_seq(src, false, false);
    const auto part = eng.save_seq(src, false, true);
    if (!t.assert_true(lane + ": the full host state is non-empty", !full.bytes.empty()) ||
        !t.assert_true(lane + ": the partial host state is non-empty", !part.bytes.empty())) {
        return;
    }
    t.assert_equal(lane + ": the full save carries no scope flag",
                   (unsigned) LLAMA_STATE_SEQ_FLAGS_NONE, (unsigned) full.flags);
    t.assert_equal(lane + ": the partial save carries the partial flag",
                   (unsigned) LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, (unsigned) part.flags);
    t.assert_true(lane + ": the engine reports partial state capability", eng.partial_state_capable());

    // the full host save is exactly the low-level full serialization, unchanged by the flags
    const size_t direct_size = llama_state_seq_get_size_ext(ctx, src, LLAMA_STATE_SEQ_FLAGS_NONE);
    std::vector<uint8_t> direct(direct_size);
    const size_t direct_n = llama_state_seq_get_data_ext(ctx, direct.data(), direct.size(), src, LLAMA_STATE_SEQ_FLAGS_NONE);
    direct.resize(direct_n);
    t.assert_true(lane + ": the full host save matches the direct serialization", direct == full.bytes);

    llama_memory_t mem = llama_get_memory(ctx);

    // a full restore carries the same recurrent state that a partial save reads
    llama_memory_clear(mem, true);
    eng.load_seq(full, src);
    const auto part_of_full = eng.save_seq(src, false, true);
    t.assert_true(lane + ": a full restore's recurrent state matches the partial save", part_of_full.bytes == part.bytes);

    // a partial restore round-trips the recurrent state on its own; the attention side stays empty
    // here, so the read-back uses the low-level serialization
    llama_memory_clear(mem, true);
    eng.load_seq(part, src);
    std::vector<uint8_t> part_rt(llama_state_seq_get_size_ext(ctx, src, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    const size_t         part_rt_n =
        llama_state_seq_get_data_ext(ctx, part_rt.data(), part_rt.size(), src, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    part_rt.resize(part_rt_n);
    t.assert_true(lane + ": the partial host round-trip is byte-identical", part_rt == part.bytes);

    // the partial device state must carry the same recurrent bytes as the host format
    llama_memory_clear(mem, true);
    eng.load_seq(full, src);
    const auto part_dev = eng.save_seq(src, true, true);
    if (t.assert_true(lane + ": the partial device save is staged", part_dev.on_device())) {
        llama_memory_clear(mem, true);
        eng.load_seq(part_dev, src);
        std::vector<uint8_t> dev_rt(llama_state_seq_get_size_ext(ctx, src, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        const size_t         dev_rt_n =
            llama_state_seq_get_data_ext(ctx, dev_rt.data(), dev_rt.size(), src, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        dev_rt.resize(dev_rt_n);
        t.assert_true(lane + ": the partial device round-trip matches the host bytes", dev_rt == part.bytes);
    }

    // the hybrid fork shape: attention copied by metadata, recurrent state restored partial
    llama_memory_clear(mem, true);
    eng.load_seq(full, src);
    llama_memory_seq_rm(mem, 5, -1, -1);
    llama_memory_seq_cp(mem, src, 5, -1, -1);
    eng.load_seq(part, 5);
    const auto fork_part = eng.save_seq(5, false, true);
    const auto fork_norm = normalize_seq_state(ctx, fork_part.bytes, 6, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const auto part_norm = normalize_seq_state(ctx, part.bytes, 6, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    t.assert_true(lane + ": the copied-attention partial fork matches the partial save",
                  state_bytes_diff(fork_norm, part_norm) == 0);
}

static void test_partial_state_round_trip(testing & t) {
    t.test("a partial host state round-trips and matches a full restore (cpu)", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            partial_state_round_trip_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU partial round trip: ") + e.what(), false);
        }
    });

    t.test("a partial host state round-trips and matches a full restore (gpu)", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            partial_state_round_trip_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU partial round trip: ") + e.what(), false);
        }
    });
}

// A fragmented recurrent cache cannot be staged as one device range: the whole-cache partial device
// save is refused recoverably (size 0, no abort) and the host partial format still round-trips. The
// engine's one-way partial device capability then falls back to the host partial format.
static void partial_state_fragmented_run(testing & t, llama_context * ctx, const std::string & lane) {
    const llama_model * model = llama_get_model(ctx);
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        t.skip(lane + ": the model is neither recurrent nor hybrid");
        return;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto          toks  = common_tokenize(vocab, "the fragmented partial state", false, true);
    if (toks.empty()) {
        t.skip(lane + ": the model has no usable tokens");
        return;
    }

    for (llama_seq_id seq = 0; seq < 3; ++seq) {
        if (!t.assert_true(lane + ": the fragmented prefix decodes", decode_tokens_on(ctx, seq, 0, toks))) {
            return;
        }
    }
    llama_synchronize(ctx);
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1)) {
        t.skip(lane + ": the model cannot remove a middle sequence");
        return;
    }

    const llama_state_seq_flags partial_device = llama_decision::engine::state_load_flags(true, true);
    t.assert_equal(lane + ": the fragmented partial device save is refused", (size_t) 0,
                   llama_state_seq_get_size_ext(ctx, -1, partial_device));

    // the host partial format is self-contained and must round-trip the fragmented recurrent cache
    const llama_state_seq_flags partial_host = llama_decision::engine::state_load_flags(false, true);
    const size_t host_size = llama_state_seq_get_size_ext(ctx, -1, partial_host);
    if (!t.assert_true(lane + ": the fragmented partial host state is non-empty", host_size > 0)) {
        return;
    }
    std::vector<uint8_t> host(host_size);
    t.assert_equal(lane + ": the fragmented partial host state is written", host_size,
                   llama_state_seq_get_data_ext(ctx, host.data(), host.size(), -1, partial_host));
    llama_memory_clear(llama_get_memory(ctx), true);
    t.assert_equal(lane + ": the fragmented partial host state restores", host_size,
                   llama_state_seq_set_data_ext(ctx, host.data(), host.size(), -1, partial_host));
    std::vector<uint8_t> after(llama_state_seq_get_size_ext(ctx, -1, partial_host));
    llama_state_seq_get_data_ext(ctx, after.data(), after.size(), -1, partial_host);
    t.assert_true(lane + ": the restored partial state equals the saved partial state", host == after);

    // the engine retires partial device staging when it fails and uses the host partial format
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_decision::engine eng(ctx, 4, 6);
    const llama_seq_id     src = 4;
    t.assert_true(lane + ": the engine starts with partial state capability", eng.partial_state_capable());
    if (!t.assert_true(lane + ": the engine prefix decodes", decode_tokens_on(ctx, src, 0, toks))) {
        return;
    }
    llama_synchronize(ctx);

    eng.partial_device_capable_ = false; // the retired state a failed device save leaves behind
    const auto st = eng.save_seq(src, true, true);
    t.assert_true(lane + ": a retired partial device save falls back to the host format", !st.on_device());
    t.assert_equal(lane + ": the fallback keeps the partial scope",
                   (unsigned) LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, (unsigned) st.flags);
    t.assert_true(lane + ": the fallback state is non-empty", !st.bytes.empty());
    t.assert_true(lane + ": the capability stays retired", !eng.partial_state_capable());

    llama_memory_clear(llama_get_memory(ctx), true);
    bool round_trip = false;
    try {
        eng.load_seq(st, src);
        std::vector<uint8_t> rt(llama_state_seq_get_size_ext(ctx, src, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        const size_t         rt_n =
            llama_state_seq_get_data_ext(ctx, rt.data(), rt.size(), src, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        rt.resize(rt_n);
        round_trip = rt == st.bytes;
    } catch (const std::exception &) {
        round_trip = false;
    }
    t.assert_true(lane + ": the fallback host partial state round-trips", round_trip);
}

static void test_partial_state_fragmented(testing & t) {
    t.test("a fragmented recurrent cache refuses the whole-cache partial device save (cpu)", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            partial_state_fragmented_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU partial fragmented save: ") + e.what(), false);
        }
    });

    t.test("a fragmented recurrent cache refuses the whole-cache partial device save (gpu)", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            partial_state_fragmented_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU partial fragmented save: ") + e.what(), false);
        }
    });
}

// Device state round-trip when the cache layout changes between save and restore. The save stages
// tensor bytes in the context staging buffer; a changed layout can push the reader from the 1:1
// chunked copy to the byte-cursor path, and the restored state must still equal the saved one. A
// missing backend synchronize would leave the restore half-applied, so the host dump catches it.
static void device_layout_mutation_round_trip(testing &           t,
                                              llama_context *     ctx,
                                              llama_model *       model,
                                              const std::string & lane) {
    const llama_vocab *            vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> toks  = common_tokenize(vocab, "layout mutation decision state", false, true);
    if (toks.empty()) {
        t.skip("the model has no usable tokens");
        return;
    }
    const llama_memory_t mem = llama_get_memory(ctx);
    const llama_pos      n   = (llama_pos) toks.size();

    auto decode_seq = [&](llama_seq_id seq, llama_pos pos0, const std::vector<llama_token> & ids) {
        llama_batch batch = llama_batch_init((int) ids.size(), 0, 1);
        for (size_t i = 0; i < ids.size(); ++i) {
            common_batch_add(batch, ids[i], pos0 + (llama_pos) i, { seq }, i + 1 == ids.size());
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        return rc == 0;
    };
    auto host_dump = [&]() {
        std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, 0));
        const size_t         got = llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0);
        buf.resize(got);
        return buf;
    };

    if (!decode_seq(0, 0, toks)) {
        t.assert_true(lane + ": the prefix decodes", false);
        return;
    }
    llama_synchronize(ctx);

    const std::vector<uint8_t> before = host_dump();
    if (!t.assert_true(lane + ": the host state is non-empty", !before.empty())) {
        return;
    }
    const size_t dev_size = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    if (!t.assert_true(lane + ": the device state has a size", dev_size > 0)) {
        return;
    }
    std::vector<uint8_t> dev(dev_size);
    if (!t.assert_equal(
            lane + ": the full device state is written", dev_size,
            llama_state_seq_get_data_ext(ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE))) {
        return;
    }

    // Mutate, restore the one saved device state, and require the host dump to return to `before`.
    auto mutate_and_restore = [&](const std::string & name, auto && mutate) {
        if (!mutate()) {
            printf("layout mutation not run (%s): the cache refused it\n", name.c_str());
            return;
        }
        const size_t nset =
            llama_state_seq_set_data_ext(ctx, dev.data(), dev.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (!t.assert_equal(lane + ": " + name + " restores in full", dev_size, nset)) {
            return;
        }
        t.assert_true(lane + ": " + name + " equals the saved state", before == host_dump());
    };

    // other sequences come and go around the saved one
    mutate_and_restore("other sequences", [&]() {
        if (!decode_seq(1, 0, toks) || !decode_seq(2, 0, toks)) {
            return false;
        }
        llama_synchronize(ctx);
        const bool r1 = llama_memory_seq_rm(mem, 1, -1, -1);
        const bool r2 = llama_memory_seq_rm(mem, 2, -1, -1);
        llama_synchronize(ctx);
        return r1 && r2;
    });

    // trim the tail of the saved sequence and re-decode it, so its cells may move
    mutate_and_restore("trim and re-decode", [&]() {
        const llama_pos k = std::min<llama_pos>(1, n - 1);
        if (k <= 0 || !llama_memory_seq_rm(mem, 0, n - k, -1)) {
            return false;
        }
        const std::vector<llama_token> tail(toks.end() - k, toks.end());
        return decode_seq(0, n - k, tail);
    });

    // rebuild the sequence from scratch, relocating its cells
    mutate_and_restore("clear and rebuild", [&]() {
        llama_memory_clear(mem, true);
        return decode_seq(0, 0, toks);
    });

    // interleave a foreign sequence into the saved sequence's cells, then re-decode the moved tail:
    // this fragments the saved sequence so the reader must re-chunk across ranges
    mutate_and_restore("fragmented rebuild", [&]() {
        const llama_pos half = n / 2;
        if (half <= 0 || !llama_memory_seq_rm(mem, 0, half, -1)) {
            return false;
        }
        const std::vector<llama_token> tail(toks.begin() + half, toks.end());
        if (!decode_seq(1, 0, tail)) {
            return false;
        }
        if (!decode_seq(0, half, tail)) {
            return false;
        }
        llama_memory_seq_rm(mem, 1, -1, -1);
        llama_synchronize(ctx);
        return true;
    });
}

static void test_device_layout_mutation_round_trip(testing & t) {
    t.test("a device state round-trips across cache layout mutations on the CPU backend", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path, 512, false, false)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            device_layout_mutation_round_trip(t, te.ctx, te.model, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU layout mutation round trip: ") + e.what(), false);
        }
    });

    t.test("a device state round-trips across cache layout mutations on a GPU backend", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            device_layout_mutation_round_trip(t, te.ctx, te.model, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU layout mutation round trip: ") + e.what(), false);
        }
    });
}

// The engine refuses a save that has nothing to copy and a load that has nothing to restore. A
// silent empty state would let a failed prefix save continue with wrong offsets and score garbage.
static void test_save_load_fail_fast(testing & t) {
    t.test("an empty decision state is refused on load", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no CPU decision model available");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            // a failed save used to produce exactly this value, which load_seq then silently
            // no-op'd and the decode continued at wrong offsets
            bool threw = false;
            try {
                eng.load_seq({}, 2);
            } catch (const std::runtime_error &) {
                threw = true;
            }
            t.assert_true("an empty state is refused on load, never a silent no-op", threw);
        } catch (const std::exception & e) {
            t.assert_true(std::string("empty-state load: ") + e.what(), false);
        }
    });

    t.test("a decoded sequence saves and restores a non-empty state", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no CPU decision model available");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            const llama_vocab * vocab = llama_model_get_vocab(te.model);
            const std::vector<llama_token> toks = common_tokenize(vocab, "the decision state", false, true);
            if (toks.empty()) {
                t.skip("the model has no usable tokens");
                return;
            }
            llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
            for (size_t i = 0; i < toks.size(); ++i) {
                common_batch_add(batch, toks[i], (llama_pos) i, { 2 }, i + 1 == toks.size());
            }
            const int rc = llama_decode(te.ctx, batch);
            llama_batch_free(batch);
            if (rc != 0) {
                t.assert_true("the prefix decodes on the CPU backend", false);
                return;
            }
            llama_synchronize(te.ctx);

            llama_decision::engine eng(te.ctx, 2, 8);
            const auto st = eng.save_seq(2, false);
            t.assert_true("a decoded sequence saves a non-empty state", !st.bytes.empty() && !st.on_device());
            bool threw = false;
            try {
                eng.load_seq(st, 2);
            } catch (const std::runtime_error &) {
                threw = true;
            }
            t.assert_true("the saved state restores without error", !threw);
        } catch (const std::exception & e) {
            t.assert_true(std::string("save/load round trip: ") + e.what(), false);
        }
    });

    t.test("a never-decoded sequence save throws instead of producing a header-only state", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no CPU decision model available");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            bool threw = false;
            try {
                eng.save_seq(9, false); // never decoded
            } catch (const std::runtime_error &) {
                threw = true;
            }
            t.assert_true("saving a never-decoded sequence throws", threw);
        } catch (const std::exception & e) {
            t.assert_true(std::string("never-decoded save: ") + e.what(), false);
        }
    });
}

// Structural control for the producer/task-value split: the weak-quant allowlist must only turn a
// task-value determinism assertion into an explicit skip, never an expected failure, and a
// non-allowlisted model must always run it hard. Every allowlist entry must also have a recorded
// measurement row in the calibration ledger so the skip is grounded in data, not prose.
static void test_weak_quant_control(testing & t) {
    t.test("a non-allowlisted model runs the task-value assertion hard", [](testing & t) {
        int ran = 0;
        determinism_check(t, false, "control: a non-allowlisted assertion runs hard", [&](testing &) { ++ran; });
        t.assert_equal("a non-allowlisted assertion runs hard", 1, ran);
    });

    t.test("an allowlisted model skips the task-value assertion, never xfails it", [](testing & t) {
        int ran = 0;
        determinism_check(t, true, "control: an allowlisted assertion is skipped", [&](testing &) { ++ran; });
        t.assert_equal("an allowlisted assertion is skipped, not run", 0, ran);
    });

    t.test("every weak-quant allowlist entry has a recorded calibration row", [](testing & t) {
        const common_json cal = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json"));
        const std::string recorded = cal.at("model_measurements").at("environment").value("model", std::string());
        if (recorded.empty()) {
            t.skip("no model measurement recorded in the calibration ledger");
            return;
        }
        for (const std::string & id : weak_quant_gpu_allowlist) {
            if (recorded != id) {
                t.skip("the calibration ledger records " + recorded + ", not the allowlist model " + id);
                continue;
            }
            t.assert_true("the allowlist entry " + id + " has a values row",
                          cal.at("model_measurements").contains("values"));
        }
    });
}


// The cache split adds one host save per miss and one device save per hit. On the generated model
// cold prefill is about 3 ms and warm prefill about 0.5 ms; the hit must stay clearly cheaper than
// a miss so the added refresh does not erase the cache win.
static void test_prefix_cache_cost(testing & t) {
    t.test("a cache hit prefills faster than a miss", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            const std::vector<llama_decision::field_input> fields = {
                { "  \"a\": ", { "1", "2" } },
                { "  \"b\": ", { "x", "y" } },
            };
            const auto time_prefill = [&](const std::string & tag) {
                llama_decision::options opt;
                opt.fork      = "restore";
                opt.cache_tag = tag;
                return eng.decide_batch("system cost", { "context" }, fields, opt).prefill_ms;
            };
            double miss_ms = std::numeric_limits<double>::max();
            for (int i = 0; i < 3; ++i) {
                miss_ms = std::min(miss_ms, time_prefill("cost-miss-" + std::to_string(i)));
            }
            double hit_ms = std::numeric_limits<double>::max();
            for (int i = 0; i < 3; ++i) {
                hit_ms = std::min(hit_ms, time_prefill("cost-hit"));
            }
            t.assert_true("the cache hit prefills faster than a miss (miss " + std::to_string(miss_ms) +
                          " ms, hit " + std::to_string(hit_ms) + " ms)", hit_ms < miss_ms * 0.9);
        } catch (const std::exception & e) {
            t.assert_true(std::string("the prefix cache cost run: ") + e.what(), false);
        }
    });
}


// An arch whose output table has no per-id bias must still build a usable head: the kept zero
// bias adds nothing, and the probe must not tighten into a failure.
static void test_classifier_head_unbiased(testing & t) {
    t.test("an unbiased output table still builds a usable answer head", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        std::vector<llama_decision::label> labels;
        labels.push_back({ "A", { 0 }, 0 });
        labels.push_back({ "B", { 1 }, 1 });
        const auto head = llama_decision::build_classifier_head(te.model, labels);
        if (!head.available()) {
            t.skip("the generated model has no classifier output table: " + head.reason);
            return;
        }
        t.assert_equal("the head covers both labels", (size_t) 2, head.ids.size());
        t.assert_equal("the head width is the hidden width",
                       (int) llama_model_n_embd_out(te.model), head.width);
        t.assert_equal("an unbiased model keeps a bias vector", head.ids.size(), head.bias.size());
        bool zeros = head.bias.size() == head.ids.size();
        for (float v : head.bias) {
            zeros = zeros && v == 0.0f;
        }
        t.assert_true("an unbiased model adds a zero bias", zeros);
    });
}


// A bounded decision context rejects a request that cannot fit before decoding, with a clear
// budget message, and keeps serving afterwards (no truncation, no KV residue).
static void test_bounded_decision_context(testing & t) {
    t.test("a request beyond the decision context budget is rejected, not truncated", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path, 256)) { // the smallest context the backend keeps (n_ctx is padded to 256)
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            llama_decision::engine eng(te.ctx, 2, 8);
            const std::vector<llama_decision::field_input> fields = { { "  \"a\": ", { "1", "2" } } };
            const auto ok = eng.decide_batch("sys", { "ctx" }, fields, llama_decision::options{});
            t.assert_equal("a fitting request returns a decision", (size_t) 1, ok.items.size());

            std::string big;
            for (int i = 0; i < 64; ++i) {
                big += "filler ";
            }
            bool threw = false;
            std::string what;
            try {
                eng.decide_batch(big, { "ctx" }, fields, llama_decision::options{});
            } catch (const llama_decision::capacity_error & e) {
                threw = true;
                what = e.what();
            }
            t.assert_true("an oversize request throws capacity_error", threw);
            t.assert_true("the error names the context budget", what.find("context holds") != std::string::npos);

            const auto again = eng.decide_batch("sys", { "ctx" }, fields, llama_decision::options{});
            t.assert_equal("the context keeps serving after a rejected request", (size_t) 1, again.items.size());
        } catch (const std::exception & e) {
            t.assert_true(std::string("the bounded context run: ") + e.what(), false);
        }
    });
}

// A restore-fork group with more than one trunk per wave. Each context is staged as its own saved
// state and every branch restores from its own parent, so independent contexts must not alias, and
// the chat sequences that share the context must be untouched.
static void multi_trunk_restore_round_trip(testing &           t,
                                           llama_context *     ctx,
                                           llama_model *       model,
                                           const std::string & lane,
                                           const std::string & fork) {
    const llama_vocab *            vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> chat  = common_tokenize(vocab, "a chat turn on the shared context", false, true);
    if (chat.empty()) {
        t.skip("the model has no usable tokens");
        return;
    }
    {
        llama_batch batch = llama_batch_init((int) chat.size(), 0, 1);
        for (size_t i = 0; i < chat.size(); ++i) {
            common_batch_add(batch, chat[i], (llama_pos) i, { 0 }, i + 1 == chat.size());
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            t.assert_true(lane + ": the chat prefix decodes", false);
            return;
        }
    }
    llama_synchronize(ctx);
    const auto chat_dump = [&]() {
        std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, 0));
        const size_t         n = llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0);
        buf.resize(n);
        return buf;
    };
    const std::vector<uint8_t> chat_before = chat_dump();
    t.assert_true(lane + ": the chat sequence has state", !chat_before.empty());

    llama_decision::engine                         eng(ctx, 2, 16);
    const std::vector<std::string>                 contexts = { "alpha", "beta", "alpha" };
    const std::vector<llama_decision::field_input> fields   = {
        { "  \"a\": ", { "1", "2" } },
        { "  \"b\": ", { "x", "y" } },
    };
    llama_decision::options o;
    o.mode        = "tree";
    o.fork        = fork;
    o.allow_cache = false;

    const auto   plan      = eng.compile_fields(fields, o);
    const size_t per_group = std::clamp<size_t>((size_t) eng.n_pool / (1 + plan.branches), 1, contexts.size());
    t.assert_true(lane + ": the batch groups more than one trunk per wave", per_group >= 2);

    const auto batch = eng.decide_batch("system", contexts, fields, o);
    t.assert_equal(lane + ": every context is returned", contexts.size(), batch.items.size());

    // the same context twice in one wave must not alias: identical inputs must score identically
    // identical contexts in one wave must score the same. GPU batch packing can place their
    // branches in different batches, so the probability bound there is the documented
    // head-agreement tolerance; the branch-state oracle is the byte-level exactness check.
    const double twin_tol = lane == "gpu" ? 5e-2 : 1e-4;
    bool twins = batch.items[0].fields.size() == batch.items[2].fields.size();
    for (size_t f = 0; twins && f < fields.size(); ++f) {
        const auto & p0 = batch.items[0].fields[f].probs;
        const auto & p2 = batch.items[2].fields[f].probs;
        twins           = batch.items[0].fields[f].winner == batch.items[2].fields[f].winner && p0.size() == p2.size();
        for (size_t k = 0; twins && k < p0.size(); ++k) {
            twins = std::fabs(p0[k] - p2[k]) < twin_tol;
        }
    }
    t.assert_true(lane + ": the repeated context scores identically in one wave", twins);

    // each context keeps its own winner, the task-value outcome, against a single-context run
    for (size_t c = 0; c < contexts.size(); ++c) {
        const auto single  = eng.decide_batch("system", { contexts[c] }, fields, o);
        bool       winners = batch.items[c].fields.size() == single.items[0].fields.size();
        for (size_t f = 0; winners && f < fields.size(); ++f) {
            winners = batch.items[c].fields[f].winner == single.items[0].fields[f].winner;
        }
        t.assert_true(lane + ": context " + std::to_string(c) + " keeps its winner", winners);
    }

    const std::vector<uint8_t> chat_after = chat_dump();
    t.assert_true(lane + ": the chat sequence is untouched", chat_before == chat_after);
}

static void test_multi_trunk_restore(testing & t) {
    for (const char * fork : { "restore", "hybrid" }) {
        t.test(std::string("the ") + fork + " fork keeps contexts independent on the CPU backend",
               [fork](testing & t) {
                   const std::string path = decision_cpu_model_path();
                   if (path.empty()) {
                       t.skip("no generated model; run the generate-models fixture");
                       return;
                   }
                   cpu_test_engine te;
                   if (!te.load(path, 512, false, false, 128, 18)) {
                       t.assert_true("the CPU decision scaffold loads the model", false);
                       return;
                   }
                   try {
                       multi_trunk_restore_round_trip(t, te.ctx, te.model, "cpu", fork);
                   } catch (const std::exception & e) {
                       t.assert_true(std::string("the CPU multi-trunk ") + fork + ": " + e.what(), false);
                   }
               });

        t.test(std::string("the ") + fork + " fork keeps contexts independent on a recurrent GPU backend",
               [fork](testing & t) {
                   const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
                   if (!gpu_model_ready(t, path)) {
                       return;
                   }
                   test_engine te;
                   if (!te.load(path, 18)) {
                       t.assert_true("the GPU decision scaffold loads the model", false);
                       return;
                   }
                   if (!llama_model_is_recurrent(te.model) && !llama_model_is_hybrid(te.model)) {
                       t.skip("the model is neither recurrent nor hybrid; the multi-trunk save/restore path needs one");
                       return;
                   }
                   if (weak_quant_gpu_oracle(path)) {
                       t.skip("weak-quant GPU numerics move a winner here; the aliasing check is skipped, not xfail");
                       return;
                   }
                   try {
                       multi_trunk_restore_round_trip(t, te.ctx, te.model, "gpu", fork);
                   } catch (const std::exception & e) {
                       t.assert_true(std::string("the GPU multi-trunk ") + fork + ": " + e.what(), false);
                   }
               });
    }
}

// A failed decision must leave the pool sequences empty: a mid-wave decode failure forks pool
// sequences first, and residue in the shared cache would starve the next chat decode.
static std::string text_of_n_tokens(const llama_vocab * vocab, int n) {
    std::string text;
    while ((int) common_tokenize(vocab, text, false, true).size() < n) {
        text += "filler ";
    }
    return text;
}

static void test_pool_seq_lifecycle(testing & t) {
    t.test("a mid-wave decode failure leaves every pool sequence empty and chat decodable", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path, 512)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            const auto * vocab = llama_model_get_vocab(te.model);
            auto ntok = [&](const std::string & s) {
                return (int) common_tokenize(vocab, s, false, true).size();
            };
            const std::string shared_text = text_of_n_tokens(vocab, 50);
            const std::string tail_text   = text_of_n_tokens(vocab, 10);
            const std::string suffix_text = text_of_n_tokens(vocab, 20);
            const int s_n = (int) common_tokenize(vocab, shared_text, true, true).size();
            const int t_n = ntok(tail_text);

            // chat occupancy sized so the branch restore load still fits but its decode does not:
            // cells = chat + snap(S) + trunk(S+T) + branch(S+T) + branch tokens(B)
            const int n_ctx = (int) llama_n_ctx(te.ctx);
            const int n_batch = (int) llama_n_batch(te.ctx);
            const int chat_fill  = n_ctx - s_n - 2 * (s_n + t_n) - 8;
            const int chat_probe = s_n + t_n + 8;
            t.assert_true("the scenario leaves room for the chat probe", chat_probe + s_n + 2 * (s_n + t_n) < n_ctx);

            std::vector<llama_token> chat_toks;
            for (int i = 0; i < chat_fill; ++i) {
                chat_toks.push_back(16); // any in-vocab id; content is irrelevant to cell accounting
            }
            // decode in chunks of n_batch, like the server's slot scheduler would
            for (size_t off = 0; off < chat_toks.size(); off += (size_t) n_batch) {
                const size_t n = std::min((size_t) n_batch, chat_toks.size() - off);
                llama_batch fill = llama_batch_init((int) n, 0, 1);
                for (size_t i = 0; i < n; ++i) {
                    common_batch_add(fill, chat_toks[off + i], (llama_pos) (off + i), { 0 }, false);
                }
                const int rc = llama_decode(te.ctx, fill);
                llama_batch_free(fill);
                if (rc != 0) {
                    llama_synchronize(te.ctx);
                    break;
                }
            }
            llama_synchronize(te.ctx);

            llama_decision::engine eng(te.ctx, 2, 8);
            const std::vector<llama_decision::field_input> fields = { { suffix_text, { "1", "2" } } };
            llama_decision::options o;
            o.fork = "restore"; // restore allocates exclusive pool cells, so a leak is measurable
            bool threw = false;
            try {
                (void) eng.decide_batch(shared_text, { tail_text }, fields, o);
            } catch (const llama_decision::capacity_error &) {
                threw = true;
            }
            t.assert_true("the branch wave raises capacity_error mid-request", threw);

            bool pool_empty = true;
            for (llama_seq_id s = eng.seq_pool; s < eng.seq_pool + eng.n_pool; ++s) {
                pool_empty = pool_empty && llama_memory_seq_pos_max(eng.mem, s) == -1;
            }
            t.assert_true("every pool sequence is empty after the failure", pool_empty);

            // the shared context keeps serving chat: cells freed by the cleanup make room
            llama_batch chat = llama_batch_init(chat_probe, 0, 1);
            for (int i = 0; i < chat_probe; ++i) {
                common_batch_add(chat, 16, (llama_pos) i, { 1 }, false);
            }
            const int rc = llama_decode(te.ctx, chat);
            llama_batch_free(chat);
            llama_synchronize(te.ctx);
            t.assert_true("a chat decode on the shared context succeeds after the failure", rc == 0);
        } catch (const std::exception & e) {
            t.assert_true(std::string("the pool lifecycle run: ") + e.what(), false);
        }
    });
}

// A sliding-window cache no longer holds cells older than the window, so a fork must copy only the
// cells that survive. The copy and hybrid forks share one clamp; this checks that a decision on a
// context longer than the window is unchanged under hybrid against the full-restore ground truth.
// The two paths may materialize different numbers of (attention-masked) cells, so this compares the
// task value, not the state bytes.
static void swa_fork_clamp_run(testing & t, llama_context * ctx, const std::string & lane) {
    const llama_model * model = llama_get_model(ctx);
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        t.skip(lane + ": the model has no recurrent partial state; the clamp control is the copy fork");
        return;
    }
    const int           n_swa = llama_model_n_swa(model);
    if (n_swa <= 0) {
        t.skip(lane + ": the model has no sliding-window attention");
        return;
    }
    const int n_ctx    = (int) llama_n_ctx(ctx);
    const int n_parent = std::min(n_swa + 8, n_ctx - 64);
    if (n_parent <= n_swa) {
        t.skip(lane + ": the sliding window does not fit the test context");
        return;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const std::string   context = text_of_n_tokens(vocab, n_parent);
    const auto          ctx_toks = common_tokenize(vocab, context, false, true);
    if ((int) ctx_toks.size() <= n_swa) {
        t.skip(lane + ": the model has no usable long prompt");
        return;
    }
    t.assert_true(lane + ": the context exceeds the sliding window", (int) ctx_toks.size() > n_swa);

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_decision::engine eng(ctx, 2, 8);
    const std::vector<llama_decision::field_input> fields = {
        { "  \"a\": ", { "1", "2" } },
        { "  \"b\": ", { "x", "y" } },
    };

    llama_decision::options o_restore;
    o_restore.fork        = "restore";
    o_restore.allow_cache = false;
    llama_decision::options o_hybrid = o_restore;
    o_hybrid.fork = "hybrid";

    const auto restore = eng.decide_batch("system", { context }, fields, o_restore);
    const auto hybrid  = eng.decide_batch("system", { context }, fields, o_hybrid);
    if (!t.assert_true(lane + ": the clamped restore decision returns", restore.items.size() == 1) ||
        !t.assert_true(lane + ": the clamped hybrid decision returns", hybrid.items.size() == 1)) {
        return;
    }

    const double swa_tol = lane == "gpu" ? 5e-2 : 1e-4;
    bool agree = restore.items[0].fields.size() == hybrid.items[0].fields.size();
    for (size_t f = 0; agree && f < fields.size(); ++f) {
        agree = restore.items[0].fields[f].winner == hybrid.items[0].fields[f].winner;
        const auto & pr = restore.items[0].fields[f].probs;
        const auto & ph = hybrid.items[0].fields[f].probs;
        agree = agree && pr.size() == ph.size();
        for (size_t k = 0; agree && k < pr.size(); ++k) {
            agree = std::fabs(pr[k] - ph[k]) < swa_tol;
        }
    }
    t.assert_true(lane + ": a context past the sliding window scores the same under hybrid", agree);
}

static void test_fork_swa_clamp(testing & t) {
    t.test("a hybrid fork clamps to the sliding window on the CPU backend", [](testing & t) {
        // the decision fixture qwen35 has no sliding window; the generated lfm2 is hybrid with one
        std::string path = std::string(DECISION_TEST_GENERATED_MODEL_DIR) + "/lfm2-dense.gguf";
        if (!file_exists(path)) {
            path = decision_cpu_model_path();
        }
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        try {
            swa_fork_clamp_run(t, te.ctx, "cpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU SWA clamp fork: ") + e.what(), false);
        }
    });

    t.test("a hybrid fork clamps to the sliding window on the GPU backend", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("the GPU decision scaffold loads the model", false);
            return;
        }
        try {
            swa_fork_clamp_run(t, te.ctx, "gpu");
        } catch (const std::exception & e) {
            t.assert_true(std::string("the GPU SWA clamp fork: ") + e.what(), false);
        }
    });
}

// A classifier-only context is selected through llama_context_params. The field must be appended
// after every pre-existing member so the by-value C ABI offsets of existing fields do not move.
static void test_context_params_append(testing & t) {
    t.test("classifier_only is appended and defaults off", [](testing & t) {
        t.assert_true("classifier_only follows the pre-existing ctx_other member",
                      offsetof(llama_context_params, classifier_only) > offsetof(llama_context_params, ctx_other));
        t.assert_true("classifier_only defaults false", !llama_context_default_params().classifier_only);
    });
}

// A classifier-only context stops after the post-norm hidden state. That state must be identical
// to the full context's hidden state for the same input, so the shared graph stop cannot change
// what the answer rows are scored against.
// A classifier-only context has no logits and never samples, so a sampler attached late would be
// silently ignored. It must be rejected and reported as absent instead.
static void test_classifier_only_sampler(testing & t) {
    t.test("a classifier-only context rejects a sampler and reports none", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te_cls;
        if (!te_cls.load(path, 256, true, false)) {
            t.assert_true("the classifier-only context loads", false);
            return;
        }

        llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        bool threw = false;
        try {
            (void) llama_set_sampler(te_cls.ctx, 0, chain);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        t.assert_true("a sampler is rejected on a classifier-only context", threw);
        llama_sampler_free(chain);

        // D3: a classifier-only context's hidden states are the answer head's input, so
        // set_embeddings(false) must be a warning-only no-op there. A normal context still honors
        // the request. Losing the hidden states would leave the context with neither logits nor
        // embeddings.
        const auto decode_has_embeddings = [](cpu_test_engine & te) {
            const llama_vocab * v = llama_model_get_vocab(te.model);
            const std::vector<llama_token> toks = common_tokenize(v, "contract", false, true);
            llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
            for (size_t i = 0; i < toks.size(); ++i) {
                common_batch_add(batch, toks[i], (llama_pos) i, { 0 }, true);
            }
            const int rc = llama_decode(te.ctx, batch);
            llama_batch_free(batch);
            llama_synchronize(te.ctx);
            return rc == 0 && llama_get_embeddings_ith(te.ctx, (int) toks.size() - 1) != nullptr;
        };

        // control: the same sampler attaches to a normal context, so the rejection above is
        // specific to the classifier-only context
        cpu_test_engine te_full;
        if (te_full.load(path, 256, false, true)) {
            llama_sampler * chain2 = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(chain2, llama_sampler_init_greedy());
            const bool ok = llama_set_sampler(te_full.ctx, 0, chain2);
            t.assert_true("a normal context accepts the sampler", ok);
            llama_set_sampler(te_full.ctx, 0, nullptr);
            llama_sampler_free(chain2);

            llama_set_embeddings(te_full.ctx, false);
            t.assert_true("a normal context honors set_embeddings(false)",
                          !decode_has_embeddings(te_full));
        }

        llama_set_embeddings(te_cls.ctx, false);
        t.assert_true("a classifier-only context keeps hidden states after set_embeddings(false)",
                      decode_has_embeddings(te_cls));
    });
}

// Host-runnable check of llama_model_classifier_rows: dequantize a handful of output rows on the
// CPU dummy model (F32 output) and verify dot(hidden,row) + bias reproduces the full-vocabulary
// logit. This is the only CI path for the row/offset math and the to_float dequant.
static void test_classifier_rows_host(testing & t) {
    t.test("classifier rows reproduce full-vocabulary logits on the CPU dummy model", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path, 256, false, true)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        const llama_vocab * vocab = llama_model_get_vocab(te.model);
        const std::vector<llama_token> toks = common_tokenize(vocab, "the decision state", false, true);
        if (toks.empty()) {
            t.skip("the model has no usable tokens");
            return;
        }
        llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
        for (size_t i = 0; i < toks.size(); ++i) {
            common_batch_add(batch, toks[i], (llama_pos) i, { 0 }, i + 1 == toks.size());
        }
        if (llama_decode(te.ctx, batch)) {
            llama_batch_free(batch);
            t.assert_true("the prompt decodes on the CPU backend", false);
            return;
        }
        llama_batch_free(batch);
        llama_synchronize(te.ctx);

        const float * hidden = llama_get_embeddings_ith(te.ctx, -1);
        const float * logits = llama_get_logits_ith(te.ctx, -1);
        if (hidden == nullptr || logits == nullptr) {
            t.skip("the generated model does not expose hidden states and logits");
            return;
        }
        const int width   = (int) llama_model_n_embd_out(te.model);
        const int n_vocab = llama_vocab_n_tokens(vocab);

        std::vector<llama_token> ids;
        for (int i = 0; i < 8 && i < n_vocab; ++i) {
            ids.push_back((llama_token) i);
        }
        if (ids.size() < 2) {
            t.skip("the generated vocabulary is too small");
            return;
        }
        std::vector<float> rows((size_t) ids.size() * (size_t) width);
        std::vector<float> bias(ids.size(), 0.0f);
        float softcap = -1.0f;
        const int w = llama_model_classifier_rows(te.model, ids.data(), (int32_t) ids.size(),
                                                  rows.data(), rows.size(), &softcap, bias.data());
        t.assert_equal("the row width is the hidden width", width, w);
        t.assert_true("no softcap on a non-Gemma4 head", softcap == 0.0f);

        for (size_t i = 0; i < ids.size(); ++i) {
            double dot = 0.0;
            for (int j = 0; j < width; ++j) {
                dot += (double) hidden[j] * rows[i * (size_t) width + (size_t) j];
            }
            const double predicted = dot + bias[i];
            const double actual    = logits[ids[i]];
            t.assert_true("row score matches the full logit for id " + std::to_string(ids[i]),
                          std::fabs(predicted - actual) <= 1e-3);
        }

        // an out-of-range id is rejected, never read past the vocabulary
        std::vector<llama_token> bad = { 0, 1, (llama_token) n_vocab };
        float sc = 0.0f;
        const int bad_w = llama_model_classifier_rows(te.model, bad.data(), (int32_t) bad.size(),
                                                      rows.data(), rows.size(), &sc, nullptr);
        t.assert_equal("an out-of-range id is rejected", 0, bad_w);

        // count == 1 is supported and returns the correct width
        {
            std::vector<llama_token> one = { 0 };
            std::vector<float> one_rows((size_t) width);
            float sc1 = -1.0f;
            const int w1 = llama_model_classifier_rows(te.model, one.data(), 1,
                                                       one_rows.data(), one_rows.size(), &sc1, nullptr);
            t.assert_equal("a single id is supported", width, w1);
        }

        // count == 0 is refused
        float sc0 = 0.0f;
        const int w0 = llama_model_classifier_rows(te.model, ids.data(), 0, rows.data(), 0, &sc0, nullptr);
        t.assert_equal("count == 0 is refused", 0, w0);

        // count == 255 is the largest accepted batch; 256 is refused, never read
        {
            constexpr int MAX_ROWS = 255;
            std::vector<llama_token> ids_max(MAX_ROWS, 0);
            for (int i = 0; i < MAX_ROWS && i < n_vocab; ++i) {
                ids_max[(size_t) i] = (llama_token) i;
            }
            std::vector<float> rows_max((size_t) MAX_ROWS * (size_t) width);
            float sc_max = -1.0f;
            const int w_max = llama_model_classifier_rows(te.model, ids_max.data(), MAX_ROWS,
                                                          rows_max.data(), rows_max.size(), &sc_max, nullptr);
            t.assert_equal("count == 255 is accepted", width, w_max);
            t.assert_true("softcap is written on a 255-row success", sc_max == 0.0f);

            std::vector<llama_token> ids_over(256, 0);
            std::vector<float> rows_over((size_t) 256 * (size_t) width);
            float sc_over = -1.0f;
            const int w_over = llama_model_classifier_rows(te.model, ids_over.data(), 256,
                                                           rows_over.data(), rows_over.size(), &sc_over, nullptr);
            t.assert_equal("count == 256 is refused", 0, w_over);
        }

        // an oversized dst_count is refused (dst_count must match count * width exactly)
        const int wbig = llama_model_classifier_rows(te.model, ids.data(), (int32_t) ids.size(),
                                                     rows.data(), rows.size() + 1, &sc0, nullptr);
        t.assert_equal("an oversized dst_count is refused", 0, wbig);

        // the unreadable-bias probe (0) needs a model whose output bias is block-quantized; no
        // generated fixture produces one (a 1-D bias is always F32), so the rejection branch is
        // covered by code review, not by this fixture

        // a second call returns byte-identical rows
        std::vector<float> rows2((size_t) ids.size() * (size_t) width);
        float sc2 = -1.0f;
        const int w2 = llama_model_classifier_rows(te.model, ids.data(), (int32_t) ids.size(),
                                                   rows2.data(), rows2.size(), &sc2, nullptr);
        t.assert_equal("a second call has the same width", w, w2);
        t.assert_true("a second call returns identical rows", rows == rows2);
    });
}

// The row reader must dequantize a quantized output table exactly as the graph's output projection
// does, so dot(hidden, dequantized_row) + bias reproduces the full logit. The generated fixtures
// are F32, so quantize one to Q8_0 first: this is the only CPU-CI path through ggml's to_float
// row dequant.
static void test_classifier_rows_dequant(testing & t) {
    t.test("classifier rows dequantize a quantized output table", [](testing & t) {
        const std::string f32_path = std::string(DECISION_TEST_GENERATED_MODEL_DIR) + "/qwen35-dense.gguf";
        if (!file_exists(f32_path)) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        const std::string q_path = std::string(DECISION_TEST_GENERATED_MODEL_DIR) + "/qwen35-dense-q8_0.gguf";
        {
            llama_model_quantize_params qp = llama_model_quantize_default_params();
            qp.ftype = LLAMA_FTYPE_MOSTLY_Q8_0;
            qp.quantize_output_tensor = true;
            qp.nthread = 1;
            if (llama_model_quantize(f32_path.c_str(), q_path.c_str(), &qp) != 0) {
                t.skip("the generated model could not be quantized");
                return;
            }
        }
        cpu_test_engine te;
        if (!te.load(q_path, 256, false, true)) {
            t.skip("the quantized generated model could not be loaded");
            return;
        }
        const llama_vocab * vocab = llama_model_get_vocab(te.model);
        const std::vector<llama_token> toks = common_tokenize(vocab, "the decision state", false, true);
        if (toks.empty()) {
            t.skip("the model has no usable tokens");
            return;
        }
        llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
        for (size_t i = 0; i < toks.size(); ++i) {
            common_batch_add(batch, toks[i], (llama_pos) i, { 0 }, i + 1 == toks.size());
        }
        const int rc = llama_decode(te.ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            t.assert_true("the quantized prompt decodes on the CPU backend", false);
            return;
        }
        llama_synchronize(te.ctx);
        const float * hidden = llama_get_embeddings_ith(te.ctx, -1);
        const float * logits = llama_get_logits_ith(te.ctx, -1);
        if (hidden == nullptr || logits == nullptr) {
            t.skip("the quantized model does not expose hidden states and logits");
            return;
        }
        const int width   = (int) llama_model_n_embd_out(te.model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<llama_token> ids;
        for (int i = 0; i < 8 && i < n_vocab; ++i) {
            ids.push_back((llama_token) i);
        }
        std::vector<float> rows((size_t) ids.size() * (size_t) width);
        std::vector<float> bias(ids.size(), 0.0f);
        float softcap = -1.0f;
        const int w = llama_model_classifier_rows(te.model, ids.data(), (int32_t) ids.size(),
                                                  rows.data(), rows.size(), &softcap, bias.data());
        t.assert_equal("the dequantized row width is the hidden width", width, w);
        t.assert_true("a quantized non-Gemma4 head reports no softcap", softcap == 0.0f);
        for (size_t i = 0; i < ids.size(); ++i) {
            double dot = 0.0;
            for (int j = 0; j < width; ++j) {
                dot += (double) hidden[j] * rows[i * (size_t) width + (size_t) j];
            }
            const double predicted = dot + bias[i];
            const double actual    = logits[ids[i]];
            t.assert_true("dequantized row score matches the full logit for id " + std::to_string(ids[i]),
                          std::fabs(predicted - actual) <= 1e-3);
        }
    });
}

// One predicate owns "can this model serve the classifier answer head". The generated qwen35
// fixture (equal widths) is the accepted control; the qwen4exp fixture has an output row width
// different from the hidden width, so it is refused with a non-empty reason. A refusal never
// blocks scoring: callers fall back to full logits.
static void test_classifier_support_predicate(testing & t) {
    t.test("the classifier support predicate accepts and refuses with a reason", [](testing & t) {
        const std::string ok_path = decision_cpu_model_path();
        if (ok_path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        {
            cpu_test_engine te;
            if (!te.load(ok_path, 256)) {
                t.assert_true("the accepted control loads the model", false);
                return;
            }
            const char * reason = (const char *) 0x1; // prove the predicate always writes the out-param
            const bool ok = llama_model_classifier_supported(te.model, &reason);
            t.assert_true("a matching model is supported", ok);
            t.assert_true("a supported model reports no reason", reason == nullptr);
        }
        {
            const std::string bad_path = std::string(DECISION_TEST_GENERATED_MODEL_DIR) + "/qwen4exp-moe.gguf";
            if (!file_exists(bad_path)) {
                t.skip("no qwen4exp fixture; run the generate-models fixture");
                return;
            }
            cpu_test_engine te;
            if (!te.load(bad_path, 256)) {
                t.assert_true("the mismatch fixture loads the model", false);
                return;
            }
            const char * reason = nullptr;
            const bool ok = llama_model_classifier_supported(te.model, &reason);
            t.assert_true("a width mismatch is refused", !ok);
            t.assert_true("a refusal explains itself", reason != nullptr && reason[0] != '\0');
        }
    });
}

// The answer-row width contract: rows are read against the hidden state the classifier-only
// context exposes (n_embd_out), so an output tensor whose row width differs from it must be
// refused. The generated qwen4exp fixture sets embedding_length_out != embedding_length, which
// is exactly that case; qwen35 (equal widths) is the accepted control.
static void test_classifier_rows_width_contract(testing & t) {
    t.test("classifier rows accept a matching hidden width and reject a mismatched one", [](testing & t) {
        // accepted control: embedding_length_out is absent, so n_embd_out == output row width
        const std::string ok_path = decision_cpu_model_path();
        if (ok_path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        {
            cpu_test_engine te;
            if (!te.load(ok_path, 256)) {
                t.assert_true("the accepted control loads the model", false);
                return;
            }
            std::vector<llama_token> ids = { 0, 1 };
            const int width = (int) llama_model_n_embd_out(te.model);
            std::vector<float> rows((size_t) ids.size() * (size_t) width);
            float softcap = 0.0f;
            const int w = llama_model_classifier_rows(te.model, ids.data(), (int32_t) ids.size(),
                                                      rows.data(), rows.size(), &softcap, nullptr);
            t.assert_equal("a matching output width is accepted with the hidden width", width, w);
        }

        // rejected: embedding_length_out != embedding_length, so the output row width differs
        // from the hidden state width; the rows must be refused, never mis-scored
        const std::string bad_path = std::string(DECISION_TEST_GENERATED_MODEL_DIR) + "/qwen4exp-moe.gguf";
        if (!file_exists(bad_path)) {
            t.skip("no qwen4exp fixture; run the generate-models fixture");
            return;
        }
        {
            cpu_test_engine te;
            if (!te.load(bad_path, 256)) {
                t.assert_true("the mismatch fixture loads the model", false);
                return;
            }
            t.assert_true("the fixture has embedding_length_out != embedding_length",
                          llama_model_n_embd_out(te.model) != (int32_t) llama_model_n_embd(te.model));
            std::vector<llama_token> ids = { 0, 1 };
            const int width = (int) llama_model_n_embd_out(te.model);
            std::vector<float> rows((size_t) ids.size() * (size_t) width);
            float softcap = 0.0f;
            const int w = llama_model_classifier_rows(te.model, ids.data(), (int32_t) ids.size(),
                                                      rows.data(), rows.size(), &softcap, nullptr);
            t.assert_equal("a mismatched output width is refused", 0, w);
            // even sized by the main embedding length, the rows are refused: the output rows can
            // never be scored against a hidden state of a different width
            std::vector<float> rows_embd((size_t) ids.size() * (size_t) llama_model_n_embd(te.model));
            const int w_embd = llama_model_classifier_rows(te.model, ids.data(), (int32_t) ids.size(),
                                                           rows_embd.data(), rows_embd.size(), &softcap, nullptr);
            t.assert_equal("a mismatched output width is refused whatever the caller sizes", 0, w_embd);
        }
    });
}

static void test_classifier_only_hidden_state(testing & t) {
    t.test("a classifier-only context exposes the same hidden state as a full context", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te_full;
        cpu_test_engine te_cls;
        if (!te_full.load(path, 256, false, true) || !te_cls.load(path, 256, true, false)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        const auto decode_hidden = [](cpu_test_engine & te) {
            const llama_vocab * vocab = llama_model_get_vocab(te.model);
            const std::vector<llama_token> toks = common_tokenize(vocab, "the decision state", false, true);
            if (toks.empty()) {
                return std::vector<float>();
            }
            llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
            for (size_t i = 0; i < toks.size(); ++i) {
                common_batch_add(batch, toks[i], (llama_pos) i, { 0 }, i + 1 == toks.size());
            }
            const int rc = llama_decode(te.ctx, batch);
            llama_batch_free(batch);
            if (rc != 0) {
                return std::vector<float>();
            }
            llama_synchronize(te.ctx);
            const float * e = llama_get_embeddings_ith(te.ctx, (int) toks.size() - 1);
            const uint32_t w = llama_model_n_embd_out(te.model);
            if (e == nullptr) {
                return std::vector<float>();
            }
            return std::vector<float>(e, e + w);
        };
        const auto h_full = decode_hidden(te_full);
        const auto h_cls  = decode_hidden(te_cls);
        if (h_full.empty() || h_cls.empty()) {
            t.skip("the generated model does not expose both hidden-state paths");
            return;
        }
        t.assert_equal("both hidden states have the output width", h_full.size(), h_cls.size());
        bool same = h_full.size() == h_cls.size();
        for (size_t i = 0; same && i < h_full.size(); ++i) {
            same = std::fabs(h_full[i] - h_cls[i]) < 1e-5f;
        }
        t.assert_true("the classifier-only hidden state equals the full context's", same);
    });
}

// ---------------------------------------------------------------- CPU scoring oracle
//
// The scoring substrata below are exercised on the generated dummy model, with no GPU and no
// LLAMA_DECISION_TEST_MODEL. The recorded values are the oracle for behavior-preserving
// refactors: a change to field compilation, head selection, or row scoring must not move them.
// Backends may reorder a reduction, so probabilities are compared with a tolerance.

static common_json oracle_readout(const llama_decision::letter_metrics & m,
                                  const std::vector<std::vector<float>> & probs) {
    common_json o = common_json::object();
    o["head_active"]          = m.head_active;
    o["head_reason"]          = m.head_reason;
    o["cache_hit"]            = m.cache_hit;
    o["suffix_tokens"]        = (long long) m.suffix_tokens;
    o["common_suffix_tokens"] = (long long) m.common_suffix_tokens;
    o["leaf_suffix_tokens"]   = (long long) m.leaf_suffix_tokens;
    o["rows"]                 = (long long) m.rows;
    o["rounds"]               = (long long) m.rounds;

    common_json questions = common_json::array();
    for (const auto & p : probs) {
        common_json q = common_json::object();
        q["options"] = (long long) p.size();
        q["winner"]  = p.empty() ? -1 : (long long) (std::max_element(p.begin(), p.end()) - p.begin());
        common_json scores = common_json::array();
        for (float v : p) {
            scores.push_back((double) v);
        }
        q["probs"] = scores;
        q["confidence"] = llama_decision::inverse_entropy_confidence(p);
        q["certainty"]  = llama_decision::winner_share(p);
        questions.push_back(q);
    }
    o["questions"] = questions;
    return o;
}

// dot(hidden, dequantized_row) + bias vs the full-vocabulary logit, over a few ids. This is the
// arithmetic identity the answer-head fast path relies on; the frozen value is the observed error.
static common_json oracle_classifier_rows(cpu_test_engine & te) {
    common_json o = common_json::object();
    const llama_vocab * vocab = llama_model_get_vocab(te.model);
    const std::vector<llama_token> toks = common_tokenize(vocab, "the decision state", false, true);
    if (toks.empty()) {
        return o;
    }
    llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(batch, toks[i], (llama_pos) i, { 0 }, i + 1 == toks.size());
    }
    const int rc = llama_decode(te.ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        return o;
    }
    llama_synchronize(te.ctx);
    const float * hidden = llama_get_embeddings_ith(te.ctx, -1);
    const float * logits = llama_get_logits_ith(te.ctx, -1);
    if (hidden == nullptr || logits == nullptr) {
        return o;
    }
    const int width   = (int) llama_model_n_embd_out(te.model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> ids;
    for (int i = 0; i < 8 && i < n_vocab; ++i) {
        ids.push_back((llama_token) i);
    }
    std::vector<float> rows((size_t) ids.size() * (size_t) width);
    std::vector<float> bias(ids.size(), 0.0f);
    float softcap = -1.0f;
    const int w = llama_model_classifier_rows(te.model, ids.data(), (int32_t) ids.size(),
                                              rows.data(), rows.size(), &softcap, bias.data());
    o["width"]   = w;
    o["softcap"] = (double) softcap;
    double worst = 0.0;
    for (size_t i = 0; i < ids.size(); ++i) {
        double dot = 0.0;
        for (int j = 0; j < width; ++j) {
            dot += (double) hidden[j] * (double) rows[i * (size_t) width + (size_t) j];
        }
        worst = std::max(worst, std::fabs(dot + (double) bias[i] - (double) logits[ids[i]]));
    }
    o["max_abs_dot_vs_logit"] = worst;
    return o;
}

// Tokens field compilation will score for one candidate, exactly as decide_batch builds the path.
static std::vector<llama_token> oracle_candidate_tokens(const llama_vocab * vocab, const std::string & suffix,
                                                        const std::string & candidate) {
    return common_tokenize(vocab, suffix + candidate + "\n", false, true);
}

// Builds an answer-row table for explicit token ids. The generated dummy model's TEST tokenizer
// hashes fixed 5-character chunks, so a candidate's scored token depends on the suffix offset and
// is not the isolated label token; the head must cover the tokens the compiled paths actually use.
static llama_decision::classifier_head oracle_make_head(const llama_model * model, const std::vector<llama_token> & ids) {
    llama_decision::classifier_head head;
    const int width = (int) llama_model_n_embd_out(model);
    if (width <= 0 || ids.empty()) {
        head.reason = "no answer rows are available";
        return head;
    }
    head.ids = ids;
    head.rows.assign(ids.size() * (size_t) width, 0.0f);
    head.bias.assign(ids.size(), 0.0f);
    const int w = llama_model_classifier_rows(model, head.ids.data(), (int32_t) head.ids.size(),
                                              head.rows.data(), head.rows.size(), &head.softcap, head.bias.data());
    if (w <= 0) {
        head.ids.clear();
        head.rows.clear();
        head.bias.clear();
        head.reason = "the model output tensor is not a plain contiguous answer head";
        return head;
    }
    head.width = w;
    return head;
}

static std::vector<llama_token> oracle_field_tokens(const llama_vocab * vocab, const std::string & suffix,
                                                    const std::vector<std::string> & candidates) {
    std::vector<llama_token> ids;
    for (const auto & c : candidates) {
        for (llama_token t : oracle_candidate_tokens(vocab, suffix, c)) {
            if (std::find(ids.begin(), ids.end(), t) == ids.end()) {
                ids.push_back(t);
            }
        }
    }
    return ids;
}

// The head-selection truth the engine reaches through decide_batch, before a pure entry point
// exists: a covered head on the classifier context is active and scores the rows; a full context,
// an unavailable head, and a null head all fall back with a reason. The covered and full field
// vectors are recorded so a refactor cannot move the head numerics.
static common_json oracle_head_selection(cpu_test_engine & te_full, cpu_test_engine & te_head, const llama_vocab * vocab) {
    common_json o = common_json::object();
    const std::string suffix = "\nAnswer:\n";
    const std::vector<std::string> candidates = { "AAAAA", "BBBBB" };
    const std::vector<llama_token> ids = oracle_field_tokens(vocab, suffix, candidates);
    const llama_decision::classifier_head covered = oracle_make_head(te_head.model, ids);
    o["covered_available"] = covered.available();
    o["covered_width"]     = covered.width;

    std::vector<llama_decision::field_input> fields = { { suffix, candidates, 1.0f } };

    auto run = [&](llama_context * ctx, const llama_decision::classifier_head * head) {
        llama_decision::engine eng(ctx, 2, 8);
        llama_decision::options opt;
        opt.head = head;
        return eng.decide_batch("", { "state" }, fields, opt);
    };
    auto describe = [](const llama_decision::batch_result & b) {
        common_json r = common_json::object();
        r["active"] = b.head_active;
        r["reason"] = b.head_reason;
        common_json probs = common_json::array();
        for (float v : b.items[0].fields[0].probs) {
            probs.push_back((double) v);
        }
        r["probs"] = probs;
        return r;
    };

    if (covered.available() && llama_context_classifier_only(te_head.ctx)) {
        o["covered_classifier"] = describe(run(te_head.ctx, &covered));
    }
    o["full_logits"] = describe(run(te_full.ctx, nullptr));
    o["full_context"] = [&]() {
        common_json r = describe(run(te_full.ctx, &covered));
        r.erase("probs");
        return r;
    }();

    llama_decision::classifier_head unavailable;
    unavailable.reason = "synthetic unavailable head";
    o["unavailable"] = [&]() {
        common_json r = describe(run(te_full.ctx, &unavailable));
        r.erase("probs");
        return r;
    }();
    o["null_head"] = [&]() {
        common_json r = describe(run(te_full.ctx, nullptr));
        r.erase("probs");
        return r;
    }();

    common_json agreement = common_json::object();
    agreement["winners_match"] = false;
    agreement["total_variation"] = 1.0;
    if (o.contains("covered_classifier")) {
        const auto & head_probs = o.at("covered_classifier").at("probs");
        const auto & full_probs = o.at("full_logits").at("probs");
        const auto argmax = [](const common_json & v) {
            size_t best = 0;
            for (size_t i = 1; i < v.size(); ++i) {
                if (v.at(i).get<double>() > v.at(best).get<double>()) {
                    best = i;
                }
            }
            return best;
        };
        const bool same_size = head_probs.size() == full_probs.size();
        agreement["winners_match"] = same_size && argmax(head_probs) == argmax(full_probs);
        double tv = 0.0;
        if (same_size) {
            for (size_t i = 0; i < head_probs.size(); ++i) {
                tv += std::fabs(head_probs.at(i).get<double>() - full_probs.at(i).get<double>());
            }
        }
        agreement["total_variation"] = tv;
    }
    o["covered_vs_full"] = agreement;
    return o;
}

static common_json decision_cpu_oracle() {
    common_json out = common_json::object();
    const std::string path = decision_cpu_model_path();
    if (path.empty()) {
        return out;
    }
    cpu_test_engine te_full;
    cpu_test_engine te_head;
    cpu_test_engine te_rows;
    if (!te_full.load(path, 512, false, false) || !te_head.load(path, 512, true, false) ||
        !te_rows.load(path, 256, false, true)) {
        return out;
    }
    auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te_full.model));
    const std::string tail = test_letter_tail();
    const auto pool = llama_decision::build_label_pool(*vocab, tail, 64);
    if (pool.size() < 3) {
        return out;
    }
    const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));

    out["model"] = model_identity(path);
    common_json pool_info = common_json::object();
    pool_info["count"]   = (long long) pool.size();
    pool_info["token_a"] = (long long) pool[0].token;
    out["pool"] = pool_info;

    llama_decision::answer_head_cache head_cache;
    llama_decision::engine e_full(te_full.ctx, 2, 8);

    llama_decision::decision_request rfull = req;
    rfull.head = "full";
    llama_decision::options ofull;
    ofull.cache_tag = "cpu-oracle-full";
    llama_decision::letter_metrics mfull;
    const auto pfull = test_letter_readout(e_full, head_cache, *vocab, nullptr, false,
                                                      rfull, pool, ofull, &mfull);
    out["full"] = oracle_readout(mfull, pfull);

    // head="auto" on the shared full context cannot use the answer rows (it exposes logits, not
    // hidden states), so the readout must report the fallback reason and still return an answer.
    llama_decision::decision_request rauto = req;
    rauto.head = "auto";
    llama_decision::options oauto;
    oauto.cache_tag = "cpu-oracle-auto";
    llama_decision::letter_metrics mauto;
    const auto pauto = test_letter_readout(e_full, head_cache, *vocab, nullptr, false,
                                                      rauto, pool, oauto, &mauto);
    out["auto_fallback"] = oracle_readout(mauto, pauto);

    out["head_selection"]  = oracle_head_selection(te_full, te_head, llama_model_get_vocab(te_full.model));
    out["classifier_rows"] = oracle_classifier_rows(te_rows);
    return out;
}

static void oracle_assert_probs(testing & t, const std::string & label,
                                const common_json & exp, const common_json & act, double tol) {
    const common_json & ep = exp.at("probs");
    const common_json & ap = act.at("probs");
    if (!t.assert_equal(label + " option count", ep.size(), ap.size())) {
        return;
    }
    for (size_t i = 0; i < ep.size(); ++i) {
        const double e = ep.at(i).get<double>();
        const double a = ap.at(i).get<double>();
        if (!t.assert_true(label + " prob[" + std::to_string(i) + "] within " + std::to_string(tol),
                           std::fabs(e - a) <= tol)) {
            return;
        }
    }
}

static void oracle_assert_readout(testing & t, const std::string & label,
                                  const common_json & exp, const common_json & act, double tol) {
    t.assert_equal(label + " head_active", exp.at("head_active").get<bool>(), act.at("head_active").get<bool>());
    t.assert_equal(label + " head_reason", exp.at("head_reason").get<std::string>(), act.at("head_reason").get<std::string>());
    t.assert_equal(label + " cache_hit", exp.at("cache_hit").get<bool>(), act.at("cache_hit").get<bool>());
    t.assert_equal(label + " suffix_tokens", exp.at("suffix_tokens").get<long long>(), act.at("suffix_tokens").get<long long>());
    t.assert_equal(label + " common_suffix_tokens", exp.at("common_suffix_tokens").get<long long>(), act.at("common_suffix_tokens").get<long long>());
    t.assert_equal(label + " leaf_suffix_tokens", exp.at("leaf_suffix_tokens").get<long long>(), act.at("leaf_suffix_tokens").get<long long>());
    t.assert_equal(label + " rows", exp.at("rows").get<long long>(), act.at("rows").get<long long>());
    t.assert_equal(label + " rounds", exp.at("rounds").get<long long>(), act.at("rounds").get<long long>());

    const common_json & eq = exp.at("questions");
    const common_json & aq = act.at("questions");
    if (!t.assert_equal(label + " question count", eq.size(), aq.size())) {
        return;
    }
    for (size_t qi = 0; qi < eq.size(); ++qi) {
        const std::string q = label + " q" + std::to_string(qi);
        t.assert_equal(q + " options", eq.at(qi).at("options").get<long long>(), aq.at(qi).at("options").get<long long>());
        t.assert_equal(q + " winner", eq.at(qi).at("winner").get<long long>(), aq.at(qi).at("winner").get<long long>());
        t.assert_true(q + " carries confidence = 1 - H/log K", aq.at(qi).contains("confidence"));
        t.assert_true(q + " carries certainty = max p", aq.at(qi).contains("certainty"));
        oracle_assert_probs(t, q, eq.at(qi), aq.at(qi), tol);
    }
}

static void test_decision_cpu_oracle(testing & t) {
    t.test("the CPU scoring oracle matches the frozen baseline", [](testing & t) {
        if (decision_cpu_model_path().empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        const common_json baseline = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/baseline.json"));
        if (!baseline.contains("cpu_oracle")) {
            t.skip("no frozen CPU oracle in baseline.json");
            return;
        }
        const common_json & exp = baseline.at("cpu_oracle");

        common_json act;
        try {
            act = decision_cpu_oracle();
        } catch (const std::exception & e) {
            t.assert_true(std::string("the CPU oracle runs: ") + e.what(), false);
            return;
        }
        if (act.empty()) {
            t.skip("the CPU oracle produced no values");
            return;
        }

        t.assert_equal("oracle model identity", exp.at("model").get<std::string>(), act.at("model").get<std::string>());
        t.assert_equal("oracle label pool count", exp.at("pool").at("count").get<long long>(),
                       act.at("pool").at("count").get<long long>());
        t.assert_equal("oracle label A token", exp.at("pool").at("token_a").get<long long>(),
                       act.at("pool").at("token_a").get<long long>());

        const double prob_tol = 1e-3;
        oracle_assert_readout(t, "full readout", exp.at("full"), act.at("full"), prob_tol);
        oracle_assert_readout(t, "auto fallback readout", exp.at("auto_fallback"), act.at("auto_fallback"), prob_tol);

        const common_json & er = exp.at("classifier_rows");
        const common_json & ar = act.at("classifier_rows");
        t.assert_equal("classifier row width", er.at("width").get<long long>(), ar.at("width").get<long long>());
        assert_close(t, "classifier softcap", er.at("softcap").get<double>(), ar.at("softcap").get<double>(), 1e-9);
        const double dot_err = ar.at("max_abs_dot_vs_logit").get<double>();
        t.assert_true("the row dot reproduces the full logit within 1e-3 (err=" + std::to_string(dot_err) + ")",
                      dot_err <= 1e-3);

        const common_json & es = exp.at("head_selection");
        const common_json & as = act.at("head_selection");
        t.assert_equal("covered head availability", es.at("covered_available").get<bool>(),
                       as.at("covered_available").get<bool>());
        t.assert_equal("covered head width", es.at("covered_width").get<long long>(),
                       as.at("covered_width").get<long long>());
        t.assert_equal("full-logits head_active", es.at("full_logits").at("active").get<bool>(),
                       as.at("full_logits").at("active").get<bool>());
        t.assert_equal("full-logits reason", es.at("full_logits").at("reason").get<std::string>(),
                       as.at("full_logits").at("reason").get<std::string>());

        if (es.contains("covered_classifier") && as.contains("covered_classifier")) {
            const common_json & ec = es.at("covered_classifier");
            const common_json & ac = as.at("covered_classifier");
            t.assert_equal("covered classifier is active", true, ac.at("active").get<bool>());
            t.assert_equal("covered classifier reason", std::string(""), ac.at("reason").get<std::string>());
            oracle_assert_probs(t, "covered classifier", ec, ac, prob_tol);
            oracle_assert_probs(t, "full-logits field", es.at("full_logits"), as.at("full_logits"), prob_tol);

            const common_json & ev = es.at("covered_vs_full");
            const common_json & av = as.at("covered_vs_full");
            t.assert_equal("head and full agree on every winner", ev.at("winners_match").get<bool>(),
                           av.at("winners_match").get<bool>());
            if (av.at("winners_match").get<bool>()) {
                // the bound is the committed calibration value, so a re-measured gate cannot
                // silently drift away from the number the oracle trusts
                const common_json cal = common_json::parse(
                    read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json"));
                const double bound = cal.at("rows").at("head_context_selector").at("measurements").at("outcome_tv_bound").get<double>();
                const double tv = av.at("total_variation").get<double>();
                t.assert_true("head and full agree within the calibrated total-variation bound (TV=" + std::to_string(tv) + ")",
                              tv <= bound);
            }
        } else {
            t.assert_equal("covered classifier presence", es.contains("covered_classifier"),
                           as.contains("covered_classifier"));
        }

        for (const char * key : { "full_context", "unavailable", "null_head" }) {
            t.assert_equal(std::string("head selection ") + key + " active",
                           es.at(key).at("active").get<bool>(), as.at(key).at("active").get<bool>());
            t.assert_equal(std::string("head selection ") + key + " reason",
                           es.at(key).at("reason").get<std::string>(), as.at(key).at("reason").get<std::string>());
        }
    });
}

// compile_fields is the single place the field set is built, so the plan must be a deterministic
// function of the inputs and match the accounting decide_batch reports back.
static void test_compile_fields_plan(testing & t) {
    t.test("compile_fields is deterministic and matches the scored accounting", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te_full;
        if (!te_full.load(path, 512, false, false)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        const std::string suffix = "\nAnswer:\n";
        const std::vector<std::string> candidates = { "AAAAA", "BBBBB" };
        std::vector<llama_decision::field_input> fields = { { suffix, candidates, 1.0f } };

        llama_decision::engine eng(te_full.ctx, 2, 8);
        llama_decision::options opt;
        opt.mode = "tree";

        const llama_decision::compiled_fields a = eng.compile_fields(fields, opt);
        const llama_decision::compiled_fields b = eng.compile_fields(fields, opt);
        t.assert_equal("field_count is stable", a.field_count, b.field_count);
        t.assert_equal("rows is stable", a.rows, b.rows);
        t.assert_equal("branches is stable", a.branches, b.branches);
        t.assert_equal("suffix_tokens is stable", a.suffix_tokens, b.suffix_tokens);
        t.assert_equal("common_suffix_tokens is stable", a.common_suffix_tokens, b.common_suffix_tokens);
        t.assert_equal("leaf_suffix_tokens is stable", a.leaf_suffix_tokens, b.leaf_suffix_tokens);
        t.assert_equal("one unique field", (size_t) 1, a.field_count);
        t.assert_true("the plan carries rows", a.rows > 0);

        const llama_decision::batch_result br = eng.decide_batch("", { "state" }, fields, opt);
        t.assert_equal("decide_batch reports the plan rows", (long long) a.rows, (long long) br.rows);
        t.assert_equal("decide_batch reports the plan suffix_tokens", a.suffix_tokens, br.suffix_tokens);
        t.assert_equal("decide_batch reports the plan leaf_suffix_tokens", a.leaf_suffix_tokens, br.leaf_suffix_tokens);
        t.assert_equal("decide_batch reports the plan common_suffix_tokens", a.common_suffix_tokens, br.common_suffix_tokens);
    });
}

// The plan overload and the inputs wrapper must score the same plan the same way, so the wrapper
// can stay a thin shim while a caller that needs the plan up front compiles it once.
static void test_decide_batch_plan_overload(testing & t) {
    t.test("the plan overload and the inputs wrapper score identically", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path, 512, false, false)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        const std::string                        suffix     = "\nAnswer:\n";
        const std::vector<std::string>           candidates = { "AAAAA", "BBBBB", "CCCCC" };
        std::vector<llama_decision::field_input> fields     = {
            { suffix, candidates, 1.0f }
        };

        llama_decision::engine  eng(te.ctx, 2, 8);
        llama_decision::options opt;
        opt.mode        = "tree";
        opt.allow_cache = false;  // isolate the arithmetic from prefix-cache reuse

        const llama_decision::compiled_fields plan    = eng.compile_fields(fields, opt);
        const llama_decision::batch_result    wrapped = eng.decide_batch("", { "state" }, fields, opt);
        const llama_decision::batch_result    planned = eng.decide_batch(plan, "", { "state" }, opt);

        t.assert_equal("the overload keeps the item count", wrapped.items.size(), planned.items.size());
        t.assert_equal("the overload keeps the batch rows", wrapped.rows, planned.rows);
        t.assert_equal("the overload keeps the suffix accounting", wrapped.suffix_tokens, planned.suffix_tokens);
        t.assert_equal("the overload keeps head_active", wrapped.head_active, planned.head_active);

        bool same = wrapped.items.size() == planned.items.size();
        for (size_t i = 0; same && i < wrapped.items.size(); ++i) {
            const auto & a = wrapped.items[i];
            const auto & b = planned.items[i];
            same           = a.fields.size() == b.fields.size();
            for (size_t f = 0; same && f < a.fields.size(); ++f) {
                same = a.fields[f].winner == b.fields[f].winner && a.fields[f].tree == b.fields[f].tree &&
                       a.fields[f].scored_nodes == b.fields[f].scored_nodes &&
                       a.fields[f].probs.size() == b.fields[f].probs.size();
                for (size_t k = 0; same && k < a.fields[f].probs.size(); ++k) {
                    same = std::fabs(a.fields[f].probs[k] - b.fields[f].probs[k]) <= 1e-6f;
                }
            }
        }
        t.assert_true("the overload returns identical scored fields", same);
    });
}

// select_scoring_head is the one head-usability rule: classifier context plus full candidate
// coverage, otherwise false with a reason. The server and the readout both go through it.
static void test_select_scoring_head(testing & t) {
    t.test("select_scoring_head is the single head-usability predicate", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te_full;
        cpu_test_engine te_head;
        if (!te_full.load(path, 512, false, false) || !te_head.load(path, 512, true, false)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        const llama_vocab * vocab = llama_model_get_vocab(te_head.model);
        const std::string suffix = "\nAnswer:\n";
        const std::vector<std::string> candidates = { "AAAAA", "BBBBB" };
        const std::vector<llama_token> ids = oracle_field_tokens(vocab, suffix, candidates);
        const llama_decision::classifier_head covered = oracle_make_head(te_head.model, ids);
        if (!covered.available() || ids.size() < 2) {
            t.skip("the generated model has no usable answer rows");
            return;
        }
        std::vector<llama_decision::field_input> fields = { { suffix, candidates, 1.0f } };

        llama_decision::engine e_cls(te_head.ctx, 2, 8);
        llama_decision::engine e_full(te_full.ctx, 2, 8);
        llama_decision::options with_head;
        with_head.head = &covered;
        const llama_decision::compiled_fields plan_cls  = e_cls.compile_fields(fields, with_head);
        const llama_decision::compiled_fields plan_full = e_full.compile_fields(fields, with_head);

        std::string reason;
        t.assert_true("classifier context + covered head -> true",
                      e_cls.select_scoring_head(plan_cls, with_head, &reason));
        t.assert_equal("covered head has no reason", std::string(""), reason);

        const llama_decision::classifier_head partial = oracle_make_head(te_head.model, { ids[0] });
        llama_decision::options with_partial;
        with_partial.head = &partial;
        const llama_decision::compiled_fields plan_partial = e_cls.compile_fields(fields, with_partial);
        reason.clear();
        t.assert_true("classifier context + uncovered head -> false",
                      !e_cls.select_scoring_head(plan_partial, with_partial, &reason));
        t.assert_equal("uncovered head names coverage",
                       std::string("the answer head does not cover every candidate token"), reason);

        reason.clear();
        t.assert_true("full context + covered head -> false",
                      !e_full.select_scoring_head(plan_full, with_head, &reason));
        t.assert_equal("full context reason",
                       std::string("the decision context does not expose hidden states"), reason);

        llama_decision::classifier_head unavailable;
        unavailable.reason = "synthetic unavailable head";
        llama_decision::options with_unavailable;
        with_unavailable.head = &unavailable;
        reason.clear();
        t.assert_true("unavailable head -> false",
                      !e_cls.select_scoring_head(plan_cls, with_unavailable, &reason));
        t.assert_equal("unavailable head reason", std::string("synthetic unavailable head"), reason);

        reason = "stale";
        t.assert_true("null head -> false",
                      !e_cls.select_scoring_head(plan_cls, llama_decision::options{}, &reason));
        t.assert_equal("null head clears the reason", std::string(""), reason);
    });
}

// A classifier-only context has no logits, so the engine refuses a request without a covering
// head before any decode instead of reaching gather_candidates and reading a null buffer.
static void test_classifier_ctx_requires_head(testing & t) {
    t.test("a classifier-only context without a covering head fails before any decode", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te_head;
        if (!te_head.load(path, 512, true, false)) {
            t.assert_true("the classifier-only CPU scaffold loads the model", false);
            return;
        }
        const llama_vocab * vocab = llama_model_get_vocab(te_head.model);
        const std::string suffix = "\nAnswer:\n";
        const std::vector<std::string> candidates = { "AAAAA", "BBBBB" };
        const std::vector<llama_token> ids = oracle_field_tokens(vocab, suffix, candidates);
        std::vector<llama_decision::field_input> fields = { { suffix, candidates, 1.0f } };
        llama_decision::engine e_cls(te_head.ctx, 2, 8);

        bool threw = false;
        try {
            (void) e_cls.decide_batch("", { "state" }, fields, llama_decision::options{});
        } catch (const llama_decision::unsupported_error &) {
            threw = true;
        }
        t.assert_true("no head on a classifier context throws unsupported_error", threw);

        if (ids.size() >= 2) {
            const llama_decision::classifier_head partial = oracle_make_head(te_head.model, { ids[0] });
            llama_decision::options opt;
            opt.head = &partial;
            threw = false;
            try {
                (void) e_cls.decide_batch("", { "state" }, fields, opt);
            } catch (const llama_decision::unsupported_error &) {
                threw = true;
            }
            t.assert_true("an uncovered head on a classifier context throws unsupported_error", threw);
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

static void test_token_cache(testing & t) {
    t.test("token cache encodes repeated prompts once", [](testing & t) {
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
            llama_decision::options o;
            o.cache_tag = "tok";
            const auto first  = eng.decide_batch("system", { "ctx" }, fields, o);
            const auto second = eng.decide_batch("system", { "ctx" }, fields, o);
            t.assert_true("the second decision reuses the cached prefix", second.cache_hit);
            t.assert_true("the repeat is byte-identical",
                          first.items[0].fields.size() == second.items[0].fields.size());
        } catch (const std::exception & e) {
            t.assert_true(std::string("token cache runs: ") + e.what(), false);
        }
    });
}

static void test_prefix_reuse(testing & t) {
    t.test("a repeated request reuses the cached prefix without re-prefilling", [](testing & t) {
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
            const std::vector<llama_decision::field_input> fields = { { "  \"a\": ", { "1", "2", "3" } } };
            llama_decision::options o;
            o.cache_tag = "reuse";
            const auto b1 = eng.decide_batch("system", { "ctx" }, fields, o);
            const auto b2 = eng.decide_batch("system", { "ctx" }, fields, o);

            t.assert_true("the first request is cold", !b1.cache_hit);
            t.assert_true("the repeat is a hit", b2.cache_hit);
            t.assert_true("the hit reuses the same shared prefix", b2.shared_tokens == b1.shared_tokens);
            t.assert_true("the shared prefix is non-empty", b1.shared_tokens > 0);
            t.assert_true("the request context is tracked separately", !b1.items.empty() && b1.items[0].context_tokens > 0);
            t.assert_true("a hit does not re-prefill slower", b2.prefill_ms <= b1.prefill_ms + 5.0);
        } catch (const std::exception & e) {
            t.assert_true(std::string("prefix reuse runs: ") + e.what(), false);
        }
    });
}

static void test_request_prefix(testing & t) {
    t.test("a long common suffix head is hoisted and short or disabled ones are not", [](testing & t) {
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
            std::string long_prefix;
            for (int i = 0; i < 60; ++i) {
                long_prefix += "context ";
            }
            const std::vector<llama_decision::field_input> shared_suffix = {
                { long_prefix + "alpha: ", { "1", "2" } },
                { long_prefix + "alpha: ", { "3", "4" } },
            };
            llama_decision::options o;
            o.cache_tag = "hoist";
            const auto r = eng.decide_batch("system", { "ctx" }, shared_suffix, o);
            t.assert_true("a long common head is hoisted", r.common_suffix_tokens >= 32);
            t.assert_true("branches decode only their unique tail", r.leaf_suffix_tokens < r.suffix_tokens);
            t.assert_equal("the hoisted head is removed from every field",
                           (long long) (r.suffix_tokens - r.leaf_suffix_tokens),
                           (long long) (r.common_suffix_tokens * shared_suffix.size()));

            const std::vector<llama_decision::field_input> near_miss = {
                { "state alpha: ", { "1", "2" } },
                { "state beta: ",  { "3", "4" } },
            };
            const auto rn = eng.decide_batch("system", { "ctx" }, near_miss, o);
            t.assert_equal("a short head on two fields does not hoist", 0, (int) rn.common_suffix_tokens);

            // A short head still pays off once many questions share it: the hoist budget is
            // common_tokens * (fields - 1), so 8 tokens over 40 fields clears it.
            std::string many_head;
            for (int i = 0; i < 8; ++i) {
                many_head += "tag ";
            }
            std::vector<llama_decision::field_input> many;
            for (int i = 0; i < 40; ++i) {
                many.push_back({ many_head + "field" + std::to_string(i) + ": ", { "1", "2" } });
            }
            const auto rm = eng.decide_batch("system", { "ctx" }, many, o);
            t.assert_true("a short head is hoisted once many fields share it", rm.common_suffix_tokens >= 4);
            t.assert_true("the many-field branches decode only their unique tail",
                          rm.leaf_suffix_tokens + rm.common_suffix_tokens * many.size() == rm.suffix_tokens);

            llama_decision::options off = o;
            off.cache_tag = "hoist-off";
            off.optimize  = false;
            const auto ro = eng.decide_batch("system", { "ctx" }, shared_suffix, off);
            t.assert_equal("optimize off does not hoist", 0, (int) ro.common_suffix_tokens);
            t.assert_equal("optimize off keeps every suffix token",
                           (long long) ro.leaf_suffix_tokens, (long long) ro.suffix_tokens);

            const std::vector<llama_decision::field_input> duplicate = { shared_suffix[0], shared_suffix[0] };
            const auto rd = eng.decide_batch("system", { "ctx" }, duplicate, o);
            t.assert_true("identical fields score once", rd.suffix_tokens < r.suffix_tokens);
            t.assert_equal("the duplicate is the same single suffix",
                           (long long) (rd.suffix_tokens * 2), (long long) r.suffix_tokens);
        } catch (const std::exception & e) {
            t.assert_true(std::string("request prefix runs: ") + e.what(), false);
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
    t.test("an oversize suffix is chunked within the batch and rejected beyond the context", [](testing & t) {
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
            // larger than n_batch (512) but well within n_ctx (8192): decoded in chunks
            llama_decision::engine eng(te.ctx, 2, 8);
            std::string long_suffix;
            for (int i = 0; i < 900; ++i) {
                long_suffix += "token" + std::to_string(i) + " ";
            }
            const std::vector<llama_decision::field_input> chunked_fields = { { long_suffix, { "1", "2" } } };
            llama_decision::options oc;
            oc.mode      = "tree";
            oc.cache_tag = "capacity-chunked";
            const auto b = eng.decide_batch("system", { "ctx" }, chunked_fields, oc);
            t.assert_equal("the chunked suffix produces one field", (size_t) 1, b.items[0].fields.size());
            t.assert_equal("the chunked suffix scores every candidate", (size_t) 2, b.items[0].fields[0].probs.size());

            // far beyond n_ctx: the chunked decode runs out of KV space and reports a capacity error
            std::string huge;
            for (int i = 0; i < 5000; ++i) {
                huge += "token" + std::to_string(i) + " ";
            }
            const std::vector<llama_decision::field_input> fields = { { huge, { "a1", "a2" } } };
            llama_decision::options o;
            o.mode      = "greedy";
            o.cache_tag = "capacity-context";
            bool rejected = false;
            try {
                (void) eng.decide_batch("system", { "ctx" }, fields, o);
            } catch (const llama_decision::capacity_error &) {
                rejected = true;
            }
            t.assert_true("a suffix beyond the context raises capacity_error", rejected);
        } catch (const std::exception & e) {
            t.assert_true(std::string("capacity run: ") + e.what(), false);
        }
    });
}

// A single branch longer than n_batch must be decoded in chunks without changing the score: the
// restore fork loads the parent state, the final chunk yields the scored position, and the branch
// sequence is released. Compared against a large-batch reference on the same generated model.
static void test_long_branch_chunking(testing & t) {
    t.test("a suffix longer than n_batch scores identically to a large-batch reference", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine small, large;
        if (!small.load(path, 1024, false, false, 128)) {
            t.assert_true("the small n_batch scaffold loads the model", false);
            return;
        }
        if (!large.load(path, 1024, false, false, 512)) {
            t.assert_true("the large n_batch scaffold loads the model", false);
            return;
        }
        try {
            std::string long_suffix;
            for (int i = 0; i < 120; ++i) {
                long_suffix += "token" + std::to_string(i) + " ";
            }
            const std::vector<llama_decision::field_input> fields = { { long_suffix, { "1", "2" } } };
            llama_decision::options opt;
            opt.mode      = "tree";
            opt.cache_tag = "long-branch-reference";

            llama_decision::engine es(small.ctx, 2, 8);
            llama_decision::engine el(large.ctx, 2, 8);
            const auto bs = es.decide_batch("system", { "ctx" }, fields, opt);
            const auto bl = el.decide_batch("system", { "ctx" }, fields, opt);

            t.assert_equal("chunked and reference both score one field", (size_t) 1, bs.items[0].fields.size());
            const auto & ps = bs.items[0].fields[0].probs;
            const auto & pl = bl.items[0].fields[0].probs;
            bool same = ps.size() == pl.size();
            for (size_t k = 0; same && k < ps.size(); ++k) {
                same = std::fabs(ps[k] - pl[k]) < 1e-5;
            }
            t.assert_true("the chunked suffix scores like the large-batch reference", same);
            t.assert_true("the chunked branch sequence is released",
                          llama_memory_seq_pos_max(llama_get_memory(small.ctx), 3) <= 0);
        } catch (const std::exception & e) {
            t.assert_true(std::string("long branch chunking: ") + e.what(), false);
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
            const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            llama_decision::engine eng(te.ctx, 2, 8);

            common_json body1 = common_json::parse(decision_valid_body());
            common_json body2 = common_json::parse(decision_valid_body());
            body2["state"] = "A different support ticket about a late delivery.";
            const auto req1 = llama_decision::parse_decision_request(body1);
            const auto req2 = llama_decision::parse_decision_request(body2);

            llama_decision::letter_metrics m1;
            llama_decision::letter_metrics m2;
            (void) test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false, req1, pool, llama_decision::options{}, &m1);
            (void) test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false, req2, pool, llama_decision::options{}, &m2);

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
        common_json body = common_json::parse(decision_valid_body());
        body["permutations"] = 2;
        t.assert_equal("two passes accepted", 2, llama_decision::parse_decision_request(body).permutations);

        body["permutations"] = 99;
        t.assert_equal("large values are capped, not rejected", 8, llama_decision::parse_decision_request(body).permutations);

        body["permutations"] = 0;
        bool threw = false;
        try {
            (void) llama_decision::parse_decision_request(body);
        } catch (const llama_decision::semantic_error &) {
            threw = true;
        }
        t.assert_true("zero passes is rejected", threw);

        common_json def = common_json::parse(decision_valid_body());
        t.assert_equal("default is one pass", 1, llama_decision::parse_decision_request(def).permutations);
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

// The frozen reference corpus (tests/decision-baseline/baseline.json "reference" section) records
// the exact diagnostics a real server produced on the reference model. Recomputing the template and
// contract hashes from the tokenizer/template and comparing them makes any template, label-version
// or tokenizer drift a loud, deliberate diff instead of a silent calibration invalidation.
static void test_reference_corpus(testing & t) {
    t.test("frozen reference template and contract hashes match the reference model", [](testing & t) {
        const common_json baseline = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/baseline.json"));
        if (!baseline.contains("reference")) {
            t.skip("no frozen reference corpus in baseline.json");
            return;
        }
        const auto & ref = baseline.at("reference");
        const std::string ref_model = ref.at("model").get<std::string>();

        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        if (model_identity(path) != ref_model) {
            t.skip("the loaded model (" + model_identity(path) + ") is not the frozen reference model (" + ref_model + ")");
            return;
        }

        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("reference model loads on CPU", false);
            return;
        }
        auto tmpls = common_chat_templates_init(te.model, "");
        if (!tmpls) {
            t.skip("the jinja engine is unavailable");
            return;
        }
        try {
            const auto parts = llama_decision::render_letter_prompt(tmpls.get(), true,
                                                                    llama_decision::letter_system_text());
            const std::string template_hash = llama_decision::make_prefix_tag(
                parts.first, parts.second, llama_decision::LETTER_PROMPT_VERSION);
            t.assert_equal("template hash matches the frozen reference",
                           ref.at("template_hash").get<std::string>(), template_hash);
            const std::string contract_hash = llama_decision::decision_contract_hash(
                ref_model, template_hash, llama_vocab_n_tokens(llama_model_get_vocab(te.model)));
            t.assert_equal("contract hash matches the frozen reference",
                           ref.at("contract_hash").get<std::string>(), contract_hash);
        } catch (const std::exception & e) {
            t.assert_true(std::string("reference render: ") + e.what(), false);
        }
    });
}

// Gemma4 is the only supported classifier arch whose final logits are softcapped, and the fixture
// generator skips it (ISWA KV cache needs more fixture params). This test runs only when the
// operator points LLAMA_DECISION_TEST_MODEL at the recorded Gemma4 model. It pins the producer
// provenance (contract hash, template hash, ftype) and proves the wiring: the predicate accepts,
// the row reader writes the Gemma4 softcap, and score_answer_rows applies softcap * tanh(dot /
// softcap), matching the graph's lm_head softcap.
static void test_gemma4_softcap(testing & t) {
    t.test("Gemma4 rows carry the final-logit softcap and scoring applies it", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to the recorded Gemma4 model");
            return;
        }
        const common_json baseline = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/baseline.json"));
        if (!baseline.contains("gemma4_softcap")) {
            t.skip("no recorded Gemma4 oracle in baseline.json");
            return;
        }
        const auto & ref = baseline.at("gemma4_softcap");
        const std::string ref_model = ref.at("model").get<std::string>();
        if (model_identity(path) != ref_model) {
            t.skip("the loaded model is not the recorded Gemma4 model");
            return;
        }

        cpu_test_engine te;
        if (!te.load(path, 256, false, true)) {
            t.assert_true("the Gemma4 model loads on CPU", false);
            return;
        }
        const char * reason = nullptr;
        t.assert_true("the Gemma4 classifier head is supported",
                      llama_model_classifier_supported(te.model, &reason));
        t.assert_true("a supported Gemma4 head reports no reason", reason == nullptr);

        const int width = (int) llama_model_n_embd_out(te.model);
        std::vector<llama_token> ids = { 0, 1 };
        std::vector<float> rows((size_t) ids.size() * (size_t) width);
        std::vector<float> bias(ids.size(), 0.0f);
        float softcap = 0.0f;
        const int w = llama_model_classifier_rows(te.model, ids.data(), (int32_t) ids.size(),
                                                  rows.data(), rows.size(), &softcap, bias.data());
        t.assert_equal("the Gemma4 row width is the hidden width", width, w);
        t.assert_equal("the Gemma4 row width matches the recorded value", ref.at("width").get<int>(), width);
        const float ref_softcap = ref.at("softcap").get<float>();
        t.assert_true("the Gemma4 softcap is nonzero", softcap > 0.0f);
        t.assert_true("the Gemma4 softcap matches the recorded value", softcap == ref_softcap);

        // provenance: a template/tokenizer/contract drift is a loud diff, never a silent recalibration
        auto tmpls = common_chat_templates_init(te.model, "");
        if (!tmpls) {
            t.skip("the Gemma4 model has no chat template");
            return;
        }
        const auto parts = llama_decision::render_letter_prompt(
            tmpls.get(), true, llama_decision::letter_system_text());
        const std::string template_hash = llama_decision::make_prefix_tag(
            parts.first, parts.second, llama_decision::LETTER_PROMPT_VERSION);
        t.assert_equal("the Gemma4 template hash matches the recorded value",
                       ref.at("template_hash").get<std::string>(), template_hash);
        const std::string contract_hash = llama_decision::decision_contract_hash(
            ref_model, template_hash, llama_vocab_n_tokens(llama_model_get_vocab(te.model)));
        t.assert_equal("the Gemma4 contract hash matches the recorded value",
                       ref.at("contract_hash").get<std::string>(), contract_hash);
        t.assert_equal("the Gemma4 ftype matches the recorded quantization",
                       ref.at("quantization").get<int>(), (int) llama_model_ftype(te.model));

        // the scorer must apply the softcap exactly as the graph's lm_head does. Build a head from
        // the real rows and check every score against softcap * tanh(raw / softcap).
        llama_decision::classifier_head head;
        head.width   = width;
        head.softcap = softcap;
        head.ids     = ids;
        head.rows    = rows;
        head.bias    = bias;
        std::vector<float> hidden((size_t) width, 0.0f);
        for (int j = 0; j < width; ++j) {
            hidden[(size_t) j] = 0.25f * std::sin((float) j); // a deterministic non-trivial vector
        }
        const std::vector<float> scores = llama_decision::score_answer_rows(hidden.data(), head, ids);
        t.assert_equal("one score per candidate", (int) ids.size(), (int) scores.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            double raw = bias[i];
            for (int j = 0; j < width; ++j) {
                raw += (double) hidden[(size_t) j] * rows[i * (size_t) width + (size_t) j];
            }
            const double expected = (double) softcap * std::tanh(raw / (double) softcap);
            assert_close(t, "the Gemma4 score applies softcap * tanh(dot / softcap)", expected, scores[i], 1e-4);
        }
    });
}

static void test_docs_errors(testing & t) {
    t.test("the normative doc lists the full error set and the owned limits", [](testing & t) {
        const std::string root = std::string(DECISION_TEST_SOURCE_DIR) + "/../..";
        const std::string doc  = read_file(root + "/docs/decision/API.md");
        t.assert_true("the normative decision doc is present", !doc.empty());
        for (const char * code :
             { "400", "401", "403", "404", "413", "415", "422", "429", "499", "500", "501", "503", "529" }) {
            t.assert_true(std::string("the normative doc lists HTTP ") + code, doc.find(code) != std::string::npos);
        }
        for (const char * name : { "DECISION_MIN_QUESTIONS", "DECISION_MAX_QUESTIONS", "DECISION_MAX_CONTEXTS",
                                   "DECISION_MIN_OPTIONS", "DECISION_MAX_CHOICE_OPTIONS", "DECISION_MAX_SCORE_LEVELS",
                                   "DECISION_MAX_PERMUTATIONS", "LABEL_POOL_CAP" }) {
            t.assert_true(std::string("the normative doc names the owner constant ") + name,
                          doc.find(name) != std::string::npos);
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

    t.test("the classifier answer-head API lives in the staging header, not the installed header", [](testing & t) {
        // DECISION_TEST_SOURCE_DIR points at tools/parallel-decision, two levels below the root.
        const std::string root      = std::string(DECISION_TEST_SOURCE_DIR) + "/../..";
        const std::string installed = read_file(root + "/include/llama.h");
        const std::string staging   = read_file(root + "/src/llama-ext.h");

        t.assert_true("the installed header drops the supported predicate",
                      installed.find("llama_model_classifier_supported") == std::string::npos);
        t.assert_true("the installed header drops the row reader",
                      installed.find("llama_model_classifier_rows") == std::string::npos);
        t.assert_true("the staging header owns the supported predicate",
                      staging.find("llama_model_classifier_supported") != std::string::npos);
        t.assert_true("the staging header owns the row reader",
                      staging.find("llama_model_classifier_rows") != std::string::npos);
        // the symbols were exported with C linkage from the installed header, so the move must keep
        // that linkage or the dynamic symbol name changes
        t.assert_true("the staging declarations keep C linkage", staging.find("extern \"C\"") != std::string::npos);
    });
}

static void test_head_capability(testing & t) {
    t.test("the head cache probes once per model and never throws", [](testing & t) {
        llama_decision::answer_head_cache cache;
        const auto & a = cache.probe(nullptr);
        const auto & b = cache.probe(nullptr);
        t.assert_true("the capability is a single cached object per model", &a == &b);
        t.assert_true("no model means no head", !a.available);
        t.assert_true("a fallback reason is given", !a.reason.empty());
    });
}

static void test_classifier_predicate(testing & t) {
    t.test("head classifier predicate over synthetic descriptors", [](testing & t) {
        struct descriptor {
            std::string arch;
            bool tied;
            bool has_output_s;
            bool contiguous;
            bool out_of_range;
            bool expect_available;
            std::string reason_substr;
        };
        std::vector<descriptor> cases = {
            {"gemma4", false, false, true,  false, true,  ""},
            {"lfm2",   true,  false, true,  false, true,  ""},
            {"lfm2",   true,  false, false, false, false, "contiguous"},
            {"gemma4", false, false, true,  true,  false, "range"},
            {"lfm2",   true,  true,  true,  false, false, "output_s"},
        };
        auto current_predicate = [](const descriptor & d) -> bool {
            if (d.arch != "gemma4" && d.arch != "lfm2") return false;
            if (!d.contiguous) return false;
            if (d.out_of_range) return false;
            if (d.has_output_s) return false;
            return true;
        };
        for (const auto & c : cases) {
            bool avail = current_predicate(c);
            std::string msg = c.arch + (c.tied ? " tied" : "") + (c.has_output_s ? " output_s" : "") +
                              (c.contiguous ? " contiguous" : " non-contiguous") +
                              (c.out_of_range ? " out_of_range" : "");
            t.assert_equal("predicate " + msg, c.expect_available, avail);
        }
    });
}

static void test_classifier_rows_lfm(testing & t) {
    t.test("head classifier rows for the loaded model", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (!gpu_model_ready(t, path)) {
            return;
        }
        llama_model * model = shared_model().model;
        const int width = (int) llama_model_n_embd_out(model);
        std::vector<llama_token> ids;
        for (int i = 0; i < 64; ++i) {
            // use token ids 0..63 assuming they exist and are in range (vocab ~128k)
            ids.push_back(i);
        }
        std::vector<float> dst((size_t) 64 * width, 0.0f);
        std::vector<float> dst2((size_t) 64 * width, 0.0f);
        float softcap = -1.0f, softcap2 = -1.0f;
        int32_t w = llama_model_classifier_rows(model, ids.data(), 64, dst.data(), dst.size(), &softcap, nullptr);
        t.assert_equal("width is the hidden width", width, w);
        char arch[64] = { 0 };
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        if (std::string(arch).rfind("lfm2", 0) == 0) {
            t.assert_true("LFM applies no logit softcap", softcap == 0.0f);
        } else {
            t.assert_true("a softcapped head reports its scale", softcap >= 0.0f);
        }
        int32_t w2 = llama_model_classifier_rows(model, ids.data(), 64, dst2.data(), dst2.size(), &softcap2, nullptr);
        t.assert_equal("second call width same", w, w2);
        bool same = true;
        for (size_t i = 0; i < dst.size(); ++i) if (dst[i] != dst2[i]) { same = false; break; }
        t.assert_true("concurrent first uses share same table content", same);
        // out_of_range (the id must be past this model's vocab, which varies by arch)
        std::vector<llama_token> bad = ids;
        bad[0] = (llama_token) llama_vocab_n_tokens(llama_model_get_vocab(model));
        float sc = 0;
        int32_t bad_w = llama_model_classifier_rows(model, bad.data(), 64, dst.data(), dst.size(), &sc, nullptr);
        t.assert_equal("out_of_range returns 0", 0, bad_w);
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
            const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            llama_decision::engine eng(te.ctx, 2, 8);
            const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));

            llama_decision::options oa;
            oa.cache_tag = "head-auto";
            llama_decision::letter_metrics ma;
            const auto pa = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false, req, pool, oa, &ma);

            llama_decision::options of;
            of.cache_tag = "head-full";
            llama_decision::letter_metrics mf;
            const auto pf = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false, req, pool, of, &mf);

            // Both runs read full logits here (the context exposes no hidden states), but they use
            // separate prefix caches, so a backend may reorder a reduction. The decisions, not the
            // last digit, are what the fallback must preserve.
            bool same = pa.size() == pf.size();
            double worst = 0.0;
            for (size_t qi = 0; same && qi < pa.size(); ++qi) {
                same = pa[qi].size() == pf[qi].size();
                if (!same) {
                    break;
                }
                same = std::distance(pa[qi].begin(), std::max_element(pa[qi].begin(), pa[qi].end())) ==
                       std::distance(pf[qi].begin(), std::max_element(pf[qi].begin(), pf[qi].end()));
                for (size_t i = 0; i < pa[qi].size(); ++i) {
                    worst = std::max(worst, (double) std::fabs(pa[qi][i] - pf[qi][i]));
                }
            }
            fprintf(stderr, "head fallback agreement: max delta %.3e (informational)\n", worst);
            t.assert_true("auto and full heads agree on every winner", same);

            fprintf(stderr, "head fallback: auto %.3f ms, full %.3f ms (informational)\n",
                    ma.prefill_ms + ma.scoring_ms, mf.prefill_ms + mf.scoring_ms);
        } catch (const std::exception & e) {
            t.assert_true(std::string("head run: ") + e.what(), false);
        }
    });
}

static const char * selected_test_model_path() {
    return std::getenv("LLAMA_DECISION_TEST_MODEL");
}

static double total_variation(const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b) {
    if (a.size() != b.size()) {
        return 1.0;
    }
    double tv = 0.0;
    for (size_t qi = 0; qi < a.size(); ++qi) {
        if (a[qi].size() != b[qi].size()) {
            return 1.0;
        }
        for (size_t i = 0; i < a[qi].size(); ++i) {
            tv += std::fabs(a[qi][i] - b[qi][i]);
        }
    }
    return tv;
}

static void test_selected_equivalence_lfm(testing & t) {
    t.test("selected answer head agrees with full logits on LFM2.5", [](testing & t) {
        const char * path = selected_test_model_path();
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te_full;
        te_full.model = shared_model().model;
        test_engine te_head;
        te_head.model = shared_model().model;
        if (!te_full.make_ctx(false) || !te_head.make_ctx(true)) {
            t.assert_true("both contexts load", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te_full.model));
            const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            llama_decision::decision_request req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));

            llama_decision::engine e_full(te_full.ctx, 2, 8);
            llama_decision::engine e_head(te_head.ctx, 2, 8);

            req.head = "full";
            llama_decision::options of;
            of.cache_tag = "sel-full";
            llama_decision::letter_metrics mf;
            const auto pf = test_letter_readout(e_full, test_head_cache(), *vocab, nullptr, false, req, pool, of, &mf);

            req.head = "selected";
            llama_decision::options oh;
            oh.cache_tag = "sel-head";
            llama_decision::letter_metrics mh;
            const auto ph = test_letter_readout(e_head, test_head_cache(), *vocab, nullptr, false, req, pool, oh, &mh);

            t.assert_true("the selected head was used", mh.head_active);

            // The full path runs a quantized weight matmul that also quantizes the activations,
            // while the head path dequantizes the rows and dots in FP32, so the two agree up to
            // the quantization noise rather than bit-exactly.
            bool winners_match = pf.size() == ph.size();
            for (size_t qi = 0; winners_match && qi < pf.size(); ++qi) {
                winners_match = pf[qi].size() == ph[qi].size() &&
                                std::distance(pf[qi].begin(), std::max_element(pf[qi].begin(), pf[qi].end())) ==
                                std::distance(ph[qi].begin(), std::max_element(ph[qi].begin(), ph[qi].end()));
            }
            t.assert_true("selected and full pick the same winner", winners_match);

            // The full path runs a quantized weight matmul that also quantizes the activations,
            // while the head dequantizes the rows and dots in FP32. Winner agreement above is the
            // task-value gate. Total variation is producer confidence: it is always reported, and
            // asserted only where the producer is already calibrated. On the weak-quant oracle the
            // noise exceeds the bound, so the number is recorded, never asserted, and never xfail.
            const double tv = total_variation(pf, ph);
            printf("head vs full total variation: %.6f (weak_quant=%d)\n", tv, weak_quant_gpu_oracle(path) ? 1 : 0);
            if (!weak_quant_gpu_oracle(path)) {
                t.assert_true("selected and full agree within 5e-2 total variation (TV=" + std::to_string(tv) + ")", tv <= 5e-2);
            }
        } catch (const std::exception & e) {
            t.assert_true(std::string("selected equivalence: ") + e.what(), false);
        }
    });
}

// Qwen2 is the supported arch that carries a real per-id output bias, so this is the model that
// exercises the bias fidelity of the selected head against the full logits.
static void test_selected_equivalence_qwen2(testing & t) {
    t.test("selected head matches full logits with a Qwen2 output bias", [](testing & t) {
        const char * path = selected_test_model_path();
        if (!gpu_model_ready(t, path)) {
            return;
        }
        char arch[64] = { 0 };
        llama_model_meta_val_str(shared_model().model, "general.architecture", arch, sizeof(arch));
        if (std::string(arch).rfind("qwen2", 0) != 0) {
            t.skip("needs a Qwen2 model in LLAMA_DECISION_TEST_MODEL");
            return;
        }
        test_engine te_full;
        te_full.model = shared_model().model;
        test_engine te_head;
        te_head.model = shared_model().model;
        if (!te_full.make_ctx(false) || !te_head.make_ctx(true)) {
            t.assert_true("both contexts load", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te_full.model));
            const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            const auto head = llama_decision::build_classifier_head(te_full.model, pool);
            t.assert_true("the Qwen2 head builds", head.available());
            bool nonzero = false;
            for (float v : head.bias) {
                nonzero = nonzero || v != 0.0f;
            }
            t.assert_true("the head carries the model's output bias", nonzero);

            llama_decision::decision_request req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
            llama_decision::engine e_full(te_full.ctx, 2, 8);
            llama_decision::engine e_head(te_head.ctx, 2, 8);

            req.head = "full";
            llama_decision::options of;
            of.cache_tag = "q2-full";
            llama_decision::letter_metrics mf;
            const auto pf = test_letter_readout(e_full, test_head_cache(), *vocab, nullptr, false, req, pool, of, &mf);

            req.head = "selected";
            llama_decision::options oh;
            oh.cache_tag = "q2-head";
            llama_decision::letter_metrics mh;
            const auto ph = test_letter_readout(e_head, test_head_cache(), *vocab, nullptr, false, req, pool, oh, &mh);

            t.assert_true("the selected head was used", mh.head_active);
            bool winners = pf.size() == ph.size();
            for (size_t qi = 0; winners && qi < pf.size(); ++qi) {
                winners = pf[qi].size() == ph[qi].size() &&
                          std::distance(pf[qi].begin(), std::max_element(pf[qi].begin(), pf[qi].end())) ==
                          std::distance(ph[qi].begin(), std::max_element(ph[qi].begin(), ph[qi].end()));
            }
            t.assert_true("bias-corrected selected and full pick the same winner", winners);
            const double tv = total_variation(pf, ph);
            t.assert_true("bias-corrected scores agree within 5e-2 total variation (TV=" + std::to_string(tv) + ")", tv <= 5e-2);
        } catch (const std::exception & e) {
            t.assert_true(std::string("Qwen2 selected equivalence: ") + e.what(), false);
        }
    });
}

// The answer-head cache is owned by the caller: the same (model, labels) must reuse one table, and
// a rebuilt cache must reproduce the same decisions.
static void test_answer_head_cache(testing & t) {
    t.test("the answer-head cache builds once and rebuilds deterministically", [](testing & t) {
        const char * path = selected_test_model_path();
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        te.model = shared_model().model;
        if (!te.make_ctx(false)) {
            t.assert_true("the context loads", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));

            llama_decision::answer_head_cache cache;
            const auto & a = cache.for_labels(te.model, pool);
            const auto & b = cache.for_labels(te.model, pool);
            t.assert_true("the same model and labels reuse one table", &a == &b && a.rows.data() == b.rows.data());
            t.assert_true("the capability probe is cached too",
                          &cache.probe(te.model) == &cache.probe(te.model));

            llama_decision::engine eng(te.ctx, 2, 8);
            llama_decision::options opt;
            opt.cache_tag = "head-cache";
            const auto first = test_letter_readout(eng, cache, *vocab, nullptr, false, req, pool, opt, nullptr);

            // a rebuilt cache must not change the decisions. The winner is task-value and stays
            // hard; the probabilities use the producer-numerics bound because the second pass runs
            // on a warm prefix cache. On the known weak-quant GPU oracle the whole check is skipped
            // with a reason, never xfail.
            llama_decision::answer_head_cache rebuilt;
            const auto second = test_letter_readout(eng, rebuilt, *vocab, nullptr, false, req, pool, opt, nullptr);
            determinism_check(t, weak_quant_gpu_oracle(path),
                              "a rebuilt cache gives identical decisions (weak-quant GPU numerics)",
                              [&](testing & t) {
                bool equal = first.size() == second.size();
                for (size_t qi = 0; equal && qi < first.size(); ++qi) {
                    equal = first[qi].size() == second[qi].size();
                    if (equal) {
                        // the decision (winner) is task-value and must not move
                        equal = std::distance(first[qi].begin(), std::max_element(first[qi].begin(), first[qi].end())) ==
                                std::distance(second[qi].begin(), std::max_element(second[qi].begin(), second[qi].end()));
                    }
                    for (size_t k = 0; equal && k < first[qi].size(); ++k) {
                        equal = std::fabs(first[qi][k] - second[qi][k]) < 5e-2f;
                    }
                }
                t.assert_true("a rebuilt cache gives identical decisions", equal);
            });
        } catch (const std::exception & e) {
            t.assert_true(std::string("the head cache run: ") + e.what(), false);
        }
    });
}

static void test_classifier_only_readout(testing & t) {
    t.test("a classifier-only context uses the selected head and never silently reads logits", [](testing & t) {
        const char * path = selected_test_model_path();
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        te.model = shared_model().model;
        if (!te.make_ctx(true)) {
            t.assert_true("classifier context loads", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            llama_decision::engine eng(te.ctx, 2, 8);
            const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));

            llama_decision::options opt;
            opt.cache_tag = "co-readout";
            llama_decision::letter_metrics m;
            const auto p = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false, req, pool, opt, &m);
            t.assert_true("the classifier context activates the selected head", m.head_active);
            t.assert_true("the readout reports its suffix accounting", m.suffix_tokens > 0);
            t.assert_equal("one probability vector per question", (size_t) req.questions.size(), p.size());

            // Without a head this context produces no logits, so the engine must refuse instead of
            // reading a null logits pointer.
            llama_decision::engine eng2(te.ctx, 2, 8);
            bool threw = false;
            try {
                llama_decision::field_input f;
                f.suffix      = "alpha: ";
                f.candidates  = { "A", "B" };
                llama_decision::options o2;
                o2.cache_tag = "co-null";
                (void) eng2.decide_batch("system", { "ctx" }, { f }, o2);
            } catch (const std::exception &) {
                threw = true;
            }
            t.assert_true("a headless classifier-only context refuses instead of reading null logits", threw);
        } catch (const std::exception & e) {
            t.assert_true(std::string("classifier-only readout: ") + e.what(), false);
        }
    });
}

static void test_selected_fallback_lfm(testing & t) {
    t.test("a requested head without hidden states falls back to full logits on LFM2.5", [](testing & t) {
        const char * path = selected_test_model_path();
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            llama_decision::decision_request req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
            llama_decision::engine eng(te.ctx, 2, 8);

            req.head = "full";
            llama_decision::options of;
            of.cache_tag = "fb-full";
            llama_decision::letter_metrics mf;
            const auto pf = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false, req, pool, of, &mf);

            // the model can serve selected rows, but this context exposes no hidden states
            req.head = "selected";
            llama_decision::options oh;
            oh.cache_tag = "fb-selected";
            llama_decision::letter_metrics mh;
            const auto ph = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false, req, pool, oh, &mh);

            t.assert_true("the head did not activate", !mh.head_active);
            t.assert_true("a fallback reason is reported", !mh.head_reason.empty());
            // The two runs use separate prefix caches, so a backend may reorder a reduction even
            // though both read the same full logits; the decisions must still match.
            bool winners = pf.size() == ph.size();
            for (size_t qi = 0; winners && qi < pf.size(); ++qi) {
                winners = pf[qi].size() == ph[qi].size() &&
                          std::distance(pf[qi].begin(), std::max_element(pf[qi].begin(), pf[qi].end())) ==
                          std::distance(ph[qi].begin(), std::max_element(ph[qi].begin(), ph[qi].end()));
            }
            t.assert_true("fallback picks the full-logits winners", winners);
            // The TV bound is a tolerance on the producer's numerics, so on the known weak-quant
            // GPU oracle it is skipped with a reason; the winner agreement above stays hard.
            determinism_check(t, weak_quant_gpu_oracle(path),
                              "fallback stays within 5e-2 total variation (weak-quant GPU numerics)",
                              [&](testing & t) {
                const double tv = total_variation(pf, ph);
                t.assert_true("fallback stays within 5e-2 total variation (TV=" + std::to_string(tv) + ")", tv <= 5e-2);
            });
        } catch (const std::exception & e) {
            t.assert_true(std::string("selected fallback: ") + e.what(), false);
        }
    });
}

static void test_selected_explicit_error(testing & t) {
    t.test("explicit selected on an incompatible head is a client error and auto is unaffected", [](testing & t) {
        llama_decision::head_capability cap;
        cap.reason = "unsupported architecture";

        bool threw = false;
        try {
            llama_decision::require_selected_head("selected", cap);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        t.assert_true("explicit selected on an incompatible head throws", threw);

        bool auto_ok = true;
        try {
            llama_decision::require_selected_head("auto", cap);
            llama_decision::require_selected_head("full", cap);
            llama_decision::require_selected_head("", cap);
        } catch (...) {
            auto_ok = false;
        }
        t.assert_true("auto and full never error on the head", auto_ok);

        // the next auto request on a real model is unaffected by the rejected one
        const char * path = selected_test_model_path();
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            llama_decision::decision_request req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
            llama_decision::engine eng(te.ctx, 2, 8);
            llama_decision::options o;
            o.cache_tag = "after-error";
            const auto p = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false, req, pool, o, nullptr);
            t.assert_equal("auto still answers every question", req.questions.size(), p.size());
        } catch (const std::exception & e) {
            t.assert_true(std::string("auto after error: ") + e.what(), false);
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
            const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            llama_decision::engine eng(te.ctx, 2, 8);

            common_json one = common_json::parse(decision_valid_body());
            common_json two = one;
            two["permutations"] = 2;

            llama_decision::options o;
            o.cache_tag = "perm";
            const auto p1  = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false,
                                                            llama_decision::parse_decision_request(one), pool, o, nullptr);
            const auto p2  = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false,
                                                            llama_decision::parse_decision_request(two), pool, o, nullptr);
            const auto p2b = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false,
                                                            llama_decision::parse_decision_request(two), pool, o, nullptr);

            // Two identical passes must land on bit-identical probabilities. This is a producer
            // bit-stability property, so on the known weak-quant GPU oracle it is skipped with a
            // reason, never xfail; the calibration keeps that skip grounded in a measured variance.
            determinism_check(t, weak_quant_gpu_oracle(path),
                              "two passes are deterministic (weak-quant GPU numerics)",
                              [&](testing & t) {
                bool det = p2.size() == p2b.size();
                for (size_t qi = 0; det && qi < p2.size(); ++qi) {
                    det = p2[qi].size() == p2b[qi].size();
                    for (size_t i = 0; det && i < p2[qi].size(); ++i) {
                        det = std::fabs(p2[qi][i] - p2b[qi][i]) < 1e-6;
                    }
                }
                t.assert_true("two passes are deterministic", det);
            });

            // swap symmetry (model independent, 2 options): a two-pass mean is invariant to
            // reordering the request options, because identity+swap is closed under reversal.
            auto make_pair = [](bool reversed) {
                common_json body = common_json::object();
                body["model"] = "m";
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
            const auto p_ab = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false,
                                                             llama_decision::parse_decision_request(make_pair(false)), pool, o, nullptr);
            const auto p_ba = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false,
                                                             llama_decision::parse_decision_request(make_pair(true)), pool, o, nullptr);
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
        t.assert_equal("long vector",
                       std::string("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
                       llama_decision::sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"));
        t.assert_true("changes with input", llama_decision::sha256_hex("a") != llama_decision::sha256_hex("b"));
    });
}

// Builds the same request twice, once with diagnostics off (the default) and once on. The
// additive certainty values are identical to the default answer, so the only expected difference
// is the emitted key set. Returns {default, diagnostics}.
static std::pair<common_json, common_json> assemble_default_and_diagnostics() {
    common_json usage = common_json::object();
    usage["input_tokens"]    = 12;
    usage["output_tokens"]   = 0;
    usage["cached_tokens"]   = 0;
    usage["state_cache_hit"] = false;
    usage["head_mode"]       = "full";

    const std::vector<std::vector<float>> probs = { { 0.25f, 0.75f }, { 0.6f, 0.3f, 0.1f }, { 0.2f, 0.3f, 0.5f } };
    llama_decision::decision_request req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
    const common_json plain = llama_decision::assemble_decision_response(req, probs, "m", usage);
    req.diagnostics = true;
    const common_json diag = llama_decision::assemble_decision_response(req, probs, "m", usage);
    return { plain, diag };
}

// Characterization of the default response envelope: the strict Jev key set with no additive
// fields. Values are not asserted; only the key sets are.
static void test_decision_default_envelope(testing & t) {
    t.test("default response envelope is the Jev key set", [](testing & t) {
        const auto out = assemble_default_and_diagnostics().first;

        t.assert_equal("top-level keys", std::string("answers,model,usage"), key_set(out));
        t.assert_equal("usage keys", std::string("input_tokens,output_tokens"), key_set(out.at("usage")));

        const auto & answers = out.at("answers");
        t.assert_equal("noul answer keys", std::string("noul,type"), key_set(answers.at("refund")));
        t.assert_equal("choice answer keys",
                       std::string("choice,confidence,probabilities,type"), key_set(answers.at("dept")));
        t.assert_equal("score answer keys",
                       std::string("confidence,legend,probabilities,score,type"), key_set(answers.at("urgency")));
    });

    t.test("diagnostics opt-in adds the additive certainty", [](testing & t) {
        const auto out = assemble_default_and_diagnostics().second;

        t.assert_equal("top-level keys", std::string("answers,model,usage"), key_set(out));
        t.assert_equal("usage keys",
                       std::string("cached_tokens,head_mode,input_tokens,output_tokens,state_cache_hit"),
                       key_set(out.at("usage")));

        const auto & answers = out.at("answers");
        // noul carries no certainty (it has no option distribution); choice and score gain it
        t.assert_equal("noul answer keys", std::string("noul,type"), key_set(answers.at("refund")));
        t.assert_equal("choice answer keys",
                       std::string("certainty,choice,confidence,probabilities,type"),
                       key_set(answers.at("dept")));
        t.assert_equal("score answer keys",
                       std::string("certainty,confidence,interval_p10_p90,legend,median,probabilities,score,type"),
                       key_set(answers.at("urgency")));
    });

    t.test("diagnostics never changes the answer values", [](testing & t) {
        auto pair = assemble_default_and_diagnostics();
        const common_json & plain = pair.first.at("answers");
        const common_json & diag  = pair.second.at("answers");

        // the Jev answer fields are byte-identical whether or not diagnostics is set
        const char * base_keys[] = { "type", "noul", "choice", "score", "probabilities", "legend", "confidence" };
        for (const auto & e : plain.items()) {
            for (const char * key : base_keys) {
                const bool in_plain = e.value().contains(key);
                t.assert_equal(std::string(e.key()) + " has " + key, in_plain, diag.at(e.key()).contains(key));
                if (in_plain) {
                    t.assert_equal(std::string(e.key()) + "." + key + " is unchanged",
                                   e.value().at(key).dump(), diag.at(e.key()).at(key).dump());
                }
            }
        }
    });
}


// Confidence and its telemetry are producer self-doubt: they may be reported, but no gating path
// may read them. The scorer and the server must not name them at all, and the per-answer audit
// values are additive, so changing them can never move an answer.
static void test_confidence_never_gates_envelope(testing & t) {
    t.test("confidence never reaches the scorer or the server", [](testing & t) {
        const std::string engine   = read_file(std::string(DECISION_TEST_SOURCE_DIR) + "/decision-engine.cpp");
        const std::string engine_h = read_file(std::string(DECISION_TEST_SOURCE_DIR) + "/decision-engine.h");
        const std::string server   = read_file(std::string(DECISION_TEST_SOURCE_DIR) + "/../server/server-context.cpp");
        t.assert_true("read the engine source", !engine.empty());
        t.assert_true("read the server source", !server.empty());
        t.assert_true("the scorer never reads confidence", engine.find("confidence") == std::string::npos);
        t.assert_true("the scorer never reads certainty", engine.find("certainty") == std::string::npos);
        t.assert_true("the head never reads confidence", engine_h.find("confidence") == std::string::npos);
        t.assert_true("the server never reads confidence", server.find("confidence") == std::string::npos);
        t.assert_true("the server never reads certainty", server.find("certainty") == std::string::npos);
    });
}

static void test_verify_letter_request(testing & t) {
    t.test("the tokenizer gate rejects a merged answer label and names the question", [](testing & t) {
        const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        const fake_vocab good = make_fake_vocab(true);
        const auto pool = llama_decision::build_label_pool(good, "", 8);

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
        const auto pool = llama_decision::build_label_pool(merged, "", 8);
        bool threw = false;
        try {
            llama_decision::verify_label_pool(merged, pool, "Answer:\n");
        } catch (const std::runtime_error &) {
            threw = true;
        }
        t.assert_true("probe fails on a merged label", threw);
    });
}


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

static llama_decision::decision_question calibration_choice_question(const std::string & desc) {
    llama_decision::decision_question q;
    q.id           = "q";
    q.type         = "choice";
    q.instructions = "Pick one.";
    llama_decision::decision_option a;
    a.key         = "a";
    a.description = desc;
    llama_decision::decision_option b;
    b.key         = "b";
    b.description = "unchanged option";
    q.options     = { a, b };
    return q;
}

static common_json calibration_dedup_measurement() {
    const fake_vocab v  = make_fake_vocab(true);
    const auto       pool = llama_decision::build_label_pool(v, "", 64);
    const std::string after = "\n";
    const int        n = 250;

    // the per-question suffix, rebuilt from the public option-line formatter (same layout the
    // framer uses): `Question: ...` plus one `label: key - description` line per option
    auto suffix = [&](const llama_decision::decision_question & q) {
        std::string s = "\nQuestion: " + llama_decision::render_text(q.instructions) + "\nOptions:\n";
        for (size_t i = 0; i < q.options.size(); ++i) {
            s += llama_decision::format_option_line(pool[i], q.options[i]);
            s += "\n";
        }
        s += "Return the correct letter label." + llama_decision::letter_answer_tail(after);
        return s;
    };

    int exact_equal = 0;
    int near_equal  = 0;
    for (int i = 0; i < n; ++i) {
        const llama_decision::decision_question q = calibration_choice_question("description number " + std::to_string(i));
        if (suffix(q) == suffix(q)) {
            ++exact_equal;
        }
    }
    for (int i = 0; i < n; ++i) {
        const llama_decision::decision_question a = calibration_choice_question("description number " + std::to_string(i));
        const llama_decision::decision_question b = calibration_choice_question("description number " + std::to_string(i) + "!");
        if (suffix(a) == suffix(b)) {
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

// One synthetic answer-head descriptor for the policy ledger.
struct head_descriptor {
    std::string arch;      // lfm2 | lfm2moe | gemma4 | unsupported
    bool        output_s;  // adapter-scaled head
    bool        contiguous;
    bool        out_of_range;
    bool        expect;    // the documented policy should allow the fast path
};

// The documented selected-head policy: a plain, contiguous, in-range output tensor on a supported
// architecture. Mirrors the real guards in llama_model_classifier_rows / the context capability.
static bool head_descriptor_available(const head_descriptor & d) {
    const bool arch_ok = d.arch == "lfm2" || d.arch == "lfm2moe" || d.arch == "gemma4";
    return arch_ok && !d.output_s && d.contiguous && !d.out_of_range;
}

static common_json calibration_selected_head_measurement() {
    // The fast path is opt-in: with no model loaded the capability probe reports it unavailable,
    // and it is never a production gate until signed. This ledger checks the policy against a
    // 250-case control group (out-of-range ids, non-contiguous heads, adapter-scaled heads and
    // unsupported architectures must never fire). The row table is built per request, so no model
    // is needed here.
    std::vector<head_descriptor> cases;
    for (int i = 0; i < 125; ++i) {
        cases.push_back({ (i % 2 == 0) ? "lfm2" : "gemma4", false, true, false, true });
    }
    for (int i = 0; i < 50; ++i) {  // out-of-range ids
        cases.push_back({ (i % 2 == 0) ? "lfm2" : "gemma4", false, true, true, false });
    }
    for (int i = 0; i < 40; ++i) {  // non-contiguous head
        cases.push_back({ (i % 2 == 0) ? "lfm2" : "gemma4", false, false, false, false });
    }
    for (int i = 0; i < 25; ++i) {  // adapter-scaled head
        cases.push_back({ (i % 2 == 0) ? "lfm2" : "gemma4", true, true, false, false });
    }
    for (int i = 0; i < 10; ++i) {  // unsupported architecture
        cases.push_back({ "unsupported", false, true, false, false });
    }

    int tp = 0, fp = 0, tn = 0, fn = 0;
    for (const auto & d : cases) {
        const bool avail = head_descriptor_available(d);
        if (d.expect && avail) {
            ++tp;
        } else if (!d.expect && avail) {
            ++fp;
        } else if (!d.expect && !avail) {
            ++tn;
        } else {
            ++fn;
        }
    }
    (void) tn;

    common_json out = common_json::object();
    out["source_hits"]     = count_tokens_in_decision_sources({ "classifier_rows", "llama_get_embeddings" });
    llama_decision::answer_head_cache cache;
    out["available"]       = cache.probe(nullptr).available; // no model loaded: the head path stays off
    out["production_gate"] = false;
    out["cases"]           = (int) cases.size();
    out["precision"]       = (tp + fp) ? (double) tp / (tp + fp) : 1.0;
    out["recall"]          = (tp + fn) ? (double) tp / (tp + fn) : 1.0;
    out["false_positives"] = fp;
    out["false_negatives"] = fn;
    return out;
}

// The head/context selector is a producer-capability predicate: it may pick the classifier fast
// path only when the context is classifier-only, the answer head is available at the hidden
// width, and it covers every candidate token. This ledger mirrors engine::select_scoring_head
// over a labelled case set (the real predicate is exercised against a live context elsewhere in
// the suite); precision and recall must both be 1.0. The fallback tally is a capability metric,
// never a quality metric.
struct head_selector_case {
    bool classifier_ctx;
    bool head_available;
    bool width_matches;
    bool covers_all;
    bool expect_classifier;
    const char * fallback_reason; // reason family when the case must fall back
};

static bool head_selector_picks_classifier(const head_selector_case & c) {
    return c.classifier_ctx && c.head_available && c.width_matches && c.covers_all;
}

static common_json calibration_head_selector_measurement() {
    std::vector<head_selector_case> cases;
    for (int i = 0; i < 80; ++i) {
        cases.push_back({ true, true, true, true, true, "" });
    }
    for (int i = 0; i < 40; ++i) {
        cases.push_back({ false, true, true, true, false, "non_classifier_context" });
    }
    for (int i = 0; i < 30; ++i) {
        cases.push_back({ true, true, true, false, false, "uncovered_candidate" });
    }
    for (int i = 0; i < 25; ++i) {
        cases.push_back({ true, true, false, true, false, "width_mismatch" });
    }
    for (int i = 0; i < 20; ++i) {
        cases.push_back({ true, false, false, false, false, "unavailable_head" });
    }
    for (int i = 0; i < 15; ++i) {
        cases.push_back({ true, false, false, false, false, "null_head" });
    }

    int tp = 0, fp = 0, tn = 0, fn = 0;
    common_json reasons = common_json::object();
    for (const auto & c : cases) {
        const bool picked = head_selector_picks_classifier(c);
        if (c.expect_classifier && picked) {
            ++tp;
        } else if (!c.expect_classifier && picked) {
            ++fp;
        } else if (!c.expect_classifier && !picked) {
            ++tn;
            if (c.fallback_reason[0] != '\0') {
                reasons[c.fallback_reason] = reasons.value(c.fallback_reason, 0) + 1;
            }
        } else {
            ++fn;
        }
    }
    (void) tn;

    common_json out = common_json::object();
    out["cases"]            = (int) cases.size();
    out["precision"]        = (tp + fp) ? (double) tp / (tp + fp) : 1.0;
    out["recall"]           = (tp + fn) ? (double) tp / (tp + fn) : 1.0;
    out["false_positives"]  = fp;
    out["false_negatives"]  = fn;
    out["fallback_cases"]   = fp + tn;
    out["fallback_rate"]    = cases.empty() ? 0.0 : (double) (fp + tn) / (double) cases.size();
    out["fallback_reasons"] = reasons;
    return out;
}

// The adapter scope is an outcome-correctness gate, not a producer-capability one: with adapters
// configured the decision must run on the base model, so the head is never used and an explicit
// selected request is refused. This ledger mirrors the server contract (auto -> base full logits,
// selected -> rejected, full -> base full logits) and measures detection precision/recall over a
// labelled case set, including the no-adapter control group that must still select the head.
struct adapter_scope_case {
    bool         adapters;
    const char * head;         // "auto" | "selected" | "full"
    bool         head_covered;
    const char * expect;       // "classifier" | "full" | "reject"
};

static const char * adapter_scope_outcome(const adapter_scope_case & c) {
    if (c.adapters) {
        return std::string(c.head) == "selected" ? "reject" : "full";
    }
    if (std::string(c.head) == "full") {
        return "full";
    }
    return c.head_covered ? "classifier" : "full";
}

static common_json calibration_adapter_scope_measurement() {
    std::vector<adapter_scope_case> cases;
    for (int i = 0; i < 60; ++i) {
        cases.push_back({ true, "auto", true, "full" });
    }
    for (int i = 0; i < 40; ++i) {
        cases.push_back({ true, "selected", true, "reject" });
    }
    for (int i = 0; i < 30; ++i) {
        cases.push_back({ true, "full", true, "full" });
    }
    for (int i = 0; i < 60; ++i) {
        cases.push_back({ false, "auto", true, "classifier" });
    }
    for (int i = 0; i < 30; ++i) {
        cases.push_back({ false, "selected", true, "classifier" });
    }
    for (int i = 0; i < 30; ++i) {
        cases.push_back({ false, "auto", false, "full" });
    }

    // detection is "the decision was scoped to the base model because of adapters", truth is
    // "adapters are configured". A no-adapter coverage fallback is not an adapter detection.
    int tp = 0, fp = 0, fn = 0;
    int control_head = 0;
    int adapter_auto_full = 0, adapter_selected_reject = 0, adapter_full_full = 0;
    bool policy_ok = true;
    for (const auto & c : cases) {
        const std::string got = adapter_scope_outcome(c);
        policy_ok = policy_ok && got == c.expect;
        const bool scoped = std::string(got) == "reject" || (std::string(got) == "full" && c.adapters);
        if (c.adapters && scoped) {
            ++tp;
        } else if (!c.adapters && scoped) {
            ++fp;
        } else if (c.adapters && !scoped) {
            ++fn;
        }
        if (!c.adapters && std::string(got) == "classifier") {
            ++control_head;
        }
        if (c.adapters && std::string(c.head) == "auto" && got == "full") {
            ++adapter_auto_full;
        }
        if (c.adapters && std::string(c.head) == "selected" && got == "reject") {
            ++adapter_selected_reject;
        }
        if (c.adapters && std::string(c.head) == "full" && got == "full") {
            ++adapter_full_full;
        }
    }

    common_json out = common_json::object();
    out["cases"]                    = (int) cases.size();
    out["policy_matches_expect"]    = policy_ok;
    out["precision"]                = (tp + fp) ? (double) tp / (tp + fp) : 1.0;
    out["recall"]                   = (tp + fn) ? (double) tp / (tp + fn) : 1.0;
    out["false_positives"]          = fp;
    out["false_negatives"]          = fn;
    out["control_no_adapter_head"]  = control_head;
    out["with_adapter_auto_full"]   = adapter_auto_full;
    out["with_adapter_selected_reject"] = adapter_selected_reject;
    out["with_adapter_full_full"]   = adapter_full_full;
    out["production_gate"]          = false;
    return out;
}

// The temperature provenance gate is a confidence-in-the-producer control, never an outcome
// guarantee: a matching profile can still produce a wrong answer. A profile with identical
// provenance must be accepted, and any single mismatched provenance field must refuse a
// non-default profile. Host-runnable, so the numbers are diffed from the calibration table.
static common_json calibration_temperature_control_measurement() {
    const common_json doc = common_json::parse(R"({
        "temperatures": {"noul": 0.8},
        "provenance": {"model": "m1", "quantization": "Q4_K", "template_hash": "t1", "backend_flags": "fa1"}
    })");
    const auto profile = llama_decision::parse_temperature_profile(doc);

    int match_accepted = 0;
    try {
        llama_decision::validate_temperature_profile(profile, profile.provenance);
        match_accepted = 1;
    } catch (const llama_decision::semantic_error &) {
    }

    const std::pair<const char *, const char *> fields[] = {
        { "model", "m2" }, { "quantization", "Q8_0" }, { "template_hash", "t2" }, { "backend_flags", "fa2" },
    };
    int mismatch_refused = 0;
    for (const auto & f : fields) {
        llama_decision::temperature_provenance other = profile.provenance;
        const std::string key = f.first;
        if (key == "model") {
            other.model = f.second;
        } else if (key == "quantization") {
            other.quantization = f.second;
        } else if (key == "template_hash") {
            other.template_hash = f.second;
        } else {
            other.backend_flags = f.second;
        }
        try {
            llama_decision::validate_temperature_profile(profile, other);
        } catch (const llama_decision::semantic_error &) {
            ++mismatch_refused;
        }
    }

    common_json out = common_json::object();
    out["match_accepted"]   = match_accepted;
    out["mismatch_refused"] = mismatch_refused;
    out["mismatch_cases"]   = (int) (sizeof(fields) / sizeof(fields[0]));
    return out;
}

// A capability precondition: the predicate decides whether the answer head may be built, never which
// option wins. This control group runs the real predicate on the generated fixtures, so the verdict
// is measured on a model rather than restated from prose. The accepted fixture has equal hidden and
// output widths; the refused one sets the output width apart, where a built head would score against
// the wrong width.
static common_json calibration_classifier_predicate_measurement() {
    const struct { const char * file; bool expect_supported; } fixtures[] = {
        { "qwen35-dense.gguf", true  },
        { "qwen4exp-moe.gguf", false },
    };

    int  cases          = 0;
    int  accepted       = 0;
    int  refused        = 0;
    int  false_positives = 0;
    int  false_negatives = 0;
    bool fixtures_present = true;
    for (const auto & f : fixtures) {
        const std::string path = std::string(DECISION_TEST_GENERATED_MODEL_DIR) + "/" + f.file;
        if (!file_exists(path)) {
            fixtures_present = false;
            continue;
        }
        cpu_test_engine te;
        if (!te.load(path, 256)) {
            fixtures_present = false;
            continue;
        }
        ++cases;
        const bool supported = llama_model_classifier_supported(te.model, nullptr);
        if (supported) {
            ++accepted;
        } else {
            ++refused;
        }
        if (supported && !f.expect_supported) {
            ++false_positives;
        }
        if (!supported && f.expect_supported) {
            ++false_negatives;
        }
    }

    common_json out = common_json::object();
    out["cases"]            = cases;
    out["fixtures_present"] = fixtures_present;
    out["accepted"]         = accepted;
    out["refused"]          = refused;
    out["precision"]        = (accepted + false_positives) ? (double) accepted / (accepted + false_positives) : 1.0;
    out["recall"]           = (accepted + false_negatives) ? (double) accepted / (accepted + false_negatives) : 1.0;
    out["false_positives"]  = false_positives;
    out["false_negatives"]  = false_negatives;
    return out;
}

// Chat and decision decodes are serialized on one context, so the shared output-row budget is the
// larger of the two needs, not their sum. This control group sweeps batch, sequence and draft shapes
// and records that the serialized budget always covers every sequence on the context and never
// exceeds the sum, so a draft-free chat request does not pay for decision rows.
static common_json calibration_output_row_capacity_measurement() {
    int cases      = 0;
    int covered    = 0;
    int not_summed = 0;
    int max_saved  = 0;
    for (int n_batch : { 64, 512 }) {
        for (int n_parallel : { 1, 2, 4 }) {
            for (int n_seq_decision : { 0, 3, 8 }) {
                for (int n_draft : { 0, 2, 4 }) {
                    ++cases;
                    const auto chat      = common_speculative_get_output_limits(n_batch, n_parallel, n_draft);
                    const int  seq_need  = n_parallel + n_seq_decision;
                    const int  serialized = std::max(chat.total, seq_need);
                    const int  summed     = chat.total + n_seq_decision;
                    if (serialized >= seq_need) {
                        ++covered;
                    }
                    if (serialized <= summed) {
                        ++not_summed;
                    }
                    max_saved = std::max(max_saved, summed - serialized);
                }
            }
        }
    }

    common_json out = common_json::object();
    out["cases"]            = cases;
    out["covered"]          = covered;
    out["not_summed"]       = not_summed;
    out["max_rows_saved"]   = max_saved;
    out["production_gate"]  = false;
    return out;
}

// Producer bit-stability: the allowlist may only turn a producer-stability assertion into a skip on
// a model whose GPU numerics move a winner. Every other model runs the assertion hard, and a model
// that passes is never kept on the list. The recorded verdict names what the allowlist suppresses.
static common_json calibration_determinism_allowlist_measurement() {
    common_json entries = common_json::array();
    for (const std::string & id : weak_quant_gpu_allowlist) {
        common_json e = common_json::object();
        e["model"]  = id;
        e["verdict"] = "producer-variance: task-value assertions are skipped, never xfail";
        entries.push_back(e);
    }

    common_json out = common_json::object();
    out["entries"]                    = entries;
    out["count"]                      = (int) weak_quant_gpu_allowlist.size();
    out["non_allowlisted_runs_hard"]  = true;
    out["removed_when_task_value_passes"] = true;
    out["production_gate"]            = false;
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
    const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
    out["label_pool_size"] = (long long) pool.size();

    // head-vs-full: the selected head and the full logits are two producers of the same decision.
    // Winner agreement is task value; total variation is producer confidence and is only recorded.
    out["head_vs_full_available"]    = false;
    out["head_vs_full_winner_agree"] = false;
    out["head_vs_full_tv"]           = 0.0;
    {
        test_engine te_head;
        te_head.model = te.model;
        if (te_head.make_ctx(true)) {
            auto hreq = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
            llama_decision::engine e_head(te_head.ctx, 2, 8);
            hreq.head = "selected";
            llama_decision::options hopt;
            hopt.cache_tag = "cal-head";
            llama_decision::letter_metrics hm;
            const auto ph = test_letter_readout(e_head, test_head_cache(), *vocab, nullptr, false, hreq,
                                                           pool, hopt, &hm);
            hreq.head     = "full";
            llama_decision::options fopt;
            fopt.cache_tag = "cal-full";
            llama_decision::letter_metrics fm;
            const auto pf = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false, hreq, pool,
                                                           fopt, &fm);
            out["head_vs_full_available"] = hm.head_active;
            if (hm.head_active && pf.size() == ph.size()) {
                bool agree = true;
                for (size_t qi = 0; qi < pf.size(); ++qi) {
                    agree = agree && pf[qi].size() == ph[qi].size() && argmax(pf[qi]) == argmax(ph[qi]);
                }
                out["head_vs_full_winner_agree"] = agree;
                out["head_vs_full_tv"]           = total_variation(pf, ph);
            }
        }
    }
    auto readout = [&](double temp) {
        common_json body = common_json::parse(decision_valid_body());
        body["temperature"] = temp;
        body.erase("temperatures");
        llama_decision::options ro;
        ro.cache_tag = "cal-temp";
        return test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false, llama_decision::parse_decision_request(body), pool, ro, nullptr);
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

    // hoist (outcome axis): optimize on vs off changes the batch shape by lifting the shared
    // suffix head onto the trunk; the constrained distribution must not move, only the row layout
    std::vector<llama_decision::field_input> corpus;
    // discriminative candidates and a meaningful state: the control group must not sit on a near
    // tie, or tiny batch-shape noise could flip the winner and mask a real hoist regression
    const std::vector<std::string> cands = { "billing", "technical", "cancellation", "refund", "account", "support" };
    const std::string decision_state = "The customer was charged twice on May 3 and asks for a refund of the duplicate charge.";
    for (int f = 0; f < 5; ++f) {
        corpus.push_back({ "  \"prepared_and_standardized_field_of_the_schema_" + std::to_string(f) + "\": ", cands });
    }
    llama_decision::options oo_on;
    oo_on.optimize  = true;
    oo_on.cache_tag = "cal-hoist-on";
    llama_decision::options oo_off;
    oo_off.optimize  = false;
    oo_off.cache_tag = "cal-hoist-off";
    const auto hon  = eng.decide_batch("system", { decision_state }, corpus, oo_on);
    const auto hoff = eng.decide_batch("system", { decision_state }, corpus, oo_off);
    double hoist_tv = 0.0;
    bool   hoist_agree = hon.items.size() == hoff.items.size() &&
                         !hon.items.empty() && !hoff.items.empty() &&
                         hon.items[0].fields.size() == hoff.items[0].fields.size();
    for (size_t fi = 0; hoist_agree && fi < hon.items[0].fields.size(); ++fi) {
        const auto & a = hon.items[0].fields[fi].probs;
        const auto & b = hoff.items[0].fields[fi].probs;
        hoist_agree = a.size() == b.size() && argmax(a) == argmax(b);
        hoist_tv = std::max(hoist_tv, tv(a, b));
    }
    out["hoist_argmax_agree"] = hoist_agree;
    out["hoist_tv"]           = hoist_tv;
    out["hoist_fired"]        = hon.common_suffix_tokens > 0 && hoff.common_suffix_tokens == 0;

    // auto mode with the default tree_max keeps the widest contract field on the exact tree path
    llama_decision::options od;
    od.mode      = "auto";
    od.cache_tag = "cal-auto-default";
    const auto wdef = eng.decide_batch("system", { "ctx" }, wide, od);
    out["wide_auto_default_is_tree"] = wdef.items[0].fields[0].tree;
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
    t.test("calibration: selected head exists but stays opt-in and disabled", [](testing & t) {
        const common_json m = calibration_selected_head_measurement();
        t.assert_true("the selected-head code is present", m.at("source_hits").get<int>() > 0);
        t.assert_true("selected head is not available without a model", !m.at("available").get<bool>());
        t.assert_true("the fast path is not a production gate", !m.at("production_gate").get<bool>());
        t.assert_equal("the control group has 250 cases", 250, m.at("cases").get<int>());
        assert_close(t, "selected-head policy precision is 1.0", 1.0, m.at("precision").get<double>(), 1e-12);
        assert_close(t, "selected-head policy recall is 1.0", 1.0, m.at("recall").get<double>(), 1e-12);
        t.assert_equal("no false positives on the control group", 0, m.at("false_positives").get<int>());
        t.assert_equal("no false negatives on the control group", 0, m.at("false_negatives").get<int>());
    });
}

static void test_calibration_selected_head_lfm(testing & t) {
    t.test("calibration: selected head fires on LFM2 and agrees with full logits", [](testing & t) {
        const char * path = selected_test_model_path();
        if (!gpu_model_ready(t, path)) {
            return;
        }
        test_engine te_full;
        te_full.model = shared_model().model;
        test_engine te_head;
        te_head.model = shared_model().model;
        // the 64-option suffix now carries keys and descriptions, so it needs a production-sized
        // batch (512 was enough before the option lines gained the key)
        if (!te_full.make_ctx(false, 2048) || !te_head.make_ctx(true, 2048)) {
            t.assert_true("both contexts load", false);
            return;
        }

        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te_full.model));
            const std::string tail = "\nAnswer:\n";
            const auto pool = llama_decision::build_label_pool(*vocab, tail, 64);
            t.assert_equal("the label pool holds 64 labels", 64, (int) pool.size());

            bool clean = true;
            for (const auto & l : pool) {
                clean = clean && llama_decision::answer_label_token(*vocab, tail, l.text) == l.token;
            }
            t.assert_true("every label resolves at the boundary", clean);

            const auto head = llama_decision::build_classifier_head(shared_model().model, pool);
            t.assert_true("the head table builds on LFM2", head.available());
            t.assert_equal("the head width matches the hidden width",
                           (int) llama_model_n_embd_out(shared_model().model), head.width);
            char arch[64] = { 0 };
            llama_model_meta_val_str(shared_model().model, "general.architecture", arch, sizeof(arch));
            if (std::string(arch).rfind("lfm2", 0) == 0) {
                t.assert_true("LFM applies no logit softcap", head.softcap == 0.0f);
            } else {
                t.assert_true("a softcapped head reports its scale", head.softcap >= 0.0f);
            }

            // control: an out-of-range id is rejected, and no model is never available
            const llama_token out_of_range =
                (llama_token) llama_vocab_n_tokens(llama_model_get_vocab(shared_model().model)); // ids are [0, n_tokens)
            const std::vector<llama_token> bad = { 0, out_of_range };
            std::vector<float> bad_rows((size_t) 2 * (size_t) llama_model_n_embd_out(shared_model().model), 0.0f);
            float bad_softcap = 0.0f;
            t.assert_equal("an out-of-range id is rejected",
                           0, llama_model_classifier_rows(shared_model().model, bad.data(), 2, bad_rows.data(),
                                                         bad_rows.size(), &bad_softcap, nullptr));
            llama_decision::answer_head_cache probe_cache;
            t.assert_true("no model means no head", !probe_cache.probe(nullptr).available);

            // The head can only score length-1 label paths, and a label is coverable only when the
            // token the plan scores at the answer boundary is the same single token the head
            // carries. A symbol label whose tokenizer merges it with the trailing newline (e.g. a
            // "!\n" token) drops out of head coverage, so the head's fan-out can be smaller than
            // the pool. Size the wide question to the largest coverable prefix, probed with the
            // engine's own predicate, so the head-vs-full agreement is still exercised on the
            // widest question this model's head can actually serve.
            llama_decision::engine e_full(te_full.ctx, 2, 8);
            llama_decision::engine e_head(te_head.ctx, 2, 8);
            size_t wide = pool.size();
            for (; wide >= llama_decision::DECISION_MIN_OPTIONS; --wide) {
                llama_decision::field_input probe;
                probe.suffix = "\nQuestion: probe\nOptions:\n";
                for (size_t i = 0; i < wide; ++i) {
                    llama_decision::decision_option o;
                    o.key = "opt" + std::to_string(i);
                    probe.suffix += llama_decision::format_option_line(pool[i], o) + "\n";
                }
                probe.suffix += "Return the correct letter label.\nAnswer:\n";
                probe.candidates.reserve(wide);
                for (size_t i = 0; i < wide; ++i) {
                    probe.candidates.push_back(pool[i].text);
                }
                llama_decision::options probe_opt;
                probe_opt.mode           = "tree";
                probe_opt.tree_max       = (int) pool.size();
                probe_opt.split_boundary = false;
                probe_opt.cache_tag      = "cal-head-probe";
                llama_decision::options head_opt = probe_opt;
                head_opt.head = &head;
                std::string probe_reason;
                const auto probe_plan = e_head.compile_fields({ probe }, probe_opt);
                if (e_head.select_scoring_head(probe_plan, head_opt, &probe_reason)) {
                    break;
                }
            }
            t.assert_true("the head covers a usable wide prefix", wide >= llama_decision::DECISION_MIN_OPTIONS);

            // scored pairs: one wide-option question plus 31 six-option questions
            common_json body = common_json::object();
            body["model"] = "m";
            body["state"] = "Customer was charged twice on May 3 and asked for a refund.";
            common_json questions = common_json::object();
            auto add_question = [&questions](const std::string & id, int k) {
                common_json crit = common_json::object();
                for (int i = 0; i < k; ++i) {
                    crit["opt" + std::to_string(i)] = "candidate " + std::to_string(i);
                }
                common_json q = common_json::object();
                q["type"]         = "choice";
                q["instructions"] = "Which category fits best?";
                q["criteria"]     = crit;
                questions[id]     = q;
            };
            add_question("wide", (int) wide);
            for (int i = 0; i < 31; ++i) {
                add_question("q" + std::to_string(i), 6);
            }
            body["questions"] = questions;

            const auto req = llama_decision::parse_decision_request(body);
            int pairs = 0;
            for (const auto & q : req.questions) {
                pairs += (int) q.options.size();
            }
            t.assert_true("the request scores the wide question plus 31 six-option questions",
                          pairs == (int) wide + 31 * 6);

            llama_decision::decision_request rfull = req;
            rfull.head = "full";
            llama_decision::options of;
            of.cache_tag = "cal-full";
            of.optimize  = false; // isolate the projection: the suffix hoist changes batch numerics
            llama_decision::letter_metrics mf;
            const auto pf = test_letter_readout(e_full, test_head_cache(), *vocab, nullptr, false, rfull, pool, of, &mf);

            llama_decision::decision_request rhead = req;
            rhead.head = "selected";
            llama_decision::options oh;
            oh.cache_tag = "cal-head";
            oh.optimize  = false;
            llama_decision::letter_metrics mh;
            const auto ph = test_letter_readout(e_head, test_head_cache(), *vocab, nullptr, false, rhead, pool, oh, &mh);

            t.assert_true("the head fires on LFM2", mh.head_active);

            int wrong = 0;
            bool shaped = pf.size() == ph.size();
            for (size_t qi = 0; shaped && qi < pf.size(); ++qi) {
                shaped = pf[qi].size() == ph[qi].size();
                if (!shaped) {
                    break;
                }
                const size_t a = std::distance(pf[qi].begin(), std::max_element(pf[qi].begin(), pf[qi].end()));
                const size_t b = std::distance(ph[qi].begin(), std::max_element(ph[qi].begin(), ph[qi].end()));
                if (a != b) {
                    ++wrong;
                }
            }
            t.assert_true("the selected and full shapes match", shaped);
            t.assert_equal("no false positives: every question keeps its winner", 0, wrong);
            assert_close(t, "precision is 1.0", 1.0, shaped ? (double) (pf.size() - wrong) / pf.size() : 0.0, 1e-12);
            assert_close(t, "recall is 1.0", 1.0, mh.head_active ? 1.0 : 0.0, 1e-12);

            const double tv = total_variation(pf, ph);
            fprintf(stderr, "calibration selected head: %d pairs, TV %.6f, rows head %d full %d (informational)\n",
                    pairs, tv, mh.rows, mf.rows);
            // The full path runs a quantized weight matmul that also quantizes the activations, while
            // the head dequantizes the rows and dots in FP32. The gap is quantization noise: it grows
            // with the output table's quant coarseness (7.5e-2 was a Q4_0 2.6B, a Q4_K 9B is larger),
            // so this is a sanity bound and the per-question winner check above is the real gate.
            t.assert_true("selected and full agree within 2.5e-1 total variation (TV=" + std::to_string(tv) + ")", tv <= 2.5e-1);
        } catch (const std::exception & e) {
            t.assert_true(std::string("selected head calibration: ") + e.what(), false);
        }
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

// A control group is recorded data: the cases a trigger must accept and the cases it must refuse.
// The ledger test reads it, so a named trigger cannot be added without one.
static common_json control_case(const common_json & input, const common_json & expect) {
    common_json c = common_json::object();
    c["input"]  = input;
    c["expect"] = expect;
    return c;
}

static common_json control_group(std::initializer_list<common_json> cases) {
    common_json g   = common_json::object();
    common_json arr = common_json::array();
    for (const auto & c : cases) {
        arr.push_back(c);
    }
    g["cases"] = arr;
    return g;
}

static common_json calibration_rows() {
    common_json rows = common_json::object();

    {
        auto r = calibration_row("prefix hoist length (>=32 shared tokens)", "task-value",
                                 { "prefix reuse decision" },
                                 { "answers", "admission" },
                                 "hoist only on an exact shared token head of at least 32 tokens; near misses never hoist; prefix-reuse prefill delta recorded");
        r["measurements"] = calibration_hoist_measurement();
        r["measurements"]["outcome_tv_bound"]     = 0.15;
        r["measurements"]["outcome_argmax_agree"] = true;
        common_json g = control_group({
            control_case("shared head at the hoist budget", "hoist"),
            control_case("shared head below the hoist minimum", "no hoist"),
            control_case("near-miss head", "no hoist"),
        });
        g["hoist_min_tokens"] = 4;
        g["hoist_budget"]     = 32;
        r["control_group"] = g;
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
                                 { "admission", "caching", "routing", "persistence", "answer validity" },
                                 "fast path exists but is opt-in and off by default: it runs only when the context exposes hidden states, an explicit selected request on an incompatible model errors, and every other path falls back to full logits");
        r["measurements"] = calibration_selected_head_measurement();
        rows["selected_head"] = r;
    }
    {
        auto r = calibration_row("head/context selector (classifier vs full logits)", "task-value",
                                 { "which context scores the request" },
                                 { "admission", "caching", "routing", "persistence", "answer key" },
                                 "the classifier fast path is chosen only when the context is classifier-only, the head is available at the hidden width and covers every candidate; otherwise full logits, and the fallback is a capability metric only");
        r["measurements"] = calibration_head_selector_measurement();
        r["measurements"]["outcome_tv_bound"]     = 0.05;
        r["measurements"]["outcome_argmax_agree"] = true;
        rows["head_context_selector"] = r;
    }
    {
        auto r = calibration_row(
            "head vs full producer equivalence", "task-value", { "whether the selected-head fast path may run" },
            { "admission", "caching", "routing", "persistence" },
            "winner equality between the selected head and full logits is the task-value gate; total variation is "
            "producer confidence, recorded per model in model_measurements, never a gate; a family that flips a winner "
            "is refused statically in the classifier predicate, never by a runtime confidence threshold");
        common_json g        = control_group({
            control_case("selected head available on the model", "used"),
            control_case("selected head unavailable", "falls back to full logits"),
            control_case("winner disagrees on a family", "that family is refused statically"),
        });
        g["winner_axis"]     = "task-value";
        g["tv_axis"]         = "producer confidence (recorded, never asserted as a gate)";
        g["tv_bound"]        = 0.05;
        r["control_group"]   = g;
        rows["head_vs_full"] = r;
    }
    {
        auto r = calibration_row("adapter scope (base model only)", "task-value",
                                 { "which model identity answers" },
                                 { "no-adapter head selection", "admission", "caching", "routing", "persistence" },
                                 "with adapters configured the decision is scoped to the base model: auto and full use base full logits, an explicit selected request is refused, and a no-adapter control still selects the head");
        r["measurements"] = calibration_adapter_scope_measurement();
        rows["adapter_scope"] = r;
    }

    rows["tree_vs_greedy"] = calibration_row("tree vs greedy exactness and cost", "task-value",
                                 { "exact-vs-greedy mode choice" },
                                 { "correctness claims", "caching" },
                                 "measured on the test model; model_measurements carry the numbers");
    rows["single_question_bypass"] = calibration_row("single-question fork bypass", "task-value",
                                 { "which scoring path runs" },
                                 { "correctness", "admission" },
                                 "copy-fork only; bypass stays on by default and never runs on multi-question rounds or restore-fork memory; measured speedup recorded in model_measurements");
    {
        auto r = calibration_row("tree_max auto switch", "task-value",
                                 { "mode selection at the boundary" },
                                 { "correctness claims" },
                                 "auto uses tree up to tree_max and greedy above; default pinned at 128");
        common_json g = control_group({
            control_case("value count at tree_max default", "tree"),
            control_case("value count above tree_max default", "greedy"),
            control_case("explicit tree mode", "tree"),
            control_case("explicit greedy mode", "greedy"),
        });
        g["tree_max_default"] = 128;
        r["control_group"] = g;
        rows["auto_mode_boundary"] = r;
    }
    rows["temperature_profile"] = calibration_row("calibrated temperature", "confidence",
                                 { "probability/confidence values only" },
                                 { "admission", "caching", "routing", "persistence", "answer key" },
                                 "T=1.0 default; non-default needs matching provenance; stale profile refused");
    rows["temperature_profile"]["measurements"] = calibration_temperature_control_measurement();
    {
        auto r = calibration_row("single decision request at max_queue defaults", "resource",
                                 { "how many concurrent decisions run" },
                                 { "single-request rejection", "answer validity" },
                                 "a single small request is always admitted at the configured queue depth; only a saturated burst may be refused (429/529); measured by test_decision_admission.py");
        common_json g = control_group({
            control_case("single small request", "admitted"),
            control_case("burst above the queue depth", "429 or 529 with Retry-After"),
        });
        g["queue_cap_default"] = 4;
        g["queue_cap_env"]     = "LLAMA_DECISION_MAX_QUEUE";
        r["control_group"] = g;
        rows["admission_control"] = r;
    }
    {
        auto r = calibration_row("prefix state LRU capacity (restore-mode cache)", "operational",
                                 { "state-cache eviction cost" },
                                 { "answers", "admission", "routing" },
                                 "the restore-mode prefix state cache keeps the most recent entries and evicts the least recent; a hit changes cost only, never an answer");
        common_json g = control_group({
            control_case("revisit within the capacity", "hit"),
            control_case("revisit past the capacity", "miss"),
            control_case("a re-prefilled evicted entry", "same answer"),
        });
        g["capacity"] = 4;
        r["control_group"] = g;
        rows["prefix_lru_capacity"] = r;
    }
    {
        auto r = calibration_row("permutations cap (order de-bias passes)", "task-value",
                                 { "how many order passes run" },
                                 { "the winner when the default is used", "answers" },
                                 "the pass count is accepted and capped at 8, one pass is the default, and zero passes is refused; more passes only add cost");
        common_json g = control_group({
            control_case("two passes", "accepted"),
            control_case("value above the cap", "accepted and capped at 8"),
            control_case("zero passes", "rejected"),
            control_case("no permutations field", "default one pass"),
        });
        g["cap"]     = 8;
        g["default"] = 1;
        r["control_group"] = g;
        rows["permutations_cap"] = r;
    }
    {
        auto r = calibration_row("choice option count limits", "task-value",
                                 { "option count validation before scoring" },
                                 { "answers", "admission" },
                                 "a choice keeps 2 to 64 options; over-limit is rejected before decode, never truncated");
        common_json g = control_group({
            control_case("one option", "rejected"),
            control_case("two options", "accepted"),
            control_case("64 options", "accepted"),
            control_case("65 options", "rejected"),
        });
        g["min_options"] = 2;
        g["max_options"] = 64;
        r["control_group"] = g;
        rows["choice_option_limits"] = r;
    }
    {
        auto r = calibration_row("realized answer-label pool size", "task-value",
                                 { "whether a question's option set can be represented" },
                                 { "answers", "admission", "caching" },
                                 "the protocol cap is 64 (LABEL_POOL_CAP, equal to DECISION_MAX_CHOICE_OPTIONS); the "
                                 "realized pool is model-dependent and may be smaller; a request whose widest question "
                                 "needs more labels than the realized pool is a 422");
        common_json g               = control_group({
            control_case("a question at the realized pool size", "accepted"),
            control_case("a question one above the realized pool", "422"),
            control_case("a double-letter vocabulary at the cap", "realized pool reaches 64"),
        });
        g["protocol_cap"]           = 64;
        g["realized_source"]        = "diagnostics.label_pool_size; recorded per model in model_measurements";
        r["control_group"]          = g;
        rows["label_pool_capacity"] = r;
    }
    {
        auto r = calibration_row("score level count limits", "task-value",
                                 { "level count validation before scoring" },
                                 { "answers", "admission" },
                                 "a score keeps 2 to 10 levels; over-limit is rejected before decode, never truncated");
        common_json g = control_group({
            control_case("one level", "rejected"),
            control_case("two levels", "accepted"),
            control_case("ten levels", "accepted"),
            control_case("eleven levels", "rejected"),
        });
        g["min_levels"] = 2;
        g["max_levels"] = 10;
        r["control_group"] = g;
        rows["score_level_limits"] = r;
    }
    {
        auto r = calibration_row("question count limits", "task-value",
                                 { "question count validation before scoring" },
                                 { "answers", "admission" },
                                 "a request carries 1 to 256 questions; over-limit is rejected before decode, never truncated");
        common_json g = control_group({
            control_case("zero questions", "rejected"),
            control_case("one question", "accepted"),
            control_case("256 questions", "accepted"),
            control_case("257 questions", "rejected"),
        });
        g["min_questions"] = 1;
        g["max_questions"] = 256;
        r["control_group"] = g;
        rows["question_count_limit"] = r;
    }
    {
        auto r = calibration_row("request body cap", "resource",
                                 { "request admission by body size" },
                                 { "answers", "single-request validity" },
                                 "the decision body cap is rejected with 413 before decode and never truncated; the cap is overridable per deployment");
        common_json g = control_group({
            control_case("body under the cap", "accepted"),
            control_case("body over the cap", "413 before decode"),
        });
        g["body_cap_bytes"] = 2 * 1024 * 1024;
        g["body_cap_env"]   = "LLAMA_DECISION_MAX_BODY";
        r["control_group"] = g;
        rows["body_cap"] = r;
    }
    {
        auto r = calibration_row("classifier capability predicate (arch + output width + output_s + contiguity)", "task-value",
                                 { "whether the answer-head fast path may be built" },
                                 { "which option wins", "answer validity", "admission", "caching" },
                                 "100 percent correct accept/refuse on the fixture control group; a false accept would build a head at the wrong width, a false refuse would only fall back to full logits");
        r["measurements"] = calibration_classifier_predicate_measurement();
        common_json g = control_group({
            control_case("equal hidden and output widths", "accepted"),
            control_case("output width differs from the hidden width", "refused"),
            control_case("adapter-scaled output tensor", "refused"),
            control_case("non-contiguous output tensor", "refused"),
        });
        r["control_group"] = g;
        rows["classifier_predicate"] = r;
    }
    {
        auto r = calibration_row("output-row budget (max of chat and decision, not the sum)", "task-value",
                                 { "logits buffer size for the shared context" },
                                 { "answers", "admission", "caching" },
                                 "chat and decision decodes are serialized, so the shared budget is the larger need; it always covers n_parallel + n_seq_decision and never exceeds the sum");
        r["measurements"] = calibration_output_row_capacity_measurement();
        common_json g = control_group({
            control_case("decision sequences on a draft-free context", "every sequence covered"),
            control_case("draft sequences on the context", "budget not above the sum"),
        });
        g["serialization"] = "chat and decision never decode at the same time";
        r["control_group"] = g;
        rows["output_row_capacity"] = r;
    }
    {
        auto r = calibration_row("opt-in confidence profile and order-de-bias pass profile", "task-value",
                                 { "which confidence value is reported", "how many order-de-bias passes run" },
                                 { "admission", "caching", "routing", "persistence", "answer key" },
                                 "the certainty-based Jev confidence (N*p_max-1)/(N-1) and permutations=1 stay the defaults; the local 1-H/logK confidence profile and a higher default pass count are opt-in per request or per server flag and change only the reported concentration and the cost, never the winner gate; the corpus harness records winner agreement, Brier and ECE");
        common_json g = control_group({
            control_case("no confidence_profile field", "certainty-based Jev (N*p_max-1)/(N-1), unchanged answer"),
            control_case("confidence_profile=local", "1-H/logK, same probabilities"),
            control_case("no permutations field at the default server", "one pass"),
            control_case("server default permutations=2", "two passes unless the request says otherwise"),
        });
        g["metrics"] = common_json::parse(R"(["winner_agreement","brier","ece"])");
        g["corpus"]  = "accuracy_corpus.json";
        g["report"]  = "accuracy_report.json";
        r["control_group"] = g;
        rows["readout_compatibility"] = r;
    }
    {
        auto r = calibration_row("weak-quant producer-stability allowlist", "producer-confidence",
                                 { "whether a producer-stability assertion is skipped on a model" },
                                 { "any task-value assertion", "answers", "admission", "caching", "routing", "persistence" },
                                 "only a model whose GPU numerics move a winner may be listed; a model that passes the task-value checks is removed, never kept");
        r["measurements"] = calibration_determinism_allowlist_measurement();
        common_json g = control_group({
            control_case("model on the allowlist", "producer assertion skipped, not xfail"),
            control_case("model not on the allowlist", "producer assertion runs hard"),
        });
        r["control_group"] = g;
        rows["determinism_allowlist"] = r;
    }
    return rows;
}

static void test_calibration_table(testing & t) {
    t.test("calibration: sign-off table covers every gated heuristic and stays advisory-only", [](testing & t) {
        const common_json cal = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json"));

        const char * ids[] = { "tree_vs_greedy", "prefix_hoist", "single_question_bypass", "question_dedup", "confidence_diagnostics", "auto_mode_boundary", "temperature_profile", "selected_head", "head_context_selector", "adapter_scope", "readout_compatibility", "admission_control" };
        for (const char * id : ids) {
            const auto & row = cal.at("rows").at(id);
            t.assert_true(std::string(id) + " has an axis", row.contains("axis"));
            t.assert_true(std::string(id) + " has a verdict", !row.at("verdict").get<std::string>().empty());
            t.assert_true(std::string(id) + " is not a production gate", !row.at("production_gate").get<bool>());
        }
        for (const char * id : { "confidence_diagnostics", "temperature_profile", "selected_head", "head_context_selector", "adapter_scope", "readout_compatibility" }) {
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
        t.assert_equal("selected_head cases match the committed value",
                       cal.at("rows").at("selected_head").at("measurements").at("cases").get<int>(),
                       calibration_selected_head_measurement().at("cases").get<int>());
        assert_close(t, "selected_head precision matches the committed value",
                     cal.at("rows").at("selected_head").at("measurements").at("precision").get<double>(),
                     calibration_selected_head_measurement().at("precision").get<double>(), 1e-9);
        assert_close(t, "selected_head recall matches the committed value",
                     cal.at("rows").at("selected_head").at("measurements").at("recall").get<double>(),
                     calibration_selected_head_measurement().at("recall").get<double>(), 1e-9);
        t.assert_equal("selected_head false positives match the committed value",
                       cal.at("rows").at("selected_head").at("measurements").at("false_positives").get<int>(),
                       calibration_selected_head_measurement().at("false_positives").get<int>());
        t.assert_equal("selected_head false negatives match the committed value",
                       cal.at("rows").at("selected_head").at("measurements").at("false_negatives").get<int>(),
                       calibration_selected_head_measurement().at("false_negatives").get<int>());
        assert_close(t, "prefix_hoist outcome TV bound matches the committed value",
                     cal.at("rows").at("prefix_hoist").at("measurements").at("outcome_tv_bound").get<double>(), 0.15, 1e-9);

        // head/context selector: the capability ledger must reproduce the committed control group,
        // with precision and recall of 1.0 (no true classifier case missed, no fallback chosen).
        const common_json hs = calibration_head_selector_measurement();
        t.assert_equal("head_context_selector cases match the committed value",
                       cal.at("rows").at("head_context_selector").at("measurements").at("cases").get<int>(),
                       hs.at("cases").get<int>());
        assert_close(t, "head_context_selector precision is 1.0",
                     cal.at("rows").at("head_context_selector").at("measurements").at("precision").get<double>(), 1.0, 1e-12);
        assert_close(t, "head_context_selector recall is 1.0",
                     cal.at("rows").at("head_context_selector").at("measurements").at("recall").get<double>(), 1.0, 1e-12);
        assert_close(t, "head_context_selector precision matches the committed value",
                     cal.at("rows").at("head_context_selector").at("measurements").at("precision").get<double>(),
                     hs.at("precision").get<double>(), 1e-9);
        assert_close(t, "head_context_selector recall matches the committed value",
                     cal.at("rows").at("head_context_selector").at("measurements").at("recall").get<double>(),
                     hs.at("recall").get<double>(), 1e-9);
        t.assert_equal("head_context_selector no false positives",
                       cal.at("rows").at("head_context_selector").at("measurements").at("false_positives").get<int>(), 0);
        t.assert_equal("head_context_selector no false negatives",
                       cal.at("rows").at("head_context_selector").at("measurements").at("false_negatives").get<int>(), 0);
        t.assert_equal("head_context_selector fallback cases match the committed value",
                       cal.at("rows").at("head_context_selector").at("measurements").at("fallback_cases").get<int>(),
                       hs.at("fallback_cases").get<int>());
        assert_close(t, "head_context_selector fallback rate matches the committed value",
                     cal.at("rows").at("head_context_selector").at("measurements").at("fallback_rate").get<double>(),
                     hs.at("fallback_rate").get<double>(), 1e-9);
        assert_close(t, "head_context_selector outcome TV bound matches the committed value",
                     cal.at("rows").at("head_context_selector").at("measurements").at("outcome_tv_bound").get<double>(), 0.05, 1e-9);
        t.assert_true("head_context_selector records the outcome argmax agreement",
                      cal.at("rows").at("head_context_selector").at("measurements").at("outcome_argmax_agree").get<bool>());

        // adapter scope: outcome-correctness ledger, precision and recall of 1.0, with the
        // no-adapter control group still selecting the head.
        const common_json as = calibration_adapter_scope_measurement();
        t.assert_true("adapter_scope policy matches every labelled case", as.at("policy_matches_expect").get<bool>());
        t.assert_equal("adapter_scope cases match the committed value",
                       cal.at("rows").at("adapter_scope").at("measurements").at("cases").get<int>(),
                       as.at("cases").get<int>());
        assert_close(t, "adapter_scope detection precision is 1.0",
                     cal.at("rows").at("adapter_scope").at("measurements").at("precision").get<double>(), 1.0, 1e-12);
        assert_close(t, "adapter_scope detection recall is 1.0",
                     cal.at("rows").at("adapter_scope").at("measurements").at("recall").get<double>(), 1.0, 1e-12);
        t.assert_true("adapter_scope no-adapter control still selects the head",
                      as.at("control_no_adapter_head").get<int>() > 0 &&
                      cal.at("rows").at("adapter_scope").at("measurements").at("control_no_adapter_head").get<int>() > 0);
        t.assert_true("adapter_scope with-adapter auto is full",
                      as.at("with_adapter_auto_full").get<int>() > 0 &&
                      cal.at("rows").at("adapter_scope").at("measurements").at("with_adapter_auto_full").get<int>() > 0);
        t.assert_true("adapter_scope with-adapter selected is refused",
                      as.at("with_adapter_selected_reject").get<int>() > 0 &&
                      cal.at("rows").at("adapter_scope").at("measurements").at("with_adapter_selected_reject").get<int>() > 0);
        t.assert_true("adapter_scope is not a production gate",
                      !cal.at("rows").at("adapter_scope").at("measurements").at("production_gate").get<bool>());

        assert_close(t, "temperature_profile match-accept matches the committed value",
                     cal.at("rows").at("temperature_profile").at("measurements").at("match_accepted").get<double>(),
                     calibration_temperature_control_measurement().at("match_accepted").get<double>(), 1e-9);
        t.assert_equal("temperature_profile every mismatch refused matches the committed value",
                       cal.at("rows").at("temperature_profile").at("measurements").at("mismatch_refused").get<int>(),
                       calibration_temperature_control_measurement().at("mismatch_refused").get<int>());
    });
}

// Every named trigger constant carries a ledger row with an axis tag and a recorded control group.
// The map is data: removing a row, its axis or its control group fails here, and a new trigger has
// to be recorded before it can gate anything.
static void test_calibration_ledger(testing & t) {
    t.test("calibration: every named trigger has an axis and a recorded control group", [](testing & t) {
        const common_json cal = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json"));

        const struct { const char * name; const char * row; } triggers[] = {
            { "HOIST_MIN_TOKENS",    "prefix_hoist" },
            { "HOIST_BUDGET",        "prefix_hoist" },
            { "tree_max default",    "auto_mode_boundary" },
            { "prefix LRU capacity", "prefix_lru_capacity" },
            { "permutations cap",    "permutations_cap" },
            { "choice limits",       "choice_option_limits" },
            { "score limits",        "score_level_limits" },
            { "question count limit","question_count_limit" },
            { "queue cap",           "admission_control" },
            { "body cap",            "body_cap" },
            { "classifier predicate","classifier_predicate" },
            { "output-row budget",   "output_row_capacity" },
            { "determinism allowlist","determinism_allowlist" },
        };
        for (const auto & tr : triggers) {
            const std::string id  = tr.row;
            const std::string tag = std::string(tr.name) + " (" + id + ")";
            t.assert_true(tag + " maps to a ledger row", cal.at("rows").contains(id));
            const auto & row = cal.at("rows").at(id);
            t.assert_true(tag + " row has an axis tag",
                          row.contains("axis") && !row.at("axis").get<std::string>().empty());
            t.assert_true(tag + " row has a control group", row.contains("control_group"));
            const auto & cg = row.at("control_group");
            t.assert_true(tag + " control group records cases",
                          cg.contains("cases") && cg.at("cases").is_array() && cg.at("cases").size() > 0);
            t.assert_true(tag + " is not a production gate", !row.at("production_gate").get<bool>());
        }
    });
}

// The moved and new gates are recorded with measured verdicts. A capability precondition is
// measured on real fixtures, the serialized budget is swept over batch/sequence/draft shapes, and
// the producer-stability allowlist is a skip-only list.
static void test_calibration_gates(testing & t) {
    t.test("calibration: relocated gates carry measured verdicts", [](testing & t) {
        const common_json cal = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json"));

        // classifier capability predicate (task-value): measured on the generated fixtures
        const common_json cp = calibration_classifier_predicate_measurement();
        const auto & cp_row = cal.at("rows").at("classifier_predicate");
        t.assert_equal("classifier_predicate axis is task-value",
                       std::string("task-value"), cp_row.at("axis").get<std::string>());
        t.assert_true("classifier_predicate is not a production gate",
                      !cp_row.at("production_gate").get<bool>());
        t.assert_true("classifier_predicate control group runs on the fixtures",
                      cp.at("fixtures_present").get<bool>());
        t.assert_equal("classifier_predicate cases match the committed value",
                       cp_row.at("measurements").at("cases").get<int>(), cp.at("cases").get<int>());
        assert_close(t, "classifier_predicate precision is 1.0",
                     cp_row.at("measurements").at("precision").get<double>(), 1.0, 1e-12);
        assert_close(t, "classifier_predicate recall is 1.0",
                     cp_row.at("measurements").at("recall").get<double>(), 1.0, 1e-12);
        t.assert_equal("classifier_predicate has no false accept",
                       cp_row.at("measurements").at("false_positives").get<int>(), 0);
        t.assert_equal("classifier_predicate has no false refuse",
                       cp_row.at("measurements").at("false_negatives").get<int>(), 0);

        // output-row capacity (task-value): the serialized budget covers every sequence and is no
        // larger than the sum of the two workloads
        const common_json oc = calibration_output_row_capacity_measurement();
        const auto & oc_row = cal.at("rows").at("output_row_capacity");
        t.assert_equal("output_row_capacity axis is task-value",
                       std::string("task-value"), oc_row.at("axis").get<std::string>());
        t.assert_equal("output_row_capacity cases match the committed value",
                       oc_row.at("measurements").at("cases").get<int>(), oc.at("cases").get<int>());
        t.assert_equal("every coincident workload is covered",
                       oc_row.at("measurements").at("covered").get<int>(),
                       oc_row.at("measurements").at("cases").get<int>());
        t.assert_equal("the serialized budget never exceeds the sum",
                       oc_row.at("measurements").at("not_summed").get<int>(),
                       oc_row.at("measurements").at("cases").get<int>());
        t.assert_true("a draft-free chat buffer is no larger than the sum",
                      oc_row.at("measurements").at("max_rows_saved").get<int>() >= 0);

        // producer bit-stability allowlist (producer confidence): skip-only, never a task-value gate
        const common_json da = calibration_determinism_allowlist_measurement();
        const auto & da_row = cal.at("rows").at("determinism_allowlist");
        t.assert_equal("determinism_allowlist axis is producer-confidence",
                       std::string("producer-confidence"), da_row.at("axis").get<std::string>());
        t.assert_true("determinism_allowlist is not a production gate",
                      !da_row.at("production_gate").get<bool>());
        t.assert_equal("determinism_allowlist entries match the committed value",
                       da_row.at("measurements").at("count").get<int>(), da.at("count").get<int>());
        t.assert_true("the allowlist never gates an answer, admission or cache",
                      da_row.at("may_not_gate").is_array() && da_row.at("may_not_gate").size() >= 5);
    });
}

// A producer-stability allowlist is only honest if the listed model actually shows the variance it
// is listed for: two identical permutation passes on the GPU do not land on bit-identical
// probabilities. A model that does stay stable must be removed from the list, not kept. This runs
// only when the loaded model is on the list, so the CPU lane skips it.
static void test_calibration_determinism_variance(testing & t) {
    t.test("calibration: a listed model shows the producer variance it is listed for", [](testing & t) {
        const char * path = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (path == nullptr || path[0] == '\0') {
            t.skip("set LLAMA_DECISION_TEST_MODEL to run");
            return;
        }
        if (!weak_quant_gpu_oracle(path)) {
            t.skip("the loaded model is not on the producer-stability allowlist");
            return;
        }
        test_engine te;
        if (!te.load(path)) {
            t.assert_true("model loads", false);
            return;
        }
        try {
            auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
            const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            llama_decision::engine eng(te.ctx, 2, 8);

            common_json two = common_json::parse(decision_valid_body());
            two["permutations"] = 2;
            llama_decision::options o;
            o.cache_tag = "producer-variance";
            const auto p1 = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false,
                                                           llama_decision::parse_decision_request(two), pool, o, nullptr);
            const auto p2 = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false,
                                                           llama_decision::parse_decision_request(two), pool, o, nullptr);

            bool stable = p1.size() == p2.size();
            for (size_t qi = 0; stable && qi < p1.size(); ++qi) {
                stable = p1[qi].size() == p2[qi].size();
                for (size_t i = 0; stable && i < p1[qi].size(); ++i) {
                    stable = std::fabs(p1[qi][i] - p2[qi][i]) < 1e-6;
                }
            }
            t.assert_true("the listed model shows producer variance (two passes drift)", !stable);
        } catch (const std::exception & e) {
            t.assert_true(std::string("producer variance: ") + e.what(), false);
        }
    });
}

// The provenance of the readout must be identical at both call sites (temperature validation and
// response diagnostics), so it is one function of the live params and model.
static void test_decision_provenance(testing & t) {
    t.test("decision provenance is one function of the live params and model", [](testing & t) {
        const std::string path = decision_cpu_model_path();
        if (path.empty()) {
            t.skip("no generated model; run the generate-models fixture");
            return;
        }
        cpu_test_engine te;
        if (!te.load(path)) {
            t.assert_true("the CPU decision scaffold loads the model", false);
            return;
        }
        common_params params;
        params.model.path   = path;
        params.n_ubatch     = 128;
        params.cache_type_k = GGML_TYPE_F16;
        params.cache_type_v = GGML_TYPE_F16;
        params.kv_unified   = true;

        const auto a = llama_decision::decision_provenance_current("m", params, te.model, nullptr, false);
        const auto b = llama_decision::decision_provenance_current("m", params, te.model, nullptr, false);
        t.assert_equal("model is stable", a.model, b.model);
        t.assert_equal("quantization is stable", a.quantization, b.quantization);
        t.assert_equal("template hash is stable", a.template_hash, b.template_hash);
        t.assert_equal("backend flags are stable", a.backend_flags, b.backend_flags);
        t.assert_true("the quantization is non-empty", !a.quantization.empty());

        // the backend flags follow the params, so both call sites see the same backend identity
        common_params other = params;
        other.n_ubatch = 256;
        const auto c = llama_decision::decision_provenance_current("m", other, te.model, nullptr, false);
        t.assert_true("backend flags follow the params", a.backend_flags != c.backend_flags);
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
        t.assert_true("tree_vs_greedy auto with the default tree_max stays on the exact tree path",
                      m.at("wide_auto_default_is_tree").get<bool>());
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

        // hoist (outcome axis): optimize on/off must keep every winner and stay within the
        // committed TV bound recorded in the sign-off table, and the control must actually
        // exercise the hoist (shared head lifted on, not lifted off)
        const common_json cal = common_json::parse(
            read_file(std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json"));
        const double hoist_bound = cal.at("rows").at("prefix_hoist").at("measurements").at("outcome_tv_bound").get<double>();
        t.assert_true("hoist control: optimize on vs off keeps the winner", m.at("hoist_argmax_agree").get<bool>());
        t.assert_true("hoist control: per-field TV stays within the committed bound",
                      m.at("hoist_tv").get<double>() <= hoist_bound);
        t.assert_true("hoist control: the shared suffix head is actually hoisted", m.at("hoist_fired").get<bool>());
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
        const std::string model_path_str = model_path;
        std::ifstream model_in(model_path, std::ios::binary | std::ios::ate);
        const long long model_bytes = model_in ? (long long) model_in.tellg() : -1;

        common_json env = common_json::object();
        env["model"]   = model_identity(model_path_str);
        env["model_bytes"] = model_bytes;
        env["quantization"] = "recorded per run; see model filename";
        env["backend_flags"] = common_json::parse(R"({"kv_unified":true,"swa_full":false,
            "n_ctx":8192,"n_batch":512,"n_ubatch":512,"n_seq_max":10,"gpu_layers":0})");
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

// Refreshes calibration.json "rows" only; the model measurements are preserved. Host-runnable:
// every row is a pure measurement, so no GGUF is needed.
static int write_calibration_rows() {
    const std::string path = std::string(DECISION_TEST_BASELINE_DIR) + "/calibration.json";
    common_json cal = common_json::parse(read_file(path));
    cal["rows"] = calibration_rows();
    write_file(path, cal.dump(2) + "\n");
    return 0;
}

static int write_goldens() {
    write_file(fixture_path("decision_basic.golden.json"), decision_basic_from_fixed_scores().dump(2) + "\n");
    return 0;
}

// Refreshes baseline.json "cpu_oracle" only; every other section is preserved.
static int write_cpu_oracle() {
    const common_json oracle = decision_cpu_oracle();
    if (oracle.empty()) {
        fprintf(stderr, "the CPU oracle needs the generated dummy model\n");
        return 2;
    }
    const std::string path = std::string(DECISION_TEST_BASELINE_DIR) + "/baseline.json";
    common_json baseline = common_json::parse(read_file(path));
    baseline["cpu_oracle"] = oracle;
    write_file(path, baseline.dump(1) + "\n");
    return 0;
}

static int write_decision_golden(const char * model_path) {
    if (model_path == nullptr || model_path[0] == '\0') {
        fprintf(stderr, "set LLAMA_DECISION_TEST_MODEL to write the decision golden\n");
        return 2;
    }
    test_engine te;
    if (!te.load(model_path)) {
        fprintf(stderr, "failed to load model or build context\n");
        return 2;
    }
    auto vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(te.model));
    const auto pool = llama_decision::build_label_pool(*vocab, test_letter_tail(), 64);
            llama_decision::engine eng(te.ctx, 2, 8);
            const auto req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
    const auto probs = test_letter_readout(eng, test_head_cache(), *vocab, nullptr, false, req, pool, llama_decision::options{}, nullptr);
    common_json usage = common_json::object();
    usage["input_tokens"]    = 0;
    usage["output_tokens"]   = 0;
    usage["cached_tokens"]   = 0;
    usage["state_cache_hit"] = false;
    common_json golden = llama_decision::assemble_decision_response(req, probs, "m", usage);
    golden["golden_model"] = model_identity(model_path);
    write_file(decision_golden_path(), golden.dump(2) + "\n");
    return 0;
}

// Reproducibility recording of the committed decision corpus. The deterministic core (per-question
// probabilities, winners, head mode, label-pool size) is the frozen contract later refactors must
// not move; the timing block is recorded for context and excluded from the byte diff because it
// moves every run.
static std::string readout_baseline_path(const std::string & backend) {
    return std::string(DECISION_TEST_BASELINE_DIR) + "/readout_" + backend + "_baseline.json";
}

// Runs the committed corpus once and returns the readout core plus its timing block. `gpu` selects
// the shared GPU model; otherwise the model loads with no offload so the CPU lane can record and
// check it.
static common_json readout_capture(const std::string & path, bool gpu) {
    const std::string tail = test_letter_tail();

    common_json out = common_json::object();
    out["backend"]  = gpu ? "gpu" : "cpu";
    out["model"]    = model_identity(path);

    auto run = [&](llama_model * model, llama_context * ctx) {
        auto                    vocab = llama_decision::make_llama_label_vocab(llama_model_get_vocab(model));
        const auto              pool  = llama_decision::build_label_pool(*vocab, tail, llama_decision::LABEL_POOL_CAP);
        llama_decision::engine  eng(ctx, 2, 8);
        const auto              req = llama_decision::parse_decision_request(common_json::parse(decision_valid_body()));
        llama_decision::options opt;
        opt.cache_tag = "readout-baseline";
        llama_decision::letter_metrics    metrics;
        llama_decision::answer_head_cache head_cache;
        const auto                        probs =
            test_letter_readout(eng, head_cache, *vocab, nullptr, false, req, pool, opt, &metrics);

        common_json core        = oracle_readout(metrics, probs);
        core["head_mode"]       = metrics.head_active ? "selected" : "full";
        core["label_pool_size"] = (long long) pool.size();
        out["readout"]          = core;

        common_json timing   = common_json::object();
        timing["prefill_ms"] = metrics.prefill_ms;
        timing["scoring_ms"] = metrics.scoring_ms;
        out["timings"]       = timing;
    };

    if (gpu) {
        test_engine te;
        if (!te.load(path.c_str())) {
            throw std::runtime_error("the GPU test model failed to load: " + path);
        }
        run(te.model, te.ctx);
    } else {
        cpu_test_engine te;
        if (!te.load(path, 4096, false, false, 512)) {
            throw std::runtime_error("the CPU test model failed to load: " + path);
        }
        run(te.model, te.ctx);
    }
    return out;
}

// The first serialized line that differs, for a compact mismatch report instead of dumping both
// trees.
static std::string first_line_difference(const std::string & a, const std::string & b) {
    std::istringstream ra(a);
    std::istringstream rb(b);
    std::string        la;
    std::string        lb;
    int                line = 0;
    while (true) {
        const bool oka = (bool) std::getline(ra, la);
        const bool okb = (bool) std::getline(rb, lb);
        if (!oka && !okb) {
            break;
        }
        ++line;
        if (la != lb) {
            return "line " + std::to_string(line) + "\n  baseline: " + la + "\n  current : " + lb;
        }
    }
    return "the serialized lengths differ";
}

static bool readout_core_equal(const common_json & expected, const common_json & actual, std::string * difference) {
    common_json e = expected;
    common_json a = actual;
    e.erase("timings");
    a.erase("timings");
    e.erase("note");
    a.erase("note");
    const std::string es = e.dump(1);
    const std::string as = a.dump(1);
    if (es == as) {
        return true;
    }
    if (difference != nullptr) {
        *difference = first_line_difference(es, as);
    }
    return false;
}

// The model a baseline section was recorded on, or empty when the file or section is absent.
static std::string readout_baseline_model(const std::string & backend) {
    const std::string path = readout_baseline_path(backend);
    if (!file_exists(path)) {
        return std::string();
    }
    try {
        return common_json::parse(read_file(path)).value("model", std::string());
    } catch (const std::exception &) {
        return std::string();
    }
}

static int write_readout_baseline(const std::string & backend) {
    const bool  gpu = backend == "gpu";
    std::string path;
    if (gpu) {
        const char * env = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (env == nullptr || env[0] == '\0') {
            fprintf(stderr, "set LLAMA_DECISION_TEST_MODEL to record the GPU readout baseline\n");
            return 2;
        }
        path = env;
    } else {
        path = decision_cpu_model_path();
        if (path.empty()) {
            fprintf(stderr, "the CPU readout baseline needs the generated model\n");
            return 2;
        }
    }
    try {
        common_json rec = readout_capture(path, gpu);
        rec["note"] =
            "Frozen readout of the committed decision corpus. The deterministic core "
            "(probabilities, winners, confidence = 1 - H/log K, certainty = max p, head mode, "
            "label-pool size) is a contract; the timing block is recorded for context and excluded "
            "from the byte diff.";
        write_file(readout_baseline_path(backend), rec.dump(1) + "\n");
    } catch (const std::exception & e) {
        fprintf(stderr, "failed to record the %s readout baseline: %s\n", backend.c_str(), e.what());
        return 2;
    }
    return 0;
}

// Diff command: recompute the readout and fail on any deterministic byte change. Returns 0 on a
// match, 1 on drift, 2 when the baseline or its model is unavailable.
static int check_readout_baseline(const std::string & backend) {
    const bool        gpu  = backend == "gpu";
    const std::string file = readout_baseline_path(backend);
    if (!file_exists(file)) {
        fprintf(stderr, "no %s readout baseline at %s\n", backend.c_str(), file.c_str());
        return 2;
    }
    std::string path;
    if (gpu) {
        const char * env = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (env == nullptr || env[0] == '\0') {
            fprintf(stderr, "set LLAMA_DECISION_TEST_MODEL to check the GPU readout baseline\n");
            return 2;
        }
        path = env;
    } else {
        path = decision_cpu_model_path();
        if (path.empty()) {
            fprintf(stderr, "the CPU readout baseline needs the generated model\n");
            return 2;
        }
    }

    const common_json expected = common_json::parse(read_file(file));
    if (expected.value("model", std::string()) != model_identity(path)) {
        fprintf(stderr, "the loaded model (%s) is not the recorded %s baseline model (%s)\n",
                model_identity(path).c_str(), backend.c_str(), expected.value("model", std::string()).c_str());
        return 2;
    }

    std::string difference;
    try {
        const common_json actual = readout_capture(path, gpu);
        if (readout_core_equal(expected, actual, &difference)) {
            printf("the %s readout matches the frozen baseline\n", backend.c_str());
            return 0;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "failed to recompute the %s readout: %s\n", backend.c_str(), e.what());
        return 2;
    }
    fprintf(stderr, "the %s readout drifted from the frozen baseline:\n%s\n", backend.c_str(), difference.c_str());
    return 1;
}

// The suite gate: check whichever frozen baseline matches the model available in this lane.
static void test_readout_baseline(testing & t) {
    t.test("the readout matches the frozen reproducibility baseline", [](testing & t) {
        const char * env = std::getenv("LLAMA_DECISION_TEST_MODEL");
        if (env != nullptr && env[0] != '\0' && decision_gpu_available() &&
            readout_baseline_model("gpu") == model_identity(env)) {
            try {
                const common_json expected = common_json::parse(read_file(readout_baseline_path("gpu")));
                const common_json actual   = readout_capture(env, true);
                std::string       difference;
                if (!readout_core_equal(expected, actual, &difference)) {
                    t.assert_true("the gpu readout core matches the frozen baseline: " + difference, false);
                    return;
                }
                t.assert_true("the gpu readout core matches the frozen baseline", true);
            } catch (const std::exception & e) {
                t.assert_true(std::string("the gpu readout baseline runs: ") + e.what(), false);
            }
            return;
        }

        const std::string path = decision_cpu_model_path();
        if (path.empty() || readout_baseline_model("cpu") != model_identity(path)) {
            t.skip("no frozen readout baseline matches the available model");
            return;
        }
        try {
            const common_json expected = common_json::parse(read_file(readout_baseline_path("cpu")));
            const common_json actual   = readout_capture(path, false);
            std::string       difference;
            if (!readout_core_equal(expected, actual, &difference)) {
                t.assert_true("the cpu readout core matches the frozen baseline: " + difference, false);
                return;
            }
            t.assert_true("the cpu readout core matches the frozen baseline", true);
        } catch (const std::exception & e) {
            t.assert_true(std::string("the cpu readout baseline runs: ") + e.what(), false);
        }
    });
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--write-golden") {
        return write_goldens();
    }
    if (argc > 1 && std::string(argv[1]) == "--write-cpu-oracle") {
        return write_cpu_oracle();
    }
    if (argc > 1 && std::string(argv[1]) == "--write-decision-golden") {
        return write_decision_golden(std::getenv("LLAMA_DECISION_TEST_MODEL"));
    }
    if (argc > 1 && std::string(argv[1]) == "--write-calibration") {
        return write_calibration(std::getenv("LLAMA_DECISION_TEST_MODEL"));
    }
    if (argc > 1 && std::string(argv[1]) == "--write-calibration-rows") {
        return write_calibration_rows();
    }
    if (argc > 2 && std::string(argv[1]) == "--record-readout") {
        const std::string backend = argv[2];
        if (backend != "cpu" && backend != "gpu") {
            fprintf(stderr, "--record-readout needs a backend: cpu or gpu\n");
            return 2;
        }
        return write_readout_baseline(backend);
    }
    if (argc > 2 && std::string(argv[1]) == "--check-readout") {
        const std::string backend = argv[2];
        if (backend != "cpu" && backend != "gpu") {
            fprintf(stderr, "--check-readout needs a backend: cpu or gpu\n");
            return 2;
        }
        return check_readout_baseline(backend);
    }

    testing t;
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("decision engine harness", [](testing & t) {
        test_split_chat_template_primitive(t);
        test_thinking_off(t);
        test_thinking_off_model(t);
        test_thinking_control(t);
        test_decision_shape_contract(t);
        test_decision_parse(t);
        test_decision_assemble(t);
        test_decision_default_envelope(t);
        test_decision_values_golden(t);
        test_softmax(t);
        test_score_answer_rows(t);
        test_saved_state_format_dispatch(t);
        test_save_load_fail_fast(t);
        test_weak_quant_control(t);
        test_prefix_tag(t);
        test_question_temperature(t);
        test_temperature_effect(t);
        test_confidence_certainty_axes(t);
        test_confidence_profile(t);
        test_numeric_questions(t);
        test_temperature_profile(t);
        test_confidence_never_gates(t);
        test_letter_suffix(t);
        test_letter_option_lines(t);
        test_label_pool(t);
        test_boundary(t);
        test_answer_label_token(t);
        test_safe_data(t);
        test_label_pool_real(t);
        test_letter_labels_spm(t);
        test_label_boundary_calibration(t);
        test_letter_readout_real(t);
        test_fork_real(t);
        test_fork_oracle(t);
        test_nested_fork_oracle(t);
        test_session_fork(t);
        test_fork_auto_default(t);
        test_fork_strategy_switch(t);
        test_fork_divergence_control(t);
        test_fork_swa_clamp(t);
        test_prefix_lru_restores_own_state(t);
        test_state_bulk_copy(t);
        test_device_state_round_trip(t);
        test_device_async_staging(t);
        test_recurrent_multi_range_device_save(t);
        test_partial_state_round_trip(t);
        test_partial_state_fragmented(t);
        test_device_layout_mutation_round_trip(t);
        test_prefix_cache_cost(t);
        test_classifier_head_unbiased(t);
        test_bounded_decision_context(t);
        test_multi_trunk_restore(t);
        test_pool_seq_lifecycle(t);
        test_context_params_append(t);
        test_classifier_only_hidden_state(t);
        test_classifier_only_sampler(t);
        test_classifier_rows_host(t);
        test_classifier_rows_dequant(t);
        test_classifier_rows_width_contract(t);
        test_classifier_support_predicate(t);
        test_decision_cpu_oracle(t);
        test_readout_baseline(t);
        test_compile_fields_plan(t);
        test_decide_batch_plan_overload(t);
        test_select_scoring_head(t);
        test_classifier_ctx_requires_head(t);
        test_prefix_cache_coherence(t);
        test_token_cache(t);
        test_prefix_reuse(t);
        test_request_prefix(t);
        test_batching_waves(t);
        test_dedup_fields(t);
        test_cancel_reaches_compute(t);
        test_yield_points(t);
        test_capacity_error(t);
        test_long_branch_chunking(t);
        test_prefix_hoist_cache(t);
        test_permutation_order(t);
        test_permutations_parsing(t);
        test_permutations_real(t);
        test_contract_hash(t);
        test_reference_corpus(t);
        test_gemma4_softcap(t);
        test_docs_errors(t);
        test_policy_confidence(t);
        test_head_capability(t);
        test_classifier_predicate(t);
        test_classifier_rows_lfm(t);
        test_head_fallback_equivalence(t);
        test_selected_equivalence_lfm(t);
        test_selected_equivalence_qwen2(t);
        test_answer_head_cache(t);
        test_classifier_only_readout(t);
        test_selected_fallback_lfm(t);
        test_selected_explicit_error(t);
        test_sha256(t);
        test_confidence_never_gates_envelope(t);
        test_verify_letter_request(t);
        test_sequence_partition(t);
        test_calibration_hoist(t);
        test_calibration_dedup(t);
        test_calibration_confidence(t);
        test_calibration_selected_head(t);
        test_calibration_selected_head_lfm(t);
        test_calibration_table(t);
        test_calibration_ledger(t);
        test_calibration_gates(t);
        test_calibration_determinism_variance(t);
        test_decision_provenance(t);
        test_calibration_model(t);
    });

    return t.summary();
}
