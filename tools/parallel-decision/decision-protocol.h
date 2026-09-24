#pragma once

// Decision request/response bridge: the wire shape on one side, the internal
// per-question option lists on the other. Pure JSON logic, no llama calls, so it
// can be unit tested without a model.

#include "json.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct common_chat_templates;

namespace llama_decision {

// Request shape limits. A request over a limit is rejected before any decode; it is never
// truncated. The choice option cap matches the answer-label pool cap, so every option gets a label.
inline constexpr size_t DECISION_MIN_QUESTIONS        = 1;
inline constexpr size_t DECISION_MAX_QUESTIONS        = 256;
inline constexpr size_t DECISION_MAX_CONTEXTS         = 256;
inline constexpr size_t DECISION_MIN_OPTIONS          = 2;
inline constexpr size_t DECISION_MAX_CHOICE_OPTIONS   = 64;
inline constexpr size_t DECISION_MAX_SCORE_LEVELS     = 10;
inline constexpr int    DECISION_MAX_PERMUTATIONS     = 8;

// Valid JSON but invalid decision content (bad type, limits, missing fields).
// The server maps this to HTTP 422; malformed JSON stays a parse error (400).
struct semantic_error : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

// FNV-1a 64 over the bytes of `s` (offset basis 1469598103934665603, prime 1099511628211).
// The single hash primitive behind the decision prefix tag and the permutation seed.
uint64_t fnv1a64(const std::string & s);

// Renders the chat template with a sentinel user message and returns the text before and after
// the sentinel. `render_prompt` and `render_letter_prompt` share it and differ only in how they
// use the split; `tmpls` must be non-null.
std::pair<std::string, std::string> split_chat_template(const common_chat_templates * tmpls,
                                                        bool use_jinja,
                                                        const std::string & system_text,
                                                        bool enable_thinking);

// The running model cannot serve the decision path at all (for example its vocabulary has no
// usable single-token answer labels). The server maps this to HTTP 501, never a client error.
struct unsupported_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// One allowed answer of a question.
struct decision_option {
    std::string key;         // choice key, level index string, or "true"/"false"
    std::string description; // text shown to the model
    common_json original;    // original criterion value, echoed by `legend`
};

struct decision_question {
    std::string              id;
    std::string              type;         // canonical: noul | choice | score
    common_json              instructions; // string/object/array, may be null
    std::vector<decision_option>  options;
    bool                     has_criteria = false;
};

struct decision_request {
    std::string               model;
    common_json               state;
    std::vector<decision_question> questions;
    double                    temperature = 1.0;
    common_json               temperatures; // object or null
    int                       permutations = 1;
    std::string               head;         // "" (auto) | "selected" | "full"
    bool                      diagnostics = false; // emit additive and audit fields
};

// Renders a string/object/array instruction or criterion value to prompt text.
std::string render_text(const common_json & value);

// Effective softmax temperature for a question: per-type override, else the global value.
double question_temperature(const decision_request & req, const decision_question & q);

// True when the body carries the decision shape (state and/or questions).
bool is_decision_request(const common_json & body);

// Throws semantic_error on any invalid decision content.
decision_request parse_decision_request(const common_json & body);

// Uniform distributions, one per question, sized to its option count.
std::vector<std::vector<float>> uniform_probs(const decision_request & req);

// Identity a calibrated temperature profile was fitted against. It is never used
// to gate answers; it only stops a profile fitted on one deployment from silently
// applying to another.
struct temperature_provenance {
    std::string model;
    std::string quantization;
    std::string template_hash;
    std::string backend_flags;

    bool operator==(const temperature_provenance & other) const;
    bool operator!=(const temperature_provenance & other) const { return !(*this == other); }
};

struct temperature_profile {
    std::map<std::string, double> temperatures; // per type: noul | choice | score
    temperature_provenance        provenance;
};

// Parses {"temperatures": {...}, "provenance": {...}}. Throws semantic_error on bad shape.
temperature_profile parse_temperature_profile(const common_json & doc);

// Throws semantic_error when any non-default temperature would run under a provenance
// that does not match the running configuration. T=1.0 is always allowed.
void validate_temperature_profile(const temperature_profile & profile, const temperature_provenance & current);

// Per-answer audit trail. Additive only: it is reported for inspection and never
// gates or changes an answer.
struct answer_audit {
    std::string                       prompt_sha256;
    std::string                       prompt_version;
    std::string                       probability_status;
    bool                              full_vocab_audit = true; // false when only answer rows are measurable
    std::vector<std::vector<int32_t>> answer_token_ids;      // per question, index-aligned with options
    std::vector<float>                allowed_token_mass;    // per question
    std::vector<int32_t>              full_vocab_argmax_id;  // per question
    std::vector<std::vector<float>>   option_logits;         // per question, index-aligned with options
};

// Lowercase hex SHA-256 of the given bytes. Used for the prompt identity in the audit trail.
std::string sha256_hex(const std::string & text);

// Canonical decision response. Additive fields (certainty, audit, extra usage counters, and the
// optional `head`/`diagnostics` payload) are emitted only when `req.diagnostics` is set; `probs` is
// index-aligned with req.questions and their options; a missing or empty entry falls back to a
// uniform distribution. This assembler is the single owner of the default-vs-diagnostics envelope.
common_json assemble_decision_response(const decision_request & req,
                                  const std::vector<std::vector<float>> & probs,
                                  const std::string & model,
                                  const common_json & usage,
                                  const answer_audit * audit = nullptr,
                                  const common_json * diagnostics = nullptr);

} // namespace llama_decision
