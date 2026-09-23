#pragma once

// Letter readout: score each Jev question as one next-token choice over the
// verified label pool, sharing one framed state prefix across all questions.
// Built on the same engine and the same branch scorer as the trie path.

#include "decision-engine.h"
#include "decision-protocol.h"
#include "labels.h"

#include <string>
#include <utility>
#include <vector>

struct common_params;

namespace llama_decision {

// Bump when the letter prompt layout changes; it is part of the prefix cache identity.
inline constexpr const char * LETTER_PROMPT_VERSION = "letter-v1";

// Identity of the decision readout contract: the tokenizer identity, the framed prompt template,
// and the label code. A template edit or a version bump changes it, so a mismatched expected hash
// means the calibration is stale and the decision path must refuse it.
std::string decision_contract_hash(const std::string & model_name, const std::string & template_hash, int vocab_size);

// Quantization label of the loaded model: the general.quantization_version / file_type / type
// metadata, falling back to the model file name. Shared by the provenance helper and the bench.
std::string decision_quantization(const llama_model * model, const std::string & fallback_path);

// Provenance of the decision readout running now: model identity, quantization, prompt template
// hash, and backend flags. Derived from the live params and model so the temperature validation
// and the response diagnostics cannot drift apart.
temperature_provenance decision_provenance_current(const std::string & model_name,
                                                   const common_params & params,
                                                   const llama_model * model,
                                                   const common_chat_templates * tmpls, bool use_jinja);

// The fixed system instruction used by the letter readout.
const char * letter_system_text();

// The assistant-answer tail a label follows: `after` plus the fixed "Answer:\n" marker. One
// definition, so the framer, the per-request gate and the server cannot drift apart.
std::string letter_answer_tail(const std::string & after);

struct letter_metrics {
    bool   cache_hit      = false;
    size_t shared_tokens  = 0;
    size_t context_tokens = 0;
    int    rows           = 0;
    int    rounds         = 0;
    double prefill_ms     = 0;
    double scoring_ms     = 0;
    bool        head_active = false; // candidates were scored against the answer rows
    std::string head_reason;         // why a requested head was not used, empty otherwise
    size_t suffix_tokens        = 0; // unique question suffixes after dedup
    size_t common_suffix_tokens = 0; // suffix head hoisted onto the shared trunk
    size_t leaf_suffix_tokens   = 0; // what each branch actually decodes
};

// Splits the rendered chat prompt at the user message: `first` is the cacheable
// system/user prefix, `second` is the text after the user content (user turn end
// plus assistant turn start, without any answer instruction). The decision readout
// is thinking-off; `enable_thinking` stays false unless a caller explicitly opts in.
std::pair<std::string, std::string> render_letter_prompt(const common_chat_templates * tmpls, bool use_jinja,
                                                         const std::string & system_text,
                                                         bool enable_thinking = false);

// The per-question branch text, ending just before the label is generated.
std::string format_letter_suffix(const jev_question & q, const std::vector<label> & labels,
                                 const std::string & after);

// Throws semantic_error when a question needs more labels than the pool provides.
void validate_label_capacity(const jev_request & req, size_t label_count);

// Deterministic option order for one de-bias pass: pass 0 is the identity, later passes are
// distinct seeded shuffles of the option indices. Seeded by the question id, so a question always
// sees the same orders across runs and languages.
std::vector<size_t> permutation_order(size_t count, const std::string & question_id, int pass);

// Selected-head capability, resolved once per process. A populated `reason` means the projection
// of only the answer rows is not usable, so the readout must fall back to full logits (never a
// hard error on the default path); an explicit `head: selected` request is the only caller that
// turns this into a plain 400.
struct head_capability {
    bool        available = false;
    int         width     = 0;
    float       softcap   = 0.0f;
    std::string reason;
};

// Dequantizes the answer rows for `labels` into an FP32 table the scorer can dot against the
// post-norm hidden state. Never throws: an unusable table comes back with width 0 and a reason.
classifier_head build_classifier_head(const llama_model * model, const std::vector<label> & labels);

// Throws std::invalid_argument (HTTP 400) when `requested` is "selected" and the model cannot
// serve it. The default paths ("auto"/"full") never throw here and always fall back.
void require_selected_head(const std::string & requested, const head_capability & cap);

// Caller-owned cache for the model-derived answer-head tables. The tables are pure functions of
// (model, label tokens), so one cache per loaded model spares each request the GPU-to-host
// dequantization. Non-copyable and not thread-safe by contract: the owner serializes calls, and
// the owner must outlive the model it caches.
class answer_head_cache {
public:
    answer_head_cache() = default;
    answer_head_cache(const answer_head_cache &) = delete;
    answer_head_cache & operator=(const answer_head_cache &) = delete;

    // Capability probe for a model. Always returns a value: an unusable output table comes back
    // with `available == false` and a reason. Never throws.
    const head_capability & probe(const llama_model * model);

    // Answer-row table for `labels`, built once per (model, label tokens). Never throws.
    const classifier_head & for_labels(const llama_model * model, const std::vector<label> & labels);

private:
    bool                     cap_set_   = false;
    const llama_model *      cap_model_ = nullptr;
    head_capability          capability_;
    bool                     head_set_   = false;
    const llama_model *      head_model_ = nullptr;
    std::vector<llama_token> head_ids_;
    classifier_head          head_;
};

// Startup vocabulary probe: every pooled label must be the single non-special token the answer
// tail produces, so the scored slot sits on a clean prompt boundary. Throws std::runtime_error
// with a clear reason when it does not, so the caller can refuse the letter path instead of
// scoring a merged token.
void verify_label_pool(const label_vocab & vocab, const std::vector<label> & labels, const std::string & tail);

// Per-request tokenizer gate: the same boundary check applied to the labels the request actually
// uses. A mismatch throws semantic_error naming the question (HTTP 422), never a silent score.
void verify_letter_request(const label_vocab & vocab, const std::string & tail,
                           const jev_request & req, const std::vector<label> & labels);

// One probability vector per question, index-aligned with q.options. Throws
// semantic_error when a question needs more labels than the pool provides.
// When `audit` is non-null the per-question diagnostics are filled in.
std::vector<std::vector<float>> letter_readout(engine & eng,
                                               answer_head_cache & head_cache,
                                               const label_vocab & vocab,
                                               const common_chat_templates * tmpls, bool use_jinja,
                                               const jev_request & req,
                                               const std::vector<label> & labels,
                                               const options & opt,
                                               letter_metrics * metrics = nullptr,
                                               answer_audit * audit = nullptr);

} // namespace llama_decision
