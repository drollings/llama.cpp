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

namespace llama_decision {

// Bump when the letter prompt layout changes; it is part of the prefix cache identity.
inline constexpr const char * LETTER_PROMPT_VERSION = "letter-v1";

// Identity of the decision readout contract: the tokenizer identity, the framed prompt template,
// and the label code. A template edit or a version bump changes it, so a mismatched expected hash
// means the calibration is stale and the decision path must refuse it.
std::string decision_contract_hash(const std::string & model_name, const std::string & template_hash, int vocab_size);

// The fixed system instruction used by the letter readout.
const char * letter_system_text();

struct letter_metrics {
    bool   cache_hit      = false;
    size_t shared_tokens  = 0;
    size_t context_tokens = 0;
    int    rows           = 0;
    int    rounds         = 0;
    double prefill_ms     = 0;
    double scoring_ms     = 0;
};

// Splits the rendered chat prompt at the user message: `first` is the cacheable
// system/user prefix, `second` is the text after the user content (user turn end
// plus assistant turn start, without any answer instruction).
std::pair<std::string, std::string> render_letter_prompt(const common_chat_templates * tmpls, bool use_jinja,
                                                         const std::string & system_text);

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
    std::string reason;
};

const head_capability & selected_head_capability();

// Startup vocabulary probe: every pooled label must be one shared token that round-trips and sits
// on a clean prompt boundary. Throws std::runtime_error with a clear reason when it does not, so
// the caller can refuse the letter path instead of scoring a merged token.
void verify_label_pool(const label_vocab & vocab, const std::vector<label> & labels, const std::string & tail);

// Per-request tokenizer gate: the same boundary check applied to the labels the request actually
// uses. A mismatch throws semantic_error naming the question (HTTP 422), never a silent score.
void verify_letter_request(const label_vocab & vocab, const std::string & tail,
                           const jev_request & req, const std::vector<label> & labels);

// One probability vector per question, index-aligned with q.options. Throws
// semantic_error when a question needs more labels than the pool provides.
// When `audit` is non-null the per-question diagnostics are filled in.
std::vector<std::vector<float>> letter_readout(engine & eng,
                                               const label_vocab & vocab,
                                               const common_chat_templates * tmpls, bool use_jinja,
                                               const jev_request & req,
                                               const std::vector<label> & labels,
                                               const options & opt,
                                               letter_metrics * metrics = nullptr,
                                               answer_audit * audit = nullptr);

} // namespace llama_decision
