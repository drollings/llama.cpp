#pragma once

// Parallel constrained decisions for finite JSON schemas, shared by llama-parallel-decision
// and llama-server's /decision endpoint.
//
// Every field of a schema has a finite set of allowed values. After a shared context, each
// field's value is scored as token paths following that field's own suffix; every scored path
// runs as its own sequence forked from the context (llama_memory_seq_cp), so all fields are
// evaluated in one batched llama_decode and cannot see each other. Small fields score every
// divergence node of their token trie at once and return the exact constrained distribution;
// larger fields walk the trie greedily.

#include "llama.h"
#include "json.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct common_chat_templates;

namespace llama_decision {

using tokens_t = std::vector<llama_token>;

// The request needs more rows or sequences than the batch or pool can hold. The server maps
// this to 422: the request is well formed but too large for the configured budget.
struct capacity_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The caller asked to stop before the evaluation finished. The server maps this to 499.
struct cancelled_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// One field as the scorer sees it: the text before its value and the allowed value texts.
struct field_input {
    std::string              suffix;              // e.g.  '  "fire": '
    std::vector<std::string> candidates;          // allowed values, with the suffix's shared prefix removed
    float                    temperature = 1.0f;  // softmax temperature for this field's score
};

struct options {
    std::string mode           = "auto"; // auto: tree up to tree_max values, else greedy; tree; greedy
    size_t      tree_max       = 128;
    bool        split_boundary = false;  // legacy: tokenise suffix and values separately
    bool        allow_cache    = true;   // reuse the cached static prefix when it matches
    std::string cache_tag;               // optional: cache is only reused when the tag also matches
    std::string fork           = "auto"; // auto | copy | restore: how branches fork the prefix
    bool        bypass         = true;   // skip the fork when a round has exactly one branch
    bool        audit          = false;  // also collect full-vocab diagnostics at the scored position
    std::function<bool()> should_stop;   // optional: checked before every decode and between waves
    std::function<void()> yield;         // optional: cooperative yield point between waves
};

// Numerically stable softmax with an optional temperature; T=1 matches the trie's
// exact normalization so the legacy path stays bit-identical.
std::vector<float> softmax(const std::vector<float> & logits, float temperature = 1.0f);

// Deterministic identity of a rendered static prefix: prompt version + chat template
// shape + the prefix text. Used to reject a cache hit produced under a different prompt.
std::string make_prefix_tag(const std::string & system_text, const std::string & after,
                            const std::string & prompt_version);

struct field_result {
    int                winner       = -1;
    float              path_score   = 1.0f;
    int                scored_nodes = 0;
    bool               tree         = false;
    std::vector<float> probs;            // tree fields: probability of every allowed value
    float              allowed_token_mass   = 1.0f; // share of full-vocab mass on the allowed tokens (audit)
    int                full_vocab_argmax_id = -1;   // argmax over the full vocabulary (audit)
    std::vector<float> logits;                      // raw logits of the allowed tokens at the scored node (audit)
};

struct result {
    std::vector<field_result> fields;
    bool   cache_hit      = false;
    size_t shared_tokens  = 0;
    size_t context_tokens = 0;
    int    rows           = 0;
    int    rounds         = 0;
    double prefill_ms     = 0;
    double scoring_ms     = 0;
};

// Several contexts decided against one schema and one cached prefix. Items carry fields,
// context_tokens and rows; timings and cache state cover the whole batch.
struct batch_result {
    std::vector<result> items;
    bool   cache_hit     = false;
    size_t shared_tokens = 0;
    int    rows          = 0;
    int    rounds        = 0;
    double prefill_ms    = 0;
    double scoring_ms    = 0;
};

// Scores decisions on an existing context with the sequence ids [seq_base, seq_base + n_seqs):
// one keeps the cached static prefix; the rest hold one trunk (prefix + context) per context in
// flight, then branches. The context needs a unified KV cache so branches share the trunk's cells.
class engine {
  public:
    engine(llama_context * ctx, llama_seq_id seq_base, int n_seqs);

    result decide(const std::string & shared_text, const std::string & context_text,
                  const std::vector<field_input> & fields, const options & opt);

    // Contexts are prefilled together and their branches scored together, in groups sized to fit
    // the sequence budget; results keep the order of the contexts.
    batch_result decide_batch(const std::string & shared_text, const std::vector<std::string> & contexts,
                              const std::vector<field_input> & fields, const options & opt);

  private:
    struct prompt_part {
        const tokens_t * toks;
        llama_pos        pos0;
        llama_seq_id     seq;
    };
    struct branch {
        llama_seq_id trunk;
        llama_pos    pos0;
        tokens_t     toks;
        tokens_t     cands;
    };
    struct branch_score {
        std::vector<float> cand_logits;
        int                full_vocab_argmax   = -1;
        float              allowed_token_mass  = 1.0f;
    };

    enum class fork_kind { copy, restore };

    llama_context     * ctx;
    const llama_model * model;
    const llama_vocab * vocab;
    llama_memory_t      mem;
    llama_seq_id        seq_snap, seq_pool;
    int                 n_pool;
    tokens_t            cached;
    std::string         cached_tag;

    std::function<bool()> stop_;
    std::function<void()> yield_;
    bool                audit_ = false;
    fork_kind           probe_fork_;
    fork_kind           active_fork_ = fork_kind::copy;
    llama_pos           swa_         = 0; // sliding-window size, 0 = none
    std::vector<uint8_t> prefix_state_;

    struct prefix_entry {
        std::string          tag;
        std::vector<uint8_t> state;
    };
    std::vector<prefix_entry> prefix_lru_; // restore mode, most-recent first
    size_t                    prefix_lru_capacity_ = 4;

    tokens_t tokenize(const std::string & text, bool add_special) const;
    void     check_cancel() const;
    void     decode_parts(const std::vector<prompt_part> & parts);
    bool     prepare_prefix(const tokens_t & shared, bool allow_cache, const std::string & tag);

    std::vector<uint8_t> save_seq(llama_seq_id seq) const;
    void                 load_seq(const std::vector<uint8_t> & state, llama_seq_id seq) const;
    void                 fork_into(llama_seq_id src, llama_seq_id dst, const std::vector<uint8_t> * src_state);
    void                 select_fork(const std::string & requested);

    std::vector<branch_score> score_branches(const std::vector<branch> & branches, llama_seq_id first, int n_free,
                                             const std::vector<std::vector<uint8_t>> * parent_states,
                                             bool allow_bypass);
};

// ---- schema compiler (the C++ counterpart of llama-mojo's tools/prepare_decisions.py)

struct field_spec {
    std::string              name;
    std::string              type;        // boolean | enum | integer | number
    std::string              description;
    std::string              aggregate;   // mode | median | mean (median/mean: numeric fields)
    std::vector<common_json> values;      // typed values; index = candidate index
    std::vector<double>      numbers;     // numeric fields: the same values as doubles
    std::vector<std::string> encoded;     // JSON text of each value
};

struct compiled_schema {
    std::string              system_text; // fixed instructions + field catalogue (cacheable)
    std::vector<field_spec>  specs;
    std::vector<field_input> inputs;
};

// Accepts compact field specs {"name": {"type": ..., "description": ..., ...}} or a JSON Schema
// object with "properties" (boolean, string+enum, integer min/max, number min/max/multipleOf).
compiled_schema compile_schema(const common_json & schema, const std::string & instructions);

// Renders system + user messages with the model's chat template (thinking disabled) and splits
// the prompt into the static prefix (cached across requests) and the per-request part: the
// context, the generation prompt and the opening brace of the JSON answer.
std::pair<std::string, std::string> render_prompt(const common_chat_templates * tmpls, bool use_jinja,
                                                  const std::string & system_text, const std::string & context);

// {"decision": {...}, "fields": {...}} from the scores, applying each numeric field's aggregate.
common_json assemble(const compiled_schema & cs, const result & r);

} // namespace llama_decision
