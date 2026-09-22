#pragma once

// Answer-label handling for scored decision readouts.
//
// A label is one alphabetic answer ("A".."Z", "AA".."ZZ") that must map to a
// single token in the model vocabulary and survive a text round-trip. The
// tokenizer is abstracted so the pool and boundary logic can be unit tested
// without a model.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_vocab;

namespace llama_decision {

struct label_vocab {
    virtual ~label_vocab() = default;
    virtual std::vector<int32_t> tokenize(const std::string & text, bool parse_special) const = 0;
    virtual std::string piece(int32_t token) const = 0;
    virtual bool is_special(int32_t token) const = 0;
};

struct label {
    std::string text;
    int32_t     token = -1;
};

// Adapter over a llama.cpp vocabulary. Owns nothing; the vocab must outlive it.
std::unique_ptr<label_vocab> make_llama_label_vocab(const llama_vocab * vocab);

// The token for `text` when it is a usable single label, else -1.
int32_t single_token(const label_vocab & vocab, const std::string & text);

// A-Z then AA-ZZ, kept only when single-token, round-tripping, unique and
// non-special; at most `cap` labels. Throws std::runtime_error if fewer than 2.
std::vector<label> build_label_pool(const label_vocab & vocab, size_t cap = 64);

// True when encode(prompt + label) equals encode(prompt) followed by exactly
// `label_token`. A false result means the scored slot would sit on a different
// boundary than the caller assumes.
bool check_boundary(const label_vocab & vocab,
                    const std::string & prompt,
                    const std::string & label,
                    int32_t label_token);

// Escapes "<" so state or option text cannot inject chat-template special
// tokens (for example "<|turn>" or "<__media__>").
std::string safe_data(const std::string & text);

} // namespace llama_decision
