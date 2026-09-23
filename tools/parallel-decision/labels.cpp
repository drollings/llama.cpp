#include "labels.h"

#include "llama.h"

#include <cstdlib>
#include <stdexcept>

namespace llama_decision {

namespace {

class llama_label_vocab : public label_vocab {
public:
    explicit llama_label_vocab(const llama_vocab * vocab) : vocab(vocab) {}

    std::vector<int32_t> tokenize(const std::string & text, bool parse_special) const override {
        const int n = llama_tokenize(vocab, text.data(), (int) text.size(), nullptr, 0, false, parse_special);
        if (n == 0) {
            return {};
        }
        std::vector<llama_token> tokens((size_t) std::abs(n));
        const int m = llama_tokenize(vocab, text.data(), (int) text.size(), tokens.data(), (int) tokens.size(), false, parse_special);
        if (m < 0) {
            throw std::runtime_error("tokenization failed");
        }
        tokens.resize((size_t) m);
        return std::vector<int32_t>(tokens.begin(), tokens.end());
    }

    std::string piece(int32_t token) const override {
        char buf[128];
        const int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
        if (n < 0) {
            return {};
        }
        return std::string(buf, (size_t) n);
    }

    bool is_special(int32_t token) const override {
        return llama_vocab_is_control(vocab, token) || llama_vocab_is_eog(vocab, token);
    }

private:
    const llama_vocab * vocab;
};

} // namespace

std::unique_ptr<label_vocab> make_llama_label_vocab(const llama_vocab * vocab) {
    return std::make_unique<llama_label_vocab>(vocab);
}

int32_t answer_label_token(const label_vocab & vocab, const std::string & tail, const std::string & text) {
    const std::vector<int32_t> with_text = vocab.tokenize(tail + text, true);
    const std::vector<int32_t> tail_only = vocab.tokenize(tail, true);
    if (with_text.size() != tail_only.size() + 1) {
        return -1;
    }
    for (size_t i = 0; i < tail_only.size(); ++i) {
        if (with_text[i] != tail_only[i]) {
            return -1;
        }
    }
    const int32_t token = with_text.back();
    if (vocab.is_special(token)) {
        return -1;
    }
    return token;
}

std::vector<label> build_label_pool(const label_vocab & vocab, const std::string & tail, size_t cap) {
    std::vector<std::string> candidates;
    for (char a = 'A'; a <= 'Z'; ++a) {
        candidates.push_back(std::string(1, a));
    }
    for (char a = 'A'; a <= 'Z'; ++a) {
        for (char b = 'A'; b <= 'Z'; ++b) {
            candidates.push_back(std::string{ a, b });
        }
    }

    std::vector<label> pool;
    for (const std::string & text : candidates) {
        const int32_t token = answer_label_token(vocab, tail, text);
        if (token < 0) {
            continue;
        }
        bool seen = false;
        for (const label & l : pool) {
            if (l.token == token) {
                seen = true;
                break;
            }
        }
        if (seen) {
            continue;
        }
        pool.push_back({ text, token });
        if (pool.size() == cap) {
            break;
        }
    }

    if (pool.size() < 2) {
        throw std::runtime_error("no suitable answer tokens in the vocabulary");
    }
    return pool;
}

bool check_boundary(const label_vocab & vocab,
                    const std::string & prompt,
                    const std::string & label,
                    int32_t label_token) {
    return answer_label_token(vocab, prompt, label) == label_token;
}

std::string safe_data(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (char c : text) {
        if (c == '<') {
            out += "\\u003c";
        } else {
            out += c;
        }
    }
    return out;
}

} // namespace llama_decision
