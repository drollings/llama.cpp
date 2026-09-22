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

int32_t single_token(const label_vocab & vocab, const std::string & text) {
    const std::vector<int32_t> ids = vocab.tokenize(text, false);
    if (ids.size() != 1) {
        return -1;
    }
    const int32_t id = ids[0];
    if (vocab.is_special(id) || vocab.piece(id) != text) {
        return -1;
    }
    return id;
}

std::vector<label> build_label_pool(const label_vocab & vocab, size_t cap) {
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
        const int32_t token = single_token(vocab, text);
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
    const std::vector<int32_t> with_label    = vocab.tokenize(prompt + label, true);
    const std::vector<int32_t> without_label = vocab.tokenize(prompt, true);
    if (with_label.size() != without_label.size() + 1) {
        return false;
    }
    for (size_t i = 0; i < without_label.size(); ++i) {
        if (with_label[i] != without_label[i]) {
            return false;
        }
    }
    return with_label.back() == label_token;
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
