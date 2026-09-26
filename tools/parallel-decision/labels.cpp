#include "labels.h"

#include "llama.h"

#include <algorithm>
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
    const std::vector<int32_t> path = answer_label_path(vocab, tail, text, 1);
    return path.size() == 1 ? path[0] : -1;
}

std::vector<int32_t> answer_label_path(const label_vocab & vocab, const std::string & tail,
                                       const std::string & text, int max_len) {
    const std::vector<int32_t> with_text = vocab.tokenize(tail + text, true);
    const std::vector<int32_t> tail_only = vocab.tokenize(tail, true);
    if (with_text.size() <= tail_only.size()) {
        return {};
    }
    const size_t extra = with_text.size() - tail_only.size();
    if ((int) extra > max_len) {
        return {};
    }
    for (size_t i = 0; i < tail_only.size(); ++i) {
        if (with_text[i] != tail_only[i]) {
            return {};
        }
    }
    std::vector<int32_t> path(with_text.begin() + (ptrdiff_t) tail_only.size(), with_text.end());
    for (const int32_t token : path) {
        if (vocab.is_special(token)) {
            return {};
        }
    }
    return path;
}

std::vector<label> build_label_pool(const label_vocab & vocab, const std::string & tail, size_t cap) {
    std::vector<std::string> candidates;
    auto one = [&](char c) { candidates.push_back(std::string(1, c)); };
    for (char a = 'A'; a <= 'Z'; ++a) {
        one(a);
    }
    for (char d = '0'; d <= '9'; ++d) {
        one(d);
    }
    auto two = [&](char a, char b) { candidates.push_back(std::string{ a, b }); };
    for (char a = 'A'; a <= 'Z'; ++a) {
        for (char b = 'A'; b <= 'Z'; ++b) {
            two(a, b); // AA..ZZ
        }
    }
    for (char a = 'A'; a <= 'Z'; ++a) {
        for (char d = '0'; d <= '9'; ++d) {
            two(a, d); // A0..Z9
        }
    }
    for (char d = '0'; d <= '9'; ++d) {
        for (char a = 'A'; a <= 'Z'; ++a) {
            two(d, a); // 0A..9Z
        }
    }
    for (char d1 = '0'; d1 <= '9'; ++d1) {
        for (char d2 = '0'; d2 <= '9'; ++d2) {
            two(d1, d2); // 00..99
        }
    }

    std::vector<label> pool;
    auto add = [&](const std::string & text) {
        const std::vector<int32_t> path = answer_label_path(vocab, tail, text, 2);
        if (path.empty()) {
            return;
        }
        for (const label & l : pool) {
            if (l.tokens == path) {
                return;
            }
        }
        label l;
        l.text   = text;
        l.tokens = path;
        l.token  = path.size() == 1 ? path[0] : -1;
        pool.push_back(std::move(l));
    };
    // single-token labels first, then two-token ones: the answer head can only score length-1
    // paths, so ordering the pool this way keeps the cheap head path available for as many
    // options as the tokenizer's single-token coverage allows
    for (const std::string & text : candidates) {
        add(text);
    }
    std::stable_partition(pool.begin(), pool.end(), [](const label & l) { return l.tokens.size() == 1; });
    if (pool.size() > cap) {
        pool.resize(cap);
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
