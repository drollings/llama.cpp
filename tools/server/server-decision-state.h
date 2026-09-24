#pragma once

// Owns every piece of /v1/decision state that is tied to one loaded model. The server context
// holds exactly one of these and calls reset() whenever the model is freed, so a sleep->wake
// reload cannot leave a stale vocab, contract, or answer-row cache pointing at dead memory.

#include "letter_readout.h"

#include "common.h"
#include "llama.h"

#include <memory>
#include <string>
#include <vector>

struct server_decision_state {
    // One live adapter-scope predicate: the decision decode answers for the base model, and
    // the registry the server applies to chat batches is the single source of truth. Entries
    // with a zero scale are registered but disabled, matching the enabled-adapter convention
    // used by the slot scheduler.
    bool adapters_configured(const std::vector<common_adapter_lora_info> & loras) const {
        for (const auto & lora : loras) {
            if (lora.scale > 0.0f) {
                return true;
            }
        }
        return false;
    }

    std::unique_ptr<llama_decision::engine> decision_engine; // trie readout, created on first use
    // Separate engine per letter-readout context: one on the shared full-logits context (seq ids
    // above the chat slots) and one on the classifier-only context (its own cache, seq ids from
    // zero). The readout picks between them from the compiled plan, so both stay alive together.
    std::unique_ptr<llama_decision::engine> decision_letter_engine;
    std::unique_ptr<llama_decision::engine> decision_letter_engine_classifier;
    std::unique_ptr<llama_decision::label_vocab> decision_label_vocab; // letter readout, built once per model
    std::vector<llama_decision::label>      decision_labels;
    llama_decision::answer_head_cache       decision_head_cache; // owns the answer-row tables for this model
    std::string                             decision_label_error; // set when the vocabulary probe fails
    // Classifier-only context for the letter readout: shares the model weights, produces hidden
    // states instead of logits, and is created on first use. Null means the letter path keeps
    // reading full logits from the shared context.
    llama_context * ctx_decision = nullptr;
    std::string     ctx_decision_error;
    std::string     decision_contract;    // identity of the decision readout contract
    bool            decision_temp_loaded = false;
    llama_decision::temperature_profile   decision_temp_profile;

    // Total reset: every member that is a function of the loaded model is dropped, so a fresh
    // model after a reload rebuilds the vocab, labels, contract, head cache and classifier
    // context from scratch. Any member added here is covered by construction.
    void reset() {
        if (ctx_decision) {
            llama_free(ctx_decision);
            ctx_decision = nullptr;
        }
        decision_engine.reset();
        decision_letter_engine.reset();
        decision_letter_engine_classifier.reset();
        decision_label_vocab.reset();
        decision_labels.clear();
        decision_head_cache.clear();
        decision_label_error.clear();
        ctx_decision_error.clear();
        decision_contract.clear();
        decision_temp_loaded = false;
        decision_temp_profile = {};
    }
};
