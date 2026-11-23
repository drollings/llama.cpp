#pragma once

#include "server-common.h"
#include "server-task.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

// typed decision models (TypeSafe /v1/systemone API): the model answers each question in one forward pass, no token is generated

// one score per column of the model output, so a question of one type cannot be read in another's
enum server_decision_question_type {
    SERVER_DECISION_QUESTION_CHOICE,
    SERVER_DECISION_QUESTION_SCORE,
    SERVER_DECISION_QUESTION_NOUL,
};

struct server_decision_option {
    std::string key;
    json description; // null if not provided
};

struct server_decision_question {
    std::string id;
    server_decision_question_type     type;
    json instructions;
    std::vector<server_decision_option> options; // in the order of the model outputs
};

// what a request asks for: its questions, and the state to evaluate them against
struct server_decision_request {
    std::vector<server_decision_question> questions;
    json                                  state;
    std::vector<raw_buffer>               files;
};

// how one model type reads a request. A new model type adds a field here and a case in init(), not an argument to every seam
struct decision_model_traits {
    size_t n_options_max   = 0; // what one question may ask
    bool   noul_true_first = false; // noul options are [true, false] instead of [false, true]
    bool   noul_criteria_order = false; // noul options are in the order of the criteria keys, if given
    bool   choice_sorted   = false; // choice options are in the order of their keys
    bool   strict_payload  = false; // state, instructions and descriptions must be text, an object or an array
    bool   canonical_legend = false; // non-string score descriptions are returned as canonical JSON text

    // OPENJEV, LEV, NIMBLE
    std::vector<std::string> label_texts; // only if the label of an option is given to the template

    // LAYA, KEV
    std::string text_marker; // only if the options are delimited by a marker in the text
};

// the tasks of one request, grouped so a shared prompt prefix is evaluated once. n_shared is the prefix tokens the children get from their parent; the caller subtracts them from usage
struct server_decision_tasks {
    std::vector<server_task> tasks;
    int32_t                  n_shared = 0;
};

struct server_decision_context {
    common_decision_type type = COMMON_DECISION_TYPE_NONE;

    // read the "<arch>.decision.*" metadata, type stays NONE if the model has none. Throws if the model is a decision model this server cannot run; the caller decides whether that is fatal.
    void init(const llama_model * model);

    // true if the questions of a request start with the same tokens, and the model can continue from them
    bool can_share_prompt() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
            case COMMON_DECISION_TYPE_GRANITE_SWITCH:
            case COMMON_DECISION_TYPE_LEV:
            case COMMON_DECISION_TYPE_KEV:
            case COMMON_DECISION_TYPE_NIMBLE:
            case COMMON_DECISION_TYPE_PPLX_DECIDER:
            case COMMON_DECISION_TYPE_VLLM_DECISION2:
            case COMMON_DECISION_TYPE_LFM2_D1:
            case COMMON_DECISION_TYPE_H2O_LIGHTNING:
                return true;
            default:
                return false;
        }
    }

    // true if all the questions of a request go in one prompt, see fill_task_joint()
    bool is_joint() const {
        return type == COMMON_DECISION_TYPE_CLEF;
    }

    bool is_lfm3_d1() const {
        return type == COMMON_DECISION_TYPE_LFM3_D1 || type == COMMON_DECISION_TYPE_LFM3_D1_OMNI;
    }

    // true if the prompt of the model has a place for images
    bool can_use_images() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
            case COMMON_DECISION_TYPE_CLEF:
            case COMMON_DECISION_TYPE_PPLX_DECIDER:
            case COMMON_DECISION_TYPE_VLLM_D3:
            case COMMON_DECISION_TYPE_LFM2_D1:
            case COMMON_DECISION_TYPE_LFM2_D1_OMNI:
            case COMMON_DECISION_TYPE_H2O_LIGHTNING:
            case COMMON_DECISION_TYPE_MINICPM_OCR_JEV:
            case COMMON_DECISION_TYPE_LFM3_D1:
            case COMMON_DECISION_TYPE_LFM3_D1_OMNI:
                return true;
            default:
                return false;
        }
    }

    // what the request asks for, checked against the limits of this model. The only entry point for parsing a body
    // support_audio_input: false if the server has no audio encoder, a spoken question is then rejected
    server_decision_request parse_request(const json & body, bool support_audio_input = false) const;

    // false for a decision-only model, it serves no text generation
    bool can_generate() const {
        const auto out = server_model_output_modalities(type, true, false);
        return std::find(out.begin(), out.end(), "text") != out.end();
    }

    // throw std::invalid_argument on bad input
    // support_audio_input: false if the server has no audio encoder, a spoken question is then rejected
    std::vector<server_decision_question> parse_questions(const json & body, bool support_audio_input) const;

    // the state as the model reads it, done once per request; returns the state for most models
    json prepare_state(const json & state) const;

    // returns the state without its images, they are appended to files in order
    // images come from "files" (alias "images") and from the image_url and input_audio parts of a state made of chat messages
    // an image can be an audio clip if the model supports it
    json parse_state(const json & body, std::vector<raw_buffer> & files) const;

    // number of prompts that are evaluated to answer this question, each one shows the options in a different order
    size_t n_variants(const server_decision_question & question) const;

    // set the prompt of one variant of this question, and where to read its result. mctx is only used if there are files
    void fill_task(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_question & question,
            size_t variant,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            server_task & task) const;

    // set the prompt of all the questions, the result has the scores of all their options, in order. mctx is only used if there are files
    void fill_task_joint(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            server_task & task) const;

    // lfm3-d1: max questions per prompt, 0 if each question has its own prompt
    size_t set_size(const std::vector<server_decision_question> & questions) const;

    // set the prompt of these questions packed after their state, the result has the scores of all their options
    void fill_task_set(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            server_task & task) const;

    // one task per pass of each question, or the single task that answers every question at once, or one task per packed set of questions, grouped so a shared prefix is evaluated once. next_id hands out the task ids. max_tasks bounds the tasks before any render (0 is unlimited); max_prompt_tokens bounds the rendered tokens.
    server_decision_tasks build_tasks(
            const server_decision_request & request,
            const std::function<int()> &    next_id,
            mtmd_context *                  mctx,
            const mtmd_helper_init_opt &    init_opt,
            size_t                          n_slots,
            size_t                          max_tasks,
            size_t                          max_prompt_tokens) const;

    // how many tasks the request builds: one for a joint head, one per packed set, else one per variant of each question. Pure: no render, no tokenizer, so build_tasks() refuses on it before rendering.
    size_t count_tasks(const server_decision_request & request) const;

    // scores: the raw model outputs of each variant
    json format_answer(const server_decision_question & question, const std::vector<std::vector<float>> & scores, bool has_media = false) const;

private:
    const llama_vocab * vocab = nullptr;
    std::shared_ptr<const common_chat_template> tmpl; // the "systemone" template

    std::map<std::string, float> temperatures; // "<type>" or "<type>.<n_options bucket>"
    decision_model_traits        traits;
    std::map<size_t, std::vector<float>> score_bias; // added to the scores of a score question, by number of levels
    size_t n_options_max   = 0;
    bool   noul_true_first = false; // noul options are [true, false] instead of [false, true]
    bool   noul_criteria_order = false; // noul options are in the order of the criteria keys, if given
    bool   choice_sorted   = false; // choice options are in the order of their keys
    bool   strict_payload  = false; // state, instructions and descriptions must be text, an object or an array
    bool   canonical_legend = false; // non-string score descriptions are returned as canonical JSON text

    // OPENJEV, LEV, NIMBLE, PPLX_DECIDER, VLLM_D3, H2O_LIGHTNING
    std::vector<llama_token> labels;
    std::vector<std::string> label_texts; // only if the label of an option is given to the template

    // LAYA, KEV
    llama_token token_marker      = LLAMA_TOKEN_NULL;
    llama_token token_sep         = LLAMA_TOKEN_NULL;
    std::string text_marker;
    size_t      max_head_tokens   = 0; // question + options
    size_t      max_option_tokens = 48;

    // H2O_LIGHTNING
    std::vector<std::string> quoted_spellings; // texts of the special tokens, quoted in the input, longest first
    std::string              quoted_first;     // the first bytes of these texts

    std::string render(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_question & question,
            size_t variant,
            size_t n_images,
            bool is_audio = false) const;
    json render_options(const server_decision_question & question, size_t variant) const;
    size_t n_outputs(const server_decision_question & question) const;
    // LFM2_D1: label text and tokens of each option
    void d1_labels(const server_decision_question & question, std::vector<std::string> & texts, std::vector<llama_tokens> & groups) const;
    // H2O_LIGHTNING: the input of the template, its texts are rendered as the reference shim does
    json render_h2o(const json & state, const server_decision_question & question) const;
    std::string h2o_cut_state(const std::string & text) const;
    std::string h2o_quote(const std::string & text) const;
    // the texts of the tokens with these attributes, quoted_spellings and quoted_first
    void init_quoted_spellings(uint32_t attrs, size_t min_size);
    // LFM3_D1: a zero-width space after the first character of each special token text (d1 prompt.py: plain)
    std::string d1_break(const std::string & text) const;
    void fill_task_laya(llama_tokens & tokens, const server_decision_question & question, server_task & task) const;
    void fill_task_d1omni(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_question & question,
            size_t variant,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            server_task & task) const;

    float get_temperature(const server_decision_question & question) const;
    double get_decision_confidence(server_decision_question_type type, const std::vector<double> & probs) const;
};

//
// free functions
//

// the two halves of a body, both refuse with server_invalid_request naming the field. decision_parse_state appends the images to files. allow_null_state is only for models that answer without a state (lfm2-d1 with images only)
// type defaults to NONE so existing tests that pass only (body, traits) keep compiling; production passes the model type for new-model branches
// support_audio_input: false if the server has no audio encoder, a spoken question of lfm3-d1-omni is then rejected
std::vector<server_decision_question> decision_parse_questions(const json & body, const decision_model_traits & traits, common_decision_type type = COMMON_DECISION_TYPE_NONE, bool support_audio_input = false);
json                                  decision_parse_state(const json & body, std::vector<raw_buffer> & files, bool allow_null_state = false, common_decision_type type = COMMON_DECISION_TYPE_NONE);

// the options of one question in template order. Variant 1 is the reverse order, so the model cannot prefer a position over an option. d1_texts carries the per-question label texts of lfm2-d1, null for every other model
json decision_template_options(
        common_decision_type             type,
        const decision_model_traits &    traits,
        const server_decision_question & question,
        size_t                           variant,
        const std::vector<std::string> * d1_texts = nullptr);

// what one prompt is rendered from. A struct, so two counts of the same type cannot be swapped non-owning: only a temporary for one decision_template_input() call, never stored
struct decision_prompt {
    common_decision_type                       type      = COMMON_DECISION_TYPE_NONE;
    const decision_model_traits &              traits;
    const json &                               state;
    const std::vector<server_decision_question> & questions;
    const server_decision_question &           question;
    size_t                                     variant   = 0;
    size_t                                     n_images  = 0;
    const std::vector<std::string> *           d1_labels = nullptr; // lfm2-d1 per-question labels, null otherwise
};

// the globals the "systemone" template is given for one variant of one question of one request
json decision_template_input(const decision_prompt & prompt);

std::string decision_render_template(const common_chat_template & tmpl, const json & inp);

// the probability of each of the n options: a softmax over each variant, then the average of the variants. Variant 1 is the reverse order; throws if a variant carries != n scores
std::vector<double> decision_average_variants(const std::vector<std::vector<float>> & scores,
                                              size_t                                  n,
                                              float                                   temperature);

// the scores of one question out of a joint head's vector, which holds the scores of every option of the request in order. Throws if the vector is too short, so a short head is a server fault
std::vector<float> decision_joint_scores(const std::vector<float> &         scores,
                                         size_t                           offset,
                                         const server_decision_question & question);

// saturating add and multiply: clamp to SIZE_MAX instead of wrapping
size_t decision_saturating_add(size_t a, size_t b);
size_t decision_saturating_mul(size_t a, size_t b);

// iterative depth check over attacker JSON with an explicit stack, so nesting cannot overflow
// the call stack. Runs on the parsed body before any dump() or recursive walk.
void decision_check_json_depth(const json & val, const char * field = "body");

// depth check on the raw body text before json::parse, whose own recursion is unbounded
void decision_check_raw_json_depth(const std::string & body);

// sum of string bytes without building a dump string, so the pre-flight itself cannot allocate
size_t decision_json_content_bytes(const json & val);

// how much of a LAYA prompt the head window keeps: option and question tokens before and after clipping to max_head_tokens. Pure, so tests pin the warning values without a model.
struct decision_head_clip {
    size_t n_option_max    = 0; // per-option cap both clippings agree on
    size_t n_options_full  = 0;
    size_t n_options_kept  = 0;
    size_t n_question_max  = 0;
    size_t n_question_full = 0;
    size_t n_question_kept = 0;
    bool   clipped_options  = false;
    bool   clipped_question = false;
};
decision_head_clip decision_clip_head(const std::vector<size_t> & option_sizes, size_t head_end, size_t max_head_tokens, size_t max_option_tokens);

// prompt tokens billed for a prompt sum minus its shared discount, floored at zero
size_t decision_usage_input_tokens(size_t n_tokens_sum, size_t n_shared);

// the queued decision tasks at which /v1/systemone starts refusing, a parent and each child counting one.
size_t decision_queue_cap(int cap, int n_parallel);

// the rendered prompt tokens one /v1/systemone request may total, counted over every task before a shared prefix is discounted.
size_t decision_prompt_budget(int cap, int n_parallel, int n_ctx_slot);

// the decision tasks one /v1/systemone request may build: sized by the server (slots x slot context), so tiny explicit token budgets keep this bound and only a larger explicit budget raises it; 0 is unlimited.
size_t decision_task_budget(int cap, int n_parallel, int n_ctx_slot);

// the Retry-After a refusal carries, in seconds. The cap counts tasks, not time, so this is a floor for the client, not an estimate of when the queue will drain
inline constexpr int DECISION_RETRY_AFTER_S = 1;

// how many images one /v1/systemone request may carry, shared by the images field and the image parts of a chat-message state. A request over it is refused whole, before any image is decoded
inline constexpr size_t DECISION_MAX_IMAGES = 8;

// group the tasks so that the common prefix of their prompts is evaluated only once
// each group is one parent and its children, it takes at most n_slots slots
// note: the order of the tasks is preserved
// build_tasks() groups with this, then sums n_tokens_shared for the usage discount
std::vector<server_task> server_decision_group_tasks(std::vector<server_task> && tasks, size_t n_slots);

// write "<|name|>" with U+00A6 in place of "|", so text from the request is not parsed as special tokens
std::string server_decision_escape_special_tokens(const std::string & text);
