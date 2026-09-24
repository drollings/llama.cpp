# Jev-Compatible Decision API (`/v1/decision`) - Introduction and Specification

Audience: a human or AI coder meeting this project for the first time. Read
Section 0 first: it fixes the endpoint name, defines every term used later,
and (Section 0.6) introduces the technical challenges of serving this API
inside llama.cpp. Sections 1 onward are the contract.

Status: implemented. A passage marked
DEFERRED is not built here; nothing is currently deferred.

The endpoint naming is the one deliberate difference from Jev, and it is
settled:

- `POST /v1/decision` is the canonical route.
- `POST /decision` is a deprecated alias for the same handler.
- `/v1/systemone` is Jev's own public path. It is NOT registered on this
  server. "Jev-compatible" means the request and response fields and their
  semantics match, so a client written for Jev is pointed here with only a
  base-URL/path change.

Other locked shapes: dual readout (letter labels for Jev, trie for generic
schemas); adapters strictly optional; temperature `T=1.0` default with a
per-type override; hybrid KV fork (`seq_cp` fast path, `llama_state_seq`
restore fallback); probabilities are closed-world and never gated on
confidence.

---

## 0. Orientation for a fresh reader

### 0.1 The problem this solves

A "decision" request asks a language model many small closed questions about
one piece of evidence, and wants CALIBRATED-ISH probabilities, not prose. For
example: given a support ticket, answer 12 questions at once - "should we
refund? (yes/no)", "which department? (billing/support/...)", "how urgent?
(low/med/high)". The caller needs the full distribution over the allowed
options for each question, with no generated text. This differs from chat:
there is no free-form output, no sampling, and every answer is guaranteed to be
one of the declared options.

### 0.2 What "Jev" means here

Jev is a hosted decision API whose public contract is `POST /v1/systemone`
with a request of `state` + typed `questions` and a response of typed
`answers`. "Jev-compatible" means our server returns the same field names and
semantics, so a client written for Jev can be pointed at our server with only a
base-URL/path change.

On this server the path is `/v1/decision`, not `/v1/systemone`:

- Jev host path: `POST /v1/systemone`.
- This server: `POST /v1/decision` (canonical), `POST /decision` (deprecated
  alias). No `/v1/systemone`.

Nothing about the request or the response changes with the path; only the URL
the client dials. The three question primitives are:
- `noul` - a yes/no probability (`noul` in [0,1]);
- `choice` - pick one of N options (`choice` + `probabilities` + `confidence`);
- `score` - a rating on an ordered scale (`score` + `legend` + `probabilities`
  + `confidence`).

### 0.3 What this repository is and where the feature lives

This is `llama.cpp`, a C/C++ inference engine for GGUF-quantized LLMs. The
decision feature is these files:

- engine (`tools/parallel-decision/decision-engine.{h,cpp}`, namespace
  `llama_decision`): the fork/score substrate shared by both readouts.
- protocol (`tools/parallel-decision/decision-protocol.{h,cpp}`): the Jev
  request/response shape, error types, temperature profile, contract hash.
- labels (`tools/parallel-decision/labels.{h,cpp}`): the single-token letter
  pool and the prompt-boundary checks.
- letter readout (`tools/parallel-decision/letter_readout.{h,cpp}`): the Jev
  scoring path built on the engine.
- HTTP wiring: `tools/server/server-context.cpp` (`handle_decision`),
  `tools/server/server.cpp` (routes), `tools/server/server-task.h`
  (`SERVER_TASK_TYPE_DECISION`).
- sizing flag: `--decision-seqs N` (`common/arg.cpp`, `common/common.h`),
  which reserves sequence ids and forces a unified KV cache.
- a standalone CLI client: `tools/parallel-decision/parallel-decision.cpp`
  (`llama-parallel-decision`).
- tests: `tests/test-decision-engine.cpp`, `tools/server/tests/test_decision_*.py`,
  and the recorded baselines under `tests/decision-baseline/`.

A separate experimental decision service (its own `/v1/systemone` route,
classifier head, exclusive mode) is not part of this work; it is reference
material only.

### 0.4 As-built map in one table

What a fresh reader can rely on:

| Area | As built |
|---|---|
| Routes | `POST /v1/decision` canonical; `POST /decision` deprecated alias; no `/v1/systemone` |
| Request | both shapes: Jev `state` + `questions`, and generic `contexts[]` + `schema` |
| Response | default Jev `{model, answers{qid}, usage{input_tokens, output_tokens}}`; `head`, `diagnostics`, per-answer audit, `certainty`, and extra usage counters only with `diagnostics: true`; generic `{object:"decision", results:[...], usage, timings}` |
| Readout | dual: letter labels (Jev) and surface-form trie (generic), one engine and one softmax |
| Errors | full contract in Section 4 (400/401/413/415/422/429/499/500/501/529) |
| Fork | probe: copy fast path (`llama_memory_seq_cp`) or `llama_state_seq` save/restore on hybrid/recurrent memory; SWA clamp |
| Prefix cache | bounded LRU (copy mode: token cache; restore mode: state LRU) |
| Temperature | `temperature` + `temperatures{}` with optional provenance; `T=1.0` default |
| Order de-bias | `permutations` (default 1, capped 8) |
| Admission | body cap (413), queue cap (429/529 + `Retry-After`), cancel (499), cooperative yield |
| Audit | `allowed_token_mass`, `full_vocab_argmax_id`, `answer_token_ids`, `option_logits`, `prompt_sha256`, `prompt_version`, `probability_status`; emitted only with `diagnostics: true`; the two full-vocab fields are omitted under the selected head (Section 3.1) |
| Provenance | contract hash (tokenizer + template + label code), logged and returned |
| Mode | `auto`/`tree`/`greedy`, `tree_max`, `cache_prompt` for the trie path |
| Head | selected-head fast path (Section 6.1): classifier-only graph + answer-row dot product; `auto`/`selected`/`full`; falls back to full logits unless `selected` is forced and unavailable |

### 0.5 Glossary

- **state**: the evidence document; opaque to the server, treated as DATA.
- **question**: one typed decision over a `state`; identified by an opaque
  `qid` that is never shown to the model.
- **option / candidate**: one allowed answer value for a question.
- **label**: a single-token letter (`A`, `B`, ... up to `BL`) the model emits
  instead of the option text; probability comes from that token's logit.
- **readout**: how option probabilities are extracted. Letter readout uses one
  label token per option; trie readout scores the model's own token paths for
  surface-form values.
- **branch / trunk / prefix**: one KV sequence per scored path; a trunk is
  `shared prefix + context`; the prefix is the cacheable head.
- **producer confidence vs task value**: two DIFFERENT axes (Section 7). Never
  use a confidence number to gate caching/admission/correctness.
- **closed-world probabilities**: `probabilities` are conditional on the
  supplied options; they do not measure the chance that all options are wrong.
- **contract hash**: a hash over the tokenizer identity, the framed prompt
  template, and the label code. If it changes, any earlier calibration is
  stale. Logged at startup and returned as `diagnostics.contract_hash`.
- **audit fields**: additive per-answer telemetry (`allowed_token_mass`,
  `full_vocab_argmax_id`, `answer_token_ids`, `option_logits`,
  `prompt_sha256`, `prompt_version`, `probability_status`). For inspection
  only; they never change an answer.

### 0.6 The technical challenges of serving this inside llama.cpp

llama.cpp is a token sampler at heart: you feed tokens, call `llama_decode`,
and read logits so you can sample. A decision request wants the opposite:
no sampling, no generation, just a clean probability distribution over a small
set of declared options. Most of the engineering is making "score, do not
generate" work well on top of a KV-cache engine built for chat. The challenges,
in the order a newcomer meets them:

1. **Score, do not sample.** You still call `llama_decode`, but you read
   `llama_get_logits` at one chosen position (the last token of the question
   suffix) and softmax only over the allowed tokens. No token is ever appended
   to the output; `output_tokens` stays 0. Getting the position right matters:
   read the logits at the position that predicts the first answer token, not
   after it.

2. **One piece of evidence, many questions, one forward pass.** Prefill the
   shared system prompt and the `state` once, then answer every question from
   that cache instead of re-prefilling per question. llama.cpp exposes this as
   sequence ids plus fork/trim calls; the engine keeps one trunk
   (`shared prefix + state`) and forks a short branch for each question.

3. **Forking a KV cache is architecture-dependent.** On plain attention you can
   copy a sequence (`llama_memory_seq_cp`) cheaply, because unified memory
   shares cells. Recurrent and hybrid models (for example Qwen3.5, Gemma, LFM2)
   cannot copy sequence state that way; there you save and restore a whole
   sequence with `llama_state_seq_get_data` / `set_data`. Sliding-window models
   only retain a window, so a copy must be clamped to that window or branch
   memory grows without bound. The engine probes the loaded model and picks a
   path, and `fork: copy|restore|auto` overrides it.

4. **Tokenization must be exact.** The model answers with a single label token
   (`A`, `B`, ...). The label token is resolved at the assistant-answer
   boundary, not in isolation: a SentencePiece / `add_space_prefix` vocabulary
   tokenizes a bare `"A"` as the space-prefixed form `" A"`, but emits the bare
   `"A"` after the framed tail, so isolation picks the wrong token. Byte-pair
   tokenizers can likewise merge a label into the previous text depending on
   whitespace and template, so `"\n" + "A"` may encode as one token. If that
   happens the score silently comes from the wrong position. Every label is
   therefore resolved and verified at startup and per request:
   `encode(tail + label) == encode(tail) + [label]`, else a hard error. The
   assistant-answer tail is template-specific and asserted exactly.

5. **Closed-world probabilities, and honesty about them.** The distribution is
   conditional on the options the caller supplied; it does not express the
   chance that every option is wrong. `confidence = 1 - H/log(K)` and
   `certainty = max(p)` measure concentration, not accuracy. They are
   never allowed to gate admission, caching, routing, or persistence.

6. **Set it inside a chat server that must keep working.** The same loaded
   model also serves `/v1/chat/completions`. A decision pass must not corrupt
   chat KV state, must not freeze the scheduler for seconds (the engine yields
   between waves so metrics and slot reads stay responsive), must respect
   admission (body cap 413, queue cap 429/529), and must cancel when the client
   leaves: cancellation has to reach the compute loop, not just the HTTP wait.

7. **Batch many questions while keeping them blind.** Questions are independent
   by contract: a `qid` never appears in the prompt and one question's options
   never condition another's answer. The engine packs branches into waves
   bounded by the sequence pool and the batch row budget, and gathers logits
   only at the scored positions.

8. **Make drift visible.** Record the model, quantization, template hash, and
   backend flags; hash the whole readout contract and refuse a mismatch; emit
   per-answer audit fields (`allowed_token_mass`, `full_vocab_argmax_id`,
   `option_logits`) so prompt drift or a wrong readout shows up as data instead
   of a silently wrong answer.

9. **Selected-head fast path.** A classifier-only context stops the graph after
   the final normalization layer, skipping the full-vocabulary projection;
   `llama_model_classifier_rows` dequantizes only the K answer rows, scored with
   a host-side dot product. When the head is unavailable (unsupported arch,
   hidden states absent, row/layout mismatch) the readout falls back to full
   logits; only an explicit `head: "selected"` on an unavailable head is an
   error (Section 6.1).

Everything after this section is the contract that these challenges produce:
request (Section 2), response (Section 3), errors (Section 4), and the chosen
implementation plus alternatives (Sections 5 and 6).

---

## 1. What the API is

One POST. No generation. A caller supplies opaque `state` (evidence, treated
as DATA, never as instructions) plus 1-256 independently-scored typed
questions. The server returns one closed-world distribution per question,
assembled by code. `output_tokens` is always 0; no text is sampled.

```
POST /v1/decision
state + N questions -> N answer distributions
```

Route policy:
- `POST /v1/decision` is the canonical route, `POST /decision` a deprecated
  alias for the same handler (`tools/server/server.cpp`).
- Do NOT serve `POST /v1/systemone` from this branch. Jev compatibility is
  achieved through request and response FIELD names and semantics, not the URL
  path. A thin client-side base-URL swap (for example `TYPESAFE_BASE_URL`) plus
  path rewrite is the supported migration; JevBench `typesafe` adapters
  otherwise talk unchanged.

Coexistence: the same model/server also serves the OpenAI-compatible API
(`/v1/chat/completions`, `/v1/models`, `/health`). Decision traffic must not
corrupt chat KV state; see Section 6.1.

---

## 2. Request specification

Content-Type: `application/json`. Body limit: 2 MiB default
(`LLAMA_DECISION_MAX_BODY`; winnow used 32 MiB, 2 MiB matches openjev-sglang
and is sufficient - raise only deliberately). Unknown top-level fields are
tolerated and ignored for forward compatibility. Unknown fields inside a
question object are still rejected (a misspelled `question`/`criteria` must not
silently default).

```json
{
  "model": "qwen3-4b",
  "state": "string | object | array (REQUIRED, non-empty)",
  "questions": {
    "<qid>": {"type": "noul|choice|score", "instructions": "string|object|array", "criteria": ...}
  },
  "temperature": 1.0,
  "temperatures": {"noul": 1.0, "choice": 1.0, "score": 1.0},
  "permutations": 1,
  "diagnostics": false
}
```

### 2.1 Fields

* `model` (optional, string): routing/echo only. Accept any string including
  `jev-latest` (treat as alias for the loaded weights). Never load weights
  per-request. Echo back in the response.
* `state` (required): string, JSON object, or JSON array; must be non-empty
  and finite (`allow_nan=False`). Chat-transcript states
  (`[{role, content}]` / `{"messages": [...]}`) MAY be accepted and are
  passed through as evidence. The state is DATA: frame it as evidence, escape
  `<` as `\u003c`, and instruct the model that state content is not
  instructions (prompt-injection hardening; full corpus in Section 7).
* `questions` (required): object mapping `qid -> question spec`, 1-256
  entries. `qid`s are opaque: they MUST NEVER be shown to the model, and
  adding/removing/reordering questions MUST NOT change other answers
  (independent branches/rows).
* `temperature` (optional, float > 0, default 1.0): global softmax
  temperature applied to gathered label logits; see Section 7 for provenance
  rules.
* `temperatures` (optional): per-primitive overrides
  `{noul, choice, score}`. Effective temperature for a question =
  `temperatures[type]` if present else `temperature`. Ship `1.0` everywhere;
  fit per deployment offline (Section 7).
* `permutations` (optional, int 1-8 or null, default 1): order-debiasing
  passes. `1` = single canonical order. `2` = identity + one seeded distinct
  shuffle, per-order softmax then mean by semantic key (seeded by
  `(seed, qid)`; noul second order = swapped). Values above 8 are accepted and
  capped at 8; document the measured cost (~1.1x for 2).
* `head` (optional, string): scoring path. Omit or `""` for auto; `"full"`
  forces full-vocabulary logits on the shared context; `"selected"` requests
  the answer-head fast path (Section 6.1). The default path uses the head when
  it is available and silently falls back to full logits otherwise; an explicit
  `head: "selected"` on an unavailable head is a client error (400).
* `diagnostics` (optional, bool, default `false`): when `true`, the response
  additionally carries the `head` object, the `diagnostics` object, per-answer
  audit fields, `certainty`, and the extra `usage` counters
  (`cached_tokens`, `state_cache_hit`, `head_mode`). The default `false` keeps
  the response to the strict Jev envelope (Section 3).

Local-only notes: `model` is optional here and echoed back verbatim; Jev
requires it. `GET /v1/models` keeps the OpenAI model-list shape
(`{"object":"list","data":[...]}`), not Jev's `{"models":[...]}` shape.

### 2.2 Question types (canonical names)

Accept `bool` as an alias for `noul` and `scale` as an alias for `score` on
INPUT; always emit canonical `noul`/`choice`/`score` on output.

* `instructions` REQUIRED and non-null on all three types (string, object, or
  array); a question with null or missing `instructions` is a 422 naming
  `instructions`.
* `noul`: binary judgment.
  `criteria` optional: `{"true": "desc", "false": "desc"}` (descriptions may
  be strings; treat missing as null desc). Internally two options
  `["false","true"]` (or lettered `A:yes/B:no` - equivalent after mapping).
* `choice`: categorical selection.
  `criteria` REQUIRED: object `{key: desc|null}`, 2-64 entries, keys in object
  order. Model sees description, or key when null. Answer uses the KEY.
* `score`: ordered rating, zero-based.
  `criteria` REQUIRED: ordered array of 2-10 level descriptions, low -> high
  (equivalently `{"0": desc, ...}` legend dict on input). Model sees each
  level; answer is the expected index.

Validation limits: `noul` fixed 2 options; `choice`
`DECISION_MIN_OPTIONS`-`DECISION_MAX_CHOICE_OPTIONS` (2-64); `score`
`DECISION_MIN_OPTIONS`-`DECISION_MAX_SCORE_LEVELS` (2-10); questions
`DECISION_MIN_QUESTIONS`-`DECISION_MAX_QUESTIONS` (1-256); permutations capped
at `DECISION_MAX_PERMUTATIONS` (8); the answer-label pool is `LABEL_POOL_CAP`
(64). `choice` and `score` limits are the letter-mode caps; trie-mode generic
schemas allow up to 255 values/field - Section 6.1. Option IDs/keys unique.
Reject over-limit; NEVER truncate. Reject empty `criteria` where required.
Structured `instructions`/`criteria` values are rendered into the prompt,
never silently stringified; `legend` echoes the ORIGINAL structured values so
they round-trip.

Each option line is rendered by `format_option_line` (one function) as
`label: <key> - <description>`, where the label is the single letter the model
answers with; when the rendered description is empty the line is
`label: <key>` only (no trailing separator). Keys are `false`/`true` for noul,
the option names for choice, and `"0".."K-1"` for score.

### 2.3 Generic-schema request (trie mode)

The branch already implements a non-Jev flat-JSON form. It stays supported
byte-identically; the Jev form is added beside it.

```json
{"contexts": ["ctx1", ...], "schema": {...}, "instructions": "...",
 "mode": "auto|tree|greedy", "tree_max": 128, "cache_prompt": true}
```

* `contexts`: 1-256 non-empty strings, decided in order, sharing one schema.
* `schema`: compact `{name: {type, description, choices/enum, minimum,
  maximum, step, aggregate}}` or JSON-Schema `{properties: {...}}`, 1-32
  fields. Types: `boolean`, `enum` (1-255), `integer`/`number` (1-255 grid
  points, fixed-width decimals). Numeric `aggregate: mode|median|mean`.
* `mode` default `auto` (tree if values <= `tree_max` else greedy).
* Response for this form stays `{object:"decision", model, created,
  results:[{decision, fields, usage}], usage, timings}` (unchanged).

When both shapes are supported, negotiate by request shape: presence of
`state`+`questions` selects the letter readout; presence of `contexts`+`schema`
selects the trie readout. Never mix both in one call.

---

## 3. Response specification

The default response is exactly the Jev envelope:

```json
{
  "model": "<echo>",
  "answers": {
    "<qid>": {"type": "noul", "noul": 0.0-1.0}
            | {"type": "choice", "choice": "<key>",
               "probabilities": {"<key>": p, ...},
               "confidence": c}
            | {"type": "score", "score": 0.0-(K-1),
               "probabilities": {"0": p, ...}, "legend": {"0": "desc", ...},
               "confidence": c}
  },
  "usage": {"input_tokens": N, "output_tokens": 0}
}
```

With `diagnostics: true` the same answers are returned with additive fields:
`certainty` on choice/score, the per-answer audit fields, the `head` and
`diagnostics` objects, and the extra `usage` counters:

```json
{
  "usage": {"input_tokens": N, "output_tokens": 0,
            "cached_tokens": M, "state_cache_hit": true|false,
            "head_mode": "selected"|"full"}
}
```

### 3.1 Semantics (normative)

* `noul.noul = P(true)` in [0,1]. No `probabilities`/`confidence` required.
  `noul` MUST NEVER carry `confidence`.
* `choice.choice = argmax(probabilities)`; `probabilities` keyed by option
  KEY, sums to 1.
* `score.score = sum(i * p_i)` (expected zero-based index, float in
  [0, K-1]); `probabilities` keys are STRINGS `"0".."K-1"`; `legend` echoes
  input criteria in order with string keys.
* `confidence = 1 - H(p)/log(K)`, `H = -sum p log p` (normalized inverse
  entropy). This is a deliberate LOCAL choice, not Jev's documented
  distribution shape: Jev's Choice confidence is derived from the winner's
  share, `(N*p_max - 1)/(N - 1)`, and its Score confidence above 3 levels is
  undocumented. Do not claim parity for this formula. Clamp to [0,1].
* `certainty = max(p)` (the winner's share). Both are returned on `choice` and
  `score` when `diagnostics` is true; `confidence` is always returned. Both
  measure concentration of the answer distribution; they are not calibrated
  correctness and never gate admission, caching, routing, or persistence.
* `probabilities` are CONDITIONAL on the supplied options
  (`probability_status: "conditional option score; uncalibrated as decision
  confidence"`). `confidence`/`certainty` measure CONCENTRATION, not
  correctness. Document this in every model card and API doc.
* `usage.output_tokens` MUST be 0 (warmup/branch tokens are accounting-only).
  `usage.input_tokens` includes cache hits + warmup and counts a shared prefix
  ONCE. The default `usage` carries only `input_tokens` and `output_tokens`;
  with `diagnostics: true` it also carries `cached_tokens` and
  `state_cache_hit`, which expose prefix-cache behavior, and `head_mode`.
* When `diagnostics: true`, the response carries two additive objects beyond the
  frozen Jev shape. `head` reports how the answer was read out
  (`mode: selected|full`, `fallback`, and a `reason` when it fell back).
  `diagnostics` reports the readout identity (`contract_hash`, `prompt_version`,
  `model`, `quantization`, `template_hash`, `backend_flags`), the adapter scope
  (`adapters_configured`, `adapter_scope`), and timings (`prefill_ms`,
  `scoring_ms`, `suffix_tokens`, `common_suffix_tokens`). These are additive and
  never change an answer. The default response omits all of them.
* Audit availability depends on the readout. `allowed_token_mass` and
  `full_vocab_argmax_id` are full-vocabulary measurements; under
  `head_mode: "selected"` only the K answer rows are read, so those two fields
  are OMITTED (never emitted as placeholder `1.0`/`-1` values). `option_logits`,
  `answer_token_ids`, and `probability_status` are available under both the
  selected head and full logits. `usage.head_mode` and the `head` object report
  which readout ran.

### 3.2 Worked example

Request:

```json
{"model": "qwen3-4b", "state": "Customer asks for a refund of $42, order arrived broken.",
 "questions": {
   "refund": {"type": "noul", "instructions": "Should we refund?",
              "criteria": {"true": "yes, refund", "false": "no refund"}},
   "dept": {"type": "choice", "instructions": "Route the ticket.",
            "criteria": {"billing": "payment/refund issues", "support": "how-to help"}},
   "urgency": {"type": "score", "instructions": "Rate urgency.",
               "criteria": ["low", "medium", "high"]}}}
```

Response (default envelope):

```json
{"model": "qwen3-4b",
 "answers": {
   "refund": {"type": "noul", "noul": 0.9995},
   "dept": {"type": "choice", "choice": "billing",
            "probabilities": {"billing": 0.9999, "support": 0.0001},
            "confidence": 0.999},
   "urgency": {"type": "score", "score": 1.66,
               "probabilities": {"0": 0.08, "1": 0.18, "2": 0.74},
               "legend": {"0": "low", "1": "medium", "2": "high"},
               "confidence": 0.33}},
 "usage": {"input_tokens": 412, "output_tokens": 0}}
```

The same request with `"diagnostics": true` keeps the answers byte-identical
and adds `certainty`, the audit fields, the `head`/`diagnostics` objects, and
the extra usage counters:

```json
{"model": "qwen3-4b",
 "answers": {
   "dept": {"type": "choice", "choice": "billing",
            "probabilities": {"billing": 0.9999, "support": 0.0001},
            "confidence": 0.999, "certainty": 0.9999,
            "probability_status": "conditional option score; uncalibrated as decision confidence"}},
 "usage": {"input_tokens": 412, "output_tokens": 0,
           "cached_tokens": 180, "state_cache_hit": false,
           "head_mode": "selected"},
 "head": {"mode": "selected", "fallback": false},
 "diagnostics": {"contract_hash": "...", "adapters_configured": false, "adapter_scope": "base"}}
```

---

## 4. Errors and headers

`handle_decision` (`tools/server/server-context.cpp`) distinguishes the error
classes by exception type, and the server maps each through
`format_error_response` (`tools/server/server-common.cpp`) to an HTTP status.
Malformed JSON and unusable values/types are 400; well-formed but semantically
invalid decision content is 422 (including an unknown field inside a question
object); unknown top-level fields are ignored; over-limit capacity is 413 or
422; a full queue is 429 or 529 with `Retry-After`; a cancel or client
disconnect is 499; an unavailable model (no usable labels, contract mismatch)
is 501. The full contract follows.

### 4.1 Full target error contract

| HTTP | `type` | When | Notes |
|---|---|---|---|
| 400 | `invalid_request_error` | malformed JSON; unusable route/role/args | do NOT use for semantic question errors |
| 401 | `authentication_error` | missing/invalid API key when auth is configured | |
| 403 | `permission_error` | authenticated but not allowed | |
| 404 | `not_found_error` | unknown model/route | in router mode the decision route follows the existing proxy behavior, like the other routes |
| 413 | `payload_too_large` | request body over the configured cap | must reject before decode |
| 415 | `unsupported_media_type` | `Content-Type` is not `application/json` | enforced by the shared HTTP layer |
| 422 | `invalid_request_error` (semantic) | valid JSON but invalid decision schema: bad question type, 0/257 questions, 1/65 options, 1 or 11+ score levels, duplicate keys, missing `instructions`, missing required `criteria`, unknown field inside a question | Jev semantic-failure code; never downgrade to 400. Unknown top-level fields are ignored, not rejected |
| 429 | `rate_limit_error` | decision queue full | include `Retry-After` |
| 499 | `client_closed_request` | client disconnected / cancelled mid-evaluation | cancel siblings; never report as a normal answer |
| 500 | `server_error` | inference/engine failure | reset engine state |
| 501 | `not_supported_error` | feature not enabled/available | existing enum value |
| 503 | `unavailable_error` | shutting down / no service | include `Retry-After` when transient |
| 529 | `overloaded_error` | server overloaded | include `Retry-After`; Jev uses this |

Error body shape (unchanged from existing server):
`{"code": <http int>, "message": <string>, "type": <string>}`.

Type-string provenance: the `type` strings live in `format_error_response`
(`tools/server/server-common.cpp`). 400/401/403/404/500/501/503 predate this
work; 413/422/429/499/529 were added for the decision contract. The HTTP code
is normative; if a future implementation uses a different `type` string it must
document the mapping.

Notes:
- 422 vs 400 is a deliberate split: malformed syntax is 400, semantically
  invalid decision content is 422. A client must be able to tell them apart.
- Over-limit capacity is rejected BEFORE decode; the path never truncates and
  never silently clips.
- 413 comes from the body cap (`LLAMA_DECISION_MAX_BODY`), 429/529 from the
  concurrent-request cap (`LLAMA_DECISION_MAX_QUEUE`); both can be overridden
  for tests.
- The control cases (requests that must NOT be rejected) are covered by the
  server tests, alongside the fixtures for every row above.

### 4.2 Headers

* `Retry-After` on 429/529 (implemented).
* Cancel/disconnect aborts the evaluation and never emits a partial answer
  (implemented; a client that is already gone cannot observe the 499, so the
  observable guarantee is that no partial answer is produced).
* `x-decision-latency-ms` and `Server-Timing: prepare,prefill,branches` are
  proposed instrumentation, not currently emitted. Per-request timings are
  available via the audit fields and the generic `timings` object on the
  `contexts`/`schema` shape.

---

## 5. Back-end approaches surveyed

| Approach | Prompt | Readout | Parallelism | Calibration | Model scope |
|---|---|---|---|---|---|
| `_codacus_parallel_decision` (substrate of this branch) | chat template + sentinel; shared head + per-context tail; suffix + common-prefix split | surface-form TRIE (exact tree <= tree_max else greedy); never generates | `seq_cp` fork, unified KV, `--decision-seqs` pool, grouped waves | exact distribution (tree); path-product (greedy) | any decoder LLM |
| `exclusive classifier` (separate experimental service) | framed state + per-question suffix ending `Answer:` | single-token LETTERS + selected-head projection + softmax; `confidence=1-H/logK` | shared-prefix KV, cross-Q dedup, bounded-SWA reclaim, exclusive suspend/restore | none (T=1.0); concentration only | Gemma-4 only, GPU only, merged adapter |
| reflex (transformers + SGLang) | ChatML evidence prefix + `# Criterion/# Options/Respond with only letter` | single-token letters, restricted softmax/T | prefix LRU + packed mask (attn-only) or batched SGLang fanout (`token_ids_logprob`) | per-primitive T (post-hoc); 2-permutation averaging | frozen Qwen 4B/27B |
| decider (custom CUDA + vLLM check) | plain `Context:/Question/Options/Answer:(` | slot hidden state x `lm_head[letter_ids]`, `A-J` narrow / `A-Z+AA..` wide to 255 | `score_shared` LCP-once + `reorder_cache` fork; micro-batched graphs | CE fine-tune + T=1.30 + RL consistency | trained Mapika 2B/35B |
| SemIf (`semif_phase1`, torch/MLX/llamacpp) | system + JSON `{evidence,criterion,options[{letter,desc}]}` | full-vocab last-pos logits, gather A-P slots | serial save/restore; shared prefill + batched suffixes | offline per-workload T only | Qwen3.5 + GGUF verify |
| Nimble (MLX/CUDA) | JSON `{context,schema}` + `Requested field` | single-token codes, union-label FP32 projection | 1 prefill + `broadcast_cache` fork + one batched suffix call | T=1.0 unfitted, contract hash | trained 2k ctx |
| openjev-sglang | state passthrough + `...answer with only its label` + UUID marker | SGLang 1-token generate, per-label logprob lookup | warmup-prefill + asyncio branches (sem 64) | T=1.0, entropy confidence | SGLang + radix required |

Key consensus across all seven: single next-token readout over verified
alphabetic labels; shared-state prefix prefill once; per-question suffix
branches; `softmax(logits/T)`; entropy-based confidence labeled
non-calibrated; assemble-by-code; never truncate (reject over-limit).

---

## 6. Selected implementation (locked) and high-quality alternatives

### 6.1 Selected: llama.cpp dual-readout + hybrid fork + optional adapters

This is what the branch implements. Every bullet below is as-built.

* Prompt (Section 2): `apply_chat_template(add_generation_prompt=true,
  enable_thinking=false)`; raw fallback `system + "\nContext:\n" + state`.
  Assert marker survival; reject if template ate it. The assistant-answer
  boundary is architecture-specific: make it a parameter, assert the EXACT
  token ids for the production template tail, and use a synthetic tail in
  model-independent tests.
* Labels: pool `A-Z,AA-ZZ`; a label is the single token the model emits after
  the framed answer tail, resolved at that boundary (so `add_space_prefix`
  tokenizers are supported); keep iff single, non-special and unique; cap 64.
  Boundary check per branch: `encode(full_prompt + label) == ids + [label]`;
  hard error otherwise. (Numeric labels break at `>=10`; never use them.)
* Readout A (Jev types, <=64 opts): letter logits -> `softmax(logits/T_type)`.
* Readout B (generic flat JSON, <=32 fields, <=255 values): codacus trie -
  `suffix = '  "name": ' + common-char-prefix`, candidates = remainders,
  tokenize `suffix+candidate`, split at longest common token prefix, score
  divergence nodes in round 1 (tree) or follow-up rounds (greedy). Numeric
  `aggregate: mode|median|mean` + `interval_p10_p90`.
* KV: one shared-prefix prefill; per-question fork via `seq_cp` when the
  architecture supports copies; whole-sequence `llama_state_seq_get_data /
  set_data` + `seq_rm` fallback for hybrid/recurrent state (Qwen3.5, Gemma),
  exactly as `SemIf/src/semif_phase1/llamacpp_backend.py` does
  (`n_seq_max=1, n_outputs_max=1`, 512-token decode chunks,
  logits-on-last-token-only). Never mutate the cached prefix; single-question
  requests skip the fork path (measured slower with it).
* Batching: exact-token dedup + common-prefix hoist (>=32 tokens); waves of
  `<= parallel` leaves fitting `n_ctx`; group by length bucket; right-pad
  AFTER the scored position; per-row last-valid gather; logits at selected
  positions only; semaphore-bounded concurrency; cooperative yield + atomic
  cancel.
* Head fast path (implemented): a classifier-only context stops the graph after
  the final normalization layer, and `llama_model_classifier_rows` dequantizes
  only the K answer rows, scored with a host-side dot product (re-applying the
  logit softcap where the architecture has one). The head is used when the
  architecture supports the classifier stop and the hidden state and answer
  rows match; on any arch/head/layout mismatch it falls back to full-logits +
  gather, so an unavailable head never changes an answer. An explicit
  `head: "selected"` on an unavailable head is a client error (400). The row
  table is cached per model. `option_logits` and probabilities are both
  exposed; full-vocab probabilities never are.
* Order de-bias: `permutations` (default 1, capped 8), identity plus seeded
  distinct shuffles, per-order softmax then mean by option key. Default 1 is
  byte-identical to a request without the field.
* Admission and cancellation: body cap (413), concurrent-request cap
  (429/529 + `Retry-After`), 499 on client disconnect, and a cancel flag that
  is checked at entry, between waves, and before every `llama_decode` so the
  compute actually stops. The decision runs inside a cooperative yield so
  metrics and slot reads stay responsive while it computes.
* Contract and audit: a contract hash over tokenizer, template, label code and
  prompt version is logged and returned as `diagnostics.contract_hash`, and
  `--decision-contract` refuses a mismatch with 501. Every answer carries the
  additive audit fields (Section 0.5).
* Adapters (optional): merged LoRA/finetune MAY be used; LoRA/specialized
  serving paths MUST NOT be required. When the fast head path is explicitly
  requested with incompatible adapters/speculative settings, return 400 with a
  plain message rather than silently degrading.
* Adapter scope (stated behavior change): the decision decode always answers
  for the BASE model. Before every decision decode the server detaches the
  adapters from the serving context; the next chat batch re-applies its own
  set, so chat is unaffected. With adapters configured on the server (any
  registered adapter with a non-zero scale), the answer head cannot serve the
  adapted model, so `head: "selected"` returns 400, the default `head` reads
  full logits on the base scope and reports why it fell back, and
  `head: "full"` is unchanged. `diagnostics` reports `adapters_configured` and
  `adapter_scope: "base"` on every decision answer. Without adapters the
  behavior is byte-identical to the head selection described above.
* Tokenizer gates (from SemIf `llamacpp_backend.py:186-207`): GGUF
  tokenization MUST equal the reference encoding (`encode_verified`), plus a
  startup vocabulary probe (all label letters single shared tokens).
  Reference-tokenizer prompt hash (`prompt_sha256`, `prompt_version`) travels
  with every result for audit.
* Chat coexistence: default shares one weight allocation; decision pass runs
  on the context thread with cooperative yield points; NO exclusive eviction
  by default. Cache behavior is surfaced in `usage` (`cached_tokens`,
  `state_cache_hit`) and the timing counters ride along in the engine metrics.

### 6.2 Alternatives implementers may choose (all Jev-compatible)

1. SGLang logprob fanout (openjev-sglang): warmup-prefill to prime radix
   cache, then parallel 1-token `/generate` with `return_logprob +
   token_ids_logprob=labels`; require exactly 1 completion token; per-label
   lookup + softmax. Requires radix + CUDA graphs; reject
   `--disable-radix-cache` etc. Simplest correct server/GPU path.
2. Transformers selected-logits (reflex/SemIf-torch): `logits_to_keep=1`
   (single) or selected positions (shared); `reorder_cache`/`deepcopy` fork;
   right-pad + per-row gather; `torch.compile + FP8` optional. Best for
   research rigs and CPU/MPS fallback (MPS: looped batch-1 suffixes).
3. Native selected-head engine (decider/Nimble): `hidden @ lm_head[labels]`
   in a custom CUDA/MLX kernel; schema-first prefix KV; per-`(batch,len)`
   suffix graphs. Fastest at scale (ms/req graphs, 7-19x shared speedups)
   but most engineering.
4. Reranker cross-encoder (SemIf `reranker.py`): per-option
   `logit(yes)-logit(no)` log-odds then cross-option softmax. Order-invariant
   but empirically weaker as a decider (0.498 in cited test); use for
   retrieval relevance, not for Jev answers.
5. Serial state-restore on CPU (SemIf `llamacpp_backend.py:309-405`):
   `clear/prefill/save_state` once, then `restore_state + branch_logits` per
   question. The minimal correct llama.cpp CPU implementation; add batching
   later.

### 6.3 What NOT to do

* No free-text generation + parsing (breaks the 100% schema-validity
  guarantee and the `output_tokens:0` contract).
* No multi-token option decoding in letter mode (unrepresentable in one
  slot; use trie mode instead).
* No truncation on over-limit (reject with 413/422).
* No `qid` leakage into prompts; no cross-question conditioning in the
  default path.
* No presenting `confidence` as accuracy; no fitted global T claimed to
  transfer across quant/backend swaps.
* No arch-gated or GPU-gated hard requirements in the default path; no
  per-request weight loading.

---

## 7. Calibration, robustness, and acceptance

Two axes, never conflated:
- CONFIDENCE (producer self-doubt): `certainty`, `allowed_token_mass`,
  full-vocab argmax agreement. A model can be confident and still wrong.
- TASK VALUE (outcome correctness/economy): `tree_max`, hoist length, wave
  packing, single-question bypass, dedup, temperature fit quality
  (NLL/Brier/ECE), permutation gain.

Rules:
* Ship `T=1.0` + per-type overrides. A non-default temperature ships with
  PROVENANCE (model hash, quantization, template hash, backend flags) and is
  refused on mismatch. Temperature is argmax-invariant: it changes
  probabilities/thresholds, not winners. Provide an offline refit script
  (NLL/Brier per primitive on own traffic). Expect NVFP4/backend swaps to
  move ECE +0.02-0.09; refit per deployment.
* Order bias: offer `permutations=2` (seeded `(seed,qid)` distinct shuffle,
  mean by semantic key). Measured: ~1.1x tokens, roughly halves pooled ECE,
  +2-3pp hard accuracy in reflex tests. Default stays 1.
* Expect close-call flips across dtype/batch (SemIf 5-6/777; Nimble up to
  1.8pp deltas). Acceptance compares with TOLERANCE, never bit-equality.
  Pin `tokenizer + template + label code` in a contract/startup hash and
  refuse on mismatch (`prompt_code_sha256` pattern). Every parity/calibration
  claim inherits the frozen backend flag set recorded at baseline (FA type,
  K/V cache types, `kv_unified`, `swa_full`, `n_ubatch`, threads); changing
  flags invalidates the claim.
* Prompt-injection corpus for `safe_data`: `<|turn>`, `{REASON:`, `__media__`,
  backticks, and nested arrays/objects.
* Diagnostics to log per request: `allowed_token_mass`
  (mass on valid labels vs full vocab), `full_vocab_argmax` (detect
  off-label winners), `prefix/suffix tokens`, `cache_hit`, `waves`,
  `queue_ms`. These catch prompt-drift and head-mismatch before users do.
* Policy: the model's `confidence`/`certainty` NEVER gates admission,
  caching, routing, or persistence.
* Benchmarks to report: JevBench easy/std/hard + ECE per primitive;
  warm/cold latency at 1/8/64 questions; cache-hit rate; argmax-flip rate
  across quant/backend. Reference points: reflex-4B frozen ~0.917 std /
  0.685 hard ECE ~0.08; reflex-27B hard 0.766; decider-35B std 0.972;
  codacus-style 137-prompt + 14-row batch ~100 ms warm on Gemma-12B/3060.

---

## 8. Minimal implementation checklist (any back-end)

Status: items 1-7 and 9, 11, 12 are implemented;
item 8 is implemented except the optional latency headers; item 10 ships the
permutation option but not an offline refit script.

1. Strict validator: state non-empty; 1-256 questions; `instructions`
   required and non-null; choice 2-64 options, score 2-10 levels; unique
   IDs/keys; `bool`/`scale` aliases accepted; over-limit rejects.
2. Prompt renderer + marker-survival assert + `safe_data` escaping + exact
   assistant-boundary id assertion.
3. Label pool builder (single-token + round-trip + unique + non-special)
   + per-branch boundary check.
4. Shared-prefix prefill + per-question branch fork (native copy OR
   save/restore; never mutate prefix).
5. Single-token readout: gather K logits, `softmax(logits/T_type)`,
   `noul=P(true)`, `choice=argmax`, `score=sum i*p_i`,
   `confidence=1-H/logK`, `certainty=max_p`.
6. (If generic schemas:) trie scorer with `tree_max`/greedy fallback +
   numeric aggregates.
7. Assembler (code, not decode) + canonical answer envelope + `legend`
   with string keys + `usage{input_tokens,output_tokens:0}`; the extra
   counters (`cached_tokens`, `state_cache_hit`, `head_mode`) and the
   additive fields appear only with `diagnostics: true`.
8. Errors (400/401/403/404/413/422/429/499/500/501/503/529), `Retry-After`,
   abort-on-cancel, latency headers.
9. Tokenizer/vocab gates + `prompt_sha256` audit + contract hash.
10. Offline temperature-refit script + order-permutation option.
11. Chat-coexistence test: interleaved `/v1/chat/completions` +
    `/v1/decision` with KV integrity asserted.
12. Tolerance-based golden tests (no exact-logit assertions).

---

## 9. Red-team notes (why this shape)

* Closed-world overconfidence is inherent: probabilities condition on the
  supplied options even when all are wrong. Mitigated by contract language,
  `allowed_token_mass` telemetry, and never calling `confidence` accuracy.
* Single-token bottleneck forces prompt discipline (bare label output).
  Mitigated by dual readout and boundary verification.
* Hybrid KV cannot use copy-or-mask tricks (reflex `packed` mask fails on
  recurrent state). Mitigated by the save/restore fallback proven in
  `llamacpp_backend.py`.
* Exclusive-mode and GPU-only designs (winnow) buy peak speed at the cost
  of generality and chat disruption. Default stays shared and portable;
  exclusivity is an operator opt-in.
* Cross-field conditioning (codacus "fields mutually blind") and multi-hop
  reasoning are out of scope for the scoring pass; callers compose them with
  multiple decision calls or a chat call. Document, do not smuggle CoT into
  the readout.
* Sequence ids are partitioned (slots vs decision); the prefix LRU is bounded
  and in-memory; cancellation reaches compute, not just the wait.
