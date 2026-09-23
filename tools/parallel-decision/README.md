# parallel-decision

Answer a finite JSON schema in one batched forward pass, instead of generating the JSON token by token.

Every field of a schema has a fixed set of allowed values (enums, booleans, bounded integers, numbers on a grid).
After the context, each field's allowed values are scored as token paths that fork from the same KV cache, so all
fields are answered in one `llama_decode` and cannot see each other. Each answer comes back with a probability, and
the JSON object is assembled by code, so it always matches the schema.

This directory holds the engine (`decision-engine.*`), a CLI (`llama-parallel-decision`), and the engine is also
served by `llama-server` as `POST /v1/decision`.

## Build

Same as llama.cpp:

```bash
cmake -B build -DGGML_CUDA=ON        # or plain `cmake -B build` for CPU / Metal
cmake --build build --config Release -j
```

## Run the server

`--decision-seqs N` reserves the sequence slots the decisions need: one holds the cached instructions, one per context
in flight, the rest are the parallel questions. It also switches the KV cache to unified, which is what lets the
branches share the context's cells.

```bash
./build/bin/llama-server -m model.gguf -ngl 99 -fa on -c 32768 --decision-seqs 24 --port 8096
```

The letter readout runs on its own classifier-only context, which by default uses the model's context size.
That context has no chat slots: its only sequences are the decision engine's, numbered from zero, while the
shared chat context keeps the decision sequences above its slots. `--decision-ctx-size N` bounds the
classifier context independently, so it does not reserve a second full-size KV cache.
`N` is a cell budget: the request is checked against its peak use (the cached instructions plus every live
question branch) before any decode. A request that does not fit is rejected with `422`, returns no decision,
and leaves no partial state; the next request still succeeds. The decision path never truncates a prompt.

With a presets file, one loaded model serves chat and decisions:

```ini
[*]
ngl = 99
fa = on
jinja = 1
parallel = 1
cache-type-k = q8_0
cache-type-v = q8_0
decision-seqs = 24

[gemma-4-12b]
model = ./models/gemma-4-12b-it-UD-Q4_K_XL.gguf
ctx-size = 32768
ubatch-size = 512
decision-seqs = 12
```

```bash
./build/bin/llama-server --models-preset models.ini --models-max 1 --port 8096
```

How many sequences a model affords depends on its attention. A plain-attention model shares the context's cells, so
128 sequences cost almost nothing. A sliding-window model (Gemma) allocates its window per sequence, so keep it low
(12 on a 12 GB card). Hybrid models with recurrent layers work, but llama.cpp splits their batches per sequence
length, so branches run in several passes instead of one.

Set `"permutations": N` (default 1, capped at 8) to de-bias option order: pass 0 keeps the caller's
order and each later pass presents the same options in a distinct order seeded by the question id,
then the per-pass distributions are averaged by option key. Two passes cost about 1.1x and pull a
position-biased model toward the balanced answer; the noul second pass is the swap. The default
single pass is byte-identical to a request without the field.

Branches fork the cached prefix two ways: `copy` uses `llama_memory_seq_cp` (fast, plain attention), `restore` saves
and reloads a sequence state with `llama_state_seq_get/set_data` (works on recurrent and hybrid memory). The engine
picks one automatically; set `"fork"` on a `contexts`/`schema` request, or `LLAMA_DECISION_FORK=copy|restore|auto`,
to force it. On a sliding-window model the copy is clamped to the retained window so branch memory does not grow.

## POST /v1/decision

`contexts` is a list of 1-256 strings. They share one schema, one set of instructions, and one cached prefix; results
come back in the same order.

```bash
curl http://localhost:8096/v1/decision -H "Content-Type: application/json" -d '{
  "model": "gemma-4-12b",
  "instructions": "Answer each question about this support request from its state.",
  "schema": {
    "category": {"type": "enum", "choices": ["billing","technical","cancellation","other"],
                 "description": "What type of support request is this?"},
    "urgent":   {"type": "boolean", "description": "Does this need urgent handling?"},
    "priority": {"type": "enum", "choices": ["low","medium","high","critical"],
                 "description": "Rate support priority."}
  },
  "contexts": ["I was charged twice and need this fixed today."]
}'
```

```json
{
  "object": "decision",
  "results": [
    {
      "decision": {"category": "billing", "urgent": true, "priority": "high"},
      "fields": {
        "category": {"value": "billing",  "probability": 1.0,  "scored_nodes": 1, "tree": true},
        "urgent":   {"value": true,       "probability": 1.0,  "scored_nodes": 1, "tree": true},
        "priority": {"value": "high",     "probability": 0.74, "scored_nodes": 1, "tree": true}
      },
      "usage": {"context_tokens": 21, "scored_rows": 14}
    }
  ],
  "usage": {"prompt_tokens": 137, "cached_tokens": 116, "context_tokens": 21, "scored_rows": 14},
  "timings": {"prefill_ms": 50.7, "scoring_ms": 50.0, "total_ms": 100.7, "rounds": 1, "per_decision_ms": 100.7}
}
```

(That response is a real one: Gemma 4 12B on an RTX 3060, warm cache.)

### Schema

Compact fields, or a JSON Schema object with `properties`:

| type | keys | notes |
|---|---|---|
| `enum` | `choices` (or `enum`) | 1-255 values |
| `boolean` | - | true / false |
| `integer` | `minimum`, `maximum` | 1-255 values |
| `number` | `minimum`, `maximum`, `step` (`multipleOf` in JSON Schema) | fixed-width decimals |

Numeric fields take `aggregate`: `mode` (default), `median` or `mean`.

### Options

| field | default | meaning |
|---|---|---|
| `instructions` | `""` | prepended to the generated field catalogue; cached with it |
| `mode` | `auto` | `tree` scores every divergence node and returns exact probabilities; `greedy` walks the trie; `auto` picks tree up to `tree_max` values |
| `tree_max` | 128 | per-field switch between tree and greedy |
| `cache_prompt` | true | reuse the cached instructions + schema prefix |

## Jev decision shape (`state` + `questions`)

`POST /v1/decision` also accepts the Jev shape: one `state` and 1-256 typed `questions`
(`noul` yes/no, `choice` pick-one, `score` ordered rating). Answers come back as one closed
distribution per question, with `output_tokens` always 0:

```json
{"state": "...",
 "questions": {"refund": {"type": "noul", "instructions": "refund?"},
               "dept": {"type": "choice", "instructions": "route", "criteria": {"billing": "payment", "support": "help"}},
               "urgency": {"type": "score", "instructions": "urgency", "criteria": ["low", "medium", "high"]}}}
```

```json
{"answers": {"refund": {"type": "noul", "noul": 0.99},
             "dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "support": 0.1},
                      "confidence": 0.9, "certainty": 0.53},
             "urgency": {"type": "score", "score": 1.6, "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65},
                         "legend": {"0": "low", "1": "medium", "2": "high"},
                         "confidence": 0.65, "certainty": 0.31}}}
```

Both shapes are served by `POST /v1/decision`, the canonical route. `POST /decision` is a deprecated
alias for the same handler; use `/v1/decision`.

### Limits and errors

| limit | value |
|---|---|
| questions per request | 1-256 |
| options / levels per question | 2-64 |
| `contexts` per legacy request | 1-256 |
| request body | 2 MiB (override `LLAMA_DECISION_MAX_BODY`) |
| concurrent decision requests | 4 (override `LLAMA_DECISION_MAX_QUEUE`), then 429/529 |

Error responses use `{"error": {"code", "message", "type"}}` and map to HTTP status:

| status | type | when |
|---|---|---|
| 400 | `invalid_request_error` | malformed JSON, unknown field, bad `head` value, invalid type |
| 401 | `authentication_error` | missing or wrong API key |
| 413 | `payload_too_large` | body over the configured cap |
| 415 | `unsupported_media_type` | `Content-Type` is not `application/json` |
| 422 | `invalid_request_error` | valid JSON, invalid semantics (empty state, too many options, label/tokenizer mismatch, request past the decision context budget) |
| 429 | `rate_limit_error` | decision queue full; `Retry-After: 1` |
| 499 | `client_closed_request` | the client disconnected before the answer was ready |
| 500 | `server_error` | unexpected internal failure |
| 501 | `not_supported_error` | the running model cannot serve decisions (no usable labels, contract mismatch) |
| 529 | `overloaded_error` | server overloaded; `Retry-After: 1` |

The decision path never truncates: an over-limit request is rejected, never silently clipped.

### Prefix cache

The shared prefix (instructions, schema, state) is cached per request tag. A cached entry is stored
in the self-contained host format, because a device-format entry references a context staging buffer
that the next save reuses. The active request keeps a device-format copy for its own trunks and
branches, so a warm prefix restores device-to-device while a later save cannot corrupt an older
cache entry.

### Selected answer head

The letter readout can score answer rows from the model's output table instead of projecting the
whole vocabulary (`head: "selected"`, or `"auto"` to use it when available). It runs on a
classifier-only context that shares the weights and stops at the post-norm hidden state. An arch is
eligible only when its graph can stop there and its output table is a plain contiguous matrix
(`llm_arch_supports_classifier`). If the model also carries a per-id output bias, that bias must be
readable as a contiguous 1-D vector over the vocabulary; an unreadable bias makes the head
unavailable instead of silently scoring without it. A zero bias is kept and adds zero; it is not
treated as "no bias". `head: "auto"` falls back to full logits when the head is unavailable; an
explicit `head: "selected"` on an incompatible model is a 400.

### Usage and audit

`usage` reports `input_tokens` (state + cached prefix), `output_tokens` (always 0, nothing is
generated), `cached_tokens`, `state_cache_hit`, and `head_mode`. Every answer also carries additive
audit fields: `answer_token_ids`, `option_logits`, `allowed_token_mass`,
`full_vocab_argmax_id`, `prompt_sha256`, `prompt_version`, and `probability_status`. These are for
inspection only; they never change an answer. The response also carries `diagnostics.contract_hash`
(see below) and a `head` object describing the readout path.

### Contract hash and diagnostics

At first use the server computes a contract hash over the tokenizer identity, the framed prompt
template, the label code, and the prompt version, logs it, and returns it as
`diagnostics.contract_hash`. Pass `--decision-contract HASH` to pin it: if the running contract
differs, the decision path is refused with a plain 501 instead of serving stale calibration.

### Frozen backend flags

Every parity and calibration claim is only valid under the backend flag set recorded in
`tests/decision-baseline/calibration.json` (flash-attention setting, K/V cache types, `kv_unified`,
`swa_full`, `n_ubatch`, threads). Change any of them and re-run the calibration gate.

## Temperature and confidence

`temperature` (default 1.0) and per-type `temperatures` scale the label logits before the
softmax. Temperature never changes the winner; it only changes how sharply the distribution
is concentrated.

`confidence` is `max(p)` and `certainty` is `1 - H/log(K)`. Both measure how concentrated the
answer is. They are **not** calibrated correctness, and they are **not** accuracy. The
probabilities are conditional on the options you supplied: if the right answer is not among
them, the distribution still sums to 1 over the wrong set. Never gate admission, caching,
routing or persistence on `confidence` or `certainty`, and never present them as probability
of being correct.

Admission is not part of this axis. `--decision-ctx-size` bounds the classifier context by token
cells, and each request is accepted or rejected by its token footprint alone, never by
`confidence`/`certainty`. A low-confidence and a high-confidence request of the same length get the
same outcome.

A calibrated temperature is deployment-specific. `--decision-temperature FILE` loads a JSON
profile `{"temperatures": {"noul": ..., "choice": ..., "score": ...}, "provenance": {"model": ...,
"quantization": ..., "template_hash": ..., "backend_flags": ...}}`. If any temperature differs
from 1.0 and the recorded provenance does not match the running model, quantization, prompt
template and backend flags, the server refuses the profile instead of silently applying it.
With no file the default stays 1.0.

### Two readouts, one engine

The same engine serves two readouts. The `contexts`/`schema` shape scores arbitrary token paths
(trie or greedy) and is the general-purpose form. The Jev `state`/`questions` shape scores declared
answer labels and returns one closed distribution per typed question. Both share the prefix cache,
the branch scorer, the softmax and the SWA clamp; the letter readout is a thin layer over the trie
scorer, not a second implementation.

Letter labels are resolved at the framed answer boundary, not in isolation. A SentencePiece /
`add_space_prefix` vocabulary tokenizes a bare `A` as the space-prefixed form in isolation but
emits the bare token after the tail, so the pool is built from the tail and the letter readout
works on both SentencePiece and BPE tokenizers.

## Model card snippet

```yaml
model: <base gguf>
task: single-pass decision / classification over a supplied state
readout: letter labels resolved at the framed answer tail (SentencePiece and BPE), or token-path trie (schema)
context: shared prefix + one state per request; branches forked on a unified KV cache
output: probability distributions only, output_tokens always 0, closed over the supplied options
confidence: max(p) and 1 - H/log(K); concentration, NOT calibrated accuracy
calibration: deployment-specific; valid only under the recorded model, quantization, template hash and backend flags
```

## CLI

`llama-parallel-decision` runs the same engine from a worker process (stdin/stdout protocol, one JSON request per
line). Environment: `DECIDE_TREE`, `DECIDE_TREE_MAX`, `DECIDE_NSEQ`, `DECIDE_SPLIT_BOUNDARY`.

## A UI for it

[decision-playground](https://github.com/thecodacus/decision-playground) is a browser-only playground: it talks
straight to your llama-server, runs a decision and the same question as a chat completion side by side with live
timers, and has a small game whose agents decide through the endpoint.
