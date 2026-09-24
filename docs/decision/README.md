## What the /v1/decision endpoint is really trying to do:

At its core, this work adds a new kind of query to llama-server:
/v1/decision, which is not "generate text" but "pick an answer from a fixed
list." Instead of asking the model to produce a sentence, you hand it a JSON
object — a state (the situation being judged) and a set of questions, each
with a small finite set of allowed answers (true/false, a choice among
options, or a score level).  The model scores each allowed answer and
returns probabilities, e.g.  "this ticket is 92% a refund question."

The request has two mutually exclusive shapes (a body using both is a 422).
The primary "state + questions" shape carries a `state` and a `questions`
map; each question carries required `instructions` and is typed `noul`
(true/false), `choice` (pick one of 2-64 options), or `score` (pick a 0..K-1
level on a 2-10 scale), and the server scores one
next-token choice over verified single-token letter labels, sharing one
framed state prefix across all questions.  The legacy "contexts + schema"
shape supplies a JSON Schema and a list of contexts and walks the schema's
value trie in one batched pass.  A request may also ask for a permutation
de-bias pass (`permutations: N`), which shuffles option order and averages,
so no one fixed ordering biases the scores.

This is classification-style work — routing, triage, RAG ranking, structured
extraction — where you care about which option wins, not about fluent prose. 
The whole feature is an attempt to make that cheap and deterministic, in
three ways:

1.  One batched forward pass instead of many.  The engine lays out every
question, every candidate, and every branch (each candidate's unique tail
tokens) as separate sequences, decodes them all in a single batched
llama_decode, and reads one scored row per branch.  A decision costs roughly
one decode step, no matter how many options exist.

2.  A "classifier-only" context.  A special context flag makes the graph
stop right after the final normalization layer, skipping the giant
full-vocabulary projection (the logits matmul).  Instead, the server
dequantizes only the handful of answer rows it needs
(llama_model_classifier_rows) and scores candidates with a host-side dot
product against the hidden state.  This is the "answer-head fast path."

3.  Faster branch forking.  To evaluate many branches at once, the engine
needs many copies of the same "so far" context.  The branch makes
saving/restoring a sequence's state much cheaper: it keeps the state on the
GPU and stages the device-to-device copies on the backend's stream with a
single sync, instead of one synchronous copy (and one cudaStreamSynchronize)
per tensor.

Plus a lot of bookkeeping that makes this safe in production: a preflight
check that rejects requests that can't fit the context, admission control
(413/429/529), a contract hash that refuses to serve a decision if the
tokenizer/prompt-template identity changed, an optional calibrated
temperature file, and per-model caching of the answer-row table.

The `head` field selects the scoring path: `"selected"` (the answer-head fast
path below), `"full"` (full-vocabulary logits), or omitted for auto.  The
fast path is fallback-safe — an unavailable head silently falls back to full
logits — except an explicit `head: "selected"` on an incompatible model,
which is a client error.  Beyond 413/429/529, the endpoint returns 422 for a
semantic or capacity error, 499 when the client disconnects, and 501 when the
loaded model cannot serve decisions at all (for example its vocabulary has no
usable single-token answer labels, or a pinned contract does not match).

The optional temperature profile is loaded with `--decision-temperature
FILE`: it maps `noul`/`choice`/`score` temperatures to non-default values,
and is only honored when the file's recorded provenance (model, quantization,
template hash, backend flags) matches the running configuration — a stale
profile is a server configuration error, never silently applied.  The
contract identity can be pinned with `--decision-contract HASH`; a mismatch
refuses the decision path.  The default response is the strict Jev envelope
(`model`, `answers`, `usage` with only input/output tokens); passing
`"diagnostics": true` adds the `head` object, the `diagnostics` identity
(contract_hash, prompt_version, timings, and the provenance of the readout),
`certainty`, and the per-answer audit trail, so callers can see exactly how an
answer was produced without changing the answers themselves.  Because the
selected answer head reads only the K answer rows, the full-vocabulary audit
fields `allowed_token_mass` and `full_vocab_argmax_id` are omitted under
`head_mode: "selected"` rather than reported as placeholder values; the
answer-row fields stay available.  The classifier-only context can be bounded with
`--decision-ctx-size N`; a request whose peak KV use does not fit is rejected
with 422 before anything is decoded.

## How the KV cache interacts with /v1/decision

The KV cache is where the model stores what it has seen so far.  The
decision engine leans on it heavily, and the server forces the "unified" KV
cache when decisions are enabled (--decision-seqs N automatically sets
kv_unified = true).  In a unified cache, all live sequences draw from one
shared pool of cells, and — critically — the code's seq_cp can make a branch
share the same physical cells as its parent instead of copying them, as long
as they're in the same stream.  So the layout for a decision request looks
like this:

- A dedicated snapshot sequence holds the static prefix (system prompt +
chat template up to the question).  It persists across requests, so a repeat
request with the same prefix is a cache hit: the prefix is never re-decoded. 

- Each trunk sequence forks off the snapshot (shares its cells) and decodes
one context (the state) plus a common suffix head.

- Each branch sequence forks off its trunk and decodes only its own unique tail
  — the few tokens that distinguish one candidate answer from another — writing
  only those new cells into the shared pool.

- At the scored position, the engine reads one output row per branch, so the
  whole request is served by: one prefix decode (on first use), one decode per
  context, and one batched decode for the branch tails.

The classifier-only context used by the letter readout has its own separate
KV cache, entirely disjoint from the chat context's cache.  Because it has no
chat slots, its decision sequences start at zero; only the shared-context
paths (the legacy schema shape, and the letter readout's full-logits
fallback) reserve ids above the chat slots as described below.

## Is the KV cache updated by decision queries?

Yes — decisions are real decodes and they do write to the KV cache.  The
model's forward pass on the trunk context and on each branch tail populates
new KV cells, exactly like generation would.  But the branch is careful
about whose cache it touches and what survives:

- On the classifier-only context, decision KV lives in that dedicated
context and never touches chat.

  - On the shared chat context (the legacy path, or the letter readout's
    fallback), decisions write into the same unified pool as chat, but into
    reserved sequence ids above the chat slots (n_parallel ..  n_parallel +
    n_seq_decision), so chat's slots and their KV are never overwritten.

  - After scoring, the engine removes the branch and trunk sequences
    (llama_memory_seq_rm), reclaiming their cells.  Only the snapshot
    sequence's prefix survives, so the next matching query can reuse it.

  - A preflight check estimates peak KV use and returns 422 rather than ever
    partially overwriting the cache, and a cancelled request leaves the pool
    dirty but the next decision clears it before reuse.

So the short answer: decisions do update the KV cache, but transiently, in a
reserved range, with a separate cache on the fast path, and with
self-cleanup — the design's whole point is that a decision never disturbs
the state chat depends on.

## What the benchmarks say: /v1/decision versus chat

The reported warm numbers (GPU, ROCm 7900 XT, flash attention off for
bit-reproducibility) for the letter readout are:

- LFM2.5-350M ~70 ms, LFM2.5-2.6B ~108 ms, Qwen3.5-2B 118 ms, Qwen3.5-9B 252
  ms, Gemma-e4b 70 ms, Gemma-4-12B-QAT 129 ms.

Head-vs-full total variation ≤ 0.014, and winners always matched.

Read against chat, the key structural difference is cost per token.  A chat
completion is a loop: decode one token, sample it, feed it back, repeat —
the cost is proportional to the number of generated tokens (tens to
hundreds).  A decision is a single batched decode that produces every
candidate's score at once, plus a prefill for the context.  So a decision's
latency is roughly one token-generation step, and it does not scale with the
number of options or fields — all branches ride in the same batch.  That's
why even a 9B model answers in ~250 ms warm, while producing a 50-token chat
reply from the same model would take roughly an order of magnitude longer.

## Three optimizations drive the gap:

- the answer-head fast path avoids the full-vocabulary projection entirely
(the dot-product shortcut), which is where the big matmul cost lives in
normal generation;

- the async state restore turns per-tensor synchronous copies into one
staged stream drain (measured ~4x on this operation); and

- prefix caching removes the prefill cost on repeat calls.

One honest caveat documented with the numbers: these are warm timings with a
cached prefix, on a specific GPU, and with flash attention disabled on ROCm
because it is not bit-reproducible.  The low head-vs-full variation and
matching winners are the meaningful "is the fast path trustworthy?" signal —
and note that's an outcome-correctness check, separate from the determinism
caveats the test suite carries on weaker quantized models.
