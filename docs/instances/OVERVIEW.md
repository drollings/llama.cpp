# llama.cpp shared-weight instances: branch spec

This document describes THIS working tree (branch
`_multi_context_stripped_master`): one `llama-server` process loads a model's
weights once and serves any number of named context windows ("instances")
from them, each with its own `ctx_size`. It is written as a diff against the
`master` baseline; stock behavior is the reference for anything not called
out here. Read with `LLAMA_CPP_ADAPTERS_ON_DEMAND_OVERVIEW.md` (what/why),
`LLAMA_CPP_ADAPTERS_ON_DEMAND_ARCHITECTURE.md` (internals), and
`LLAMA_CPP_ADAPTERS_ON_DEMAND_API.md` (how to call it).

Status notes (verified against the implementation, 2026-09):

- Lazy windows: configured `--instance` entries are registered without
  allocating KV/compute; each window materializes on first demand for that
  instance (first dispatched request or explicit management op). Only the
  legacy no-`--instance` default is built eagerly.
- No auto-sleep / residency state machine exists in this branch (section 2).
  `--sleep-idle-seconds` with instances is refused at load and stripped per
  window (section 4.7).
- Per-instance LoRA adapters are loaded on demand from a pool-level registry
  (section 4.8): `POST/GET/DELETE /instances/:name/adapters` attach, list,
  and detach adapter sets per window, with shared weights loaded once and
  refcounted. Snapshot files stamp the adapter fingerprint, and restores
  reject a mismatched set.
- `tools/server/README.md` section "Using multiple instances" documents the
  same contract as this spec: `state` includes `"unloaded"`, snapshots are
  per-instance (`DIR/<model_key>/<instance>/<snapshot>.bin`), adapter routes
  and `adapter_bytes` are listed, and resize clears bindings. This spec is
  authoritative on any conflict.

`master` and `_swarm_api` are references only; this branch leaves both
untouched (it is a diff against `master`, not a change to it):

- `master` is the baseline. The change is a diff against it; stock behavior is
  the reference for anything this document does not call out.
- `_swarm_api` (the `server-instances` branch) is the reference implementation
  for what to port and what to deliberately leave behind.

Goal: simple, robust, minimal changes to llama.cpp. Coral Router owns
orchestration and VRAM policy; llama.cpp provides a thin serving primitive.

## 1. Responsibilities

llama-server (one process per model weights file):

- Load a model's weights exactly once.
- Allocate N named contexts ("instances") from those weights; each instance is
  an independent context window of exactly its own `ctx_size`, plus its own
  number of slots (`parallel`) that share that window (section 4.6).
- Serve OpenAI-compatible requests routed to a specific instance by model id.
- Expose the `/instances` management API: create, destroy, resize, pin
  (advisory), list (with memory breakdown), snapshot save/list/delete.
- Expose KV-cache snapshot save/restore to a slot-save path.

Explicitly NOT llama.cpp's job (all owned by Coral Router):

- Multi-model orchestration, spawning, process supervision, health of other
  servers. Coral Router is the deployed router; the llama.cpp router mode is
  never in the serving path.
- VRAM policy, eviction timing, load/unload decisions across models.

The llama.cpp router mode is kept for parity/debugging only. It aggregates
`GET /instances` across its children (concatenated instance rows, snapshots
tagged with their owning `model`, 64-bit-summed totals; `?model=<child>`
scopes to one child) so a multi-model deployment stays inspectable without
Coral Router. Management writes stay child-direct; the router never owns
VRAM policy.

## 2. Explicit non-goals (absent from THIS branch)

The following subsystems are deliberately absent here (some exist in the
_swarm_api reference; none were ported):

- **Router mode** (`server-models.{h,cpp}` child management) is kept in the
  binary but is never in the serving path: Coral Router spawns one
  `llama-server` per weights file and talks to each directly. This branch's
  router changes are read-only: a `GET /instances[?model=<child>]` aggregate
  (concatenated rows, snapshots tagged with owning `model`, 64-bit-summed
  totals, unreachable children skipped) for inspectability, plus a fix so a
  child running in instance mode reports READY without dereferencing the
  absent single-context metadata.
- **Auto-sleep / residency state machine**: no `sleep_idle_seconds` handling,
  no `SERVER_STATE_SLEEPING`/`READY` pairing, no `resident_count`, no
  residency worker, no `re_warm()`. Unloading is purely imperative
  (`DELETE /instances/:name`). `--sleep-idle-seconds` is refused outright:
  `load()` fails when instances are configured and sleep is enabled, and every
  built window's effective value is stripped to -1 (4.7). Run with sleep
  disabled.
- **Global `--parallel` as a default.** A declared instance's slot count is
  its own `parallel` value (default 1). Declared instances never read the
  global `--parallel` to size or divide a context; multiple contexts per
  single model are used, and parallel instances may be specified. Exception:
  with no `--instance` configured, the pool registers one legacy default that
  copies the global `--parallel`, preserving stock single-context startup.
- **`instance_guard`-adjacent drain complexity beyond the minimum.** The
  exclusive-access guard is: destroy/resize set a `removing` flag and wait
  for in-flight dispatches to drain (`instance_drain_guard`); direct
  instance-addressing handlers hold an `active_route_guard` count. Keep the
  flags and waits; nothing more.

Instance-mode endpoint semantics that are in scope and defined (not non-goals):

- `GET /metrics` renders the resolved target instance's own context (the default
  instance when untargeted; `404` for an unbuilt target; `400` for a GROUP or
  unknown target). A scrape never builds a window.
- The legacy `GET/POST /lora-adapters` targets the default instance in instance
  mode: `POST` requires a built default (`404` otherwise), refreshes that
  instance's resolved set so a later attach cannot revert the scale, and revokes
  its slot-snapshot bindings.
- A cold-reload adapter-registry mismatch aborts in debug builds and returns
  `507` in release builds.

## 3. Architecture

One process, one pool. The pool owns the shared weights and a vector of
instances; each instance owns a `server_context` (stock scheduler) and a
`server_routes` (stock routing table for that context).

```
llama-server (one per weights file)
  server_instances (the pool)
    model_init   : shared weights, loaded once (llama_init_from_model, model_only)
    instances[]  : each = { cfg, effective params, server_context, server_routes }
```

A request carrying a `model` field is resolved to an instance and dispatched
onto that instance's `server_context` via its `server_routes`. There is no
child process, no auto-sleep, no residency accounting.

### 3.1 Concurrency model

One process multiplexes any number of simultaneous requests. Concurrency comes
from the instances, never from extra processes:

- Each instance owns one scheduler thread (`server_context::start_loop`) and
  one task queue. N instances = N concurrent schedulers, each driving its own
  `llama_context`: on CPU they run in parallel across cores; on a GPU they
  share the device but are scheduled concurrently, one batch per context.
- Each instance's `parallel` slots add continuous batching WITHIN that one
  context (its single window), for cheap fan-out when requests do not need the
  full window (section 4.6).
- The two combine: a request lands on one instance, that instance's scheduler
  runs it (batched with its other slots' requests), and the other instances'
  schedulers run their own requests concurrently.

So four instances at `parallel` 1 serve four simultaneous full-window
conversations from one weight load; adding `parallel > 1` to an instance
serves more simultaneous short requests against that same window.

## 4. Core changes (diff against the `master` baseline)

Reference symbols are named after the _swarm_api branch; port the minimal
subset.

### 4.1 common: the instance grammar

`common/common.{h,cpp}` and `common/arg.cpp`:

- `struct common_instance { std::string name, group; int32_t ctx_size = 0;
  int32_t parallel = 0; bool pinned = false; bool is_default = false; }`.
  No `sleep_idle_seconds`, no `no_sleep`.
- `--instance "name[:group=G][:ctx=N][:parallel=M][:pinned][:default]"`,
  repeatable. `0` for `ctx`/`parallel` means "inherit"; `parallel` inherits to
  `1`, never to the global `--parallel`. The only path that copies the global
  `--parallel` is the legacy no-`--instance` default synthesized by the pool
  loader, which preserves stock startup.
- `common_instances_parse(spec)` splits on `,`, validates names/groups against
  `[A-Za-z0-9._-]`, rejects duplicates and name/group collisions
  (same logic as the _swarm_api `common_instance_validate`).
- `common_instance_params(base, inst)` builds the effective per-instance
  `common_params`: `n_ctx` from `ctx_size` (or the base default), `n_parallel`
  from `parallel` (default 1). It must not carry `instances` into the copy.

Also keep `--instance-wait SECONDS` (group wait, default 60) and
`--slot-save-path DIR`.

### 4.2 server-context: context from shared weights

`tools/server/server-context.{h,cpp}`. The one invasive change to shared code:

- Add a load path that takes an externally-owned model:
  `load_model(common_params & params, llama_model * shared_model = nullptr)`
  that calls `llama_new_context_with_model(shared_model, ...)` instead of
  loading weights itself. This is the _swarm_api branch's
  `server_context::load_model(params, model, mctx)` overload, kept minimal.
- Do not touch the stock single-context load path used when no instances are
  configured.
- The memory getters (`get_model_bytes/context/compute_bytes`) already exist
  in master; use them as-is.

### 4.3 server-instances: the pool manager

Port `tools/server/server-instances.{h,cpp}` in a stripped form:

- `struct server_instance { common_instance cfg; common_params effective;
  std::unique_ptr<server_context> ctx_server;
  std::unique_ptr<server_routes> routes; bool removing; int n_active_dispatch;
  bool running; bool built; bool adapter_failed;
  std::shared_ptr<common_init_result> model_owner; }`.
  `model_owner` is the window's shared reference to the pool weights (freed
  only when the last reference drops); `adapter_failed` distinguishes an
  adapter load failure (400) from an allocation failure (507) at build.
- `server_instances` owns `model_init`, `model`, `instances`, a dispatch
  mutex + condition variable, and the management handlers.
- `load()`: parse the pool identity from the first alias, load weights once, and
  register one entry per configured `--instance`. Registration is cheap: NO context
  window is allocated here. Each window materializes on first demand for that
  instance (first dispatched request, or an explicit management op), so declaring N
  instances never costs N windows upfront; one demand builds exactly one window.
  With NO instances configured, `load()` registers and eagerly builds a single
  default instance named `default` (`is_default = true`), so the pool is never
  left empty when driven directly. Note: in the server binary the pool is only
  engaged when at least one `--instance` is given (`use_instances`); with none,
  the stock single-context path runs and the pool is not used.
- `build_instance(cfg)`: effective params, the shared-model context
  (4.2), identity (`instance_id`, aliases), slot bookkeeping, `server_routes`,
  and the slot-release callback for group waits.
- `resolve(model_id, explicit_instance, error)`: the name-resolution grammar
  in 5. Exact instance-name match wins over group; no match -> "model or
  instance not found". A bare `<base>` / empty model id picks the default
  instance, or the sole instance, or 400 "ambiguous".
- `dispatch_group(group, ...)`: pick the member with an idle slot, then
  least-loaded, then least-recently-used by the monotonic clock (4.7),
  deterministic tie-break by instance
  order; wait up to `--instance-wait` when every member is busy (503 on
  expiry).
- Management handlers (see the full table in section 6): instance
  list/create/destroy/resize/pin/unpin, snapshot save/list/delete, and the
  per-instance adapter attach/list/detach routes (4.8).
- The shared weights are freed when the last instance is destroyed
  (`model_init.reset(); model = nullptr;`). The process stays alive; the next
  `POST /instances` (or dispatch to a cold pool) reloads the weights under the
  same mutex. No residency bookkeeping, no cold/warm flags.

`tools/server/server.cpp` wiring:

- Replace the single-context handlers with the manager's dispatch handlers
  when instances are configured (the _swarm_api `use_instances` swap). A child
  spawn mode is not needed (Coral Router spawns processes directly), so the
  `is_router_server`/`server_child` branches are removed.

### 4.4 snapshots

Port `tools/server/server-snapshot.{h,cpp}` and the request-time switch
semantics from the _swarm_api branch:

- `server_snapshot_write(path, data)` / `server_snapshot_read(path)` /
  `server_snapshot_read_status(path)` (`MISSING` vs `CORRUPT` vs `OK`) /
  `server_snapshot_list(dir)`. The `.bin` format wraps
  `llama_state_save`/`llama_state_load` plus magic, version, and `n_ctx_seq`.
- `--slot-save-path DIR`; files live under
  `DIR/<model_key>/<instance>/<snapshot>.bin`, `model_key` = sanitized base
  name (`/` and `:` replaced by `_`) plus a short deterministic hash of the
  original name, so distinct identities that sanitize alike never share a
  directory. New saves always write under the hashed key. Snapshots are per-instance: two
  instances saving the same name never share a file. Files written before
  per-instance scoping live flat at `DIR/<model_key>/<snapshot>.bin` (legacy
  layout, unhashed key); reads fall back to the unhashed scoped file and then
  to them, and a fresh save supersedes them.
- One FIFO I/O worker per pool (a single thread + bounded job deque) so
  snapshot file I/O never runs on a scheduler thread and a client disconnect
  cannot interrupt an in-flight write.
- Request-time restore: `snapshot=X` on a generation request binds/restores
  the target slot (`id_slot`, default 0) before serving. Repeated `snapshot=X`
  serves from the live KV; a different snapshot saves the KV back to the
  bound snapshot (if any) then loads the new one; an unbound working
   context is discarded. Missing -> `404`, corrupt -> `400`, timeout/busy ->
   `503`, no slot-save path -> `501`. Bindings are cleared by instance
  destroy, instance resize (which rebuilds the context and re-zeros the
  binding table), and per-snapshot delete (unbinds every slot bound to that
  name); ordinary generation requests never clear them.
- Switch budget (router tuning): compose deadline is 1 s + 1 s per 64 k of
  slot context; at most 2 concurrent switches per pool; the I/O queue holds
  at most 4 jobs (each queued write pins a full KV host buffer) and rejects
  with retriable `503` when full. Per-slot mutexes let switches on different
  slots of one instance run concurrently.

### 4.5 per-instance params and the --parallel non-assumption

Each instance's effective `common_params` come from `common_instance_params`:

- `ctx_size` maps to `n_ctx` (the instance's context window), defaulting to
  the model default when 0.
- `parallel` maps to `n_parallel`/`n_seq_max` (slots within the instance),
  defaulting to 1 when 0. The global `--parallel` is never read for this.

### 4.6 Context window vs slots: the exact mapping (load-bearing)

This is the rule every implementation must follow, stated explicitly so no
caller can get it wrong:

- An instance is ONE context window. `ctx_size` is the size of that window,
  allocated exactly. It is NEVER multiplied or divided by `parallel`.
- `parallel` is the number of continuous-batching slots that SHARE that one
  window. This is the stock llama.cpp mapping
  `n_seq_max = n_parallel` (`common/common.cpp:1639`) and
  `n_ctx_seq = n_ctx / n_seq_max` (`src/llama-context.cpp:290`). With
  `parallel = 4`, the window stays `ctx_size` and each slot may use at most
  `ctx_size / 4` of its KV. The KV buffer of an instance is sized for
  `ctx_size` total, never `ctx_size * parallel`.
- `parallel > 1` therefore does NOT create more windows and does NOT enlarge
  the window. To run N concurrent full-size conversations on one model,
  declare N instances (e.g. `count: N`), each with `parallel` 1. Do NOT set
  `parallel: N` on one instance and expect N full windows.
- `ctx_size` should be divisible by `parallel`; llama.cpp rounds a
  non-divisible `n_ctx` down to `n_ctx_seq * n_seq_max` with a warning
  (`src/llama-context.cpp:297-299`).

Worked example (Coral Router reference config):

- `swarm { count: 4, num_ctx: 8192 }` -> four independent 8192-token windows,
  each with `parallel` 1.
- `scratch { num_ctx: 131072 }` -> one 131072-token window.
- If `swarm { count: 1, num_ctx: 8192, parallel: 4 }` were set instead, the
  result is ONE 8192-token window shared by four slots (each slot capped at
  2048 tokens of KV), not four 8192 windows and not one 32768 window.

Neither 8192 nor 131072 is ever scaled by a `parallel` value. The configured
`ctx_size` is authoritative for the window; `parallel` only decides how many
sequences contend for it.

### 4.7 Sleep, idle timeouts, and idle tracking (read this before deploying)

There is no sleep state machine in this branch and no timeout is ever
reported:

- `state` is only `"loaded"` / `"unloaded"`. There is no `"sleeping"`, no
  per-instance idle countdown, and no "time remaining" field anywhere.
- Idle tracking is two fields, stamped together on slot release (i.e. last
  completion, not request arrival: in-flight work does not advance either
  field). Each is the max over the instance's slots, `-1` when the instance
  is unused or unbuilt:
  - `last_used`: monotonic microseconds (`ggml_time_us`). Ordering signal
    only: valid for comparing instances within one server process lifetime
    (group routing's least-recently-used pick uses it), never against a
    wall clock and never across processes or restarts.
  - `last_used_epoch`: unix epoch seconds (`system_clock`). The
    cross-process signal: Coral Router computes idle time as
    `now_epoch - last_used_epoch`. Second resolution is by design (it
    matches the snapshot `mtime` convention); rapid successive completions
    within one second are indistinguishable.
- Both fields reset to `-1` on `resize` (the context rebuild re-creates the
  slot vector) and vanish with the instance on `DELETE`. A router that
  resizes an instance must treat it as never-used for idle purposes until
  its next completion.
- `--sleep-idle-seconds` is refused with instances: `load()` fails when
  instances are configured and sleep is enabled, and every built window has
  its effective value stripped to -1 (the pool has no residency state machine
  and reports no sleeping state, so a per-instance sleep would park a
  scheduler with zero manager visibility). Idle reclamation is imperative
  `DELETE /instances/:name` from Coral Router.

### 4.8 LoRA adapters: per-instance, on demand

GGUF LoRA adapters attach per window through a pool-level registry keyed by
canonical path, loaded once against the shared model and refcounted:

- Each built instance resolves its set (`--instance ...:lora=PATH[:SCALE]`
  declarations plus the inherited base `--lora` set) into shared registry
  entries; one ref per built instance per entry, unbuilt windows hold none.
  The registry drains before the weights are freed, so no entry outlives the
  model.
- `POST /instances/:name/adapters {path, scale}` attaches (re-attach updates
  the scale with no refcount change), `GET` lists `[{path, scale}]`,
  `DELETE` detaches and frees at refcount 0. Unknown instance/adapter is 404;
  an unloadable file is 400. Attaching to an unbuilt window records the
  declaration without allocating; resolution happens at build.
- `GET/POST /lora-adapters` stay wired to the default instance only
  (legacy single-context contract, unchanged).
- Snapshots stamp the adapter fingerprint (`common_lora_fingerprint`, FNV-1a
  over path-sorted `path + raw scale bits`); restores under a different set
  are rejected, so KV is never served under the wrong tensors. Design detail
  lives in `LLAMA_CPP_ADAPTERS_ON_DEMAND_ARCHITECTURE.md`.

## 5. Request routing grammar

The llama-server (one pool) accepts these routing fields from body and query
(body wins): `model`, `instance`, `snapshot`, `id_slot`.

```
<base>                     default instance (or 400 "ambiguous" if none)
<base>:<instance>          exact instance
<base>:<group>             first available member of the group
<base>:latest              default instance
<base>:latest:<instance>   exact instance via the reserved pin
<base>:latest:<group>      first available member of the group via the reserved pin
```

`latest` is reserved: it is rejected as an instance name or group, and ids
with more than three `:`-separated components never resolve.

The `instance` field overrides any instance/group component of `model`.
`<base>` must equal the server's model name (the first alias), or be empty.
Since one process serves exactly one pool, there is no variant component.

Every generation endpoint carries these fields and is routed by instance:
`/v1/chat/completions`, `/v1/completions`, infill, embeddings, rerank,
transcribe, responses, Anthropic messages and token count.

### 5.1 Targeting a specific context window

The grammar above is how a request addresses a particular model and a
particular context window on it. A client has three spellings, all equivalent:

- `model: "<base>:<instance>"` - the exact instance, e.g.
  `model: "abiray/lfm2.5-2.6b-heretic-abliterated:ledger"`.
- `instance: "<instance>"` - overrides any instance/group in `model`, e.g.
  `{ "model": "abiray/lfm2.5-2.6b-heretic-abliterated", "instance": "ledger" }`.
- `id_slot: N` - a slot inside the resolved instance (snapshot switches and
  multi-turn session identity).

Which spelling to use depends on the role of the window:

- Swarm-style fan-out needs no instance awareness: `model` of the form
  `<base>` or `<base>:<group>` routes to the first available member of the
  group (group routing, any member).
- A large-context window such as `ledger` or `scratch` must be pinned exactly,
  by name, using one of the three spellings above. Group routing never picks
  a member implicitly for these; Coral Router addresses them by name.

### 5.2 Snapshot targeting

Snapshot work uses the same targeting, so save and restore always land on the
window the caller means:

- `POST /instances/ledger/snapshot { "name": "work" }` saves the `ledger`
  instance's slot-0 KV into its own snapshot namespace
  (`DIR/<model_key>/ledger/work.bin`, section 6); `{ "name": "work",
  "id_slot": 2 }` saves slot 2 instead (400 on an out-of-range slot).
- A later generation request restores it into that same window:
  `{ "model": "<base>", "instance": "ledger", "snapshot": "work" }`. The
   `snapshot` field applies to whatever instance the `model`/`instance` fields
   resolve to; it never selects the instance on its own.
 - `GET /instances/ledger/snapshots` and
   `DELETE /instances/ledger/snapshot/:snapshot` operate on that instance's
   window explicitly (section 6). A per-instance listing returns that
   instance's own files plus legacy flat files (tagged `"instance": null`).

The restore semantics themselves are in 4.4; this section only nails down
that routing and snapshot operations both reference one specific instance.

Endpoints with no model (`/tokenize`, `/detokenize`, `/apply-template`,
`/props`, `/health`, `/control`, LoRA) run on the default instance
(`default_instance()`: the `is_default` member, else the first registered).
NOTE the asymmetry: generation routing via `resolve()` is strict (bare
`<base>` with no default and several instances -> `400` "ambiguous"), while
these default-instance endpoints leniently fall back to the first instance.
`/props` adds `total_slots` and an `instances` array. `/v1/models` and
`/models` list one entry per instance (unbuilt instances appear with
`"status": "unloaded"` and no model data; no auto-generated suffixes).
`/slots` with no target aggregates every built instance's slots, each tagged
with its `"instance"` name; a targeted call runs on that instance (building
it on demand). Per-instance identity is
`instance_id = "<base>:<name>"` with aliases `{<base>, <base>:<group>,
<base>:latest, <base>:latest:<name>, <base>:latest:<group>}`.

## 6. The /instances management API

The contract mirrors the _swarm_api branch's child contract minus sleep fields.
See `CORAL_ROUTER_INSTANCES.md` for the public-facing version; this is the
same wire contract.

`GET /instances` envelope:

```json
{ "instances": [ { "id": "abiray/lfm2.5-2.6b-heretic-abliterated:ledger",
    "aliases": [...], "group": "ledger", "n_ctx": 65536, "parallel": 1,
    "pinned": true, "is_default": true, "state": "loaded",
    "model_bytes": ..., "context_bytes": ..., "compute_bytes": ...,
    "adapter_bytes": ..., "total_bytes": ..., "vram_bytes": ...,
    "last_used": -1, "last_used_epoch": -1 } ],
  "snapshots": [ { "name": "readfiles", "size": ..., "mtime": ...,
                   "n_ctx_seq": ..., "adapter_fp": ..., "version": ...,
                   "instance": "ledger" } ],
  "total": { "model": ..., "context": ..., "compute": ..., "adapter": ...,
             "total": ... } }
```

- `state` has only two values: `"loaded"` once the window exists, `"unloaded"`
  registered but never demanded (no KV/compute allocated, all byte fields zero);
  there is no sleeping state. A mere listing (`GET /instances`, `/v1/models`,
  `/slots` aggregate, `/health` on an unbuilt default) never materializes a window.
  Saving a snapshot to an unbuilt instance is `404` (a save must not build a
  window as a side effect); generation and slot/prop/model-targeted reads build
  on demand.
- `pinned` is stored and reported but never enforced (no auto-eviction exists).
- every built instance reports its own `model_bytes` (the shared weights, the
  same value on each built row); `total.model` counts the weights once. Summing
  per-row `total_bytes` therefore double-counts the weights - use `total` for
  pool-wide accounting.
- `adapter_bytes` counts the window's own resolved set (a shared entry counts
  toward every instance referencing it); `total.adapter` counts each registry
  file once. Per-instance `total_bytes` is `model + context + compute +
  adapter`.
- Snapshot rows carry `adapter_fp` ("" for v1 files / unreadable headers)
  and `version` (1 or 2 from a readable header, 0 when unreadable).
- `last_used` (monotonic us, in-process ordering only) and `last_used_epoch`
  (unix epoch seconds, for router idle math) are max-over-slots, `-1` when
  unused or unbuilt; see 4.7 for clocks, units, and reset semantics.
- A cold pool (all instances deleted, weights freed) returns `instances: []`,
  `total: 0`, and the `snapshots` array still populated. Note: deleting some
  but not all instances never frees the weights; cold requires zero remaining
  instances. Every live instance holds a `shared_ptr` copy of the model, so
  weights are freed only after the last reference (including transient
  in-flight handler references) drops.

Operations:

| Endpoint | Body | Response |
| --- | --- | --- |
| `GET /instances` | - | 200 envelope |
| `POST /instances` | `{name, group, ctx_size, parallel, pinned, default}` | 201 instance |
| `DELETE /instances/:name` | - | 200 `{success:true}`; last instance frees weights |
| `POST /instances/:name/resize` | `{ctx_size}` | 200 `{"n_ctx": <actual>}`; 507 leaves the instance unbuilt (a later demand retries at the new size) |
| `POST /instances/:name/pin` | - | 200 instance |
| `POST /instances/:name/unpin` | - | 200 instance |
| `POST /instances/:name/snapshot` | `{name, id_slot?}` | 201 `{id_slot, n_tokens, n_bytes}` |
| `GET /instances/:name/snapshots` | - | 200 `{snapshots:[...]}` |
| `DELETE /instances/:name/snapshot/:snapshot` | - | 200 `{success:true}` |
| `POST /instances/:name/adapters` | `{path, scale?}` | 200 instance (4.8) |
| `GET /instances/:name/adapters` | - | 200 `[{path, scale}]` |
| `DELETE /instances/:name/adapters` | `{path}` or `?path=` | 200 `{success:true}` |

Errors (OpenAI-shaped): `409` duplicate name/collision, `507` OOM, `404`
unknown, `400` invalid name/body or unloadable adapter, `501` snapshot
management without `--slot-save-path`, `503` group wait / apply timeout
(rolled back) / read-back verify timeout (applied - re-`GET` to confirm).
A demand-build with a bad declared adapter is `400` (unloadable file),
not `507`. `POST /instances`
on a cold pool reloads the weights first; a failed reload is `507` and nothing
is freed.

`DELETE /instances/:name` is the primary VRAM-return path. `?force=true` is
accepted for compatibility and ignored (nothing enforces `pinned`).

## 7. Concurrency and lifecycle rules

- One instance = one scheduler thread (`server_context::start_loop`),
  owned by the pool so instances can be created/destroyed/resized at runtime.
- Dispatch to an instance increments `n_active_dispatch` under the dispatch
  mutex; a management op that rebuilds or destroys the context
  (`DELETE`, `resize`) sets `removing = true`, waits for
  `n_active_dispatch == 0`, then tears down and joins the scheduler thread.
  New dispatches to a `removing` instance get a retriable error.
- Weight load/unload and instance create/destroy serialize on one pool mutex.
  There is no scheduler-thread-initiated teardown (no auto-sleep), so the
  _swarm_api branch's "never destroy a context from inside its own state
  callback" constraint does not apply: all teardown is HTTP-thread driven.

## 8. Derivation from the _swarm_api reference (historical)

This branch is a fresh implementation diffed against `master`; the
_swarm_api branch was consulted as a reference, not copied. The mapping
below records what came from where, for readers tracing a symbol:

Ported (adjusted to this branch's minimal shape):

- `common_instance`, `common_instances_parse`, `common_instance_validate`,
  `common_instance_params`, `--instance`/`--instance-wait`/`--slot-save-path`
  arg registration.
- `server_context::load_model(params, shared_model, ...)` shared-weight path
  and its use in `server_instances::build_instance`.
- `server_instances` pool manager: `load`, `build_instance`, `resolve`,
  `dispatch`, `dispatch_instance/group`, `pick_best_available`, identity and
  aliases, `get_instances_json`, `instance_to_json`, `pool_snapshots_json`.
- The management handlers: `handle_get/post_instances`,
  `handle_delete_instance`, `handle_post_instance_resize`,
  `handle_post_instance_pin/unpin`, `handle_post_instance_snapshot`,
  `handle_get_instance_snapshots`, `handle_delete_instance_snapshot`.
- `server-snapshot.{h,cpp}` and the pool I/O worker.
- Request-time snapshot switch (`apply_snapshot`) with the simplified
  single-threaded restore semantics of 4.4.
- From `tests/test-common-instances.cpp`: the parse/validation tests.
- From `tests/test-server-snapshot.cpp`: the format/status tests.
- From `tools/server/tests/unit/test_instances.py`: the create/destroy/
  resize/snapshot/resolve tests. DROP every test that exercises sleep,
  residency, cold/warm re-warm, or the router (they do not apply).

Written fresh here (no direct counterpart in the reference):

- The `server_instances` vector/drain lifecycle (simpler than the branch's
  `instance_guard`).
- Imperative weight free + reload on `DELETE`/`POST` (the branch's
  cold-unload/re-warm machinery is replaced by a plain mutex + null check).

Not ported from the reference (the branch builds on `master`'s own copies of
these instead; the router is kept for parity with a read-only aggregate
added, never in the serving path):

- The reference's `server-models` child-management machinery, `server-child`
  pipe protocol extensions, the instances refresher, `/models/sse`.
- Sleep/wake: `server_queue` sleep block, `sleep_idle_seconds`,
  `SERVER_STATE_SLEEPING`, the residency worker, `load_source`.
- `pinned` enforcement, `no_sleep`, `?force=` semantics.
- Variant (`base:variant`) resolution: one process is one weights file.

## 9. Testing

- `test-common-instances`: grammar (incl. strict numerics and the scale
  lookahead), defaults, validation, cross-flag duplicate/collision checks,
  the "no instances creates a default instance" behavior, name resolution
  with the `instance` override, and adapter parametrization/fingerprint.
- `test-server-snapshot`: write/read/list, MISSING vs CORRUPT, version
  plumbing, exact-size checks, short-write durability, and the v1/v2-empty
  fingerprint matrix.
- `test-server-instances-snapshots`: on-disk layout, the aggregate merge and
  its saturation/additive rules, the fan-out deadline skip, envelope shape.
- Server unit tests (`tools/server/tests/unit/test_instances.py`,
  `test_instances_aggregate_e2e.py`, `test_basic.py`): create/destroy,
  resize, `test_instances_snapshot_*`, two-pool listing, group routing and
  `--instance-wait`, default-instance resolution, the adapter attach/list/
  detach flow, the memory envelope, the slots aggregate markers, and the
  router aggregate over live children.
- Lazy-window coverage: configured instances list as `"unloaded"` with zero
  bytes until first demand; one demand builds exactly one window (siblings
  stay unloaded); group demand with no built member materializes one member.
- Manual (driven by Coral Router): spawn two servers, create instances of
  different `n_ctx`, delete the last instance and confirm `total.model`
  returns to 0, snapshot/restore across a process restart of Coral Router.

## 10. Divergences from the _swarm_api branch (summary)

| Area | _swarm_api reference | This branch |
| --- | --- | --- |
| Multi-model orchestration | llama.cpp router + children | Coral Router |
| Auto-sleep / residency | full state machine | none (imperative only) |
| Slot default | inherits `--parallel` | 1 for declared instances; legacy no-`--instance` default copies `--parallel` |
| Context vs slots | `--parallel` splits one shared window | `ctx_size` is the exact window; `parallel` slots share it; `count: N` creates N full windows (4.6) |
| `pinned` | eviction/sleep exemption | advisory flag, not enforced |
| Cold pool | `re_warm()` + residency worker | plain reload on demand |
| Snapshots | deadline/switch machinery | minimal FIFO worker + restore |
| Router observability | refresher, SSE, child cache | read-only aggregate only |

## 11. Performance trade-offs (what sharing weights does and does not buy)

Weights (the large, read-only `model_bytes`) are loaded once and shared: N
instances cost one model, not N. Everything else is per-instance and real:

- KV + compute buffers are allocated per window (`context_bytes` +
  `compute_bytes` per instance). Four 8192-token windows cost ~4x the KV of
  one. Lazy registration (section 4.3) avoids paying for windows nobody
  demands, but there is no free lunch for windows actually serving.
- No cross-instance batching. Each instance owns one scheduler thread and
  one task queue; each builds its own batch against the shared device.
  On CPU the schedulers run in parallel across cores; on GPU they share the
  device but submit independently, so small concurrent requests spread over
  instances utilize the GPU worse than one large `parallel` pool would.
  Rule of thumb: use instances for isolation (different `ctx_size`,
  different conversations that must not contend), `parallel` slots for cheap
  fan-out inside one window.
- Adapter buffers are shared through the pool registry (4.8): one load per
  file, refcounted across the built windows that reference it.
- Snapshot switches briefly occupy the target instance's scheduler (bounded
  GPU<->host copies) and are capped at 2 concurrent pool-wide; file I/O is
  off-scheduler on the pool worker, so generation on other instances is
  unaffected.
- Fixable without redesign: the `get_slot_info()` group-routing read is a
  lock-free scan of scheduler-owned slot state (best-effort/stale by
  design); the instance queue remains the authority and defers on a lost
  race, so a stale pick degrades to a short wait, never wrong execution.
  If stricter picks are needed, snapshot the busy/idle counters atomically.

## 12. Router cookbook: serving models, thinking levels, context sizes

Coral Router spawns one `llama-server` per weights file and maps its config
onto instances. Each bullet is one `POST /instances` (or one `--instance`
flag at boot):

- Various models: separate servers per weights file; the router translates
  the public model id to the owning server (`CORAL_ROUTER_INSTANCES.md`
  1.1). One process never serves two weights files.
- Thinking levels: convention, not a primitive. Run one instance per level
  (e.g. `think-low` with small `ctx_size`, `think-high` with large
  `ctx_size`, different system prompts / `n_predict` per request) and route
  by `<model>:<instance>`; or share one window and vary the request params.
- Context window sizes on demand: `POST /instances {name, ctx_size}` builds
  exactly that window from the already-loaded weights (no restart, no
  weight reload); `POST /instances/:name/resize {ctx_size}` rebuilds in
  place (KV and bindings dropped); `DELETE /instances/:name` returns the
  KV/compute immediately, and deleting the last instance unloads the
  weights (cold pool keeps snapshots listed for re-warm + `snapshot=X`
  restore).
- Fan-out vs isolation: `count: N` in Coral Router config creates N full
  windows (`<key>-0..N-1` in one group, group routing picks least-busy);
  `parallel: M` on one instance creates M slots sharing one window
  (section 4.6). Pick by whether the N conversations each need the full
  window.
- Group routing + wait: `<base>:<group>` dispatches to the member with an
  idle slot, then least-loaded, then LRU; when all members are busy the
  request waits up to `--instance-wait` (default 60 s, `-1` forever) for a
  slot-release wakeup, else `503` (retriable). Size groups so the wait is
  the exception, not the steady state.
