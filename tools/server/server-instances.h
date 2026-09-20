#pragma once

#include "server-common.h"
#include "server-context.h"
#include "server-http.h"
#include "server-queue.h"
#include "server-snapshot.h"
#include "server-task.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

// pure firing rule for the huge-implicit-window guardrail: warn iff the window
// inherits its size (effective n_ctx == 0) and the model's train context
// reaches the threshold. tested directly (truth table); warn_if_huge_implicit_ctx
// applies it after the null-model check.
bool server_should_warn_huge_implicit_ctx(int32_t effective_n_ctx, int32_t n_ctx_train);

// one named context sharing a pool's loaded weights. owns the effective params
// (referenced by server_routes), the server_context, and the server_routes.
struct server_instance {
    common_instance                 cfg;
    common_params                   effective;
    // shared ownership of the pool's weights: the context borrows the raw model, so this
    // copy keeps the model alive until the instance (and its context) is gone. declared
    // before ctx_server so it is destroyed AFTER the context, guaranteeing the model
    // outlives every context that references it even when the pool is freed early.
    std::shared_ptr<common_init_result> model_owner;
    std::unique_ptr<server_context> ctx_server;
    std::unique_ptr<server_routes>  routes;

    // KV snapshot currently bound to each slot (index = slot id), empty = unbound.
    // guarded by server_instances::mutex_dispatch
    std::vector<std::string> slot_snapshots;

    // one lock per slot, so switching different slots of this instance do not
    // serialize each other. each element is heap-allocated so the vector survives
    // resize() (std::mutex is not movable).
    std::vector<std::unique_ptr<std::mutex>> mutex_snapshot;

    // scheduler thread for this instance; owned by the manager so instances can be
    // created / destroyed / resized at runtime
    std::thread loop_thread;

    // manager-internal lifecycle state, guarded by server_instances::mutex_dispatch:
    // removing = a management op owns the instance (new requests get a retriable error)
    // running  = the scheduler thread is alive; set false the moment destroy begins so a
    //            stale shared_ptr can never post a task to a terminated queue
    // n_active_dispatch = in-flight requests routed to this instance
    // built    = the context window (KV + compute), routes and scheduler exist. a
    //            registered instance starts unbuilt; the first demand for it
    //            materializes exactly one window (see ensure_built_instance).
    // loop_started = the scheduler thread was started exactly once, only by
    //            start_instance_loop_locked. built implies a context exists;
    //            loop_started implies its scheduler is (or was) running.
    bool removing          = false;
    bool running           = true;
    int  n_active_dispatch = 0;
    bool built             = false;
    bool loop_started      = false;
};

// manages the pool: one shared model load, many named contexts (instances).
// owns the shared weights, resolves requests by model id, routes them to the
// owning instance, and exposes the instance management API.
struct server_instances {
    // context construction seam: fills ctx_server + routes for an instance from
    // the shared model, true on success. the production default builds them for
    // real; tests may inject a stub to drive build-failure paths. runs on the
    // calling thread with no manager lock held.
    using context_builder_fn = std::function<bool(server_instance &)>;

    explicit server_instances(context_builder_fn builder = nullptr);

    // production context construction (the default builder): model ownership,
    // KV + compute alloc from the shared weights, identity, slot bookkeeping,
    // routes, and the slot-release callback. leaves null context/routes on failure.
    bool build_context_default(server_instance & inst);

    // test-only: replace the builder (nullptr restores the default). call only
    // when no build is in flight.
    void set_context_builder(context_builder_fn builder);

    // shared weights, loaded exactly once (model_only mode). shared_ptr so every
    // instance holds a copy (server_instance::model_owner); the weights are freed only
    // when the pool and every instance have released their copy, so a context that still
    // references the model can never outlive it.
    std::shared_ptr<common_init_result> model_init = nullptr;
    llama_model *                       model      = nullptr;

    common_params params;  // base params (global defaults + instances config)

    // owned through shared_ptr so an in-flight dispatch keeps the instance alive while a
    // management op (destroy/resize) removes it from this vector
    std::vector<std::shared_ptr<server_instance>> instances;

    // this pool's identity, parsed from the first alias (or the model name)
    std::string base_name;

    // dispatch mutex: guards slot_snapshots, the instances vector, and group waits
    mutable std::mutex      mutex_dispatch;
    std::condition_variable cond_dispatch;

    // management mutex: serializes create/destroy/resize/pin/snapshot ops so two
    // management calls can never race on the same instance (e.g. destroy + resize).
    // dispatches never take this mutex; a management op holds it for its whole body
    // and takes mutex_dispatch underneath. set once when the server shuts down.
    std::mutex mutex_mgmt;

    // the shared model's training context, cached on every weight (re)load. lets
    // reporting show the real size of an inherit-size window without touching the
    // model pointer off the management lock (see displayed_n_ctx).
    std::atomic<int32_t> train_ctx_cached{0};

    // window size shown in /instances, /props and /models: built windows report
    // their real size; unbuilt windows report the requested size, or the model
    // default when inheriting (0 when the weights are not loaded). lock-free.
    int32_t displayed_n_ctx(const server_instance & inst) const;

    // one-shot teardown flag, guarded by mutex_dispatch (see terminate())
    bool terminated = false;

    // load the shared model once and register one entry per configured instance.
    // with no instances configured, a single default instance is created AND built so the
    // manager can act as a drop-in replacement for the legacy single-context server.
    // configured instances are NOT built here: each context window materializes on
    // first demand for that instance (see ensure_built_instance), so declaring N
    // instances never costs N windows upfront.
    bool load(const common_params & params);

    // --- name resolution ---
    enum class target_kind { NONE, INSTANCE, GROUP };

    struct resolve_target {
        target_kind                      kind = target_kind::NONE;
        std::shared_ptr<server_instance> inst;   // INSTANCE target; kept alive for the caller
        std::string                      group;  // GROUP target
    };

    // routing input for one group member, derived from a single stats snapshot.
    // the fresh policy lives here: a member with no slots (unbuilt) is never
    // picked; a fresh built member (last_used_us == -1) sorts before any used
    // member among equally-busy candidates (cold-spread).
    struct route_candidate {
        size_t  index        = 0;  // position in the manager's instances vector (tie-break)
        int     n_slots      = 0;
        int     n_busy       = 0;
        int64_t last_used_us = -1; // max stamp over the member's slots, -1 = never used
    };

    // pure ordering over candidates: fewest busy slots, then least recently
    // used, then registration order. nullopt when no candidate has a free slot.
    static std::optional<size_t> pick_best_candidate(const std::vector<route_candidate> & members);

    resolve_target resolve(const std::string & model_id,
                           const std::string & explicit_instance,
                           std::string &       error) const;

    // route a request that carries model / instance / snapshot fields (body and/or query)
    // to the owning instance. `forward` runs on the chosen instance's server_routes.
    using forward_fn = std::function<server_http_res_ptr(server_routes &, const server_http_req &)>;
    server_http_res_ptr dispatch(const server_http_req & req, const forward_fn & forward);

    // --- management API ---
    server_http_res_ptr handle_get_instances(const server_http_req & req);
    server_http_res_ptr handle_post_instances(const server_http_req & req);
    server_http_res_ptr handle_post_instance_pin(const server_http_req & req);
    server_http_res_ptr handle_post_instance_unpin(const server_http_req & req);
    server_http_res_ptr handle_post_instance_resize(const server_http_req & req);
    server_http_res_ptr handle_delete_instance(const server_http_req & req);
    server_http_res_ptr handle_post_instance_snapshot(const server_http_req & req);
    server_http_res_ptr handle_get_instance_snapshots(const server_http_req & req);
    server_http_res_ptr handle_delete_instance_snapshot(const server_http_req & req);

    // --- HTTP handlers (wired by server.cpp, one per endpoint) ---
    server_http_res_ptr handle_get_health(const server_http_req & req);
    server_http_res_ptr handle_get_slots(const server_http_req & req);
    server_http_res_ptr handle_post_slots(const server_http_req & req);
    server_http_res_ptr handle_get_props(const server_http_req & req);
    server_http_res_ptr handle_post_props(const server_http_req & req);
    server_http_res_ptr handle_post_infill(const server_http_req & req);
    server_http_res_ptr handle_post_completions(const server_http_req & req);
    server_http_res_ptr handle_post_completions_oai(const server_http_req & req);
    server_http_res_ptr handle_post_chat_completions(const server_http_req & req);
    server_http_res_ptr handle_post_chat_completions_tok(const server_http_req & req);
    server_http_res_ptr handle_post_control(const server_http_req & req);
    server_http_res_ptr handle_post_responses_oai(const server_http_req & req);
    server_http_res_ptr handle_post_responses_tok_oai(const server_http_req & req);
    server_http_res_ptr handle_post_transcriptions_oai(const server_http_req & req);
    server_http_res_ptr handle_post_anthropic_messages(const server_http_req & req);
    server_http_res_ptr handle_post_anthropic_count_tokens(const server_http_req & req);
    server_http_res_ptr handle_post_apply_template(const server_http_req & req);
    server_http_res_ptr handle_get_models(const server_http_req & req);
    server_http_res_ptr handle_post_tokenize(const server_http_req & req);
    server_http_res_ptr handle_post_detokenize(const server_http_req & req);
    server_http_res_ptr handle_post_embeddings(const server_http_req & req);
    server_http_res_ptr handle_post_embeddings_oai(const server_http_req & req);
    server_http_res_ptr handle_post_rerank(const server_http_req & req);
    server_http_res_ptr handle_get_lora_adapters(const server_http_req & req);
    server_http_res_ptr handle_post_lora_adapters(const server_http_req & req);

    ~server_instances();  // safe shutdown on any exit path: joins every scheduler thread

    void start_loops();
    void terminate();

    // --- pool snapshot I/O worker lifecycle: file read/write never runs on a
    //     scheduler thread. start_loops() owns the single start together with
    //     the scheduler starts (M2 choke point); terminate() owns the single
    //     stop. both are idempotent. a post before start or after stop returns
    //     nullopt so callers fail fast with the existing retriable 503 instead
    //     of waiting on a future that will never complete.
    void start_io_worker();
    void stop_io_worker();
    // post a job to the single FIFO pool I/O worker. the queue is hard-bounded by
    // max_io_jobs (a queued write holds a full KV host buffer); returns nullopt when
    // the worker is not running or the queue is full so a caller can reject with a
    // retriable error instead of accumulating unbounded host memory.
    std::optional<std::future<void>> snapshot_io_post(std::function<void()> && fn);

    // per-instance snapshot paths under <slot_save_path>/<model_key>/. the
    // instance-scoped path (hashed key) is the only write target;
    // resolve_snapshot_path prefers it, then the previous-key scoped file,
    // then the legacy flat file for migration reads.
    std::string snapshot_instance_path(const std::string & instance, const std::string & snapshot) const;
    std::string snapshot_instance_path_prev(const std::string & instance, const std::string & snapshot) const;
    std::string snapshot_legacy_path(const std::string & snapshot) const;
    std::string resolve_snapshot_path(const std::string & instance, const std::string & snapshot) const;

    // RAII switch-semaphore guard; acquisition bounded by the compose deadline.
    // at most max_concurrent_switches (2) may be held pool-wide; further
    // acquisitions fail so callers answer the retriable 503.
    struct switch_guard {
        server_instances & mgr;
        bool               acquired = false;
        switch_guard(server_instances & m, int64_t deadline_ms);
        ~switch_guard();
    };

  private:
    // the only place a scheduler thread is constructed: starts the loop once per
    // built context, no-op afterwards. caller must hold mutex_dispatch (debug
    // assert); every teardown joins the thread before the instance is released,
    // so the captured reference never outlives the instance.
    void start_instance_loop_locked(server_instance & inst);

    // active context builder (the injected stub or the production default).
    context_builder_fn context_builder;

    // shared construction used by create, demand-build and resize: (re)computes
    // effective params from cfg, then materializes ctx_server + routes through
    // the injected builder. false = build failed, nothing installed.
    bool build_context_into(server_instance & inst);

    // manager-owned teardown: aborts in-flight work, stops the scheduler and
    // joins its thread, then releases the context, routes, weights ref and slot
    // bookkeeping, leaving a well-defined unbuilt instance. caller must hold
    // mutex_mgmt (and usually the drain guard) but NOT mutex_dispatch while
    // joining.
    void teardown_instance_context(server_instance & inst);

    resolve_target        resolve_instance_or_group(const std::string & target, std::string & error) const;
    std::optional<size_t> pick_best_available(const std::string & group) const;
    // materialize one registered instance on first demand: reloads the shared weights
    // when the pool went cold, allocates only this instance's context window from them,
    // and starts its scheduler. serializes on mutex_mgmt (lock order everywhere is
    // mgmt -> dispatch), so concurrent first demands for one instance collapse onto a
    // single build. call with NO locks held. nullptr = ready to serve.
    server_http_res_ptr ensure_built_instance(const std::shared_ptr<server_instance> & inst);
    // shared prologue of every default-instance endpoint: resolve the default,
    // build it on demand, guard against a racing destroy/resize, then run the
    // instance's own route handler. custom endpoints keep their own bodies.
    server_http_res_ptr default_instance_forward(const server_http_req & req,
                                                 server_http_context::handler_t server_routes::* method);
    server_http_res_ptr   dispatch_group(const server_http_req & req,
                                         const std::string &     group,
                                         const std::string &     snapshot,
                                         int                     id_slot,
                                         const forward_fn &      forward);
    server_http_res_ptr   dispatch_instance(const server_http_req &                  req,
                                            const std::shared_ptr<server_instance> & inst,
                                            const std::string &                      snapshot,
                                            int                                      id_slot,
                                            const forward_fn &                       forward);
    server_http_res_ptr   apply_snapshot(server_instance & inst, const std::string & snapshot, int id_slot);
    void                  clear_slot_binding(server_instance & inst, int id_slot);

    // --- two-phase snapshot compose ---
    // KV-size-scaled compose deadline: 1s floor + 1s per 64k context
    int64_t snapshot_deadline_ms(const server_instance & inst);
    // deadline-bounded file read on the pool I/O worker. busy = the I/O job queue was
    // full (a retriable 503, not a timeout); timed_out distinguishes a read that did not
    // finish in time (503) from one that completed and classified the
    // file (MISSING -> 404, CORRUPT -> 400, OK with data).
    struct server_snapshot_read_result {
        bool                                busy      = false;
        bool                                timed_out = false;
        server_snapshot_status              status    = server_snapshot_status::MISSING;
        std::optional<server_snapshot_data> data;
    };
    server_snapshot_read_result snapshot_io_read(const std::string & path, int64_t deadline_ms);
    // deadline-bounded snapshot file write on the pool I/O worker. busy = the I/O job
    // queue was full; timed_out distinguishes a write that did not finish in time (the
    // write still completes in the background) from one that completed and failed or
    // succeeded. bindings are only updated after a successful write, so a failure never
    // leaves a slot bound to a file whose content does not match the slot's KV.
    struct server_snapshot_write_result {
        bool busy      = false;
        bool timed_out = false;
        bool ok        = false;
    };
    server_snapshot_write_result snapshot_io_write(const std::string & path, server_snapshot_data data, int64_t deadline_ms);

    // management API internals
    // the single place an instance is constructed from the already-loaded shared
    // model (effective params, n_parallel clamp, context allocation, identity, slot
    // bookkeeping, routes, slot-release callback). returns nullptr on context
    // allocation failure; the caller owns registration in the pool and starting the
    // scheduler loop thread.
    std::shared_ptr<server_instance> build_instance(const common_instance & cfg);
    server_http_res_ptr              create_instance(const common_instance & cfg);
    server_http_res_ptr              destroy_instance(const std::string & name, bool force);
    server_http_res_ptr              resize_instance(const std::string & name, int32_t new_ctx);
    server_http_res_ptr              set_instance_pinned(const std::string & name, bool pinned);
    std::shared_ptr<server_instance> get_instance(const std::string & name) const;
    // copy of the instance list under mutex_dispatch, so an aggregate handler can iterate
    // without holding the lock across per-instance route calls
    std::vector<std::shared_ptr<server_instance>> snapshot_instances() const;
    json                             instance_to_json(const server_instance & inst) const;
    json                             instance_to_json(const server_instance & inst,
                                                      uint64_t                model_bytes,
                                                      uint64_t                context_bytes,
                                                      uint64_t                compute_bytes) const;
    // the full {"instances": [...], "snapshots": [...], "total": {...}} envelope; the
    // shared weights are counted once per pool
    json get_instances_json() const;
    // on-disk KV snapshots for this pool (name, size, mtime, n_ctx_seq); empty if no
    // slot-save-path. surfaced in the /instances envelope so a cold pool's snapshots
    // stay discoverable for reactivation.
    json                             pool_snapshots_json() const;
    // snapshots visible to one instance: its own instance-scoped directory plus
    // legacy flat files (migration read path), each tagged with "instance"
    // (the owning name, or null for legacy files).
    json                             instance_snapshots_json(const std::string & instance) const;
    std::string                      instance_id(const server_instance & inst) const;
    std::set<std::string>            instance_aliases(const std::string & name, const std::string & group) const;
    void                             apply_identity(server_instance & inst);
    std::shared_ptr<server_instance> default_instance();
    std::shared_ptr<server_instance> default_instance() const;

    server_http_res_ptr make_error(const std::string & message, error_type type) const;
    server_http_res_ptr make_error(const json & error) const;
    server_http_res_ptr make_error(int code, const std::string & type, const std::string & message) const;
    server_http_res_ptr make_ok(const json & data, int status = 200) const;

    // --- pool snapshot I/O worker: file read/write never runs on a scheduler thread.
    //     a bounded job queue decoupled from the HTTP-thread lifecycle, so a client
    //     disconnect can never interrupt an in-flight write.
    void                                   io_loop();
    std::mutex                             mutex_io;
    std::condition_variable                cond_io;
    // worker running state, guarded by mutex_io: false before the single start in
    // start_loops() and after the single stop in terminate(). posts while false
    // refuse fast (nullopt) instead of queueing onto a thread that will never run.
    bool                                   io_running = false;
    // hard-bounded job queue: each queued snapshot write holds a full KV host buffer, so
    // the queue is capped at max_io_jobs (rejects with a retriable 503 when full) instead
    // of accumulating unbounded host memory under slow-disk churn. 4 = two concurrent
    // switches (max_concurrent_switches) times one job in flight each, plus headroom.
    std::deque<std::packaged_task<void()>> io_jobs;
    std::thread                            io_thread;
    bool                                   io_stop = false;
    static constexpr size_t                max_io_jobs = 4;

    // per-pool cap on concurrent snapshot switches: bounds peak host-buffer memory
    std::mutex              mutex_switch;
    std::condition_variable cond_switch;
    size_t                  n_active_switches       = 0;
    static constexpr size_t max_concurrent_switches = 2;
};
