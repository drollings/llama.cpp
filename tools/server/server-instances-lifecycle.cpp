#include "server-instances.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>

#define IST_INF(fmt, ...) LOG_INF("inst %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define IST_WRN(fmt, ...) LOG_WRN("inst %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define IST_ERR(fmt, ...) LOG_ERR("inst %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define IST_DBG(fmt, ...) LOG_DBG("inst %12.*s: " fmt, 12, __func__, __VA_ARGS__)

// pool lifecycle: model/model_init, the instances vector, scheduler threads.
// lock contract: serialize on mutex_mgmt, take mutex_dispatch underneath
// (mgmt -> dispatch everywhere). never take mutex_mgmt under mutex_dispatch.

// size the per-slot snapshot locks: one heap-allocated mutex per slot so the vector
// survives resize() (std::mutex is not movable) and switching different slots of one
// instance do not serialize each other.
static void server_instance_size_slot_locks(std::vector<std::unique_ptr<std::mutex>> & locks, int n_parallel) {
    locks.clear();
    locks.reserve((size_t) n_parallel);
    for (int i = 0; i < n_parallel; ++i) {
        locks.push_back(std::make_unique<std::mutex>());
    }
}

// guardrail: a context window this large that nobody asked for explicitly almost
// always means the train-ctx default fired on a long-context model (e.g. a 9B model
// with 262144 train ctx silently reserving ~10 GiB of KV on a 21 GiB card). warn
// loudly, before the allocation, with the remedy. explicit sizes stay silent: the
// operator asked for those.
static constexpr int32_t HUGE_CTX_WARN_TOKENS = 32768;

bool server_should_warn_huge_implicit_ctx(int32_t effective_n_ctx, int32_t n_ctx_train) {
    return effective_n_ctx == 0 && n_ctx_train >= HUGE_CTX_WARN_TOKENS;
}

static void warn_if_huge_implicit_ctx(const llama_model * model, const common_params & effective, const char * instance_name) {
    if (effective.n_ctx != 0) {
        return; // explicit size: the answer needs no model data
    }
    if (model == nullptr) {
        return; // nothing to compare against: not the footgun
    }
    if (!server_should_warn_huge_implicit_ctx(effective.n_ctx, llama_model_n_ctx_train(model))) {
        return; // explicit size, or a train context too small to matter
    }
    const int32_t n_ctx_train = llama_model_n_ctx_train(model);
    // rough KV footprint for the message: layers * kv_heads * head_dim * K+V * cache
    // bytes. head_dim ~= n_embd / n_head holds for the GQA families this branch serves;
    // hybrid (SSM) state is extra, so this is a lower bound.
    const int64_t n_layer   = llama_model_n_layer(model);
    const int64_t n_head    = llama_model_n_head(model);
    const int64_t head_dim  = n_head > 0 ? llama_model_n_embd(model) / n_head : 0;
    const int64_t kv_elems  = (int64_t) n_ctx_train * n_layer * llama_model_n_head_kv(model) * head_dim * 2;
    const double  bytes_per = (double) ggml_type_size(effective.cache_type_k) / (double) ggml_blck_size(effective.cache_type_k);
    IST_WRN("instance '%s' has no explicit context size and inherits the full train context (%d tokens, KV cache alone ~%.1f GiB): "
            "pass --ctx-size or instance ctx= to avoid filling device memory\n",
            instance_name, n_ctx_train, (double) kv_elems * bytes_per / 1073741824.0);
}


instance_drain_guard::instance_drain_guard(server_instances & m, std::shared_ptr<server_instance> i)
    : mgr(m), inst(std::move(i)) {
    std::unique_lock<std::mutex> lock(mgr.mutex_dispatch);
    inst->removing = true;
    mgr.cond_dispatch.notify_all();
    mgr.cond_dispatch.wait(lock, [&]() { return inst->n_active_dispatch == 0; });
}

instance_drain_guard::~instance_drain_guard() {
    std::lock_guard<std::mutex> lock(mgr.mutex_dispatch);
    inst->removing = false;
    mgr.cond_dispatch.notify_all();
}


bool server_instances::load(const common_params & params) {
    this->params = params;

    // fail fast on duplicate names or name/group collisions accumulated
    // across repeated flags or built programmatically: registration below
    // assumes every name and group reference is unambiguous
    try {
        common_instances_validate_all(params.instances);
    } catch (const std::invalid_argument & e) {
        IST_ERR("invalid instance configuration: %s\n", e.what());
        return false;
    }
    // each instance inherits the base sleep value into its own scheduler loop
    // with no manager visibility (no sleeping state, no countdown), so refuse
    // the combination instead of sleeping silently. idle reclamation is
    // imperative DELETE /instances/:name.
    if (!params.instances.empty() && params.sleep_idle_seconds >= 0) {
        IST_ERR("%s", "instances do not support --sleep-idle-seconds (would sleep per-instance with no manager visibility); run with sleep disabled\n");
        return false;
    }

    // the pool identity is the first alias (or the model name or the file basename)
    if (!params.model_alias.empty()) {
        base_name = *params.model_alias.begin();
    } else if (!params.model.get_name().empty()) {
        base_name = params.model.get_name();
    } else {
        base_name = std::filesystem::path(params.model.path).filename().string();
    }
    IST_INF("pool identity: base = '%s'\n", base_name.c_str());

    // load the shared weights exactly once
    common_params model_params = params;
    model_init                 = common_init_from_params(model_params, true);
    model                      = model_init ? model_init->model() : nullptr;
    if (model == nullptr) {
        IST_ERR("failed to load model weights '%s'\n", params.model.path.c_str());
        return false;
    }
    IST_INF("loaded shared model weights '%s'\n", params.model.path.c_str());
    train_ctx_cached.store(llama_model_n_ctx_train(model), std::memory_order_relaxed);

    std::vector<common_instance> inst_cfgs = params.instances;
    if (inst_cfgs.empty()) {
        // single default instance, so a bare <base> request always has a target. it takes the
        // global --parallel/-c, preserving stock single-context startup behavior, because
        // common_instance_params never falls back to the base values for a declared instance.
        common_instance inst;
        inst.name       = "default";
        inst.group      = "default";
        inst.is_default = true;
        inst.parallel   = params.n_parallel;
        inst_cfgs.push_back(std::move(inst));
    }

    instances.reserve(inst_cfgs.size());

    // register every configured instance WITHOUT materializing it: each context window
    // (KV + compute) is allocated on first demand for that instance, so declaring N
    // instances never costs N windows upfront. the weights loaded above are the only
    // thing this function loads.
    const bool legacy_default = params.instances.empty();
    for (const auto & cfg : inst_cfgs) {
        auto inst       = std::make_shared<server_instance>();
        inst->cfg       = cfg;
        inst->effective = common_instance_params(params, cfg);

        if (inst->effective.n_parallel < 1) {
            IST_WRN("instance '%s' has no valid n_parallel, defaulting to 1\n", cfg.name.c_str());
            inst->effective.n_parallel = 1;
        }
        if (inst->effective.n_ctx == 0) {
            IST_INF("instance '%s' inherits the model's default context size\n", cfg.name.c_str());
        }

        // guardrail at registration: a huge implicit window is almost never intended,
        // and this is the first line an operator sees for it (before any demand builds it)
        warn_if_huge_implicit_ctx(model, inst->effective, cfg.name.c_str());

        // publish the display caches before the instance is reachable from the vector
        inst->n_ctx_effective.store(inst->effective.n_ctx, std::memory_order_relaxed);
        inst->n_parallel_effective.store(inst->effective.n_parallel, std::memory_order_relaxed);

        instances.push_back(std::move(inst));

        IST_INF("instance '%s' (group '%s', ctx = %d, parallel = %d%s%s) registered, builds on first demand\n",
                cfg.name.c_str(), cfg.group.c_str(), instances.back()->effective.n_ctx,
                instances.back()->effective.n_parallel, cfg.pinned ? ", pinned" : "",
                cfg.is_default ? ", default" : "");
    }

    // legacy drop-in: with no instances configured the single default instance is built
    // eagerly, preserving stock single-context startup behavior (exactly one window).
    if (legacy_default) {
        if (auto err = ensure_built_instance(instances.front())) {
            IST_ERR("failed to build default instance '%s'\n", instances.front()->cfg.name.c_str());
            return false;
        }
    }

    return true;
}


server_instances::server_instances(context_builder_fn builder) {
    set_context_builder(std::move(builder));
}


void server_instances::set_context_builder(context_builder_fn builder) {
    if (builder) {
        context_builder = std::move(builder);
    } else {
        context_builder = [this](server_instance & inst) { return build_context_default(inst); };
    }
}

void server_instances::start_instance_loop_locked(server_instance & inst) {
    // non-recursive mutex: try_lock fails exactly when the caller holds it
    assert(mutex_dispatch.try_lock() == false);
    if (inst.loop_started) {
        return;
    }
    inst.loop_thread = std::thread([&inst]() { inst.ctx_server->start_loop(); });
    inst.loop_started = true;
    // take thread ownership here, the one place a scheduler thread is created: false
    // means a thread is running and owes a join, and the teardown sites consume it
    inst.scheduler_joined = false;
}

bool server_instances::build_context_default(server_instance & inst) {
    // resolve the adapter set into pool-owned entries BEFORE the context load:
    // init_from_model skips (never re-loads) entries with a non-null ptr.
    // the registry access relies on the caller holding mutex_mgmt.
    auto resolved = resolve_adapter_set(inst.cfg);
    if (!resolved) {
        inst.adapter_failed = true;
        return false;
    }
    inst.effective.lora_adapters = std::move(*resolved);

    // shared ownership of the pool's weights: the context borrows `model`, so this copy
    // guarantees the model outlives the context (and any transient shared_ptr reference
    // to this instance held by an aggregate handler) even after the pool frees its copy
    inst.model_owner = model_init;

    // allocate only this instance's KV + compute buffers from the already-loaded weights
    inst.ctx_server = std::make_unique<server_context>();
    if (!inst.ctx_server->load_model(inst.effective, model)) {
        // drop the refs taken above and restore the declared set so a later
        // demand retries the same adapters (never a dangling ptr)
        release_adapter_set(inst.effective.lora_adapters);
        inst.effective.lora_adapters = common_instance_params(params, inst.cfg).lora_adapters;
        inst.ctx_server.reset();
        inst.model_owner.reset();
        return false;
    }

    apply_identity(inst);

    inst.slot_snapshots.resize((size_t) inst.effective.n_parallel);
    server_instance_size_slot_locks(inst.mutex_snapshot, inst.effective.n_parallel);

    inst.routes = std::make_unique<server_routes>(inst.effective, *inst.ctx_server);
    inst.routes->update_meta(*inst.ctx_server);

    // wake group waiters on the first 0->capacity transition (the wake
    // accounting lives in on_slot_released, shared with the test driver)
    inst.ctx_server->set_slot_release_callback([this](int) {
        on_slot_released();
    });

    return true;
}

bool server_instances::build_context_into(server_instance & inst) {
    inst.effective = common_instance_params(params, inst.cfg);
    inst.adapter_failed = false;

    // instances never sleep: the pool has no residency state machine and
    // reports no sleeping state, so an inherited idle timeout would park a
    // scheduler with zero manager visibility. load() already refuses the
    // combination at startup; strip it here too for runtime-created windows.
    inst.effective.sleep_idle_seconds = -1;

    if (inst.effective.n_parallel < 1) {
        IST_WRN("instance '%s' has no valid n_parallel, defaulting to 1\n", inst.cfg.name.c_str());
        inst.effective.n_parallel = 1;
    }

    if (inst.effective.n_ctx == 0) {
        IST_INF("instance '%s' inherits the model's default context size\n", inst.cfg.name.c_str());
    }

    // guardrail before the allocation: a huge implicit window is almost never intended
    warn_if_huge_implicit_ctx(model, inst.effective, inst.cfg.name.c_str());

    // publish the display caches for the unbuilt window (and again after a successful
    // build, where they match the context exactly); the build below is not observable
    // through `built` until the caller release-stores true
    inst.n_ctx_effective.store(inst.effective.n_ctx, std::memory_order_relaxed);
    inst.n_parallel_effective.store(inst.effective.n_parallel, std::memory_order_relaxed);

    return context_builder(inst);
}

void server_instances::teardown_instance_context(server_instance & inst) {
    // close the barrier first: a lock-free reader that acquire-loads false after
    // this point never touches ctx_server, and one that still sees true is held off
    // by the drain guard the caller holds (`removing`), which also waits out every
    // in-flight dispatch before the context is freed below
    inst.built.store(false, std::memory_order_release);
    // abort first so in-flight generations finish promptly; the drain guard the
    // caller holds then waits only for stragglers, never a never-ending decode.
    // this abort stays unbounded on purpose: the only caller (resize) runs a
    // bounded abort first and only reaches here once the scheduler proved
    // live, so this second abort is a fast-path no-op, not a stall risk.
    if (inst.ctx_server) {
        server_context_control_ops(*inst.ctx_server).abort_slots("instance '" + inst.cfg.name + "' resized");
        inst.ctx_server->terminate();
    }
    // never join under mutex_dispatch: a manager thread holding the dispatch
    // lock while the scheduler tears down would deadlock the teardown path
    bool do_join;
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        do_join               = !inst.scheduler_joined;
        inst.scheduler_joined = true;
    }
    if (do_join && inst.loop_thread.joinable()) {
        inst.loop_thread.join();
    }
    inst.ctx_server.reset();
    inst.routes.reset();
    inst.model_owner.reset();
    // drop this window's adapter refs while the model is still alive; the rebuild
    // re-resolves from cfg. restore the declared set so no dangling ptr survives
    // when the rebuild fails and the window stays unbuilt.
    release_adapter_set(inst.effective.lora_adapters);
    inst.effective.lora_adapters = common_instance_params(params, inst.cfg).lora_adapters;
    inst.slot_snapshots.clear();
    inst.mutex_snapshot.clear();
    inst.loop_started = false;
}

std::shared_ptr<server_instance> server_instances::build_instance(const common_instance & cfg, bool & adapter_failed) {
    auto inst = std::make_shared<server_instance>();
    inst->cfg = cfg;

    if (!build_context_into(*inst)) {
        adapter_failed = inst->adapter_failed;
        return nullptr;
    }

    adapter_failed = false;
    return inst;
}

// materialize one registered instance on first demand. the build itself runs under
// mutex_mgmt (lock order everywhere is mgmt -> dispatch), but the cold weight
// reload releases mgmt while the file loads (see cold_reload), so concurrent
// first demands for one instance collapse onto a single install and a reload
// can never wedge create/destroy/resize or race create_instance's reload. the
// built context is installed under mutex_dispatch; terminate() null-guards
// unbuilt instances and create_instance's shutdown check is mirrored here, so a
// build that loses a race with teardown frees what it allocated and reports
// unavailable instead of leaking a scheduler thread.
server_http_res_ptr server_instances::cold_reload() {
    // one flight at a time: a second concurrent demand waits here (not on
    // mutex_mgmt), then collapses onto the winner's install below instead of
    // loading the weights twice. management ops never take this mutex.
    std::lock_guard<std::mutex> reload_lock(mutex_reload);

    common_params model_params;
    {
        std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
        // reload the shared weights when the pool went cold (last instance destroyed)
        if (model != nullptr) {
            return nullptr;
        }
        // the registry drains at delete-last, so a cold pool never holds stale entries:
        // inherited adapters re-resolve from the config, not from old pointers. a
        // non-empty registry here means a refcount accounting bug. debug builds abort
        // loudly at the source; release builds must not take a live server down for an
        // internal accounting bug, so they log and refuse with a retriable 507.
        if (!adapter_registry.empty()) {
#ifndef NDEBUG
            GGML_ABORT("adapter registry not drained before cold reload");
#else
            IST_ERR("cold reload with %zu leaked adapter entries, refusing\n", adapter_registry.size());
            return make_error(507, "insufficient_memory_error", "adapter registry is inconsistent, restart the server");
#endif
        }
        model_params = params;
    }

    // seconds-long file load with NO lock held: management ops proceed during it
    auto fresh = common_init_from_params(model_params, true);
    if (fresh == nullptr || fresh->model() == nullptr) {
        IST_ERR("failed to reload model weights '%s'\n", model_params.model.path.c_str());
        return make_error(507, "insufficient_memory_error", "failed to reload shared weights");
    }

    {
        std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
        if (model != nullptr) {
            // a racing flight installed while we loaded: drop ours, weights ready
            IST_INF("%s", "cold reload collapsed onto a racing install, dropping duplicate load\n");
            return nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_dispatch);
            if (terminated) {
                return make_error(503, "unavailable_error", "server is shutting down");
            }
        }
        if (!adapter_registry.empty()) {
            // entries appeared mid-flight (refcount bug, see above): refuse rather
            // than install weights under stale adapter pointers
#ifndef NDEBUG
            GGML_ABORT("adapter registry not drained before cold reload");
#else
            IST_ERR("cold reload with %zu leaked adapter entries, refusing\n", adapter_registry.size());
            return make_error(507, "insufficient_memory_error", "adapter registry is inconsistent, restart the server");
#endif
        }
        model_init = std::move(fresh);
        model      = model_init ? model_init->model() : nullptr;
        if (model == nullptr) {
            IST_ERR("failed to reload model weights '%s'\n", model_params.model.path.c_str());
            return make_error(507, "insufficient_memory_error", "failed to reload shared weights");
        }
        IST_INF("reloaded shared model weights '%s'\n", model_params.model.path.c_str());
        train_ctx_cached.store(llama_model_n_ctx_train(model), std::memory_order_relaxed);
        cold_reload_count += (cold_reload_count < UINT64_MAX);
    }
    return nullptr;
}

void server_instances::drop_built_context_locked(server_instance & inst) {
    // the scheduler was never started, so there is no thread to join: drop the
    // fresh context with no thread to join, release the refs the build took
    // and restore the declared set for a later retry
    inst.routes.reset();
    inst.ctx_server->terminate();
    inst.ctx_server.reset();
    inst.model_owner.reset();
    release_adapter_set(inst.effective.lora_adapters);
    inst.effective.lora_adapters = common_instance_params(params, inst.cfg).lora_adapters;
}

server_http_res_ptr server_instances::ensure_built_instance(const std::shared_ptr<server_instance> & inst) {
    std::unique_lock<std::mutex> mgmt_lock(mutex_mgmt);

    // pre-build validation: still registered, not under teardown, and not
    // already built (a finished racing build collapses here, never rebuilds).
    // nullopt = ready to build; an engaged value (possibly null) returns directly.
    const auto validate_demand = [&]() -> std::optional<server_http_res_ptr> {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        if (inst->built.load(std::memory_order_acquire)) {
            // another demand built (or is building and finished first) while
            // this one waited: collapse onto it, never rebuild
            demand_build_collapsed += (demand_build_collapsed < UINT64_MAX);
            return server_http_res_ptr(nullptr);
        }
        bool registered = false;
        for (const auto & it : instances) {
            if (it == inst) {
                registered = true;
                break;
            }
        }
        if (!registered) {
            return make_error("instance '" + inst->cfg.name + "' no longer exists", ERROR_TYPE_NOT_FOUND);
        }
        if (inst->removing || !inst->running || terminated) {
            return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
        }
        return std::nullopt;
    };

    if (auto done = validate_demand()) {
        return std::move(*done);
    }

    if (model == nullptr) {
        // cold pool: the weight load releases mgmt (see cold_reload), so a
        // racing destroy can unregister us mid-flight: validate again below
        mgmt_lock.unlock();
        server_http_res_ptr err = cold_reload();
        mgmt_lock.lock();
        if (err) {
            return err;
        }
        if (auto done = validate_demand()) {
            return std::move(*done);
        }
    }

    // materialize the window: effective params are recomputed from cfg, then the
    // injected builder allocates (the default builder resolves the adapter set
    // into pool-owned entries first; a bad adapter path is a 400, never a 507).
    // expensive work runs with no dispatch lock held; the install below is gated
    // on a teardown race, mirroring create_instance's shutdown check.
    if (!build_context_into(*inst)) {
        if (inst->adapter_failed) {
            return make_error("failed to load lora adapter for instance '" + inst->cfg.name + "'", ERROR_TYPE_INVALID_REQUEST);
        }
        IST_ERR("failed to allocate instance '%s', shared model stays loaded\n", inst->cfg.name.c_str());
        return make_error(507, "insufficient_memory_error",
                          "failed to allocate instance '" + inst->cfg.name + "', not enough device memory");
    }

    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        if (inst->removing || !inst->running || terminated) {
            // lost a race with teardown after the build: drop the fresh context,
            // nothing is installed, no scheduler is started.
            drop_built_context_locked(*inst);
            return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
        }
        // start the scheduler only now that the instance is fully installed, so a
        // racing terminate() cannot leak the thread. the release-store publishes the
        // context writes above to any lock-free reader that acquire-loads built.
        start_instance_loop_locked(*inst);
        inst->built.store(true, std::memory_order_release);
        cond_dispatch.notify_all();  // wake group waiters so the new window can be picked
    }

    IST_INF("instance '%s' (group '%s', ctx = %d, parallel = %d) built on demand\n", inst->cfg.name.c_str(),
            inst->cfg.group.c_str(), inst->effective.n_ctx, inst->effective.n_parallel);
    return nullptr;
}


server_http_res_ptr server_instances::create_instance(const common_instance & cfg) {
    // management ops serialize on mutex_mgmt, so the check below stays
    // authoritative only while it is held: it re-runs after the unlocked
    // reload window, and the weight reload never races a last-instance unload
    std::unique_lock<std::mutex> mgmt_lock(mutex_mgmt);

    // duplicate check (see above): engaged = clash, return it directly
    const auto check_duplicate = [&]() -> server_http_res_ptr {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        for (const auto & inst : instances) {
            if (inst->cfg.name == cfg.name) {
                return make_error(409, "invalid_request_error", "instance already exists: '" + cfg.name + "'");
            }
            if (inst->cfg.group == cfg.name || inst->cfg.name == cfg.group) {
                return make_error(409, "invalid_request_error",
                                  "instance name collides with an existing group: '" + cfg.name + "'");
            }
        }
        return nullptr;
    };

    // reject duplicates before any expensive weight reload
    if (auto clash = check_duplicate()) {
        return clash;
    }

    if (model == nullptr) {
        // cold pool: the weight load releases mgmt (see ensure_built_instance),
        // so a racing create can take the name mid-flight: check again below
        mgmt_lock.unlock();
        server_http_res_ptr err = cold_reload();
        mgmt_lock.lock();
        if (err) {
            return err;
        }
        if (auto clash = check_duplicate()) {
            return clash;
        }
    }

    bool adapter_failed = false;
    auto inst = build_instance(cfg, adapter_failed);
    if (!inst) {
        if (adapter_failed) {
            return make_error("failed to load lora adapter for instance '" + cfg.name + "'", ERROR_TYPE_INVALID_REQUEST);
        }
        IST_ERR("failed to allocate instance '%s', shared model stays loaded\n", cfg.name.c_str());
        return make_error(507, "insufficient_memory_error",
                          "failed to allocate instance '" + cfg.name + "', not enough device memory");
    }

    IST_INF("creating instance '%s' (group '%s', ctx = %d, parallel = %d)\n", cfg.name.c_str(), cfg.group.c_str(),
            inst->effective.n_ctx, inst->effective.n_parallel);

    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        if (terminated) {
            // the server is shutting down mid-create: the scheduler was never
            // started, so drop the fresh context with no thread to join
            drop_built_context_locked(*inst);
            return make_error(503, "unavailable_error", "server is shutting down");
        }
        // explicitly created means explicitly demanded: push first, then start the
        // scheduler under the same lock, so terminate() cannot slip between them
        // and leak a joinable thread. release-store publishes the fresh context.
        inst->built.store(true, std::memory_order_release);
        instances.push_back(inst);
        start_instance_loop_locked(*inst);
        cond_dispatch.notify_all();  // wake group waiters so the new member can be picked
    }

    IST_INF("instance '%s' created at runtime\n", cfg.name.c_str());
    return make_ok(instance_to_json(*inst), 201);
}

server_http_res_ptr server_instances::destroy_instance(const std::string & name, bool) {
    // serialized with the other management ops so this can never race a resize or a
    // concurrent create/destroy of the same (or any) instance
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    // pinned is advisory in this branch; the force flag is accepted and ignored
    std::shared_ptr<server_instance> inst;
    size_t inst_pos = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        for (size_t i = 0; i < instances.size(); ++i) {
            if (instances[i]->cfg.name != name) {
                continue;
            }
            inst = instances[i];
            inst_pos = i;
            instances.erase(instances.begin() + i);
            // flip running under the same lock as the erase: any dispatch that incremented
            // before this point is drained by the guard below, any dispatch that checks
            // after sees running == false and never posts to the about-to-stop scheduler
            inst->running = false;
            cond_dispatch.notify_all();  // a group waiter must re-pick without this member
            break;
        }
    }

    if (!inst) {
        return make_error("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND);
    }

    // an unbuilt window never started a scheduler and owns no context: nothing to
    // abort, drain or join (no dispatch can be inside it: ensure_built_instance only
    // reports success once built is set)
    if (inst->built.load(std::memory_order_acquire)) {
        // abort in-flight generation, then drain the remaining dispatched requests so no
        // HTTP reader is left hanging when the scheduler is stopped below.
        // bounded: the management lock is held across this call, so a stalled
        // scheduler answers 503 instead of wedging every other management op
        // behind it. on timeout the unlink above is rolled back (same position,
        // running flipped back) so the instance is fully intact and a retry
        // finds it where it always was; without the rollback the last local
        // reference would tear down a live scheduler here.
        if (!server_context_control_ops(*inst->ctx_server).abort_slots("instance '" + name + "' evicted", snapshot_deadline_ms(*inst))) {
            {
                std::lock_guard<std::mutex> lock(mutex_dispatch);
                inst->running = true;
                instances.insert(instances.begin() + std::min(inst_pos, instances.size()), inst);
                cond_dispatch.notify_all();  // group waiters re-pick with this member
            }
            return make_error("timed out aborting in-flight work, retry", ERROR_TYPE_UNAVAILABLE);
        }

        {
            instance_drain_guard guard(*this, inst);
            inst->ctx_server->terminate();
            bool do_join;
            {
                std::lock_guard<std::mutex> lock(mutex_dispatch);
                do_join               = !inst->scheduler_joined;
                inst->scheduler_joined = true;
            }
            if (do_join && inst->loop_thread.joinable()) {
                inst->loop_thread.join();
            }
        }
    }

    // release this manager reference now; the weights are NOT freed here if any other
    // reference is still alive. each instance holds a shared copy of the pool model
    // (model_owner) that outlives its context, so the model is freed only when the last
    // reference -- including a transient shared_ptr held by an in-flight aggregate
    // handler -- actually drops, never while a context can still dereference it.
    // drop this instance's adapter refs first: entries at refcount 0 free here while
    // the model is still alive (an unbuilt window holds no refs; its null ptrs skip).
    release_adapter_set(inst->effective.lora_adapters);
    inst.reset();

    // delete-last: no live instances remain, so free the shared weights
    bool last_instance;
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        last_instance = instances.empty();
    }
    if (last_instance) {
        // drain the registry while the model is still alive (llama_adapter_lora_free
        // erases each adapter from model->loras). freeing the model first would leave
        // the registry unique_ptrs dangling. empty by construction (every instance
        // released its refs above), so this only guards against refcount bugs.
        adapter_registry.clear();
        model_init.reset();
        model = nullptr;
        IST_INF("pool '%s' has no instances left, shared weights unloaded\n", base_name.c_str());
    }

    IST_INF("instance '%s' destroyed\n", name.c_str());
    return make_ok({
        { "success", true }
    });
}

server_http_res_ptr server_instances::resize_instance(const std::string & name, int32_t new_ctx) {
    if (new_ctx <= 0) {
        return make_error("ctx_size must be positive", ERROR_TYPE_INVALID_REQUEST);
    }

    // serialized with the other management ops: a resize and a destroy of the same
    // instance can no longer race (destroy must never stop the queue under resize)
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    std::shared_ptr<server_instance> inst;
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        for (const auto & it : instances) {
            if (it->cfg.name == name) {
                inst = it;
                break;
            }
        }
        if (!inst) {
            return make_error("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND);
        }
        // an unbuilt window has no context to rebuild: record the new size, it applies
        // when the window materializes on first demand. the display cache follows the
        // declaration so a concurrent reader never sees the old size
        if (!inst->built.load(std::memory_order_acquire)) {
            inst->cfg.ctx_size = new_ctx;
            inst->effective     = common_instance_params(params, inst->cfg);
            inst->n_ctx_effective.store(inst->effective.n_ctx, std::memory_order_relaxed);
            inst->n_parallel_effective.store(inst->effective.n_parallel, std::memory_order_relaxed);
            IST_INF("instance '%s' resized to ctx = %d (applies on first demand)\n", name.c_str(), new_ctx);
            return make_ok(instance_to_json(*inst));
        }
    }

    // exclusive access while the old context is torn down and the new one is
    // built at the requested size. the guard rejects new dispatches and drains
    // in-flight ones; its destructor restores `removing` even when the rebuild
    // fails or throws.
    // abort in-flight generations before draining: the drain below would otherwise
    // wait forever on a never-ending decode (same order as destroy_instance).
    // bounded like destroy: a stalled scheduler answers 503 with the window
    // untouched instead of stalling the management plane.
    if (!server_context_control_ops(*inst->ctx_server).abort_slots("instance '" + name + "' resized", snapshot_deadline_ms(*inst))) {
        return make_error("timed out aborting in-flight work, retry", ERROR_TYPE_UNAVAILABLE);
    }
    {
        instance_drain_guard guard(*this, inst);
        // pin the post-resize adapter entries BEFORE the teardown drops the old
        // refs: the overlapping refs keep shared entries alive, so the rebuild
        // borrows them instead of reloading the files. a resolve failure answers
        // 400 with the window untouched (never a destructive teardown for a
        // predictable failure).
        const int32_t prev_ctx = inst->cfg.ctx_size;
        inst->cfg.ctx_size = new_ctx;
        auto pinned = resolve_adapter_set(inst->cfg);
        if (!pinned) {
            inst->cfg.ctx_size = prev_ctx;
            return make_error("failed to load lora adapter for instance '" + name + "'", ERROR_TYPE_INVALID_REQUEST);
        }
        teardown_instance_context(*inst);

        if (!build_context_into(*inst)) {
            // the rebuild failed: a well-defined unbuilt window at the new size.
            // a later demand retries the build instead of serving a dangling context.
            // teardown already released the old refs; a resolve failure took none.
            release_adapter_set(*pinned);
            IST_ERR("failed to resize instance '%s' to ctx = %d, leaving it unbuilt\n", name.c_str(), new_ctx);
            if (inst->adapter_failed) {
                return make_error("failed to load lora adapter for instance '" + name + "'", ERROR_TYPE_INVALID_REQUEST);
            }
            return make_error(507, "insufficient_memory_error",
                              "failed to resize instance '" + name + "', not enough device memory");
        }
        // the build resolved its own refs; drop the pins
        release_adapter_set(*pinned);

        // the window was rebuilt (bindings were dropped with the old context);
        // start the scheduler exactly once under the dispatch lock. the release-store
        // publishes the rebuilt context to lock-free readers.
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        start_instance_loop_locked(*inst);
        inst->built.store(true, std::memory_order_release);
        cond_dispatch.notify_all();
    }

    IST_INF("instance '%s' resized to ctx = %d\n", name.c_str(), new_ctx);
    return make_ok({ { "n_ctx", displayed_n_ctx(*inst) } });
}

server_http_res_ptr server_instances::set_instance_pinned(const std::string & name, bool pinned) {
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
    std::shared_ptr<server_instance> inst;
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        for (const auto & it : instances) {
            if (it->cfg.name == name) {
                inst = it;
                break;
            }
        }
        if (!inst) {
            return make_error("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND);
        }
        inst->cfg.pinned = pinned;
    }

    IST_INF("instance '%s' %s\n", name.c_str(), pinned ? "pinned" : "unpinned");
    return make_ok(instance_to_json(*inst));
}

//
// HTTP handlers
//
// endpoints carrying a `model` are dispatched to the owning instance; endpoints without
// one (tokenize, props, health, ...) run on the default instance. aggregate endpoints
// (/models, /slots) merge the per-instance state.


void server_instances::start_loops() {
    // idempotent and safe to call after demand-driven builds: only instances that
    // own a context and never started a loop get one. serialized on mutex_dispatch
    // so a racing demand build cannot interleave a second start (assigning over a
    // joinable thread would std::terminate).
    std::lock_guard<std::mutex> lock(mutex_dispatch);
    if (terminated) {
        return;
    }
    for (const auto & inst : instances) {
        // unbuilt windows have no scheduler yet; their loop starts on first demand
        if (!inst->ctx_server) {
            continue;
        }
        start_instance_loop_locked(*inst);
    }
    start_io_worker();
}

server_instances::~server_instances() {
    // last-resort shutdown on any exit path (e.g. an exception after start_loops): joins
    // every scheduler thread so no joinable thread survives pool destruction (a joinable
    // std::thread destructor would std::terminate the process). a no-op after a normal
    // terminate(), which the flag below makes one-shot.
    terminate();
}

void server_instances::terminate() {
    // one-shot: the signal handler, the main path, and the destructor may each call
    // terminate(); only the first call tears anything down. runs under mutex_dispatch so
    // no concurrent create_instance can slip a new instance past the teardown.
    std::vector<std::shared_ptr<server_instance>> live;
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        if (terminated) {
            return;
        }
        terminated = true;
        // reject any dispatch that arrives during teardown (group waiters re-pick and
        // find nothing)
        for (const auto & inst : instances) {
            inst->running = false;
        }
        cond_dispatch.notify_all();
        // snapshot so teardown never iterates a vector that a management op may mutate
        live = instances;
    }

    for (const auto & inst : live) {
        // unbuilt windows never started a scheduler and own no context
        if (inst->ctx_server) {
            inst->ctx_server->terminate();
        }
    }
    for (const auto & inst : live) {
        bool do_join;
        {
            std::lock_guard<std::mutex> lock(mutex_dispatch);
            do_join               = !inst->scheduler_joined;
            inst->scheduler_joined = true;
        }
        if (do_join && inst->loop_thread.joinable()) {
            inst->loop_thread.join();
        }
    }
    stop_io_worker();

    // drain the adapter registry under mutex_mgmt with NO dispatch lock held (the
    // joins above already ran lock-free): every scheduler is terminated, so no
    // params_base.lora_adapters raw ptr can still be dereferenced, and the pool's
    // model_init is still alive so llama_adapter_lora_free can erase from model->loras.
    {
        std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
        adapter_registry.clear();
    }
}

