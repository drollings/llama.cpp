#include "server-instances.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <thread>

#define IST_INF(fmt, ...) LOG_INF("inst %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define IST_WRN(fmt, ...) LOG_WRN("inst %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define IST_ERR(fmt, ...) LOG_ERR("inst %12.*s: " fmt, 12, __func__, __VA_ARGS__)

// sanitize the pool identity into a directory name for snapshots: both '/' and ':'
// are replaced by '_' so the path is filesystem-safe
static std::string server_instance_model_key(const std::string & base_name) {
    std::string key = base_name;
    std::replace(key.begin(), key.end(), '/', '_');
    std::replace(key.begin(), key.end(), ':', '_');
    return key;
}

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

// RAII exclusive access for destroy/resize: set removing = true, wait for in-flight
// dispatches to drain, then restore the flag on scope exit.
struct instance_drain_guard {
    server_instances &              mgr;
    std::shared_ptr<server_instance> inst;

    instance_drain_guard(server_instances & m, std::shared_ptr<server_instance> i)
        : mgr(m), inst(std::move(i)) {
        std::unique_lock<std::mutex> lock(mgr.mutex_dispatch);
        inst->removing = true;
        mgr.cond_dispatch.notify_all();
        mgr.cond_dispatch.wait(lock, [&]() { return inst->n_active_dispatch == 0; });
    }

    ~instance_drain_guard() {
        std::lock_guard<std::mutex> lock(mgr.mutex_dispatch);
        inst->removing = false;
        mgr.cond_dispatch.notify_all();
    }
};

// RAII count of a direct route call on an instance (aggregate / management handlers
// that address an instance without going through dispatch()). destroy/resize drain on
// n_active_dispatch, so the guard keeps the scheduler alive until the route call
// finishes. the check-and-increment is atomic with the `running` flip in destroy, so a
// stale shared_ptr is rejected (acquired == false) instead of posting to a dead queue.
struct active_route_guard {
    server_instances & mgr;
    server_instance &  inst;
    bool               acquired = false;

    active_route_guard(server_instances & m, server_instance & i) : mgr(m), inst(i) {
        std::lock_guard<std::mutex> lock(mgr.mutex_dispatch);
        if (inst.removing || !inst.running) {
            return;
        }
        inst.n_active_dispatch++;
        acquired = true;
    }

    ~active_route_guard() {
        if (acquired) {
            std::lock_guard<std::mutex> lock(mgr.mutex_dispatch);
            inst.n_active_dispatch--;
            mgr.cond_dispatch.notify_all();
        }
    }
};

bool server_instances::load(const common_params & params) {
    this->params = params;

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

    std::vector<common_instance> inst_cfgs = params.instances;
    if (inst_cfgs.empty()) {
        // legacy drop-in: a single default instance. the pool is never left empty, so
        // a bare <base> request always has a target.
        common_instance inst;
        inst.name       = "default";
        inst.group      = "default";
        inst.is_default = true;
        inst_cfgs.push_back(std::move(inst));
    }

    instances.reserve(inst_cfgs.size());

    for (const auto & cfg : inst_cfgs) {
        auto inst = build_instance(cfg);
        if (!inst) {
            IST_ERR("failed to load instance '%s'\n", cfg.name.c_str());
            return false;
        }

        IST_INF("instance '%s' (group '%s', ctx = %d, parallel = %d%s%s) ready\n", cfg.name.c_str(), cfg.group.c_str(),
                inst->effective.n_ctx, inst->effective.n_parallel, cfg.pinned ? ", pinned" : "",
                cfg.is_default ? ", default" : "");

        instances.push_back(std::move(inst));
    }

    return true;
}

//
// name resolution
//

// caller must hold mutex_dispatch (the instances vector may be mutated by management ops)
server_instances::resolve_target server_instances::resolve_instance_or_group(const std::string & target,
                                                                             std::string &       error) const {
    resolve_target res;

    // exact instance name wins over group
    for (const auto & inst : instances) {
        if (inst->cfg.name == target) {
            res.kind = target_kind::INSTANCE;
            res.inst = inst;
            return res;
        }
    }
    for (const auto & inst : instances) {
        if (inst->cfg.group == target) {
            res.kind  = target_kind::GROUP;
            res.group = target;
            return res;
        }
    }

    error = "model or instance not found: '" + target + "'";
    return res;
}

server_instances::resolve_target server_instances::resolve(const std::string & model_id,
                                                           const std::string & explicit_instance,
                                                           std::string &       error) const {
    resolve_target              res;
    std::lock_guard<std::mutex> lock(mutex_dispatch);

    const auto pick_default = [&]() {
        for (const auto & inst : instances) {
            if (inst->cfg.is_default) {
                res.kind = target_kind::INSTANCE;
                res.inst = inst;
                return true;
            }
        }
        if (instances.size() == 1) {
            res.kind = target_kind::INSTANCE;
            res.inst = instances.front();
            return true;
        }
        return false;
    };

    // default instance for empty model id; an explicit instance field still takes precedence
    if (model_id.empty()) {
        if (!explicit_instance.empty()) {
            return resolve_instance_or_group(explicit_instance, error);
        }
        if (pick_default()) {
            return res;
        }
        error = "ambiguous: multiple instances and no default, specify one via 'instance' or the model id";
        return res;
    }

    const auto comps = string_split<std::string>(model_id, ':');

    if (comps[0] != base_name) {
        // this process serves exactly one pool; everything else belongs to another server
        error = "model not found on this child: '" + model_id + "'";
        return res;
    }

    // <base> alone: the default instance (or the sole instance, or ambiguous)
    if (comps.size() == 1) {
        if (pick_default()) {
            return res;
        }
        error = "ambiguous: multiple instances and no default for '" + model_id + "'";
        return res;
    }

    // <base>:latest -> the default instance
    if (comps.size() == 2) {
        std::string target;
        if (comps[1] == "latest") {
            if (pick_default()) {
                return res;
            }
            error = "ambiguous: multiple instances and no default for '" + model_id + "'";
            return res;
        }
        target = comps[1];

        // the explicit `instance` request field overrides the model's instance/group component
        if (!explicit_instance.empty()) {
            target = explicit_instance;
        }
        return resolve_instance_or_group(target, error);
    }

    error = "model not found: '" + model_id + "'";
    return res;
}

std::optional<size_t> server_instances::pick_best_available(const std::string & group) const {
    std::optional<size_t> best;
    int                   best_busy = INT32_MAX;
    int64_t               best_lru  = INT64_MAX;

    // caller must hold mutex_dispatch
    for (size_t i = 0; i < instances.size(); ++i) {
        const server_instance & inst = *instances[i];
        // removing = a management op owns the instance; running = false once a destroy or
        // a pool-wide shutdown has begun, so a group waiter never picks a dying instance
        if (inst.cfg.group != group || inst.removing || !inst.running) {
            continue;
        }

        const auto slots    = inst.ctx_server->get_slot_info();
        int        busy     = 0;
        bool       any_used = false;
        int64_t    lru      = INT64_MAX;
        for (const auto & slot : slots) {
            if (!slot.idle) {
                busy++;
            }
            if (slot.t_last_used >= 0) {
                any_used = true;
                lru      = std::min(lru, slot.t_last_used);
            }
        }
        if (!any_used) {
            lru = 0;  // a fresh, warm instance is the most recently used
        }

        // only members with at least one idle slot are candidates; when every member is
        // busy the caller waits for a slot release instead of dispatching onto a queue
        if (busy >= (int) slots.size()) {
            continue;
        }

        // prefer fewer busy slots, then least-recently-used; tie-break by instance order
        bool better = false;
        if (!best) {
            better = true;
        } else if (busy != best_busy) {
            better = busy < best_busy;
        } else {
            better = lru < best_lru;
        }

        if (better) {
            best      = i;
            best_busy = busy;
            best_lru  = lru;
        }
    }
    return best;
}

//
// request dispatch
//

server_http_res_ptr server_instances::dispatch(const server_http_req & req, const forward_fn & forward) {
    std::string model_id;
    std::string instance_field;
    std::string snapshot;
    int         id_slot = -1;

    try {
        json body = json::parse(req.body);
        if (body.is_object()) {
            model_id       = json_value(body, "model", std::string());
            instance_field = json_value(body, "instance", std::string());
            snapshot       = json_value(body, "snapshot", std::string());
            id_slot        = json_value(body, "id_slot", -1);
        }
    } catch (const std::exception &) {
        // a malformed body is reported by the instance's own handler
    }

    if (model_id.empty()) {
        model_id = req.get_param("model");
    }
    if (instance_field.empty()) {
        instance_field = req.get_param("instance");
    }
    if (snapshot.empty()) {
        snapshot = req.get_param("snapshot");
    }
    if (id_slot < 0) {
        const std::string & s = req.get_param("id_slot");
        if (!s.empty()) {
            try {
                id_slot = std::stoi(s);
            } catch (const std::exception &) {
            }
        }
    }

    std::string          error;
    const resolve_target target = resolve(model_id, instance_field, error);
    switch (target.kind) {
        case target_kind::INSTANCE:
            return dispatch_instance(req, target.inst, snapshot, id_slot, forward);
        case target_kind::GROUP:
            return dispatch_group(req, target.group, snapshot, id_slot, forward);
        case target_kind::NONE:
            return make_error(error, ERROR_TYPE_INVALID_REQUEST);
    }
    return make_error("unreachable", ERROR_TYPE_SERVER);
}

server_http_res_ptr server_instances::dispatch_instance(const server_http_req &                  req,
                                                        const std::shared_ptr<server_instance> & inst,
                                                        const std::string &                      snapshot,
                                                        int                                      id_slot,
                                                        const forward_fn &                       forward) {
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        inst->n_active_dispatch++;
        // check under the same lock that increments the count, so a destroy that starts
        // after this point must drain this dispatch before stopping the scheduler
        if (inst->removing || !inst->running) {
            inst->n_active_dispatch--;
            return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
        }
    }

    const auto release_dispatch = [this, &inst]() {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        inst->n_active_dispatch--;
        cond_dispatch.notify_all();
    };

    server_http_res_ptr err;
    if (!snapshot.empty()) {
        // a plain request leaves the slot (and any bound snapshot) untouched; only a
        // named snapshot can change the binding
        err = apply_snapshot(*inst, snapshot, id_slot);
    }
    if (err) {
        release_dispatch();
        return err;
    }

    try {
        server_http_res_ptr res = forward(*inst->routes, req);
        release_dispatch();
        return res;
    } catch (...) {
        // a handler must not throw (ex_wrapper guarantees it at the HTTP layer), but keep the
        // dispatch counter balanced anyway so a destroy is never blocked on a leaked count
        release_dispatch();
        throw;
    }
}

// group dispatch waits for a free member instead of dispatching onto a queue.
//
// NOTIFY COVERAGE: the wait is predicate-based; every 0->capacity transition that
// affects pick_best_available must notify cond_dispatch under mutex_dispatch. with the
// predicate wait the deadline is the only bound; a missed notify degrades to the 503
// deadline, never an indefinite hang. notify sources:
//   - slot-release callback (every instance, via the instance-creation helper)
//   - create_instance  after push
//   - destroy_instance after erase and after clearing `removing`
//   - resize_instance  after the removing toggles
//   - the drain guard on both set and clear of `removing`
//   - release_dispatch on every dispatch exit
server_http_res_ptr server_instances::dispatch_group(const server_http_req & req,
                                                     const std::string &     group,
                                                     const std::string &     snapshot,
                                                     int                     id_slot,
                                                     const forward_fn &      forward) {
    // a single steady-clock deadline computed once. "wait forever" is time_point::max(),
    // never INT64_MAX arithmetic (which would overflow to a negative deadline and 503
    // a busy-but-eventually-free group immediately).
    const bool wait_forever = params.instance_wait_seconds < 0;
    const auto deadline     = wait_forever
                                  ? std::chrono::steady_clock::time_point::max()
                                  : std::chrono::steady_clock::now() +
                                        std::chrono::seconds(params.instance_wait_seconds);

    while (true) {
        std::shared_ptr<server_instance> inst;
        {
            std::lock_guard<std::mutex> lock(mutex_dispatch);
            const auto                  best = pick_best_available(group);
            if (best) {
                inst = instances[*best];
                inst->n_active_dispatch++;
                // the removing/running check stays under the same lock as the count, so a
                // destroy that begins after this point must drain this dispatch first
                if (inst->removing || !inst->running) {
                    inst->n_active_dispatch--;
                    cond_dispatch.notify_all();
                    inst.reset();  // re-pick
                }
            }
        }

        if (inst) {
            const auto release_dispatch = [this, &inst]() {
                std::lock_guard<std::mutex> lock(mutex_dispatch);
                inst->n_active_dispatch--;
                cond_dispatch.notify_all();
            };

            server_http_res_ptr err;
            if (!snapshot.empty()) {
                // see dispatch_instance: a plain request never unbinds a slot
                err = apply_snapshot(*inst, snapshot, id_slot);
            }
            if (err) {
                release_dispatch();
                return err;
            }

            try {
                server_http_res_ptr res = forward(*inst->routes, req);
                release_dispatch();
                return res;
            } catch (...) {
                release_dispatch();
                throw;
            }
        }

        // predicate wait (no poll): re-picks under mutex_dispatch on every notify
        std::unique_lock<std::mutex> lock(mutex_dispatch);
        const bool                   ready = cond_dispatch.wait_until(lock, deadline, [&]() {
            return pick_best_available(group).has_value();
        });
        if (!ready) {
            return make_error("no free instance in group '" + group + "'", ERROR_TYPE_UNAVAILABLE);
        }
    }
}

void server_instances::clear_slot_binding(server_instance & inst, int id_slot) {
    if (id_slot < 0) {
        id_slot = 0;
    }
    std::lock_guard<std::mutex> lock(mutex_dispatch);
    if ((size_t) id_slot < inst.slot_snapshots.size()) {
        inst.slot_snapshots[id_slot].clear();
    }
}

//
// two-phase snapshot compose
//

int64_t server_instances::snapshot_deadline_ms(const server_instance & inst) {
    // 1s floor + 1s per 64k context, an approximation of the KV byte cost. the
    // transfer itself is bounded by the scheduler tasks; this bounds the whole switch.
    return ggml_time_ms() + 1000 + (int64_t) inst.ctx_server->get_slot_n_ctx() / 65536 * 1000;
}

server_instances::switch_guard::switch_guard(server_instances & m, int64_t deadline_ms) : mgr(m) {
    std::unique_lock<std::mutex> lock(mgr.mutex_switch);
    const int64_t                remain = deadline_ms - ggml_time_ms();
    if (remain <= 0) {
        return;
    }
    const bool ok = mgr.cond_switch.wait_for(lock, std::chrono::milliseconds(remain), [&]() {
        return mgr.n_active_switches < max_concurrent_switches;
    });
    if (ok) {
        mgr.n_active_switches++;
        acquired = true;
    }
}

server_instances::switch_guard::~switch_guard() {
    if (acquired) {
        std::lock_guard<std::mutex> lock(mgr.mutex_switch);
        mgr.n_active_switches--;
    }
    mgr.cond_switch.notify_all();
}

server_instances::server_snapshot_read_result server_instances::snapshot_io_read(const std::string & path,
                                                                                  int64_t             deadline_ms) {
    auto result = std::make_shared<std::optional<server_snapshot_read_out>>();
    auto future = snapshot_io_post([result, path]() {
        *result = server_snapshot_read_status(path);
    });
    if (!future) {
        // the I/O queue is full (hard-bound on queued host memory); a retriable 503
        server_snapshot_read_result busy;
        busy.busy = true;
        return busy;
    }
    const int64_t remain = deadline_ms - ggml_time_ms();
    if (remain <= 0 || future->wait_for(std::chrono::milliseconds(remain)) != std::future_status::ready) {
        server_snapshot_read_result timed_out;
        timed_out.timed_out = true;
        return timed_out;
    }
    future->get();
    server_snapshot_read_result done;
    done.status = (*result)->status;
    done.data   = std::move((*result)->data);
    return done;
}

server_instances::server_snapshot_write_result server_instances::snapshot_io_write(const std::string & path,
                                                                                    server_snapshot_data data,
                                                                                    int64_t             deadline_ms) {
    auto result = std::make_shared<bool>(false);
    auto future = snapshot_io_post([result, path, data = std::move(data)]() {
        *result = server_snapshot_write(path, data);
        if (!*result) {
            IST_WRN("failed to write snapshot '%s'\n", path.c_str());
        }
    });
    if (!future) {
        // the I/O queue is full; the slot must not be bound to a snapshot whose file was
        // never written
        server_snapshot_write_result busy;
        busy.busy = true;
        return busy;
    }
    const int64_t remain = deadline_ms - ggml_time_ms();
    if (remain <= 0 || future->wait_for(std::chrono::milliseconds(remain)) != std::future_status::ready) {
        // timed out: the write still completes in the background on the pool I/O worker;
        // the caller must not bind the slot to this snapshot, it may not exist on disk yet
        server_snapshot_write_result timed_out;
        timed_out.timed_out = true;
        return timed_out;
    }
    future->get();
    server_snapshot_write_result done;
    done.ok = *result;
    return done;
}

server_http_res_ptr server_instances::apply_snapshot(server_instance &   inst,
                                                     const std::string & snapshot,
                                                     int                 id_slot) {
    if (!fs_validate_filename(snapshot)) {
        return make_error("invalid snapshot name", ERROR_TYPE_INVALID_REQUEST);
    }
    if (params.slot_save_path.empty()) {
        return make_error("snapshot switching requires --slot-save-path", ERROR_TYPE_INVALID_REQUEST);
    }
    if (id_slot < 0) {
        id_slot = 0;
    }
    if ((size_t) id_slot >= inst.slot_snapshots.size()) {
        return make_error("invalid slot id", ERROR_TYPE_INVALID_REQUEST);
    }

    // per-instance resolve: the instance-scoped file wins, the legacy flat
    // file is the migration fallback ("" when neither exists, read as 404).
    const std::string filepath = resolve_snapshot_path(inst.cfg.name, snapshot);
    if (filepath.empty()) {
        return make_error(format_error_response("snapshot not found: '" + snapshot + "'", ERROR_TYPE_NOT_FOUND));
    }

    // 1. per-slot lock: switching different slots of one instance do not serialize
    std::lock_guard<std::mutex> slot_lock(*inst.mutex_snapshot[id_slot]);

    // 2. acquire the switch semaphore, bounded by the compose deadline
    const int64_t deadline_ms = snapshot_deadline_ms(inst);
    switch_guard  sw_guard(*this, deadline_ms);
    if (!sw_guard.acquired) {
        return make_error("too many concurrent snapshot switches, retry", ERROR_TYPE_UNAVAILABLE);
    }

    // 3. read the current binding under mutex_dispatch. a slot already bound to this
    //    snapshot serves immediately from the live KV.
    std::string current;
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        current = inst.slot_snapshots[id_slot];
        if (current == snapshot) {
            // already bound: continue the conversation from the live KV
            return nullptr;
        }
    }

    // 4. persist the slot's current KV (if any) before switching away from it. the write is
    //    fire-and-forget on the single FIFO pool I/O worker, which serializes all file ops.
    if (!current.empty()) {
        auto result = inst.ctx_server->slot_save_copy(id_slot, deadline_ms);
        if (!result) {
            return make_error("snapshot switch timed out while saving", ERROR_TYPE_UNAVAILABLE);
        }
        if (result->is_error()) {
            // a busy slot fails with a retriable 503 instead of deferring
            return make_error(result->to_json());
        }
        auto * copy = dynamic_cast<server_task_result_slot_copy *>(result.get());
        GGML_ASSERT(copy != nullptr);
        server_snapshot_data data;
        data.n_ctx_seq = inst.ctx_server->get_slot_n_ctx();
        data.tokens    = copy->tokens;
        data.kv        = std::move(copy->buffer);
        // save the old snapshot back where it was read from: the
        // instance-scoped file when present, else the legacy flat file it was
        // restored from (migration), else the instance-scoped path. the old
        // snapshot file must match the slot's KV before we switch away,
        // otherwise a later restore of `current` silently loses the
        // conversation that happened while it was bound. on failure the switch
        // is aborted and the slot keeps its live KV bound to `current`.
        std::string cur_path = resolve_snapshot_path(inst.cfg.name, current);
        if (cur_path.empty()) {
            cur_path = snapshot_instance_path(inst.cfg.name, current);
        }
        auto write = snapshot_io_write(cur_path, std::move(data), deadline_ms);
        if (write.busy) {
            return make_error("snapshot io worker is busy, retry", ERROR_TYPE_UNAVAILABLE);
        }
        if (write.timed_out) {
            return make_error("snapshot switch timed out while saving", ERROR_TYPE_UNAVAILABLE);
        }
        if (!write.ok) {
            return make_error("failed to save snapshot '" + current + "' to disk", ERROR_TYPE_SERVER);
        }
    }

    // 5. MISSING (404) vs CORRUPT (400) is decided by the snapshot module, not an
    //    HTTP-thread existence check (no TOCTOU with a concurrent fire-and-forget write).
    auto read = snapshot_io_read(filepath, deadline_ms);
    if (read.busy) {
        return make_error("snapshot io worker is busy, retry", ERROR_TYPE_UNAVAILABLE);
    }
    if (read.timed_out) {
        return make_error("snapshot read timed out", ERROR_TYPE_UNAVAILABLE);
    }
    if (read.status == server_snapshot_status::MISSING) {
        return make_error(format_error_response("snapshot not found: '" + snapshot + "'", ERROR_TYPE_NOT_FOUND));
    }
    if (read.status == server_snapshot_status::CORRUPT || !read.data) {
        return make_error(format_error_response("corrupt or unreadable snapshot: '" + snapshot + "'",
                                                ERROR_TYPE_INVALID_REQUEST));
    }

    // 6. a snapshot saved under a different context size is incompatible with this slot
    if (read.data->n_ctx_seq != inst.ctx_server->get_slot_n_ctx()) {
        return make_error(format_error_response(
            "snapshot context size does not match this slot", ERROR_TYPE_INVALID_REQUEST));
    }

    // 7. apply the host KV buffer on the scheduler; busy/timeout -> 503, apply failure
    //    clears the slot so it is never left partially loaded
    auto result = inst.ctx_server->slot_restore_apply(id_slot, std::move(read.data->kv), read.data->tokens, deadline_ms);
    if (!result) {
        clear_slot_binding(inst, id_slot);
        return make_error("snapshot switch timed out while restoring", ERROR_TYPE_UNAVAILABLE);
    }
    if (result->is_error()) {
        clear_slot_binding(inst, id_slot);
        return make_error(result->to_json());
    }

    // 8. bind on success
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        inst.slot_snapshots[id_slot] = snapshot;
    }
    return nullptr;
}

//
// management API
//

std::shared_ptr<server_instance> server_instances::get_instance(const std::string & name) const {
    std::lock_guard<std::mutex> lock(mutex_dispatch);
    for (const auto & inst : instances) {
        if (inst->cfg.name == name) {
            return inst;
        }
    }
    return nullptr;
}

std::vector<std::shared_ptr<server_instance>> server_instances::snapshot_instances() const {
    std::lock_guard<std::mutex> lock(mutex_dispatch);
    return instances;
}

std::string server_instances::instance_id(const server_instance & inst) const {
    return base_name + ":" + inst.cfg.name;
}

std::set<std::string> server_instances::instance_aliases(const std::string & name, const std::string & group) const {
    std::set<std::string> aliases;
    aliases.insert(base_name);
    aliases.insert(base_name + ":" + group);
    aliases.insert(base_name + ":latest");
    aliases.insert(base_name + ":latest:" + name);
    aliases.insert(base_name + ":latest:" + group);
    return aliases;
}

void server_instances::apply_identity(server_instance & inst) {
    inst.ctx_server->set_model_name(instance_id(inst));
    inst.ctx_server->set_model_aliases(instance_aliases(inst.cfg.name, inst.cfg.group));
}

// most recent slot use across an instance (max t_last_used over its slots), -1 when unused
static int64_t instance_last_used(const server_instance & inst) {
    int64_t t_last_used = -1;
    for (const auto & slot : inst.ctx_server->get_slot_info()) {
        if (slot.t_last_used >= 0) {
            t_last_used = std::max(t_last_used, slot.t_last_used);
        }
    }
    return t_last_used;
}

json server_instances::instance_to_json(const server_instance & inst) const {
    // all memory fields derive from the three instance getters (single source of truth)
    const uint64_t model_bytes   = inst.ctx_server->get_model_bytes();
    const uint64_t context_bytes = inst.ctx_server->get_context_bytes();
    const uint64_t compute_bytes = inst.ctx_server->get_compute_bytes();
    return instance_to_json(inst, model_bytes, context_bytes, compute_bytes);
}

json server_instances::instance_to_json(const server_instance & inst,
                                        uint64_t                model_bytes,
                                        uint64_t                context_bytes,
                                        uint64_t                compute_bytes) const {
    const int64_t t_last_used = instance_last_used(inst);

    return json{
        { "id", instance_id(inst) },
        { "aliases", instance_aliases(inst.cfg.name, inst.cfg.group) },
        { "group", inst.cfg.group },
        { "n_ctx", inst.effective.n_ctx },
        { "parallel", inst.effective.n_parallel },
        { "pinned", inst.cfg.pinned },
        { "is_default", inst.cfg.is_default },
        // this branch has no auto-sleep: an instance is always loaded
        { "state", "loaded" },
        // memory breakdown
        { "model_bytes", model_bytes },
        { "context_bytes", context_bytes },
        { "compute_bytes", compute_bytes },
        { "total_bytes", model_bytes + context_bytes + compute_bytes },
        // documented alias kept for the list() contract: context + compute
        { "vram_bytes", context_bytes + compute_bytes },
        { "last_used", t_last_used },
    };
}

json server_instances::get_instances_json() const {
    json     instances_arr = json::array();
    // 64-bit sums; the shared weights are counted once per pool
    uint64_t total_model   = 0;
    uint64_t total_context = 0;
    uint64_t total_compute = 0;
    bool     model_counted = false;

    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        for (const auto & inst : instances) {
            const uint64_t model_bytes   = inst->ctx_server->get_model_bytes();
            const uint64_t context_bytes = inst->ctx_server->get_context_bytes();
            const uint64_t compute_bytes = inst->ctx_server->get_compute_bytes();

            instances_arr.push_back(instance_to_json(*inst, model_bytes, context_bytes, compute_bytes));

            if (!model_counted && model_bytes > 0) {
                total_model += model_bytes;
                model_counted = true;
            }
            total_context += context_bytes;
            total_compute += compute_bytes;
        }
    }

    // snapshots are on-disk and independent of the live instances; list them so a cold
    // (weight-unloaded) pool's KV snapshots stay discoverable for reactivation
    json snapshots = pool_snapshots_json();

    return json{
        { "instances", std::move(instances_arr) },
        { "snapshots", std::move(snapshots) },
        { "total",
         {
              { "model", total_model },
              { "context", total_context },
              { "compute", total_compute },
              { "total", total_model + total_context + total_compute },
          }                                     },
    };
}

json server_instances::pool_snapshots_json() const {
    // null-safe string field read for ordering (common_json::value() throws
    // on a null, and legacy entries are explicitly tagged null)
    const auto str_field = [](const json & e, const std::string & key) -> std::string {
        if (!e.is_object() || !e.contains(key) || !e.at(key).is_string()) {
            return "";
        }
        return e.at(key).get<std::string>();
    };
    std::vector<json> entries;
    if (!params.slot_save_path.empty()) {
        const std::string model_key = server_instance_model_key(base_name);
        const std::string dir       = params.slot_save_path + model_key;
        const auto push = [&entries](const server_snapshot_meta & meta, const json & instance_tag) {
            entries.push_back({
                { "name",      meta.name      },
                { "size",      meta.size      },
                { "mtime",     meta.mtime     },
                { "n_ctx_seq", meta.n_ctx_seq },
                { "instance",  instance_tag   },
            });
        };
        // legacy flat files (pre-per-instance layout), tagged with a null instance
        for (const auto & meta : server_snapshot_list(dir)) {
            push(meta, nullptr);
        }
        // per-instance subdirectories, tagged with the owning instance name. a
        // snapshot outlives its instance (cold-pool discoverability), so every
        // subdirectory is listed, not just live instances.
        std::error_code          ec;
        std::vector<std::string> subdirs;
        std::filesystem::directory_iterator it(dir, ec);
        if (!ec) {
            for (const auto & entry : it) {
                if (ec) {
                    break;
                }
                if (entry.is_directory(ec)) {
                    subdirs.push_back(entry.path().filename().string());
                }
            }
        }
        std::sort(subdirs.begin(), subdirs.end());
        for (const auto & sub : subdirs) {
            for (const auto & meta : server_snapshot_list(dir + "/" + sub)) {
                push(meta, sub);
            }
        }
    }
    // deterministic envelope: order by (instance, name), legacy (null) first
    std::sort(entries.begin(), entries.end(), [&str_field](const json & a, const json & b) {
        const std::string ai = str_field(a, "instance");
        const std::string bi = str_field(b, "instance");
        return ai != bi ? ai < bi : str_field(a, "name") < str_field(b, "name");
    });
    json snapshots = json::array();
    for (auto & e : entries) {
        snapshots.push_back(std::move(e));
    }
    return snapshots;
}

// snapshots visible to one instance: its own instance-scoped directory plus
// the legacy flat files (migration read path, tagged with a null instance).
json server_instances::instance_snapshots_json(const std::string & instance) const {
    std::vector<json> entries;
    if (!params.slot_save_path.empty()) {
        const std::string model_key = server_instance_model_key(base_name);
        const std::string dir       = params.slot_save_path + model_key;
        for (const auto & meta : server_snapshot_list(dir + "/" + instance)) {
            entries.push_back({
                { "name",      meta.name      },
                { "size",      meta.size      },
                { "mtime",     meta.mtime     },
                { "n_ctx_seq", meta.n_ctx_seq },
                { "instance",  instance       },
            });
        }
        for (const auto & meta : server_snapshot_list(dir)) {
            entries.push_back({
                { "name",      meta.name      },
                { "size",      meta.size      },
                { "mtime",     meta.mtime     },
                { "n_ctx_seq", meta.n_ctx_seq },
                { "instance",  nullptr        },
            });
        }
    }
    std::stable_sort(entries.begin(), entries.end(), [](const json & a, const json & b) {
        const auto str_field = [](const json & e) -> std::string {
            if (!e.is_object() || !e.contains("name") || !e.at("name").is_string()) {
                return "";
            }
            return e.at("name").get<std::string>();
        };
        return str_field(a) < str_field(b);
    });
    json snapshots = json::array();
    for (auto & e : entries) {
        snapshots.push_back(std::move(e));
    }
    return snapshots;
}

std::string server_instances::snapshot_instance_path(const std::string & instance,
                                                     const std::string & snapshot) const {
    return server_snapshot_instance_path(params.slot_save_path, server_instance_model_key(base_name),
                                         instance, snapshot);
}

std::string server_instances::snapshot_legacy_path(const std::string & snapshot) const {
    return server_snapshot_legacy_path(params.slot_save_path, server_instance_model_key(base_name), snapshot);
}

// resolve a snapshot for read (and save-back): the instance-scoped file wins;
// the legacy flat file is the migration fallback. "" when neither exists.
std::string server_instances::resolve_snapshot_path(const std::string & instance,
                                                    const std::string & snapshot) const {
    std::error_code ec;
    const std::string inst_path = snapshot_instance_path(instance, snapshot);
    if (std::filesystem::exists(inst_path, ec)) {
        return inst_path;
    }
    const std::string leg_path = snapshot_legacy_path(snapshot);
    if (std::filesystem::exists(leg_path, ec)) {
        return leg_path;
    }
    return "";
}

std::shared_ptr<server_instance> server_instances::build_instance(const common_instance & cfg) {
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

    // shared ownership of the pool's weights: the context borrows `model`, so this copy
    // guarantees the model outlives the context (and any transient shared_ptr reference
    // to this instance held by an aggregate handler) even after the pool frees its copy
    inst->model_owner = model_init;

    // allocate only this instance's KV + compute buffers from the already-loaded weights
    inst->ctx_server = std::make_unique<server_context>();
    if (!inst->ctx_server->load_model(inst->effective, model)) {
        return nullptr;
    }

    apply_identity(*inst);

    inst->slot_snapshots.resize((size_t) inst->effective.n_parallel);
    server_instance_size_slot_locks(inst->mutex_snapshot, inst->effective.n_parallel);

    inst->routes = std::make_unique<server_routes>(inst->effective, *inst->ctx_server);
    inst->routes->update_meta(*inst->ctx_server);

    // wake group waiters on the first 0->capacity transition
    inst->ctx_server->set_slot_release_callback([this](int) {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        cond_dispatch.notify_all();
    });

    return inst;
}

server_http_res_ptr server_instances::create_instance(const common_instance & cfg) {
    // management ops serialize on mutex_mgmt, so the duplicate check below stays
    // authoritative through the push at the bottom (no concurrent create/destroy can
    // interleave) and the weight reload never races a last-instance unload
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    // reject duplicates before any expensive weight reload
    {
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
    }

    // reload the shared weights first when the pool went cold (last instance destroyed)
    if (model == nullptr) {
        common_params model_params = params;
        model_init                 = common_init_from_params(model_params, true);
        model                      = model_init ? model_init->model() : nullptr;
        if (model == nullptr) {
            IST_ERR("failed to reload model weights '%s'\n", params.model.path.c_str());
            return make_error(507, "insufficient_memory_error", "failed to reload shared weights");
        }
        IST_INF("reloaded shared model weights '%s'\n", params.model.path.c_str());
    }

    auto inst = build_instance(cfg);
    if (!inst) {
        IST_ERR("failed to allocate instance '%s', shared model stays loaded\n", cfg.name.c_str());
        return make_error(507, "insufficient_memory_error",
                          "failed to allocate instance '" + cfg.name + "', not enough device memory");
    }

    IST_INF("creating instance '%s' (group '%s', ctx = %d, parallel = %d)\n", cfg.name.c_str(), cfg.group.c_str(),
            inst->effective.n_ctx, inst->effective.n_parallel);

    // start the scheduler BEFORE registering the instance: a terminate() that runs
    // after the push (but before the thread was created) would otherwise leak an
    // un-joined thread. the terminated check under the same lock closes that race.
    inst->loop_thread = std::thread([inst]() { inst->ctx_server->start_loop(); });

    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        if (terminated) {
            // the server is shutting down mid-create: stop the scheduler we just started
            // and join it so nothing is left joinable when the pool is destroyed
            inst->ctx_server->terminate();
            if (inst->loop_thread.joinable()) {
                inst->loop_thread.join();
            }
            return make_error(503, "unavailable_error", "server is shutting down");
        }
        instances.push_back(inst);
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
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        for (auto it = instances.begin(); it != instances.end(); ++it) {
            if ((*it)->cfg.name != name) {
                continue;
            }
            inst = *it;
            instances.erase(it);
            // flip running under the same lock as the erase: any dispatch that incremented
            // before this point is drained by the guard below, any dispatch that checks
            // after sees running == false and never posts to the about-to-stop scheduler
            inst->running = false;
            cond_dispatch.notify_all();  // a group waiter must re-pick without this member
            break;
        }
    }

    if (!inst) {
        return make_error(format_error_response("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND));
    }

    // abort in-flight generation, then drain the remaining dispatched requests so no
    // HTTP reader is left hanging when the scheduler is stopped below
    inst->ctx_server->abort_slots("instance '" + name + "' evicted");

    {
        instance_drain_guard guard(*this, inst);
        inst->ctx_server->terminate();
        if (inst->loop_thread.joinable()) {
            inst->loop_thread.join();
        }
    }

    // release this manager reference now; the weights are NOT freed here if any other
    // reference is still alive. each instance holds a shared copy of the pool model
    // (model_owner) that outlives its context, so the model is freed only when the last
    // reference -- including a transient shared_ptr held by an in-flight aggregate
    // handler -- actually drops, never while a context can still dereference it.
    inst.reset();

    // delete-last: no live instances remain, so free the shared weights
    bool last_instance;
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        last_instance = instances.empty();
    }
    if (last_instance) {
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
            return make_error(format_error_response("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND));
        }
    }

    // exclusive access while the context is rebuilt and its bookkeeping refreshed. the
    // guard rejects new dispatches and drains in-flight ones; its destructor restores
    // `removing` even when resize fails or throws.
    server_task_result_ptr result;
    {
        instance_drain_guard guard(*this, inst);
        result = inst->ctx_server->resize(new_ctx);
        if (!result || result->is_error()) {
            return make_error(result ? result->to_json() :
                                       format_error_response("failed to resize instance", ERROR_TYPE_SERVER));
        }

        // the context was rebuilt; refresh identity, routes meta and snapshot bookkeeping
        inst->effective.n_ctx = new_ctx;
        apply_identity(*inst);
        inst->routes->update_meta(*inst->ctx_server);
        inst->slot_snapshots.assign((size_t) inst->effective.n_parallel, std::string());
        server_instance_size_slot_locks(inst->mutex_snapshot, inst->effective.n_parallel);
    }

    IST_INF("instance '%s' resized to ctx = %d\n", name.c_str(), new_ctx);
    return make_ok(instance_to_json(*inst));
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
            return make_error(format_error_response("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND));
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
//

server_http_res_ptr server_instances::handle_get_health(const server_http_req & req) {
    auto inst = default_instance();
    if (!inst) {
        // a cold/empty pool is healthy but has nothing loaded; a liveness probe must be
        // able to distinguish cold from dead
        return make_ok({
            { "status",    "ok"      },
            { "instances", 0         },
        });
    }
    // every remaining instance is loaded by construction; a destroyed instance is removed
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    return inst->routes->get_health(req);
}

server_http_res_ptr server_instances::handle_get_slots(const server_http_req & req) {
    std::string model_id       = req.get_param("model");
    std::string instance_field = req.get_param("instance");

    // no target: aggregate the slots of every instance, tagged with the instance name
    if (model_id.empty() && instance_field.empty()) {
        json all_slots = json::array();
        for (const auto & inst : snapshot_instances()) {
            active_route_guard guard(*this, *inst);
            if (!guard.acquired) {
                continue;  // being destroyed/resized; skip it
            }
            auto res = inst->routes->get_slots(req);
            if (res->status != 200) {
                return res;
            }
            for (auto & slot : json::parse(res->data)) {
                slot["instance"] = inst->cfg.name;
                all_slots.push_back(std::move(slot));
            }
        }
        return make_ok(all_slots);
    }

    std::string          error;
    const resolve_target target = resolve(model_id, instance_field, error);
    if (target.kind != target_kind::INSTANCE) {
        return make_error(error.empty() ? "invalid instance for slots" : error, ERROR_TYPE_INVALID_REQUEST);
    }
    active_route_guard guard(*this, *target.inst);
    if (!guard.acquired) {
        return make_error("instance '" + target.inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    return target.inst->routes->get_slots(req);
}

server_http_res_ptr server_instances::handle_post_slots(const server_http_req & req) {
    // resolve the target instance from an optional model/instance field in the body,
    // defaulting to the default instance
    std::string model_id;
    std::string instance_field;
    try {
        json body = json::parse(req.body);
        if (body.is_object()) {
            model_id       = json_value(body, "model", std::string());
            instance_field = json_value(body, "instance", std::string());
        }
    } catch (const std::exception &) {
        // a malformed body is reported by the instance's own handler
    }
    if (model_id.empty()) {
        model_id = req.get_param("model");
    }
    if (instance_field.empty()) {
        instance_field = req.get_param("instance");
    }

    std::string          error;
    const resolve_target target = resolve(model_id, instance_field, error);
    if (target.kind != target_kind::INSTANCE) {
        return make_error(error.empty() ? "invalid instance for slot action" : error, ERROR_TYPE_INVALID_REQUEST);
    }
    active_route_guard guard(*this, *target.inst);
    if (!guard.acquired) {
        return make_error("instance '" + target.inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    return target.inst->routes->post_slots(req);
}

server_http_res_ptr server_instances::handle_get_props(const server_http_req & req) {
    auto inst = default_instance();
    if (!inst) {
        return make_error("no instances loaded", ERROR_TYPE_SERVER);
    }
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    auto res = inst->routes->get_props(req);
    if (res->status != 200) {
        return res;
    }
    try {
        json                        props         = json::parse(res->data);
        int                         total_slots   = 0;
        json                        instances_arr = json::array();
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        for (const auto & it : instances) {
            total_slots += it->effective.n_parallel;
            instances_arr.push_back({
                { "name",  it->cfg.name        },
                { "group", it->cfg.group       },
                { "n_ctx", it->effective.n_ctx },
            });
        }
        props["total_slots"] = total_slots;
        props["instances"]   = instances_arr;
        res->data            = safe_json_to_str(props);
    } catch (const std::exception & e) {
        IST_WRN("failed to merge /props: %s\n", e.what());
    }
    return res;
}

server_http_res_ptr server_instances::handle_post_props(const server_http_req & req) {
    auto inst = default_instance();
    if (!inst) {
        return make_error("no instances loaded", ERROR_TYPE_SERVER);
    }
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    return inst->routes->post_props(req);
}

server_http_res_ptr server_instances::handle_post_infill(const server_http_req & req) {
    return dispatch(req, [](server_routes & routes, const server_http_req & req) { return routes.post_infill(req); });
}

server_http_res_ptr server_instances::handle_post_completions(const server_http_req & req) {
    return dispatch(req,
                    [](server_routes & routes, const server_http_req & req) { return routes.post_completions(req); });
}

server_http_res_ptr server_instances::handle_post_completions_oai(const server_http_req & req) {
    return dispatch(
        req, [](server_routes & routes, const server_http_req & req) { return routes.post_completions_oai(req); });
}

server_http_res_ptr server_instances::handle_post_chat_completions(const server_http_req & req) {
    return dispatch(
        req, [](server_routes & routes, const server_http_req & req) { return routes.post_chat_completions(req); });
}

server_http_res_ptr server_instances::handle_post_chat_completions_tok(const server_http_req & req) {
    return dispatch(
        req, [](server_routes & routes, const server_http_req & req) { return routes.post_chat_completions_tok(req); });
}

server_http_res_ptr server_instances::handle_post_control(const server_http_req & req) {
    auto inst = default_instance();
    if (!inst) {
        return make_error("no instances loaded", ERROR_TYPE_SERVER);
    }
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    return inst->routes->post_control(req);
}

server_http_res_ptr server_instances::handle_post_responses_oai(const server_http_req & req) {
    return dispatch(req,
                    [](server_routes & routes, const server_http_req & req) { return routes.post_responses_oai(req); });
}

server_http_res_ptr server_instances::handle_post_responses_tok_oai(const server_http_req & req) {
    return dispatch(
        req, [](server_routes & routes, const server_http_req & req) { return routes.post_responses_tok_oai(req); });
}

server_http_res_ptr server_instances::handle_post_transcriptions_oai(const server_http_req & req) {
    return dispatch(
        req, [](server_routes & routes, const server_http_req & req) { return routes.post_transcriptions_oai(req); });
}

server_http_res_ptr server_instances::handle_post_anthropic_messages(const server_http_req & req) {
    return dispatch(
        req, [](server_routes & routes, const server_http_req & req) { return routes.post_anthropic_messages(req); });
}

server_http_res_ptr server_instances::handle_post_anthropic_count_tokens(const server_http_req & req) {
    return dispatch(req, [](server_routes & routes, const server_http_req & req) {
        return routes.post_anthropic_count_tokens(req);
    });
}

server_http_res_ptr server_instances::handle_post_apply_template(const server_http_req & req) {
    auto inst = default_instance();
    if (!inst) {
        return make_error("no instances loaded", ERROR_TYPE_SERVER);
    }
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    return inst->routes->post_apply_template(req);
}

server_http_res_ptr server_instances::handle_get_models(const server_http_req & req) {
    // one entry per instance: reuse each instance's get_models handler and merge, keeping
    // the legacy /models shape ({"models": [...], "object": "list", "data": [...]})
    json models = json::array();
    json data   = json::array();
    for (const auto & inst : snapshot_instances()) {
        active_route_guard guard(*this, *inst);
        if (!guard.acquired) {
            continue;  // being destroyed/resized; skip it
        }
        auto res = inst->routes->get_models(req);
        if (res->status != 200) {
            return res;
        }
        try {
            json entry = json::parse(res->data);
            for (auto & m : entry["models"]) {
                models.push_back(std::move(m));
            }
            for (auto & d : entry["data"]) {
                d["n_ctx"]    = inst->effective.n_ctx;
                d["parallel"] = inst->effective.n_parallel;
                d["status"]   = "loaded";
                data.push_back(std::move(d));
            }
        } catch (const std::exception & e) {
            IST_WRN("failed to merge /models: %s\n", e.what());
        }
    }
    return make_ok({
        { "models", models },
        { "object", "list" },
        { "data",   data   }
    });
}

server_http_res_ptr server_instances::handle_post_tokenize(const server_http_req & req) {
    auto inst = default_instance();
    if (!inst) {
        return make_error("no instances loaded", ERROR_TYPE_SERVER);
    }
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    return inst->routes->post_tokenize(req);
}

server_http_res_ptr server_instances::handle_post_detokenize(const server_http_req & req) {
    auto inst = default_instance();
    if (!inst) {
        return make_error("no instances loaded", ERROR_TYPE_SERVER);
    }
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    return inst->routes->post_detokenize(req);
}

server_http_res_ptr server_instances::handle_post_embeddings(const server_http_req & req) {
    return dispatch(req,
                    [](server_routes & routes, const server_http_req & req) { return routes.post_embeddings(req); });
}

server_http_res_ptr server_instances::handle_post_embeddings_oai(const server_http_req & req) {
    return dispatch(
        req, [](server_routes & routes, const server_http_req & req) { return routes.post_embeddings_oai(req); });
}

server_http_res_ptr server_instances::handle_post_rerank(const server_http_req & req) {
    return dispatch(req, [](server_routes & routes, const server_http_req & req) { return routes.post_rerank(req); });
}

server_http_res_ptr server_instances::handle_get_lora_adapters(const server_http_req & req) {
    auto inst = default_instance();
    if (!inst) {
        return make_error("no instances loaded", ERROR_TYPE_SERVER);
    }
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    return inst->routes->get_lora_adapters(req);
}

server_http_res_ptr server_instances::handle_post_lora_adapters(const server_http_req & req) {
    auto inst = default_instance();
    if (!inst) {
        return make_error("no instances loaded", ERROR_TYPE_SERVER);
    }
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    return inst->routes->post_lora_adapters(req);
}

//
// management API handlers
//

server_http_res_ptr server_instances::handle_get_instances(const server_http_req &) {
    return make_ok(get_instances_json());
}

server_http_res_ptr server_instances::handle_post_instances(const server_http_req & req) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception &) {
        return make_error("invalid JSON body", ERROR_TYPE_INVALID_REQUEST);
    }
    if (!body.is_object()) {
        return make_error("body must be a JSON object", ERROR_TYPE_INVALID_REQUEST);
    }

    common_instance cfg;
    cfg.name       = json_value(body, "name", std::string());
    cfg.group      = json_value(body, "group", std::string());
    cfg.ctx_size   = json_value(body, "ctx_size", 0);
    cfg.parallel   = json_value(body, "parallel", 0);
    cfg.pinned     = json_value(body, "pinned", false);
    cfg.is_default = json_value(body, "default", false);

    if (cfg.name.empty()) {
        return make_error("'name' is required", ERROR_TYPE_INVALID_REQUEST);
    }
    if (cfg.group.empty()) {
        cfg.group = cfg.name;
    }
    if (cfg.ctx_size < 0 || cfg.parallel < 0) {
        return make_error("'ctx_size' and 'parallel' must be non-negative", ERROR_TYPE_INVALID_REQUEST);
    }

    try {
        common_instance_validate(cfg);
    } catch (const std::invalid_argument & e) {
        return make_error(e.what(), ERROR_TYPE_INVALID_REQUEST);
    }

    return create_instance(cfg);
}

server_http_res_ptr server_instances::handle_post_instance_pin(const server_http_req & req) {
    return set_instance_pinned(req.get_param("name"), true);
}

server_http_res_ptr server_instances::handle_post_instance_unpin(const server_http_req & req) {
    return set_instance_pinned(req.get_param("name"), false);
}

server_http_res_ptr server_instances::handle_post_instance_resize(const server_http_req & req) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception &) {
        return make_error("invalid JSON body", ERROR_TYPE_INVALID_REQUEST);
    }
    const int32_t new_ctx = json_value(body, "ctx_size", 0);
    return resize_instance(req.get_param("name"), new_ctx);
}

server_http_res_ptr server_instances::handle_delete_instance(const server_http_req & req) {
    // force is accepted for compatibility and ignored (nothing enforces pinned)
    return destroy_instance(req.get_param("name"), !req.get_param("force").empty());
}

server_http_res_ptr server_instances::handle_post_instance_snapshot(const server_http_req & req) {
    // POST /instances/:name/snapshot  body { "name": "<snapshot>", "id_slot": N? }
    // saves one slot's KV into the instance's own snapshot namespace
    // (<slot_save_path>/<model_key>/<instance>/<snapshot>.bin). id_slot selects
    // the slot (default 0); snapshots are per-instance, never silently slot 0
    // of another instance's namespace. serialized with resize/destroy so the
    // slot-save task on the scheduler never races a context rebuild that
    // destroys the very context it is copying from
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    const std::string name = req.get_param("name");
    json              body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception &) {
        return make_error("invalid JSON body", ERROR_TYPE_INVALID_REQUEST);
    }
    const std::string snapshot = json_value(body, "name", std::string());
    const int         id_slot  = json_value(body, "id_slot", 0);
    if (!fs_validate_filename(snapshot)) {
        return make_error("invalid snapshot name", ERROR_TYPE_INVALID_REQUEST);
    }
    if (params.slot_save_path.empty()) {
        return make_error("snapshot management requires --slot-save-path", ERROR_TYPE_NOT_SUPPORTED);
    }

    auto inst = get_instance(name);
    if (!inst) {
        return make_error(format_error_response("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND));
    }
    if (id_slot < 0 || (size_t) id_slot >= inst->slot_snapshots.size()) {
        return make_error("invalid slot id", ERROR_TYPE_INVALID_REQUEST);
    }

    const std::string dir = server_snapshot_instance_dir(params.slot_save_path,
                                                         server_instance_model_key(base_name), name);
    std::error_code   ec;
    std::filesystem::create_directories(dir, ec);

    const std::string filepath = snapshot_instance_path(name, snapshot);

    // the same compose pipeline as a request-time switch: per-slot lock, switch
    // semaphore, deadline-bounded KV copy on the scheduler, then an awaited
    // file write on the pool I/O worker
    {
        std::lock_guard<std::mutex> slot_lock(*inst->mutex_snapshot[id_slot]);
        const int64_t               deadline_ms = snapshot_deadline_ms(*inst);
        switch_guard                sw_guard(*this, deadline_ms);
        if (!sw_guard.acquired) {
            return make_error("too many concurrent snapshot switches, retry", ERROR_TYPE_UNAVAILABLE);
        }

        auto result = inst->ctx_server->slot_save_copy(id_slot, deadline_ms);
        if (!result) {
            return make_error("snapshot save timed out", ERROR_TYPE_UNAVAILABLE);
        }
        if (result->is_error()) {
            return make_error(result->to_json());
        }
        auto * copy = dynamic_cast<server_task_result_slot_copy *>(result.get());
        GGML_ASSERT(copy != nullptr);
        server_snapshot_data data;
        data.n_ctx_seq = inst->ctx_server->get_slot_n_ctx();
        data.tokens    = copy->tokens;
        data.kv        = std::move(copy->buffer);
        // await the write so a failed save never binds the slot to a file that does not
        // exist on disk (the slot KV is unchanged either way)
        auto write = snapshot_io_write(filepath, std::move(data), deadline_ms);
        if (write.busy) {
            return make_error("snapshot io worker is busy, retry", ERROR_TYPE_UNAVAILABLE);
        }
        if (write.timed_out) {
            return make_error("snapshot save timed out", ERROR_TYPE_UNAVAILABLE);
        }
        if (!write.ok) {
            return make_error("failed to write snapshot '" + snapshot + "' to disk", ERROR_TYPE_SERVER);
        }

        // the instance-scoped file is now authoritative: drop a legacy flat
        // file of the same name (migration dedup) so listings never show the
        // snapshot twice and future reads cannot ambiguate.
        {
            std::error_code lec;
            std::filesystem::remove(snapshot_legacy_path(snapshot), lec);
        }

        // the slot's KV now matches the snapshot content, so bind it to the snapshot
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        inst->slot_snapshots[id_slot] = snapshot;

        return make_ok(result->to_json(), 201);
    }
}

server_http_res_ptr server_instances::handle_get_instance_snapshots(const server_http_req & req) {
    const std::string name = req.get_param("name");
    auto              inst = get_instance(name);
    if (!inst) {
        return make_error(format_error_response("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND));
    }
    if (params.slot_save_path.empty()) {
        return make_error("snapshot management requires --slot-save-path", ERROR_TYPE_NOT_SUPPORTED);
    }

    return make_ok({
        { "snapshots", instance_snapshots_json(name) }
    });
}

server_http_res_ptr server_instances::handle_delete_instance_snapshot(const server_http_req & req) {
    // serialized with resize so a binding cleanup can never race a context rebuild that
    // re-allocates the slot-snapshot bookkeeping
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    const std::string name     = req.get_param("name");
    const std::string snapshot = req.get_param("snapshot");
    if (!fs_validate_filename(snapshot)) {
        return make_error("invalid snapshot name", ERROR_TYPE_INVALID_REQUEST);
    }
    if (params.slot_save_path.empty()) {
        return make_error("snapshot management requires --slot-save-path", ERROR_TYPE_NOT_SUPPORTED);
    }

    auto inst = get_instance(name);
    if (!inst) {
        return make_error(format_error_response("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND));
    }

    // per-instance delete: the instance-scoped file first, then the legacy
    // flat file (migration). each removed file unbinds slots: the named
    // instance's slots for its own file, every instance's slots for a legacy
    // file (which any instance may have restored before scoping existed).
    const std::string inst_path = snapshot_instance_path(name, snapshot);
    const std::string leg_path  = snapshot_legacy_path(snapshot);

    std::error_code ec;
    const bool removed_inst = std::filesystem::remove(inst_path, ec);
    ec.clear();
    const bool removed_leg = std::filesystem::remove(leg_path, ec);

    if (!removed_inst && !removed_leg) {
        return make_error(format_error_response("snapshot not found: '" + snapshot + "'", ERROR_TYPE_NOT_FOUND));
    }

    // unbind any slot that was bound to the deleted snapshot
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        if (removed_leg) {
            for (const auto & it : instances) {
                for (auto & s : it->slot_snapshots) {
                    if (s == snapshot) {
                        s.clear();
                    }
                }
            }
        } else {
            for (auto & s : inst->slot_snapshots) {
                if (s == snapshot) {
                    s.clear();
                }
            }
        }
    }

    return make_ok({
        { "success", true }
    });
}

//
// lifecycle
//

void server_instances::start_loops() {
    for (const auto & inst : instances) {
        inst->loop_thread = std::thread([inst]() { inst->ctx_server->start_loop(); });
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
        inst->ctx_server->terminate();
    }
    for (const auto & inst : live) {
        if (inst->loop_thread.joinable()) {
            inst->loop_thread.join();
        }
    }
    stop_io_worker();
}

// pool snapshot I/O worker: a single FIFO thread owns every on-disk snapshot read/write,
// so a snapshot switch never does file I/O on a scheduler thread and a client disconnect
// cannot interrupt an in-flight write. jobs are bounded by the switch semaphore.
void server_instances::io_loop() {
    while (true) {
        std::packaged_task<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_io);
            cond_io.wait(lock, [&]() { return io_stop || !io_jobs.empty(); });
            if (io_jobs.empty()) {
                return;  // io_stop set and the queue is drained
            }
            task = std::move(io_jobs.front());
            io_jobs.pop_front();
        }
        task();
    }
}

std::optional<std::future<void>> server_instances::snapshot_io_post(std::function<void()> && fn) {
    std::packaged_task<void()> task(std::move(fn));
    auto                       future = task.get_future();
    {
        std::lock_guard<std::mutex> lock(mutex_io);
        // hard bound on queued jobs: each queued write holds a full KV host buffer, so a
        // full queue rejects with a retriable error instead of accumulating host memory
        if (io_jobs.size() >= max_io_jobs) {
            return std::nullopt;
        }
        io_jobs.push_back(std::move(task));
    }
    cond_io.notify_one();
    return future;
}

void server_instances::start_io_worker() {
    std::lock_guard<std::mutex> lock(mutex_io);
    if (!io_thread.joinable()) {
        io_thread = std::thread([this]() { io_loop(); });
    }
}

void server_instances::stop_io_worker() {
    {
        std::lock_guard<std::mutex> lock(mutex_io);
        io_stop = true;
        cond_io.notify_all();
    }
    // the loop drains any pending jobs before it returns, so an HTTP thread waiting
    // on a job future is never left hanging at shutdown
    if (io_thread.joinable()) {
        io_thread.join();
    }
}

std::shared_ptr<server_instance> server_instances::default_instance() {
    std::lock_guard<std::mutex> lock(mutex_dispatch);
    for (const auto & inst : instances) {
        if (inst->cfg.is_default) {
            return inst;
        }
    }
    return instances.empty() ? nullptr : instances.front();
}

std::shared_ptr<server_instance> server_instances::default_instance() const {
    std::lock_guard<std::mutex> lock(mutex_dispatch);
    for (const auto & inst : instances) {
        if (inst->cfg.is_default) {
            return inst;
        }
    }
    return instances.empty() ? nullptr : instances.front();
}

server_http_res_ptr server_instances::make_error(const std::string & message, error_type type) const {
    return make_error(format_error_response(message, type));
}

server_http_res_ptr server_instances::make_error(const json & error) const {
    auto res          = std::make_unique<server_http_res>();
    res->status       = json_value(error, "code", 500);
    res->content_type = "application/json; charset=utf-8";
    res->data         = safe_json_to_str({
        { "error", error }
    });
    return res;
}

server_http_res_ptr server_instances::make_error(int                 code,
                                                 const std::string & type,
                                                 const std::string & message) const {
    auto res          = std::make_unique<server_http_res>();
    res->status       = code;
    res->content_type = "application/json; charset=utf-8";
    res->data         = safe_json_to_str({
        { "error", { { "code", code }, { "message", message }, { "type", type } } }
    });
    return res;
}

server_http_res_ptr server_instances::make_ok(const json & data, int status) const {
    auto res          = std::make_unique<server_http_res>();
    res->status       = status;
    res->content_type = "application/json; charset=utf-8";
    res->data         = safe_json_to_str(data);
    return res;
}
