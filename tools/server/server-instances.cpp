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

// facade: routing, aggregates, management handlers. bodies live in the
// lifecycle, snapshot and adapter units; handlers here only parse and call down.

// one stored adapter spelling matches a request key when both canonicalize to
// the same file (covers relative, dot-dot and symlink spellings of one file,
// including cfg declarations recorded before the file resolved), or by
// filesystem identity when the keys differ (hardlink, or a spelling that
// resolved after the entry was stored). identity never merges on missing or
// unreadable spellings: detach a deleted file with the exact spelling shown
// by GET (the stored entry key, which then matches lexically).
static bool adapter_key_matches(const std::string & stored, const std::string & key) {
    return server_instances::adapter_key(stored) == server_instances::adapter_key(key) ||
           server_instances::adapter_identity_equal(stored, key);
}

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
        std::shared_ptr<server_instance> sole;
        for (const auto & inst : instances) {
            if (sole) {
                sole = nullptr;
                break;
            }
            sole = inst;
        }
        if (sole) {
            res.kind = target_kind::INSTANCE;
            res.inst = sole;
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

    // <base> alone: the default instance (or the sole instance, or ambiguous).
    // the explicit `instance` request field overrides the absent component.
    if (comps.size() == 1) {
        if (!explicit_instance.empty()) {
            return resolve_instance_or_group(explicit_instance, error);
        }
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
            // the explicit `instance` request field overrides the reserved pin
            if (!explicit_instance.empty()) {
                return resolve_instance_or_group(explicit_instance, error);
            }
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

    // <base>:latest:<name|group> -> that member. 'latest' is reserved, so a
    // middle component holding anything else cannot be a legal id; more than
    // three components is never a legal id either.
    if (comps.size() == 3) {
        if (comps[1] != "latest") {
            error = "model not found: '" + model_id + "'";
            return res;
        }
        std::string target = comps[2];

        // the explicit `instance` request field overrides the model's instance/group component
        if (!explicit_instance.empty()) {
            target = explicit_instance;
        }
        return resolve_instance_or_group(target, error);
    }

    error = "model not found: '" + model_id + "'";
    return res;
}

// single reduction of an instance's slot activity, shared by group routing and
// last-used reporting: unbuilt windows report never-used stamps; built windows
// report the scheduler-published aggregates (max stamps, busy count). this is
// the one place the fresh policy is defined: last_used_us == -1 sorts a cold
// member before any used member among equally-busy candidates.
static server_context_stats instance_stats(const server_instance & inst) {
    if (!inst.built.load(std::memory_order_acquire)) {
        return server_context_stats{};
    }
    return inst.ctx_server->get_stats();
}

std::optional<size_t> server_instances::pick_best_candidate(const std::vector<route_candidate> & members) {
    std::optional<size_t> best;
    int                   best_busy = INT32_MAX;
    int64_t               best_lru  = INT64_MAX;

    for (const auto & cand : members) {
        // unbuilt (no slots) and fully busy members are never candidates; when
        // every member is busy the caller waits for a slot release instead
        if (cand.n_slots <= 0 || cand.n_busy >= cand.n_slots) {
            continue;
        }

        // prefer fewer busy slots, then least-recently-used; tie-break by
        // instance order (strict compare keeps the first registration)
        bool better = false;
        if (!best) {
            better = true;
        } else if (cand.n_busy != best_busy) {
            better = cand.n_busy < best_busy;
        } else {
            better = cand.last_used_us < best_lru;
        }

        if (better) {
            best      = cand.index;
            best_busy = cand.n_busy;
            best_lru  = cand.last_used_us;
        }
    }
    return best;
}

std::optional<size_t> server_instances::pick_best_available(const std::string & group) const {
    // caller must hold mutex_dispatch
    std::vector<route_candidate> members;
    for (size_t i = 0; i < instances.size(); ++i) {
        const server_instance & inst = *instances[i];
        // removing = a management op owns the instance; running = false once a destroy or
        // a pool-wide shutdown has begun, so a group waiter never picks a dying instance.
        // unbuilt members have no slots yet; group demand materializes the first one
        // (see dispatch_group) instead of picking it here.
        if (inst.cfg.group != group || inst.removing || !inst.running || !inst.built.load(std::memory_order_acquire)) {
            continue;
        }

        const server_context_stats stats = instance_stats(inst);
        members.push_back({ i, inst.n_parallel_effective.load(std::memory_order_relaxed), stats.n_processing, stats.last_used_us });
    }
    return pick_best_candidate(members);
}

void server_instances::on_slot_released() {
    std::lock_guard<std::mutex> lock(mutex_dispatch);
    wake_count_total += (wake_count_total < UINT64_MAX);
    size_t parked = 0;
    for (const auto & kv : group_waiters_now) {
        parked += kv.second;
    }
    if (parked == 0) {
        wake_count_dropped += (wake_count_dropped < UINT64_MAX);
    }
    cond_dispatch.notify_all();
}

server_instances::wake_stats server_instances::get_wake_stats() const {
    std::lock_guard<std::mutex> lock(mutex_dispatch);
    wake_stats out;
    out.wakes_total       = wake_count_total;
    out.wakes_dropped     = wake_count_dropped;
    out.group_waits_total = group_waits_total;
    out.group_waiters_now = group_waiters_now;
    return out;
}

server_instances::build_stats server_instances::get_build_stats() const {
    // lock order everywhere is mgmt -> dispatch: take mgmt first so a racing
    // demand holding mgmt and waiting on dispatch can never deadlock us
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
    std::lock_guard<std::mutex> lock(mutex_dispatch);
    return build_stats { cold_reload_count, demand_build_collapsed };
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

server_instances::dispatch_guard::dispatch_guard(server_instances & m, const std::shared_ptr<server_instance> & i)
    : mgr(m), inst(i) {
    std::lock_guard<std::mutex> lock(mgr.mutex_dispatch);
    inst->n_active_dispatch++;
    // check under the same lock that increments the count, so a destroy that starts
    // after this point must drain this dispatch before stopping the scheduler
    if (inst->removing || !inst->running) {
        inst->n_active_dispatch--;
        inst.reset();
        return;
    }
    acquired = true;
}

server_instances::dispatch_guard::~dispatch_guard() {
    if (acquired) {
        std::lock_guard<std::mutex> lock(mgr.mutex_dispatch);
        inst->n_active_dispatch--;
        mgr.cond_dispatch.notify_all();
    }
}

server_instances::dispatch_guard::dispatch_guard(dispatch_guard && o) noexcept : mgr(o.mgr), inst(std::move(o.inst)) {
    acquired   = o.acquired;
    o.acquired = false;
    o.inst.reset();
}

// shared tail of both dispatch heads: the guard already holds the dispatch
// count for this call, so every return path (including a throwing route)
// releases it. a handler must not throw (ex_wrapper guarantees it at the HTTP
// layer); the guard keeps the counter balanced anyway so a destroy never
// blocks on a leaked count.
server_http_res_ptr server_instances::dispatch_resolved(const server_http_req & req,
                                                        server_instance &        inst,
                                                        const std::string &      snapshot,
                                                        int                      id_slot,
                                                        const forward_fn &       forward,
                                                        dispatch_guard /*guard*/) {
    if (!snapshot.empty()) {
        // a plain request leaves the slot (and any bound snapshot) untouched; only a
        // named snapshot can change the binding
        if (server_http_res_ptr err = apply_snapshot(inst, snapshot, id_slot)) {
            return err;
        }
    }
    return forward(*inst.routes, req);
}
server_http_res_ptr server_instances::dispatch_instance(const server_http_req &                  req,
                                                        const std::shared_ptr<server_instance> & inst,
                                                        const std::string &                      snapshot,
                                                        int                                      id_slot,
                                                        const forward_fn &                       forward) {
    // first demand for this window materializes it: exactly one context is ever built
    // per demand, never the whole pool upfront
    if (auto err = ensure_built_instance(inst)) {
        return err;
    }

    dispatch_guard guard(*this, inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }

    return dispatch_resolved(req, *inst, snapshot, id_slot, forward, std::move(guard));
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
//   - dispatch guard release on every dispatch exit
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
        std::shared_ptr<server_instance> unbuilt;
        {
            std::lock_guard<std::mutex> lock(mutex_dispatch);
            if (const auto best = pick_best_available(group)) {
                inst = instances[*best];
            } else {
                // no built member has a free slot: materialize the first registered
                // member instead of waiting. one demand builds exactly one window.
                for (const auto & it : instances) {
                    if (it->cfg.group == group && !it->built.load(std::memory_order_acquire) && !it->removing && it->running) {
                        unbuilt = it;
                        break;
                    }
                }
            }
        }

        if (unbuilt) {
            // a persistently failing build degrades to the 503 deadline below, never a hang
            // and never a hot spin: only a successful build re-picks immediately
            if (auto err = ensure_built_instance(unbuilt)) {
                IST_WRN("on-demand build of instance '%s' failed, waiting for a free member\n",
                        unbuilt->cfg.name.c_str());
            } else {
                continue;
            }
        }

        if (inst) {
            // the count and the removing/running check stay atomic inside the
            // guard; a lost teardown race re-picks instead of serving a dead queue
            dispatch_guard guard(*this, inst);
            if (!guard.acquired) {
                continue;
            }
            return dispatch_resolved(req, *inst, snapshot, id_slot, forward, std::move(guard));
        }

        // predicate wait (no poll): re-picks under mutex_dispatch on every notify.
        // the waiter parks itself in the per-group counters first, so a
        // slot-release wake can tell a useful wake (waiters parked) from a
        // dropped one; unparked on every exit path below.
        std::unique_lock<std::mutex> lock(mutex_dispatch);
        group_waiters_now[group]++;
        group_waits_total[group] += (group_waits_total[group] < UINT64_MAX);
        const bool                   ready = cond_dispatch.wait_until(lock, deadline, [&]() {
            return pick_best_available(group).has_value();
        });
        if (--group_waiters_now[group] == 0) {
            group_waiters_now.erase(group);
        }
        if (!ready) {
            return make_error("no free instance in group '" + group + "'", ERROR_TYPE_UNAVAILABLE);
        }
    }
}


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

int32_t server_instances::displayed_n_ctx(const server_instance & inst) const {
    if (inst.built.load(std::memory_order_acquire)) {
        return inst.ctx_server->get_n_ctx();
    }
    const int32_t cached = inst.n_ctx_effective.load(std::memory_order_relaxed);
    if (cached > 0) {
        return cached;
    }
    return train_ctx_cached.load(std::memory_order_relaxed);
}

// most recent slot use across an instance (max t_last_used over its slots), -1 when unused
// (or unbuilt: a registered window has no slots yet)
static int64_t instance_last_used(const server_instance & inst) {
    return instance_stats(inst).last_used_us;
}

// wall-clock twin of instance_last_used: unix epoch seconds of the most recent slot
// release, -1 when unused or unbuilt. safe to compare against the caller's own clock
// and across processes, unlike the monotonic last_used.
static int64_t instance_last_used_epoch(const server_instance & inst) {
    return instance_stats(inst).last_used_wall_s;
}

json server_instances::instance_to_json(const server_instance & inst) const {
    // adapter accounting reads effective.lora_adapters: every caller holds mutex_mgmt
    // (create/resize/pin paths) or takes it first (get_instances_json below)
    const uint64_t adapter_bytes = instance_adapter_bytes(inst);
    // an unbuilt (registered but never demanded) window owns no buffers yet
    if (!inst.built.load(std::memory_order_acquire)) {
        return instance_to_json(inst, 0, 0, 0, adapter_bytes);
    }
    // all memory fields derive from the three instance getters (single source of truth)
    const uint64_t model_bytes   = inst.ctx_server->get_model_bytes();
    const uint64_t context_bytes = inst.ctx_server->get_context_bytes();
    const uint64_t compute_bytes = inst.ctx_server->get_compute_bytes();
    return instance_to_json(inst, model_bytes, context_bytes, compute_bytes, adapter_bytes);
}

json server_instances::instance_to_json(const server_instance & inst,
                                        uint64_t                model_bytes,
                                        uint64_t                context_bytes,
                                        uint64_t                compute_bytes,
                                        uint64_t                adapter_bytes) const {
    const int64_t t_last_used = instance_last_used(inst);
    const int64_t t_last_used_epoch = instance_last_used_epoch(inst);

    return json{
        { "id", instance_id(inst) },
        { "aliases", instance_aliases(inst.cfg.name, inst.cfg.group) },
        { "group", inst.cfg.group },
        { "n_ctx", displayed_n_ctx(inst) },
        { "parallel", inst.n_parallel_effective.load(std::memory_order_relaxed) },
        { "pinned", inst.cfg.pinned },
        { "is_default", inst.cfg.is_default },
        // unbuilt = registered but never demanded; its window (and bytes) do not exist yet
        { "state", inst.built.load(std::memory_order_acquire) ? "loaded" : "unloaded" },
        // memory breakdown
        { "model_bytes", model_bytes },
        { "context_bytes", context_bytes },
        { "compute_bytes", compute_bytes },
        // this instance's own adapter footprint (a shared entry counts toward every
        // instance referencing it: evicting any one of them does not free it)
        { "adapter_bytes", adapter_bytes },
        // every leg of the window adds up, adapters included
        { "total_bytes", model_bytes + context_bytes + compute_bytes + adapter_bytes },
        // documented alias kept for the list() contract: context + compute
        { "vram_bytes", context_bytes + compute_bytes },
        { "last_used", t_last_used },
        { "last_used_epoch", t_last_used_epoch },
    };
}

json server_instances::get_instances_json() const {
    json     instances_arr = json::array();
    // 64-bit sums; the shared weights are counted once per pool
    uint64_t total_model   = 0;
    uint64_t total_context = 0;
    uint64_t total_compute = 0;
    uint64_t total_adapter = 0;
    bool     model_counted = false;

    // mgmt first (lock order mgmt -> dispatch): adapter accounting reads the
    // registry and effective.lora_adapters, both mgmt-guarded. the scope covers
    // exactly the instance rows, the totals and the registry sum; the snapshot
    // listing below does file I/O and must NOT hold this lock, because
    // GET /instances is the router's polling endpoint and slow storage would
    // otherwise serialize create/destroy/resize behind every poll.
    {
        std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);
        {
            std::lock_guard<std::mutex> lock(mutex_dispatch);
            for (const auto & inst : instances) {
                // unbuilt windows own no buffers; they contribute identity with zero bytes.
                // both locks are held here (mgmt -> dispatch), so the context is stable
                const bool     is_built      = inst->built.load(std::memory_order_acquire);
                const uint64_t model_bytes   = is_built ? inst->ctx_server->get_model_bytes()   : 0;
                const uint64_t context_bytes = is_built ? inst->ctx_server->get_context_bytes() : 0;
                const uint64_t compute_bytes = is_built ? inst->ctx_server->get_compute_bytes() : 0;
                const uint64_t adapter_bytes = instance_adapter_bytes(*inst);

                instances_arr.push_back(instance_to_json(*inst, model_bytes, context_bytes, compute_bytes, adapter_bytes));

                if (!model_counted && model_bytes > 0) {
                    total_model += model_bytes;
                    model_counted = true;
                }
                total_context += context_bytes;
                total_compute += compute_bytes;
            }
        }

        // pool-wide adapter footprint: each registry file counted once
        for (const auto & kv : adapter_registry) {
            total_adapter += llama_adapter_lora_buf_size(kv.second.adapter.get());
        }
    }

    // snapshots are on-disk and independent of the live instances; list them so a cold
    // (weight-unloaded) pool's KV snapshots stay discoverable for reactivation.
    // this reads only params and base_name, both immutable after load(), so it runs
    // unlocked and never serializes management behind slow snapshot storage.
    json snapshots = pool_snapshots_json();

    return json{
        { "instances", std::move(instances_arr) },
        { "snapshots", std::move(snapshots) },
        { "total",
         {
              { "model", total_model },
              { "context", total_context },
              { "compute", total_compute },
              { "adapter", total_adapter },
              { "total", total_model + total_context + total_compute + total_adapter },
          }                                     },
    };
}

// one snapshot row builder for both JSON producers: {name, size, mtime,


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
    // a registered-but-undemanded default is healthy too, and probing it must NOT
    // materialize its window: orchestrators poll /health constantly, and a probe
    // is not a demand. the guard is taken BEFORE any field is read, so the shared_ptr
    // cannot outlive a concurrent destroy/resize into a torn context.
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    if (!inst->built.load(std::memory_order_acquire)) {
        return make_ok({
            { "status",    "ok"      },
            { "instances", (int) snapshot_instances().size() },
        });
    }
    return inst->routes->get_health(req);
}

server_http_res_ptr server_instances::handle_get_metrics(const server_http_req & req) {
    // a scrape is an observability read, never a demand: it must not materialize
    // an unbuilt window. unlike /slots there is no aggregate form, because the
    // metrics are context-local counters and merging them would invent a gauge
    // policy; no target therefore means the default instance.
    const std::string model_id       = req.get_param("model");
    const std::string instance_field = req.get_param("instance");

    std::shared_ptr<server_instance> inst;
    if (model_id.empty() && instance_field.empty()) {
        inst = default_instance();
    } else {
        std::string          error;
        const resolve_target target = resolve(model_id, instance_field, error);
        if (target.kind != target_kind::INSTANCE) {
            return make_error(error.empty() ? "invalid instance for metrics" : error, ERROR_TYPE_INVALID_REQUEST);
        }
        inst = target.inst;
    }
    if (!inst) {
        return make_error("no instances loaded", ERROR_TYPE_NOT_FOUND);
    }
    // the target window does not exist yet: render nothing rather than allocate
    // it as a scrape side effect. this is an exact outcome condition, not a
    // health or quality score. the guard is taken first so a concurrent
    // destroy/resize cannot turn this read into a torn context dereference.
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    if (!inst->built.load(std::memory_order_acquire)) {
        return make_error("instance '" + inst->cfg.name + "' has no loaded context", ERROR_TYPE_NOT_FOUND);
    }
    return inst->routes->get_metrics(req);
}

json server_instances::instance_error_marker(const server_http_res_ptr & res, const std::string & name) {
    json marker = { { "instance", name } };
    try {
        const json body = json::parse(res->data);
        marker["error"] = (body.is_object() && body.contains("error")) ? body.at("error") : body;
    } catch (const std::exception &) {
        marker["error"] = res->status;
    }
    return marker;
}

std::vector<server_instances::per_instance_route_result> server_instances::collect_instance_routes(
        const server_http_req & req,
        server_http_context::handler_t server_routes::* handler) {
    std::vector<per_instance_route_result> out;
    for (const auto & inst : snapshot_instances()) {
        // guard first, then read `built`: a concurrent destroy/resize refuses here
        // instead of letting the field read race its teardown
        active_route_guard guard(*this, *inst);
        if (!guard.acquired) {
            continue;  // being destroyed/resized; not reported at all
        }
        per_instance_route_result row;
        row.inst  = inst;
        // a registered-but-undemanded window is never materialized by a listing;
        // the caller decides whether to show it as unloaded (models) or skip it (slots)
        row.built = inst->built.load(std::memory_order_acquire);
        if (!row.built) {
            out.push_back(std::move(row));
            continue;
        }
        // the hook is a test-only seam (null in production)
        server_http_res_ptr res;
        if (aggregate_route_hook) {
            if (auto forced = aggregate_route_hook(*inst, req)) {
                res = std::move(*forced);
            }
        }
        if (!res) {
            res = (inst->routes.get()->*handler)(req);
        }
        if (res->status != 200) {
            row.error_marker = instance_error_marker(res, inst->cfg.name);
        } else {
            try {
                row.body = json::parse(res->data);
            } catch (const std::exception &) {
                // a 200 whose body cannot be parsed is still a failing member
                row.error_marker = instance_error_marker(res, inst->cfg.name);
            }
        }
        out.push_back(std::move(row));
    }
    return out;
}

server_http_res_ptr server_instances::handle_get_slots(const server_http_req & req) {
    std::string model_id       = req.get_param("model");
    std::string instance_field = req.get_param("instance");

    // no target: aggregate the slots of every instance, tagged with the instance name
    if (model_id.empty() && instance_field.empty()) {
        json all_slots = json::array();
        for (auto & row : collect_instance_routes(req, &server_routes::get_slots)) {
            if (!row.built) {
                continue;  // unbuilt windows have no slots yet; listing must not build them
            }
            if (!row.body) {
                // a dying member must not fail the whole aggregate: skip it with a
                // marker row carrying the instance (router-visible, never silent)
                all_slots.push_back(std::move(row.error_marker));
                continue;
            }
            for (auto & slot : *row.body) {
                slot["instance"] = row.inst->cfg.name;
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
    if (auto err = ensure_built_instance(target.inst)) {
        return err;
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
    if (auto err = ensure_built_instance(target.inst)) {
        return err;
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
    if (auto err = ensure_built_instance(inst)) {
        return err;
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
            // the display cache is lock-free; cfg.name/group are immutable after registration.
            // memory fields are size accessors on the context (they lock only mutex_mem, an
            // independent lock, so no lock-order concern under mutex_dispatch). an unbuilt
            // window owns no buffers yet and reports zero bytes.
            const bool     is_built      = it->built.load(std::memory_order_acquire);
            const uint64_t model_bytes   = is_built ? it->ctx_server->get_model_bytes()   : 0;
            const uint64_t context_bytes = is_built ? it->ctx_server->get_context_bytes() : 0;
            const uint64_t compute_bytes = is_built ? it->ctx_server->get_compute_bytes() : 0;
            total_slots += it->n_parallel_effective.load(std::memory_order_relaxed);
            instances_arr.push_back({
                { "name",          it->cfg.name        },
                { "group",         it->cfg.group       },
                { "n_ctx",         displayed_n_ctx(*it) },
                // per-instance memory; matches the /instances breakdown
                { "model_bytes",   model_bytes   },
                { "context_bytes", context_bytes },
                { "compute_bytes", compute_bytes },
                { "vram_bytes",    context_bytes + compute_bytes },
                { "total_bytes",   model_bytes + context_bytes + compute_bytes },
                { "state",         is_built ? "loaded" : "unloaded" },
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
    return default_instance_forward(req, &server_routes::post_props);
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
    return default_instance_forward(req, &server_routes::post_control);
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
    return default_instance_forward(req, &server_routes::post_apply_template);
}

server_http_res_ptr server_instances::handle_get_models(const server_http_req & req) {
    // one entry per instance: reuse each instance's get_models handler and merge, keeping
    // the legacy /models shape ({"models": [...], "object": "list", "data": [...]})
    json models = json::array();
    json data   = json::array();
    for (auto & row : collect_instance_routes(req, &server_routes::get_models)) {
        // a registered-but-undemanded window is listed without building it: a mere
        // listing must never materialize a context. the display caches are lock-free
        if (!row.built) {
            data.push_back({
                { "id",       instance_id(*row.inst)    },
                { "n_ctx",    displayed_n_ctx(*row.inst) },
                { "parallel", row.inst->n_parallel_effective.load(std::memory_order_relaxed) },
                { "status",   "unloaded"                },
            });
            continue;
        }
        // a failing built member no longer fails the whole aggregate: emit a marker
        // row carrying the instance (router-visible, never silent)
        if (!row.body) {
            data.push_back(std::move(row.error_marker));
            continue;
        }
        for (auto & m : (*row.body)["models"]) {
            models.push_back(std::move(m));
        }
        for (auto & d : (*row.body)["data"]) {
            d["n_ctx"]    = displayed_n_ctx(*row.inst);
            d["parallel"] = row.inst->n_parallel_effective.load(std::memory_order_relaxed);
            d["status"]   = "loaded";
            data.push_back(std::move(d));
        }
    }
    return make_ok({
        { "models", models },
        { "object", "list" },
        { "data",   data   }
    });
}

server_http_res_ptr server_instances::handle_post_tokenize(const server_http_req & req) {
    return default_instance_forward(req, &server_routes::post_tokenize);
}

server_http_res_ptr server_instances::handle_post_detokenize(const server_http_req & req) {
    return default_instance_forward(req, &server_routes::post_detokenize);
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
    return default_instance_forward(req, &server_routes::get_lora_adapters);
}

server_http_res_ptr server_instances::handle_post_lora_adapters(const server_http_req & req) {
    // the legacy writer commits to the scheduler through the stock handler (its
    // body grammar and zero-for-unlisted scale semantics stay the only
    // implementation). the manager mirror is then refreshed from the scheduler,
    // so the next attach does not re-apply a stale scale. the drain is exclusive:
    // no dispatch may read the effective list (apply_snapshot fingerprint) while
    // the refresh below rewrites it, so the write is never torn.
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    auto inst = default_instance();
    if (!inst) {
        return make_error("no instances loaded", ERROR_TYPE_NOT_FOUND);
    }
    // the legacy grammar needs a scheduler to apply to; an unbuilt default is the
    // same tier as save-to-unbuilt (404)
    if (!inst->built.load(std::memory_order_acquire)) {
        return make_error("instance '" + inst->cfg.name + "' has no loaded context", ERROR_TYPE_NOT_FOUND);
    }
    // like swap_adapter_set: management ops are serialized by mutex_mgmt, so no
    // two drains contend; the scheduler posts run while removing (direct queue)
    instance_drain_guard guard(*this, inst);
    auto res = inst->routes->post_lora_adapters(req);
    if (res->status != 200) {
        return res;
    }
    if (!refresh_effective_from_scheduler(*inst, snapshot_deadline_ms(*inst))) {
        return make_error("could not read the committed adapter set", ERROR_TYPE_UNAVAILABLE);
    }
    // the scale is part of the snapshot fingerprint: bound slots would otherwise
    // keep serving KV computed under the old adapters
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        inst->slot_snapshots.assign(inst->slot_snapshots.size(), std::string());
    }
    return res;
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


server_http_res_ptr server_instances::handle_post_instance_adapters(const server_http_req & req) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception &) {
        return make_error("invalid JSON body", ERROR_TYPE_INVALID_REQUEST);
    }
    if (!body.is_object()) {
        return make_error("body must be a JSON object", ERROR_TYPE_INVALID_REQUEST);
    }
    const std::string path  = json_value(body, "path", std::string());
    const float         scale = json_value(body, "scale", 1.0f);
    if (path.empty()) {
        return make_error("'path' is required", ERROR_TYPE_INVALID_REQUEST);
    }
    // no separator restriction here: the body is JSON, so ':' and ',' need no
    // escaping (the --instance grammar still rejects them as separators).
    if (!(scale > 0.0f) || !std::isfinite(scale)) {
        return make_error("'scale' must be a positive finite number", ERROR_TYPE_INVALID_REQUEST);
    }
    // the message carries no echo of the path: a malformed path may hold
    // bytes (such as NUL) that truncate log lines and confuse readers
    const auto key_opt = try_adapter_key(path);
    if (!key_opt) {
        return make_error("invalid adapter path", ERROR_TYPE_INVALID_REQUEST);
    }
    const std::string key = *key_opt;

    // serialized with the other management ops; dispatches never take this mutex, so
    // the drain below cannot deadlock (resize precedent, same lock order)
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    auto inst = get_instance(req.get_param("name"));
    if (!inst) {
        return make_error("instance not found: '" + req.get_param("name") + "'", ERROR_TYPE_NOT_FOUND);
    }

    // an unbuilt window has no scheduler to apply to: record the declaration, it
    // resolves (takes refs) when the window materializes. matches the resize-on-
    // unbuilt precedent, and avoids forcing a window allocation for a bookkeeping op.
    if (!inst->built.load(std::memory_order_acquire)) {
        bool present = false;
        for (auto & gl : inst->cfg.lora) {
            if (adapter_key_matches(gl.first, key)) {
                gl = { key, scale };
                present = true;
            }
        }
        if (!present) {
            inst->cfg.lora.emplace_back(key, scale);
        }
        inst->effective.lora_adapters.clear();
        for (const auto & gl : inst->cfg.lora) {
            inst->effective.lora_adapters.push_back({ gl.first, gl.second, "", "", nullptr });
        }
        return make_ok(instance_to_json(*inst));
    }

    // a built window swaps through the shared core: the mutator below only
    // edits the two lists (upsert + scale), everything else lives in one place
    adapter_set_mutator upsert = [&](std::vector<std::pair<std::string, float>> & cfg_lora,
                                     std::vector<common_adapter_lora_info> &     effective,
                                     adapter_set_delta &                         delta) -> server_http_res_ptr {
        bool present = false;
        for (auto & la : effective) {
            if (adapter_key_matches(la.path, key)) {
                la.scale = scale; // re-attach updates the scale; the ref is already held
                present  = true;
            }
        }
        if (!present) {
            // the entry key is the one spelling per file (first spelling
            // wins), so a shared entry is stored under its own key and the
            // release pairing still matches
            std::string entry_key;
            llama_adapter_lora * ptr = ensure_adapter(key, entry_key);
            if (ptr == nullptr) {
                return make_error("failed to load lora adapter '" + key + "'", ERROR_TYPE_INVALID_REQUEST);
            }
            delta.acquired.push_back({ entry_key, scale, "", "", ptr });
            effective.push_back({ entry_key, scale, "", "", ptr });
        }
        // cfg.lora tracks the same set in grammar form; matched independently so the
        // two lists can never diverge even if a previous op left them inconsistent
        bool cfg_present = false;
        for (auto & gl : cfg_lora) {
            if (adapter_key_matches(gl.first, key)) {
                gl           = { key, scale };
                cfg_present  = true;
            }
        }
        if (!cfg_present) {
            cfg_lora.emplace_back(key, scale);
        }
        return nullptr;
    };
    if (auto err = swap_adapter_set(inst, upsert)) {
        return err;
    }

    IST_INF("instance '%s' attached adapter '%s' (scale = %g)\n", inst->cfg.name.c_str(), key.c_str(), scale);
    return make_ok(instance_to_json(*inst));
}

server_http_res_ptr server_instances::handle_delete_instance_adapters(const server_http_req & req) {
    // filesystem paths contain '/', so no :path URL segment can hold them: the path
    // arrives in the JSON body, with ?path= accepted as an alias for empty bodies
    std::string path = req.get_param("path");
    if (!req.body.empty()) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception &) {
            return make_error("invalid JSON body", ERROR_TYPE_INVALID_REQUEST);
        }
        if (!body.is_object()) {
            return make_error("body must be a JSON object", ERROR_TYPE_INVALID_REQUEST);
        }
        const std::string body_path = json_value(body, "path", std::string());
        if (!body_path.empty()) {
            path = body_path;
        }
    }
    if (path.empty()) {
        return make_error("'path' is required (body or ?path= query)", ERROR_TYPE_INVALID_REQUEST);
    }
    // same boundary as attach: a malformed path is a caller bug (400),
    // answered before any instance lookup, drain, or registry touch
    const auto key_opt = try_adapter_key(path);
    if (!key_opt) {
        return make_error("invalid adapter path", ERROR_TYPE_INVALID_REQUEST);
    }
    const std::string key = *key_opt;

    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    auto inst = get_instance(req.get_param("name"));
    if (!inst) {
        return make_error("instance not found: '" + req.get_param("name") + "'", ERROR_TYPE_NOT_FOUND);
    }

    // an unbuilt window has no scheduler and no in-flight dispatch can be inside
    // it: mutate the declaration lists directly (no drain needed).
    if (!inst->built.load(std::memory_order_acquire)) {
        std::vector<common_adapter_lora_info> removed;
        {
            auto & eff = inst->effective.lora_adapters;
            for (auto it = eff.begin(); it != eff.end();) {
                if (adapter_key_matches(it->path, key)) {
                    removed.push_back(*it);
                    it = eff.erase(it);
                } else {
                    ++it;
                }
            }
        }
        if (removed.empty()) {
            return make_error("adapter not attached to instance '" + inst->cfg.name + "'", ERROR_TYPE_NOT_FOUND);
        }
        {
            auto & cfg_lora = inst->cfg.lora;
            for (auto it = cfg_lora.begin(); it != cfg_lora.end();) {
                if (adapter_key_matches(it->first, key)) {
                    it = cfg_lora.erase(it);
                } else {
                    ++it;
                }
            }
        }
        release_adapter_set(removed); // no-op: unbuilt entries hold no refs
        IST_INF("instance '%s' detached adapter '%s'\n", inst->cfg.name.c_str(), key.c_str());
        return make_ok({
            { "success", true }
        });
    }

    // fast 404 before the drain: an adapter that was never attached stalls no
    // traffic and holds the management plane only for the scan below. the
    // scan inside the drain stays authoritative: a racing attach can only add
    // the key (turning this into a success), and management ops already
    // serialize on the management lock, so the two scans cannot disagree.
    {
        bool attached = false;
        for (const auto & la : inst->effective.lora_adapters) {
            if (adapter_key_matches(la.path, key)) {
                attached = true;
                break;
            }
        }
        if (!attached) {
            return make_error("adapter not attached to instance '" + inst->cfg.name + "'", ERROR_TYPE_NOT_FOUND);
        }
    }

    // a built window swaps through the shared core: the mutator below only
    // edits the two lists (erase), everything else lives in one place. the
    // peek above already 404s the never-attached case, so the miss return
    // below is unreachable in practice, kept as the authoritative check.
    adapter_set_mutator erase = [&](std::vector<std::pair<std::string, float>> & cfg_lora,
                                    std::vector<common_adapter_lora_info> &     effective,
                                    adapter_set_delta &                         delta) -> server_http_res_ptr {
        // remove the key from both lists (they are kept in sync); the removed
        // entries carry the refs to drop on success
        for (auto it = effective.begin(); it != effective.end();) {
            if (adapter_key_matches(it->path, key)) {
                delta.released.push_back(*it);
                it = effective.erase(it);
            } else {
                ++it;
            }
        }
        if (delta.released.empty()) {
            return make_error("adapter not attached to instance '" + inst->cfg.name + "'", ERROR_TYPE_NOT_FOUND);
        }
        for (auto it = cfg_lora.begin(); it != cfg_lora.end();) {
            if (adapter_key_matches(it->first, key)) {
                it = cfg_lora.erase(it);
            } else {
                ++it;
            }
        }
        return nullptr;
    };
    if (auto err = swap_adapter_set(inst, erase)) {
        return err;
    }

    IST_INF("instance '%s' detached adapter '%s'\n", inst->cfg.name.c_str(), key.c_str());
    return make_ok({
        { "success", true }
    });
}

server_http_res_ptr server_instances::handle_get_instance_adapters(const server_http_req & req) {
    // reader side of the locking invariant: the effective list mutates under mutex_mgmt
    std::lock_guard<std::mutex> mgmt_lock(mutex_mgmt);

    auto inst = get_instance(req.get_param("name"));
    if (!inst) {
        return make_error("instance not found: '" + req.get_param("name") + "'", ERROR_TYPE_NOT_FOUND);
    }

    json adapters = json::array();
    for (const auto & la : inst->effective.lora_adapters) {
        adapters.push_back({
            { "path",  la.path  },
            { "scale", la.scale },
        });
    }
    return make_ok(adapters);
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
        return make_error("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND);
    }
    // saving needs live KV; unlike a generation request, a save must not materialize
    // a window as a side effect
    if (!inst->built.load(std::memory_order_acquire)) {
        return make_error("instance '" + name + "' has no loaded context", ERROR_TYPE_NOT_FOUND);
    }
    if (id_slot < 0 || (size_t) id_slot >= inst->slot_snapshots.size()) {
        return make_error("invalid slot id", ERROR_TYPE_INVALID_REQUEST);
    }

    const std::string dir = server_snapshot_instance_dir(params.slot_save_path,
                                                         server_snapshot_model_key_hashed(base_name), name);
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

        auto copied = kv_copy_to_data(*inst, id_slot, deadline_ms, "snapshot save timed out");
        if (!copied.ok) {
            return std::move(copied.error);
        }
        server_snapshot_data data = std::move(copied.data);
        // await the write so a failed save never binds the slot to a file that does not
        // exist on disk (the slot KV is unchanged either way)
        auto write = snapshot_io_write(filepath, std::move(data), deadline_ms);
        if (write.busy) {
            return make_error("snapshot io worker is busy, retry", ERROR_TYPE_UNAVAILABLE);
        }
        if (write.timed_out) {
            return make_error("snapshot save timed out, retry the save to confirm", ERROR_TYPE_UNAVAILABLE);
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

        // copy metadata: same shape as server_task_result_slot_copy::to_json
        // (that object was consumed building data above; a move preserves sizes)
        return make_ok({ { "id_slot", id_slot }, { "n_tokens", data.tokens.size() }, { "n_bytes", data.kv.size() } }, 201);
    }
}

server_http_res_ptr server_instances::handle_get_instance_snapshots(const server_http_req & req) {
    const std::string name = req.get_param("name");
    auto              inst = get_instance(name);
    if (!inst) {
        return make_error("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND);
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
        return make_error("instance not found: '" + name + "'", ERROR_TYPE_NOT_FOUND);
    }

    // per-instance delete: the hashed-key scoped file, the previous-key scoped
    // file (pre-hashing saves), then the legacy flat file (migration). each
    // removed file unbinds slots: the named instance's slots for its own files,
    // every instance's slots for a legacy file (which any instance may have
    // restored before scoping existed).
    const std::string inst_path = snapshot_instance_path(name, snapshot);
    const std::string prev_path = snapshot_instance_path_prev(name, snapshot);
    const std::string leg_path  = snapshot_legacy_path(snapshot);

    std::error_code ec;
    const bool removed_inst = std::filesystem::remove(inst_path, ec);
    ec.clear();
    const bool removed_prev = std::filesystem::remove(prev_path, ec);
    ec.clear();
    const bool removed_leg = std::filesystem::remove(leg_path, ec);

    if (!removed_inst && !removed_prev && !removed_leg) {
        return make_error("snapshot not found: '" + snapshot + "'", ERROR_TYPE_NOT_FOUND);
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


std::shared_ptr<server_instance> server_instances::default_instance() {
    // one implementation (see the const overload); the dispatch mutex is mutable
    return static_cast<const server_instances *>(this)->default_instance();
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

// the shared prologue of every default-instance endpoint: resolve the default,
// build it on demand, guard against a racing destroy/resize, then run the
// instance's own route handler. custom endpoints (health, props, slots,
// models) keep their own bodies.
server_http_res_ptr server_instances::default_instance_forward(const server_http_req & req,
                                                               server_http_context::handler_t server_routes::* method) {
    auto inst = default_instance();
    if (!inst) {
        return make_error("no instances loaded", ERROR_TYPE_SERVER);
    }
    if (auto err = ensure_built_instance(inst)) {
        return err;
    }
    active_route_guard guard(*this, *inst);
    if (!guard.acquired) {
        return make_error("instance '" + inst->cfg.name + "' is being reconfigured", ERROR_TYPE_UNAVAILABLE);
    }
    return (inst->routes.get()->*method)(req);
}

server_http_res_ptr server_instances::make_error(const std::string & message, error_type type) const {
    auto res          = std::make_unique<server_http_res>();
    res->status       = error_status(type);
    res->content_type = "application/json; charset=utf-8";
    // a 503 here is always a transient condition (an instance being reconfigured, a busy snapshot I/O
    // worker, an op that outran its deadline), so it is retriable. A retrying client needs to be told
    // how long to wait; without the header it either spins or backs off by a guess.
    if (res->status == 503) {
        res->headers["Retry-After"] = "1";
    }
    res->data         = safe_json_to_str({
        { "error", format_error_response(message, type) }
    });
    return res;
}

server_http_res_ptr server_instances::make_error_from_result(server_task_result & result) {
    auto * err = dynamic_cast<server_task_result_error *>(&result);
    // the error class travels with the result: the body is the class's own render and
    // the status comes from the same table, never from a field of that body. a result
    // that is not an error means the caller reached an error branch unexpectedly.
    auto res          = std::make_unique<server_http_res>();
    res->status       = err != nullptr ? error_status(err->err_type) : error_status(ERROR_TYPE_SERVER);
    res->content_type = "application/json; charset=utf-8";
    res->data         = safe_json_to_str({
        { "error", err != nullptr ? err->to_json() :
                                    format_error_response("unexpected instance task result", ERROR_TYPE_SERVER) }
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

