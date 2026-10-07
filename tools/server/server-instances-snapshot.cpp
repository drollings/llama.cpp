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

// snapshots: on-disk files, the pool I/O worker, per-slot bindings.
// lock contract: bindings mutate under mutex_dispatch; file I/O runs on the
// pool worker, never on a scheduler thread.

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
    // notify only when the count changed: a failed acquisition waited out its
    // deadline without taking a slot, so there is nothing to hand over
    if (acquired) {
        std::lock_guard<std::mutex> lock(mgr.mutex_switch);
        mgr.n_active_switches--;
        mgr.cond_switch.notify_all();
    }
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

server_instances::kv_copy_out server_instances::kv_copy_to_data(server_instance &   inst,
                                                                  int                 id_slot,
                                                                  int64_t             deadline_ms,
                                                                  const std::string & timeout_msg) {
    kv_copy_out out;
    auto result = server_context_snapshot_ops(*inst.ctx_server).slot_save_copy(id_slot, deadline_ms);
    if (!result) {
        out.error = make_error(timeout_msg, ERROR_TYPE_UNAVAILABLE);
        return out;
    }
    if (result->is_error()) {
        // a busy slot fails with a retriable 503 instead of deferring
        out.error = make_error_from_result(*result);
        return out;
    }
    auto * copy = dynamic_cast<server_task_result_slot_copy *>(result.get());
    GGML_ASSERT(copy != nullptr);
    out.data.n_ctx_seq = inst.ctx_server->get_slot_n_ctx();
    out.data.tokens    = copy->tokens;
    out.data.kv        = std::move(copy->buffer);
    // record the adapter set this KV was computed under; a restore under a
    // different set is rejected. the fingerprint read is serialized against
    // attach/detach: the drain waits out this dispatch's n_active_dispatch.
    out.data.adapter_fp = common_lora_fingerprint(inst.effective.lora_adapters);
    out.ok = true;
    return out;
}

server_http_res_ptr server_instances::apply_snapshot(server_instance &   inst,
                                                     const std::string & snapshot,
                                                     int                 id_slot) {
    if (!fs_validate_filename(snapshot)) {
        return make_error("invalid snapshot name", ERROR_TYPE_INVALID_REQUEST);
    }
    if (params.slot_save_path.empty()) {
        return make_error("snapshot switching requires --slot-save-path", ERROR_TYPE_NOT_SUPPORTED);
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
        return make_error("snapshot not found: '" + snapshot + "'", ERROR_TYPE_NOT_FOUND);
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

    // 4. persist the slot's current KV (if any) before switching away from it.
    //    the write runs awaited on the single FIFO pool I/O worker, which
    //    serializes all file ops; the deadline bounds the wait (see below).
    if (!current.empty()) {
        auto copied = kv_copy_to_data(inst, id_slot, deadline_ms, "snapshot switch timed out while saving");
        if (!copied.ok) {
            return std::move(copied.error);
        }
        server_snapshot_data data = std::move(copied.data);
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
            return make_error("snapshot switch timed out while saving, retry the switch to confirm", ERROR_TYPE_UNAVAILABLE);
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
        return make_error("snapshot read timed out, retry the switch to confirm", ERROR_TYPE_UNAVAILABLE);
    }
    if (read.status == server_snapshot_status::MISSING) {
        return make_error("snapshot not found: '" + snapshot + "'", ERROR_TYPE_NOT_FOUND);
    }
    if (read.status == server_snapshot_status::CORRUPT || !read.data) {
        return make_error("corrupt or unreadable snapshot: '" + snapshot + "'", ERROR_TYPE_INVALID_REQUEST);
    }

    // 6. a snapshot saved under a different context size is incompatible with this slot
    if (read.data->n_ctx_seq != inst.ctx_server->get_slot_n_ctx()) {
        return make_error("snapshot context size does not match this slot", ERROR_TYPE_INVALID_REQUEST);
    }

    // 6b. a snapshot saved under a different adapter set is incompatible with this
    // instance: the KV was computed under different tensors. the file version
    // decides what an empty fingerprint means: v1 never had the field
    // (warn-allow, legacy), while v2 always stamps it, so a v2 empty fp means
    // "saved with zero adapters" and must match exactly. the fingerprint read
    // is serialized against attach/detach (see step 4).
    if (!read.data->adapter_fp.empty() || read.data->version > 1) {
        const std::string current_fp = common_lora_fingerprint(inst.effective.lora_adapters);
        if (read.data->adapter_fp != current_fp) {
            return make_error("snapshot adapter set does not match this instance", ERROR_TYPE_INVALID_REQUEST);
        }
    } else {
        IST_WRN("snapshot '%s' has no adapter fingerprint (pre-v2 file), skipping the adapter check\n",
                snapshot.c_str());
    }

    // 7. apply the host KV buffer on the scheduler; busy/timeout -> 503, apply failure
    //    clears the slot so it is never left partially loaded
    auto result = server_context_snapshot_ops(*inst.ctx_server).slot_restore_apply(id_slot, std::move(read.data->kv), read.data->tokens, deadline_ms);
    if (!result) {
        clear_slot_binding(inst, id_slot);
        return make_error("snapshot switch timed out while restoring", ERROR_TYPE_UNAVAILABLE);
    }
    if (result->is_error()) {
        clear_slot_binding(inst, id_slot);
        return make_error_from_result(*result);
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


// n_ctx_seq, adapter_fp, instance}. the tag is the owning instance name, or
// null for legacy flat files.
static void push_snapshot_row(std::vector<json> & entries, const server_snapshot_meta & meta, const json & instance_tag) {
    entries.push_back({
        { "name",       meta.name       },
        { "size",       meta.size       },
        { "mtime",      meta.mtime      },
        { "n_ctx_seq",  meta.n_ctx_seq  },
        { "adapter_fp", meta.adapter_fp },
        { "version",    meta.version    },
        { "instance",   instance_tag    },
    });
}

// list one model-key directory: legacy flat files (null instance tag) plus every
// instance subdirectory (a snapshot outlives its instance, so every subdirectory
// is listed for cold-pool discoverability, not just live instances). `seen`
// dedupes scoped entries already listed from the newer (hashed) key so a
// re-saved snapshot appears once; legacy rows predate hashing and are never
// deduped.
static void list_snapshot_key_dir(const std::string & dir,
                                  bool include_legacy,
                                  const std::function<void(const server_snapshot_meta &, const json &)> & push,
                                  std::set<std::string> & seen) {
    if (include_legacy) {
        for (const auto & meta : server_snapshot_list(dir)) {
            push(meta, nullptr);
        }
    }
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
            if (!seen.insert(sub + '\0' + meta.name).second) {
                continue;
            }
            push(meta, sub);
        }
    }
}

// null-safe string field read for ordering (common_json::value() throws
// on a null, and legacy snapshot entries are explicitly tagged null).
static std::string snapshot_str_field(const json & e, const std::string & key) {
    if (!e.is_object() || !e.contains(key) || !e.at(key).is_string()) {
        return "";
    }
    return e.at(key).get<std::string>();
}


json server_instances::pool_snapshots_json() const {
    std::vector<json> entries;
    if (!params.slot_save_path.empty()) {
        const std::string model_key     = server_snapshot_model_key(base_name);
        const std::string model_key_new = server_snapshot_model_key_hashed(base_name);
        const std::string dir           = params.slot_save_path + model_key;
        const std::string dir_new       = params.slot_save_path + model_key_new;
        const auto push = [&entries](const server_snapshot_meta & meta, const json & instance_tag) {
            push_snapshot_row(entries, meta, instance_tag);
        };
        // a snapshot re-saved after the key change exists under both keys; the
        // hashed key is listed first so its row wins and the older one is skipped.
        // legacy flat files predate hashing and live only under the old key.
        std::set<std::string> seen_scoped;
        list_snapshot_key_dir(dir_new, false, push, seen_scoped);
        // legacy flat files (pre-per-instance layout), tagged with a null instance
        list_snapshot_key_dir(dir, true, push, seen_scoped);
    }
    // deterministic envelope: order by (instance, name), legacy (null) first
    std::sort(entries.begin(), entries.end(), [](const json & a, const json & b) {
        const std::string ai = snapshot_str_field(a, "instance");
        const std::string bi = snapshot_str_field(b, "instance");
        return ai != bi ? ai < bi : snapshot_str_field(a, "name") < snapshot_str_field(b, "name");
    });
    json snapshots = json::array();
    for (auto & e : entries) {
        snapshots.push_back(std::move(e));
    }
    return snapshots;
}

// snapshots visible to one instance: its own instance-scoped directory (new key
// first, then the previous key) plus the legacy flat files (migration read
// path, tagged with a null instance).
json server_instances::instance_snapshots_json(const std::string & instance) const {
    std::vector<json> entries;
    if (!params.slot_save_path.empty()) {
        const std::string model_key     = server_snapshot_model_key(base_name);
        const std::string model_key_new = server_snapshot_model_key_hashed(base_name);
        const std::string dir           = params.slot_save_path + model_key;
        const std::string dir_new       = params.slot_save_path + model_key_new;
        const auto push_scoped = [&entries, &instance](const server_snapshot_meta & meta) {
            push_snapshot_row(entries, meta, instance);
        };
        // the hashed key wins over the previous key for re-saved snapshots
        std::set<std::string> seen;
        for (const auto & meta : server_snapshot_list(dir_new + "/" + instance)) {
            seen.insert(meta.name);
            push_scoped(meta);
        }
        for (const auto & meta : server_snapshot_list(dir + "/" + instance)) {
            if (!seen.insert(meta.name).second) {
                continue;
            }
            push_scoped(meta);
        }
        for (const auto & meta : server_snapshot_list(dir)) {
            push_snapshot_row(entries, meta, nullptr);
        }
    }
    std::stable_sort(entries.begin(), entries.end(), [](const json & a, const json & b) {
        return snapshot_str_field(a, "name") < snapshot_str_field(b, "name");
    });
    json snapshots = json::array();
    for (auto & e : entries) {
        snapshots.push_back(std::move(e));
    }
    return snapshots;
}


std::string server_instances::snapshot_instance_path(const std::string & instance,
                                                     const std::string & snapshot) const {
    // the only write target: the hashed key, so identities that sanitize alike
    // never share a directory
    return server_snapshot_instance_path(params.slot_save_path, server_snapshot_model_key_hashed(base_name),
                                         instance, snapshot);
}

std::string server_instances::snapshot_instance_path_prev(const std::string & instance,
                                                          const std::string & snapshot) const {
    // read fallback for files written before key hashing (same shape as the
    // scoped -> legacy fallback below)
    return server_snapshot_instance_path(params.slot_save_path, server_snapshot_model_key(base_name),
                                         instance, snapshot);
}

std::string server_instances::snapshot_legacy_path(const std::string & snapshot) const {
    return server_snapshot_legacy_path(params.slot_save_path, server_snapshot_model_key(base_name), snapshot);
}

// resolve a snapshot for read (and save-back): the hashed-key scoped file wins,
// then the previous-key scoped file, then the legacy flat file (migration
// fallbacks). "" when none exists.
std::string server_instances::resolve_snapshot_path(const std::string & instance,
                                                    const std::string & snapshot) const {
    std::error_code ec;
    const std::string inst_path = snapshot_instance_path(instance, snapshot);
    if (std::filesystem::exists(inst_path, ec)) {
        return inst_path;
    }
    const std::string prev_path = snapshot_instance_path_prev(instance, snapshot);
    if (std::filesystem::exists(prev_path, ec)) {
        return prev_path;
    }
    const std::string leg_path = snapshot_legacy_path(snapshot);
    if (std::filesystem::exists(leg_path, ec)) {
        return leg_path;
    }
    return "";
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
        // no worker, no queue: before the single start or after the single stop a
        // posted job would never run, so refuse fast with a retriable error
        if (!io_running) {
            return std::nullopt;
        }
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
    if (io_running) {
        return;
    }
    io_stop    = false;
    io_running = true;
    if (!io_thread.joinable()) {
        io_thread = std::thread([this]() { io_loop(); });
    }
}

void server_instances::stop_io_worker() {
    {
        std::lock_guard<std::mutex> lock(mutex_io);
        if (!io_running) {
            return;
        }
        io_running = false;
        io_stop    = true;
        cond_io.notify_all();
    }
    // the loop drains any pending jobs before it returns, so an HTTP thread waiting
    // on a job future is never left hanging at shutdown
    if (io_thread.joinable()) {
        io_thread.join();
    }
}

