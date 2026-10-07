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

// adapters: the pool registry plus per-instance effective adapter lists.
// lock contract: every entry point requires mutex_mgmt held by the caller.

// read-back agreement for an adapter-set switch: the scheduler's committed set
// must equal the manager's expected list. on drift the expected set is posted
// once more; false = still diverged or a bound expired (caller answers
// retriable). fingerprints are computed on demand, so a successful re-apply
// makes every later fingerprint observe the corrected set.
static bool verify_attached(server_instances & mgr, server_instance & inst, int64_t deadline_ms) {
    // effective.lora_adapters: callers hold mutex_mgmt and the instance_drain_guard
    server_context_adapter_ops adapters(*inst.ctx_server);
    auto installed = adapters.get_lora_adapters(deadline_ms);
    if (installed && are_lora_sets_identical(*installed, inst.effective.lora_adapters)) {
        return true;
    }
    auto result = mgr.manager_apply_adapters(inst, inst.effective.lora_adapters, deadline_ms);
    if (!result || result->is_error()) {
        return false;
    }
    installed = adapters.get_lora_adapters(deadline_ms);
    return installed && are_lora_sets_identical(*installed, inst.effective.lora_adapters);
}


std::string server_instances::adapter_key(const std::string & path) {
    // canonical first: one file, one key across relative, dot-dot and
    // symlink spellings. anything the filesystem cannot resolve (missing,
    // loop, permission) keeps the lexical fallback, so the load fails later
    // with the unloadable-adapter error instead of here. never throws.
    try {
        return std::filesystem::canonical(path).string();
    } catch (const std::filesystem::filesystem_error &) {
    }
    try {
        return std::filesystem::absolute(path).lexically_normal().string();
    } catch (const std::filesystem::filesystem_error &) {
        return path;
    }
}

bool server_instances::adapter_identity_equal(const std::string & a, const std::string & b) {
    if (a.find('\0') != std::string::npos || b.find('\0') != std::string::npos) {
        return false;
    }
    // both spellings must name something readable; anything else (missing,
    // dangling, loop) is never a merge. equivalent() is the OS sameness
    // check (inode on POSIX, file index on Windows), so a true answer can
    // never join distinct files.
    std::error_code ec;
    if (!std::filesystem::exists(a, ec) || ec || !std::filesystem::exists(b, ec) || ec) {
        return false;
    }
    try {
        return std::filesystem::equivalent(a, b);
    } catch (const std::filesystem::filesystem_error &) {
        return false;
    }
}

std::optional<std::string> server_instances::try_adapter_key(const std::string & path) {
    // absolute() is purely lexical on common platforms, so an embedded NUL
    // would sail through into registry keys, fingerprints and declarations
    // that can never load. reject it here, before any of that happens.
    if (path.find('\0') != std::string::npos) {
        return std::nullopt;
    }
    try {
        return std::filesystem::absolute(path).lexically_normal().string();
    } catch (const std::filesystem::filesystem_error &) {
        return std::nullopt;
    }
}

llama_adapter_lora * server_instances::ensure_adapter(const std::string & path, std::string & entry_key) {
    // NUL guard first: adapter_key never fails, so a NUL path would sail
    // through into a registry key for a file that can never load
    if (!try_adapter_key(path)) {
        return nullptr;
    }
    // canonical key: one file, one entry across relative, dot-dot and symlink
    // spellings. missing files keep the lexical fallback and fail below at
    // llama_adapter_lora_init with the unloadable-adapter error.
    const std::string key = adapter_key(path);
    auto it = adapter_registry.find(key);
    if (it != adapter_registry.end()) {
        it->second.refcount++;
        entry_key = it->first;
        return it->second.adapter.get();
    }
    // same file under another spelling (hardlink, or a spelling that resolved
    // after the entry was stored): share the entry instead of loading twice.
    // the scan runs only on a registry miss; registries stay tiny. the entry
    // keeps its key (first spelling wins), so release pairing still matches.
    for (auto & kv : adapter_registry) {
        if (adapter_identity_equal(kv.first, key)) {
            kv.second.refcount++;
            entry_key = kv.first;
            return kv.second.adapter.get();
        }
    }
    llama_adapter_lora_ptr adapter;
    adapter.reset(llama_adapter_lora_init(model, key.c_str()));
    if (adapter == nullptr) {
        IST_ERR("failed to load lora adapter '%s'\n", key.c_str());
        return nullptr;
    }
    llama_adapter_lora * ptr = adapter.get();
    adapter_registry[key] = { std::move(adapter), 1 };
    entry_key = key;
    return ptr;
}

void server_instances::release_adapter(const std::string & key) {
    auto it = adapter_registry.find(key);
    if (it == adapter_registry.end()) {
        return;
    }
    if (--it->second.refcount == 0) {
        adapter_registry.erase(it);
    }
}

void server_instances::release_adapter_set(const std::vector<common_adapter_lora_info> & loras) {
    for (const auto & la : loras) {
        if (la.ptr != nullptr) {
            release_adapter(adapter_key(la.path));
        }
    }
}

std::optional<std::vector<common_adapter_lora_info>> server_instances::resolve_adapter_set(const common_instance & cfg) {
    std::vector<common_adapter_lora_info> resolved = common_instance_params(params, cfg).lora_adapters;
    for (auto & la : resolved) {
        // the entry key is the one spelling per file, so the snapshot
        // fingerprint is stable across restarts and re-resolution
        std::string entry_key;
        llama_adapter_lora * ptr = ensure_adapter(la.path, entry_key);
        if (ptr == nullptr) {
            release_adapter_set(resolved);
            return std::nullopt;
        }
        la.path = entry_key;
        la.ptr  = ptr;
        // the report-only meta travels with the mirror so a GET shape stays
        // consistent whether the entry came from a build or a refresh
        common_adapter_lora_fill_meta(la);
    }
    return resolved;
}

uint64_t server_instances::instance_adapter_bytes(const server_instance & inst) const {
    uint64_t bytes = 0;
    for (const auto & la : inst.effective.lora_adapters) {
        bytes += llama_adapter_lora_buf_size(la.ptr);
    }
    return bytes;
}


bool server_instances::refresh_effective_from_scheduler(server_instance & inst, int64_t deadline_ms) {
    if (!inst.ctx_server) {
        return false;
    }
    auto committed = server_context_adapter_ops(*inst.ctx_server).get_lora_adapters(deadline_ms);
    if (!committed) {
        return false;
    }
    // one committed list feeds both forms: the effective list (ptrs, meta) and
    // the grammar list (path + scale) can therefore never drift apart
    inst.effective.lora_adapters = std::move(*committed);
    inst.cfg.lora.clear();
    inst.cfg.lora.reserve(inst.effective.lora_adapters.size());
    for (const auto & la : inst.effective.lora_adapters) {
        inst.cfg.lora.emplace_back(la.path, la.scale);
    }
    return true;
}

server_task_result_ptr server_instances::manager_apply_adapters(server_instance & inst,
                                                               std::vector<common_adapter_lora_info> adapters,
                                                               int64_t deadline_ms) {
    // admission before the queue: the drain flag is the proof that no slot is
    // decoding and no save/restore can interleave, and running/terminated
    // proves the scheduler is still (or again) alive to take the post: a swap
    // racing terminate() or a destroy fails fast here instead of queueing onto
    // a dying scheduler and burning the deadline. read under the dispatch
    // lock (the caller already holds mgmt, so the order stays mgmt -> dispatch).
    // the only admitted caller holds the drain guard (see swap_adapter_set),
    // so removing is true here; the running check is the teardown race.
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        if (!inst.removing) {
            auto rejected = std::make_unique<server_task_result_error>();
            rejected->err_type = ERROR_TYPE_SERVER;
            rejected->err_msg  = "adapter swap needs the instance drain";
            return rejected;
        }
        if (!inst.running || terminated) {
            auto rejected = std::make_unique<server_task_result_error>();
            rejected->err_type = ERROR_TYPE_UNAVAILABLE;
            rejected->err_msg  = "instance '" + inst.cfg.name + "' is being reconfigured";
            return rejected;
        }
    }
    return server_context_adapter_ops(*inst.ctx_server).set_lora_adapters(std::move(adapters), deadline_ms);
}

server_http_res_ptr server_instances::swap_adapter_set(const std::shared_ptr<server_instance> & inst,
                                                       const adapter_set_mutator &               mutate) {
    // exclusive access while the set changes: the drain rejects new dispatches and
    // waits out in-flight generations, so the swap below never frees adapter tensors
    // under a decoding slot (use-after-free) nor corrupts its output. the drain
    // comes before the mutation because apply_snapshot reads the effective list
    // without the management lock while its dispatch is counted.
    instance_drain_guard guard(*this, inst);

    // the scheduler owns the live set (a legacy POST writes it directly), so the
    // mirror is refreshed before the mutation: an attach/detach is then additive
    // over what the scheduler actually holds instead of reverting a legacy scale.
    if (!refresh_effective_from_scheduler(*inst, snapshot_deadline_ms(*inst))) {
        return make_error("could not read the committed adapter set", ERROR_TYPE_UNAVAILABLE);
    }

    // rollback copies: the lists stay untouched until the synchronous apply succeeds
    const auto cfg_prev       = inst->cfg.lora;
    const auto effective_prev = inst->effective.lora_adapters;

    adapter_set_delta delta;
    if (auto err = mutate(inst->cfg.lora, inst->effective.lora_adapters, delta)) {
        return err;
    }

    // synchronous apply through the guarded manager path (the drain above is
    // the admission proof): posted directly to this instance's queue, never
    // through dispatch(), which the guard's removing flag would reject. the
    // drain makes the scheduler quiescent, so the task runs immediately; the
    // deadline only trips on a stalled scheduler.
    auto result = manager_apply_adapters(*inst, inst->effective.lora_adapters, snapshot_deadline_ms(*inst));
    if (!result || result->is_error()) {
        inst->cfg.lora                = cfg_prev;
        inst->effective.lora_adapters = effective_prev;
        release_adapter_set(delta.acquired);
        return make_error("timed out applying the adapter set", ERROR_TYPE_UNAVAILABLE);
    }
    // read-back agreement: the committed set must equal the expected list.
    // no rollback here: the set above succeeded, so the lists already describe
    // the scheduler; a failed read-back is answered retriable and the idempotent
    // retry re-verifies.
    if (!verify_attached(*this, *inst, snapshot_deadline_ms(*inst))) {
        return make_error("could not verify the adapter set", ERROR_TYPE_UNAVAILABLE);
    }
    release_adapter_set(delta.released); // frees at refcount 0 while the model is alive

    // the swap invalidated every slot's live KV (the scheduler cleared the prompts)
    // AND the recorded bindings: bound slots would otherwise keep serving KV computed
    // under the old adapters (resize semantics)
    {
        std::lock_guard<std::mutex> lock(mutex_dispatch);
        inst->slot_snapshots.assign(inst->slot_snapshots.size(), std::string());
    }
    return nullptr;
}

