#pragma once

#include "common.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// server_snapshot: the single owner of the on-disk .bin snapshot format.
// pure file I/O - no HTTP, no scheduler, no pool state. the format is
// model-agnostic: it stores a raw KV byte buffer plus the prompt tokens that
// match it, so any context size / n_ctx_seq is representable.
//
// on-disk layout (plain, host-endian):
//   offset  size  field
//   0       4     magic   = 0x534C5041  ("SLPA")
//   4       4     version = 2
//   8       4     n_ctx_seq   (int32)
//   12      4     n_tokens    (int32)
//   16      8     kv_size     (uint64)
//   24      4     adapter_fp_len (uint32, v2 only)
//   28      ...   adapter_fp: adapter_fp_len bytes (v2 only)
//   ...     ...   tokens:  n_tokens * int32
//   ...     ...   kv:      kv_size bytes
//
// version 1 files have no adapter_fp field; readers treat them as fp = "".

struct server_snapshot_data {
    llama_tokens         tokens;
    std::vector<uint8_t> kv;
    int32_t              n_ctx_seq = 0;
    // fingerprint of the adapter set active when the snapshot was written
    // ("" = none / pre-v2 file)
    std::string adapter_fp;
    // on-disk version the data was read as (1 or 2). fresh data defaults to
    // the current write version; the restore gate needs the distinction: a
    // v1 empty fp is legacy, a v2 empty fp means "saved with zero adapters".
    uint32_t version = 2;
};

// atomic: write <path>.tmp, then rename over <path>. returns false on I/O error
// (the .tmp is removed on failure). caller serializes concurrent writes per file.
bool server_snapshot_write(const std::string & path, const server_snapshot_data & data);

// file-state classifier for the snapshot module. MISSING = the file does not open;
// CORRUPT = bad magic/version/size or a truncated payload (version > 2 included).
enum class server_snapshot_status { OK, MISSING, CORRUPT };

struct server_snapshot_read_out {
    server_snapshot_status              status = server_snapshot_status::MISSING;
    std::optional<server_snapshot_data> data;
};

// validate magic/version and exact file size; carries MISSING vs CORRUPT so the caller
// can map to 404/400 without a separate existence check. accepts version 1 (fp = "")
// and 2. never throws.
server_snapshot_read_out server_snapshot_read_status(const std::string & path);

// thin wrapper over server_snapshot_read_status: data on OK, nullopt otherwise.
// never throws.
std::optional<server_snapshot_data> server_snapshot_read(const std::string & path);

struct server_snapshot_meta {
    std::string name;
    uint64_t    size;
    int64_t     mtime;    // unix seconds
    int32_t     n_ctx_seq;
    std::string adapter_fp; // "" for v1 files / unreadable headers
    uint32_t    version = 0; // 1 or 2 from a readable header, 0 when unreadable
};

// list *.bin in dir: one row per file, header-parsed n_ctx_seq + adapter_fp +
// version (read header only), mtime in unix seconds. a file whose header does
// not parse still gets a row, with version 0 and zeroed fields, so one bad
// file can never hide or end the good rows; only a file that vanishes
// mid-listing (unstattable) is skipped.
std::vector<server_snapshot_meta> server_snapshot_list(const std::string & dir);

// pool identity to a filesystem-safe directory name: '/' and ':' become '_'.
// the legacy mapping, kept for reading snapshots written before key hashing;
// two distinct identities can sanitize alike ("a/b" vs "a:b"), so it must
// never be the write target for new files.
std::string server_snapshot_model_key(const std::string & base_name);
// collision-resistant write key: the legacy mapping plus a short deterministic
// hash of the original name, so identities that sanitize alike never share a
// directory. new saves always write under this key; readers try it first and
// fall back to the legacy key (same shape as the scoped -> flat fallback).
std::string server_snapshot_model_key_hashed(const std::string & base_name);
// per-instance on-disk layout. snapshots are scoped to their instance:
//   <slot_save_path>/<model_key>/<instance>/<snapshot>.bin
// where model_key is the sanitized pool identity and instance/snapshot are
// validated [A-Za-z0-9._-] names (safe as path segments, no traversal).
//
// files written before per-instance scoping live flat at
//   <slot_save_path>/<model_key>/<snapshot>.bin
// (legacy layout). readers fall back to it for migration; new saves always
// write the instance-scoped path.
std::string server_snapshot_instance_dir(const std::string & slot_save_path,
                                         const std::string & model_key,
                                         const std::string & instance);
std::string server_snapshot_instance_path(const std::string & slot_save_path,
                                          const std::string & model_key,
                                          const std::string & instance,
                                          const std::string & snapshot);
std::string server_snapshot_legacy_path(const std::string & slot_save_path,
                                        const std::string & model_key,
                                        const std::string & snapshot);
