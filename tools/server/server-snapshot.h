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
//   4       4     version = 1
//   8       4     n_ctx_seq   (int32)
//   12      4     n_tokens    (int32)
//   16      8     kv_size     (uint64)
//   24      ...   tokens:  n_tokens * int32
//   ...     ...   kv:      kv_size bytes

struct server_snapshot_data {
    llama_tokens         tokens;
    std::vector<uint8_t> kv;
    int32_t              n_ctx_seq = 0;
};

// atomic: write <path>.tmp, then rename over <path>. returns false on I/O error
// (the .tmp is removed on failure). caller serializes concurrent writes per file.
bool server_snapshot_write(const std::string & path, const server_snapshot_data & data);

// file-state classifier for the snapshot module. MISSING = the file does not open;
// CORRUPT = bad magic/version/size or a truncated payload.
enum class server_snapshot_status { OK, MISSING, CORRUPT };

struct server_snapshot_read_out {
    server_snapshot_status              status = server_snapshot_status::MISSING;
    std::optional<server_snapshot_data> data;
};

// validate magic/version and exact file size; carries MISSING vs CORRUPT so the caller
// can map to 404/400 without a separate existence check. never throws.
server_snapshot_read_out server_snapshot_read_status(const std::string & path);

// thin wrapper over server_snapshot_read_status: data on OK, nullopt otherwise.
// never throws.
std::optional<server_snapshot_data> server_snapshot_read(const std::string & path);

struct server_snapshot_meta {
    std::string name;
    uint64_t    size;
    int64_t     mtime;    // unix seconds
    int32_t     n_ctx_seq;
};

// list *.bin in dir, header-parsed n_ctx_seq (read header only), mtime in unix
// seconds. skips unreadable files (n_ctx_seq = 0 on an unreadable header).
std::vector<server_snapshot_meta> server_snapshot_list(const std::string & dir);
