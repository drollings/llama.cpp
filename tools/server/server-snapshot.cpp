#include "server-snapshot.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace {

constexpr uint32_t SNAPSHOT_MAGIC   = 0x534C5041;  // "SLPA"
constexpr uint32_t SNAPSHOT_VERSION = 2;
constexpr size_t   SNAPSHOT_HEADER_V1 = 24;  // magic + version + n_ctx_seq + n_tokens + kv_size
constexpr size_t   SNAPSHOT_HEADER_V2 = 28;  // v1 header + adapter_fp_len (fp bytes follow)
// max adapter fingerprint accepted from a file header. the writer emits a short
// hex hash (16 chars); anything larger is corrupt. bounds the header-driven
// allocation below so a crafted fp_len cannot force a huge resize.
constexpr uint32_t SNAPSHOT_FP_MAX = 1024;

// write the header followed by the payload. returns the payload offset on
// success, or 0 if the payload does not fit int32.
size_t write_snapshot(std::ostream & out, const server_snapshot_data & data) {
    if (data.tokens.size() > (size_t) INT32_MAX || data.adapter_fp.size() > (size_t) UINT32_MAX) {
        return 0;
    }
    const uint64_t kv_size = (uint64_t) data.kv.size();
    const int32_t  n_tokens = (int32_t) data.tokens.size();
    const uint32_t fp_len = (uint32_t) data.adapter_fp.size();

    out.write((const char *) &SNAPSHOT_MAGIC, 4);
    out.write((const char *) &SNAPSHOT_VERSION, 4);
    out.write((const char *) &data.n_ctx_seq, 4);
    out.write((const char *) &n_tokens, 4);
    out.write((const char *) &kv_size, 8);
    out.write((const char *) &fp_len, 4);
    if (fp_len > 0) {
        out.write(data.adapter_fp.data(), (std::streamsize) fp_len);
    }

    for (const llama_token t : data.tokens) {
        out.write((const char *) &t, sizeof(t));
    }
    if (!data.kv.empty()) {
        out.write((const char *) data.kv.data(), (std::streamsize) data.kv.size());
    }
    return SNAPSHOT_HEADER_V2 + fp_len;
}

int64_t file_mtime_unix(const std::filesystem::path & path) {
    std::error_code ec;
    const auto      age = std::chrono::duration_cast<std::chrono::seconds>(
        std::filesystem::file_time_type::clock::now() - std::filesystem::last_write_time(path, ec));
    if (ec) {
        return 0;
    }
    return (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count() -
           age.count();
}

}  // namespace

// validate the header and the exact file size; classify the outcome. never throws.
server_snapshot_read_out server_snapshot_read_status(const std::string & path) {
    server_snapshot_read_out out;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        out.status = server_snapshot_status::MISSING;
        return out;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    int32_t  n_ctx_seq = 0;
    int32_t  n_tokens = 0;
    uint64_t kv_size = 0;
    in.read((char *) &magic, 4);
    in.read((char *) &version, 4);
    in.read((char *) &n_ctx_seq, 4);
    in.read((char *) &n_tokens, 4);
    in.read((char *) &kv_size, 8);
    if (!in || magic != SNAPSHOT_MAGIC || (version != 1 && version != 2) || n_tokens < 0) {
        out.status = server_snapshot_status::CORRUPT;
        return out;
    }

    // v2 carries the adapter fingerprint between kv_size and the payload; v1 has
    // no such field and reads as fp = "".
    std::string adapter_fp;
    size_t      header = SNAPSHOT_HEADER_V1;
    if (version == 2) {
        uint32_t fp_len = 0;
        in.read((char *) &fp_len, 4);
        if (!in || fp_len > SNAPSHOT_FP_MAX) {
            out.status = server_snapshot_status::CORRUPT;
            return out;
        }
        adapter_fp.resize(fp_len);
        if (fp_len > 0) {
            in.read(adapter_fp.data(), (std::streamsize) fp_len);
            if (!in) {
                out.status = server_snapshot_status::CORRUPT;
                return out;
            }
        }
        header = SNAPSHOT_HEADER_V2 + fp_len;
    }

    // exact size check: header + fp + tokens + kv. any leftover or shortfall
    // means the file is corrupt or truncated.
    const uint64_t expect = (uint64_t) header + (uint64_t) n_tokens * sizeof(llama_token) + kv_size;
    in.seekg(0, std::ios::end);
    if (in.tellg() != (std::streampos) expect) {
        out.status = server_snapshot_status::CORRUPT;
        return out;
    }
    // back to the payload for the reads below
    in.seekg((std::streamoff) header, std::ios::beg);

    server_snapshot_data data;
    data.version    = version;
    data.n_ctx_seq  = n_ctx_seq;
    data.adapter_fp = std::move(adapter_fp);
    data.tokens.resize((size_t) n_tokens);
    for (llama_token & t : data.tokens) {
        in.read((char *) &t, sizeof(t));
    }
    data.kv.resize((size_t) kv_size);
    if (kv_size > 0) {
        in.read((char *) data.kv.data(), (std::streamsize) kv_size);
    }
    if (!in) {
        out.status = server_snapshot_status::CORRUPT;
        return out;
    }
    out.status = server_snapshot_status::OK;
    out.data   = std::move(data);
    return out;
}

bool server_snapshot_write(const std::string & path, const server_snapshot_data & data) {
    const std::string tmp_path = path + ".tmp";
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            return false;
        }
        if (write_snapshot(out, data) == 0) {
            out.close();
            std::filesystem::remove(tmp_path);
            return false;
        }
        // buffered flush failures surface at close: a short disk write must
        // never rename a truncated .tmp over the target
        out.close();
        if (!out) {
            std::error_code ec;
            std::filesystem::remove(tmp_path, ec);
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        // rename failed (e.g. cross-device); fall back to remove-on-failure so no
        // .tmp is left behind
        std::filesystem::remove(tmp_path, ec);
        return false;
    }
    return true;
}

std::optional<server_snapshot_data> server_snapshot_read(const std::string & path) {
    return server_snapshot_read_status(path).data;
}

// join helper: exactly one '/' between segments regardless of a trailing slash
// on the root (slot_save_path may or may not end in '/').
static std::string snapshot_join(const std::string & root, const std::string & rest) {
    if (!root.empty() && root.back() == '/') {
        return root + rest;
    }
    return root + "/" + rest;
}

std::string server_snapshot_model_key(const std::string & base_name) {
    std::string key = base_name;
    std::replace(key.begin(), key.end(), '/', '_');
    std::replace(key.begin(), key.end(), ':', '_');
    return key;
}

std::string server_snapshot_model_key_hashed(const std::string & base_name) {
    // FNV-1a 32-bit over the original name: deterministic across runs and
    // platforms (unlike std::hash), 8 hex chars, filesystem-safe. a collision
    // merely reproduces the old shared-directory behavior, never corruption.
    uint32_t hash = 2166136261u;
    for (const char c : base_name) {
        hash ^= (uint8_t) c;
        hash *= 16777619u;
    }
    char suffix[9];
    snprintf(suffix, sizeof(suffix), "%08x", hash);
    return server_snapshot_model_key(base_name) + "-" + suffix;
}

std::string server_snapshot_instance_dir(const std::string & slot_save_path,
                                         const std::string & model_key,
                                         const std::string & instance) {
    return snapshot_join(slot_save_path, model_key + "/" + instance);
}

std::string server_snapshot_instance_path(const std::string & slot_save_path,
                                          const std::string & model_key,
                                          const std::string & instance,
                                          const std::string & snapshot) {
    return snapshot_join(slot_save_path, model_key + "/" + instance + "/" + snapshot + ".bin");
}

std::string server_snapshot_legacy_path(const std::string & slot_save_path,
                                        const std::string & model_key,
                                        const std::string & snapshot) {
    return snapshot_join(slot_save_path, model_key + "/" + snapshot + ".bin");
}

std::vector<server_snapshot_meta> server_snapshot_list(const std::string & dir) {
    std::vector<server_snapshot_meta> out;
    std::error_code                   ec;
    if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec)) {
        return out;
    }

    for (const auto & entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) {
            break; // the iteration itself failed; entries collected so far stand
        }
        std::error_code fec;
        if (!entry.is_regular_file(fec) || fec || entry.path().extension() != ".bin") {
            continue;
        }
        const std::string fname = entry.path().filename().string();
        const std::string name  = fname.substr(0, fname.size() - 4);  // strip ".bin"

        // header-only read for n_ctx_seq + adapter_fp + version; an unreadable
        // header reports version 0 with zeroed fields (same bucket as the read
        // path's CORRUPT, including negative n_tokens)
        int32_t     n_ctx_seq  = 0;
        std::string adapter_fp;
        uint32_t    version    = 0;
        {
            std::ifstream in(entry.path(), std::ios::binary);
            uint32_t      magic = 0;
            uint32_t      v     = 0;
            int32_t       n_tokens = 0;
            in.read((char *) &magic, 4);
            in.read((char *) &v, 4);
            in.read((char *) &n_ctx_seq, 4);
            in.read((char *) &n_tokens, 4);
            bool ok = (bool) in && magic == SNAPSHOT_MAGIC && (v == 1 || v == 2) && n_tokens >= 0;
            if (ok) {
                version = v;
            }
            if (ok && version == 2) {
                // skip kv_size, then read the fp (offsets 16-24 + fp at 28)
                uint64_t kv_size = 0;
                uint32_t fp_len  = 0;
                in.read((char *) &kv_size, 8);
                in.read((char *) &fp_len, 4);
                ok = (bool) in && fp_len <= SNAPSHOT_FP_MAX;
                if (ok && fp_len > 0) {
                    adapter_fp.resize(fp_len);
                    in.read(adapter_fp.data(), (std::streamsize) fp_len);
                    ok = (bool) in;
                }
            }
            if (!ok) {
                version    = 0;
                n_ctx_seq  = 0;
                adapter_fp.clear();
            }
        }

        // a file that vanished between iteration and stat carries no size: skip
        // it rather than listing a row with a bogus size
        const uint64_t fsize = entry.file_size(ec);
        if (ec) {
            continue;
        }
        out.push_back({
            name,
            fsize,
            file_mtime_unix(entry.path()),
            n_ctx_seq,
            adapter_fp,
            version,
        });
    }
    return out;
}
