#include "server-snapshot.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace {

constexpr uint32_t SNAPSHOT_MAGIC   = 0x534C5041;  // "SLPA"
constexpr uint32_t SNAPSHOT_VERSION = 1;
constexpr size_t   SNAPSHOT_HEADER  = 24;  // magic + version + n_ctx_seq + n_tokens + kv_size

// write the 24-byte header followed by the payload. returns the payload offset
// (== SNAPSHOT_HEADER) on success, or 0 if the payload does not fit int32.
size_t write_snapshot(std::ostream & out, const server_snapshot_data & data) {
    if (data.tokens.size() > (size_t) INT32_MAX) {
        return 0;
    }
    const uint64_t kv_size = (uint64_t) data.kv.size();
    const int32_t  n_tokens = (int32_t) data.tokens.size();

    out.write((const char *) &SNAPSHOT_MAGIC, 4);
    out.write((const char *) &SNAPSHOT_VERSION, 4);
    out.write((const char *) &data.n_ctx_seq, 4);
    out.write((const char *) &n_tokens, 4);
    out.write((const char *) &kv_size, 8);

    for (const llama_token t : data.tokens) {
        out.write((const char *) &t, sizeof(t));
    }
    if (!data.kv.empty()) {
        out.write((const char *) data.kv.data(), (std::streamsize) data.kv.size());
    }
    return SNAPSHOT_HEADER;
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
    if (!in || magic != SNAPSHOT_MAGIC || version != SNAPSHOT_VERSION || n_tokens < 0) {
        out.status = server_snapshot_status::CORRUPT;
        return out;
    }

    // exact size check: 24-byte header + tokens + kv. any leftover or shortfall
    // means the file is corrupt or truncated.
    const uint64_t expect = SNAPSHOT_HEADER + (uint64_t) n_tokens * sizeof(llama_token) + kv_size;
    in.seekg(0, std::ios::end);
    if (in.tellg() != (std::streampos) expect) {
        out.status = server_snapshot_status::CORRUPT;
        return out;
    }
    // back to the payload for the reads below
    in.seekg(SNAPSHOT_HEADER, std::ios::beg);

    server_snapshot_data data;
    data.n_ctx_seq = n_ctx_seq;
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

std::vector<server_snapshot_meta> server_snapshot_list(const std::string & dir) {
    std::vector<server_snapshot_meta> out;
    std::error_code                   ec;
    if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec)) {
        return out;
    }

    for (const auto & entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec) || entry.path().extension() != ".bin") {
            continue;
        }
        const std::string fname = entry.path().filename().string();
        const std::string name  = fname.substr(0, fname.size() - 4);  // strip ".bin"

        // header-only read for n_ctx_seq; an unreadable header reports n_ctx_seq = 0
        int32_t n_ctx_seq = 0;
        {
            std::ifstream in(entry.path(), std::ios::binary);
            uint32_t      magic = 0;
            uint32_t      version = 0;
            int32_t       n = 0;
            in.read((char *) &magic, 4);
            in.read((char *) &version, 4);
            in.read((char *) &n_ctx_seq, 4);
            in.read((char *) &n, 4);
            if (!(in && magic == SNAPSHOT_MAGIC && version == SNAPSHOT_VERSION)) {
                n_ctx_seq = 0;
            }
        }

        out.push_back({
            name,
            entry.file_size(ec),
            file_mtime_unix(entry.path()),
            n_ctx_seq,
        });
    }
    return out;
}
