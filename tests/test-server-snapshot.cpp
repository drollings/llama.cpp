#include "server-snapshot.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifndef _WIN32
#include <csignal>
#include <sys/resource.h>
#endif

#undef NDEBUG
#include <cassert>

static std::filesystem::path make_tmpdir() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() /
        ("server-snapshot-test-" + std::to_string(
            std::chrono::system_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    return dir;
}

static server_snapshot_data sample() {
    server_snapshot_data data;
    data.n_ctx_seq = 4096;
    data.tokens    = { 1, 42, 43, 44, 0 };
    data.kv        = { 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77 };
    return data;
}

static void test_round_trip() {
    const auto dir  = make_tmpdir();
    const auto path = (dir / "snap.bin").string();
    const auto data = sample();

    assert(server_snapshot_write(path, data));
    const auto got = server_snapshot_read(path);
    assert(got.has_value());
    assert(got->n_ctx_seq == data.n_ctx_seq);
    assert(got->tokens == data.tokens);
    assert(got->kv == data.kv);

    std::filesystem::remove_all(dir);
}

static void test_truncation_nullopt() {
    const auto dir  = make_tmpdir();
    const auto path = (dir / "snap.bin").string();
    const auto data = sample();

    assert(server_snapshot_write(path, data));

    // cut the file short: the read must fail without throwing
    const auto truncated = dir / "trunc.bin";
    {
        std::ifstream in(path, std::ios::binary);
        std::vector<char> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        buf.resize(buf.size() - 4);  // drop the last 4 bytes of kv
        std::ofstream out(truncated, std::ios::binary | std::ios::trunc);
        out.write(buf.data(), (std::streamsize) buf.size());
    }
    assert(!server_snapshot_read(truncated.string()).has_value());

    // a truncated header (only 10 bytes) also returns nullopt
    const auto half = dir / "half.bin";
    {
        std::ofstream out(half, std::ios::binary | std::ios::trunc);
        out.write("SLPAv1", 6);
    }
    assert(!server_snapshot_read(half.string()).has_value());

    std::filesystem::remove_all(dir);
}

static void test_bad_magic_nullopt() {
    const auto dir  = make_tmpdir();
    const auto path = (dir / "snap.bin").string();
    const auto data = sample();

    assert(server_snapshot_write(path, data));

    // corrupt the magic byte
    {
        std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
        out.seekp(0);
        out.put('X');
    }
    assert(!server_snapshot_read(path).has_value());

    std::filesystem::remove_all(dir);
}

static void test_read_status() {
    const auto dir  = make_tmpdir();
    const auto path = (dir / "snap.bin").string();
    const auto data = sample();

    // absent file -> MISSING
    auto missing = server_snapshot_read_status((dir / "nope.bin").string());
    assert(missing.status == server_snapshot_status::MISSING);
    assert(!missing.data.has_value());

    // a good file -> OK with data
    assert(server_snapshot_write(path, data));
    auto ok = server_snapshot_read_status(path);
    assert(ok.status == server_snapshot_status::OK);
    assert(ok.data.has_value());
    assert(ok.data->n_ctx_seq == data.n_ctx_seq);
    assert(ok.data->tokens == data.tokens);
    assert(ok.data->kv == data.kv);

    // a truncated file -> CORRUPT
    const auto truncated = dir / "trunc.bin";
    {
        std::ifstream in(path, std::ios::binary);
        std::vector<char> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        buf.resize(buf.size() - 4);
        std::ofstream out(truncated, std::ios::binary | std::ios::trunc);
        out.write(buf.data(), (std::streamsize) buf.size());
    }
    auto corrupt = server_snapshot_read_status(truncated.string());
    assert(corrupt.status == server_snapshot_status::CORRUPT);
    assert(!corrupt.data.has_value());

    // a bad-magic file -> CORRUPT
    {
        std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
        out.seekp(0);
        out.put('X');
    }
    auto bad = server_snapshot_read_status(path);
    assert(bad.status == server_snapshot_status::CORRUPT);
    assert(!bad.data.has_value());

    std::filesystem::remove_all(dir);
}

static void test_list() {
    const auto dir = make_tmpdir();

    server_snapshot_data a = sample();
    a.n_ctx_seq            = 2048;
    server_snapshot_data b = sample();
    b.n_ctx_seq            = 8192;

    assert(server_snapshot_write((dir / "one.bin").string(), a));
    assert(server_snapshot_write((dir / "two.bin").string(), b));

    // a stray non-snapshot file is skipped
    {
        std::ofstream out(dir / "notasnapshot.txt", std::ios::trunc);
        out << "hello";
    }
    // an unreadable .bin header reports n_ctx_seq = 0 but is still listed
    {
        std::ofstream out(dir / "bad.bin", std::ios::binary | std::ios::trunc);
        out.write("garbagegarbagegarbage", 21);
    }

    const auto list = server_snapshot_list(dir.string());
    assert(list.size() == 3);

    bool found_one = false;
    bool found_two = false;
    bool found_bad = false;
    for (const auto & meta : list) {
        if (meta.name == "one") {
            found_one = true;
            assert(meta.n_ctx_seq == 2048);
        } else if (meta.name == "two") {
            found_two = true;
            assert(meta.n_ctx_seq == 8192);
        } else if (meta.name == "bad") {
            found_bad = true;
            assert(meta.n_ctx_seq == 0);
        }
    }
    assert(found_one && found_two && found_bad);

    std::filesystem::remove_all(dir);
}

static void test_fp_round_trip() {
    const auto dir  = make_tmpdir();
    const auto path = (dir / "snap.bin").string();

    server_snapshot_data data = sample();
    data.adapter_fp           = "deadbeef";
    assert(server_snapshot_write(path, data));
    const auto got = server_snapshot_read(path);
    assert(got.has_value());
    assert(got->adapter_fp == "deadbeef");
    assert(got->tokens == data.tokens);
    assert(got->kv == data.kv);

    const auto list = server_snapshot_list(dir.string());
    assert(list.size() == 1);
    assert(list[0].adapter_fp == "deadbeef");

    std::filesystem::remove_all(dir);
}

// hand-craft a version-1 file (no adapter_fp field): it must read with fp = ""
static void test_v1_backcompat() {
    const auto dir  = make_tmpdir();
    const auto path = (dir / "v1.bin").string();
    const auto data = sample();
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        const uint32_t magic = 0x534C5041;
        const uint32_t version = 1;
        const uint64_t kv_size = (uint64_t) data.kv.size();
        const int32_t  n_tokens = (int32_t) data.tokens.size();
        out.write((const char *) &magic, 4);
        out.write((const char *) &version, 4);
        out.write((const char *) &data.n_ctx_seq, 4);
        out.write((const char *) &n_tokens, 4);
        out.write((const char *) &kv_size, 8);
        for (const llama_token t : data.tokens) {
            out.write((const char *) &t, sizeof(t));
        }
        out.write((const char *) data.kv.data(), (std::streamsize) data.kv.size());
    }
    const auto got = server_snapshot_read(path);
    assert(got.has_value());
    assert(got->adapter_fp.empty());
    assert(got->tokens == data.tokens);
    assert(got->kv == data.kv);

    const auto list = server_snapshot_list(dir.string());
    assert(list.size() == 1);
    assert(list[0].adapter_fp.empty());

    std::filesystem::remove_all(dir);
}

// the read path reports the on-disk version: current writes read back as 2
// (including fp_len 0, which means "saved with zero adapters"), hand-made v1
// files read back as 1. the restore gate needs the distinction.
static void test_version_plumbed() {
    const auto dir  = make_tmpdir();
    const auto data = sample();

    const auto v2_path = (dir / "v2.bin").string();
    assert(server_snapshot_write(v2_path, data));
    const auto v2 = server_snapshot_read(v2_path);
    assert(v2.has_value());
    assert(v2->version == 2);

    server_snapshot_data no_fp = data;
    no_fp.adapter_fp.clear();
    const auto empty_path = (dir / "empty.bin").string();
    assert(server_snapshot_write(empty_path, no_fp));
    const auto empty = server_snapshot_read(empty_path);
    assert(empty.has_value());
    assert(empty->version == 2);
    assert(empty->adapter_fp.empty());

    const auto v1_path = (dir / "v1.bin").string();
    {
        std::ofstream out(v1_path, std::ios::binary | std::ios::trunc);
        const uint32_t magic = 0x534C5041;
        const uint32_t version = 1;
        const uint64_t kv_size = (uint64_t) data.kv.size();
        const int32_t  n_tokens = (int32_t) data.tokens.size();
        out.write((const char *) &magic, 4);
        out.write((const char *) &version, 4);
        out.write((const char *) &data.n_ctx_seq, 4);
        out.write((const char *) &n_tokens, 4);
        out.write((const char *) &kv_size, 8);
        for (const llama_token t : data.tokens) {
            out.write((const char *) &t, sizeof(t));
        }
        out.write((const char *) data.kv.data(), (std::streamsize) data.kv.size());
    }
    const auto v1 = server_snapshot_read(v1_path);
    assert(v1.has_value());
    assert(v1->version == 1);
    assert(v1->adapter_fp.empty());

    std::filesystem::remove_all(dir);
}

// one listing over mixed content: every .bin gets a row, each classified from
// its own header, so one bad file can never hide or end the good rows. bad
// headers report version 0 with zeroed fields.
static void test_list_versions_and_bad_files() {
    const auto dir  = make_tmpdir();
    const auto data = sample();

    server_snapshot_data stamped = data;
    stamped.adapter_fp           = "abc123";
    assert(server_snapshot_write((dir / "good.bin").string(), stamped));

    server_snapshot_data no_fp = data;
    no_fp.adapter_fp.clear();
    assert(server_snapshot_write((dir / "nofp.bin").string(), no_fp));

    {
        std::ofstream out((dir / "old.bin").string(), std::ios::binary | std::ios::trunc);
        const uint32_t magic = 0x534C5041;
        const uint32_t version = 1;
        const uint64_t kv_size = (uint64_t) data.kv.size();
        const int32_t  n_tokens = (int32_t) data.tokens.size();
        out.write((const char *) &magic, 4);
        out.write((const char *) &version, 4);
        out.write((const char *) &data.n_ctx_seq, 4);
        out.write((const char *) &n_tokens, 4);
        out.write((const char *) &kv_size, 8);
        for (const llama_token t : data.tokens) {
            out.write((const char *) &t, sizeof(t));
        }
        out.write((const char *) data.kv.data(), (std::streamsize) data.kv.size());
    }
    {
        std::ofstream out((dir / "junk.bin").string(), std::ios::binary | std::ios::trunc);
        out.write("not a snapshot", 14);
    }

    auto by_name = [&](const std::vector<server_snapshot_meta> & rows, const std::string & name) {
        for (const auto & m : rows) {
            if (m.name == name) {
                return m;
            }
        }
        assert(false);
        return server_snapshot_meta{};
    };

    const auto rows = server_snapshot_list(dir.string());
    assert(rows.size() == 4);
    assert(by_name(rows, "good").version == 2);
    assert(by_name(rows, "good").adapter_fp == "abc123");
    assert(by_name(rows, "good").n_ctx_seq == data.n_ctx_seq);
    assert(by_name(rows, "nofp").version == 2);
    assert(by_name(rows, "nofp").adapter_fp.empty());
    assert(by_name(rows, "old").version == 1);
    assert(by_name(rows, "old").adapter_fp.empty());
    assert(by_name(rows, "junk").version == 0);
    assert(by_name(rows, "junk").adapter_fp.empty());
    assert(by_name(rows, "junk").n_ctx_seq == 0);

    std::filesystem::remove_all(dir);
}

// a negative n_tokens is CORRUPT in the read path and a zeroed (version 0)
// row in the list path: the two classifiers agree.
static void test_n_tokens_negative_parity() {
    const auto dir  = make_tmpdir();
    const auto path = (dir / "neg.bin").string();
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        const uint32_t magic     = 0x534C5041;
        const uint32_t version   = 2;
        const int32_t  n_ctx_seq = 64;
        const int32_t  n_tokens  = -3;
        const uint64_t kv_size   = 0;
        const uint32_t fp_len    = 0;
        out.write((const char *) &magic, 4);
        out.write((const char *) &version, 4);
        out.write((const char *) &n_ctx_seq, 4);
        out.write((const char *) &n_tokens, 4);
        out.write((const char *) &kv_size, 8);
        out.write((const char *) &fp_len, 4);
    }
    auto st = server_snapshot_read_status(path);
    assert(st.status == server_snapshot_status::CORRUPT);

    const auto rows = server_snapshot_list(dir.string());
    assert(rows.size() == 1);
    assert(rows[0].version == 0);
    assert(rows[0].n_ctx_seq == 0);

    std::filesystem::remove_all(dir);
}

// version 3 (or any unknown version) is CORRUPT, even with a consistent size
static void test_version3_corrupt() {
    const auto dir  = make_tmpdir();
    const auto path = (dir / "v3.bin").string();
    const auto data = sample();
    assert(server_snapshot_write(path, data));
    {
        std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
        const uint32_t version = 3;
        io.seekp(4);
        io.write((const char *) &version, 4);
    }
    auto st = server_snapshot_read_status(path);
    assert(st.status == server_snapshot_status::CORRUPT);
    assert(!st.data.has_value());

    std::filesystem::remove_all(dir);
}

// a huge fp_len in the header must be rejected without allocating: a 28-byte
// crafted file claims gigabytes of fingerprint.
static void test_huge_fp_len_corrupt() {
    const auto dir  = make_tmpdir();
    const auto path = (dir / "evil.bin").string();
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        const uint32_t magic   = 0x534C5041;
        const uint32_t version = 2;
        const int32_t  n_ctx_seq = 64;
        const int32_t  n_tokens  = 0;
        const uint64_t kv_size   = 0;
        const uint32_t fp_len    = 0xFFFFFFF0u;
        out.write((const char *) &magic, 4);
        out.write((const char *) &version, 4);
        out.write((const char *) &n_ctx_seq, 4);
        out.write((const char *) &n_tokens, 4);
        out.write((const char *) &kv_size, 8);
        out.write((const char *) &fp_len, 4);
    }
    auto st = server_snapshot_read_status(path);
    assert(st.status == server_snapshot_status::CORRUPT);
    assert(!st.data.has_value());

    // the header-only list path must also refuse without allocating
    const auto list = server_snapshot_list(dir.string());
    assert(list.size() == 1);
    assert(list[0].adapter_fp.empty());

    std::filesystem::remove_all(dir);
}

static void test_no_tmp_leftover() {
    const auto dir  = make_tmpdir();
    const auto path = (dir / "snap.bin").string();
    const auto data = sample();

    assert(server_snapshot_write(path, data));
    // the write is atomic: no .tmp may remain after a successful write
    assert(!std::filesystem::exists(path + ".tmp"));

    // a failed write (unwritable dir) must not leave a .tmp either
    {
        const auto bad = dir / "nonexistent" / "snap.bin";
        assert(!server_snapshot_write(bad.string(), data));
        assert(!std::filesystem::exists(bad.string() + ".tmp"));
    }

    std::filesystem::remove_all(dir);
}

// a snapshot write that fails partway (short disk write) reports failure,
// leaves any pre-existing target untouched, and leaves no .tmp behind. the
// fault below is deterministic: a tiny RLIMIT_FSIZE makes the payload flush
// fail (SIGXFSZ ignored so the write fails instead of killing the process).
static void test_short_write_keeps_target() {
    const auto dir  = make_tmpdir();
    const auto path = (dir / "snap.bin").string();
    const auto data = sample();

    assert(server_snapshot_write(path, data));
    std::ifstream before_stream(path, std::ios::binary);
    const std::vector<char> before((std::istreambuf_iterator<char>(before_stream)),
                                   std::istreambuf_iterator<char>());
    assert(!before.empty());

#ifndef _WIN32
    struct rlimit old_lim;
    assert(getrlimit(RLIMIT_FSIZE, &old_lim) == 0);
    // lower only the soft limit: lowering the hard limit cannot be undone
    // without privilege, which would break the restore below
    struct rlimit small_lim = { 16, old_lim.rlim_max };
    assert(setrlimit(RLIMIT_FSIZE, &small_lim) == 0);
    auto * old_handler = std::signal(SIGXFSZ, SIG_IGN);

    const bool ok = server_snapshot_write(path, data);

    std::signal(SIGXFSZ, old_handler);
    assert(setrlimit(RLIMIT_FSIZE, &old_lim) == 0);

    // the short write must fail, never rename a truncated file over the target
    assert(!ok);
    std::ifstream after_stream(path, std::ios::binary);
    const std::vector<char> after((std::istreambuf_iterator<char>(after_stream)),
                                  std::istreambuf_iterator<char>());
    assert(after == before);
    assert(!std::filesystem::exists(path + ".tmp"));
#else
    // no deterministic short-write fault on this platform; the good write
    // above plus the round-trip test cover the success path
    (void) before;
#endif

    std::filesystem::remove_all(dir);
}

int main() {
    test_round_trip();
    test_truncation_nullopt();
    test_bad_magic_nullopt();
    test_read_status();
    test_list();
    test_fp_round_trip();
    test_v1_backcompat();
    test_version3_corrupt();
    test_huge_fp_len_corrupt();
    test_version_plumbed();
    test_list_versions_and_bad_files();
    test_n_tokens_negative_parity();
    test_no_tmp_leftover();
    test_short_write_keeps_target();
    printf("test-server-snapshot: all tests passed\n");
    return 0;
}
