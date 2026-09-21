#include "server-database.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>

static std::string test_db_path(const std::string & name) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const auto dir = fs::temp_directory_path(ec) / "llama-m9-db";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    return (dir / name).string();
}

// open creates the file; close releases; reopen works.
static void test_db_open_close() {
    const std::string path = test_db_path("open.db");
    server_database db;
    assert(!db.is_open());
    assert(db.open(path));
    assert(db.is_open());
    assert(db.path() == path);
    assert(std::filesystem::exists(path));
    db.close();
    assert(!db.is_open());
    assert(db.open(path));
    assert(db.is_open());
}

// WAL journal mode is on after open.
static void test_db_wal() {
    const std::string path = test_db_path("wal.db");
    server_database db;
    assert(db.open(path));
    std::string mode;
    assert(db.query("PRAGMA journal_mode", [&](int, const char * const *, const char * const * vals) {
        mode = vals[0] != nullptr ? vals[0] : "";
        return false;
    }));
    assert(mode == "wal");
}

// plain CRUD round-trip through exec/query.
static void test_db_crud() {
    const std::string path = test_db_path("crud.db");
    server_database db;
    assert(db.open(path));
    assert(db.exec("CREATE TABLE kv(k TEXT PRIMARY KEY, v TEXT NOT NULL)"));
    assert(db.exec("INSERT INTO kv VALUES('a','1')"));

    std::string got;
    int rows = 0;
    assert(db.query("SELECT v FROM kv WHERE k='a'", [&](int n, const char * const *, const char * const * vals) {
        assert(n == 1);
        got = vals[0] != nullptr ? vals[0] : "";
        rows++;
        return true;
    }));
    assert(rows == 1 && got == "1");

    assert(db.exec("UPDATE kv SET v='2' WHERE k='a'"));
    assert(db.query("SELECT v FROM kv WHERE k='a'", [&](int, const char * const *, const char * const * vals) {
        got = vals[0] != nullptr ? vals[0] : "";
        return false;
    }));
    assert(got == "2");

    assert(db.exec("DELETE FROM kv WHERE k='a'"));
    rows = 0;
    assert(db.query("SELECT v FROM kv", [&](int, const char * const *, const char * const *) {
        rows++;
        return true;
    }));
    assert(rows == 0);
}

// a concurrent writer blocks and retries instead of failing busy: the second
// connection's write waits out the first connection's held lock, then succeeds.
static void test_db_busy_retry() {
    const std::string path = test_db_path("busy.db");
    server_database db1;
    server_database db2;
    assert(db1.open(path));
    assert(db2.open(path));
    assert(db1.exec("CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT NOT NULL)"));
    assert(db1.exec("BEGIN IMMEDIATE"));
    assert(db1.exec("INSERT INTO t VALUES(1,'a')"));

    std::thread releaser([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        assert(db1.exec("COMMIT"));
    });
    const auto start = std::chrono::steady_clock::now();
    assert(db2.exec("INSERT INTO t VALUES(2,'b')"));
    const auto waited = std::chrono::steady_clock::now() - start;
    releaser.join();
    // must have blocked on the lock (not failed fast) and then committed
    assert(waited >= std::chrono::milliseconds(150));

    int rows = 0;
    assert(db2.query("SELECT v FROM t ORDER BY id", [&](int, const char * const *, const char * const *) {
        rows++;
        return true;
    }));
    assert(rows == 2);
}

// constraint violations are errors with text, not crashes.
static void test_db_not_null() {
    const std::string path = test_db_path("notnull.db");
    server_database db;
    assert(db.open(path));
    assert(db.exec("CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT NOT NULL)"));
    assert(!db.exec("INSERT INTO t VALUES(1,NULL)"));
    assert(db.last_error().find("NOT NULL") != std::string::npos);
}

int main() {
    test_db_open_close();
    test_db_wal();
    test_db_crud();
    test_db_busy_retry();
    test_db_not_null();
    printf("test-server-database: all tests passed\n");
    return 0;
}
