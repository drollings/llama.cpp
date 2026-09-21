#pragma once

#include <cstdint>
#include <functional>
#include <string>

// server_database: minimal RAII wrapper over one sqlite file handle. every
// statement path sets a busy timeout and lets sqlite retry a locked database
// internally, so a busy database blocks instead of failing. one handle is not
// safe for concurrent use from several threads; use one handle per thread.
struct server_database {
    // row callback: column count, column names, text values (nullptr for SQL
    // NULL); return false to stop the query early (not an error)
    using row_callback = std::function<bool(int, const char * const *, const char * const *)>;

    // how long sqlite retries a locked database before reporting busy
    static constexpr int BUSY_TIMEOUT_MS = 5000;

    server_database() = default;
    ~server_database();

    server_database(const server_database &) = delete;
    server_database & operator=(const server_database &) = delete;

    // open (creating) the file at path; enables WAL; closes any previous handle
    bool open(const std::string & path);
    void close();

    bool        is_open() const { return db != nullptr; }
    std::string path() const { return db_path; }

    // run a statement without rows (DDL/DML/PRAGMA); false on error
    bool exec(const std::string & sql);
    // run a select, invoking row per result row; false on error
    bool query(const std::string & sql, const row_callback & row);

    std::string last_error() const { return last_err; }

private:
    struct sqlite3 * db = nullptr;
    std::string db_path;
    std::string last_err;
};
