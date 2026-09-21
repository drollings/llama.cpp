#include "server-database.h"

#include <sqlite3.h>

server_database::~server_database() {
    close();
}

bool server_database::open(const std::string & path) {
    close();
    sqlite3 * handle = nullptr;
    if (sqlite3_open(path.c_str(), &handle) != SQLITE_OK) {
        last_err = handle != nullptr ? sqlite3_errmsg(handle) : "out of memory";
        sqlite3_close(handle);
        return false;
    }
    db      = handle;
    db_path = path;
    sqlite3_busy_timeout(db, BUSY_TIMEOUT_MS);
    // readers never block writers and vice versa; required for the pool's
    // concurrent snapshot bookkeeping
    if (!exec("PRAGMA journal_mode=WAL;")) {
        close();
        return false;
    }
    return true;
}

void server_database::close() {
    if (db != nullptr) {
        sqlite3_close(db);
        db = nullptr;
    }
    db_path.clear();
}

bool server_database::exec(const std::string & sql) {
    if (db == nullptr) {
        last_err = "database is not open";
        return false;
    }
    char * msg = nullptr;
    const int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &msg);
    if (rc != SQLITE_OK) {
        last_err = msg != nullptr ? msg : sqlite3_errmsg(db);
        sqlite3_free(msg);
        return false;
    }
    return true;
}

bool server_database::query(const std::string & sql, const row_callback & row) {
    if (db == nullptr) {
        last_err = "database is not open";
        return false;
    }
    auto trampoline = [](void * arg, int ncols, char ** vals, char ** names) -> int {
        const auto & cb = *static_cast<const row_callback *>(arg);
        return cb(ncols, (const char * const *) names, (const char * const *) vals) ? 0 : 1;
    };
    char * msg = nullptr;
    const int rc = sqlite3_exec(db, sql.c_str(), trampoline, (void *) &row, &msg);
    // ABORT means the callback stopped early: a clean stop, not an error
    if (rc != SQLITE_OK && rc != SQLITE_ABORT) {
        last_err = msg != nullptr ? msg : sqlite3_errmsg(db);
        sqlite3_free(msg);
        return false;
    }
    return true;
}
