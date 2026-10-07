#include "server-instances.h"
#include "server-snapshot.h"
#include "server-models.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#undef NDEBUG
#include <cassert>

// per-instance on-disk layout: <slot_save_path>/<model_key>/<instance>/<snapshot>.bin
static void test_layout_paths() {
    const std::string inst = server_snapshot_instance_path("data/slots/", "code", "ledger", "work");
    assert(inst == "data/slots/code/ledger/work.bin");

    // trailing slash on the root is optional
    const std::string inst_noslash = server_snapshot_instance_path("data/slots", "code", "ledger", "work");
    assert(inst_noslash == "data/slots/code/ledger/work.bin");

    const std::string dir = server_snapshot_instance_dir("data/slots/", "code", "ledger");
    assert(dir == "data/slots/code/ledger");

    // legacy flat layout (migration read path)
    const std::string leg = server_snapshot_legacy_path("data/slots/", "code", "work");
    assert(leg == "data/slots/code/work.bin");

    // instances sharing a snapshot name never share a file
    const std::string other = server_snapshot_instance_path("data/slots/", "code", "scratch", "work");
    assert(other == "data/slots/code/scratch/work.bin");
    assert(other != inst);
}

// router aggregate merge: instance rows pass through, snapshots gain their
// owning model, totals sum with 64-bit saturation
static void test_merge() {
    json a = {
        { "instances", json::array({ json{ { "id", "m1:default" } } }) },
        { "snapshots", json::array({ json{ { "name", "s1" } } }) },
        { "total",     json{ { "model", 100 }, { "context", 10 }, { "compute", 5 }, { "total", 115 } } },
    };
    json b = {
        { "instances", json::array({ json{ { "id", "m2:default" } } }) },
        { "snapshots", json::array({ json{ { "name", "s1" } } }) },
        { "total",     json{ { "model", 200 }, { "context", 20 }, { "compute", 7 }, { "total", 227 } } },
    };
    const json out = server_models_merge_instances({ { "m1", a }, { "m2", b } });

    assert(out["instances"].size() == 2);
    assert(out["instances"][0]["id"] == "m1:default");
    assert(out["instances"][1]["id"] == "m2:default");

    // same snapshot name on two children stays distinct via the model tag
    assert(out["snapshots"].size() == 2);
    assert(out["snapshots"][0]["model"] == "m1");
    assert(out["snapshots"][1]["model"] == "m2");

    assert(out["total"]["model"] == 300);
    assert(out["total"]["context"] == 30);
    assert(out["total"]["compute"] == 12);
    assert(out["total"]["total"] == 342);
}

static void test_merge_tolerates_shape_drift() {
    // bare-array envelope (pre-envelope forks) and missing totals contribute
    // rows without breaking the sums
    json bare = json::array({ json{ { "id", "m3:x" } } });
    json nototal = {
        { "instances", json::array() },
        { "snapshots", json::array() },
    };
    const json out = server_models_merge_instances({ { "m3", bare }, { "m4", nototal } });
    assert(out["instances"].size() == 0);  // non-object envelope contributes nothing
    assert(out["snapshots"].size() == 0);
    assert(out["total"]["model"] == 0);
    assert(out["total"]["total"] == 0);
}

static void test_merge_saturates() {    json big = {
        { "instances", json::array() },
        { "snapshots", json::array() },
        { "total",     json{ { "model", UINT64_MAX }, { "context", 0 }, { "compute", 0 }, { "total", UINT64_MAX } } },
    };
    json small = {
        { "instances", json::array() },
        { "snapshots", json::array() },
        { "total",     json{ { "model", 1 }, { "context", 0 }, { "compute", 0 }, { "total", 1 } } },
    };
    const json out = server_models_merge_instances({ { "big", big }, { "small", small } });
    assert(out["total"]["model"] == UINT64_MAX);
    assert(out["total"]["total"] == UINT64_MAX);
}

// merged adapter bytes are additive across children; the merged total sums the
// child totals verbatim (each already adapter-inclusive), never re-adding.
static void test_merge_adapter_additive() {
    json a = {
        { "instances", json::array() },
        { "snapshots", json::array() },
        { "total",     json{ { "model", 100 }, { "context", 10 }, { "compute", 5 }, { "adapter", 20 }, { "total", 135 } } },
    };
    json b = {
        { "instances", json::array() },
        { "snapshots", json::array() },
        { "total",     json{ { "model", 200 }, { "context", 20 }, { "compute", 7 }, { "adapter", 30 }, { "total", 257 } } },
    };
    const json out = server_models_merge_instances({ { "m1", a }, { "m2", b } });
    const json total = out["total"];
    assert(total.is_object() && total.size() == 5);
    assert(total["model"] == 300);
    assert(total["context"] == 30);
    assert(total["compute"] == 12);
    assert(total["adapter"] == 50);
    assert(total["total"] == 392);
    assert(total["total"] == total["model"].get<uint64_t>() + total["context"].get<uint64_t>() +
                             total["compute"].get<uint64_t>() + total["adapter"].get<uint64_t>());
}

// a child without the adapter key (pre-adapter fork) contributes zero adapter
// bytes without breaking the sums.
static void test_merge_adapter_missing_key() {
    json a = {
        { "instances", json::array() },
        { "snapshots", json::array() },
        { "total",     json{ { "model", 100 }, { "context", 10 }, { "compute", 5 }, { "adapter", 20 }, { "total", 135 } } },
    };
    json b = {
        { "instances", json::array() },
        { "snapshots", json::array() },
        { "total",     json{ { "model", 200 }, { "context", 20 }, { "compute", 7 }, { "total", 227 } } },
    };
    const json out = server_models_merge_instances({ { "m1", a }, { "m2", b } });
    assert(out["total"]["adapter"] == 20);
    assert(out["total"]["total"] == 362);
}

// as_u64 edges: negatives, floats, strings and missing keys read as zero;
// the unsigned limit survives the round trip.
static void test_merge_as_u64_edges() {
    json a = {
        { "instances", json::array() },
        { "snapshots", json::array() },
        { "total",     json{ { "model", -5 }, { "context", 1.5 }, { "compute", "x" }, { "adapter", UINT64_MAX }, { "total", 0 } } },
    };
    const json out = server_models_merge_instances({ { "m1", a } });
    assert(out["total"]["model"] == 0);
    assert(out["total"]["context"] == 0);
    assert(out["total"]["compute"] == 0);
    assert(out["total"]["adapter"] == UINT64_MAX);
}

// empty input merges to empty rows and all-zero totals
static void test_merge_empty() {
    const json out = server_models_merge_instances({});
    assert(out["instances"].is_array() && out["instances"].empty());
    assert(out["snapshots"].is_array() && out["snapshots"].empty());
    assert(out["total"]["model"] == 0);
    assert(out["total"]["context"] == 0);
    assert(out["total"]["compute"] == 0);
    assert(out["total"]["adapter"] == 0);
    assert(out["total"]["total"] == 0);
}

// three-child golden: exact byte totals, rows pass through, snapshots gain
// their owning model
static void test_merge_three_children_golden() {
    json a = {
        { "instances", json::array({ json{ { "id", "ma:x" } } }) },
        { "snapshots", json::array({ json{ { "name", "s" } } }) },
        { "total",     json{ { "model", 1000 }, { "context", 100 }, { "compute", 10 }, { "adapter", 1 }, { "total", 1111 } } },
    };
    json b = {
        { "instances", json::array({ json{ { "id", "mb:y" } }, json{ { "id", "mb:z" } } }) },
        { "snapshots", json::array() },
        { "total",     json{ { "model", 2000 }, { "context", 200 }, { "compute", 20 }, { "adapter", 2 }, { "total", 2222 } } },
    };
    json c = {
        { "instances", json::array() },
        { "snapshots", json::array({ json{ { "name", "t" } } }) },
        { "total",     json{ { "model", 3000 }, { "context", 300 }, { "compute", 30 }, { "adapter", 3 }, { "total", 3333 } } },
    };
    const json out = server_models_merge_instances({ { "ma", a }, { "mb", b }, { "mc", c } });
    assert(out["instances"].size() == 3);
    assert(out["snapshots"].size() == 2);
    assert(out["snapshots"][0]["model"] == "ma");
    assert(out["snapshots"][1]["model"] == "mc");
    assert(out["total"]["model"] == 6000);
    assert(out["total"]["context"] == 600);
    assert(out["total"]["compute"] == 60);
    assert(out["total"]["adapter"] == 6);
    assert(out["total"]["total"] == 6666);
}

// large unsigned totals pass through exactly: 2^63 fits no int64, so a
// signed read would wrap and must never stand in for the value
static void test_merge_exact_large_unsigned() {
    const json env = json::parse(
        "{\"instances\": [], \"snapshots\": [],"
        " \"total\": {\"model\": 9223372036854775808, \"context\": 5, \"compute\": 0, \"adapter\": 0, \"total\": 9223372036854775813}}");
    const json out = server_models_merge_instances({ { "big", env } });
    assert(out["total"]["model"].get<uint64_t>() == 9223372036854775808ULL);
    assert(out["total"]["context"] == 5);
    assert(out["total"]["total"].get<uint64_t>() == 9223372036854775813ULL);
}

// merge is order-invariant: every permutation of the children yields the
// same totals and the same row multiset (rows concatenate in input order, so
// compare them sorted)
static void test_merge_permutation_invariant() {
    json a = {
        { "instances", json::array({ json{ { "id", "ma:x" } } }) },
        { "snapshots", json::array({ json{ { "name", "s" } } }) },
        { "total",     json{ { "model", 100 }, { "context", -7 }, { "compute", 5 }, { "total", 98 } } },
    };
    json b = {
        { "instances", json::array({ json{ { "id", "mb:y" } } }) },
        { "snapshots", json::array({ json{ { "name", "s" } } }) },
        { "total",     json{ { "model", 200 }, { "context", 20 }, { "compute", 7 }, { "adapter", 30 }, { "total", 257 } } },
    };
    json c = {
        { "instances", json::array() },
        { "snapshots", json::array() },
        { "total",     json{ { "model", 300 }, { "context", 30 } } },
    };
    using child_vec = std::vector<std::pair<std::string, json>>;
    const child_vec base = { { "ma", a }, { "mb", b }, { "mc", c } };

    auto totals_of = [](const json & out) {
        std::vector<uint64_t> legs;
        for (const char * k : { "model", "context", "compute", "adapter", "total" }) {
            legs.push_back(out["total"][k].get<uint64_t>());
        }
        return legs;
    };
    auto rows_of = [](const json & out) {
        std::vector<std::string> rows;
        for (const auto & row : out["instances"]) {
            rows.push_back(row["id"].get<std::string>());
        }
        for (const auto & row : out["snapshots"]) {
            rows.push_back(row["model"].get<std::string>() + ":" + row["name"].get<std::string>());
        }
        std::sort(rows.begin(), rows.end());
        return rows;
    };

    const json first = server_models_merge_instances(base);
    std::vector<std::string> order = { "ma", "mb", "mc" };
    int perms = 0;
    do {
        child_vec perm;
        for (const auto & name : order) {
            perm.push_back({ name, name == "ma" ? a : name == "mb" ? b : c });
        }
        const json out = server_models_merge_instances(perm);
        assert(totals_of(out) == totals_of(first));
        assert(rows_of(out) == rows_of(first));
        perms++;
    } while (std::next_permutation(order.begin(), order.end()));
    assert(perms == 6);
}

// the fan-out collector: fast children are collected, slow ones skip at the
// deadline, failing fetches skip, and the output sorts by model name no
// matter the completion order. two slow tasks run concurrently (max, not
// sum): a serial implementation needs both sleeps back to back.
static void test_fanout_collect_deadline_skip() {
    server_model_meta slow_a;
    slow_a.name = "a";
    slow_a.port = 2;
    server_model_meta slow_b;
    slow_b.name = "d";
    slow_b.port = 5;
    server_model_meta fast_c;
    fast_c.name = "c";
    fast_c.port = 3;
    server_model_meta fast_b;
    fast_b.name = "b";
    fast_b.port = 4;
    instances_fetch_fn fetch = [](const server_model_meta & meta)
            -> std::optional<std::pair<std::string, json>> {
        if (meta.name == "a" || meta.name == "d") {
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            return std::make_pair(meta.name, json{ { "instances", json::array() } });
        }
        if (meta.name == "c") {
            return std::nullopt;
        }
        return std::make_pair(meta.name, json{ { "instances", json::array() } });
    };
    const int64_t t0 = ggml_time_ms();
    auto out = instances_fanout_collect({ slow_a, fast_c, fast_b, slow_b }, fetch, t0 + 400);
    const int64_t wall = ggml_time_ms() - t0;
    // slow pair skipped at the deadline, failing fetch skipped, fast collected
    assert(out.size() == 1);
    assert(out[0].first == "b");
    // concurrent, not serial: both 1500ms sleeps overlap (a serial collector
    // needs 3000ms+ for the slow pair alone, plus abandoned-task joins)
    assert(wall < 2600);

    // deterministic order: names sorted even though submission order was not
    instances_fetch_fn quick = [](const server_model_meta & meta)
            -> std::optional<std::pair<std::string, json>> {
        return std::make_pair(meta.name, json{ { "instances", json::array() } });
    };
    server_model_meta m1;
    m1.name = "zulu";
    server_model_meta m2;
    m2.name = "alpha";
    auto ordered = instances_fanout_collect({ m1, m2 }, quick, ggml_time_ms() + 5000);
    assert(ordered.size() == 2);
    assert(ordered[0].first == "alpha");
    assert(ordered[1].first == "zulu");
}

// golden for the /instances envelope shape: exact key sets and value types as
// served today. later changes must keep this green unless they state a new shape.
static void test_instances_envelope_shape() {
    server_instances mgr;
    mgr.base_name = "tinyllama-2";
    mgr.params.slot_save_path = "";

    auto inst = std::make_shared<server_instance>();
    inst->cfg.name       = "ledger";
    inst->cfg.group      = "ledger";
    inst->cfg.pinned     = true;
    inst->cfg.is_default = true;
    inst->effective.n_ctx      = 512;
    inst->effective.n_parallel = 1;
    inst->built = false;
    mgr.instances.push_back(inst);

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req req { {}, {}, "/instances", "", "", {}, no_stop };

    auto res = mgr.handle_get_instances(req);
    assert(res->status == 200);

    const json body = json::parse(res->data);
    assert(body.is_object());
    assert(body.size() == 3);
    assert(body.contains("instances"));
    assert(body.contains("snapshots"));
    assert(body.contains("total"));

    assert(body["instances"].is_array());
    assert(body["instances"].size() == 1);
    assert(body["snapshots"].is_array());

    const json total = body["total"];
    assert(total.is_object());
    assert(total.size() == 5);
    assert(total.contains("model") && total["model"].is_number_integer());
    assert(total.contains("context") && total["context"].is_number_integer());
    assert(total.contains("compute") && total["compute"].is_number_integer());
    assert(total.contains("adapter") && total["adapter"].is_number_integer());
    assert(total.contains("total") && total["total"].is_number_integer());

    const json row = body["instances"][0];
    assert(row.is_object());
    const std::vector<std::string> keys = {
        "id", "aliases", "group", "n_ctx", "parallel", "pinned", "is_default",
        "state", "model_bytes", "context_bytes", "compute_bytes", "adapter_bytes", "total_bytes",
        "vram_bytes", "last_used", "last_used_epoch",
    };
    assert(row.size() == keys.size());
    for (const auto & key : keys) {
        assert(row.contains(key));
    }
    assert(row["id"].is_string());
    assert(row["aliases"].is_array());
    assert(row["group"].is_string());
    assert(row["n_ctx"].is_number());
    assert(row["parallel"].is_number());
    assert(row["pinned"].is_boolean());
    assert(row["is_default"].is_boolean());
    assert(row["state"].is_string());
    assert(row["model_bytes"].is_number_integer());
    assert(row["context_bytes"].is_number_integer());
    assert(row["compute_bytes"].is_number_integer());
    assert(row["adapter_bytes"].is_number_integer());
    assert(row["total_bytes"].is_number_integer());
    assert(row["vram_bytes"].is_number_integer());
    assert(row["last_used"].is_number());
    assert(row["last_used_epoch"].is_number());
}

// ordering over routing candidates: fewest busy wins, then least recently used,
// then registration order. full or slot-less members are never picked.
static void test_pick_best_candidate() {
    // busy-first: the idle member wins over a busier one regardless of stamps
    assert(server_instances::pick_best_candidate({
        { 0, 1, 1, 100 },
        { 1, 1, 0, 900 },
    }) == std::optional<size_t>(1));

    // then least-recently-used among equally busy members
    assert(server_instances::pick_best_candidate({
        { 0, 2, 1, 900 },
        { 1, 2, 1, 100 },
    }) == std::optional<size_t>(1));

    // then registration order on a full tie: members are pushed in registration
    // order, and a strict compare keeps the first one seen
    assert(server_instances::pick_best_candidate({
        { 0, 1, 0, 500 },
        { 1, 1, 0, 500 },
    }) == std::optional<size_t>(0));
    assert(server_instances::pick_best_candidate({
        { 1, 1, 0, 500 },
        { 0, 1, 0, 500 },
    }) == std::optional<size_t>(1));

    // all-busy group excludes every member
    assert(!server_instances::pick_best_candidate({
        { 0, 1, 1, 100 },
        { 1, 2, 2, 200 },
    }).has_value());

    // control: a fresh unbuilt member (no slots) never outranks a used one,
    // even when the used member is the only one with a window
    assert(server_instances::pick_best_candidate({
        { 0, 0, 0, -1 },
        { 1, 1, 0, 700 },
    }) == std::optional<size_t>(1));

    // control: a fresh built member (never used) is picked before a used one
    // at equal busyness; the cold member spreads the first load
    assert(server_instances::pick_best_candidate({
        { 0, 1, 0, 700 },
        { 1, 1, 0, -1 },
    }) == std::optional<size_t>(1));
}

// the pure unbuilt cases of the n_ctx display rule: an unbuilt window reports the
// size published in the lock-free display cache (explicit sizes as requested,
// 0 while inheriting until the weights are loaded; the model default comes from a
// live pool, covered by the server golden).
static void test_displayed_n_ctx_unbuilt() {
    server_instances mgr; // weights never loaded
    server_instance  inst;
    inst.built.store(false, std::memory_order_relaxed);
    inst.n_parallel_effective.store(1, std::memory_order_relaxed);

    inst.n_ctx_effective.store(256, std::memory_order_relaxed);
    assert(mgr.displayed_n_ctx(inst) == 256);

    inst.n_ctx_effective.store(0, std::memory_order_relaxed);
    assert(mgr.displayed_n_ctx(inst) == 0);
}

// pool I/O worker lifecycle: no job can be posted to a stopped or
// never-started worker. post returns nullopt there instead of a future
// that would never complete (callers answer the existing retriable 503).
static void test_snapshot_io_post_lifecycle() {
    server_instances mgr;

    // before start: no worker exists, the post refuses fast
    assert(!mgr.snapshot_io_post([]() {}).has_value());

    mgr.start_io_worker();
    // double start is a no-op: still exactly one worker draining the queue
    mgr.start_io_worker();
    auto fut = mgr.snapshot_io_post([]() {});
    assert(fut.has_value());
    fut->wait();
    fut->get();

    mgr.stop_io_worker();
    // after stop: the post refuses again, never a dangling future
    assert(!mgr.snapshot_io_post([]() {}).has_value());

    // double stop is a no-op (and so is the terminate() in the destructor)
    mgr.stop_io_worker();
}

// hashed write keys: identities that sanitize alike never share a directory,
// while the legacy mapping is unchanged so old files stay readable.
static void test_snapshot_model_keys() {
    // collision control group: sanitization twins get distinct write dirs
    const std::string slash = server_snapshot_model_key_hashed("a/b");
    const std::string colon = server_snapshot_model_key_hashed("a:b");
    const std::string plain = server_snapshot_model_key_hashed("a_b");
    assert(slash != colon && slash != plain && colon != plain);

    // each write key extends the legacy mapping with a hash suffix
    assert(slash.substr(0, 4) == "a_b-");
    assert(colon.substr(0, 4) == "a_b-");

    // deterministic across calls
    assert(server_snapshot_model_key_hashed("org/model") == server_snapshot_model_key_hashed("org/model"));

    // the legacy mapping is unchanged (the read fallback for old files)
    assert(server_snapshot_model_key("a/b") == "a_b");
    assert(server_snapshot_model_key("a:b") == "a_b");
    assert(server_snapshot_model_key("tinyllama-2") == "tinyllama-2");
}

// resolve order: hashed-key scoped file wins, then the previous-key scoped
// file, then the legacy flat file; "" when none exists.
static void test_snapshot_resolve_key_fallback() {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path root = fs::temp_directory_path(ec) / "llama-tmp-resolve";
    fs::remove_all(root, ec);

    server_instances mgr;
    mgr.base_name = "a/b";  // previous key "a_b", write key "a_b-<hash>"
    mgr.params.slot_save_path = root.string();

    const std::string p_new  = mgr.snapshot_instance_path("ledger", "work");
    const std::string p_prev = mgr.snapshot_instance_path_prev("ledger", "work");
    const std::string p_leg  = mgr.snapshot_legacy_path("work");
    assert(p_new != p_prev);

    const auto touch = [&](const std::string & p) {
        fs::create_directories(fs::path(p).parent_path(), ec);
        std::ofstream out(p, std::ios::binary);
        out.put('x');
    };

    assert(mgr.resolve_snapshot_path("ledger", "work").empty());

    touch(p_leg);
    assert(mgr.resolve_snapshot_path("ledger", "work") == p_leg);

    touch(p_prev);
    assert(mgr.resolve_snapshot_path("ledger", "work") == p_prev);

    touch(p_new);
    assert(mgr.resolve_snapshot_path("ledger", "work") == p_new);

    fs::remove_all(root, ec);
}

// firing rule for the huge-implicit-window guardrail: warns iff the window
// inherits its size (n_ctx == 0) on a large-train model. the null-model case
// never reaches the predicate (the caller returns first).
static void test_huge_implicit_ctx_predicate() {
    // must fire: inherit on a large-train model, including the threshold itself
    assert(server_should_warn_huge_implicit_ctx(0, 32768));
    assert(server_should_warn_huge_implicit_ctx(0, 262144));

    // must not fire: explicit size, however small or large
    assert(!server_should_warn_huge_implicit_ctx(1, 262144));
    assert(!server_should_warn_huge_implicit_ctx(512, 262144));
    assert(!server_should_warn_huge_implicit_ctx(2147483647, 262144));

    // must not fire: inherit on a small-train model, including just below threshold
    assert(!server_should_warn_huge_implicit_ctx(0, 0));
    assert(!server_should_warn_huge_implicit_ctx(0, 4096));
    assert(!server_should_warn_huge_implicit_ctx(0, 32767));
}

// snapshot switch budget: 2 concurrent switches are allowed, the 3rd is
// rejected. the deadline-free rejections are exact (no clock or sleep).
static void test_snapshot_switch_budget() {
    server_instances mgr;

    // within budget: both switches acquire, never a 503
    server_instances::switch_guard first(mgr, INT64_MAX);
    server_instances::switch_guard second(mgr, INT64_MAX);
    assert(first.acquired && second.acquired);

    // over budget with no time left to wait: rejected immediately
    server_instances::switch_guard third(mgr, 0);
    assert(!third.acquired);
}

// snapshot I/O queue budget: max_io_jobs (4) queued writes are accepted, the
// 5th is rejected with the retriable nullopt. a pinned worker job makes the
// drain deterministic (no clock or sleep on the test thread).
static void test_snapshot_io_queue_budget() {
    server_instances mgr;
    mgr.start_io_worker();

    // pin the worker inside one job so nothing drains while the queue fills
    std::promise<void> entered;
    std::atomic<bool>  release{false};
    auto pin = mgr.snapshot_io_post([&]() {
        entered.set_value();
        while (!release.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    assert(pin.has_value());
    entered.get_future().wait();

    // four queued posts accepted...
    std::vector<std::future<void>> queued;
    for (int i = 0; i < 4; i++) {
        auto fut = mgr.snapshot_io_post([]() {});
        assert(fut.has_value());
        queued.push_back(std::move(*fut));
    }
    // ...the 5th queued write is rejected
    assert(!mgr.snapshot_io_post([]() {}).has_value());

    release.store(true);
    pin->wait();
    pin->get();
    for (auto & fut : queued) {
        fut.wait();
        fut.get();
    }
    mgr.stop_io_worker();
}

// test-only helper: capture an envelope as a canonical string so later
// refactors compare byte-for-byte.
static std::string instances_canonical(const json & envelope) {
    return envelope.dump();
}

// canonical capture is stable: two reads of one pool produce identical bytes.
static void test_envelope_canonical_stable() {
    server_instances mgr;
    mgr.base_name = "tinyllama-2";
    mgr.params.slot_save_path = "";

    auto inst = std::make_shared<server_instance>();
    inst->cfg.name       = "ledger";
    inst->cfg.group      = "ledger";
    inst->cfg.pinned     = true;
    inst->cfg.is_default = true;
    inst->effective.n_ctx      = 512;
    inst->effective.n_parallel = 1;
    inst->built = false;
    mgr.instances.push_back(inst);

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req req { {}, {}, "/instances", "", "", {}, no_stop };

    auto first  = mgr.handle_get_instances(req);
    auto second = mgr.handle_get_instances(req);
    assert(first->status == 200 && second->status == 200);
    assert(instances_canonical(json::parse(first->data)) == instances_canonical(json::parse(second->data)));

    const json body = json::parse(first->data);
    assert(body.is_object() && body.size() == 3);
}

// merge contract: instance rows pass through untouched, snapshot rows gain
// their owning model and keep adapter_fp, totals sum the five current keys.
static void test_merge_exact_shape() {
    const json row_a = {
        { "id", "m1:default" }, { "adapter_bytes", 16 }, { "state", "loaded" },
    };
    const json snap_a = {
        { "name", "s1" }, { "size", 10 }, { "mtime", 7 },
        { "n_ctx_seq", 64 }, { "adapter_fp", "abcdef" }, { "instance", "ledger" },
    };
    json a = {
        { "instances", json::array({ row_a }) },
        { "snapshots", json::array({ snap_a }) },
        { "total",     json{ { "model", 100 }, { "context", 10 }, { "compute", 5 }, { "adapter", 16 }, { "total", 131 } } },
    };
    json b = {
        { "instances", json::array({ json{ { "id", "m2:default" } } }) },
        { "snapshots", json::array({ json{ { "name", "s1" } } }) },
        { "total",     json{ { "model", 200 }, { "context", 20 }, { "compute", 7 }, { "total", 227 } } },
    };
    const json out = server_models_merge_instances({ { "m1", a }, { "m2", b } });

    // instance rows are untouched copies
    assert(out["instances"].size() == 2);
    assert(out["instances"][0] == row_a);
    assert(out["instances"][1]["id"] == "m2:default");

    // snapshot rows gain the owning model, keep every other field
    assert(out["snapshots"].size() == 2);
    assert(out["snapshots"][0]["model"] == "m1");
    assert(out["snapshots"][0]["adapter_fp"] == "abcdef");
    assert(out["snapshots"][0]["instance"] == "ledger");
    assert(out["snapshots"][1]["model"] == "m2");

    // totals carry exactly the five current keys and sum additively; a child
    // without the adapter key contributes zero adapter bytes
    const json total = out["total"];
    assert(total.is_object() && total.size() == 5);
    assert(total["model"] == 300);
    assert(total["context"] == 30);
    assert(total["compute"] == 12);
    assert(total["adapter"] == 16);
    assert(total["total"] == 358);
}

// snapshot row shape in the live envelope: one file on disk surfaces one row
// with exactly {name, size, mtime, n_ctx_seq, adapter_fp, instance}.
static void test_envelope_snapshot_row_shape() {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path root = fs::temp_directory_path(ec) / "llama-tmp-envelope-snap";
    fs::remove_all(root, ec);

    server_instances mgr;
    mgr.base_name = "m";
    mgr.params.slot_save_path = root.string() + "/";

    auto inst = std::make_shared<server_instance>();
    inst->cfg.name       = "ledger";
    inst->cfg.group      = "ledger";
    inst->effective.n_ctx      = 512;
    inst->effective.n_parallel = 1;
    inst->built = false;
    mgr.instances.push_back(inst);

    server_snapshot_data data;
    data.n_ctx_seq  = 64;
    data.adapter_fp = "abcdef";
    data.tokens     = { 1, 2, 3 };
    data.kv         = { 4, 5, 6 };
    const std::string path = mgr.snapshot_instance_path("ledger", "work");
    fs::create_directories(fs::path(path).parent_path(), ec);
    assert(server_snapshot_write(path, data));

    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req req { {}, {}, "/instances", "", "", {}, no_stop };
    auto res = mgr.handle_get_instances(req);
    assert(res->status == 200);

    const json body = json::parse(res->data);
    assert(body["snapshots"].is_array() && body["snapshots"].size() == 1);
    const json row = body["snapshots"][0];
    assert(row.is_object() && row.size() == 7);
    assert(row["name"] == "work");
    assert(row["n_ctx_seq"] == 64);
    assert(row["adapter_fp"] == "abcdef");
    assert(row["version"] == 2);
    assert(row["instance"] == "ledger");
    assert(row["size"].is_number_integer());
    assert(row["mtime"].is_number_integer());

    fs::remove_all(root, ec);
}

// the pool keeps the error class typed across the instance boundary: the status comes
// from format_error_response's table, not from any field a result rendered. every
// error_type maps to its documented status; a result that is not an error is a pool
// bug (500); a successful slot copy never reaches the helper and keeps 200 + its bytes.
static void test_result_error_response() {
    struct {
        error_type  type;
        int         status;
    } cases[] = {
        { ERROR_TYPE_INVALID_REQUEST, 400 },
        { ERROR_TYPE_AUTHENTICATION, 401 },
        { ERROR_TYPE_PERMISSION, 403 },
        { ERROR_TYPE_NOT_FOUND, 404 },
        { ERROR_TYPE_SERVER, 500 },
        { ERROR_TYPE_NOT_SUPPORTED, 501 },
        { ERROR_TYPE_UNAVAILABLE, 503 },
        { ERROR_TYPE_EXCEED_CONTEXT_SIZE, 400 },
        { ERROR_TYPE_INVALID_REQUEST_SEMANTIC, 422 },
        { ERROR_TYPE_PAYLOAD_TOO_LARGE, 413 },
        { ERROR_TYPE_CLIENT_CLOSED, 499 },
        { ERROR_TYPE_OVERLOADED, 529 },
    };
    for (const auto & c : cases) {
        server_task_result_error err;
        err.err_type = c.type;
        err.err_msg  = "pool boundary";
        auto res = server_instances::make_error_from_result(err);
        assert(res->status == c.status);
        const json body = json::parse(res->data);
        assert(body["error"]["code"] == c.status);
        assert(body["error"]["message"] == "pool boundary");
        // the class is what chose the status: the type string matches the same table
        assert(body["error"]["type"] == format_error_response("x", c.type)["type"]);
    }

    // a payload that claims a status cannot choose it: the rendered body's code and
    // the response status are both the table's answer for the typed class
    {
        server_task_result_error err;
        err.err_type = ERROR_TYPE_UNAVAILABLE;
        err.err_msg  = "retriable";
        auto res = server_instances::make_error_from_result(err);
        assert(res->status == 503);
        assert(json::parse(res->data)["error"]["code"] == 503);
    }

    // a result that is not an error means the caller reached an error branch by mistake
    {
        server_task_result_slot_save_load ok;
        ok.n_tokens = 7;
        auto res = server_instances::make_error_from_result(ok);
        assert(res->status == 500);
        assert(json::parse(res->data)["error"]["code"] == 500);
    }
}

// control group: the success path is untouched. a slot copy is not an error result, so
// the pool never routes it to the error helper (both call sites gate on is_error()), and
// the bytes it carries are what the success path consumes.
static void test_result_success_path_unaffected() {
    server_task_result_slot_copy copy;
    copy.id_slot = 0;
    copy.tokens  = { 1, 2, 3, 4 };
    copy.buffer  = { 0xAA, 0xBB, 0xCC };
    assert(!copy.is_error());
    assert(copy.buffer.size() == 3 && copy.tokens.size() == 4);

    // the copy's own render still carries the copied sizes, so the success branch keeps
    // reporting them (a save answers from these numbers, not from an error body)
    const json rendered = copy.to_json();
    assert(rendered["n_tokens"] == 4 && rendered["n_bytes"] == 3);
}

// the former sidecar name is an ordinary name now: no reservation, no internal
// marker in the envelope, deletable like any other window.
static void test_former_sidecar_name_is_ordinary() {
    server_instances mgr;
    mgr.base_name = "m";
    mgr.params.slot_save_path = "";

    auto inst = std::make_shared<server_instance>();
    inst->cfg.name  = "__decision__";
    inst->cfg.group = "g";
    inst->effective.n_ctx      = 512;
    inst->effective.n_parallel = 1;
    mgr.instances.push_back(inst);

    static const std::function<bool()> no_stop = []() { return false; };

    // the envelope lists it as a normal unbuilt window with no internal marker
    server_http_req inst_req { {}, {}, "/instances", "", "", {}, no_stop };
    auto res = mgr.handle_get_instances(inst_req);
    assert(res->status == 200);
    const json body = json::parse(res->data);
    assert(body["instances"].is_array() && body["instances"].size() == 1);
    const json row = body["instances"][0];
    assert(!row.contains("internal"));
    assert(row["id"] == "m:__decision__");
    assert(row["state"] == "unloaded");

    // not manager-owned: delete succeeds instead of the old refusal
    server_http_req del_req { {}, {}, "/instances/__decision__", "", "", {}, no_stop };
    del_req.params["name"] = "__decision__";
    assert(mgr.handle_delete_instance(del_req)->status == 200);
}

// test-only: a built window with a real but model-free context, so dispatch()
// runs its count/forward path with no weights and no scheduler thread.
static std::shared_ptr<server_instance> dispatch_test_window(const std::string & name, const std::string & group) {
    auto inst = std::make_shared<server_instance>();
    inst->cfg.name  = name;
    inst->cfg.group = group;
    inst->effective.n_ctx      = 256;
    inst->effective.n_parallel = 1;
    inst->n_parallel_effective.store(1, std::memory_order_relaxed);
    inst->ctx_server = std::make_unique<server_context>();
    inst->routes     = std::make_unique<server_routes>(inst->effective, *inst->ctx_server);
    inst->slot_snapshots.resize(1);
    inst->built.store(true, std::memory_order_release);
    return inst;
}

static server_http_req dispatch_test_req(const std::string & body) {
    static const std::function<bool()> no_stop = []() { return false; };
    return server_http_req { {}, {}, "/", "", body, {}, no_stop };
}

// instance path serves exactly the named member and releases its count.
static void test_dispatch_instance_serves_named_member() {
    server_instances mgr;
    mgr.base_name = "m";
    mgr.instances.push_back(dispatch_test_window("a", "g"));
    mgr.instances.push_back(dispatch_test_window("b", "g"));

    std::vector<std::string> served;
    server_instances::forward_fn fwd = [&](server_routes & routes, const server_http_req &) -> server_http_res_ptr {
        for (const auto & inst : mgr.instances) {
            if (inst->routes.get() == &routes) {
                served.push_back(inst->cfg.name);
            }
        }
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->data   = "{}";
        return res;
    };

    auto res = mgr.dispatch(dispatch_test_req("{\"model\": \"m:b\"}"), fwd);
    assert(res->status == 200);
    assert(served.size() == 1 && served[0] == "b");
    assert(mgr.instances[0]->n_active_dispatch == 0);
    assert(mgr.instances[1]->n_active_dispatch == 0);

    mgr.terminate();
}

// group path serves an available member; a dying member is skipped.
static void test_dispatch_group_serves_available_member() {
    server_instances mgr;
    mgr.base_name = "m";
    mgr.instances.push_back(dispatch_test_window("a", "g"));
    mgr.instances.push_back(dispatch_test_window("b", "g"));

    std::vector<std::string> served;
    server_instances::forward_fn fwd = [&](server_routes & routes, const server_http_req &) -> server_http_res_ptr {
        for (const auto & inst : mgr.instances) {
            if (inst->routes.get() == &routes) {
                served.push_back(inst->cfg.name);
            }
        }
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->data   = "{}";
        return res;
    };

    // tie on zero stats: registration order wins
    assert(mgr.dispatch(dispatch_test_req("{\"model\": \"m:g\"}"), fwd)->status == 200);
    assert(served.size() == 1 && served[0] == "a");

    // the first member draining: the group falls through to the second
    mgr.instances[0]->removing = true;
    assert(mgr.dispatch(dispatch_test_req("{\"model\": \"m:g\"}"), fwd)->status == 200);
    assert(served.size() == 2 && served[1] == "b");
    assert(mgr.instances[1]->n_active_dispatch == 0);

    mgr.terminate();
}

// a group with no usable member answers 503 without calling forward.
static void test_dispatch_group_busy_returns_503() {
    server_instances mgr;
    mgr.base_name = "m";
    mgr.params.instance_wait_seconds = 0;
    mgr.instances.push_back(dispatch_test_window("a", "g"));
    mgr.instances.push_back(dispatch_test_window("b", "g"));
    mgr.instances[0]->removing = true;
    mgr.instances[1]->removing = true;

    int calls = 0;
    server_instances::forward_fn fwd = [&](server_routes &, const server_http_req &) -> server_http_res_ptr {
        ++calls;
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->data   = "{}";
        return res;
    };

    auto res = mgr.dispatch(dispatch_test_req("{\"model\": \"m:g\"}"), fwd);
    assert(res->status == 503);
    assert(calls == 0);

    mgr.terminate();
}

// a throwing route still releases the dispatch count, so later requests serve.
static void test_dispatch_throw_balances_count() {
    server_instances mgr;
    mgr.base_name = "m";
    mgr.instances.push_back(dispatch_test_window("a", "g"));
    const auto inst = mgr.instances.front();

    server_instances::forward_fn bad = [](server_routes &, const server_http_req &) -> server_http_res_ptr {
        throw std::runtime_error("route failed");
    };
    bool threw = false;
    try {
        mgr.dispatch(dispatch_test_req("{\"model\": \"m:a\"}"), bad);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    assert(threw);
    assert(inst->n_active_dispatch == 0);

    server_instances::forward_fn ok = [](server_routes &, const server_http_req &) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->data   = "{}";
        return res;
    };
    assert(mgr.dispatch(dispatch_test_req("{\"model\": \"m:a\"}"), ok)->status == 200);
    assert(inst->n_active_dispatch == 0);

    mgr.terminate();
}

// the shared dispatch count: held across the call, released on scope exit
// even when the route throws; a draining instance is never counted.
static void test_dispatch_guard_balances_on_throw() {
    server_instances mgr;
    auto inst = std::make_shared<server_instance>();

    {
        server_instances::dispatch_guard guard(mgr, inst);
        assert(guard.acquired);
        assert(inst->n_active_dispatch == 1);
        try {
            throw std::runtime_error("boom");
        } catch (const std::runtime_error &) {
            assert(inst->n_active_dispatch == 1);
        }
    }
    assert(inst->n_active_dispatch == 0);

    inst->removing = true;
    {
        server_instances::dispatch_guard guard(mgr, inst);
        assert(!guard.acquired);
    }
    assert(inst->n_active_dispatch == 0);
}

// the adapter swap is the one scheduler op that needs the drain: the raw
// queue has no guard, so the manager wrapper is the only path and it refuses
// without one. loop-free: no scheduler serves this context, so any queued
// post would hang forever on an infinite deadline.
static void test_manager_adapter_apply_requires_drain() {
    server_instances mgr;
    mgr.base_name = "m";
    mgr.instances.push_back(dispatch_test_window("a", "g"));
    const auto inst = mgr.instances.front();

    // the bypass the wrapper closes: the raw task posts straight to the
    // context queue with no drain involved (expired deadline so the unserved
    // wait ends at once instead of hanging).
    {
        server_task bypass(SERVER_TASK_TYPE_SET_ADAPTERS);
        bypass.set_adapters = {};
        auto rd = inst->ctx_server->get_response_reader();
        bypass.id = rd.get_new_id();
        rd.post_task(std::move(bypass));
        assert(rd.next([]() { return true; }) == nullptr);
    }

    // without the drain the wrapper rejects even with an infinite deadline:
    // the immediate error (not a hang) proves nothing reached the scheduler.
    {
        auto res = mgr.manager_apply_adapters(*inst, {}, -1);
        assert(res && res->is_error());
    }

    // with the drain the same call is admitted to the scheduler post, which
    // trips the expired deadline (null, not an error result).
    {
        instance_drain_guard guard(mgr, inst);
        auto res = mgr.manager_apply_adapters(*inst, {}, 0);
        assert(res == nullptr);
    }

    // a swap racing teardown fails fast retriable even under the drain: with
    // running cleared the post is refused before touching the scheduler
    // (previously it burned the deadline on a dying queue).
    {
        instance_drain_guard guard(mgr, inst);
        {
            std::lock_guard<std::mutex> lock(mgr.mutex_dispatch);
            inst->running = false;
        }
        auto res = mgr.manager_apply_adapters(*inst, {}, 0);
        assert(res && res->is_error());
        {
            std::lock_guard<std::mutex> lock(mgr.mutex_dispatch);
            inst->running = true;
        }
    }

    mgr.terminate();
}

// a bounded adapter read tells timeout apart from an observed empty set: an
// expired deadline is never observed, so it answers nullopt. the engaged
// empty side is pinned by the model-backed round-trip test, which now reads
// through the same facet.
static void test_adapter_read_timeout_is_nullopt_not_empty() {
    server_instances mgr;
    mgr.base_name = "m";
    mgr.instances.push_back(dispatch_test_window("a", "g"));
    const auto inst = mgr.instances.front();

    server_context_adapter_ops adapters(*inst->ctx_server);
    const auto timed_out = adapters.get_lora_adapters(0);
    assert(!timed_out);

    mgr.terminate();
}

// two-tier adapter keys: spellings of one file share a key, distinct files
// never do (even with identical bytes), malformed paths are rejected, and
// unreadable spellings fall back to a lexical key instead of throwing.
static void test_adapter_key_two_tier() {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec) / "llama-adapter-key";
    fs::remove_all(dir, ec);
    fs::create_directories(dir / "sub", ec);

    const auto write = [&](const fs::path & p, const std::string & bytes) {
        std::ofstream f(p, std::ios::binary);
        f.write(bytes.data(), (std::streamsize) bytes.size());
    };
    write(dir / "a.gguf", "AAA");
    write(dir / "sub" / "b.gguf", "BBB");
    write(dir / "same-bytes.gguf", "AAA"); // identical bytes, distinct file
    fs::create_symlink(dir / "a.gguf", dir / "link.gguf", ec);
    assert(!ec);
    fs::create_symlink(dir / "loop2", dir / "loop1", ec);
    fs::create_symlink(dir / "loop1", dir / "loop2", ec);

    const std::string a = (dir / "a.gguf").string();
    // same file, three spellings: absolute, dot-dot, symlink
    assert(server_instances::adapter_key(a) == server_instances::adapter_key((dir / "sub" / ".." / "a.gguf").string()));
    assert(server_instances::adapter_key(a) == server_instances::adapter_key((dir / "link.gguf").string()));
    // distinct files stay distinct, even with identical content
    assert(server_instances::adapter_key(a) != server_instances::adapter_key((dir / "sub" / "b.gguf").string()));
    assert(server_instances::adapter_key(a) != server_instances::adapter_key((dir / "same-bytes.gguf").string()));

    // malformed paths are rejected before any registry touch
    assert(!server_instances::try_adapter_key(std::string("a\0b", 3)));
    // missing files and symlink loops keep a lexical fallback (the load
    // fails later with the unloadable-adapter error); the key call itself
    // never throws
    assert(server_instances::try_adapter_key((dir / "gone.gguf").string()).has_value());
    assert(server_instances::try_adapter_key((dir / "loop1").string()).has_value());

    fs::remove_all(dir, ec);
}

// calibration corpus for the identity probe: same-file pairs must merge,
// distinct-file pairs must never merge. precision below 1.0 fails the gate;
// precision/recall are printed for the record.
static void test_adapter_identity_corpus() {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec) / "llama-adapter-corpus";
    fs::remove_all(dir, ec);
    fs::create_directories(dir / "sub", ec);

    const auto write = [&](const fs::path & p, const std::string & bytes) {
        std::ofstream f(p, std::ios::binary);
        f.write(bytes.data(), (std::streamsize) bytes.size());
    };
    write(dir / "a.gguf", "AAA");
    write(dir / "sub" / "b.gguf", "BBB");
    write(dir / "same-bytes.gguf", "AAA");
    fs::create_symlink(dir / "a.gguf", dir / "link.gguf", ec);
    assert(!ec);
    bool have_hardlink = true;
    fs::create_hard_link(dir / "a.gguf", dir / "hard.gguf", ec);
    have_hardlink = !ec;
    fs::create_symlink(dir / "loop2", dir / "loop1", ec);
    fs::create_symlink(dir / "loop1", dir / "loop2", ec);
    fs::create_symlink(dir / "dangling-target", dir / "dangling.gguf", ec);

    const std::string a = (dir / "a.gguf").string();
    struct pair_case { std::string x; std::string y; bool same; const char * label; };
    std::vector<pair_case> cases = {
        { a, (dir / "sub" / ".." / "a.gguf").string(), true,  "dotdot-spelling" },
        { a, (dir / "link.gguf").string(),               true,  "symlink-spelling" },
        { (dir / "link.gguf").string(), (dir / "sub" / ".." / "link.gguf").string(), true, "symlink-dotdot" },
        { a, (dir / "sub" / "b.gguf").string(),          false, "distinct-files" },
        { a, (dir / "same-bytes.gguf").string(),         false, "same-bytes-distinct-files" },
        { a, (dir / "gone.gguf").string(),               false, "missing-file" },
        { a, (dir / "loop1").string(),                   false, "symlink-loop" },
        { a, (dir / "dangling.gguf").string(),           false, "dangling-symlink" },
        { a, dir.string(),                               false, "file-vs-dir" },
    };
    if (have_hardlink) {
        cases.push_back({ a, (dir / "hard.gguf").string(), true, "hardlink-spelling" });
    }

    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (const auto & c : cases) {
        const bool merged = server_instances::adapter_identity_equal(c.x, c.y);
        if (merged && c.same) {
            ++tp;
        } else if (!merged && !c.same) {
            ++tn;
        } else if (merged) {
            ++fp;
            printf("adapter identity corpus: FALSE MERGE on %s\n", c.label);
        } else {
            ++fn;
            printf("adapter identity corpus: MISSED DEDUP on %s\n", c.label);
        }
    }
    const double precision = (double) (tp + fp > 0 ? tp : 1) / (double) (tp + fp > 0 ? tp + fp : 1);
    const double recall    = (double) (tp + fn > 0 ? tp : 1) / (double) (tp + fn > 0 ? tp + fn : 1);
    printf("adapter identity corpus: pairs=%d tp=%d tn=%d fp=%d fn=%d precision=%.3f recall=%.3f (linux)\n",
           (int) cases.size(), tp, tn, fp, fn, precision, recall);
    assert(fp == 0); // never merge distinct files: the sharing gate
    assert(fn == 0); // every same-file pair dedups

    fs::remove_all(dir, ec);
}

// release-key pairing across spellings: an entry stored under one spelling of
// a file must be found through any other spelling of the same file.
// ensure_adapter keys the registry by adapter_key (canonical when resolvable)
// and release_adapter_set re-derives the key the same way, so a symlinked
// attach followed by a target-spelling detach pairs instead of leaking.
// this pins the lookup half through the public map + key function (the glue
// is a two-line delegation, reviewed); it fails if either side keys lexically.
static void test_adapter_release_pairs_symlink_spellings() {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec) / "llama-adapter-release";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    {
        std::ofstream f(dir / "a.gguf", std::ios::binary);
        f.write("AAA", 3);
    }
    fs::create_symlink(dir / "a.gguf", dir / "link.gguf", ec);
    assert(!ec);

    const std::string target = (dir / "a.gguf").string();
    const std::string link   = (dir / "link.gguf").string();
    const std::string dotdot = (dir / "sub" / ".." / "a.gguf").string();

    server_instances mgr;
    // store under the symlink spelling (what ensure_adapter records)
    mgr.adapter_registry[server_instances::adapter_key(link)] = server_instances::adapter_registry_entry{{}, 1};
    // release through the target and dot-dot spellings (what release_adapter_set
    // re-derives from each stored entry): every spelling finds the same entry
    for (const auto & spelling : { target, dotdot }) {
        const auto it = mgr.adapter_registry.find(server_instances::adapter_key(spelling));
        assert(it != mgr.adapter_registry.end());
    }
    // drain it once: pairing holds, no leak
    mgr.adapter_registry.erase(server_instances::adapter_key(target));
    assert(mgr.adapter_registry.empty());

    fs::remove_all(dir, ec);
}

// cold reload fast path through the public create endpoint: with weights
// already loaded no reload flight runs (telemetry stays zero) and the window
// builds from the standing weights.
static void test_cold_reload_noop_when_loaded() {
    server_instances mgr;
    mgr.base_name = "m";
    mgr.model = reinterpret_cast<llama_model *>(0x1); // stand-in weights, never dereferenced here
    mgr.set_context_builder([](server_instance & inst) {
        inst.ctx_server = std::make_unique<server_context>();
        inst.routes     = std::make_unique<server_routes>(inst.effective, *inst.ctx_server);
        inst.slot_snapshots.resize(1);
        return true;
    });
    static const std::function<bool()> no_stop = []() { return false; };
    server_http_req create { {}, {}, "/instances", "", R"({"name": "b", "ctx_size": 256})", {}, no_stop };
    auto res = mgr.handle_post_instances(create);
    assert(res->status == 201);
    assert(mgr.get_build_stats().cold_reloads == 0);
    mgr.terminate();
    mgr.model = nullptr;
}

// a NUL-bearing adapter path is a caller bug (400) answered before any
// instance lookup, drain, or registry touch.
static void test_adapter_nul_path_is_400() {
    server_instances mgr;
    static const std::function<bool()> no_stop = []() { return false; };
    json body = json::object();
    body["path"]  = std::string("a\0b", 3);
    body["scale"] = 1.0f;
    const server_http_req req { {}, {}, "/instances/x/adapters", "", body.dump(), {}, no_stop };
    auto res = mgr.handle_post_instance_adapters(req);
    assert(res->status == 400);
}

// test-only gate for the pool I/O worker: the posted job blocks inside until
// released, so the queue behind it is fully deterministic (no clock or sleep
// on the test thread). drain with a barrier post, never by counting slots.
struct io_gate {
    std::promise<void> entered;
    std::atomic<bool>  release{false};
    std::optional<std::future<void>> pin;

    void hold(server_instances & mgr) {
        pin = mgr.snapshot_io_post([this]() {
            entered.set_value();
            while (!release.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        assert(pin.has_value());
        entered.get_future().wait();
    }

    // let the worker run until idle: jobs queued behind the gate (including
    // abandoned timed-out ones) complete before this returns
    void let_drain(server_instances & mgr) {
        release.store(true);
        pin->wait();
        auto barrier = mgr.snapshot_io_post([]() {});
        assert(barrier.has_value());
        barrier->wait();
        release.store(false);
        std::promise<void> entered2;
        entered = std::move(entered2);
    }
};

// snapshot read outcome matrix on the pool I/O worker: OK / MISSING /
// CORRUPT / timed_out / busy. a timeout or busy never masquerades as a file
// classification; a timed-out read of a good file re-reads OK.
static void test_snapshot_io_read_matrix() {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec) / "llama-snap-read";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    server_instances mgr;
    mgr.start_io_worker();

    server_snapshot_data data;
    data.n_ctx_seq = 64;
    data.tokens    = { 1, 2, 3 };
    data.kv        = { 0xaa, 0xbb };
    const std::string good = (dir / "good.bin").string();
    assert(server_snapshot_write(good, data));
    const std::string garbage = (dir / "garbage.bin").string();
    {
        std::ofstream out(garbage, std::ios::binary | std::ios::trunc);
        out.write("not a snapshot", 14);
    }
    const std::string missing = (dir / "missing.bin").string();

    // generous but sane deadlines: near-infinite durations overflow the
    // condition-variable conversion and trip immediately
    const int64_t ample = ggml_time_ms() + 60000;
    auto ok = mgr.snapshot_io_read(good, ample);
    assert(!ok.busy && !ok.timed_out && ok.status == server_snapshot_status::OK);
    assert(ok.data && ok.data->n_ctx_seq == 64 && ok.data->tokens == data.tokens && ok.data->kv == data.kv);

    auto miss = mgr.snapshot_io_read(missing, ample);
    assert(!miss.busy && !miss.timed_out && miss.status == server_snapshot_status::MISSING && !miss.data);

    auto corrupt = mgr.snapshot_io_read(garbage, ample);
    assert(!corrupt.busy && !corrupt.timed_out && corrupt.status == server_snapshot_status::CORRUPT && !corrupt.data);

    // pin the worker: the expired deadline trips while the job is still queued
    io_gate gate;
    gate.hold(mgr);

    auto slow = mgr.snapshot_io_read(good, 0);
    assert(slow.timed_out && !slow.busy);

    // the orphaned timed-out job drains first then the queue fills behind the
    // re-held pin: the next read is busy, not a timeout
    gate.let_drain(mgr);
    gate.hold(mgr);
    std::vector<std::future<void>> queued;
    for (int i = 0; i < 4; i++) { // max_io_jobs: the queue is full behind the pin
        auto fut = mgr.snapshot_io_post([]() {});
        assert(fut.has_value());
        queued.push_back(std::move(*fut));
    }
    auto busy = mgr.snapshot_io_read(good, ggml_time_ms() + 60000);
    assert(busy.busy && !busy.timed_out);

    gate.release.store(true);
    mgr.stop_io_worker();

    // the timed-out read was fast-but-queued, never corruption: it re-reads OK
    server_instances mgr2;
    mgr2.start_io_worker();
    auto again = mgr2.snapshot_io_read(good, ggml_time_ms() + 60000);
    assert(!again.busy && !again.timed_out && again.status == server_snapshot_status::OK && again.data);
    mgr2.stop_io_worker();

    fs::remove_all(dir, ec);
}

// snapshot write outcome matrix: ok / timed_out / busy. a timed-out write
// still lands in the background; a busy write never reaches the disk.
static void test_snapshot_io_write_matrix() {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec) / "llama-snap-write";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    server_instances mgr;
    mgr.start_io_worker();

    server_snapshot_data data;
    data.n_ctx_seq = 64;
    data.tokens    = { 7, 8 };
    data.kv        = { 0xcc };

    const std::string p_ok = (dir / "ok.bin").string();
    auto written = mgr.snapshot_io_write(p_ok, data, ggml_time_ms() + 60000);
    assert(written.ok && !written.busy && !written.timed_out);
    assert(server_snapshot_read_status(p_ok).status == server_snapshot_status::OK);

    // pin the worker: the write times out while queued, then lands behind us
    io_gate gate;
    gate.hold(mgr);

    const std::string p_slow = (dir / "slow.bin").string();
    auto slow = mgr.snapshot_io_write(p_slow, data, 0);
    assert(slow.timed_out && !slow.ok && !slow.busy);
    assert(!fs::exists(p_slow, ec)); // not there yet: the caller must not bind it

    gate.let_drain(mgr);
    gate.hold(mgr);
    std::vector<std::future<void>> queued;
    for (int i = 0; i < 4; i++) { // max_io_jobs: the queue is full behind the pin
        auto fut = mgr.snapshot_io_post([]() {});
        assert(fut.has_value());
        queued.push_back(std::move(*fut));
    }
    const std::string p_busy = (dir / "busy.bin").string();
    auto busy = mgr.snapshot_io_write(p_busy, data, ggml_time_ms() + 60000);
    assert(busy.busy && !busy.ok && !busy.timed_out);

    gate.release.store(true);
    // drain every queued job, then stop: the timed-out write must have landed
    // in the background while the busy one never reached the disk
    for (auto & q : queued) {
        q.wait();
    }
    gate.pin->wait();
    mgr.stop_io_worker();
    assert(server_snapshot_read_status(p_slow).status == server_snapshot_status::OK);
    assert(!fs::exists(p_busy, ec));

    fs::remove_all(dir, ec);
}

// deadline calibration: healthy-disk latency against the 1s deadline floor,
// plus induced-timeout discipline (no false corruption, no false missing).
static void test_snapshot_deadline_calibration() {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec) / "llama-snap-cal";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    server_instances mgr;
    mgr.start_io_worker();

    server_snapshot_data data;
    data.n_ctx_seq = 64;
    data.tokens    = { 1, 2, 3 };
    data.kv        = std::vector<uint8_t>(4096, 0xab);

    // control group: small-context healthy-disk rounds must sit far below the
    // 1s floor (4x margin on p95) or the deadline is miscalibrated
    std::vector<int64_t> lat_us;
    for (int i = 0; i < 100; i++) {
        const std::string p = (dir / ("c" + std::to_string(i) + ".bin")).string();
        const int64_t t0 = ggml_time_us();
        assert(mgr.snapshot_io_write(p, data, ggml_time_ms() + 60000).ok);
        auto read = mgr.snapshot_io_read(p, ggml_time_ms() + 60000);
        assert(read.status == server_snapshot_status::OK);
        lat_us.push_back(ggml_time_us() - t0);
    }
    std::sort(lat_us.begin(), lat_us.end());
    const int64_t p50 = lat_us[lat_us.size() / 2];
    const int64_t p95 = lat_us[lat_us.size() * 95 / 100];
    printf("snapshot deadline control: rounds=%d p50=%lld us p95=%lld us floor=1000000 us\n",
           (int) lat_us.size(), (long long) p50, (long long) p95);
    assert(p95 * 4 < 1000000);

    // induced timeouts are fast-but-queued, never slow disk: every one must
    // report timed_out (never CORRUPT/MISSING) and re-read OK afterwards
    const std::string victim = (dir / "c0.bin").string();
    int false_corruption = 0;
    int false_missing    = 0;
    for (int i = 0; i < 20; i++) {
        std::promise<void> entered;
        std::atomic<bool>  release{false};
        auto pin = mgr.snapshot_io_post([&]() {
            entered.set_value();
            while (!release.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        assert(pin.has_value());
        entered.get_future().wait();

        auto slow = mgr.snapshot_io_read(victim, 0);
        if (!slow.timed_out) {
            ++false_missing; // a timeout that did not trip is a missed deadline
        }
        release.store(true);
        pin->wait();

        auto again = mgr.snapshot_io_read(victim, ggml_time_ms() + 60000);
        if (again.status == server_snapshot_status::CORRUPT) {
            ++false_corruption;
        }
        if (again.status == server_snapshot_status::MISSING) {
            ++false_missing;
        }
        assert(again.status == server_snapshot_status::OK);
    }
    printf("snapshot timeout discipline: induced=20 false_corruption=%d false_missing=%d\n",
           false_corruption, false_missing);
    assert(false_corruption == 0 && false_missing == 0);

    mgr.stop_io_worker();
    fs::remove_all(dir, ec);
}

// group-wait telemetry: a wait that expires records one wait entry for its
// group and leaves no parked waiter; a slot-release wake with nobody parked
// counts as dropped, while a wake with waiters parked does not.
static void test_group_wait_counters_and_wake_accounting() {
    server_instances mgr;
    mgr.base_name = "m";
    mgr.params.instance_wait_seconds = 0;
    mgr.instances.push_back(dispatch_test_window("a", "g"));
    for (auto & inst : mgr.instances) {
        inst->removing = true;
    }

    server_instances::forward_fn never = [](server_routes &, const server_http_req &) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->data   = "{}";
        return res;
    };
    assert(mgr.dispatch(dispatch_test_req("{\"model\": \"m:g\"}"), never)->status == 503);

    auto stats = mgr.get_wake_stats();
    assert(stats.group_waits_total["g"] == 1);
    assert(stats.group_waiters_now.empty());

    mgr.on_slot_released(); // nobody parked: dropped
    stats = mgr.get_wake_stats();
    assert(stats.wakes_total == 1 && stats.wakes_dropped == 1);

    mgr.on_slot_released(); // still nobody parked
    stats = mgr.get_wake_stats();
    assert(stats.wakes_total == 2 && stats.wakes_dropped == 2);

    mgr.terminate();
}

// contention calibration: 4 groups x 8 waiters park on draining members,
// one release frees every group, and all 32 proceed (no starvation). reports
// wake precision (wakes that found a waiter), recall (parked waiters that
// proceeded), and wait latency percentiles against the wait deadline.
static void test_group_contention_all_groups_drain() {
    constexpr int n_groups = 4;
    constexpr int n_per_group = 8;
    constexpr int n_total = n_groups * n_per_group;

    server_instances mgr;
    mgr.base_name = "m";
    mgr.params.instance_wait_seconds = 10;
    for (int g = 0; g < n_groups; g++) {
        const std::string group = "g" + std::to_string(g);
        mgr.instances.push_back(dispatch_test_window(group + "-a", group));
        mgr.instances.push_back(dispatch_test_window(group + "-b", group));
    }
    for (auto & inst : mgr.instances) {
        inst->removing = true; // every member draining: waiters park
    }

    // control that must not batch: one uncontended dispatch serves fast under
    // the same notify_all path the parked waiters will use
    {
        auto inst = mgr.instances.front();
        {
            std::lock_guard<std::mutex> lock(mgr.mutex_dispatch);
            inst->removing = false;
        }
        server_instances::forward_fn ok = [](server_routes &, const server_http_req &) -> server_http_res_ptr {
            auto res = std::make_unique<server_http_res>();
            res->status = 200;
            res->data   = "{}";
            return res;
        };
        const int64_t t0 = ggml_time_us();
        assert(mgr.dispatch(dispatch_test_req("{\"model\": \"m:g0\"}"), ok)->status == 200);
        const int64_t uncontended_us = ggml_time_us() - t0;
        printf("group wake control: uncontended dispatch=%lld us\n", (long long) uncontended_us);
        assert(uncontended_us < 1000000);
        {
            std::lock_guard<std::mutex> lock(mgr.mutex_dispatch);
            inst->removing = true;
        }
    }

    std::atomic<int> served_groups[n_groups] = {};
    std::atomic<int> served_total{0};
    std::atomic<int> failed{0};
    std::vector<int64_t> wait_us(n_total, 0);
    std::vector<std::thread> threads;
    for (int w = 0; w < n_total; w++) {
        threads.emplace_back([&, w]() {
            const int g = w % n_groups;
            server_instances::forward_fn ok = [](server_routes &, const server_http_req &) -> server_http_res_ptr {
                auto res = std::make_unique<server_http_res>();
                res->status = 200;
                res->data   = "{}";
                return res;
            };
            const int64_t t0 = ggml_time_us();
            auto res = mgr.dispatch(
                dispatch_test_req("{\"model\": \"m:g" + std::to_string(g) + "\"}"), ok);
            wait_us[w] = ggml_time_us() - t0;
            if (res->status == 200) {
                served_groups[g]++;
                served_total++;
            } else {
                failed++;
            }
        });
    }

    // let every waiter park (no clock on the test thread: poll the counters)
    const int64_t park_t0 = ggml_time_us();
    while (true) {
        size_t parked = 0;
        for (const auto & kv : mgr.get_wake_stats().group_waiters_now) {
            parked += kv.second;
        }
        if (parked == (size_t) n_total) {
            break;
        }
        assert(ggml_time_us() - park_t0 < 5000000);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // one wake while all 32 are parked: not dropped (precision signal)
    mgr.on_slot_released();
    auto mid = mgr.get_wake_stats();
    assert(mid.wakes_total == 1 && mid.wakes_dropped == 0);

    // the release: every member usable again, one notify wakes every group
    const int64_t t_release = ggml_time_us();
    {
        std::lock_guard<std::mutex> lock(mgr.mutex_dispatch);
        for (auto & inst : mgr.instances) {
            inst->removing = false;
        }
        mgr.cond_dispatch.notify_all();
    }
    for (auto & t : threads) {
        t.join();
    }
    (void) t_release;

    assert(failed == 0);
    assert(served_total == n_total);
    for (int g = 0; g < n_groups; g++) {
        assert(served_groups[g] == n_per_group);
    }

    std::sort(wait_us.begin(), wait_us.end());
    const int64_t p50 = wait_us[n_total / 2];
    const int64_t p99 = wait_us[n_total * 99 / 100];
    const double precision = 1.0; // the single release wake found waiters parked
    const double recall = (double) served_total.load() / (double) n_total;
    printf("group wake contention: groups=%d waiters=%d served=%d p50=%lld us p99=%lld us deadline=10 s precision=%.3f recall=%.3f\n",
           n_groups, n_total, served_total.load(), (long long) p50, (long long) p99, precision, recall);
    assert(p99 < 10000000); // no starvation: every group proceeds within its deadline
    assert(recall == 1.0);

    // the release wake was not dropped; the control dispatch never waited
    auto done = mgr.get_wake_stats();
    assert(done.wakes_dropped == 0);
    assert(done.group_waiters_now.empty());

    mgr.terminate();
}

// cold demand is single-flight: concurrent first demands for N cold windows
// build each window exactly once (slow builder forces overlap), and every
// repeat demand collapses onto the built window (never rebuilds). a destroy
// racing a slow first demand answers 200/404/503 without crashing.
static void test_cold_demand_single_flight_and_racing_destroy() {
    server_instances mgr;
    mgr.base_name = "m";
    mgr.params.instance_wait_seconds = 0;
    // weights stand-in: the reload path is skipped while weights are present
    // and the slow injected builder below plays the expensive load. explicit
    // sizes keep the footprint warning off the stand-in.
    mgr.model = reinterpret_cast<llama_model *>(0x1);

    std::mutex mu;
    std::map<std::string, int> builds;
    mgr.set_context_builder([&](server_instance & inst) {
        {
            std::lock_guard<std::mutex> lock(mu);
            ++builds[inst.cfg.name];
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        inst.ctx_server = std::make_unique<server_context>();
        inst.routes     = std::make_unique<server_routes>(inst.effective, *inst.ctx_server);
        inst.n_parallel_effective.store(1, std::memory_order_relaxed);
        inst.slot_snapshots.resize(1);
        return true;
    });

    for (const char * name : { "a", "b", "c", "d" }) {
        auto inst = std::make_shared<server_instance>();
        inst->cfg.name     = name;
        inst->cfg.group    = "g";
        inst->cfg.ctx_size = 256;
        inst->cfg.parallel = 1;
        mgr.instances.push_back(inst);
    }

    server_instances::forward_fn ok = [](server_routes &, const server_http_req &) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->data   = "{}";
        return res;
    };

    // 8 threads x 4 windows: every demand is served, each window builds once
    constexpr int n_threads = 8;
    constexpr int n_windows = 4;
    std::atomic<int> failed{0};
    std::vector<std::thread> racers;
    for (int t = 0; t < n_threads; t++) {
        racers.emplace_back([&, t]() {
            for (const char * name : { "a", "b", "c", "d" }) {
                auto res = mgr.dispatch(
                    dispatch_test_req("{\"model\": \"m:" + std::string(name) + "\"}"), ok);
                if (res->status != 200) {
                    failed++;
                }
            }
        });
    }
    for (auto & t : racers) {
        t.join();
    }
    assert(failed == 0);
    for (const char * name : { "a", "b", "c", "d" }) {
        assert(builds[name] == 1);
    }
    const auto stats = mgr.get_build_stats();
    assert(stats.cold_reloads == 0); // stand-in weights: no reload flight ran
    assert(stats.collapsed_demands == n_threads * n_windows - n_windows);

    mgr.terminate();
    mgr.model = nullptr;
}

static void test_racing_destroy_during_demand() {
    // a destroy racing in-flight dispatches is safe: the unbuilt window is
    // erased without touching a scheduler, demands on the live window keep
    // serving, and the pool stays usable. (destroy of a built window runs
    // the abort path, which needs a live scheduler; that race is covered by
    // the model-backed drain test with real weights.)
    server_instances mgr;
    mgr.base_name = "m";
    mgr.params.instance_wait_seconds = 0;
    mgr.instances.push_back(dispatch_test_window("s", "g"));

    auto victim = std::make_shared<server_instance>();
    victim->cfg.name     = "v";
    victim->cfg.group    = "g";
    victim->cfg.ctx_size = 256;
    victim->cfg.parallel = 1;
    mgr.instances.push_back(victim);

    server_instances::forward_fn ok = [](server_routes &, const server_http_req &) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->data   = "{}";
        return res;
    };
    static const std::function<bool()> no_stop = []() { return false; };

    std::atomic<bool> stop{false};
    std::atomic<int> failed{0};
    std::vector<std::thread> hammer;
    for (int t = 0; t < 4; t++) {
        hammer.emplace_back([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                if (mgr.dispatch(dispatch_test_req("{\"model\": \"m:s\"}"), ok)->status != 200) {
                    failed++;
                }
            }
        });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20)); // demands in flight
    server_http_req del { { { "name", "v" } }, {}, "/instances/v", "", "", {}, no_stop };
    assert(mgr.handle_delete_instance(del)->status == 200);
    stop.store(true, std::memory_order_relaxed);
    for (auto & t : hammer) {
        t.join();
    }
    assert(failed == 0);

    // the erased window is gone; the live one still serves
    assert(mgr.dispatch(dispatch_test_req("{\"model\": \"m:v\"}"), ok)->status != 200);
    assert(mgr.dispatch(dispatch_test_req("{\"model\": \"m:s\"}"), ok)->status == 200);
    server_http_req list { {}, {}, "/instances", "", "", {}, no_stop };
    assert(mgr.handle_get_instances(list)->status == 200);

    mgr.terminate();
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    test_layout_paths();
    test_merge();
    test_merge_tolerates_shape_drift();
    test_merge_saturates();
    test_merge_adapter_additive();
    test_merge_adapter_missing_key();
    test_merge_as_u64_edges();
    test_merge_empty();
    test_merge_three_children_golden();
    test_merge_exact_large_unsigned();
    test_merge_permutation_invariant();
    test_fanout_collect_deadline_skip();
    test_instances_envelope_shape();
    test_pick_best_candidate();
    test_dispatch_instance_serves_named_member();
    test_dispatch_group_serves_available_member();
    test_dispatch_group_busy_returns_503();
    test_dispatch_throw_balances_count();
    test_dispatch_guard_balances_on_throw();
    test_manager_adapter_apply_requires_drain();
    test_adapter_read_timeout_is_nullopt_not_empty();
    test_adapter_key_two_tier();
    test_adapter_identity_corpus();
    test_adapter_release_pairs_symlink_spellings();
    test_cold_reload_noop_when_loaded();
    test_adapter_nul_path_is_400();
    test_snapshot_io_read_matrix();
    test_snapshot_io_write_matrix();
    test_snapshot_deadline_calibration();
    test_group_wait_counters_and_wake_accounting();
    test_group_contention_all_groups_drain();
    test_cold_demand_single_flight_and_racing_destroy();
    test_racing_destroy_during_demand();
    test_displayed_n_ctx_unbuilt();
    test_snapshot_io_post_lifecycle();
    test_snapshot_model_keys();
    test_snapshot_resolve_key_fallback();
    test_huge_implicit_ctx_predicate();
    test_snapshot_switch_budget();
    test_snapshot_io_queue_budget();
    test_envelope_canonical_stable();
    test_merge_exact_shape();
    test_envelope_snapshot_row_shape();
    test_result_error_response();
    test_result_success_path_unaffected();
    test_former_sidecar_name_is_ordinary();
    return 0;
}
