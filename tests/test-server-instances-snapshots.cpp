#include "server-snapshot.h"
#include "server-models.h"

#include <cstdint>
#include <string>
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

static void test_merge_saturates() {
    json big = {
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

int main() {
    test_layout_paths();
    test_merge();
    test_merge_tolerates_shape_drift();
    test_merge_saturates();
    return 0;
}
