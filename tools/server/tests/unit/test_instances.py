import threading
import pytest
from utils import *
import os
import requests
import shutil
import tempfile
import urllib.parse

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _get_instances():
    res = server.make_request("GET", "/instances")
    assert res.status_code == 200
    return res.body


def _get_instance_ids() -> set[str]:
    return {inst["id"] for inst in _get_instances()["instances"]}


def test_instances_two_pool_list():
    global server
    server.instances = [
        "swarm0:group=swarm:ctx=512",
        "swarm1:group=swarm:ctx=512",
        "ledger:ctx=512:pinned:default",
    ]
    server.n_ctx = 512
    server.start()

    body = _get_instances()
    ids = {inst["id"] for inst in body["instances"]}
    assert ids == {
        "tinyllama-2:swarm0",
        "tinyllama-2:swarm1",
        "tinyllama-2:ledger",
    }

    # no demand yet: every configured window is registered but unbuilt, so the
    # entries carry identity with zero bytes and an unloaded state
    for inst in body["instances"]:
        assert "model_bytes" in inst
        assert "context_bytes" in inst
        assert "compute_bytes" in inst
        assert "total_bytes" in inst
        assert inst["total_bytes"] == inst["model_bytes"] + inst["context_bytes"] + inst["compute_bytes"]
        assert "vram_bytes" in inst
        assert inst["vram_bytes"] == inst["context_bytes"] + inst["compute_bytes"]
        assert inst["state"] == "unloaded"
        assert inst["context_bytes"] == 0
        assert inst["compute_bytes"] == 0
        assert inst["last_used"] == -1
        assert inst["last_used_epoch"] == -1
        # this branch has no auto-sleep: neither field may be reported
        assert "sleep_idle_seconds" not in inst
        assert "no_sleep" not in inst

    # one demand builds exactly one window: swarm0 loads, the rest stay unloaded
    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:swarm0",
        "prompt": "What is the capital of France?",
        "n_predict": 4,
    })
    assert res.status_code == 200

    states = {inst["id"]: inst["state"] for inst in _get_instances()["instances"]}
    assert states == {
        "tinyllama-2:swarm0": "loaded",
        "tinyllama-2:swarm1": "unloaded",
        "tinyllama-2:ledger": "unloaded",
    }

    # last_used_epoch is the wall-clock twin of the monotonic last_used: unix epoch
    # seconds of the most recent slot release, so a router can compute idle time
    # against its own clock. unused/unbuilt instances report -1 for both.
    import time
    now = int(time.time())
    epochs = {inst["id"]: inst["last_used_epoch"] for inst in _get_instances()["instances"]}
    assert epochs["tinyllama-2:swarm1"] == -1
    assert epochs["tinyllama-2:ledger"] == -1
    assert now - 300 <= epochs["tinyllama-2:swarm0"] <= now + 60
    used = {inst["id"]: inst for inst in _get_instances()["instances"]}["tinyllama-2:swarm0"]
    assert used["last_used"] >= 0

    # the envelope sums a 64-bit total; the shared model bytes are counted once
    assert "total" in body
    total = body["total"]
    assert set(total.keys()) == {"model", "context", "compute", "adapter", "total"}
    assert total["total"] == total["model"] + total["context"] + total["compute"] + total["adapter"]

    # the default instance is the pinned ledger
    defaults = [inst for inst in body["instances"] if inst["is_default"]]
    assert len(defaults) == 1
    assert defaults[0]["id"].endswith(":ledger")
    assert defaults[0]["pinned"] is True


def test_instances_routing():
    global server
    server.instances = [
        "swarm0:group=swarm:ctx=512",
        "swarm1:group=swarm:ctx=512",
        "ledger:ctx=512:pinned:default",
    ]
    server.n_ctx = 512
    server.temperature = 0.0
    server.start()

    # exact instance via the model id
    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:ledger",
        "prompt": "What is the capital of France?",
        "n_predict": 8,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0

    # group routing selects the first available member
    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:swarm",
        "prompt": "What is the capital of France?",
        "n_predict": 8,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_instances_crud():
    global server
    server.instances = ["swarm0:group=swarm:ctx=512"]
    server.n_ctx = 512
    server.start()

    # create
    res = server.make_request("POST", "/instances", data={
        "name": "work",
        "group": "swarm",
        "ctx_size": 512,
        "parallel": 1,
    })
    assert res.status_code == 201
    assert any(i.endswith(":work") for i in _get_instance_ids())

    # duplicate -> 409
    res = server.make_request("POST", "/instances", data={
        "name": "work",
        "group": "swarm",
        "ctx_size": 512,
    })
    assert res.status_code == 409

    # pin / unpin
    res = server.make_request("POST", "/instances/work/pin")
    assert res.status_code == 200
    res = server.make_request("POST", "/instances/work/unpin")
    assert res.status_code == 200

    # resize
    res = server.make_request("POST", "/instances/work/resize", data={"ctx_size": 1024})
    assert res.status_code == 200
    assert res.body["n_ctx"] == 1024

    # delete
    res = server.make_request("DELETE", "/instances/work")
    assert res.status_code == 200
    assert not any(i.endswith(":work") for i in _get_instance_ids())

    # unknown instance -> 404
    res = server.make_request("DELETE", "/instances/nope")
    assert res.status_code == 404


def test_instances_envelope():
    """The /instances envelope carries the summed memory breakdown; there is no /memory
    endpoint in this branch."""
    global server
    server.instances = ["swarm0:group=swarm:ctx=512"]
    server.n_ctx = 512
    server.start()

    body = _get_instances()
    assert len(body["instances"]) == 1
    assert "total" in body
    total = body["total"]
    assert set(total.keys()) == {"model", "context", "compute", "adapter", "total"}
    assert total["total"] == total["model"] + total["context"] + total["compute"] + total["adapter"]

    inst = body["instances"][0]
    assert inst["total_bytes"] == inst["model_bytes"] + inst["context_bytes"] + inst["compute_bytes"]
    # no demand yet: registered but unbuilt
    assert inst["state"] == "unloaded"
    assert inst["context_bytes"] == 0
    assert inst["compute_bytes"] == 0


def test_instances_props_shape():
    """Golden for the /props shape on an instance server: total_slots plus the
    per-instance array key set."""
    global server
    server.instances = ["swarm0:group=swarm:ctx=512"]
    server.n_ctx = 512
    server.start()

    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert "total_slots" in res.body
    assert res.body["total_slots"] == 1
    assert "instances" in res.body
    assert isinstance(res.body["instances"], list)
    assert len(res.body["instances"]) == 1
    assert set(res.body["instances"][0].keys()) == {"name", "group", "n_ctx"}
    assert res.body["instances"][0]["name"] == "swarm0"
    assert res.body["instances"][0]["group"] == "swarm"


def test_instances_concurrency_overlap():
    """Two simultaneous long generations on two different instances overlap (1b).

    Each instance runs its own scheduler thread, so two requests to different
    instances must not serialize: running both concurrently takes roughly one
    generation, not two.
    """
    global server
    server.instances = [
        "a:group=swarm:ctx=1024",
        "b:group=swarm:ctx=1024",
    ]
    server.n_ctx = 1024
    server.temperature = 0.0
    server.start()

    def generate(instance: str) -> float:
        t0 = time.time()
        res = server.make_request("POST", "/completion", data={
            "model": f"tinyllama-2:{instance}",
            "prompt": "The quick brown fox. ",
            "n_predict": 900,
        })
        assert res.status_code == 200
        return time.time() - t0

    # windows materialize on first demand (and first demands serialize on the pool
    # lock), so warm both windows first: this test measures scheduler overlap,
    # not build time
    generate("a")
    generate("b")

    # two requests to different instances, fired concurrently
    results = parallel_function_calls([(generate, ("a",)), (generate, ("b",))])
    concurrent = max(results)

    # two requests to the same instance, serialized (one scheduler thread per
    # instance means a single instance can only process one request at a time)
    serial = generate("a") + generate("a")

    # concurrent should be close to one generation, not two; assert it is
    # strictly less than the serialized pair
    assert concurrent < serial, (
        f"concurrent generations to two instances did not overlap: "
        f"concurrent={concurrent:.2f}s serial={serial:.2f}s"
    )


# --- H2: a plain request must not clear a slot's snapshot binding ---
def test_instances_snapshot_binding_persists():
    """A request without `snapshot` used to clear the slot binding, so the extended KV was
    never saved back to the bound snapshot when switching away (foo.bin did not grow). After
    the fix, a plain request leaves the binding intact and a later switch-away saves the
    extended KV to the bound snapshot file."""
    global server
    snap_dir = tempfile.mkdtemp()
    server.instances = ["a:group=swarm:ctx=512"]
    server.n_ctx = 512
    server.temperature = 0.0
    server.slot_save_path = snap_dir
    server.start()

    key = "tinyllama-2"
    # writes land in the hashed per-instance dir: <snap>/<key>-<fnv1a(key)>/a/
    h = 2166136261
    for c in key.encode():
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    model_dir = os.path.join(snap_dir, "%s-%08x" % (key, h), "a")
    os.makedirs(model_dir, exist_ok=True)
    foo_path = os.path.join(model_dir, "foo.bin")

    # seed a fallback snapshot `bar` so the switch-away below can load it
    server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a", "prompt": "bar seed", "n_predict": 1,
    })
    server.make_request("POST", "/instances/a/snapshot", data={"name": "bar"})

    # save `foo` over a SHORT KV, note its size S1
    server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a", "prompt": "short", "n_predict": 1,
    })
    res = server.make_request("POST", "/instances/a/snapshot", data={"name": "foo"})
    assert res.status_code == 201
    s1 = os.path.getsize(foo_path)

    # a plain request (no snapshot) with a LONG prompt greatly extends the slot KV while
    # keeping the `foo` binding
    server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a",
        "prompt": "The quick brown fox jumps over the lazy dog and runs far away into the deep dark forest where no one can ever find it again",
        "n_predict": 16,
    })

    # switch away to `bar`: the slot's extended KV is saved back to `foo`, growing it
    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a", "snapshot": "bar", "prompt": " cont", "n_predict": 4,
    })
    assert res.status_code == 200

    deadline = time.time() + 5
    s2 = s1
    while time.time() < deadline:
        if os.path.exists(foo_path):
            s2 = os.path.getsize(foo_path)
        if s2 > s1:
            break
        time.sleep(0.2)
    assert s2 > s1, f"foo.bin did not grow after the plain request + switch away (S1={s1}, S2={s2})"


# --- L1: POST /instances rejects names/groups that break the model-id grammar (400) ---
def test_instances_post_invalid_name_400():
    global server
    server.instances = ["a:group=swarm:ctx=512"]
    server.n_ctx = 512
    server.start()

    for bad_name in ["a:b", "a=b", "a/b", "a b", "latest"]:
        res = server.make_request("POST", "/instances", data={
            "name": bad_name, "group": "swarm", "ctx_size": 512,
        })
        assert res.status_code == 400, f"name {bad_name!r} should be rejected with 400"

    # 'latest' is a reserved routing token, also rejected as a group
    res = server.make_request("POST", "/instances", data={
        "name": "work", "group": "latest", "ctx_size": 512,
    })
    assert res.status_code == 400

    # a valid name still creates an instance
    res = server.make_request("POST", "/instances", data={
        "name": "work", "group": "swarm", "ctx_size": 512,
    })
    assert res.status_code == 201


# --- M2: snapshot management round-trip (save / list / apply / delete) ---
def test_instances_snapshot_management():
    global server
    snap_dir = tempfile.mkdtemp()
    server.instances = ["a:group=swarm:ctx=512"]
    server.n_ctx = 512
    server.slot_save_path = snap_dir
    server.start()

    # generate some context so slot 0 has KV to save
    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a",
        "prompt": "The quick brown fox jumps over the lazy dog",
        "n_predict": 4,
    })
    assert res.status_code == 200

    # save a snapshot of slot 0
    res = server.make_request("POST", "/instances/a/snapshot", data={"name": "foo"})
    assert res.status_code == 201

    # list snapshots
    res = server.make_request("GET", "/instances/a/snapshots")
    assert res.status_code == 200
    assert any(s["name"] == "foo" for s in res.body["snapshots"])

    # apply the snapshot via a request-time switch
    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a",
        "snapshot": "foo",
        "prompt": " continues",
        "n_predict": 4,
    })
    assert res.status_code == 200

    # delete the snapshot; it unbinds and is gone from the list
    res = server.make_request("DELETE", "/instances/a/snapshot/foo")
    assert res.status_code == 200
    res = server.make_request("GET", "/instances/a/snapshots")
    assert not any(s["name"] == "foo" for s in res.body["snapshots"])


# --- M7: a snapshot-save burst at startup resolves fast and never hangs ---
def test_instances_snapshot_startup_burst():
    """Snapshot saves fired the moment the server accepts requests must each
    resolve with a valid status, never hang, and never 500. Saves landing
    before the pool I/O worker starts fail fast with the retriable 503;
    saves landing on the still-unbuilt window 404 (a save must not build
    a window); saves after first demand succeed."""
    global server
    snap_dir = tempfile.mkdtemp()
    server.instances = ["a:group=swarm:ctx=512"]
    server.n_ctx = 512
    server.slot_save_path = snap_dir
    server.start()

    def save(name: str) -> int:
        res = server.make_request("POST", "/instances/a/snapshot", data={"name": name})
        return res.status_code

    # burst immediately: no demand has built the window yet
    t0 = time.time()
    results = parallel_function_calls([(save, (f"burst-{i}",)) for i in range(8)])
    burst_s = time.time() - t0

    # every save resolved (no hang), each with a valid startup status
    assert len(results) == 8
    for status in results:
        assert status in (201, 404, 503), f"unexpected startup save status: {status}"
    # 8 deadline-bounded saves must not serialize into a long stall
    assert burst_s < 60, f"startup snapshot burst stalled: {burst_s:.1f}s"

    # after first demand the window is built and a save succeeds
    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a",
        "prompt": "The quick brown fox jumps over the lazy dog",
        "n_predict": 4,
    })
    assert res.status_code == 200
    res = server.make_request("POST", "/instances/a/snapshot", data={"name": "after"})
    assert res.status_code == 201


# --- M10: sequential snapshot switches stay within budget (never 503) ---
def test_instances_snapshot_switch_within_budget():
    """A save / switch-away / switch-back cycle runs one switch at a time with
    an empty I/O queue, so every step must succeed: a switch within the
    documented budgets (2 concurrent switches, 4 queued writes) never 503s."""
    global server
    snap_dir = tempfile.mkdtemp()
    server.instances = ["a:group=swarm:ctx=512"]
    server.n_ctx = 512
    server.slot_save_path = snap_dir
    server.start()

    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a",
        "prompt": "The quick brown fox jumps over the lazy dog",
        "n_predict": 4,
    })
    assert res.status_code == 200

    # save the starting KV as foo (a save binds the slot)
    res = server.make_request("POST", "/instances/a/snapshot", data={"name": "foo"})
    assert res.status_code == 201

    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a",
        "prompt": " continues",
        "n_predict": 4,
    })
    assert res.status_code == 200

    # save the extended KV as bar (binds bar)
    res = server.make_request("POST", "/instances/a/snapshot", data={"name": "bar"})
    assert res.status_code == 201

    # switch back to foo: saves bar back, restores foo
    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a",
        "snapshot": "foo",
        "prompt": " resumes",
        "n_predict": 4,
    })
    assert res.status_code == 200

    # and back to bar: saves foo back, restores bar
    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a",
        "snapshot": "bar",
        "prompt": " resumes",
        "n_predict": 4,
    })
    assert res.status_code == 200


# --- L5: a missing snapshot yields a clean 404 (not a 500) ---
def test_instances_snapshot_missing_404():
    global server
    snap_dir = tempfile.mkdtemp()
    server.instances = ["a:group=swarm:ctx=512"]
    server.n_ctx = 512
    server.slot_save_path = snap_dir
    server.start()

    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a",
        "snapshot": "does-not-exist",
        "prompt": "hi",
        "n_predict": 4,
    })
    assert res.status_code == 404


# --- M2: a corrupt .bin yields a clean 400 (not a 500) ---
def test_instances_snapshot_corrupt_400():
    global server
    snap_dir = tempfile.mkdtemp()
    server.instances = ["a:group=swarm:ctx=512"]
    server.n_ctx = 512
    server.slot_save_path = snap_dir
    server.start()

    # the pool's snapshot dir is the alias (':'/'/' replaced by '_'); tinyllama-2 is unchanged
    key = "tinyllama-2"
    os.makedirs(os.path.join(snap_dir, key), exist_ok=True)
    with open(os.path.join(snap_dir, key, "corrupt.bin"), "wb") as f:
        f.write(b"this is not a snapshot file")

    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a",
        "snapshot": "corrupt",
        "prompt": "hi",
        "n_predict": 4,
    })
    assert res.status_code == 400


# --- M2: restoring a snapshot saved at a different ctx size is rejected (400) ---
def test_instances_snapshot_wrong_nctx_400():
    global server
    snap_dir = tempfile.mkdtemp()
    server.instances = ["a:group=swarm:ctx=512"]
    server.n_ctx = 512
    server.slot_save_path = snap_dir
    server.start()

    server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a", "prompt": "hi", "n_predict": 4,
    })
    res = server.make_request("POST", "/instances/a/snapshot", data={"name": "s512"})
    assert res.status_code == 201

    # resize the instance; its n_ctx_seq changes, so the 512 snapshot no longer fits
    res = server.make_request("POST", "/instances/a/resize", data={"ctx_size": 1024})
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a", "snapshot": "s512", "prompt": "hi", "n_predict": 4,
    })
    assert res.status_code == 400


# --- M3: a group waiter with --instance-wait -1 never 503s on a busy group that frees ---
def test_instances_group_wait_forever():
    global server
    server.instances = [
        "a:group=swarm:ctx=512",
        "b:group=swarm:ctx=512",
    ]
    server.n_ctx = 512
    server.temperature = 0.0
    server.instance_wait_seconds = -1  # wait forever
    server.start()

    # occupy both group instances with long generations
    def gen(i: str):
        return server.make_request("POST", "/completion", data={
            "model": f"tinyllama-2:{i}",
            "prompt": "The quick brown fox. ",
            "n_predict": 900,
        })

    t1 = threading.Thread(target=gen, args=("a",))
    t2 = threading.Thread(target=gen, args=("b",))
    t1.start()
    t2.start()
    time.sleep(0.5)  # let both long generations start before the group request

    # a group request waits (predicate wait, no poll) instead of 503ing; it is served
    # once one of the two instances frees
    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:swarm",
        "prompt": "hello",
        "n_predict": 4,
    })
    t1.join()
    t2.join()
    assert res.status_code == 200, "group waiter 503ed on a busy group that frees"


# --- 1c: deleting the last instance unloads the shared weights; POST reloads them ---
def test_instances_delete_last_unloads():
    """Deleting the last instance unloads the shared weights, and a subsequent
    POST /instances reloads them (the pool goes cold purely via DELETE)."""
    global server
    server.instances = ["a:group=swarm:ctx=512"]
    server.n_ctx = 512
    server.start()

    # one demand materializes the window, so the shared weights are counted
    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a",
        "prompt": "demand",
        "n_predict": 1,
    })
    assert res.status_code == 200
    assert _get_instances()["total"]["model"] > 0

    res = server.make_request("DELETE", "/instances/a")
    assert res.status_code == 200

    # delete-last unloads the weights
    deadline = time.time() + 30
    body = {}
    while time.time() < deadline:
        body = _get_instances()
        if len(body["instances"]) == 0 and body["total"]["model"] == 0:
            break
        time.sleep(1)
    assert len(body["instances"]) == 0, "delete-last did not empty the pool"
    assert body["total"]["model"] == 0, "delete-last did not unload the weights"

    # POST /instances on the now-cold pool reloads the weights and creates the instance
    res = server.make_request("POST", "/instances", data={
        "name": "work",
        "group": "swarm",
        "ctx_size": 512,
        "parallel": 1,
    })
    assert res.status_code == 201
    body = _get_instances()
    assert body["total"]["model"] > 0, "POST /instances did not reload the shared weights"


def test_instances_kv_unified_per_slot_plain_server():
    """Control: --kv-unified-per-slot on a server without instances sizes the KV
    pool as upstream (n_parallel * per-slot) and still serves."""
    global server
    server.instances = None
    server.n_ctx = None
    server.n_slots = 2
    server.kv_unified_per_slot = 256
    server.start()

    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["total_slots"] == 2
    # pool sized to 2 * 256 = 512, split across 2 slots
    assert res.body["default_generation_settings"]["n_ctx"] == 256

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "n_predict": 4,
    })
    assert res.status_code == 200


def test_instances_last_used_advances_on_completion():
    """last_used/last_used_epoch are stamped on slot release: -1 while unused,
    non-decreasing across completions, and untouched by mere listings."""
    global server
    server.instances = ["a:group=swarm:ctx=512"]
    server.n_ctx = 512
    server.start()

    def _last_used():
        inst = _get_instances()["instances"][0]
        return inst["last_used"], inst["last_used_epoch"]

    # unused and unbuilt: both clocks report -1
    assert _last_used() == (-1, -1)

    import time
    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a",
        "prompt": "What is the capital of France?",
        "n_predict": 4,
    })
    assert res.status_code == 200

    first_used, first_epoch = _last_used()
    assert first_used >= 0
    now = int(time.time())
    assert now - 300 <= first_epoch <= now + 60

    # listings never advance either clock
    assert _last_used() == (first_used, first_epoch)

    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a",
        "prompt": "And the capital of Spain?",
        "n_predict": 4,
    })
    assert res.status_code == 200

    second_used, second_epoch = _last_used()
    assert second_used >= first_used
    assert second_epoch >= first_epoch


def test_instances_routing_matrix():
    """Full routing matrix: every advertised alias form resolves, illegal forms
    are rejected, and the explicit `instance` field overrides the model id."""
    global server
    server.instances = [
        "a:group=g:ctx=512",
        "b:group=g:ctx=512",
        "c:group=h:ctx=512:pinned:default",
    ]
    server.n_ctx = 512
    server.start()

    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    ids = {d["id"] for d in res.body["data"]}
    assert ids == {"tinyllama-2:a", "tinyllama-2:b", "tinyllama-2:c"}

    def complete(model=None, instance=None):
        data = {"prompt": "What is the capital of France?", "n_predict": 4}
        if model is not None:
            data["model"] = model
        if instance is not None:
            data["instance"] = instance
        return server.make_request("POST", "/completion", data=data)

    def states():
        return {inst["id"]: inst["state"] for inst in _get_instances()["instances"]}

    # base:name routes to that member only
    assert complete("tinyllama-2:a").status_code == 200
    assert states()["tinyllama-2:a"] == "loaded"
    assert states()["tinyllama-2:b"] == "unloaded"

    # the explicit instance field overrides the model id (b serves, a untouched)
    assert complete("tinyllama-2:a", instance="b").status_code == 200
    assert states()["tinyllama-2:b"] == "loaded"

    # base:group, base:latest:name, base:latest:group, bare base, base:latest
    assert complete("tinyllama-2:g").status_code == 200
    assert complete("tinyllama-2:latest:b").status_code == 200
    assert complete("tinyllama-2:latest:g").status_code == 200
    assert complete("tinyllama-2").status_code == 200
    assert states()["tinyllama-2:c"] == "loaded"
    assert complete("tinyllama-2:latest").status_code == 200

    # a foreign pool name is strictly rejected, never defaulted
    assert complete("other-pool:a").status_code == 400
    # more than three components is never a legal id
    assert complete("tinyllama-2:a:b:c").status_code == 400
    # 'latest' anywhere but the reserved middle slot is rejected
    assert complete("tinyllama-2:a:latest").status_code == 400
    assert complete("tinyllama-2:latest:latest").status_code == 400


def test_instances_nctx_inherit_reports_default():
    """An instance with no `ctx=` reports the model default (non-zero): the
    train size while unbuilt, the real window once built. An explicit size is
    reported unchanged both ways (control group)."""
    global server
    server.instances = ["plain:group=g", "sized:group=g:ctx=256"]
    server.n_ctx = None  # inherit the model default (effective 0)
    server.start()

    rows = {inst["id"]: inst for inst in _get_instances()["instances"]}
    assert rows["tinyllama-2:plain"]["state"] == "unloaded"
    assert rows["tinyllama-2:plain"]["n_ctx"] == 2048
    assert rows["tinyllama-2:sized"]["n_ctx"] == 256

    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:plain",
        "prompt": "What is the capital of France?",
        "n_predict": 4,
    })
    assert res.status_code == 200

    rows = {inst["id"]: inst for inst in _get_instances()["instances"]}
    assert rows["tinyllama-2:plain"]["state"] == "loaded"
    assert rows["tinyllama-2:plain"]["n_ctx"] == 2048
    assert rows["tinyllama-2:sized"]["n_ctx"] == 256


def test_instances_resize_unbuilt_applies_on_demand():
    """Resizing an unbuilt window only records the size; the first demand
    builds at the new size (the retry half of the teardown contract; a failed
    rebuild 507s and stays unbuilt, covered deterministically in C++)."""
    global server
    server.instances = ["a:group=g:ctx=512"]
    server.n_ctx = 512
    server.start()

    res = server.make_request("POST", "/instances/a/resize", data={"ctx_size": 1024})
    assert res.status_code == 200
    rows = {inst["id"]: inst for inst in _get_instances()["instances"]}
    assert rows["tinyllama-2:a"]["state"] == "unloaded"
    assert rows["tinyllama-2:a"]["n_ctx"] == 1024

    res = server.make_request("POST", "/completion", data={
        "model": "tinyllama-2:a",
        "prompt": "What is the capital of France?",
        "n_predict": 4,
    })
    assert res.status_code == 200
    rows = {inst["id"]: inst for inst in _get_instances()["instances"]}
    assert rows["tinyllama-2:a"]["state"] == "loaded"
    assert rows["tinyllama-2:a"]["n_ctx"] == 1024


# --- adapters on demand (ROADMAP_20260809_ADAPTERS_ON_DEMAND) ---
LORA_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"


def _adapter_path() -> str:
    # cached in tests/tmp after the first download (no re-download across tests)
    return os.path.abspath(download_file(LORA_FILE_URL))


def _moe_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.n_ctx = 512
    server.temperature = 0.0
    return server


def _adapters_of(name: str):
    res = server.make_request("GET", f"/instances/{name}/adapters")
    assert res.status_code == 200
    return res.body


def _complete(instance: str, **kw):
    data = {"model": f"stories15m-moe:{instance}", "prompt": "Hello", "n_predict": 4}
    data.update(kw)
    return server.make_request("POST", "/completion", data=data)


def test_instances_adapter_attach_detach():
    """Attach is per-instance, GET lists [{path, scale}], re-attach updates the
    scale (never 409, never a duplicate), detach removes."""
    _moe_server()
    server.instances = ["a:ctx=512", "b:ctx=512"]
    server.start()
    lora = _adapter_path()

    res = server.make_request("POST", "/instances/a/adapters", data={"path": lora, "scale": 0.5})
    assert res.status_code == 200
    assert _adapters_of("a") == [{"path": lora, "scale": 0.5}]

    # re-attach updates the scale in place
    res = server.make_request("POST", "/instances/a/adapters", data={"path": lora, "scale": 1.0})
    assert res.status_code == 200
    assert _adapters_of("a") == [{"path": lora, "scale": 1.0}]

    # make_request DELETE sends no body: use requests directly
    url = f"http://{server.server_host}:{server.server_port}/instances/a/adapters"
    res = requests.delete(url, json={"path": lora})
    assert res.status_code == 200
    assert _adapters_of("a") == []


def test_instances_adapter_shared_single_load():
    """One file attached to two instances loads once: both report adapter_bytes,
    total.adapter counts it once."""
    _moe_server()
    server.instances = ["a:ctx=512", "b:ctx=512"]
    server.start()
    lora = _adapter_path()

    assert _complete("a").status_code == 200
    assert _complete("b").status_code == 200
    res = server.make_request("POST", "/instances/a/adapters", data={"path": lora})
    assert res.status_code == 200
    res = server.make_request("POST", "/instances/b/adapters", data={"path": lora})
    assert res.status_code == 200

    body = _get_instances()
    by_id = {inst["id"]: inst for inst in body["instances"]}
    a_bytes = by_id["stories15m-moe:a"]["adapter_bytes"]
    b_bytes = by_id["stories15m-moe:b"]["adapter_bytes"]
    assert a_bytes > 0
    assert a_bytes == b_bytes
    assert body["total"]["adapter"] == a_bytes, "adapter file loaded more than once"
    assert body["total"]["total"] == (
        body["total"]["model"] + body["total"]["context"] + body["total"]["compute"] + body["total"]["adapter"]
    )


def test_instances_adapter_inherited_base():
    """The base --lora set resolves into the same shared entries on every instance
    (dedup with zero attach calls); attach is additive."""
    _moe_server()
    lora = _adapter_path()
    server.lora_files = [lora]
    server.instances = ["a:ctx=512", "b:ctx=512"]
    server.start()

    assert _complete("a").status_code == 200
    assert _complete("b").status_code == 200
    assert _adapters_of("a") == [{"path": lora, "scale": 1.0}]
    assert _adapters_of("b") == [{"path": lora, "scale": 1.0}]

    body = _get_instances()
    by_id = {inst["id"]: inst for inst in body["instances"]}
    assert by_id["stories15m-moe:a"]["adapter_bytes"] > 0
    assert body["total"]["adapter"] == by_id["stories15m-moe:a"]["adapter_bytes"]

    # a second file (same bytes, other path) attaches additively
    lora2 = lora + ".copy.gguf"
    shutil.copyfile(lora, lora2)
    try:
        res = server.make_request("POST", "/instances/a/adapters", data={"path": lora2})
        assert res.status_code == 200
        got = _adapters_of("a")
        assert [e["path"] for e in got] == [lora, lora2]
        assert _adapters_of("b") == [{"path": lora, "scale": 1.0}]
    finally:
        os.remove(lora2)


def test_instances_adapter_isolated():
    """Instance A's adapter never appears on B."""
    _moe_server()
    server.instances = ["a:ctx=512", "b:ctx=512"]
    server.start()
    lora = _adapter_path()

    res = server.make_request("POST", "/instances/a/adapters", data={"path": lora})
    assert res.status_code == 200
    assert _complete("a").status_code == 200
    assert _complete("b").status_code == 200
    assert _adapters_of("a") == [{"path": lora, "scale": 1.0}]
    assert _adapters_of("b") == []

    body = _get_instances()
    by_id = {inst["id"]: inst for inst in body["instances"]}
    assert by_id["stories15m-moe:a"]["adapter_bytes"] > 0
    assert by_id["stories15m-moe:b"]["adapter_bytes"] == 0


def test_instances_adapter_detach_busy_503_or_waits():
    """Attach during an in-flight generation on the same instance: no corruption,
    no crash (the drain waits or 503s, never a torn swap)."""
    _moe_server()
    server.instances = ["a:ctx=512"]
    server.start()
    lora = _adapter_path()
    assert _complete("a").status_code == 200

    results = {}

    def gen():
        results["gen"] = server.make_request("POST", "/completion", data={
            "model": "stories15m-moe:a",
            "prompt": "The quick brown fox jumps over the lazy dog. ",
            "n_predict": 128,
        })

    def attach():
        results["attach"] = server.make_request("POST", "/instances/a/adapters", data={"path": lora})

    t = threading.Thread(target=gen)
    t.start()
    time.sleep(0.5)  # let the long generation occupy the slot first
    attach()
    t.join()

    assert results["gen"].status_code == 200
    assert results["attach"].status_code in (200, 503)
    # the server is still healthy and the adapter state is coherent
    assert _complete("a", n_predict=2).status_code == 200
    got = _adapters_of("a")
    assert got in ([], [{"path": lora, "scale": 1.0}])

def test_instances_adapter_clears_bindings():
    """A slot bound to a snapshot is unbound by attach (like resize): after the
    swap the request must reach the fingerprint check (400 here) instead of
    short-circuiting on the stale binding (which would serve 200 from live KV)."""
    _moe_server()
    server.instances = ["a:ctx=512"]
    server.slot_save_path = tempfile.mkdtemp()
    server.start()
    lora = _adapter_path()

    res = server.make_request("POST", "/instances/a/adapters", data={"path": lora})
    assert res.status_code == 200
    assert _complete("a", n_predict=8).status_code == 200

    res = server.make_request("POST", "/instances/a/snapshot", data={"name": "work"})
    assert res.status_code == 201
    # bind slot 0 to the snapshot
    assert _complete("a", snapshot="work", n_predict=2).status_code == 200

    # attach a second adapter: revokes the binding and changes the set, so asking
    # for the snapshot must 400 on the fingerprint check rather than serve bound KV
    lora2 = lora + ".copy.gguf"
    shutil.copyfile(lora, lora2)
    try:
        res = server.make_request("POST", "/instances/a/adapters", data={"path": lora2})
        assert res.status_code == 200
        res = server.make_request("POST", "/completion", data={
            "model": "stories15m-moe:a", "snapshot": "work", "prompt": "x", "n_predict": 2,
        })
        assert res.status_code == 400
    finally:
        os.remove(lora2)


def test_instances_adapter_snapshot_mismatch_400():
    """A snapshot saved under adapter A refuses restore after detach; re-attaching
    the original set (fp excludes ptr, so re-resolution matches) restores fine."""
    _moe_server()
    server.instances = ["a:ctx=512"]
    server.slot_save_path = tempfile.mkdtemp()
    server.start()
    lora = _adapter_path()

    res = server.make_request("POST", "/instances/a/adapters", data={"path": lora})
    assert res.status_code == 200
    assert _complete("a", n_predict=8).status_code == 200
    res = server.make_request("POST", "/instances/a/snapshot", data={"name": "work"})
    assert res.status_code == 201

    url = f"http://{server.server_host}:{server.server_port}/instances/a/adapters"
    res = requests.delete(url, json={"path": lora})
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "model": "stories15m-moe:a", "snapshot": "work", "prompt": "x", "n_predict": 2,
    })
    assert res.status_code == 400

    res = server.make_request("POST", "/instances/a/adapters", data={"path": lora})
    assert res.status_code == 200
    res = server.make_request("POST", "/completion", data={
        "model": "stories15m-moe:a", "snapshot": "work", "prompt": "x", "n_predict": 2,
    })
    assert res.status_code == 200


def test_instances_adapter_cold_reload():
    """Delete-last frees adapter bytes (registry drained); recreate + attach works
    and generation still runs."""
    _moe_server()
    server.instances = ["a:ctx=512"]
    server.start()
    lora = _adapter_path()

    res = server.make_request("POST", "/instances/a/adapters", data={"path": lora})
    assert res.status_code == 200
    assert _complete("a").status_code == 200
    assert _get_instances()["total"]["adapter"] > 0

    res = server.make_request("DELETE", "/instances/a")
    assert res.status_code == 200
    body = _get_instances()
    assert body["instances"] == []
    assert body["total"]["adapter"] == 0
    assert body["total"]["model"] == 0

    res = server.make_request("POST", "/instances", data={"name": "b", "ctx_size": 512})
    assert res.status_code == 201
    res = server.make_request("POST", "/instances/b/adapters", data={"path": lora})
    assert res.status_code == 200
    assert _complete("b").status_code == 200
    assert _get_instances()["total"]["adapter"] > 0


def test_instances_adapter_bad_path_400():
    _moe_server()
    server.instances = ["a:ctx=512"]
    server.start()

    # pure body validation rejects regardless of build state
    res = server.make_request("POST", "/instances/a/adapters", data={"path": ""})
    assert res.status_code == 400
    res = server.make_request("POST", "/instances/a/adapters",
                              data={"path": _adapter_path(), "scale": 0})
    assert res.status_code == 400

    # a missing file fails the load: only attempted on a built window (an
    # unbuilt window records the declaration and resolves on first demand)
    assert _complete("a").status_code == 200
    res = server.make_request("POST", "/instances/a/adapters", data={"path": "/nonexistent/x.gguf"})
    assert res.status_code == 400
    assert _adapters_of("a") == []
    assert _get_instances()["total"]["adapter"] == 0


def test_instances_adapter_unknown_404():
    _moe_server()
    server.instances = ["a:ctx=512"]
    server.start()
    lora = _adapter_path()

    res = server.make_request("GET", "/instances/nope/adapters")
    assert res.status_code == 404
    res = server.make_request("POST", "/instances/nope/adapters", data={"path": lora})
    assert res.status_code == 404

    base = f"http://{server.server_host}:{server.server_port}"
    res = requests.delete(base + "/instances/nope/adapters", json={"path": lora})
    assert res.status_code == 404
    # unknown adapter on a known instance is also 404
    res = requests.delete(base + "/instances/a/adapters", json={"path": lora})
    assert res.status_code == 404
    # the ?path= alias works for detach
    res = server.make_request("POST", "/instances/a/adapters", data={"path": lora})
    assert res.status_code == 200
    res = requests.delete(base + "/instances/a/adapters?path=" + urllib.parse.quote(lora, safe=""))
    assert res.status_code == 200
    assert _adapters_of("a") == []
