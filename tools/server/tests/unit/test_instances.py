import threading
import pytest
from utils import *
import os
import tempfile

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

    # each entry carries the memory breakdown and an always-loaded state
    for inst in body["instances"]:
        assert "model_bytes" in inst
        assert "context_bytes" in inst
        assert "compute_bytes" in inst
        assert "total_bytes" in inst
        assert inst["total_bytes"] == inst["model_bytes"] + inst["context_bytes"] + inst["compute_bytes"]
        assert "vram_bytes" in inst
        assert inst["vram_bytes"] == inst["context_bytes"] + inst["compute_bytes"]
        assert inst["state"] == "loaded"
        # this branch has no auto-sleep: neither field may be reported
        assert "sleep_idle_seconds" not in inst
        assert "no_sleep" not in inst

    # the envelope sums a 64-bit total; the shared model bytes are counted once
    assert "total" in body
    total = body["total"]
    assert set(total.keys()) == {"model", "context", "compute", "total"}
    assert total["total"] == total["model"] + total["context"] + total["compute"]

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
    assert set(total.keys()) == {"model", "context", "compute", "total"}
    assert total["total"] == total["model"] + total["context"] + total["compute"]

    inst = body["instances"][0]
    assert inst["total_bytes"] == inst["model_bytes"] + inst["context_bytes"] + inst["compute_bytes"]
    assert inst["state"] == "loaded"


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
    model_dir = os.path.join(snap_dir, key)
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

    for bad_name in ["a:b", "a=b", "a/b", "a b"]:
        res = server.make_request("POST", "/instances", data={
            "name": bad_name, "group": "swarm", "ctx_size": 512,
        })
        assert res.status_code == 400, f"name {bad_name!r} should be rejected with 400"

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
