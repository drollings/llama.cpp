"""Failing-first repros for server-level wire bugs.

Each test asserts the specified contract from LLAMA_INSTANCES.md and is
marked xfail until the server behavior is fixed, at which point the marker
must be removed (strict mode fails the suite on an unexpected pass, so a
fixed bug cannot hide behind a stale marker):

- bare ``<base>`` and ``<base>:latest`` must honor the ``instance`` request
  field (section 5: the field overrides any instance/group component).
- a request-time snapshot switch with no ``--slot-save-path`` must answer
  501 like its three management siblings (section 4.4).
- a malformed adapter path must answer 400, never escape as a 500.
"""
import os
import tempfile
import pytest
import requests
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _complete(model=None, instance=None, **kw):
    data = {"prompt": "What is the capital of France?", "n_predict": 4}
    if model is not None:
        data["model"] = model
    if instance is not None:
        data["instance"] = instance
    data.update(kw)
    return server.make_request("POST", "/completion", data=data)


def test_bare_base_honors_instance_override():
    global server
    server.instances = ["a:group=g:ctx=512", "b:group=g:ctx=512:default", "c:ctx=512"]
    server.n_ctx = 512
    server.start()

    def states():
        return {inst["id"]: inst["state"] for inst in server.make_request("GET", "/instances").body["instances"]}

    # the instance field overrides the (absent) component of a bare base id:
    # a serves, so only a materializes
    assert _complete("tinyllama-2", instance="a").status_code == 200
    assert states()["tinyllama-2:a"] == "loaded"
    assert states()["tinyllama-2:b"] == "unloaded"
    # an explicit group routes by group policy
    assert _complete("tinyllama-2", instance="g").status_code == 200
    # control group: no override keeps the long-standing behavior
    assert _complete("tinyllama-2").status_code == 200
    assert _complete("tinyllama-2:c", instance="c").status_code == 200
    # unknown override still misses, on the generation-routing tier
    assert _complete("tinyllama-2", instance="nope").status_code == 400
    # a foreign pool name is rejected even with an override present
    assert _complete("other-pool", instance="a").status_code == 400


def test_latest_honors_instance_override():
    global server
    server.instances = ["a:ctx=512", "b:ctx=512:default"]
    server.n_ctx = 512
    server.start()
    assert _complete("tinyllama-2:latest", instance="a").status_code == 200
    states = {inst["id"]: inst["state"] for inst in server.make_request("GET", "/instances").body["instances"]}
    assert states["tinyllama-2:a"] == "loaded"
    assert states["tinyllama-2:b"] == "unloaded"
    # control group: latest without an override still hits the default
    assert _complete("tinyllama-2:latest").status_code == 200


def test_snapshot_switch_without_path_is_501():
    global server
    server.instances = ["a:ctx=512"]
    server.n_ctx = 512
    # no slot_save_path: switching must report "not supported"
    server.start()
    res = _complete("tinyllama-2:a", snapshot="foo")
    assert res.status_code == 501


def test_snapshot_switch_status_tiers_unchanged():
    """Control group for the missing-path contract: every neighboring status
    tier keeps its code while the missing-path guard moves."""
    global server
    snap_dir = tempfile.mkdtemp()
    server.instances = ["a:ctx=512"]
    server.n_ctx = 512
    server.slot_save_path = snap_dir
    server.start()

    # forbidden characters in the name are a caller bug
    assert _complete("tinyllama-2:a", snapshot="a/b").status_code == 400
    # an out-of-range slot is a caller bug
    assert _complete("tinyllama-2:a", snapshot="foo", id_slot=999).status_code == 400
    # a well-formed name with no file behind it is missing
    assert _complete("tinyllama-2:a", snapshot="missing-nope").status_code == 404

    # a truncated file is corrupt: plant garbage at the hashed scoped path
    key = "tinyllama-2"
    h = 2166136261
    for c in key.encode():
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    model_dir = os.path.join(snap_dir, "%s-%08x" % (key, h), "a")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "junk.bin"), "wb") as f:
        f.write(b"not a snapshot")
    assert _complete("tinyllama-2:a", snapshot="junk").status_code == 400


def test_malformed_adapter_path_is_400():
    global server
    server.instances = ["a:ctx=512"]
    server.n_ctx = 512
    server.start()

    def total_adapter():
        return server.make_request("GET", "/instances").body["total"]["adapter"]

    assert total_adapter() == 0
    res = server.make_request("POST", "/instances/a/adapters", data={"path": "a\x00b"})
    assert res.status_code == 400
    # nothing reached the registry: no entry, no bytes counted
    assert total_adapter() == 0
    res = server.make_request("GET", "/instances/a/adapters")
    assert res.status_code == 200
    assert res.body == []
    # detach of a malformed path is the same tier, not a 500 (make_request
    # DELETE sends no body, so use requests directly like the adapter suite)
    url = f"http://{server.server_host}:{server.server_port}/instances/a/adapters"
    res = requests.delete(url, json={"path": "a\x00b"})
    assert res.status_code == 400
    assert total_adapter() == 0


def test_adapter_path_single_spelling():
    """Control group: an unbuilt window records adapter declarations without
    loading, so path spelling can be checked with no model files. Relative
    and absolute spellings of one file are one entry; re-attach updates the
    scale in place and never duplicates."""
    global server
    server.instances = ["a:ctx=512"]
    server.n_ctx = 512
    server.start()

    res = server.make_request("POST", "/instances/a/adapters", data={"path": "/tmp/x.gguf"})
    assert res.status_code == 200
    res = server.make_request("GET", "/instances/a/adapters")
    assert res.status_code == 200
    assert res.body == [{"path": "/tmp/x.gguf", "scale": 1.0}]

    res = server.make_request("POST", "/instances/a/adapters", data={"path": "/tmp/x.gguf", "scale": 0.5})
    assert res.status_code == 200
    res = server.make_request("GET", "/instances/a/adapters")
    assert res.body == [{"path": "/tmp/x.gguf", "scale": 0.5}]

    # unbuilt entries hold no refs: the pool counts zero adapter bytes
    assert server.make_request("GET", "/instances").body["total"]["adapter"] == 0


def test_detach_unknown_adapter_on_built_instance_is_404():
    """Control group for the detach ordering: detaching an adapter that was
    never attached answers 404 on a built window too, without stalling
    traffic and without touching the registry."""
    global server
    server.instances = ["a:ctx=512"]
    server.n_ctx = 512
    server.start()

    # build the window first so the detach runs under the drain
    assert _complete("tinyllama-2:a").status_code == 200

    import time
    base = f"http://{server.server_host}:{server.server_port}"
    t0 = time.time()
    res = requests.delete(base + "/instances/a/adapters", json={"path": "/tmp/never-attached.gguf"})
    assert res.status_code == 404
    assert time.time() - t0 < 20
    assert server.make_request("GET", "/instances").body["total"]["adapter"] == 0

    # pin / unpin stays a pure config write with immediate visibility
    res = server.make_request("POST", "/instances/a/pin")
    assert res.status_code == 200
    assert res.body["pinned"] is True
    res = server.make_request("POST", "/instances/a/unpin")
    assert res.status_code == 200
    assert res.body["pinned"] is False
