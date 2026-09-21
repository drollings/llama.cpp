"""Failing-first repros for the multi-instance and adapters-on-demand branch.

Each repro is marked ``xfail(strict=True)`` so it must turn green (and have the
marker removed) when the corresponding behavior is fixed; a stale marker then
fails the suite instead of hiding the bug. A passing control sits beside each
repro so the shared setup is known to be sound before the fix lands.

- instance mode with ``--metrics`` must render the demanded window's metrics
  instead of blocking on the unloaded stock context;
- a legacy ``POST /lora-adapters`` scale must survive a later per-instance
  attach and be reflected by ``GET /instances/:name/adapters``;
- the router's ``GET /instances`` aggregate must stop scheduling children once
  its caller-latency budget is spent.
"""
import json
import os
import signal
import socket
import subprocess
import tempfile
import time

import pytest
import requests
from utils import *

LORA_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"


# --- metrics in instance mode -------------------------------------------------

def _metrics_request(srv: ServerProcess, timeout: float = 10):
    return srv.make_request("GET", "/metrics", timeout=timeout)


def test_metrics_stock_mode_is_non_empty():
    """Control A: stock single-context mode with ``--metrics`` already serves a
    non-empty body; this is the reference shape the instance mode must match."""
    server = ServerPreset.tinyllama2()
    server.server_metrics = True
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": "I believe", "n_predict": 4})
    assert res.status_code == 200

    metrics = _metrics_request(server)
    assert metrics.status_code == 200
    assert isinstance(metrics.body, str)
    assert metrics.body.strip() != ""
    assert "llamacpp:tokens_predicted_total" in metrics.body


def test_metrics_instance_mode_is_non_empty():
    """Repro A: instance mode must render the demanded window's metrics."""
    server = ServerPreset.tinyllama2()
    server.server_metrics = True
    server.instances = ["a:ctx=512"]
    server.start()

    res = server.make_request("POST", "/completion", data={"model": "tinyllama-2:a", "prompt": "I believe", "n_predict": 4})
    assert res.status_code == 200

    metrics = _metrics_request(server)
    assert metrics.status_code == 200
    assert isinstance(metrics.body, str)
    assert metrics.body.strip() != ""
    assert "llamacpp:tokens_predicted_total" in metrics.body


def _metric_value(text: str, name: str):
    for line in text.splitlines():
        if line.startswith(name + " "):
            return float(line.split(" ", 1)[1])
    return None


def test_metrics_targets_named_instance():
    """A named target renders that instance's context-local counters, not the
    default's."""
    server = ServerPreset.tinyllama2()
    server.server_metrics = True
    server.instances = ["a:ctx=512", "b:ctx=512"]
    server.start()

    for name in ("a", "b"):
        assert server.make_request("POST", "/completion", data={"model": f"tinyllama-2:{name}", "prompt": "I believe", "n_predict": 1}).status_code == 200
    # extra work only on a, so its predicted-token counter must be strictly higher
    for _ in range(3):
        assert server.make_request("POST", "/completion", data={"model": "tinyllama-2:a", "prompt": "I believe", "n_predict": 8}).status_code == 200

    a = _metrics_request(server, timeout=10)
    b = server.make_request("GET", "/metrics?instance=b", timeout=10)
    assert a.status_code == 200 and b.status_code == 200
    a_pred = _metric_value(a.body, "llamacpp:tokens_predicted_total")
    b_pred = _metric_value(b.body, "llamacpp:tokens_predicted_total")
    assert a_pred is not None and b_pred is not None
    assert a_pred > b_pred


def test_metrics_unbuilt_is_404():
    """An unbuilt target is 404 and, crucially, the scrape does not build it."""
    server = ServerPreset.tinyllama2()
    server.server_metrics = True
    server.instances = ["a:group=g:ctx=512", "b:group=g:ctx=512"]
    server.start()

    # demand only a: b stays registered but unbuilt
    assert server.make_request("POST", "/completion", data={"model": "tinyllama-2:a", "prompt": "I believe", "n_predict": 1}).status_code == 200

    res = server.make_request("GET", "/metrics?instance=b", timeout=10)
    assert res.status_code == 404
    # a group is not a renderable target
    assert server.make_request("GET", "/metrics?instance=g", timeout=10).status_code == 400
    # an unknown target is the generation-routing tier
    assert server.make_request("GET", "/metrics?instance=nope", timeout=10).status_code == 400
    # the loaded target still renders
    assert server.make_request("GET", "/metrics?instance=a", timeout=10).status_code == 200

    states = {inst["id"]: inst["state"] for inst in server.make_request("GET", "/instances").body["instances"]}
    assert states["tinyllama-2:b"] == "unloaded"



# --- legacy lora scale survival ----------------------------------------------

def _adapter_path() -> str:
    return os.path.abspath(download_file(LORA_FILE_URL))


def _lora_scale(body, path: str):
    for entry in body:
        if entry.get("path") == path:
            return entry.get("scale")
    return None


def _moe_instance_server(instances, lora):
    server = ServerPreset.stories15m_moe()
    server.n_ctx = 512
    server.temperature = 0.0
    server.lora_files = [lora]
    server.instances = instances
    return server


def test_legacy_lora_scale_survives_attach():
    """Repro B: a legacy scale change must stay visible after an attach.

    The legacy writer updates the scheduler only; the manager mirror (read back
    by ``GET /instances/:name/adapters`` and re-applied by the next attach) must
    be refreshed from the scheduler so the attach does not revert the scale.
    """
    lora = _adapter_path()
    server = _moe_instance_server(["a:ctx=512"], lora)
    server.start()

    assert server.make_request("POST", "/completion", data={"model": "stories15m-moe:a", "prompt": "Hello", "n_predict": 4}).status_code == 200

    res = server.make_request("POST", "/lora-adapters", data=[{"id": 0, "scale": 0.5}])
    assert res.status_code == 200
    assert _lora_scale(server.make_request("GET", "/lora-adapters").body, lora) == 0.5
    # the manager mirror must already follow the legacy writer
    assert _lora_scale(server.make_request("GET", "/instances/a/adapters").body, lora) == 0.5

    # attach a second file (same bytes, other path): additive, never reverts
    lora2 = lora + ".copy.gguf"
    import shutil
    shutil.copyfile(lora, lora2)
    try:
        res = server.make_request("POST", "/instances/a/adapters", data={"path": lora2})
        assert res.status_code == 200
        assert _lora_scale(server.make_request("GET", "/lora-adapters").body, lora) == 0.5
        assert _lora_scale(server.make_request("GET", "/instances/a/adapters").body, lora) == 0.5
    finally:
        os.remove(lora2)


# --- router aggregate caller-latency bound -----------------------------------

AGG_PRESET = """\
[agg-a]
hf-repo = ggml-org/test-model-stories260K:F32
instance = w:ctx=512

[agg-b]
hf-repo = ggml-org/test-model-stories260K:F32
instance = w:ctx=512

[agg-c]
hf-repo = ggml-org/test-model-stories260K:F32
instance = w:ctx=512
"""

# small scheduling windows so three children exercise three batches. the
# production values are 8 children per batch and multi-second windows.
AGG_TOTAL_MS = 100
AGG_CHILD_TIMEOUT_MS = 300
AGG_FANOUT_MAX = 1


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _router_with_children(monkeypatch):
    preset_path = os.path.join(tempfile.mkdtemp(), "agg.ini")
    with open(preset_path, "w") as f:
        f.write(AGG_PRESET)
    monkeypatch.setenv("LLAMA_SERVER_TEST_AGG_TOTAL_MS", str(AGG_TOTAL_MS))
    monkeypatch.setenv("LLAMA_SERVER_TEST_AGG_CHILD_TIMEOUT_MS", str(AGG_CHILD_TIMEOUT_MS))
    monkeypatch.setenv("LLAMA_SERVER_TEST_AGG_FANOUT_MAX", str(AGG_FANOUT_MAX))

    srv = ServerPreset.router()
    srv.server_port = _free_port()
    srv.models_preset = preset_path
    srv.models_max = 4
    srv.start()
    for model in ("agg-a", "agg-b", "agg-c"):
        res = srv.make_request("POST", "/models/load", data={"model": model}, timeout=180)
        assert res.status_code == 200, res.body
    return srv


def _wait_for_model(srv, model, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = srv.make_request("GET", "/models")
        for item in res.body.get("data", []):
            status = item.get("status", {})
            value = status.get("value") if isinstance(status, dict) else status
            if item.get("id") == model and value == "loaded":
                return
        time.sleep(0.5)
    raise AssertionError(f"model {model} never loaded")


def _child_pids(router_pid):
    out = subprocess.check_output(["pgrep", "-P", str(router_pid)]).decode().split()
    return [int(x) for x in out]


def _pid_for_alias(pid, alias):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return alias.encode() in f.read().split(b"\0")
    except OSError:
        return False


def _suspend_children(router_pid, aliases):
    suspended = []
    for pid in _child_pids(router_pid):
        for alias in aliases:
            if _pid_for_alias(pid, alias):
                os.kill(pid, signal.SIGSTOP)
                suspended.append(pid)
    return suspended


def test_aggregate_fast_child_present():
    """Control C: a healthy child is always reported even when a sibling is
    frozen past its per-child socket timeout."""
    monkeypatch = pytest.MonkeyPatch()
    try:
        srv = _router_with_children(monkeypatch)
        try:
            for model in ("agg-a", "agg-b", "agg-c"):
                _wait_for_model(srv, model)
            suspended = _suspend_children(srv.process.pid, ["agg-b", "agg-c"])
            try:
                assert len(suspended) == 2, "could not find the two children to suspend"
                env = srv.make_request("GET", "/instances", timeout=60).body
                ids = {inst["id"] for inst in env["instances"]}
                assert any(i.startswith("agg-a:") for i in ids)
            finally:
                for pid in suspended:
                    os.kill(pid, signal.SIGCONT)
        finally:
            srv.stop()
    finally:
        monkeypatch.undo()


def test_aggregate_deadline_bound():
    """Repro C: once the caller-latency budget is spent no further batch may be
    scheduled; the hard worst case is budget plus one per-child socket timeout."""
    monkeypatch = pytest.MonkeyPatch()
    try:
        srv = _router_with_children(monkeypatch)
        try:
            for model in ("agg-a", "agg-b", "agg-c"):
                _wait_for_model(srv, model)
            suspended = _suspend_children(srv.process.pid, ["agg-b", "agg-c"])
            try:
                assert len(suspended) == 2, "could not find the two children to suspend"
                t0 = time.monotonic()
                env = srv.make_request("GET", "/instances", timeout=60).body
                elapsed_ms = (time.monotonic() - t0) * 1000.0

                ids = {inst["id"] for inst in env["instances"]}
                assert any(i.startswith("agg-a:") for i in ids)
                # two batches were skipped by the budget; only one per-child
                # timeout may be paid by an in-flight read
                bound = AGG_TOTAL_MS + AGG_CHILD_TIMEOUT_MS
                assert elapsed_ms <= bound + 150, f"aggregate took {elapsed_ms:.0f} ms, budget {bound} ms"
            finally:
                for pid in suspended:
                    os.kill(pid, signal.SIGCONT)
        finally:
            srv.stop()
    finally:
        monkeypatch.undo()
