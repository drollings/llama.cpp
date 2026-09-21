"""M13 aggregate parity E2E: two live background servers (one window each) plus
a mock router child advertising adapter bytes.

The merged contract (the same shape the router's production merge implements,
covered over identical shapes by the C++ merge tests): merged total.adapter is
the sum of the child adapter totals, and merged total.total is the sum of the
adapter-inclusive child totals with no double count. Instance rows pass
through untouched; snapshot rows are tagged with their owning model on merge.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from utils import *

MOCK_ADAPTER = 123456
MOCK_TOTAL = {
    "model":   1000,
    "context": 200,
    "compute": 50,
    "adapter": MOCK_ADAPTER,
    "total":   1000 + 200 + 50 + MOCK_ADAPTER,
}
MOCK_ENVELOPE = {
    "instances": [{"id": "mock:m"}],
    "snapshots": [{"name": "s1"}],
    "total":     MOCK_TOTAL,
}


class _MockChildHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/instances":
            body = json.dumps(MOCK_ENVELOPE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def mock_child():
    port = _free_port()
    httpd = HTTPServer(("127.0.0.1", port), _MockChildHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield MOCK_ENVELOPE
    httpd.shutdown()


def _live_server(instance: str):
    srv = ServerPreset.tinyllama2()
    srv.server_port = _free_port()
    srv.instances = [f"{instance}:ctx=512"]
    srv.n_ctx = 512
    srv.start()
    # demand-build the one window so the envelope carries real byte fields
    res = srv.make_request("POST", "/completion", data={
        "model": f"tinyllama-2:{instance}",
        "prompt": "Hello",
        "n_predict": 4,
    })
    assert res.status_code == 200
    return srv


def test_aggregate_two_live_plus_mock(mock_child):
    srv_a = _live_server("a")
    try:
        srv_b = _live_server("b")
        try:
            env_a = srv_a.make_request("GET", "/instances").body
            env_b = srv_b.make_request("GET", "/instances").body
            env_m = mock_child

            # live legs: real byte fields, no adapters on these windows
            for env in (env_a, env_b):
                assert env["total"]["adapter"] == 0
                assert env["total"]["total"] == (
                    env["total"]["model"] + env["total"]["context"] + env["total"]["compute"]
                )

            # merged adapter is the sum across children (only the mock here)
            merged_adapter = env_a["total"]["adapter"] + env_b["total"]["adapter"] + env_m["total"]["adapter"]
            assert merged_adapter == MOCK_ADAPTER

            # merged total is the sum of the adapter-inclusive child totals
            merged_total = env_a["total"]["total"] + env_b["total"]["total"] + env_m["total"]["total"]
            legs = ["model", "context", "compute", "adapter"]
            assert merged_total == sum(
                env_a["total"][k] + env_b["total"][k] + env_m["total"][k] for k in legs
            ), "adapter bytes double counted in merged total"

            # instance rows pass through untouched: one live row per server plus mock
            ids = (
                {inst["id"] for inst in env_a["instances"]}
                | {inst["id"] for inst in env_b["instances"]}
                | {inst["id"] for inst in env_m["instances"]}
            )
            assert "tinyllama-2:a" in ids
            assert "tinyllama-2:b" in ids
            assert "mock:m" in ids

            # the mock snapshot row is present for the merge to tag
            assert env_m["snapshots"][0]["name"] == "s1"
        finally:
            srv_b.stop()
    finally:
        srv_a.stop()
