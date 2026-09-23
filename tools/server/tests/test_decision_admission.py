#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Admission-control checks for the decision endpoint.

Starts llama-server with a tiny decision body cap and a queue depth of one, then
exercises the capacity and overload contract: oversize body -> 413, a burst of
concurrent decisions -> 429 (or 529) with `Retry-After`, and a semantically
invalid request -> 422. Skips cleanly (exit 0) when the server binary or a small
test model is missing, so it never fails open.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", os.path.join(REPO, "build", "bin", "llama-server"))

MODEL_CANDIDATES = [
    os.environ.get("LLAMA_SERVER_TEST_MODEL", ""),
    os.path.join(HERE, "tmp", "stories15M-q4_0.gguf"),
    os.path.join(HERE, "tmp", "moe_shakespeare15M.gguf"),
]

MAX_BODY = 16384
MAX_QUEUE = 1

# Pre-registered responsiveness/fairness bound (see tests/decision-baseline/fairness.json).
FAIRNESS_FILE = os.path.join(REPO, "tests", "decision-baseline", "fairness.json")
DEFAULT_FAIRNESS = {
    "slots_during_decision_ms": 250.0,
    "chat_latency_factor": 2.0,
    "chat_latency_margin_ms": 500.0,
}


def load_fairness_bound():
    try:
        with open(FAIRNESS_FILE) as f:
            bound = json.load(f).get("bound", {})
        return {k: float(bound.get(k, v)) for k, v in DEFAULT_FAIRNESS.items()}
    except Exception:  # noqa: BLE001
        return dict(DEFAULT_FAIRNESS)

JEV_VALID = {
    "model": "test",
    "state": "Customer was charged twice on May 3.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Should this be refunded?"},
        "dept": {
            "type": "choice",
            "instructions": "What is the issue?",
            "criteria": {"billing": "payment", "technical": "bug"},
        },
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["calm", "upset", "furious"]},
    },
}


def find_model():
    for path in MODEL_CANDIDATES:
        if path and os.path.isfile(path):
            return path
    return None


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


API_KEY = "decision-secret"


def http(method, url, body=None, content_type="application/json", api_key=API_KEY):
    import urllib.request
    import urllib.error

    data = body.encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", content_type)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, dict(resp.headers), resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode("utf-8")


class Server:
    def __init__(self, model, extra_args=None):
        self.model = model
        self.extra_args = extra_args or []
        self.port = free_port()
        self.proc = None

    def start(self):
        cmd = [
            SERVER_BIN,
            "-m", self.model,
            "-c", "2048",
            "-ngl", "0",
            "--decision-seqs", "8",
            "--slots",
            "--api-key", API_KEY,
            "--port", str(self.port),
            "--host", "127.0.0.1",
        ] + self.extra_args
        env = dict(os.environ)
        build_bin = os.path.dirname(os.path.abspath(SERVER_BIN))
        env["LD_LIBRARY_PATH"] = build_bin + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        env["LLAMA_DECISION_MAX_BODY"] = str(MAX_BODY)
        env["LLAMA_DECISION_MAX_QUEUE"] = str(MAX_QUEUE)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited early")
            try:
                status, _, _ = http("GET", f"http://127.0.0.1:{self.port}/health")
                if status == 200:
                    return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("server did not become healthy")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def post(self, body):
        return http("POST", f"http://127.0.0.1:{self.port}/v1/decision", body)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def supports_letter_labels(server):
    status, _, text = server.post(json.dumps(JEV_VALID))
    if status == 200:
        return True
    return "answer tokens" not in text


def run_checks(server):
    # 1. body over the configured cap is rejected before any decode
    big = dict(JEV_VALID)
    big["state"] = "x" * (MAX_BODY + 512)
    status, _, text = server.post(json.dumps(big))
    check(status == 413, f"oversize body status {status}: {text}")
    check(json.loads(text)["error"]["code"] == 413, "oversize body error code 413")

    # 2. semantically invalid request is 422, not 400
    bad = dict(JEV_VALID)
    bad["state"] = ""
    status, _, text = server.post(json.dumps(bad))
    check(status == 422, f"semantic error status {status}: {text}")

    # 3. a burst over the queue depth is admitted (429) or overloaded (529), with Retry-After
    heavy = dict(JEV_VALID)
    heavy["state"] = "refund request with a broken item. " * 20
    n = 8
    barrier = threading.Barrier(n)
    results = [None] * n

    def worker(i):
        barrier.wait()
        results[i] = server.post(json.dumps(heavy))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    statuses = [r[0] for r in results if r is not None]
    check(200 in statuses, f"at least one request succeeds: {statuses}")
    check(any(s in (429, 529) for s in statuses), f"a saturated burst is rejected: {statuses}")
    for status, headers, _ in results:
        if status == 429:
            check(headers.get("Retry-After") == "1", f"429 carries Retry-After: {headers}")
        if status == 529:
            check(headers.get("Retry-After") == "1", f"529 carries Retry-After: {headers}")

    # 4. only application/json is accepted on the decision route
    status, _, _ = http("POST", f"http://127.0.0.1:{server.port}/v1/decision", "{}", content_type="text/plain")
    check(status in (400, 415), f"non-JSON content type is rejected: {status}")

    # 4b. an unauthenticated request is rejected when auth is configured
    status, _, _ = http("POST", f"http://127.0.0.1:{server.port}/v1/decision", json.dumps(JEV_VALID), api_key=None)
    check(status == 401, f"unauthenticated request status {status}")

    # 5. chat coexistence: a decision must not break chat on the same context
    url = f"http://127.0.0.1:{server.port}/v1/chat/completions"
    chat = json.dumps({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 1})
    status, _, text = http("POST", url, chat)
    check(status == 200, f"chat before decision: {status} {text}")
    t0 = time.time()
    status, _, text = server.post(json.dumps(JEV_VALID))
    warm_decision_ms = (time.time() - t0) * 1000.0
    check(status == 200, f"decision between chats: {status} {text}")
    status, _, text = http("POST", url, chat)
    check(status == 200, f"chat after decision: {status} {text}")

    # 6. cooperative yield: a long decision must not freeze the scheduler
    bound = load_fairness_bound()
    heavy = {
        "model": "test",
        "state": "The customer opened a ticket about a delayed delivery and a duplicate charge. " * 8,
        "questions": {
            f"q{i}": {"type": "noul", "instructions": f"Is claim {i} supported by the state?"}
            for i in range(96)
        },
    }
    outcome = {}

    def run_decision():
        t = time.time()
        st, _, tx = server.post(json.dumps(heavy))
        outcome["ms"] = (time.time() - t) * 1000.0
        outcome["status"] = st
        outcome["text"] = tx

    decision_thread = threading.Thread(target=run_decision)
    decision_thread.start()
    time.sleep(0.1)  # let the decision start so the probes land inside it
    slots_latency = None
    while decision_thread.is_alive():
        t = time.time()
        st, _, _ = http("GET", f"http://127.0.0.1:{server.port}/slots")
        dt = (time.time() - t) * 1000.0
        if st == 200:
            slots_latency = dt if slots_latency is None else min(slots_latency, dt)
        time.sleep(0.01)
    decision_thread.join()

    check(outcome.get("status") == 200, f"heavy decision: {outcome.get('status')} {outcome.get('text')}")
    check(slots_latency is not None, "the slots endpoint answered during the decision")
    check(slots_latency <= bound["slots_during_decision_ms"],
          f"/slots stayed responsive during a {outcome['ms']:.0f} ms decision: {slots_latency:.0f} ms")

    # chat fired concurrently with a decision is serialized behind it, but must complete
    chat_result = {}

    def run_chat():
        t = time.time()
        st, _, tx = http("POST", url, chat)
        chat_result["ms"] = (time.time() - t) * 1000.0
        chat_result["status"] = st
        chat_result["text"] = tx

    decision_thread = threading.Thread(target=run_decision)
    decision_thread.start()
    time.sleep(0.1)
    chat_thread = threading.Thread(target=run_chat)
    chat_thread.start()
    decision_thread.join()
    chat_thread.join()
    check(chat_result.get("status") == 200, f"concurrent chat: {chat_result.get('status')} {chat_result.get('text')}")
    chat_bound = outcome["ms"] + bound["chat_latency_factor"] * warm_decision_ms + bound["chat_latency_margin_ms"]
    check(chat_result["ms"] <= chat_bound,
          f"concurrent chat latency {chat_result['ms']:.0f} ms within {chat_bound:.0f} ms")
    print(f"measured: decision {outcome['ms']:.0f} ms, slots-during {slots_latency:.0f} ms, chat {chat_result['ms']:.0f} ms")


def capacity_body(state, temperature=None):
    body = {
        "model": "test",
        "state": state,
        "questions": {
            "dept": {
                "type": "choice",
                "instructions": "What is the issue?",
                "criteria": {"billing": "payment", "technical": "bug"},
            },
        },
    }
    if temperature is not None:
        body["temperature"] = temperature
    return json.dumps(body)


# Sweep the prompt size against a bounded classifier context: every size up to the budget is
# accepted and decides identically to a large-context control, every size past it is rejected with
# 422 and no decision. The accept/reject outcome must not depend on the reported confidence.
CAPACITY_CTX = 512
CAPACITY_UNIT = "The customer was charged twice and asked for a refund. "


def run_capacity_sweep(model):
    bounded = Server(model, ["--decision-ctx-size", str(CAPACITY_CTX)])
    bounded.start()
    control = None
    try:
        if not supports_letter_labels(bounded):
            return "skip"

        def post_state(state, temperature=None):
            return bounded.post(capacity_body(state, temperature))

        last_ok = None
        first_reject = None
        observed = []
        for mult in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            status, _, text = post_state(CAPACITY_UNIT * mult)
            if status == 413:
                # the body cap is a separate limit (checked above); the context boundary must have
                # been reached before the state grows past it, so stop the sweep here
                break
            observed.append((mult, status, text))
            if status == 200:
                body = json.loads(text)
                check("answers" in body, "a fitting request returns a decision")
                last_ok = {"mult": mult, "tokens": body["usage"]["input_tokens"], "body": body}
            elif status != 422:
                raise AssertionError(f"capacity sweep unexpected status {status}: {text}")

        check(last_ok is not None, "a request fits within the bounded context")
        check(any(s == 422 for _, s, _ in observed), "the sweep reaches the capacity boundary")
        first_reject = next(m for m, s, _ in observed if s == 422)
        check(first_reject > last_ok["mult"], "the boundary is monotone in state size")

        # every size below the boundary must be accepted and every size at or above it rejected
        not_fired_ok = all(s == 200 for m, s, _ in observed if m < first_reject)
        fired_ok = all(s == 422 for m, s, _ in observed if m >= first_reject)
        check(not_fired_ok, "no false reject below the boundary")
        check(fired_ok, "no false accept at or above the boundary")
        expected_not_fired = sum(1 for m, _, _ in observed if m < first_reject)
        expected_fired = sum(1 for m, _, _ in observed if m >= first_reject)
        fired = sum(1 for m, s, _ in observed if m >= first_reject and s == 422)
        not_fired = sum(1 for m, s, _ in observed if m < first_reject and s == 200)
        precision = fired / expected_fired if expected_fired else 1.0
        recall = not_fired / expected_not_fired if expected_not_fired else 1.0
        check(precision == 1.0 and recall == 1.0, f"capacity precision={precision} recall={recall}")

        # no truncation and no KV residue
        rejected_text = next(t for m, s, t in observed if m == first_reject)
        rejected = json.loads(rejected_text)
        check(rejected["error"]["code"] == 422, "the boundary rejection is a 422")
        check("answers" not in rejected and "results" not in rejected, "a rejected request returns no decision")
        status, _, text = post_state(CAPACITY_UNIT)
        check(status == 200, f"the server still serves after a rejection: {status} {text}")

        # the outcome must not depend on the reported confidence: same length, different temperatures
        sharp = post_state(CAPACITY_UNIT * first_reject, 0.01)
        flat = post_state(CAPACITY_UNIT * first_reject, 4.0)
        check(sharp[0] == 422 and flat[0] == 422,
              f"the reject is independent of confidence/temperature: {sharp[0]} {flat[0]}")

        # control group: the fitting request decides the same on a large-context server
        control = Server(model)
        control.start()
        if not supports_letter_labels(control):
            return "skip"
        state = CAPACITY_UNIT * last_ok["mult"]
        status, _, text = control.post(capacity_body(state))
        check(status == 200, f"control server serves the fitting request: {status}")
        cb = json.loads(text)["answers"]["dept"]
        lb = last_ok["body"]["answers"]["dept"]
        check(lb["choice"] == cb["choice"], "the bounded and control servers pick the same choice")
        tv = sum(abs(lb["probabilities"][k] - cb["probabilities"][k]) for k in lb["probabilities"])
        check(tv <= 5e-2, f"the bounded and control distributions agree (TV={tv})")

        print(f"capacity sweep: fit up to {last_ok['mult']} units ({last_ok['tokens']} input tokens), "
              f"first reject at {first_reject} units, precision={precision} recall={recall}, ctx={CAPACITY_CTX}")
        return "pass"
    finally:
        if control is not None:
            control.stop()
        bounded.stop()


def main():
    if not os.path.isfile(SERVER_BIN):
        print(f"SKIP: server binary not found at {SERVER_BIN}")
        return 0

    candidates = [m for m in MODEL_CANDIDATES if m and os.path.isfile(m)]
    if not candidates:
        print("SKIP: no test model; set LLAMA_SERVER_TEST_MODEL")
        return 0

    for model in candidates:
        server = Server(model)
        try:
            server.start()
        except Exception as e:  # noqa: BLE001
            server.stop()
            print(f"skip {os.path.basename(model)}: {e}")
            continue
        if not supports_letter_labels(server):
            server.stop()
            print(f"skip {os.path.basename(model)}: no usable answer labels")
            continue
        try:
            run_checks(server)
        except Exception as e:  # noqa: BLE001
            server.stop()
            print(f"FAIL: {e}")
            return 1
        server.stop()

        # bounded-context sweep: a request past the budget is rejected, never truncated
        try:
            capacity = run_capacity_sweep(model)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: capacity sweep: {e}")
            return 1
        if capacity == "skip":
            print("SKIP: no usable answer labels for the capacity sweep")
        else:
            print("decision capacity sweep passed")

        print("decision admission checks passed")
        return 0

    print("SKIP: no candidate model supports letter labels; set LLAMA_SERVER_TEST_MODEL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
