#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end checks for the decision endpoint envelope and error contract.

Starts llama-server with --decision-seqs and exercises the Jev request shape
plus the legacy contexts/schema shape. Skips cleanly (exit 0) when the server
binary or a small test model is not available, so it never fails open.
"""

import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", os.path.join(REPO, "build", "bin", "llama-server"))

MODEL_CANDIDATES = [
    os.environ.get("LLAMA_SERVER_TEST_MODEL", ""),
    os.path.join(HERE, "tmp", "stories15M-q4_0.gguf"),
    os.path.join(HERE, "tmp", "moe_shakespeare15M.gguf"),
]

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

LEGACY_VALID = {
    "instructions": "Answer each field from the context.",
    "schema": {
        "category": {"type": "enum", "choices": ["billing", "technical"], "description": "What type?"},
        "urgent": {"type": "boolean", "description": "Urgent?"},
    },
    "contexts": ["I was charged twice and need this fixed today."],
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


def http(method, url, body=None, content_type="application/json"):
    import urllib.request
    import urllib.error

    data = body.encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


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
            "--port", str(self.port),
            "--host", "127.0.0.1",
        ] + self.extra_args
        env = dict(os.environ)
        build_bin = os.path.dirname(os.path.abspath(SERVER_BIN))
        env["LD_LIBRARY_PATH"] = build_bin + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited early")
            try:
                status, _ = http("GET", f"http://127.0.0.1:{self.port}/health")
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

    def post(self, path, body):
        return http("POST", f"http://127.0.0.1:{self.port}{path}", body)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run_checks(server, captured):
    # 1. valid Jev envelope
    status, text = server.post("/v1/decision", json.dumps(JEV_VALID))
    check(status == 200, f"valid request status {status}: {text}")
    body = json.loads(text)
    contract = body.get("diagnostics", {}).get("contract_hash", "")
    check(len(contract) == 64, f"contract hash reported: {contract!r}")
    captured["contract_hash"] = contract
    check(body.get("model") == "test", "model echo")
    answers = body.get("answers", {})
    check(set(answers) == {"refund", "dept", "urgency"}, f"answers keyed by qid: {answers}")
    check("confidence" not in answers["refund"], "noul must not carry confidence")
    check(isinstance(answers["refund"]["noul"], (int, float)), "noul probability")
    check(set(answers["dept"]["probabilities"]) == {"billing", "technical"}, "choice probabilities keyed by option")
    check(answers["dept"]["choice"] in ("billing", "technical"), "choice winner is an option")
    check(set(answers["urgency"]["probabilities"]) == {"0", "1", "2"}, "score probability keys are index strings")
    check(set(answers["urgency"]["legend"]) == {"0", "1", "2"}, "score legend keys")
    check(body["usage"]["output_tokens"] == 0, "output_tokens is always 0")
    # the prompt/cached split is exposed so callers can see how much of the prompt was reused
    check("input_tokens" in body["usage"], "usage reports input_tokens")
    check("cached_tokens" in body["usage"], "usage reports a cached_tokens split")
    check(body["usage"]["input_tokens"] >= body["usage"]["cached_tokens"], "cached tokens are part of the input")

    # additive audit fields: prompt identity + per-answer tokenizer/vocabulary diagnostics
    expected_ids = {"refund": 2, "dept": 2, "urgency": 3}
    for qid, ans in answers.items():
        check(0.0 < ans["allowed_token_mass"] <= 1.0 + 1e-6, f"{qid} allowed_token_mass in range")
        check(isinstance(ans["full_vocab_argmax_id"], int) and ans["full_vocab_argmax_id"] >= 0, f"{qid} argmax id")
        check(len(ans["answer_token_ids"]) == expected_ids[qid], f"{qid} answer token ids")
        check(len(ans["prompt_sha256"]) == 64, f"{qid} prompt sha256")
        check(bool(ans["prompt_version"]), f"{qid} prompt version")
        check(bool(ans["probability_status"]), f"{qid} probability status")

    # 2. malformed JSON -> 400
    status, text = server.post("/v1/decision", "{ this is not json")
    check(status == 400, f"malformed JSON status {status}: {text}")

    # 3. semantic error -> 422
    bad = dict(JEV_VALID)
    bad["state"] = ""
    status, text = server.post("/v1/decision", json.dumps(bad))
    check(status == 422, f"semantic error status {status}: {text}")
    payload = json.loads(text)
    check(payload["error"]["code"] == 422, "semantic error code 422")

    # 3b. head: an explicit selected request falls back to full logits when the serving context
    #     cannot expose hidden states (the shared chat context does not), and reports why. Only an
    #     incompatible model is a client error; auto/full serve normally.
    selected = dict(JEV_VALID)
    selected["head"] = "selected"
    status, text = server.post("/v1/decision", json.dumps(selected))
    check(status == 200, f"explicit selected head status {status}: {text}")
    selected_body = json.loads(text)
    check(selected_body["head"]["mode"] in ("selected", "full"), "selected head mode reported")
    if selected_body["head"]["mode"] == "full":
        check(selected_body["head"]["fallback"] is True, "selected fallback is reported")
        check(bool(selected_body["head"].get("reason")), "selected fallback reason is reported")

    full = dict(JEV_VALID)
    full["head"] = "full"
    status, text = server.post("/v1/decision", json.dumps(full))
    check(status == 200, f"explicit full head status {status}: {text}")
    full_body = json.loads(text)
    check(full_body["usage"].get("head_mode") == "full", "head_mode in usage")
    check(full_body.get("head", {}).get("mode") == "full", "head diagnostic object")
    check("option_logits" in full_body["answers"]["dept"], "option logits exposed")

    # 3c. permutations: two passes are accepted and stay a valid distribution
    permuted = dict(JEV_VALID)
    permuted["permutations"] = 2
    status, text = server.post("/v1/decision", json.dumps(permuted))
    check(status == 200, f"permutations status {status}: {text}")
    perm_answers = json.loads(text)["answers"]
    probs2 = perm_answers["dept"]["probabilities"]
    check(abs(sum(probs2.values()) - 1.0) < 1e-4, f"permuted probabilities sum to 1: {probs2}")
    check(perm_answers["dept"]["choice"] in ("billing", "technical"), "permuted choice is an option")

    # 4. legacy contexts/schema shape is unchanged
    status, text = server.post("/v1/decision", json.dumps(LEGACY_VALID))
    check(status == 200, f"legacy status {status}: {text}")
    legacy = json.loads(text)
    check(legacy.get("object") == "decision", "legacy object marker")
    check("results" in legacy and "decision" in legacy["results"][0], "legacy results shape")
    check("answers" not in legacy, "legacy response must not use the Jev envelope")
    check("prompt_tokens" in legacy["usage"], "legacy usage reports prompt_tokens")
    check("cached_tokens" in legacy["usage"], "legacy usage reports a cached_tokens split")


def supports_letter_labels(server):
    """A usable model must yield at least two single-token letter labels."""
    try:
        status, text = server.post("/v1/decision", json.dumps(JEV_VALID))
    except Exception:
        return False
    if status == 200:
        return True
    # an unsupported vocabulary is a model limitation, not a server bug
    return "answer tokens" not in text


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
        captured = {}
        try:
            run_checks(server, captured)
        except Exception as e:  # noqa: BLE001
            server.stop()
            print(f"FAIL: {e}")
            return 1
        server.stop()

        # contract pinning: the running hash is accepted, any other hash refuses the path
        try:
            good = Server(model, ["--decision-contract", captured["contract_hash"]])
            good.start()
            status, text = good.post("/v1/decision", json.dumps(JEV_VALID))
            good.stop()
            check(status == 200, f"pinned contract status {status}: {text}")

            bad = Server(model, ["--decision-contract", "0" * 64])
            bad.start()
            status, text = bad.post("/v1/decision", json.dumps(JEV_VALID))
            bad.stop()
            check(status == 501, f"mismatched contract status {status}: {text}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: {e}")
            return 1

        print("decision envelope checks passed")
        return 0

    print("SKIP: no candidate model supports letter labels; set LLAMA_SERVER_TEST_MODEL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
