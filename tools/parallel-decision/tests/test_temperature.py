#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runs the temperature/confidence subset of the decision engine tests.

Thin driver over the single C++ implementation so there is no second copy of the
temperature logic. Skips cleanly (exit 0) when the test binary is not built.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
NEEDLES = ("temperature", "confidence", "certainty", "provenance")
# Run only the temperature/confidence/certainty/provenance tests, so this driver is not coupled to
# model-family-specific tests (selected heads, LFM2.5 fixtures) that an SPM model cannot serve.
FILTER = r"decision engine harness(\.(.*(temperature|confidence|certainty|provenance).*))?"


def find_bin():
    explicit = os.environ.get("LLAMA_DECISION_TEST_BIN", "")
    if explicit:
        return explicit
    for name in ("build-synthesis", "build"):
        cand = os.path.join(REPO, name, "bin", "test-decision-engine")
        if os.path.isfile(cand):
            return cand
    return os.path.join(REPO, "build", "bin", "test-decision-engine")


BIN = find_bin()


def main():
    if not os.path.isfile(BIN):
        print(f"SKIP: {BIN} not built")
        return 0

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.path.dirname(BIN) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.run([BIN, FILTER], capture_output=True, text=True, errors="replace", env=env)
    output = proc.stdout + proc.stderr
    # the harness exit code and its summary are authoritative; do not match the literal "FAIL",
    # which also appears in the expected-failure and xpass status markers
    failed = proc.returncode != 0
    for line in output.splitlines():
        key, _, value = line.strip().partition(":")
        if key.strip() in ("failures", "xpass"):
            try:
                failed = failed or int(value.strip()) != 0
            except ValueError:
                pass
    if failed:
        print(output)
        print("FAIL: temperature checks failed")
        return 1
    low = output.lower()
    if not all(needle in low for needle in NEEDLES):
        print(output)
        print("FAIL: temperature/confidence checks did not run")
        return 1
    print("temperature checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
