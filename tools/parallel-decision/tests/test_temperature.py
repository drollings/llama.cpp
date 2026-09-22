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
BIN = os.path.join(REPO, "build", "bin", "test-decision-engine")
NEEDLES = ("temperature", "confidence", "certainty", "provenance")


def main():
    if not os.path.isfile(BIN):
        print(f"SKIP: {BIN} not built")
        return 0

    proc = subprocess.run([BIN], capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    if proc.returncode != 0 or "FAIL" in output:
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
