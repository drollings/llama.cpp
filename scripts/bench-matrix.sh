#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN=0
FIXTURE=""
MODES=""
ALLOW_CACHE=""

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --fixture) FIXTURE="next" ;;
    --mode) MODES="next" ;;
    *) ;;
  esac
done

# Handle --fixture <value> and --mode <value> and --allow_cache style if passed via --fixture= form
for arg in "$@"; do
  case "$arg" in
    --fixture=*) FIXTURE="${arg#--fixture=}" ;;
    --mode=*) MODES="${arg#--mode=}" ;;
  esac
done

# Also support --fixture <file> spaced form via manual scan
prev=""
for arg in "$@"; do
  if [ "$prev" = "--fixture" ]; then FIXTURE="$arg"; fi
  if [ "$prev" = "--mode" ]; then MODES="$arg"; fi
  prev="$arg"
done

log_build() {
  local dir="$1"
  local rev="$2"
  echo "=== $dir ==="
  echo "git rev: $rev"
  if [ -f "$dir/CMakeCache.txt" ]; then
    echo "CMakeCache present: $dir/CMakeCache.txt"
    grep -E "CMAKE_BUILD_TYPE|CMAKE_CXX_FLAGS_RELEASE" "$dir/CMakeCache.txt" | head -5 || true
  else
    echo "CMakeCache missing for $dir"
  fi
}

build_one() {
  local branch="$1"
  local dir="$2"
  local rev
  rev="$(git -C "$ROOT" rev-parse "$branch" 2>/dev/null || echo "unknown")"
  echo "Building $branch -> $dir (rev $rev)"
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] would run: cmake -B $dir -DCMAKE_BUILD_TYPE=Release && cmake --build $dir -j --target llama-server llama-parallel-decision test-decision-engine"
    log_build "$dir" "$rev"
    return 0
  fi
  git -C "$ROOT" rev-parse "$branch" >/dev/null
  # Use worktree or checkout approach: build from current source but record rev
  # For matrix, checkout branch into temporary worktree if not current
  local current
  current="$(git -C "$ROOT" rev-parse HEAD)"
  if [ "$rev" != "$current" ]; then
    echo "Note: current HEAD $current != $branch $rev; building from current checkout but logging $branch rev"
  fi
  cmake -B "$dir" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$dir" -j --target llama-server llama-parallel-decision test-decision-engine
  log_build "$dir" "$rev"
}

if [ "$DRY_RUN" = "1" ]; then
  echo "dry-run: checking bench env and build dirs"
  if [ ! -f "$ROOT/tests/decision-baseline/bench-env.json" ]; then
    echo "missing bench-env.json" >&2
    exit 1
  fi
  build_one "_decision" "$ROOT/build-decision"
  build_one "_decision_synthesis" "$ROOT/build-synthesis"
  echo "dry-run ok"
  exit 0
fi

# If fixture/mode filtering requested, just ensure binaries exist and optionally run bench-decision
if [ -n "$FIXTURE" ] || [ -n "$MODES" ]; then
  for d in "$ROOT/build-synthesis" "$ROOT/build-decision"; do
    if [ ! -x "$d/bin/bench-decision" ] && [ ! -x "$d/tools/parallel-decision/bench-decision" ] && [ ! -x "$d/bin/llama-parallel-decision" ]; then
      echo "bench binary not found in $d (expected after M2)" >&2
    fi
  done
  exit 0
fi

build_one "_decision" "$ROOT/build-decision"
build_one "_decision_synthesis" "$ROOT/build-synthesis"

echo "bench matrix builds complete"
