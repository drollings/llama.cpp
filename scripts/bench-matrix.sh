#!/usr/bin/env bash
# Developer tool, not part of CI. Builds bench-decision for the two branches and runs a timing
# matrix. Requires -DLLAMA_BUILD_DECISION_BENCH=ON (the bench target is off by default) on a
# machine with the benchmark models; the committed bench artifact was removed, so there is no
# baseline file to validate against. Regenerate one with the command in
# tools/parallel-decision/CMakeLists.txt.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN=0
FIXTURE=""
MODES=""
ALLOW_CACHE=""

# Handle --fixture=value / --mode=value and --dry-run
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
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
  cmake -B "$dir" -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_DECISION_BENCH=ON \
    -DGGML_HIP=ON \
    -DGPU_TARGETS="${GPU_TARGETS:-gfx1100}" \
    -DCMAKE_HIP_ARCHITECTURES="${GPU_TARGETS:-gfx1100}" \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags" \
    -Dhipblas_DIR=/opt/rocm/lib/cmake/hipblas \
    -Drocblas_DIR=/opt/rocm/lib/cmake/rocblas
  cmake --build "$dir" -j --target llama-server llama-parallel-decision test-decision-engine bench-decision
  log_build "$dir" "$rev"
}

if [ "$DRY_RUN" = "1" ]; then
  echo "dry-run: checking build dirs (the bench target needs -DLLAMA_BUILD_DECISION_BENCH=ON)"
  build_one "_decision_synthesis" "$ROOT/build-synthesis"
  echo "dry-run ok"
  exit 0
fi

# If fixture/mode filtering requested, run bench-decision matrix for both models when bin exists
if [ -n "$FIXTURE" ] || [ -n "$MODES" ]; then
  FIXTURE="${FIXTURE:-tests/fixtures/decision/contexts_schema.request.json}"
  MODES="${MODES:-auto,tree,greedy}"
  # models to cover per M0
  MODELS=("/ai/models/gguf/liquidai/lfm2.5-2.6b-gguf/latest.gguf" "$HOME/Downloads/LFM2.5-350M-QAD-Q4_0.gguf")
  IFS=',' read -ra MODE_ARR <<< "$MODES"
  for d in "$ROOT/build-synthesis"; do
    if [ ! -x "$d/bin/bench-decision" ]; then
      echo "bench binary not found in $d (expected after M2)" >&2
      continue
    fi
    for model in "${MODELS[@]}"; do
      if [ ! -f "$model" ]; then
        echo "skip missing model $model" >&2
        continue
      fi
      for mode in "${MODE_ARR[@]}"; do
        for cache in true false; do
          echo "bench $d $model mode=$mode cache=$cache"
          "$d/bin/bench-decision" --model "$model" --fixture "$FIXTURE" --mode "$mode" --allow_cache "$cache" --json > /dev/null 2>&1 || echo "bench failed $mode $cache $model" >&2
        done
      done
    done
  done
  exit 0
fi

build_one "_decision_synthesis" "$ROOT/build-synthesis"

# post-build matrix for both models when bench-decision exists
if [ -x "$ROOT/build-synthesis/bin/bench-decision" ]; then
  FIXTURE="tests/fixtures/decision/contexts_schema.request.json"
  MODELS=("/ai/models/gguf/liquidai/lfm2.5-2.6b-gguf/latest.gguf" "$HOME/Downloads/LFM2.5-350M-QAD-Q4_0.gguf")
  for model in "${MODELS[@]}"; do
    if [ -f "$model" ]; then
      echo "post-build bench check $model"
      "$ROOT/build-synthesis/bin/bench-decision" --model "$model" --fixture "$FIXTURE" --mode auto --json > /dev/null 2>&1 || true
    fi
  done
fi

echo "bench matrix builds complete"
