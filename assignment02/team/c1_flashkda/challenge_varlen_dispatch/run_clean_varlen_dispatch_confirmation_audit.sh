#!/usr/bin/env bash
# Clean one-GPU B300 execution for the pre-registered packed-varlen gate.
# It never rebuilds FlashKDA.  A pinned Torch-reference helper is prewarmed in
# a dedicated cache before the timed confirmation and its binary SHA is logged.
set -Eeuo pipefail

if [[ "${C1_VARLEN_DISPATCH_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_VARLEN_DISPATCH_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT to the already-built audited comparison worktree}"
: "${REFERENCE_ROOT:?set REFERENCE_ROOT to the pinned Torch-reference worktree}"
: "${LABEL:?set LABEL}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_varlen_dispatch"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
EXPECTED_SO_SHA256="8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_REFERENCE_HELPER_SHA256="8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
EXPECTED_RUNNER_SHA256="9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b"
EXPECTED_PATCHED_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_REFERENCE_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_varlen_dispatch_confirmation_${LABEL}_job${SLURM_JOB_ID:-none}.log"
exec > >(tee "$LOG") 2>&1

gpu_query() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader; }
app_query() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; }
memory_query() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
require_sha256() {
    local expected="$1" path="$2" actual
    [[ -f "$path" ]] || { echo "missing identity-gated file: $path" >&2; return 1; }
    actual="$(sha256sum "$path" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || {
        echo "SHA256 identity gate failed: expected=$expected actual=$actual path=$path" >&2
        return 1
    }
}
require_clean() {
    local stage="$1" apps used
    apps="$(app_query)" || { echo "$stage: compute-app query failed" >&2; return 92; }
    used="$(memory_query)" || { echo "$stage: memory query failed" >&2; return 92; }
    echo "${stage}_COMPUTE_APPS_BEGIN"; printf '%s\n' "$apps"; echo "${stage}_COMPUTE_APPS_END"
    printf '%s_MEMORY_USED_MIB=%s\n' "$stage" "$used"
    [[ -z "$apps" ]] || return 1
    [[ "$(printf '%s\n' "$used" | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || return 1
    while IFS= read -r line; do [[ "$line" =~ ^[[:space:]]*0[[:space:]]*$ ]] || return 1; done <<<"$used"
}
finish() {
    local rc=$?; trap - EXIT
    echo "===== POST_AUDIT ====="; date -Is; gpu_query || rc=92
    require_clean POST || rc=91
    echo "FINAL_RC=$rc"; exit "$rc"
}
trap finish EXIT

echo "===== ENVIRONMENT_GATE ====="
command -v "$PYTHON_BIN"
[[ -f "$OWNED/run_varlen_dispatch_confirmation.py" ]] || { echo "missing owned runner" >&2; exit 86; }
[[ -f "$REFERENCE_ROOT/tests/torch_ref.py" ]] || { echo "missing pinned Torch reference tests/torch_ref.py" >&2; exit 86; }
require_sha256 "$EXPECTED_RUNNER_SHA256" "$OWNED/run_varlen_dispatch_confirmation.py" || exit 87
SO_PATHS=("$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so)
[[ "${#SO_PATHS[@]}" -eq 1 && -f "${SO_PATHS[0]}" ]] || {
    echo "expected exactly one prebuilt flash_kda_C.cpython-*-linux-gnu.so in PATCHED_ROOT" >&2
    exit 89
}
[[ "$(sha256sum "${SO_PATHS[0]}" | awk '{print $1}')" == "$EXPECTED_SO_SHA256" ]] || {
    echo "audited extension SHA256 gate failed" >&2; exit 87
}
[[ "$(git -C "$PATCHED_ROOT" rev-parse HEAD)" == "$EXPECTED_PATCHED_COMMIT" ]] || {
    echo "patched-root commit gate failed" >&2; exit 87
}
[[ "$(git -C "$REFERENCE_ROOT" rev-parse HEAD)" == "$EXPECTED_REFERENCE_COMMIT" ]] || {
    echo "reference-root commit gate failed" >&2; exit 87
}
git -C "$REFERENCE_ROOT" diff --quiet || { echo "reference-root tracked diff gate failed" >&2; exit 87; }
git -C "$REFERENCE_ROOT" diff --cached --quiet || {
    echo "reference-root staged diff gate failed" >&2; exit 87
}
require_sha256 "bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5" \
    "$REFERENCE_ROOT/tests/torch_ref.py" || exit 87
require_sha256 "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84" \
    "$PATCHED_ROOT/flash_kda/__init__.py" || exit 87
require_sha256 "4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f" \
    "$A02_ROOT/team/c1_flashkda/challenge_seqcount_dispatch/run_seqcount_dispatch.py" || exit 87
require_sha256 "ff771c0b2f1b66f3062bc310c14634bf23830f706aec39f1b8ff03ff8b567621" \
    "$A02_ROOT/team/c1_flashkda/challenge_varlen_tail/run_varlen_tail.py" || exit 87
require_sha256 "752126488487ac317a7ee167b660b0895562e4877aedbc4ce2a599c1f59a10d0" \
    "$A02_ROOT/team/c1_flashkda/challenge_prefetch2/prefetch2.py" || exit 87
require_sha256 "445c90815919fb8c1db3bc79289d10029ea77b1d9accb393c326c120ba1f8385" \
    "$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2/vshard4_prefetch2.py" || exit 87
require_sha256 "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52" \
    "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" || exit 87
[[ "$(gpu_query | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || exit 88

echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
require_clean PRE_BEFORE_REFERENCE_HELPER || exit 90

echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'PATCHED_STATUS_BEGIN\n'; git -C "$PATCHED_ROOT" status --short; printf 'PATCHED_STATUS_END\n'
printf 'REFERENCE_COMMIT=%s\n' "$(git -C "$REFERENCE_ROOT" rev-parse HEAD)"
printf 'REFERENCE_TRACKED_STATUS_BEGIN\n'; git -C "$REFERENCE_ROOT" status --short --untracked-files=no; printf 'REFERENCE_TRACKED_STATUS_END\n'
sha256sum \
    "$OWNED/run_varlen_dispatch_confirmation.py" \
    "$OWNED/run_clean_varlen_dispatch_confirmation_audit.sh" \
    "$OWNED/README.zh-CN.md" \
    "$A02_ROOT/team/c1_flashkda/challenge_seqcount_dispatch/run_seqcount_dispatch.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_varlen_tail/run_varlen_tail.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_prefetch2/prefetch2.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2/vshard4_prefetch2.py" \
    "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" \
    "$REFERENCE_ROOT/tests/torch_ref.py" \
    "${SO_PATHS[0]}" \
    "$PATCHED_ROOT/flash_kda/__init__.py" \
    "$PATCHED_ROOT/csrc/flash_kda.cpp" \
    "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu"

echo "===== PYTHON_COMPILE_GATE ====="
"$PYTHON_BIN" -m py_compile "$OWNED/run_varlen_dispatch_confirmation.py"

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset TORCH_EXTENSIONS_DIR
REFERENCE_HELPER_CACHE="$HOME/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so"
require_sha256 "$EXPECTED_REFERENCE_HELPER_SHA256" "$REFERENCE_HELPER_CACHE" || exit 87
export C1_PINNED_REFERENCE_HELPER_PATH="$REFERENCE_HELPER_CACHE"
export C1_PINNED_REFERENCE_HELPER_SHA256="$EXPECTED_REFERENCE_HELPER_SHA256"
echo "===== PINNED_REFERENCE_HELPER_PREWARM ====="
"$PYTHON_BIN" - "$REFERENCE_HELPER_CACHE" "$EXPECTED_REFERENCE_HELPER_SHA256" <<'PY'
import hashlib
import importlib.util
from pathlib import Path
import sys
import torch

helper = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("sigmoid_ext", helper)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load pinned reference helper from {helper}")
module = importlib.util.module_from_spec(spec)
sys.modules["sigmoid_ext"] = module
spec.loader.exec_module(module)
actual = hashlib.sha256(helper.read_bytes()).hexdigest()
print(f"PINNED_REFERENCE_HELPER_PATH={helper}")
print(f"PINNED_REFERENCE_HELPER_SHA256={actual}")
if actual != sys.argv[2]:
    raise RuntimeError(f"pinned reference helper SHA mismatch: expected={sys.argv[2]} actual={actual}")
PY
require_clean PRE_CONFIRMATION || exit 90
export C1_VARLEN_DISPATCH_CLEAN_GPU_GATES=1
cd "$PATCHED_ROOT"
echo "===== VARLEN_DISPATCH_CONFIRMATION ====="
"$PYTHON_BIN" "$OWNED/run_varlen_dispatch_confirmation.py" \
    --reference-root "$REFERENCE_ROOT" \
    --json "$RESULTS_DIR/c1_varlen_dispatch_confirmation_${LABEL}.json"
require_clean AFTER_CONFIRMATION || exit 93
