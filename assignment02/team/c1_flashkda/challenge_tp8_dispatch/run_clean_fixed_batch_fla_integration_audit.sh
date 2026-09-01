#!/usr/bin/env bash
# Clean one-GPU B300 audit for fixed-batch FLA public-registry integration.
# This script never rebuilds FlashKDA or patches FLA.
set -Eeuo pipefail

if [[ "${C1_FIXED_BATCH_FLA_INTEGRATION_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_FIXED_BATCH_FLA_INTEGRATION_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT to the already-built audited comparison worktree}"
: "${FLA_ROOT:?set FLA_ROOT to the pinned FLA checkout}"
: "${LABEL:?set LABEL}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
EXPECTED_SO_SHA256="8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_FLASH_KDA_PYTHON_SHA256="9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
EXPECTED_FLA_COMMIT="a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
EXPECTED_FLA_INIT_SHA256="b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d"
EXPECTED_FLA_BACKENDS_SHA256="a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635"
EXPECTED_FLA_KDA_INIT_SHA256="24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb"
EXPECTED_FLA_KDA_BACKENDS_SHA256="86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797"
EXPECTED_FLA_FLASH_KDA_SHA256="0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2"
EXPECTED_FLA_CHUNK_SHA256="a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_fixed_batch_fla_integration_${LABEL}_job${SLURM_JOB_ID:-none}.log"
exec > >(tee "$LOG") 2>&1

gpu_query() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader; }
app_query() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; }
memory_query() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
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
SO_PATHS=("$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so)
[[ "${#SO_PATHS[@]}" -eq 1 && -f "${SO_PATHS[0]}" ]] || exit 89
[[ -f "$PATCHED_ROOT/flash_kda/__init__.py" \
    && -f "$FLA_ROOT/fla/__init__.py" \
    && -f "$FLA_ROOT/fla/ops/backends/__init__.py" \
    && -f "$FLA_ROOT/fla/ops/kda/__init__.py" \
    && -f "$FLA_ROOT/fla/ops/kda/backends/__init__.py" \
    && -f "$FLA_ROOT/fla/ops/kda/backends/flash_kda.py" \
    && -f "$FLA_ROOT/fla/ops/kda/chunk.py" ]] || exit 89
[[ "$(git -C "$FLA_ROOT" rev-parse HEAD)" == "$EXPECTED_FLA_COMMIT" ]] || exit 88
[[ "$(sha256sum "${SO_PATHS[0]}" | awk '{print $1}')" == "$EXPECTED_SO_SHA256" ]] || exit 87
[[ "$(sha256sum "$PATCHED_ROOT/flash_kda/__init__.py" | awk '{print $1}')" == "$EXPECTED_FLASH_KDA_PYTHON_SHA256" ]] || exit 81
[[ "$(sha256sum "$FLA_ROOT/fla/__init__.py" | awk '{print $1}')" == "$EXPECTED_FLA_INIT_SHA256" ]] || exit 79
[[ "$(sha256sum "$FLA_ROOT/fla/ops/backends/__init__.py" | awk '{print $1}')" == "$EXPECTED_FLA_BACKENDS_SHA256" ]] || exit 86
[[ "$(sha256sum "$FLA_ROOT/fla/ops/kda/__init__.py" | awk '{print $1}')" == "$EXPECTED_FLA_KDA_INIT_SHA256" ]] || exit 78
[[ "$(sha256sum "$FLA_ROOT/fla/ops/kda/backends/__init__.py" | awk '{print $1}')" == "$EXPECTED_FLA_KDA_BACKENDS_SHA256" ]] || exit 80
[[ "$(sha256sum "$FLA_ROOT/fla/ops/kda/backends/flash_kda.py" | awk '{print $1}')" == "$EXPECTED_FLA_FLASH_KDA_SHA256" ]] || exit 85
[[ "$(sha256sum "$FLA_ROOT/fla/ops/kda/chunk.py" | awk '{print $1}')" == "$EXPECTED_FLA_CHUNK_SHA256" ]] || exit 84
[[ "${FLA_DISABLE_BACKEND_DISPATCH:-0}" != 1 ]] || exit 83
[[ "$(gpu_query | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || exit 82

echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
require_clean PRE || exit 90

echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'FLA_COMMIT=%s\n' "$(git -C "$FLA_ROOT" rev-parse HEAD)"
printf 'PATCHED_STATUS_BEGIN\n'; git -C "$PATCHED_ROOT" status --short; printf 'PATCHED_STATUS_END\n'
printf 'FLA_STATUS_BEGIN\n'; git -C "$FLA_ROOT" status --short; printf 'FLA_STATUS_END\n'
sha256sum \
    "$OWNED/run_fixed_batch_fla_integration.py" \
    "$OWNED/run_clean_fixed_batch_fla_integration_audit.sh" \
    "$OWNED/auto_dispatch.py" \
    "$OWNED/fla_backend.py" \
    "$OWNED/test_auto_dispatch_policy.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_seqcount_dispatch/run_seqcount_dispatch.py" \
    "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" \
    "${SO_PATHS[0]}" \
    "$PATCHED_ROOT/flash_kda/__init__.py" \
    "$PATCHED_ROOT/csrc/flash_kda.cpp" \
    "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu" \
    "$FLA_ROOT/fla/__init__.py" \
    "$FLA_ROOT/fla/ops/backends/__init__.py" \
    "$FLA_ROOT/fla/ops/kda/__init__.py" \
    "$FLA_ROOT/fla/ops/kda/backends/__init__.py" \
    "$FLA_ROOT/fla/ops/kda/backends/flash_kda.py" \
    "$FLA_ROOT/fla/ops/kda/chunk.py"

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT:$FLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export C1_B300_FLASH_KDA=1
export FLA_FLASH_KDA=1
export C1_FIXED_BATCH_FLA_INTEGRATION_CLEAN_GPU=1
cd "$PATCHED_ROOT"
echo "===== CPU_POLICY_TESTS ====="
"$PYTHON_BIN" "$OWNED/test_auto_dispatch_policy.py"
echo "===== PYTHON_COMPILE_GATE ====="
"$PYTHON_BIN" -m py_compile "$OWNED/run_fixed_batch_fla_integration.py"
echo "===== FIXED_BATCH_FLA_INTEGRATION ====="
"$PYTHON_BIN" "$OWNED/run_fixed_batch_fla_integration.py" \
    --json "$RESULTS_DIR/c1_fixed_batch_fla_integration_${LABEL}.json"
require_clean AFTER_FLA_INTEGRATION || exit 93
