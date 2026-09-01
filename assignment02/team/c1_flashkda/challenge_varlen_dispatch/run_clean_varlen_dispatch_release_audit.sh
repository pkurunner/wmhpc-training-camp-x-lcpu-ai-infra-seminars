#!/usr/bin/env bash
# Clean one-GPU B300 execution for the packed-varlen per-cell release gate.
# It uses only prebuilt artifacts and pinned evidence; no compile is allowed.
set -Eeuo pipefail

if [[ "${C1_VARLEN_DISPATCH_RELEASE_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_VARLEN_DISPATCH_RELEASE_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT to the already-built audited comparison worktree}"
: "${REFERENCE_ROOT:?set REFERENCE_ROOT to the pinned Torch-reference worktree}"
: "${LABEL:?set LABEL}"
: "${C1_PINNED_REFERENCE_HELPER_PATH:?set C1_PINNED_REFERENCE_HELPER_PATH to the prebuilt helper module}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_varlen_dispatch"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
SEQCOUNT_DISCOVERY_JSON="${SEQCOUNT_DISCOVERY_JSON:-$A02_ROOT/team/c1_flashkda/challenge_seqcount_dispatch/results/c1_seqcount_dispatch_b300_sm103a_r2.json}"
MIXED_DISCOVERY_JSON="${MIXED_DISCOVERY_JSON:-$A02_ROOT/team/c1_flashkda/challenge_varlen_tail/results/c1_varlen_tail_b300_sm103a_r1.json}"
CONFIRMATION_JSON="${CONFIRMATION_JSON:-$OWNED/results/c1_varlen_dispatch_confirmation_b300_sm103a_r4.json}"
EXPECTED_SO_SHA256="8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_SEQCOUNT_SHA256="46cd27f2fbdcaeeb61011c49c6175a0c05d15d4365bfda800cf52040dbe414f7"
EXPECTED_MIXED_SHA256="b2dae8d42f43c3e42c44ca20fdc2c8443ec8b6b1b1ff2b81aff74be5b877fcd3"
EXPECTED_CONFIRMATION_SHA256="447d7f49a624fa5b92adc431b350450f99d53f5b20f3a07a1bf4d2f76a64e51c"
EXPECTED_CONFIRMATION_RUNNER_SHA256="9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b"
EXPECTED_RELEASE_RUNNER_SHA256="1e8ff86e79683dd3b1266abe2013e7cec8c95b6b099d4c315ecb79419b2d2a42"
EXPECTED_PINNED_REFERENCE_HELPER_SHA256="8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
EXPECTED_TORCH_REF_SHA256="bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5"
EXPECTED_PATCHED_FLASH_KDA_INIT_SHA256="9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
EXPECTED_SHARED_SEQCOUNT_RUNNER_SHA256="4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f"
EXPECTED_VARLEN_TAIL_RUNNER_SHA256="ff771c0b2f1b66f3062bc310c14634bf23830f706aec39f1b8ff03ff8b567621"
EXPECTED_PREFETCH2_SHA256="752126488487ac317a7ee167b660b0895562e4877aedbc4ce2a599c1f59a10d0"
EXPECTED_VSHARD4_PREFETCH2_SHA256="445c90815919fb8c1db3bc79289d10029ea77b1d9accb393c326c120ba1f8385"
EXPECTED_HARNESS_SHA256="5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52"
EXPECTED_PATCHED_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_varlen_dispatch_release_${LABEL}_job${SLURM_JOB_ID:-none}.log"
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

require_sha() {
    local path="$1" expected="$2" label="$3"
    [[ -f "$path" ]] || { echo "missing $label: $path" >&2; return 86; }
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
        echo "$label SHA256 gate failed: $path" >&2; return 87
    }
}
require_commit() {
    local root="$1" label="$2"
    [[ "$(git -C "$root" rev-parse HEAD)" == "$EXPECTED_PATCHED_COMMIT" ]] || {
        echo "$label commit gate failed" >&2; return 85
    }
}
require_reference_tracked_clean() {
    [[ -z "$(git -C "$REFERENCE_ROOT" status --short --untracked-files=no)" ]] || {
        echo "reference worktree has tracked or staged changes" >&2; return 84
    }
}

echo "===== ENVIRONMENT_GATE ====="
command -v "$PYTHON_BIN"
[[ -f "$REFERENCE_ROOT/tests/torch_ref.py" ]] || { echo "missing pinned Torch reference" >&2; exit 86; }
require_commit "$PATCHED_ROOT" "patched worktree"
require_commit "$REFERENCE_ROOT" "reference worktree"
require_reference_tracked_clean
[[ "${C1_PINNED_REFERENCE_HELPER_SHA256:-}" == "$EXPECTED_PINNED_REFERENCE_HELPER_SHA256" ]] || {
    echo "pinned reference helper SHA environment drift" >&2; exit 87
}
export C1_PINNED_REFERENCE_HELPER_SHA256="$EXPECTED_PINNED_REFERENCE_HELPER_SHA256"
require_sha "$SEQCOUNT_DISCOVERY_JSON" "$EXPECTED_SEQCOUNT_SHA256" "seqcount discovery"
require_sha "$MIXED_DISCOVERY_JSON" "$EXPECTED_MIXED_SHA256" "mixed discovery"
require_sha "$CONFIRMATION_JSON" "$EXPECTED_CONFIRMATION_SHA256" "confirmation"
require_sha "$OWNED/run_varlen_dispatch_release.py" "$EXPECTED_RELEASE_RUNNER_SHA256" "release runner"
require_sha "$OWNED/run_varlen_dispatch_confirmation.py" "$EXPECTED_CONFIRMATION_RUNNER_SHA256" "confirmation runner"
require_sha "$C1_PINNED_REFERENCE_HELPER_PATH" "$EXPECTED_PINNED_REFERENCE_HELPER_SHA256" "pinned reference helper"
require_sha "$REFERENCE_ROOT/tests/torch_ref.py" "$EXPECTED_TORCH_REF_SHA256" "pinned Torch reference"
require_sha "$PATCHED_ROOT/flash_kda/__init__.py" "$EXPECTED_PATCHED_FLASH_KDA_INIT_SHA256" "patched flash_kda init"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_seqcount_dispatch/run_seqcount_dispatch.py" "$EXPECTED_SHARED_SEQCOUNT_RUNNER_SHA256" "shared seqcount runner"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_varlen_tail/run_varlen_tail.py" "$EXPECTED_VARLEN_TAIL_RUNNER_SHA256" "varlen tail runner"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_prefetch2/prefetch2.py" "$EXPECTED_PREFETCH2_SHA256" "prefetch2 wrapper"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2/vshard4_prefetch2.py" "$EXPECTED_VSHARD4_PREFETCH2_SHA256" "vshard4 wrapper"
require_sha "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" "$EXPECTED_HARNESS_SHA256" "harness"
SO_PATHS=("$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so)
[[ "${#SO_PATHS[@]}" -eq 1 && -f "${SO_PATHS[0]}" ]] || { echo "expected exactly one audited extension SO" >&2; exit 89; }
require_sha "${SO_PATHS[0]}" "$EXPECTED_SO_SHA256" "audited extension"
[[ "$(gpu_query | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || exit 88

echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
require_clean PRE || exit 90

echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'PATCHED_STATUS_BEGIN\n'; git -C "$PATCHED_ROOT" status --short --untracked-files=no; printf 'PATCHED_STATUS_END\n'
printf 'REFERENCE_COMMIT=%s\n' "$(git -C "$REFERENCE_ROOT" rev-parse HEAD)"
printf 'REFERENCE_STATUS_BEGIN\n'; git -C "$REFERENCE_ROOT" status --short --untracked-files=no; printf 'REFERENCE_STATUS_END\n'
sha256sum \
    "$OWNED/run_varlen_dispatch_release.py" \
    "$OWNED/run_clean_varlen_dispatch_release_audit.sh" \
    "$OWNED/run_varlen_dispatch_confirmation.py" \
    "$OWNED/analyze_varlen_confirmation.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_seqcount_dispatch/run_seqcount_dispatch.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_varlen_tail/run_varlen_tail.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_prefetch2/prefetch2.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2/vshard4_prefetch2.py" \
    "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" \
    "$REFERENCE_ROOT/tests/torch_ref.py" \
    "$C1_PINNED_REFERENCE_HELPER_PATH" \
    "$SEQCOUNT_DISCOVERY_JSON" \
    "$MIXED_DISCOVERY_JSON" \
    "$CONFIRMATION_JSON" \
    "${SO_PATHS[0]}" \
    "$PATCHED_ROOT/flash_kda/__init__.py" \
    "$PATCHED_ROOT/csrc/flash_kda.cpp" \
    "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu"

echo "===== PYTHON_COMPILE_GATE ====="
"$PYTHON_BIN" -m py_compile "$OWNED/run_varlen_dispatch_release.py"

# No setup.py, Ninja, NVCC, patch generator, source mutation, or extension rebuild below.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export C1_VARLEN_DISPATCH_RELEASE_CLEAN_GPU_GATES=1
export C1_VARLEN_DISPATCH_RELEASE_RUNNER_SHA256="$EXPECTED_RELEASE_RUNNER_SHA256"
cd "$PATCHED_ROOT"
echo "===== VARLEN_DISPATCH_RELEASE ====="
"$PYTHON_BIN" "$OWNED/run_varlen_dispatch_release.py" \
    --reference-root "$REFERENCE_ROOT" \
    --seqcount-discovery-json "$SEQCOUNT_DISCOVERY_JSON" \
    --mixed-discovery-json "$MIXED_DISCOVERY_JSON" \
    --confirmation-json "$CONFIRMATION_JSON" \
    --json "$RESULTS_DIR/c1_varlen_dispatch_release_${LABEL}.json"
require_clean AFTER_RELEASE || exit 93
