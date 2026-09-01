#!/usr/bin/env bash
# Clean single-B300 diagnostic-only packed-varlen public-overhead attribution.
# This script invokes no compiler, build system, setup.py, Ninja, NVCC, or CMake.
set -Eeuo pipefail

if [[ "${C1_VARLEN_PUBLIC_OVERHEAD_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU diagnostic: set C1_VARLEN_PUBLIC_OVERHEAD_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT}"
: "${REFERENCE_ROOT:?set REFERENCE_ROOT}"
: "${FLA_ROOT:?set FLA_ROOT}"
: "${LABEL:?set LABEL}"
: "${C1_PINNED_REFERENCE_HELPER_PATH:?set C1_PINNED_REFERENCE_HELPER_PATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_varlen_dispatch"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
EXPECTED_PATCHED_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLA_COMMIT="a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
EXPECTED_FLA_INIT_SHA256="b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d"
EXPECTED_FLA_BACKENDS_SHA256="a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635"
EXPECTED_FLA_KDA_INIT_SHA256="24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb"
EXPECTED_FLA_KDA_BACKENDS_SHA256="86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797"
EXPECTED_FLA_FLASH_KDA_SHA256="0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2"
EXPECTED_FLA_CHUNK_SHA256="a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8"
EXPECTED_SO_SHA256="8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_HELPER_SHA256="8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
EXPECTED_DIAGNOSTIC_RUNNER_SHA256="651ff9af72ddd423d094b018ab7b29438a4283cb2cc50f39254874b8fd84e866"
EXPECTED_INTEGRATION_RUNNER_SHA256="5db71f29335220496ca9540924e17c5f160b0bc8237060921cffaecb708f22bb"
EXPECTED_AUTO_SHA256="2b817adb7d21d1f223e8df4616eeccd74e34a5b1944492211f0f0254147ba883"
EXPECTED_BACKEND_SHA256="6321b1a75713560d25fd92bb94e8e4e15401d206269a8fc10ca5b8ab4433174f"
EXPECTED_METADATA_SHA256="16c01cfc2a8aeee4d80362435053009c3b6397ab09e01d390ac14a38a29b822d"
EXPECTED_CONFIRMATION_SHA256="9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b"
EXPECTED_SHARED_SHA256="4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f"
EXPECTED_PREFETCH2_SHA256="752126488487ac317a7ee167b660b0895562e4877aedbc4ce2a599c1f59a10d0"
EXPECTED_VSHARD4_SHA256="445c90815919fb8c1db3bc79289d10029ea77b1d9accb393c326c120ba1f8385"
EXPECTED_HARNESS_SHA256="5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52"
EXPECTED_TORCH_REF_SHA256="bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5"
EXPECTED_FLASH_KDA_PY_SHA256="9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_varlen_public_overhead_${LABEL}_job${SLURM_JOB_ID:-none}.log"
exec > >(tee "$LOG") 2>&1

gpu_query() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader; }
app_query() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; }
memory_query() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
require_clean() {
    local stage="$1" apps used
    apps="$(app_query)" || return 92; used="$(memory_query)" || return 92
    echo "${stage}_COMPUTE_APPS_BEGIN"; printf '%s\n' "$apps"; echo "${stage}_COMPUTE_APPS_END"
    printf '%s_MEMORY_USED_MIB=%s\n' "$stage" "$used"
    [[ -z "$apps" ]] || return 1
    [[ "$(printf '%s\n' "$used" | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || return 1
    while IFS= read -r line; do [[ "$line" =~ ^[[:space:]]*0[[:space:]]*$ ]] || return 1; done <<<"$used"
}
finish() { local rc=$?; trap - EXIT; echo "===== POST_AUDIT ====="; date -Is; gpu_query || rc=92; require_clean POST || rc=91; echo "FINAL_RC=$rc"; exit "$rc"; }
trap finish EXIT
require_sha() { local path="$1" expected="$2" label="$3"; [[ -f "$path" ]] || { echo "missing $label" >&2; return 86; }; [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || { echo "$label SHA256 gate failed" >&2; return 87; }; }
require_commit() { [[ "$(git -C "$1" rev-parse HEAD)" == "$2" ]] || { echo "commit gate failed for $1" >&2; return 85; }; }

echo "===== ENVIRONMENT_GATE ====="
command -v "$PYTHON_BIN"
require_commit "$PATCHED_ROOT" "$EXPECTED_PATCHED_COMMIT"
require_commit "$REFERENCE_ROOT" "$EXPECTED_PATCHED_COMMIT"
require_commit "$FLA_ROOT" "$EXPECTED_FLA_COMMIT"
[[ -z "$(git -C "$REFERENCE_ROOT" status --short --untracked-files=no)" ]] || { echo "reference tracked/staged diff is not clean" >&2; exit 84; }
[[ -z "$(git -C "$FLA_ROOT" status --short --untracked-files=no)" ]] || { echo "FLA tracked/staged diff is not clean" >&2; exit 84; }
[[ "${C1_PINNED_REFERENCE_HELPER_SHA256:-}" == "$EXPECTED_HELPER_SHA256" ]] || { echo "helper SHA environment drift" >&2; exit 87; }
export C1_PINNED_REFERENCE_HELPER_SHA256="$EXPECTED_HELPER_SHA256"
SO_PATHS=("$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so)
[[ "${#SO_PATHS[@]}" -eq 1 && -f "${SO_PATHS[0]}" ]] || { echo "expected one audited extension SO" >&2; exit 89; }
require_sha "${SO_PATHS[0]}" "$EXPECTED_SO_SHA256" "extension"
require_sha "$PATCHED_ROOT/flash_kda/__init__.py" "$EXPECTED_FLASH_KDA_PY_SHA256" "flash_kda Python"
require_sha "$REFERENCE_ROOT/tests/torch_ref.py" "$EXPECTED_TORCH_REF_SHA256" "Torch reference"
require_sha "$FLA_ROOT/fla/__init__.py" "$EXPECTED_FLA_INIT_SHA256" "FLA package"
require_sha "$FLA_ROOT/fla/ops/backends/__init__.py" "$EXPECTED_FLA_BACKENDS_SHA256" "FLA backend registry"
require_sha "$FLA_ROOT/fla/ops/kda/__init__.py" "$EXPECTED_FLA_KDA_INIT_SHA256" "FLA public KDA package"
require_sha "$FLA_ROOT/fla/ops/kda/backends/__init__.py" "$EXPECTED_FLA_KDA_BACKENDS_SHA256" "FLA KDA registry package"
require_sha "$FLA_ROOT/fla/ops/kda/backends/flash_kda.py" "$EXPECTED_FLA_FLASH_KDA_SHA256" "FLA pinned FlashKDA"
require_sha "$FLA_ROOT/fla/ops/kda/chunk.py" "$EXPECTED_FLA_CHUNK_SHA256" "FLA public chunk_kda"
require_sha "$C1_PINNED_REFERENCE_HELPER_PATH" "$EXPECTED_HELPER_SHA256" "reference helper"
require_sha "$OWNED/run_varlen_public_overhead_diagnosis.py" "$EXPECTED_DIAGNOSTIC_RUNNER_SHA256" "diagnostic runner"
require_sha "$OWNED/run_varlen_fla_integration.py" "$EXPECTED_INTEGRATION_RUNNER_SHA256" "current integration runner"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py" "$EXPECTED_AUTO_SHA256" "auto dispatcher"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py" "$EXPECTED_BACKEND_SHA256" "C1 FLA backend"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/varlen_metadata.py" "$EXPECTED_METADATA_SHA256" "varlen metadata"
require_sha "$OWNED/run_varlen_dispatch_confirmation.py" "$EXPECTED_CONFIRMATION_SHA256" "confirmation runner"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_seqcount_dispatch/run_seqcount_dispatch.py" "$EXPECTED_SHARED_SHA256" "shared runner"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_prefetch2/prefetch2.py" "$EXPECTED_PREFETCH2_SHA256" "v2 wrapper"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2/vshard4_prefetch2.py" "$EXPECTED_VSHARD4_SHA256" "v4 wrapper"
require_sha "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" "$EXPECTED_HARNESS_SHA256" "harness"
[[ "$(gpu_query | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || exit 88

echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query; require_clean PRE || exit 90
echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"; printf 'PATCHED_STATUS_BEGIN\n'; git -C "$PATCHED_ROOT" status --short --untracked-files=no; printf 'PATCHED_STATUS_END\n'
printf 'REFERENCE_COMMIT=%s\n' "$(git -C "$REFERENCE_ROOT" rev-parse HEAD)"; printf 'REFERENCE_STATUS_BEGIN\n'; git -C "$REFERENCE_ROOT" status --short --untracked-files=no; printf 'REFERENCE_STATUS_END\n'
printf 'FLA_COMMIT=%s\n' "$(git -C "$FLA_ROOT" rev-parse HEAD)"; printf 'FLA_STATUS_BEGIN\n'; git -C "$FLA_ROOT" status --short --untracked-files=no; printf 'FLA_STATUS_END\n'
sha256sum "$OWNED/run_varlen_public_overhead_diagnosis.py" "$OWNED/run_clean_varlen_public_overhead_diagnosis.sh" "$OWNED/run_varlen_fla_integration.py" "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py" "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py" "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/varlen_metadata.py" "$OWNED/run_varlen_dispatch_confirmation.py" "$C1_PINNED_REFERENCE_HELPER_PATH" "${SO_PATHS[0]}" "$PATCHED_ROOT/flash_kda/__init__.py" "$REFERENCE_ROOT/tests/torch_ref.py" "$FLA_ROOT/fla/__init__.py" "$FLA_ROOT/fla/ops/backends/__init__.py" "$FLA_ROOT/fla/ops/kda/__init__.py" "$FLA_ROOT/fla/ops/kda/backends/__init__.py" "$FLA_ROOT/fla/ops/kda/backends/flash_kda.py" "$FLA_ROOT/fla/ops/kda/chunk.py"
echo "===== PYTHON_COMPILE_GATE ====="; "$PYTHON_BIN" -m py_compile "$OWNED/run_varlen_public_overhead_diagnosis.py"

# The runner only direct-loads the SHA-pinned reference helper.  It has no
# extension build path, and this shell invokes no compilation command.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$FLA_ROOT:$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export C1_VARLEN_PUBLIC_OVERHEAD_CLEAN_GPU=1 C1_B300_FLASH_KDA=1 C1_B300_VARLEN_CPU_DESCRIPTOR=1 FLA_FLASH_KDA=1
export C1_VARLEN_PUBLIC_OVERHEAD_RUNNER_SHA256="$EXPECTED_DIAGNOSTIC_RUNNER_SHA256"
cd "$PATCHED_ROOT"
echo "===== VARLEN_PUBLIC_OVERHEAD_DIAGNOSIS ====="
"$PYTHON_BIN" "$OWNED/run_varlen_public_overhead_diagnosis.py" --reference-root "$REFERENCE_ROOT" --json "$RESULTS_DIR/c1_varlen_public_overhead_${LABEL}.json"
require_clean AFTER_DIAGNOSIS || exit 93
