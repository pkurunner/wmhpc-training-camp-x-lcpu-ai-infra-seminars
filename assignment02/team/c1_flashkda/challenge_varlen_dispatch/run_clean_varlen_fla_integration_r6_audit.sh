#!/usr/bin/env bash
# Clean single-B300 r6 public-FLA packed-varlen production freeze.  No build path.
set -Eeuo pipefail

if [[ "${C1_VARLEN_FLA_INTEGRATION_R6_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_VARLEN_FLA_INTEGRATION_R6_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
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
EXPECTED_RUNNER_SHA256="71a016307d385d846dfc9e58fefeb041446616a08d3ee36d73f2ac2d3d5ac058"
EXPECTED_ANALYZER_SHA256="9cb9c626fdfa2426e64611f1596f1f71148b855b9aa7ab7b2ad363b0aa28cb0b"
EXPECTED_AUTO_SHA256="2b817adb7d21d1f223e8df4616eeccd74e34a5b1944492211f0f0254147ba883"
EXPECTED_BACKEND_SHA256="8555995c04ecd666a580ddee02eae1d34820ef1a601cbad5d10f9c6b8505974b"
EXPECTED_METADATA_SHA256="f89a97ba284c7a24a3df54efca7bb60eb70f9cbda659200bd6cb3e8dfdaf4ccd"
EXPECTED_AUTO_POLICY_TEST_SHA256="9ab690ea977dd18ecc6d41d451b442e38c7d524d97e06da07180f37ec7b6f480"
EXPECTED_METADATA_POLICY_TEST_SHA256="85ef2c1dcec8ca2c000b76553b921dcd1d5c92417c40617b2541194dc3435826"
EXPECTED_CONFIRMATION_SHA256="9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b"
EXPECTED_SHARED_SHA256="4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f"
EXPECTED_PREFETCH2_SHA256="752126488487ac317a7ee167b660b0895562e4877aedbc4ce2a599c1f59a10d0"
EXPECTED_VSHARD4_SHA256="445c90815919fb8c1db3bc79289d10029ea77b1d9accb393c326c120ba1f8385"
EXPECTED_HARNESS_SHA256="5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52"
EXPECTED_TORCH_REF_SHA256="bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5"
EXPECTED_FLASH_KDA_PY_SHA256="9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_varlen_fla_integration_r6_${LABEL}_job${SLURM_JOB_ID:-none}.log"
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
require_sha "$OWNED/run_varlen_fla_integration_r6.py" "$EXPECTED_RUNNER_SHA256" "r6 integration runner"
require_sha "$OWNED/analyze_varlen_fla_integration_r6.py" "$EXPECTED_ANALYZER_SHA256" "r6 integration analyzer"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py" "$EXPECTED_AUTO_SHA256" "auto dispatcher"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py" "$EXPECTED_BACKEND_SHA256" "C1 FLA backend"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/varlen_metadata.py" "$EXPECTED_METADATA_SHA256" "varlen metadata"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/test_auto_dispatch_policy.py" "$EXPECTED_AUTO_POLICY_TEST_SHA256" "auto dispatcher policy tests"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/test_varlen_metadata_policy.py" "$EXPECTED_METADATA_POLICY_TEST_SHA256" "varlen metadata policy tests"
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
echo "===== R6_SOURCE_IDENTITY ====="
sha256sum "$OWNED/run_varlen_fla_integration_r6.py" "$OWNED/analyze_varlen_fla_integration_r6.py" "$OWNED/run_clean_varlen_fla_integration_r6_audit.sh" "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/test_auto_dispatch_policy.py" "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/test_varlen_metadata_policy.py"
sha256sum "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py" "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py" "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/varlen_metadata.py" "$OWNED/run_varlen_dispatch_confirmation.py" "$A02_ROOT/team/c1_flashkda/challenge_seqcount_dispatch/run_seqcount_dispatch.py" "$A02_ROOT/team/c1_flashkda/challenge_prefetch2/prefetch2.py" "$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2/vshard4_prefetch2.py" "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" "$C1_PINNED_REFERENCE_HELPER_PATH" "${SO_PATHS[0]}" "$PATCHED_ROOT/flash_kda/__init__.py" "$REFERENCE_ROOT/tests/torch_ref.py" "$FLA_ROOT/fla/__init__.py" "$FLA_ROOT/fla/ops/backends/__init__.py" "$FLA_ROOT/fla/ops/kda/__init__.py" "$FLA_ROOT/fla/ops/kda/backends/__init__.py" "$FLA_ROOT/fla/ops/kda/backends/flash_kda.py" "$FLA_ROOT/fla/ops/kda/chunk.py"
"$PYTHON_BIN" -m py_compile "$OWNED/run_varlen_fla_integration_r6.py" "$OWNED/analyze_varlen_fla_integration_r6.py"

# The only reference-helper load is the pinned direct-load interception in the runner; no setup.py, Ninja, NVCC, or extension build is permitted.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$FLA_ROOT:$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export C1_VARLEN_FLA_INTEGRATION_R6_CLEAN_GPU=1 C1_B300_FLASH_KDA=1 C1_B300_VARLEN_CPU_DESCRIPTOR=1 FLA_FLASH_KDA=1
export C1_VARLEN_FLA_INTEGRATION_R6_RUNNER_SHA256="$EXPECTED_RUNNER_SHA256"
cd "$PATCHED_ROOT"
echo "===== CPU_POLICY_TESTS ====="
"$PYTHON_BIN" "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/test_auto_dispatch_policy.py"
"$PYTHON_BIN" "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/test_varlen_metadata_policy.py"
echo "===== CPU_R6_DESCRIBE_AND_CONSTRUCTION ====="
"$PYTHON_BIN" "$OWNED/run_varlen_fla_integration_r6.py" --describe --json "$RESULTS_DIR/c1_varlen_fla_integration_r6_${LABEL}.plan.json"
"$PYTHON_BIN" "$OWNED/run_varlen_fla_integration_r6.py" --cpu-construction-check --json "$RESULTS_DIR/c1_varlen_fla_integration_r6_${LABEL}.cpu.json"
echo "===== VARLEN_FLA_INTEGRATION_R6 ====="
ARTIFACT="$RESULTS_DIR/c1_varlen_fla_integration_r6_${LABEL}.json"
AUDIT="$RESULTS_DIR/c1_varlen_fla_integration_r6_${LABEL}.independent_audit.json"
"$PYTHON_BIN" "$OWNED/run_varlen_fla_integration_r6.py" --reference-root "$REFERENCE_ROOT" --json "$ARTIFACT"
require_clean AFTER_R6_INTEGRATION || exit 93
ARTIFACT_SHA256="$(sha256sum "$ARTIFACT" | awk '{print $1}')"
echo "===== INDEPENDENT_AUDIT ====="
"$PYTHON_BIN" "$OWNED/analyze_varlen_fla_integration_r6.py" "$ARTIFACT" --expected-sha256 "$ARTIFACT_SHA256" --json "$AUDIT"
require_clean AFTER_INDEPENDENT_AUDIT || exit 94
