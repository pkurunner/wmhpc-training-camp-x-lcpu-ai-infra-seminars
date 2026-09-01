#!/usr/bin/env bash
# Directed correctness/liveness gate for a reviewed v6 register-numerator DSO.
# This consumes an explicit, immutable AOT plugin only; it makes no performance
# claim and must pass before a v6 candidate can enter ABBA timing.
#SBATCH --job-name=c2-native-plugin-v6-directed
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v6-directed-artifacts-20260830/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONPATH= PYTHONDONTWRITEBYTECODE=1

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v6-directed-artifacts-20260830
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
BASE_HARNESS=/home/lcpu/85117379/native_c2_operator_bench_20260829.py
V5_DIRECTED_HARNESS=/home/lcpu/85117379/native_c2_v5_softmax_directed_20260829.py
DIRECTED_HARNESS=/home/lcpu/85117379/native_c2_v6_register_numerator_directed_20260830.py
EXPECTED_BASE_HARNESS_SHA=0b51fede1e6258eea0b535acba2a86e371b284e9655352156cdeb3d310d90c28
EXPECTED_V5_DIRECTED_HARNESS_SHA=43ad54c5ad72cc40bf8b8a6e17592fa16cacc1d1e62f5f8c42ac2a4d3206cdd1
EXPECTED_V6_CANDIDATE_PLUGIN=/home/lcpu/85117379/c2-native-plugin-v6-aot-artifacts-20260830/job12278/vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_V6_CANDIDATE_SHA=0d5492bee11b2a785d5f16681614b0a1fc6c74ddb33351d44f50b124580e21a4
EXPECTED_V6_SOURCE=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v6-register-numerator-20260830-job12278/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
EXPECTED_V6_SOURCE_SHA=c7f05928fd2ba3a0e29d3154f2bf56849e9d7ad6fd3eddc24b0890722e117f02

: "${SLURM_JOB_ID:?must run under Slurm}"
: "${C2_V6_CANDIDATE_PLUGIN:?set an accepted v6 plugin absolute path}"
: "${C2_EXPECTED_V6_CANDIDATE_SHA:?set the reviewed v6 plugin SHA-256}"
: "${C2_V6_CANDIDATE_SOURCE:?set the derived v6 source absolute path}"
: "${C2_EXPECTED_V6_DIRECTED_HARNESS_SHA:?set the frozen v6 harness SHA-256}"
: "${C2_EXPECTED_V6_DIRECTED_SCRIPT_SHA:?set the frozen v6 Slurm script SHA-256}"
[[ "${C2_V6_CANDIDATE_PLUGIN}" == /* && "${C2_V6_CANDIDATE_SOURCE}" == /* ]]
[[ "${C2_EXPECTED_V6_CANDIDATE_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ "${C2_EXPECTED_V6_DIRECTED_HARNESS_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ "${C2_EXPECTED_V6_DIRECTED_SCRIPT_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ "${C2_V6_CANDIDATE_PLUGIN}" == "${EXPECTED_V6_CANDIDATE_PLUGIN}" ]]
[[ "${C2_EXPECTED_V6_CANDIDATE_SHA}" == "${EXPECTED_V6_CANDIDATE_SHA}" ]]
[[ "${C2_V6_CANDIDATE_SOURCE}" == "${EXPECTED_V6_SOURCE}" ]]
RUN_ID=${C2_RUN_ID:-${SLURM_JOB_ID}}
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${RUN_ID}
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
[[ "${SCRIPT_PATH}" == /* ]]
[[ -d "${ARTIFACT_ROOT}" && ! -e "${ARTIFACT_DIR}" ]] || {
  echo 'artifact root missing or run-scoped output already exists' >&2; exit 2;
}
[[ -f "${C2_V6_CANDIDATE_PLUGIN}" && -f "${C2_V6_CANDIDATE_SOURCE}" ]] || {
  echo 'candidate plugin/source missing' >&2; exit 2;
}
[[ -f "${BASE_HARNESS}" && -f "${V5_DIRECTED_HARNESS}" && -f "${DIRECTED_HARNESS}" ]] || {
  echo 'frozen directed-harness input missing' >&2; exit 2;
}
mkdir -p "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1

export PATH=/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin
# Keep py_compile/import cache writes under the immutable run artifact rather
# than beside any frozen harness input.
export PYTHONPYCACHEPREFIX="${ARTIFACT_DIR}/pycache"
post_checked=0
compute_apps() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}
snapshot() {
  local label=$1
  {
    printf '===== %s UTC=%s =====\n' "${label}" "$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap \
      --format=csv,noheader,nounits
    printf '%s\n' 'COMPUTE_APPS'
    compute_apps || true
  } | tee "${ARTIFACT_DIR}/gpu-${label,,}-job${SLURM_JOB_ID}.txt"
}
ensure_clean_post() {
  snapshot POST
  local apps_after
  apps_after=$(compute_apps) || { echo 'unable to query post-run compute apps' >&2; return 1; }
  [[ -z "${apps_after}" ]] || { echo "post-run compute apps remain: ${apps_after}" >&2; return 1; }
  post_checked=1
}
finalize() {
  local rc=$?
  trap - EXIT
  set +e
  if [[ "${post_checked}" -eq 0 ]]; then
    snapshot FINAL_POST
    local apps_after
    apps_after=$(compute_apps)
    if [[ "${rc}" -eq 0 && -n "${apps_after}" ]]; then
      echo "post-run compute apps remain: ${apps_after}" >&2
      rc=1
    fi
  fi
  printf 'FINAL_RC=%s UTC=%s\n' "${rc}" "$(date -u +%FT%TZ)" \
    | tee "${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt"
  exit "${rc}"
}
trap finalize EXIT

snapshot PRE
gpu_line=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
apps_before=$(compute_apps) || { echo 'unable to query pre-run compute apps' >&2; exit 1; }
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* && -z "${apps_before}" ]] || {
  echo 'requires a clean B300 capability-10.3 GPU' >&2; exit 1;
}

printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${C2_EXPECTED_V6_CANDIDATE_SHA}" "${C2_V6_CANDIDATE_PLUGIN}" \
  "${EXPECTED_V6_SOURCE_SHA}" "${C2_V6_CANDIDATE_SOURCE}" \
  "${EXPECTED_BASE_HARNESS_SHA}" "${BASE_HARNESS}" \
  "${EXPECTED_V5_DIRECTED_HARNESS_SHA}" "${V5_DIRECTED_HARNESS}" \
  "${C2_EXPECTED_V6_DIRECTED_HARNESS_SHA}" "${DIRECTED_HARNESS}" \
  "${C2_EXPECTED_V6_DIRECTED_SCRIPT_SHA}" "${SCRIPT_PATH}" | sha256sum -c -
"${PYTHON}" -m py_compile "${BASE_HARNESS}" "${V5_DIRECTED_HARNESS}" "${DIRECTED_HARNESS}"
"${PYTHON}" - "${DIRECTED_HARNESS}" <<'PY'
import ast
import sys
from pathlib import Path

path = Path(sys.argv[1])
ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("AST_OK", path)
PY

RUN_MANIFEST=${ARTIFACT_DIR}/run-manifest-job${SLURM_JOB_ID}.json
"${PYTHON}" - "${RUN_MANIFEST}" \
  "${C2_V6_CANDIDATE_PLUGIN}" "${C2_V6_CANDIDATE_SOURCE}" \
  "${BASE_HARNESS}" "${V5_DIRECTED_HARNESS}" "${DIRECTED_HARNESS}" "${SCRIPT_PATH}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
labels = (
    "plugin", "v6_source", "base_harness", "v5_directed_harness",
    "v6_directed_harness", "v6_directed_script",
)
paths = [Path(value).resolve() for value in sys.argv[2:]]
assert len(labels) == len(paths)
entries = []
for label, path in zip(labels, paths, strict=True):
    assert path.is_absolute() and path.is_file(), (label, path)
    entries.append({
        "label": label,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    })
manifest_path.write_text(
    json.dumps(
        {"schema": "c2-native-c2-v6-directed-input-manifest-v1", "inputs": entries},
        allow_nan=False, indent=2, sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY
INPUT_MANIFEST=${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256
sha256sum "${C2_V6_CANDIDATE_PLUGIN}" "${C2_V6_CANDIDATE_SOURCE}" \
  "${BASE_HARNESS}" "${V5_DIRECTED_HARNESS}" "${DIRECTED_HARNESS}" "${SCRIPT_PATH}" \
  "${RUN_MANIFEST}" "${ARTIFACT_DIR}/gpu-pre-job${SLURM_JOB_ID}.txt" > "${INPUT_MANIFEST}"

RESULT=${ARTIFACT_DIR}/v6-register-numerator-directed-job${SLURM_JOB_ID}.json
timeout --foreground --preserve-status --kill-after=10s 600s \
  "${PYTHON}" "${DIRECTED_HARNESS}" --library "${C2_V6_CANDIDATE_PLUGIN}" \
  --base-harness "${BASE_HARNESS}" --v5-directed-harness "${V5_DIRECTED_HARNESS}" \
  --sync-repeats 4 --max-iteration-seconds 15 --atol 1e-4 --rtol 1e-3 > "${RESULT}"

"${PYTHON}" - "${RESULT}" "${C2_V6_CANDIDATE_PLUGIN}" <<'PY'
import json
import math
import sys
from pathlib import Path

result_path, library = map(Path, sys.argv[1:])
payload = json.loads(result_path.read_text(encoding="utf-8"))
assert payload["schema"] == "c2-native-c2-v6-register-numerator-directed-v1"
assert payload["all_gates_pass"] is True
assert payload["no_monkeypatch"] is True
assert payload["dispatch"] == "direct_torch.ops._C.native_c2_msa_decode"
assert Path(payload["operator_library"]).resolve() == library.resolve()
assert payload["contract"]["producer_ctas"] == 4
assert payload["contract"]["pages_per_producer"] == 4
assert payload["contract"]["selected_pages"] == 16
assert payload["contract"]["page_size"] == 128

def assert_case(case: dict) -> None:
    assert case["all_gates_pass"] is True
    assert case["caller_output"]["pointer_unchanged"] is True
    assert case["caller_output"]["return_is_none"] is True
    assert case["stability"]["bitwise_repeatable"] is True
    assert case["oracle"]["finite_output"] is True
    assert case["oracle"]["reference_fp64_fp32_agree"] is True
    assert case["oracle"]["fp64"]["allclose"] is True
    assert case["oracle"]["fp32"]["allclose"] is True
    elapsed = case["sync_watchdog"]["elapsed_seconds"]
    assert len(elapsed) == 4
    assert all(math.isfinite(float(value)) and 0.0 <= float(value) <= 15.0 for value in elapsed)

assert_case(payload["head_dim_encoding"])
encoding = payload["head_dim_encoding"]["head_dim_encoding"]
assert encoding["dim_code_count"] == 128
assert encoding["head_signature_count"] == 64
assert len(encoding["head_signature_offsets"]) == 64
assert len(set(encoding["head_signature_offsets"])) == 64
assert len(encoding["target_tokens"]) == 64
assert encoding["exact_location_match"] is True
assert encoding["nonzero_mismatch_count"] == 0
assert encoding["max_abs_to_exact_expected"] == 0.0

assert payload["rank_directed_case_count"] == 4
for expected_rank, case in enumerate(payload["rank_directed"]):
    assert_case(case)
    assert case["producer_rank"] == expected_rank
    assert case["selected_page_indices"] == list(range(4 * expected_rank, 4 * expected_rank + 4))
    assert case["rank_expected_value"]["allclose"] is True

assert payload["softmax_directed_case_count"] == 2
expected_softmax = (("all-valid-online-rescale", 2048, False), ("mixed-causal-tail-37", 37, True))
for case, (label, sequence_length, mixed_tail) in zip(payload["softmax_directed"], expected_softmax, strict=True):
    assert_case(case)
    assert case["case"] == label
    assert case["sequence_length"] == sequence_length
    assert case["mixed_causal_subgroup"] is mixed_tail

assert_case(payload["all_invalid"])
assert payload["all_invalid"]["case"] == "all-invalid"
assert payload["all_invalid"]["all_zero"] is True
assert payload["all_invalid"]["oracle"]["fp64"]["max_abs"] == 0.0
assert payload["all_invalid"]["oracle"]["fp32"]["max_abs"] == 0.0

profiler = payload["profiler"]
assert profiler["native_dispatcher_hit"] is True
assert profiler["native_kernel_hit"] is True
assert profiler["exactly_one_dispatcher_event"] is True
assert profiler["exactly_one_native_kernel_event"] is True
assert profiler["dispatcher_events"] == [
    {"key": "_C::native_c2_msa_decode", "count": 1}
]
assert len(profiler["kernel_events"]) == 1
assert profiler["kernel_events"][0]["count"] == 1
assert profiler["caller_output_pointer_unchanged"] is True
assert profiler["return_is_none"] is True
assert profiler["finite_output"] is True
PY

ensure_clean_post
sha256sum -c "${INPUT_MANIFEST}"
OUTPUT_MANIFEST=${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256
sha256sum "${RESULT}" "${RUN_MANIFEST}" "${INPUT_MANIFEST}" \
  "${ARTIFACT_DIR}/gpu-pre-job${SLURM_JOB_ID}.txt" \
  "${ARTIFACT_DIR}/gpu-post-job${SLURM_JOB_ID}.txt" > "${OUTPUT_MANIFEST}"
