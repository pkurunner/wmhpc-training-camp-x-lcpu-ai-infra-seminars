#!/usr/bin/env bash
# Diagnostic only: run the same v5 softmax-directed inputs against frozen v4
# and v5 DSOs in separate processes.  This job cannot promote either plugin.
#SBATCH --job-name=c2-native-v5-softmax-diagnose
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v5-directed-diagnosis-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONPATH=

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v5-directed-diagnosis-artifacts-20260829
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
BASE_HARNESS=/home/lcpu/85117379/native_c2_operator_bench_20260829.py
DIRECTED_HARNESS=/home/lcpu/85117379/native_c2_v5_softmax_directed_20260829.py
REFERENCE_DSO=/home/lcpu/85117379/c2-native-plugin-v4-aot-artifacts-20260829/job11496/vllm/_native_c2_msa_decode_plugin.abi3.so
CANDIDATE_DSO=/home/lcpu/85117379/c2-native-plugin-v5-aot-artifacts-20260829/job11760/vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_BASE_HARNESS_SHA=0b51fede1e6258eea0b535acba2a86e371b284e9655352156cdeb3d310d90c28
EXPECTED_DIRECTED_HARNESS_SHA=43ad54c5ad72cc40bf8b8a6e17592fa16cacc1d1e62f5f8c42ac2a4d3206cdd1
EXPECTED_REFERENCE_DSO_SHA=e00877088ac94bc17d7c501dff418561f195f6e349af46f532ead0a6aa3763d5
EXPECTED_CANDIDATE_DSO_SHA=9ab755faf79ec7bc574a8f5afa09b22b9acedc8257d3ccfad6fc98cc3f1d2538

[[ -d "${ARTIFACT_ROOT}" && ! -e "${ARTIFACT_DIR}" ]] || { echo 'artifact root missing or job output reused' >&2; exit 2; }
mkdir -p "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1
finalize() { local rc=$?; trap - EXIT; set +e; printf 'FINAL_RC=%s UTC=%s\n' "${rc}" "$(date -u +%FT%TZ)" | tee "${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt"; exit "${rc}"; }
trap finalize EXIT
export PATH=/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin
compute_apps() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'; }
snapshot() { printf '\n===== %s UTC=%s =====\n' "$1" "$(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap --format=csv,noheader,nounits; compute_apps || true; }

snapshot PRE
gpu_line=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
apps_before=$(compute_apps) || { echo 'unable to query pre-run compute apps' >&2; exit 1; }
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* && -z "${apps_before}" ]]
printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${EXPECTED_REFERENCE_DSO_SHA}" "${REFERENCE_DSO}" \
  "${EXPECTED_CANDIDATE_DSO_SHA}" "${CANDIDATE_DSO}" \
  "${EXPECTED_BASE_HARNESS_SHA}" "${BASE_HARNESS}" \
  "${EXPECTED_DIRECTED_HARNESS_SHA}" "${DIRECTED_HARNESS}" | sha256sum -c -
sha256sum "${REFERENCE_DSO}" "${CANDIDATE_DSO}" "${BASE_HARNESS}" "${DIRECTED_HARNESS}" > "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"

run_arm() {
  local library=$1 result=$2 rc
  set +e
  timeout --foreground --preserve-status --kill-after=10s 600s \
    "${PYTHON}" "${DIRECTED_HARNESS}" --library "${library}" \
    --base-harness "${BASE_HARNESS}" --sync-repeats 4 \
    --max-iteration-seconds 15 --atol 1e-4 --rtol 1e-3 > "${result}"
  rc=$?
  set -e
  [[ ${rc} -eq 0 || ${rc} -eq 1 ]] || return "${rc}"
}

REFERENCE_JSON=${ARTIFACT_DIR}/reference-v4-job${SLURM_JOB_ID}.json
CANDIDATE_JSON=${ARTIFACT_DIR}/candidate-v5-job${SLURM_JOB_ID}.json
COMPARISON_JSON=${ARTIFACT_DIR}/v5-softmax-diagnosis-job${SLURM_JOB_ID}.json
run_arm "${REFERENCE_DSO}" "${REFERENCE_JSON}"
run_arm "${CANDIDATE_DSO}" "${CANDIDATE_JSON}"

"${PYTHON}" - "${REFERENCE_JSON}" "${CANDIDATE_JSON}" "${COMPARISON_JSON}" <<'PY'
import json
import math
import sys
from pathlib import Path

reference_path, candidate_path, output_path = map(Path, sys.argv[1:])
reference = json.loads(reference_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
for label, payload in (("reference_v4", reference), ("candidate_v5", candidate)):
    assert payload["schema"] == "c2-native-c2-v5-softmax-directed-v2", label
    assert payload["rank_directed_case_count"] == 4, label
    assert all(case["all_gates_pass"] is True for case in payload["rank_directed"]), label
    assert payload["all_invalid"]["all_gates_pass"] is True, label
    assert payload["softmax_directed_case_count"] == 2, label
    for case in payload["softmax_directed"]:
        assert case["oracle"]["finite_output"] is True, label
        assert case["caller_output"]["pointer_unchanged"] is True, label
        assert case["caller_output"]["return_is_none"] is True, label
        assert case["stability"]["bitwise_repeatable"] is True, label

comparisons = []
for reference_case, candidate_case in zip(
    reference["softmax_directed"], candidate["softmax_directed"], strict=True
):
    assert reference_case["case"] == candidate_case["case"]
    reference_max = float(reference_case["oracle"]["max_abs"])
    candidate_max = float(candidate_case["oracle"]["max_abs"])
    comparisons.append({
        "case": reference_case["case"],
        "candidate_v5_max_abs": candidate_max,
        "candidate_v5_mean_abs": float(candidate_case["oracle"]["mean_abs"]),
        "max_abs_ratio_v5_over_v4": candidate_max / reference_max if reference_max else None,
        "reference_v4_max_abs": reference_max,
        "reference_v4_mean_abs": float(reference_case["oracle"]["mean_abs"]),
    })
payload = {
    "diagnostic_only": True,
    "not_an_acceptance_or_promotion_gate": True,
    "schema": "c2-native-c2-v5-softmax-directed-diagnosis-v2",
    "softmax_case_comparisons": comparisons,
}
output_path.write_text(
    json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY

snapshot POST
apps_after=$(compute_apps) || { echo 'unable to query post-run compute apps' >&2; exit 1; }
[[ -z "${apps_after}" ]]
sha256sum "${REFERENCE_JSON}" "${CANDIDATE_JSON}" "${COMPARISON_JSON}" \
  "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
