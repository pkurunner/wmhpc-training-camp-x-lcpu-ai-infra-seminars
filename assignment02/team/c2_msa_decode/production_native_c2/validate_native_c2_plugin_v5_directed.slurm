#!/usr/bin/env bash
# Four-producer plus subgroup-softmax correctness/liveness gate for an already
# accepted v5 direct-performance candidate.  No performance claim is derived here.
#SBATCH --job-name=c2-native-plugin-v5-directed
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v5-directed-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONPATH=

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v5-directed-artifacts-20260829
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
BASE_HARNESS=/home/lcpu/85117379/native_c2_operator_bench_20260829.py
DIRECTED_HARNESS=/home/lcpu/85117379/native_c2_v5_softmax_directed_20260829.py
EXPECTED_BASE_HARNESS_SHA=0b51fede1e6258eea0b535acba2a86e371b284e9655352156cdeb3d310d90c28
EXPECTED_DIRECTED_HARNESS_SHA=43ad54c5ad72cc40bf8b8a6e17592fa16cacc1d1e62f5f8c42ac2a4d3206cdd1
EXPECTED_SOURCE_SHA=6409a54689ebbdd1f72fad76c2d72dfe588f79d3b8dd6c4dbaf1df5b630f0931

: "${C2_CANDIDATE_PLUGIN:?submit the accepted v5 plugin absolute path}"
: "${C2_EXPECTED_CANDIDATE_SHA:?submit the accepted v5 plugin SHA-256}"
: "${C2_CANDIDATE_SOURCE:?submit the accepted v5 source absolute path}"
[[ "${C2_EXPECTED_CANDIDATE_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ "${C2_CANDIDATE_PLUGIN}" == /* && "${C2_CANDIDATE_SOURCE}" == /* ]]
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
[[ -f "${C2_CANDIDATE_PLUGIN}" && -f "${C2_CANDIDATE_SOURCE}" && -f "${BASE_HARNESS}" && -f "${DIRECTED_HARNESS}" ]]
printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${C2_EXPECTED_CANDIDATE_SHA}" "${C2_CANDIDATE_PLUGIN}" \
  "${EXPECTED_SOURCE_SHA}" "${C2_CANDIDATE_SOURCE}" \
  "${EXPECTED_BASE_HARNESS_SHA}" "${BASE_HARNESS}" \
  "${EXPECTED_DIRECTED_HARNESS_SHA}" "${DIRECTED_HARNESS}" | sha256sum -c -
sha256sum "${C2_CANDIDATE_PLUGIN}" "${C2_CANDIDATE_SOURCE}" "${BASE_HARNESS}" "${DIRECTED_HARNESS}" > "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"

RESULT=${ARTIFACT_DIR}/v5-softmax-directed-job${SLURM_JOB_ID}.json
timeout --foreground --preserve-status --kill-after=10s 600s \
  "${PYTHON}" "${DIRECTED_HARNESS}" --library "${C2_CANDIDATE_PLUGIN}" \
  --base-harness "${BASE_HARNESS}" --sync-repeats 4 \
  --max-iteration-seconds 15 --atol 1e-4 --rtol 1e-3 > "${RESULT}"

"${PYTHON}" - "${RESULT}" "${C2_CANDIDATE_PLUGIN}" <<'PY'
import json
import math
import sys
from pathlib import Path

result_path, library = map(Path, sys.argv[1:])
payload = json.loads(result_path.read_text(encoding="utf-8"))
assert payload["schema"] == "c2-native-c2-v5-softmax-directed-v2"
assert payload["all_gates_pass"] is True
assert payload["no_monkeypatch"] is True
assert payload["dispatch"] == "direct_torch.ops._C.native_c2_msa_decode"
assert Path(payload["operator_library"]).resolve() == library.resolve()
assert payload["contract"]["producer_ctas"] == 4
assert payload["contract"]["pages_per_producer"] == 4
assert payload["contract"]["selected_pages"] == 16
assert payload["contract"]["page_size"] == 128
assert payload["rank_directed_case_count"] == 4
assert payload["all_invalid"]["all_gates_pass"] is True
assert payload["all_invalid"]["all_zero"] is True
assert payload["all_invalid"]["finite_output"] is True
assert payload["all_invalid"]["caller_output"]["pointer_unchanged"] is True
assert payload["all_invalid"]["caller_output"]["return_is_none"] is True
assert payload["all_invalid"]["stability"]["bitwise_repeatable"] is True
invalid_elapsed = payload["all_invalid"]["sync_watchdog"]["elapsed_seconds"]
assert len(invalid_elapsed) == 4
assert all(math.isfinite(float(value)) and 0.0 <= float(value) <= 15.0 for value in invalid_elapsed)
for expected_rank, case in enumerate(payload["rank_directed"]):
    assert case["producer_rank"] == expected_rank
    assert case["selected_page_indices"] == list(range(4 * expected_rank, 4 * expected_rank + 4))
    assert case["all_gates_pass"] is True
    assert case["oracle"]["allclose"] is True and case["oracle"]["finite_output"] is True
    assert case["rank_expected_value"]["allclose"] is True
    assert case["caller_output"]["pointer_unchanged"] is True and case["caller_output"]["return_is_none"] is True
    assert case["stability"]["bitwise_repeatable"] is True
    elapsed = case["sync_watchdog"]["elapsed_seconds"]
    assert len(elapsed) == 4
    assert all(math.isfinite(float(value)) and 0.0 <= float(value) <= 15.0 for value in elapsed)
assert payload["softmax_directed_case_count"] == 2
expected_softmax = (
    ("all-valid-online-rescale", 2048, False),
    ("mixed-causal-tail-37", 37, True),
)
for case, (label, sequence_length, mixed_causal_subgroup) in zip(
    payload["softmax_directed"], expected_softmax, strict=True
):
    assert case["case"] == label
    assert case["sequence_length"] == sequence_length
    assert case["mixed_causal_subgroup"] is mixed_causal_subgroup
    assert case["all_gates_pass"] is True
    assert case["oracle"]["allclose"] is True and case["oracle"]["finite_output"] is True
    assert case["caller_output"]["pointer_unchanged"] is True
    assert case["caller_output"]["return_is_none"] is True
    assert case["stability"]["bitwise_repeatable"] is True
    elapsed = case["sync_watchdog"]["elapsed_seconds"]
    assert len(elapsed) == 4
    assert all(math.isfinite(float(value)) and 0.0 <= float(value) <= 15.0 for value in elapsed)
PY

snapshot POST
apps_after=$(compute_apps) || { echo 'unable to query post-run compute apps' >&2; exit 1; }
[[ -z "${apps_after}" ]]
sha256sum "${RESULT}" "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
