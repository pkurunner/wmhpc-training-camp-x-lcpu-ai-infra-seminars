#!/usr/bin/env bash
# Direct-plugin v5 promotion gate.  Freeze job11496's promoted v4 DSO as the
# reference and compare warp-parallel-softmax v5 through an ABBA sequence in
# four distinct Python processes.  v5 is promotable only when its
# 16-seed-median latency is <= 95% of v4.

#SBATCH --job-name=c2-native-plugin-v5-stress
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v5-stress-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONPATH=

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v5-stress-artifacts-20260829
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
BASE_HARNESS=/home/lcpu/85117379/native_c2_operator_bench_20260829.py
STRESS_HARNESS=/home/lcpu/85117379/native_c2_stress_perf_20260829.py
EXPECTED_BASE_HARNESS_SHA=0b51fede1e6258eea0b535acba2a86e371b284e9655352156cdeb3d310d90c28
EXPECTED_STRESS_HARNESS_SHA=0a929ead3a6d0513c8555ec74df543f2f69cc0ce0e3d84fffb0a924913420068
EXPECTED_V5_SOURCE_SHA=6409a54689ebbdd1f72fad76c2d72dfe588f79d3b8dd6c4dbaf1df5b630f0931
EXPECTED_REFERENCE_LIBRARY=/home/lcpu/85117379/c2-native-plugin-v4-aot-artifacts-20260829/job11496/vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_REFERENCE_SHA=e00877088ac94bc17d7c501dff418561f195f6e349af46f532ead0a6aa3763d5

: "${C2_REFERENCE_PLUGIN:?submit the fixed v4 job11496 plugin path}"
: "${C2_EXPECTED_REFERENCE_SHA:?submit the fixed v4 job11496 plugin SHA-256}"
: "${C2_CANDIDATE_PLUGIN:?submit the v5 independent plugin shared library}"
: "${C2_CANDIDATE_SOURCE:?submit the v5 patched native_c2_decode.cu}"
: "${C2_EXPECTED_CANDIDATE_SHA:?submit the reviewed v5 plugin SHA-256}"
for digest in "${C2_EXPECTED_REFERENCE_SHA}" "${C2_EXPECTED_CANDIDATE_SHA}"; do [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]; done
[[ "${C2_REFERENCE_PLUGIN}" == "${EXPECTED_REFERENCE_LIBRARY}" && "${C2_EXPECTED_REFERENCE_SHA}" == "${EXPECTED_REFERENCE_SHA}" ]]
[[ "${C2_REFERENCE_PLUGIN}" == /* && "${C2_CANDIDATE_PLUGIN}" == /* && "${C2_CANDIDATE_SOURCE}" == /* ]]
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
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* ]]
[[ -z "${apps_before}" ]]
[[ -f "${C2_REFERENCE_PLUGIN}" && -f "${C2_CANDIDATE_PLUGIN}" && -f "${C2_CANDIDATE_SOURCE}" && -f "${BASE_HARNESS}" && -f "${STRESS_HARNESS}" ]]
printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${C2_EXPECTED_REFERENCE_SHA}" "${C2_REFERENCE_PLUGIN}" \
  "${C2_EXPECTED_CANDIDATE_SHA}" "${C2_CANDIDATE_PLUGIN}" \
  "${EXPECTED_V5_SOURCE_SHA}" "${C2_CANDIDATE_SOURCE}" \
  "${EXPECTED_BASE_HARNESS_SHA}" "${BASE_HARNESS}" | sha256sum -c -
[[ "$(sha256sum "${STRESS_HARNESS}" | awk '{print $1}')" == "${EXPECTED_STRESS_HARNESS_SHA}" ]]
sha256sum "${C2_REFERENCE_PLUGIN}" "${C2_CANDIDATE_PLUGIN}" "${C2_CANDIDATE_SOURCE}" "${BASE_HARNESS}" "${STRESS_HARNESS}" | tee "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
"${PYTHON}" -c 'import platform, torch; print({"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda})'
ldd "${C2_REFERENCE_PLUGIN}" | tee "${ARTIFACT_DIR}/reference-v4-plugin-ldd-job${SLURM_JOB_ID}.txt"
ldd "${C2_CANDIDATE_PLUGIN}" | tee "${ARTIFACT_DIR}/candidate-v5-plugin-ldd-job${SLURM_JOB_ID}.txt"

REF_A=${ARTIFACT_DIR}/reference-v4-a-stress-perf-job${SLURM_JOB_ID}.json
CAN_B1=${ARTIFACT_DIR}/candidate-v5-b1-stress-perf-job${SLURM_JOB_ID}.json
CAN_B2=${ARTIFACT_DIR}/candidate-v5-b2-stress-perf-job${SLURM_JOB_ID}.json
REF_A2=${ARTIFACT_DIR}/reference-v4-a2-stress-perf-job${SLURM_JOB_ID}.json
DECISION=${ARTIFACT_DIR}/v5-vs-v4-direct-stress-decision-job${SLURM_JOB_ID}.json
run_fresh() {
  timeout --foreground --preserve-status --kill-after=10s 2400s "${PYTHON}" "${STRESS_HARNESS}" \
    --library "$1" --base-harness "${BASE_HARNESS}" \
    --seeds 17,23,42,2024,314159,20260801,20260815,20260829 \
    --oracle-dtype float64 --atol 1e-4 --rtol 1e-3 --warmup 30 --iters 200 --stability-repeats 4 > "$2"
}
# Each call is a separate Python process, preventing TORCH_LIBRARY reuse.
run_fresh "${C2_REFERENCE_PLUGIN}" "${REF_A}"
run_fresh "${C2_CANDIDATE_PLUGIN}" "${CAN_B1}"
run_fresh "${C2_CANDIDATE_PLUGIN}" "${CAN_B2}"
run_fresh "${C2_REFERENCE_PLUGIN}" "${REF_A2}"

set +e
"${PYTHON}" - "${REF_A}" "${CAN_B1}" "${CAN_B2}" "${REF_A2}" "${DECISION}" \
  "${C2_REFERENCE_PLUGIN}" "${C2_CANDIDATE_PLUGIN}" "${C2_CANDIDATE_SOURCE}" "${BASE_HARNESS}" "${STRESS_HARNESS}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

ref_a_path, can_b1_path, can_b2_path, ref_a2_path, decision_path = map(Path, sys.argv[1:6])
ref_library, can_library, candidate_source, base_harness, stress_harness = map(Path, sys.argv[6:])
SEEDS = [17, 23, 42, 2024, 314159, 20260801, 20260815, 20260829]

def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check(label, path, library):
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "c2-native-c2-stress-perf-v1", label
    assert payload["all_gates_pass"] is True and payload["no_monkeypatch"] is True, label
    assert payload["dispatch"] == "direct_torch.ops._C.native_c2_msa_decode", label
    assert payload["correctness_gate"] == {"atol": 1e-4, "oracle_dtype": "float64", "rtol": 1e-3}, label
    assert payload["seed_count"] == 8 and [x["seed"] for x in payload["seeds"]] == SEEDS, label
    assert Path(payload["operator_library"]).resolve() == library.resolve(), label
    means = []
    for record in payload["seeds"]:
        assert record["all_gates_pass"] is True, label
        assert record["correctness"]["allclose"] is True and record["correctness"]["finite_output"] is True, label
        assert record["caller_output"]["pointer_unchanged"] is True, label
        assert record["caller_output"]["timing_pointer_unchanged"] is True, label
        assert record["caller_output"]["return_is_none"] is True, label
        assert record["stability"]["bitwise_repeatable"] is True, label
        assert record["stability"]["repeat_pointers_unchanged"] is True, label
        assert record["stability"]["repeat_returns_none"] is True, label
        value = float(record["latency"]["mean_ms"])
        assert math.isfinite(value) and value > 0.0, (label, value)
        means.append(value)
    return payload, means

ref_a, ref_a_means = check("reference_v4_a", ref_a_path, ref_library)
can_b1, can_b1_means = check("candidate_v5_b1", can_b1_path, can_library)
can_b2, can_b2_means = check("candidate_v5_b2", can_b2_path, can_library)
ref_a2, ref_a2_means = check("reference_v4_a2", ref_a2_path, ref_library)
contract_keys = ("schema", "base_harness", "contract", "correctness_gate", "device", "dispatch", "measurement", "no_monkeypatch", "seed_count", "torch")
reference_contract = {key: ref_a[key] for key in contract_keys}
for label, payload in (("candidate_v5_b1", can_b1), ("candidate_v5_b2", can_b2), ("reference_v4_a2", ref_a2)):
    assert {key: payload[key] for key in contract_keys} == reference_contract, label
assert Path(ref_a["base_harness"]).resolve() == base_harness.resolve()
assert Path(can_b1["base_harness"]).resolve() == base_harness.resolve()
inputs = {"reference_v4_job11496_plugin": checksum(ref_library), "candidate_v5_plugin": checksum(can_library), "candidate_v5_source": checksum(candidate_source), "base_harness": checksum(base_harness), "stress_harness": checksum(stress_harness)}
assert inputs["reference_v4_job11496_plugin"] == "e00877088ac94bc17d7c501dff418561f195f6e349af46f532ead0a6aa3763d5"
assert inputs["reference_v4_job11496_plugin"] != inputs["candidate_v5_plugin"]
reference_means = sorted(ref_a_means + ref_a2_means)
candidate_means = sorted(can_b1_means + can_b2_means)
reference_median = (reference_means[7] + reference_means[8]) / 2.0
candidate_median = (candidate_means[7] + candidate_means[8]) / 2.0
threshold = reference_median * 0.95
accepted = candidate_median <= threshold
decision = {
    "schema": "c2-native-plugin-v5-vs-v4-direct-stress-promotion-v1",
    "reference": {"job": 11496, "library": str(ref_library.resolve()), "median_of_16_seed_means_ms": reference_median, "seed_mean_ms": reference_means},
    "candidate": {"library": str(can_library.resolve()), "median_of_16_seed_means_ms": candidate_median, "seed_mean_ms": candidate_means},
    "shared_contract": {"seeds": SEEDS, "oracle_dtype": "float64", "atol": 1e-4, "rtol": 1e-3, "warmup": 30, "iters": 200, "stability_repeats": 4, "comparison_processes": "ABBA: four distinct Python processes", "run_order": ["reference_v4_a", "candidate_v5_b1", "candidate_v5_b2", "reference_v4_a2"], "four_run_contract_identical": True},
    "input_checksums": inputs, "required_v5_max_ms": threshold,
    "relative_improvement_fraction": 1.0 - candidate_median / reference_median,
    "required_improvement_fraction": 0.05, "all_correctness_and_comparability_gates_pass": True,
    "accepted": accepted, "rejection_policy": "If accepted is false, do not submit a v5 wheel job; retain v4."}
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(decision, sort_keys=True))
raise SystemExit(0 if accepted else 3)
PY
decision_rc=$?
set -e
snapshot POST
apps_after=$(compute_apps) || { echo 'unable to query post-run compute apps' >&2; exit 1; }
[[ -z "${apps_after}" ]]
sha256sum "${REF_A}" "${CAN_B1}" "${CAN_B2}" "${REF_A2}" "${DECISION}" "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
exit "${decision_rc}"

