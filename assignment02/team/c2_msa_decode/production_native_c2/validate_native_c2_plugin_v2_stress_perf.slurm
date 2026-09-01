#!/usr/bin/env bash
# Direct-plugin v2 gate.  It runs ABBA in four distinct Python processes:
# v1_a, v2_b1, v2_b2, v1_a2.  Every run has identical FP64-oracle workload
# arguments.  v2 may reach wheel packaging only if it is at least 5% faster by
# the pinned 16-value median-of-seed-means metric; no stable extension changes.

#SBATCH --job-name=c2-native-plugin-v2-stress
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v2-stress-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONOPTIMIZE=0
export PYTHONPATH=

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v2-stress-artifacts-20260829
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
BASE_HARNESS=/home/lcpu/85117379/native_c2_operator_bench_20260829.py
STRESS_HARNESS=/home/lcpu/85117379/native_c2_stress_perf_20260829.py
EXPECTED_BASE_HARNESS_SHA=0b51fede1e6258eea0b535acba2a86e371b284e9655352156cdeb3d310d90c28
EXPECTED_STRESS_HARNESS_SHA=0a929ead3a6d0513c8555ec74df543f2f69cc0ce0e3d84fffb0a924913420068
EXPECTED_V2_SOURCE_SHA=f2aae9c9c62e646b3cd38b136f224009d400298c21a99b9b1cf5a402f944acc4
EXPECTED_V1_PLUGIN_LIBRARY=/home/lcpu/85117379/c2-native-plugin-aot-artifacts-20260829/job11443/vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_V1_PLUGIN_SHA=f464b2b24eb6b8f9d7d45a187f44f43d4467f2c9106e87ad45ff86000478bf8d

: "${C2_V1_PLUGIN_LIBRARY:?submit the fixed v1 job11443 plugin path}"
: "${C2_EXPECTED_V1_PLUGIN_SHA:?submit the fixed v1 job11443 plugin SHA-256}"
: "${C2_PLUGIN_LIBRARY:?submit with v2 independent plugin shared library}"
: "${C2_PLUGIN_SOURCE:?submit with v2 patched native_c2_decode.cu}"
: "${C2_EXPECTED_PLUGIN_SHA:?submit with reviewed v2 plugin SHA-256}"
for c2_digest in "${C2_EXPECTED_V1_PLUGIN_SHA}" "${C2_EXPECTED_PLUGIN_SHA}"; do
  [[ "${c2_digest}" =~ ^[0-9a-f]{64}$ ]]
done
[[ "${C2_V1_PLUGIN_LIBRARY}" == "${EXPECTED_V1_PLUGIN_LIBRARY}" ]]
[[ "${C2_EXPECTED_V1_PLUGIN_SHA}" == "${EXPECTED_V1_PLUGIN_SHA}" ]]
[[ "${C2_V1_PLUGIN_LIBRARY}" == /* && "${C2_PLUGIN_LIBRARY}" == /* && "${C2_PLUGIN_SOURCE}" == /* ]]
[[ -d "${ARTIFACT_ROOT}" ]] || { echo "missing artifact root: ${ARTIFACT_ROOT}" >&2; exit 2; }
[[ ! -e "${ARTIFACT_DIR}" ]] || { echo "refusing to reuse ${ARTIFACT_DIR}" >&2; exit 2; }
mkdir -p "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1
finalize() {
  local rc=$?
  trap - EXIT
  set +e
  printf 'FINAL_RC=%s UTC=%s\n' "${rc}" "$(date -u +%FT%TZ)"
  printf 'FINAL_RC=%s UTC=%s\n' "${rc}" "$(date -u +%FT%TZ)" \
    > "${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt"
  exit "${rc}"
}
trap finalize EXIT
export PATH="/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin"

compute_apps() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}
snapshot() {
  printf '\n===== %s UTC=%s =====\n' "$1" "$(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap \
    --format=csv,noheader,nounits
  compute_apps || true
}

snapshot PRE
gpu_line=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* ]]
apps_before=$(compute_apps) || { echo "unable to query pre-run compute apps" >&2; exit 1; }
[[ -z "${apps_before}" ]]
[[ -f "${C2_V1_PLUGIN_LIBRARY}" && -f "${C2_PLUGIN_LIBRARY}" && -f "${C2_PLUGIN_SOURCE}" && -f "${BASE_HARNESS}" && -f "${STRESS_HARNESS}" ]]
printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${C2_EXPECTED_V1_PLUGIN_SHA}" "${C2_V1_PLUGIN_LIBRARY}" \
  "${C2_EXPECTED_PLUGIN_SHA}" "${C2_PLUGIN_LIBRARY}" \
  "${EXPECTED_V2_SOURCE_SHA}" "${C2_PLUGIN_SOURCE}" \
  "${EXPECTED_BASE_HARNESS_SHA}" "${BASE_HARNESS}" | sha256sum -c -
[[ "$(sha256sum "${STRESS_HARNESS}" | awk '{print $1}')" == "${EXPECTED_STRESS_HARNESS_SHA}" ]]
sha256sum "${C2_V1_PLUGIN_LIBRARY}" "${C2_PLUGIN_LIBRARY}" "${C2_PLUGIN_SOURCE}" "${BASE_HARNESS}" "${STRESS_HARNESS}" \
  | tee "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
"${PYTHON}" -c 'import platform, torch; print({"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda})'
ldd "${C2_V1_PLUGIN_LIBRARY}" | tee "${ARTIFACT_DIR}/v1-plugin-ldd-job${SLURM_JOB_ID}.txt"
ldd "${C2_PLUGIN_LIBRARY}" | tee "${ARTIFACT_DIR}/v2-plugin-ldd-job${SLURM_JOB_ID}.txt"

V1_A_RESULT="${ARTIFACT_DIR}/plugin-v1-job11443-a-stress-perf-job${SLURM_JOB_ID}.json"
V2_B1_RESULT="${ARTIFACT_DIR}/plugin-v2-b1-stress-perf-job${SLURM_JOB_ID}.json"
V2_B2_RESULT="${ARTIFACT_DIR}/plugin-v2-b2-stress-perf-job${SLURM_JOB_ID}.json"
V1_A2_RESULT="${ARTIFACT_DIR}/plugin-v1-job11443-a2-stress-perf-job${SLURM_JOB_ID}.json"
DECISION="${ARTIFACT_DIR}/v2-vs-v1-direct-stress-decision-job${SLURM_JOB_ID}.json"
run_stress_in_fresh_python() {
  local library=$1 output=$2
  timeout --foreground --preserve-status --kill-after=10s 2400s \
    "${PYTHON}" "${STRESS_HARNESS}" \
    --library "${library}" \
    --base-harness "${BASE_HARNESS}" \
    --seeds 17,23,42,2024,314159,20260801,20260815,20260829 \
    --oracle-dtype float64 --atol 1e-4 --rtol 1e-3 \
    --warmup 30 --iters 200 --stability-repeats 4 > "${output}"
}

# Every invocation execs a different Python process, so a DSO's TORCH_LIBRARY
# registration cannot be reused, while ABBA limits a one-direction drift bias.
run_stress_in_fresh_python "${C2_V1_PLUGIN_LIBRARY}" "${V1_A_RESULT}"
run_stress_in_fresh_python "${C2_PLUGIN_LIBRARY}" "${V2_B1_RESULT}"
run_stress_in_fresh_python "${C2_PLUGIN_LIBRARY}" "${V2_B2_RESULT}"
run_stress_in_fresh_python "${C2_V1_PLUGIN_LIBRARY}" "${V1_A2_RESULT}"

set +e
"${PYTHON}" - "${V1_A_RESULT}" "${V2_B1_RESULT}" "${V2_B2_RESULT}" "${V1_A2_RESULT}" \
  "${DECISION}" "${C2_V1_PLUGIN_LIBRARY}" "${C2_PLUGIN_LIBRARY}" \
  "${BASE_HARNESS}" "${STRESS_HARNESS}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

v1_a_path, v2_b1_path, v2_b2_path, v1_a2_path, decision_path = map(Path, sys.argv[1:6])
v1_library, v2_library, base_harness, stress_harness = map(Path, sys.argv[6:])

def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_payload(label, path, library):
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "c2-native-c2-stress-perf-v1", label
    assert payload["all_gates_pass"] is True, label
    assert payload["no_monkeypatch"] is True, label
    assert payload["dispatch"] == "direct_torch.ops._C.native_c2_msa_decode", label
    assert payload["correctness_gate"] == {"atol": 1e-4, "oracle_dtype": "float64", "rtol": 1e-3}, label
    assert payload["seed_count"] == 8, label
    assert [record["seed"] for record in payload["seeds"]] == [
        17, 23, 42, 2024, 314159, 20260801, 20260815, 20260829
    ], label
    assert Path(payload["operator_library"]).resolve() == library.resolve(), label
    means = []
    for record in payload["seeds"]:
        assert record["all_gates_pass"] is True, label
        assert record["correctness"]["allclose"] is True, label
        assert record["correctness"]["finite_output"] is True, label
        assert record["caller_output"]["pointer_unchanged"] is True, label
        assert record["caller_output"]["timing_pointer_unchanged"] is True, label
        assert record["caller_output"]["return_is_none"] is True, label
        assert record["stability"]["bitwise_repeatable"] is True, label
        assert record["stability"]["repeat_pointers_unchanged"] is True, label
        assert record["stability"]["repeat_returns_none"] is True, label
        mean_ms = float(record["latency"]["mean_ms"])
        assert math.isfinite(mean_ms) and mean_ms > 0.0, (label, mean_ms)
        means.append(mean_ms)
    return payload, means

v1_a, v1_a_means = check_payload("v1_a", v1_a_path, v1_library)
v2_b1, v2_b1_means = check_payload("v2_b1", v2_b1_path, v2_library)
v2_b2, v2_b2_means = check_payload("v2_b2", v2_b2_path, v2_library)
v1_a2, v1_a2_means = check_payload("v1_a2", v1_a2_path, v1_library)
contract_keys = ("schema", "base_harness", "contract", "correctness_gate", "device", "dispatch",
                 "measurement", "no_monkeypatch", "seed_count", "torch")
reference_contract = {key: v1_a[key] for key in contract_keys}
for label, payload in (("v2_b1", v2_b1), ("v2_b2", v2_b2), ("v1_a2", v1_a2)):
    assert {key: payload[key] for key in contract_keys} == reference_contract, label
assert Path(v1_a["base_harness"]).resolve() == base_harness.resolve()
assert Path(v2_b1["base_harness"]).resolve() == base_harness.resolve()
input_checksums = {
    "v1_job11443_plugin": file_sha256(v1_library),
    "v2_plugin": file_sha256(v2_library),
    "base_harness": file_sha256(base_harness),
    "stress_harness": file_sha256(stress_harness),
}
assert input_checksums["v1_job11443_plugin"] == "f464b2b24eb6b8f9d7d45a187f44f43d4467f2c9106e87ad45ff86000478bf8d"
assert input_checksums["v1_job11443_plugin"] != input_checksums["v2_plugin"]
v1_means = sorted(v1_a_means + v1_a2_means)
v2_means = sorted(v2_b1_means + v2_b2_means)
v1_median = 0.5 * (v1_means[7] + v1_means[8])
v2_median = 0.5 * (v2_means[7] + v2_means[8])
threshold = v1_median * 0.95
accepted = v2_median <= threshold
decision = {
    "schema": "c2-native-plugin-v2-vs-v1-direct-stress-promotion-v1",
    "reference": {"job": 11443, "library": str(v1_library.resolve()),
                  "median_of_16_seed_means_ms": v1_median, "seed_mean_ms": v1_means},
    "candidate": {"library": str(v2_library.resolve()),
                  "median_of_16_seed_means_ms": v2_median, "seed_mean_ms": v2_means},
    "shared_contract": {"seeds": [17, 23, 42, 2024, 314159, 20260801, 20260815, 20260829],
                        "oracle_dtype": "float64", "atol": 1e-4, "rtol": 1e-3,
                        "warmup": 30, "iters": 200, "stability_repeats": 4,
                        "comparison_processes": "ABBA: four distinct Python processes",
                        "run_order": ["v1_a", "v2_b1", "v2_b2", "v1_a2"],
                        "four_run_contract_identical": True},
    "input_checksums": input_checksums,
    "required_v2_max_ms": threshold,
    "relative_improvement_fraction": 1.0 - v2_median / v1_median,
    "required_improvement_fraction": 0.05,
    "all_correctness_and_comparability_gates_pass": True,
    "accepted": accepted,
    "rejection_policy": "If accepted is false, do not submit the v2 wheel job; retain v1.",
}
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(decision, sort_keys=True))
raise SystemExit(0 if accepted else 3)
PY
decision_rc=$?
set -e

snapshot POST
apps_after=$(compute_apps) || { echo "unable to query post-run compute apps" >&2; exit 1; }
[[ -z "${apps_after}" ]]
sha256sum "${V1_A_RESULT}" "${V2_B1_RESULT}" "${V2_B2_RESULT}" "${V1_A2_RESULT}" "${DECISION}" \
  "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" \
  > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
exit "${decision_rc}"
