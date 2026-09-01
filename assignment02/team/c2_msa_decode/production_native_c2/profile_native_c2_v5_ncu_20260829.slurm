#!/usr/bin/env bash
# One-kernel NCU mechanism diagnosis for v5 warp-parallel softmax.
# This is deliberately not a benchmark and cannot promote or reject v5: clean
# CUDA-event ABBA correctness/performance evidence makes that decision.  The
# sole historical mechanism comparator is frozen v4 NCU job11506.

#SBATCH --job-name=c2-native-v5-ncu
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v5-ncu-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONPATH=

ARTIFACT_ROOT=${C2_NCU_ARTIFACT_DIR:-/home/lcpu/85117379/c2-native-plugin-v5-ncu-artifacts-20260829}
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID}
PYTHON=${C2_PYTHON:-/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python}
HARNESS=${C2_BASE_HARNESS:-/home/lcpu/85117379/native_c2_operator_bench_20260829.py}
LIB=${C2_V5_PLUGIN:-/home/lcpu/85117379/c2-native-plugin-v5-aot-artifacts-20260829/job11760/vllm/_native_c2_msa_decode_plugin.abi3.so}
SOURCE=${C2_V5_SOURCE:-/home/lcpu/85117379/vllm-d4-native-c2-plugin-v5-warp-parallel-softmax-20260829-job11760/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu}
V4_SUMMARY=${C2_V4_NCU_SUMMARY:-/home/lcpu/85117379/c2-native-v4-ncu-artifacts-20260829/native-c2-ncu-job11506.json}

EXPECTED_PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
EXPECTED_HARNESS_SHA=0b51fede1e6258eea0b535acba2a86e371b284e9655352156cdeb3d310d90c28
EXPECTED_V5_PLUGIN=/home/lcpu/85117379/c2-native-plugin-v5-aot-artifacts-20260829/job11760/vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_V5_PLUGIN_SHA=9ab755faf79ec7bc574a8f5afa09b22b9acedc8257d3ccfad6fc98cc3f1d2538
EXPECTED_V5_SOURCE_SHA=6409a54689ebbdd1f72fad76c2d72dfe588f79d3b8dd6c4dbaf1df5b630f0931
EXPECTED_V4_SUMMARY=/home/lcpu/85117379/c2-native-v4-ncu-artifacts-20260829/native-c2-ncu-job11506.json
EXPECTED_V4_SUMMARY_SHA=1dc181ea69e771574949e192cffcd417831384ec6937371bcd6135aeafa22f4b

REPORT=${ARTIFACT_DIR}/native-c2-v5-job${SLURM_JOB_ID}.ncu-rep
RAW_CSV=${ARTIFACT_DIR}/native-c2-v5-job${SLURM_JOB_ID}.raw.csv
TARGET_LOG=${ARTIFACT_DIR}/native-c2-v5-target-job${SLURM_JOB_ID}.log
SUMMARY=${ARTIFACT_DIR}/native-c2-v5-ncu-job${SLURM_JOB_ID}.json
COMPARISON=${ARTIFACT_DIR}/native-c2-v5-v4-mechanism-job${SLURM_JOB_ID}.json
CATALOG=${ARTIFACT_DIR}/metric-catalog-job${SLURM_JOB_ID}.txt

# Keep this vector byte-for-byte semantically identical to the v4 job11506
# collection: same 14 counters, same seed/input contract and one-kernel filter.
metrics=(
  gpu__time_duration.sum
  dram__bytes_read.sum
  dram__bytes_write.sum
  lts__t_bytes.sum
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
  smsp__inst_executed_pipe_tensor.sum
  smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct
  smsp__average_warps_issue_stalled_wait_per_issue_active.pct
  smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
  smsp__warps_active.avg.per_cycle_active
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
)
metrics_csv="$(IFS=,; printf '%s' "${metrics[*]}")"

compute_apps() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}
snapshot() {
  printf '\n===== %s UTC=%s =====\n' "$1" "$(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap \
    --format=csv,noheader,nounits
  printf '%s\n' '-- compute apps --'; compute_apps || true
}
finish() {
  local rc=$?; trap - EXIT; snapshot POST || true
  printf 'FINAL_RC=%s UTC=%s\n' "${rc}" "$(date -u +%FT%TZ)" \
    | tee "${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt"
  exit "${rc}"
}
trap finish EXIT

[[ "${LIB}" == "${EXPECTED_V5_PLUGIN}" && "${V4_SUMMARY}" == "${EXPECTED_V4_SUMMARY}" ]]
[[ -d "${ARTIFACT_ROOT}" && ! -e "${ARTIFACT_DIR}" ]] || { echo 'artifact root missing or output reused' >&2; exit 2; }
mkdir -p "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1

snapshot PRE
gpu_line=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
apps_before=$(compute_apps) || { echo 'failed to query pre-run compute apps' >&2; exit 1; }
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* && -z "${apps_before}" ]]
[[ -f "${LIB}" && -f "${HARNESS}" && -f "${SOURCE}" && -f "${V4_SUMMARY}" ]]
[[ "$(readlink -f "${PYTHON}")" == "$(readlink -f "${EXPECTED_PYTHON}")" ]]
command -v ncu; ncu --version
printf '%s  %s\n' \
  "${EXPECTED_V5_PLUGIN_SHA}" "${LIB}" \
  "${EXPECTED_HARNESS_SHA}" "${HARNESS}" \
  "${EXPECTED_V5_SOURCE_SHA}" "${SOURCE}" \
  "${EXPECTED_V4_SUMMARY_SHA}" "${V4_SUMMARY}" | sha256sum -c -
sha256sum "${LIB}" "${HARNESS}" "${SOURCE}" "${V4_SUMMARY}" | tee "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"

"${PYTHON}" - "${EXPECTED_PYTHON}" "${ARTIFACT_DIR}/python-env-job${SLURM_JOB_ID}.json" <<'PY'
import json
from pathlib import Path
import sys
import torch

if not __debug__:
    raise SystemExit("Python optimization unexpectedly disabled evidence gates")
invocation, output = map(Path, sys.argv[1:])
venv_root = invocation.parent.parent.resolve()
expected = invocation.resolve(); actual = Path(sys.executable).resolve()
torch_file = Path(torch.__file__).resolve()
if actual != expected or Path(sys.prefix).resolve() != venv_root or venv_root not in torch_file.parents:
    raise SystemExit("unexpected pinned Python/Torch environment")
if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
    raise SystemExit(f"unexpected torch/CUDA build: {torch.__version__} / {torch.version.cuda}")
payload = {"schema": "c2-native-c2-ncu-python-environment-v1", "python": str(actual),
           "python_invocation": str(invocation), "python_debug": __debug__,
           "python_prefix": str(Path(sys.prefix).resolve()), "torch_file": str(torch_file),
           "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

grep_args=(); for metric in "${metrics[@]}"; do grep_args+=(--regexp="${metric}"); done
timeout --signal=TERM --kill-after=15s 180s ncu --query-metrics --query-metrics-mode all \
  | grep --fixed-strings "${grep_args[@]}" > "${CATALOG}"
"${PYTHON}" - "${CATALOG}" "${metrics[@]}" <<'PY'
import re, sys
from pathlib import Path
catalog = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
available = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*(?:__[A-Za-z0-9_]+)+(?:\.[A-Za-z0-9_]+)+\b", catalog))
missing = [metric for metric in sys.argv[2:] if metric not in available]
if missing: raise SystemExit("missing exact NCU metrics: " + ", ".join(missing))
PY

timeout --signal=TERM --kill-after=30s 300s ncu --target-processes all \
  --kernel-name-base function --kernel-name 'regex:.*native_c2_msa_decode_kernel.*' \
  --launch-skip 0 --launch-count 1 --clock-control none --metrics "${metrics_csv}" \
  --force-overwrite --export "${REPORT}" \
  "${PYTHON}" "${HARNESS}" --library "${LIB}" --seed 20260829 \
    --oracle-dtype float64 --atol 1e-4 --rtol 1e-3 > "${TARGET_LOG}" 2>&1
[[ -s "${REPORT}" && -s "${TARGET_LOG}" ]]

"${PYTHON}" - "${TARGET_LOG}" "${LIB}" <<'PY'
import json, sys
from pathlib import Path
records = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
    try: value = json.loads(line)
    except json.JSONDecodeError: continue
    if isinstance(value, dict) and value.get("schema") == "c2-native-c2-direct-operator-oracle-v1": records.append(value)
if len(records) != 1: raise SystemExit(f"expected one direct-operator record, got {len(records)}")
record = records[0]
if Path(record.get("operator_library", "")).resolve() != Path(sys.argv[2]).resolve(): raise SystemExit("unexpected operator library")
expected = {"batch": 16, "head_dim": 128, "kv_heads": 4, "max_logical_pages": 32,
            "num_physical_pages": 64, "page_size": 128, "q_heads": 64, "topk": 16}
for key, value in expected.items():
    if record.get("contract", {}).get(key) != value:
        raise SystemExit(f"unexpected NCU input contract {key}: {record.get('contract', {}).get(key)}")
checks = {"all_gates_pass": record.get("all_gates_pass") is True, "seed": record.get("seed") == 20260829,
          "device_capability": record.get("device_capability") == [10, 3],
          "direct_dispatch": record.get("dispatch") == "direct_torch.ops._C.native_c2_msa_decode",
          "no_monkeypatch": record.get("no_monkeypatch") is True,
          "caller_pointer": record.get("caller_output", {}).get("pointer_unchanged") is True,
          "return_is_none": record.get("caller_output", {}).get("return_is_none") is True,
          "oracle_dtype": record.get("correctness", {}).get("oracle_dtype") == "float64",
          "atol": record.get("correctness", {}).get("atol") == 1e-4,
          "rtol": record.get("correctness", {}).get("rtol") == 1e-3}
failed = sorted(name for name, passed in checks.items() if not passed)
if failed: raise SystemExit("failed target-log gates: " + ", ".join(failed))
PY

timeout --signal=TERM --kill-after=15s 120s ncu --import "${REPORT}" --csv --page raw --print-units base --print-fp > "${RAW_CSV}"
[[ -s "${RAW_CSV}" ]]

"${PYTHON}" - "${RAW_CSV}" "${SUMMARY}" "${metrics[@]}" <<'PY'
import csv, json, math, sys
from collections import Counter
from pathlib import Path
csv_path, output_path, *required = sys.argv[1:]
if len(required) != 14 or len(set(required)) != 14: raise SystemExit("expected 14 unique required metrics")
rows = list(csv.reader(Path(csv_path).open(encoding="utf-8", errors="replace", newline="")))
header_index = next((i for i, row in enumerate(rows) if "ID" in row and "Kernel Name" in row and all(m in row for m in required)), None)
if header_index is None: raise SystemExit("raw CSV lacks one wide header containing all requested metrics")
header = rows[header_index]; counts = Counter(header)
for column in ("ID", "Kernel Name", *required):
    if counts[column] != 1: raise SystemExit(f"required column {column!r} count is {counts[column]}")
indices = {name: header.index(name) for name in ("ID", "Kernel Name", *required)}; last_index = max(indices.values())
if header_index + 1 >= len(rows) or len(rows[header_index + 1]) <= last_index: raise SystemExit("raw CSV lacks a complete units row")
unit_row = rows[header_index + 1]; units = {m: unit_row[indices[m]].strip() for m in required}
for m in ("dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes.sum"):
    if units[m] != "byte": raise SystemExit(f"{m} is not expressed in base bytes: {units[m]!r}")
kernel_rows = [row for row in rows[header_index + 2:] if len(row) > last_index and "native_c2_msa_decode_kernel" in row[indices["Kernel Name"]]]
if len(kernel_rows) != 1: raise SystemExit(f"expected exactly one native kernel action row, got {len(kernel_rows)}")
row = kernel_rows[0]; values = {}
for m in required:
    raw = row[indices[m]].strip().replace(",", ""); value = float(raw)
    if not math.isfinite(value) or value < 0.0: raise SystemExit(f"invalid {m}: {raw!r}")
    values[m] = value
if values["smsp__inst_executed_pipe_tensor.sum"] <= 0.0: raise SystemExit("target kernel recorded no tensor-pipe instructions")
optional_launch = ("launch__cluster_size", "launch__cluster_max_potential_size", "launch__cluster_max_active", "launch__occupancy_cluster_pct", "launch__occupancy_cluster_gpu_pct", "launch__registers_per_thread", "launch__shared_mem_per_block", "launch__barrier_count")
launch = {name: row[header.index(name)].strip() for name in optional_launch if counts[name] == 1 and len(row) > header.index(name)}
payload = {"schema": "c2-native-c2-ncu-counters-v1", "kernel_name": row[indices["Kernel Name"]],
           "action_id": row[indices["ID"]].strip(), "matching_kernel_rows": 1,
           "metrics": {m: {"value": values[m], "unit": units[m]} for m in required},
           "launch_metadata_if_reported": launch, "timing_valid_for_benchmark": False,
           "timing_note": "NCU replay and instrumentation invalidate benchmark timing; use separately collected clean non-NCU CUDA-event ABBA.",
           "all_gates_pass": True}
Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

# The comparison is evidence, not a threshold gate.  It keeps the v4 reference
# immutable and records all counter ratios so a regression remains visible
# without wrongly overriding the independent clean-performance decision.
"${PYTHON}" - "${V4_SUMMARY}" "${SUMMARY}" "${COMPARISON}" \
  "${EXPECTED_V4_SUMMARY_SHA}" "${EXPECTED_V5_PLUGIN_SHA}" "${EXPECTED_V5_SOURCE_SHA}" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
v4_path, v5_path, output_path = map(Path, sys.argv[1:4])
expected_v4_sha, expected_v5_plugin_sha, expected_v5_source_sha = sys.argv[4:]
if hashlib.sha256(v4_path.read_bytes()).hexdigest() != expected_v4_sha: raise SystemExit("v4 job11506 summary checksum changed")
v4 = json.loads(v4_path.read_text(encoding="utf-8")); v5 = json.loads(v5_path.read_text(encoding="utf-8"))
for label, payload in (("v4", v4), ("v5", v5)):
    if payload.get("schema") != "c2-native-c2-ncu-counters-v1" or payload.get("all_gates_pass") is not True:
        raise SystemExit(f"invalid {label} counter summary")
    if payload.get("matching_kernel_rows") != 1 or payload.get("timing_valid_for_benchmark") is not False:
        raise SystemExit(f"invalid {label} one-kernel/timing contract")
v4_metrics, v5_metrics = v4["metrics"], v5["metrics"]
if set(v4_metrics) != set(v5_metrics) or len(v4_metrics) != 14: raise SystemExit("v4/v5 metric set mismatch")
ratios = {}; deltas = {}
for metric in sorted(v4_metrics):
    old, new = float(v4_metrics[metric]["value"]), float(v5_metrics[metric]["value"])
    if v4_metrics[metric]["unit"] != v5_metrics[metric]["unit"]: raise SystemExit(f"unit mismatch: {metric}")
    if not math.isfinite(old) or not math.isfinite(new) or old < 0 or new < 0: raise SystemExit(f"invalid counter: {metric}")
    ratios[metric] = None if old == 0.0 else new / old
    deltas[metric] = new - old
def decreased(metric):
    ratio = ratios[metric]
    return None if ratio is None else ratio < 1.0
launch_v4, launch_v5 = v4.get("launch_metadata_if_reported", {}), v5.get("launch_metadata_if_reported", {})
payload = {
  "schema": "c2-native-c2-v5-v4-ncu-mechanism-comparison-v1",
  "reference": {"version": "v4", "job": 11506, "summary": str(v4_path.resolve()), "summary_sha256": expected_v4_sha,
                "kernel_name": v4["kernel_name"], "action_id": v4["action_id"]},
  "candidate": {"version": "v5", "summary": str(v5_path.resolve()), "plugin_sha256": expected_v5_plugin_sha,
                "source_sha256": expected_v5_source_sha, "kernel_name": v5["kernel_name"], "action_id": v5["action_id"]},
  "same_collection_contract": {"seed": 20260829, "same_metric_set": True, "same_kernel_filter": "regex:.*native_c2_msa_decode_kernel.*", "single_kernel_action_each": True},
  "v5_counter_values": {m: v5_metrics[m] for m in sorted(v5_metrics)},
  "v5_minus_v4": deltas, "v5_over_v4": ratios,
  "launch_metadata": {"v4_job11506": launch_v4, "v5": launch_v5,
                      "expected_v4_registers_per_thread": 58, "expected_v5_registers_per_thread": 64,
                      "v5_aot_resource_expectation": {"stack": 0, "shared_bytes": 38752, "local": 0}},
  "mechanism_observations": {
      "tensor_instruction_count_unchanged": ratios["smsp__inst_executed_pipe_tensor.sum"] == 1.0,
      "shared_load_bank_conflicts_decreased": decreased("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"),
      "shared_store_bank_conflicts_decreased": decreased("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"),
      "shared_load_wavefronts_decreased": decreased("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
      "shared_store_wavefronts_decreased": decreased("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"),
      "barrier_stall_decreased": decreased("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"),
      "wait_stall_decreased": decreased("smsp__average_warps_issue_stalled_wait_per_issue_active.pct"),
      "external_traffic_counters": ["dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes.sum"],
  },
  "decision_boundary": "NCU replay counters are mechanism evidence only. No aggregate counter direction, including shared wavefronts, is a v5 promotion/rejection gate; FP64 correctness and clean non-NCU ABBA performance make that decision.",
  "all_collection_and_comparison_gates_pass": True,
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

printf '%s  %s\n' \
  "${EXPECTED_V5_PLUGIN_SHA}" "${LIB}" \
  "${EXPECTED_HARNESS_SHA}" "${HARNESS}" \
  "${EXPECTED_V5_SOURCE_SHA}" "${SOURCE}" \
  "${EXPECTED_V4_SUMMARY_SHA}" "${V4_SUMMARY}" | sha256sum -c -
sha256sum "${REPORT}" "${RAW_CSV}" "${TARGET_LOG}" "${SUMMARY}" "${COMPARISON}" "${CATALOG}" \
  "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" "${ARTIFACT_DIR}/python-env-job${SLURM_JOB_ID}.json" \
  > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
apps_after=$(compute_apps) || { echo 'failed to query post-run compute apps' >&2; exit 1; }
[[ -z "${apps_after}" ]]
