#!/usr/bin/env bash
# One-kernel Nsight Compute diagnosis for the production-facing native C2 op.
# NCU replay timing is mechanism evidence only; CUDA-event ABBA remains the
# performance scoreboard.
#SBATCH --job-name=c2-native-ncu
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/lcpu/85117379/c2-native-ncu-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONOPTIMIZE=0
export PYTHONPATH=

ARTIFACT_DIR=${C2_NCU_ARTIFACT_DIR:-/home/lcpu/85117379/c2-native-ncu-artifacts-20260829}
PYTHON=${C2_PYTHON:-/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python}
LIB=${C2_LIB:-/home/lcpu/85117379/vllm-d4-native-c2-build-20260829/_C_stable_libtorch.abi3.so}
HARNESS=${C2_BASE_HARNESS:-/home/lcpu/85117379/native_c2_operator_bench_20260829.py}
SOURCE=${C2_NATIVE_SOURCE:-/home/lcpu/85117379/vllm-d4-native-c2-20260829/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu}
EXPECTED_PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
EXPECTED_LIB_SHA=${C2_EXPECTED_LIB_SHA:-beed8c557858da9bdd9e5f4c3681a04c29c8435dd92994f43c9935ce715ed064}
EXPECTED_HARNESS_SHA=${C2_EXPECTED_HARNESS_SHA:-0b51fede1e6258eea0b535acba2a86e371b284e9655352156cdeb3d310d90c28}
EXPECTED_SOURCE_SHA=${C2_EXPECTED_SOURCE_SHA:-f67495586759b3b8f718dbbf60627c499f917842f2ed6ecf89d2344f1c82bc4b}
REPORT="${ARTIFACT_DIR}/native-c2-job${SLURM_JOB_ID}.ncu-rep"
RAW_CSV="${ARTIFACT_DIR}/native-c2-job${SLURM_JOB_ID}.raw.csv"
TARGET_LOG="${ARTIFACT_DIR}/native-c2-target-job${SLURM_JOB_ID}.log"
SUMMARY="${ARTIFACT_DIR}/native-c2-ncu-job${SLURM_JOB_ID}.json"
CATALOG="${ARTIFACT_DIR}/metric-catalog-job${SLURM_JOB_ID}.txt"

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

mkdir -p "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1

compute_apps() {
  local raw
  raw=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null) || return 1
  printf '%s\n' "${raw}" | sed '/^[[:space:]]*$/d'
}

snapshot() {
  local label=$1
  printf '\n===== %s UTC=%s =====\n' "${label}" "$(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap \
    --format=csv,noheader,nounits
  printf '%s\n' '-- compute apps --'
  compute_apps || true
}

finish() {
  local rc=$?
  local final_line
  trap - EXIT
  snapshot POST || true
  final_line=$(printf 'FINAL_RC=%s UTC=%s' "${rc}" "$(date -u +%FT%TZ)")
  printf '%s\n' "${final_line}"
  if ! printf '%s\n' "${final_line}" \
    > "${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt"; then
    printf 'failed to write job-scoped final status\n' >&2
  fi
  exit "${rc}"
}
trap finish EXIT

snapshot PRE
gpu_line=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* ]]
apps_before=$(compute_apps) || { echo "failed to query pre-run compute apps" >&2; exit 1; }
[[ -z "${apps_before}" ]]
[[ -f "${LIB}" && -f "${HARNESS}" && -f "${SOURCE}" ]]
[[ "$(readlink -f "${PYTHON}")" == "$(readlink -f "${EXPECTED_PYTHON}")" ]]
command -v ncu
ncu --version
printf '%s  %s\n' \
  "${EXPECTED_LIB_SHA}" "${LIB}" \
  "${EXPECTED_HARNESS_SHA}" "${HARNESS}" \
  "${EXPECTED_SOURCE_SHA}" "${SOURCE}" \
  | sha256sum -c -
sha256sum "${LIB}" "${HARNESS}" "${SOURCE}" \
  | tee "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
"${PYTHON}" - "${EXPECTED_PYTHON}" \
  "${ARTIFACT_DIR}/python-env-job${SLURM_JOB_ID}.json" <<'PY'
import json
from pathlib import Path
import sys
import torch

if not __debug__:
    raise SystemExit("Python optimization unexpectedly disabled evidence gates")
invocation = Path(sys.argv[1])
venv_root = invocation.parent.parent.resolve()
expected = invocation.resolve()
output = Path(sys.argv[2])
actual = Path(sys.executable).resolve()
torch_file = Path(torch.__file__).resolve()
if actual != expected:
    raise SystemExit(f"unexpected Python executable: {actual} != {expected}")
if Path(sys.prefix).resolve() != venv_root:
    raise SystemExit(f"unexpected sys.prefix: {sys.prefix} != {venv_root}")
if venv_root not in torch_file.parents:
    raise SystemExit(f"torch is outside pinned venv: {torch_file}")
if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
    raise SystemExit(
        f"unexpected torch/CUDA build: {torch.__version__} / {torch.version.cuda}"
    )
payload = {
    "schema": "c2-native-c2-ncu-python-environment-v1",
    "python": str(actual),
    "python_invocation": str(invocation),
    "python_debug": __debug__,
    "python_prefix": str(Path(sys.prefix).resolve()),
    "torch_file": str(torch_file),
    "torch_version": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

grep_args=()
for metric in "${metrics[@]}"; do
  grep_args+=(--regexp="${metric}")
done
timeout --signal=TERM --kill-after=15s 180s \
  ncu --query-metrics --query-metrics-mode all \
  | grep --fixed-strings "${grep_args[@]}" > "${CATALOG}"
"${PYTHON}" - "${CATALOG}" "${metrics[@]}" <<'PY'
import re
import sys
from pathlib import Path

catalog = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
available = set(re.findall(
    r"\b[A-Za-z][A-Za-z0-9_]*(?:__[A-Za-z0-9_]+)+(?:\.[A-Za-z0-9_]+)+\b",
    catalog,
))
missing = [metric for metric in sys.argv[2:] if metric not in available]
if missing:
    raise SystemExit("missing exact NCU metrics: " + ", ".join(missing))
PY

timeout --signal=TERM --kill-after=30s 300s \
  ncu --target-processes all \
    --kernel-name-base function \
    --kernel-name 'regex:.*native_c2_msa_decode_kernel.*' \
    --launch-skip 0 --launch-count 1 \
    --clock-control none \
    --metrics "${metrics_csv}" \
    --force-overwrite --export "${REPORT}" \
    "${PYTHON}" "${HARNESS}" --library "${LIB}" --seed 20260829 \
      --oracle-dtype float64 --atol 1e-4 --rtol 1e-3 \
  > "${TARGET_LOG}" 2>&1
[[ -s "${REPORT}" && -s "${TARGET_LOG}" ]]

"${PYTHON}" - "${TARGET_LOG}" "${LIB}" <<'PY'
import json
import sys
from pathlib import Path

records = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(value, dict) and value.get("schema") == "c2-native-c2-direct-operator-oracle-v1":
        records.append(value)
if len(records) != 1:
    raise SystemExit(f"expected one direct-operator record, got {len(records)}")
record = records[0]
expected_library = Path(sys.argv[2]).resolve()
reported_library = Path(record.get("operator_library", "")).resolve()
if reported_library != expected_library:
    raise SystemExit(
        f"unexpected operator library: {reported_library} != {expected_library}"
    )
expected_contract = {
    "batch": 16,
    "head_dim": 128,
    "kv_heads": 4,
    "max_logical_pages": 32,
    "num_physical_pages": 64,
    "page_size": 128,
    "q_heads": 64,
    "topk": 16,
}
for key, value in expected_contract.items():
    if record.get("contract", {}).get(key) != value:
        raise SystemExit(f"unexpected contract {key}: {record.get('contract', {}).get(key)}")
checks = {
    "all_gates_pass": record.get("all_gates_pass") is True,
    "seed": record.get("seed") == 20260829,
    "device_capability": record.get("device_capability") == [10, 3],
    "direct_dispatch": record.get("dispatch") == "direct_torch.ops._C.native_c2_msa_decode",
    "no_monkeypatch": record.get("no_monkeypatch") is True,
    "caller_pointer": record.get("caller_output", {}).get("pointer_unchanged") is True,
    "return_is_none": record.get("caller_output", {}).get("return_is_none") is True,
    "oracle_dtype": record.get("correctness", {}).get("oracle_dtype") == "float64",
    "atol": record.get("correctness", {}).get("atol") == 1.0e-4,
    "rtol": record.get("correctness", {}).get("rtol") == 1.0e-3,
}
failed = sorted(name for name, passed in checks.items() if not passed)
if failed:
    raise SystemExit("failed target-log gates: " + ", ".join(failed))
PY

timeout --signal=TERM --kill-after=15s 120s \
  ncu --import "${REPORT}" --csv --page raw --print-units base --print-fp \
  > "${RAW_CSV}"
[[ -s "${RAW_CSV}" ]]

"${PYTHON}" - "${RAW_CSV}" "${SUMMARY}" "${metrics[@]}" <<'PY'
import csv
from collections import Counter
import json
import math
import sys
from pathlib import Path

csv_path, output_path, *required = sys.argv[1:]
if len(required) != 14 or len(set(required)) != 14:
    raise SystemExit("expected 14 unique required metrics")
rows = list(csv.reader(Path(csv_path).open(
    encoding="utf-8", errors="replace", newline="")))
header_index = next((
    index for index, row in enumerate(rows)
    if "ID" in row and "Kernel Name" in row and all(metric in row for metric in required)
), None)
if header_index is None:
    raise SystemExit("raw CSV lacks one wide header containing all requested metrics")
header = rows[header_index]
counts = Counter(header)
for column in ("ID", "Kernel Name", *required):
    if counts[column] != 1:
        raise SystemExit(f"required column {column!r} count is {counts[column]}")
indices = {name: header.index(name) for name in ("ID", "Kernel Name", *required)}
last_index = max(indices.values())
if header_index + 1 >= len(rows) or len(rows[header_index + 1]) <= last_index:
    raise SystemExit("raw CSV lacks a complete units row")
unit_row = rows[header_index + 1]
units = {metric: unit_row[indices[metric]].strip() for metric in required}
for metric in ("dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes.sum"):
    if units[metric] != "byte":
        raise SystemExit(f"{metric} is not expressed in base bytes: {units[metric]!r}")
kernel_rows = [
    row for row in rows[header_index + 2:]
    if len(row) > last_index
    and "native_c2_msa_decode_kernel" in row[indices["Kernel Name"]]
]
if len(kernel_rows) != 1:
    raise SystemExit(f"expected exactly one native kernel action row, got {len(kernel_rows)}")
row = kernel_rows[0]
values = {}
for metric in required:
    raw = row[indices[metric]].strip().replace(",", "")
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise SystemExit(f"invalid {metric}: {raw!r}")
    values[metric] = value
if values["smsp__inst_executed_pipe_tensor.sum"] <= 0.0:
    raise SystemExit("target kernel recorded no tensor-pipe instructions")
optional_launch = (
    "launch__cluster_size",
    "launch__cluster_max_potential_size",
    "launch__cluster_max_active",
    "launch__occupancy_cluster_pct",
    "launch__occupancy_cluster_gpu_pct",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__barrier_count",
)
launch = {
    name: row[header.index(name)].strip()
    for name in optional_launch
    if counts[name] == 1 and len(row) > header.index(name)
}
payload = {
    "schema": "c2-native-c2-ncu-counters-v1",
    "kernel_name": row[indices["Kernel Name"]],
    "action_id": row[indices["ID"]].strip(),
    "matching_kernel_rows": 1,
    "metrics": {
        metric: {"value": values[metric], "unit": units[metric]}
        for metric in required
    },
    "launch_metadata_if_reported": launch,
    "timing_valid_for_benchmark": False,
    "timing_note": "NCU replay and instrumentation invalidate benchmark timing; use separately collected clean non-NCU CUDA-event ABBA.",
    "all_gates_pass": True,
}
Path(output_path).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

printf '%s  %s\n' \
  "${EXPECTED_LIB_SHA}" "${LIB}" \
  "${EXPECTED_HARNESS_SHA}" "${HARNESS}" \
  "${EXPECTED_SOURCE_SHA}" "${SOURCE}" \
  | sha256sum -c -
sha256sum "${REPORT}" "${RAW_CSV}" "${TARGET_LOG}" "${SUMMARY}" "${CATALOG}" \
  "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" \
  "${ARTIFACT_DIR}/python-env-job${SLURM_JOB_ID}.json" \
  > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
apps_after=$(compute_apps) || { echo "failed to query post-run compute apps" >&2; exit 1; }
[[ -z "${apps_after}" ]]
