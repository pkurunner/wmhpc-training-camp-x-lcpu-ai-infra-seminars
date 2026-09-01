#!/usr/bin/env bash
# Strict direct-op gate plus four-producer-specific directed liveness tests.
# Submit only after the v2+4P build has printed its reviewed candidate AOT
# digest into next-submit-job*.env.
#SBATCH --job-name=c2-native-v2-4p-stress
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-v2-four-producer-stress-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail

: "${C2_EXPECTED_LIB_SHA:?submit with the reviewed v2+4P AOT SHA-256}"
[[ "${C2_EXPECTED_LIB_SHA}" =~ ^[0-9a-f]{64}$ ]]
export C2_EXPECTED_SOURCE_SHA=a46830564aa62e30e26de67dc98882043a239985f6d1c7e1b2dc2a03630fd8a1
export C2_STRESS_ARTIFACT_DIR=/home/lcpu/85117379/c2-native-v2-four-producer-stress-artifacts-20260829
export C2_NATIVE_SOURCE=/home/lcpu/85117379/c2-native-v2-four-producer-build-artifacts-20260829/native_c2_decode.v2-four-producer.cu

LIB=${C2_LIB:-/home/lcpu/85117379/vllm-d4-native-c2-build-20260829/_C_stable_libtorch.abi3.so}
export C2_LIB="${LIB}"
PYTHON=${C2_PYTHON:-/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python}
BASE_HARNESS=${C2_BASE_HARNESS:-/home/lcpu/85117379/native_c2_operator_bench_20260829.py}
DIRECTED_HARNESS=${C2_FOUR_PRODUCER_HARNESS:-/home/lcpu/85117379/native_c2_four_producer_directed_20260829.py}
DIRECTED_RESULT="${C2_STRESS_ARTIFACT_DIR}/four-producer-directed-job${SLURM_JOB_ID}.json"

mkdir -p "${C2_STRESS_ARTIFACT_DIR}"
[[ -f "${LIB}" && -f "${C2_NATIVE_SOURCE}" && -f "${BASE_HARNESS}" && -f "${DIRECTED_HARNESS}" ]]
[[ "$(sha256sum "${LIB}" | awk '{print $1}')" == "${C2_EXPECTED_LIB_SHA}" ]]

# The reused eight-seed FP64-oracle/stability/performance gate remains the
# primary correctness check; it validates the exact production ABI.
bash /home/lcpu/85117379/validate_native_c2_stress_perf_20260829.sh

# Each invocation is synchronized and individually bounded.  The outer
# timeout catches a true GPU/cluster rendezvous hang; the Python harness also
# rejects any completed iteration that exceeds its host-side liveness budget.
timeout --foreground --preserve-status --kill-after=10s 600s \
  "${PYTHON}" "${DIRECTED_HARNESS}" \
  --library "${LIB}" \
  --base-harness "${BASE_HARNESS}" \
  --sync-repeats 4 --max-iteration-seconds 15 \
  > "${DIRECTED_RESULT}"

"${PYTHON}" - "${DIRECTED_RESULT}" <<'PY'
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["schema"] == "c2-native-c2-four-producer-directed-v1"
assert payload["all_gates_pass"] is True
assert payload["no_monkeypatch"] is True
assert payload["dispatch"] == "direct_torch.ops._C.native_c2_msa_decode"
assert payload["rank_directed_case_count"] == 4
assert payload["all_invalid"]["all_gates_pass"] is True
assert payload["all_invalid"]["all_zero"] is True
for case in payload["rank_directed"]:
    assert case["all_gates_pass"] is True
    assert case["oracle"]["allclose"] is True
    assert case["oracle"]["finite_output"] is True
    assert case["caller_output"]["pointer_unchanged"] is True
    assert case["caller_output"]["return_is_none"] is True
    assert case["stability"]["bitwise_repeatable"] is True
    assert len(case["sync_watchdog"]["elapsed_seconds"]) >= 4
    assert all(math.isfinite(float(value)) and float(value) >= 0.0
               for value in case["sync_watchdog"]["elapsed_seconds"])
PY

sha256sum "${DIRECTED_RESULT}" "${C2_NATIVE_SOURCE}" "${LIB}" \
  > "${C2_STRESS_ARTIFACT_DIR}/four-producer-directed-outputs-job${SLURM_JOB_ID}.sha256"
