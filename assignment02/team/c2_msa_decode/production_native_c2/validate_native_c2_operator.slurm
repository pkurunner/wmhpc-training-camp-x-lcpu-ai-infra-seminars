#!/usr/bin/env bash
#SBATCH --job-name=c2-native-op
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-op-artifacts-20260829/slurm-%j.log

set -euo pipefail

ARTIFACT_DIR=/home/lcpu/85117379/c2-native-op-artifacts-20260829
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
LIB=/home/lcpu/85117379/vllm-d4-native-c2-build-20260829/_C_stable_libtorch.abi3.so
HARNESS=/home/lcpu/85117379/native_c2_operator_bench_20260829.py
SRC=/home/lcpu/85117379/vllm-d4-native-c2-20260829/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu

mkdir -p "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1
export PATH="/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin"

snapshot() {
  local label=$1
  printf '\n===== %s UTC=%s =====\n' "${label}" "$(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap \
    --format=csv,noheader,nounits
  printf '%s\n' '-- compute apps --'
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true
}

snapshot PRE
gpu_line=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* ]]
[[ -f "${LIB}" && -f "${HARNESS}" && -f "${SRC}" ]]
sha256sum "${LIB}" "${HARNESS}" "${SRC}"
"${PYTHON}" -c 'import torch; print(torch.__version__, torch.version.cuda)'

timeout --preserve-status --kill-after=10s 900s \
  "${PYTHON}" "${HARNESS}" --library "${LIB}" --seed 17 \
  > "${ARTIFACT_DIR}/operator-seed17-job${SLURM_JOB_ID}.json"
timeout --preserve-status --kill-after=10s 900s \
  "${PYTHON}" "${HARNESS}" --library "${LIB}" --seed 20260829 \
  > "${ARTIFACT_DIR}/operator-seed20260829-job${SLURM_JOB_ID}.json"

"${PYTHON}" - "${ARTIFACT_DIR}/operator-seed17-job${SLURM_JOB_ID}.json" \
  "${ARTIFACT_DIR}/operator-seed20260829-job${SLURM_JOB_ID}.json" <<'PY'
import json, math, sys
from pathlib import Path
for name in sys.argv[1:]:
    payload = json.loads(Path(name).read_text())
    assert payload["schema"] == "c2-native-c2-direct-operator-oracle-v1"
    assert payload["all_gates_pass"] is True
    assert payload["no_monkeypatch"] is True
    assert payload["dispatch"] == "direct_torch.ops._C.native_c2_msa_decode"
    assert payload["correctness"]["allclose"] is True
    assert payload["correctness"]["finite_output"] is True
    assert payload["caller_output"]["pointer_unchanged"] is True
    for key in ("max_abs", "max_rel", "mean_abs"):
        assert math.isfinite(float(payload["correctness"][key]))
PY

cuobjdump --dump-resource-usage "${LIB}" \
  > "${ARTIFACT_DIR}/resource-usage-job${SLURM_JOB_ID}.log"
cuobjdump --dump-sass "${LIB}" \
  > "${ARTIFACT_DIR}/all-sass-job${SLURM_JOB_ID}.log"
"${PYTHON}" - "${ARTIFACT_DIR}/all-sass-job${SLURM_JOB_ID}.log" \
  "${ARTIFACT_DIR}/native-c2-instructions-job${SLURM_JOB_ID}.json" <<'PY'
import json, re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(errors="replace")
needle = "native_c2_msa_decode_kernel"
starts = [m.start() for m in re.finditer(r"Function : .*" + needle, text)]
assert len(starts) == 1, starts
start = starts[0]
match = re.search(r"\n\s*Function : ", text[start + 1:])
end = len(text) if match is None else start + 1 + match.start()
body = text[start:end]
record = {
    "schema": "c2-native-c2-instruction-evidence-v1",
    "function": needle,
    "hmma_16816_f32_bf16": body.count("HMMA.16816.F32.BF16"),
    "mbarrier_arrive": body.count("SYNCS.ARRIVE"),
    "mbarrier_wait": body.count("SYNCS.PHASECHK"),
    "cluster_arrive": body.count("UCGABAR_ARV"),
    "cluster_wait": body.count("UCGABAR_WAIT"),
}
assert record["hmma_16816_f32_bf16"] > 0, record
assert record["mbarrier_arrive"] > 0 and record["mbarrier_wait"] > 0, record
assert record["cluster_arrive"] > 0 and record["cluster_wait"] > 0, record
Path(sys.argv[2]).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY

snapshot POST
sha256sum "${ARTIFACT_DIR}"/*job"${SLURM_JOB_ID}"* \
  > "${ARTIFACT_DIR}/operator-job${SLURM_JOB_ID}.sha256"
printf 'FINAL_RC=0 UTC=%s\n' "$(date -u +%FT%TZ)"
