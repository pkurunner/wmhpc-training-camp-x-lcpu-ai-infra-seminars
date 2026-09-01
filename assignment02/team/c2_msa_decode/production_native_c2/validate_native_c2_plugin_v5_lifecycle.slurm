#!/usr/bin/env bash
# Fail-closed fresh-wheel lifecycle and boundary-shape gate for promoted v5.
# This is a downstream reproducible integration gate, not an upstream release
# or full-model service test.

#SBATCH --job-name=c2-v5-lifecycle
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v5-lifecycle-artifacts-20260830/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONOPTIMIZE=0
export PYTHONPATH=
export PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v5-lifecycle-artifacts-20260830
RUN_ID=${C2_RUN_ID:-${SLURM_JOB_ID}}
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${RUN_ID}
RUNTIME_ROOT=/home/lcpu/85117379/c2-native-plugin-v5-lifecycle-runtime-20260830
RUNTIME=${RUNTIME_ROOT}/job${RUN_ID}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
UV=/usr/local/bin/uv
C2_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02/team/c2_msa_decode
BASE_HARNESS=/home/lcpu/85117379/native_c2_full_backend_bench_20260829.py
HARNESS=/home/lcpu/85117379/native_c2_v5_lifecycle_20260830.py
HARNESS_INIT=${C2_ROOT}/harness/__init__.py
HARNESS_DATA=${C2_ROOT}/harness/data.py
HARNESS_REFERENCE=${C2_ROOT}/harness/reference.py
HARNESS_TRITON_BASELINE=${C2_ROOT}/harness/triton_baseline.py
VENDORED_SPARSE_ATTN=${C2_ROOT}/vllm_msa_ref/sparse_attn.py
SOURCE_SNAPSHOT=${ARTIFACT_DIR}/reviewed-c2-source
RESULT=${ARTIFACT_DIR}/native-c2-v5-lifecycle-job${RUN_ID}.json
INSTALL_AUDIT=${ARTIFACT_DIR}/installed-wheel-audit-job${RUN_ID}.json
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
EXPECTED_BASE_WHEEL_SHA=91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06
EXPECTED_STABLE_SHA=cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442
EXPECTED_PLUGIN_MEMBER=vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_ADAPTER_MEMBER=vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py
EXPECTED_HARNESS_INIT_SHA=df9dd0b01c3e91afeb1a490785fab1a0554a16f001e118005192f55f891358a8
EXPECTED_HARNESS_DATA_SHA=38faefb89b38e3bc26b135b762a6bbbd813dcc695423982b423a806fe41c11d6
EXPECTED_HARNESS_REFERENCE_SHA=712c1632367700137392d2c43bd5c6962130d945f01f1f585cc6338788ce53cc
EXPECTED_HARNESS_TRITON_BASELINE_SHA=6b0a40463b4f1c05ca53649d6689377b4f755d1efb58b10700dad05705b53f71
EXPECTED_VENDORED_SPARSE_ATTN_SHA=5db5d5e692044e54bb57418cbb7dcf5c9cc94f789eaccf128eec6fa8df86d967

: "${C2_WHEEL_ABSPATH:?absolute frozen v5 overlay wheel required}"
: "${C2_PROVENANCE_ABSPATH:?absolute matching provenance required}"
: "${C2_EXPECTED_WHEEL_SHA:?reviewed wheel SHA-256 required}"
: "${C2_EXPECTED_PROVENANCE_SHA:?reviewed provenance SHA-256 required}"
: "${C2_EXPECTED_PLUGIN_SHA:?reviewed plugin SHA-256 required}"
: "${C2_EXPECTED_PACKAGER_SHA:?reviewed packager SHA-256 required}"
: "${C2_EXPECTED_BASE_HARNESS_SHA:?reviewed base harness SHA-256 required}"
: "${C2_EXPECTED_LIFECYCLE_HARNESS_SHA:?reviewed lifecycle harness SHA-256 required}"
: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?reviewed lifecycle Slurm SHA-256 required}"

WHEEL=$(readlink -f "${C2_WHEEL_ABSPATH}")
PROVENANCE=$(readlink -f "${C2_PROVENANCE_ABSPATH}")
for digest in "${C2_EXPECTED_WHEEL_SHA}" "${C2_EXPECTED_PROVENANCE_SHA}" \
  "${C2_EXPECTED_PLUGIN_SHA}" "${C2_EXPECTED_PACKAGER_SHA}" \
  "${C2_EXPECTED_BASE_HARNESS_SHA}" "${C2_EXPECTED_LIFECYCLE_HARNESS_SHA}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]
done
[[ "${WHEEL}" == /* && "${PROVENANCE}" == /* ]]
[[ "$(dirname "${WHEEL}")" == "$(dirname "${PROVENANCE}")" ]]
[[ ! -e "${ARTIFACT_DIR}" && ! -e "${RUNTIME}" ]] || {
  echo "refusing to reuse lifecycle artifact/runtime path" >&2
  exit 2
}
mkdir -p "${ARTIFACT_ROOT}" "${RUNTIME_ROOT}" "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${RUN_ID}.log") 2>&1

finalize() {
  local rc=$?
  trap - EXIT
  set +e
  printf 'FINAL_RC=%s UTC=%s\n' "${rc}" "$(date -u +%FT%TZ)" | \
    tee "${ARTIFACT_DIR}/final-status-job${RUN_ID}.txt"
  exit "${rc}"
}
trap finalize EXIT

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
[[ -f "${WHEEL}" && -f "${PROVENANCE}" && -f "${BASE_HARNESS}" \
   && -f "${HARNESS}" && -f "${SCRIPT_PATH}" && -f "${HARNESS_INIT}" \
   && -f "${HARNESS_DATA}" && -f "${HARNESS_REFERENCE}" \
   && -f "${HARNESS_TRITON_BASELINE}" && -f "${VENDORED_SPARSE_ATTN}" ]]
[[ -x "${PYTHON}" && -x "${UV}" && -d "${C2_ROOT}" ]]
"${PYTHON}" -c 'raise SystemExit(0 if __debug__ else 1)'
printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${C2_EXPECTED_WHEEL_SHA}" "${WHEEL}" \
  "${C2_EXPECTED_PROVENANCE_SHA}" "${PROVENANCE}" \
  "${C2_EXPECTED_BASE_HARNESS_SHA}" "${BASE_HARNESS}" \
  "${C2_EXPECTED_LIFECYCLE_HARNESS_SHA}" "${HARNESS}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" \
  "${EXPECTED_HARNESS_INIT_SHA}" "${HARNESS_INIT}" \
  "${EXPECTED_HARNESS_DATA_SHA}" "${HARNESS_DATA}" \
  "${EXPECTED_HARNESS_REFERENCE_SHA}" "${HARNESS_REFERENCE}" \
  "${EXPECTED_HARNESS_TRITON_BASELINE_SHA}" "${HARNESS_TRITON_BASELINE}" \
  "${EXPECTED_VENDORED_SPARSE_ATTN_SHA}" "${VENDORED_SPARSE_ATTN}" | sha256sum -c -
sha256sum "${WHEEL}" "${PROVENANCE}" "${BASE_HARNESS}" "${HARNESS}" \
  "${SCRIPT_PATH}" "${HARNESS_INIT}" "${HARNESS_DATA}" "${HARNESS_REFERENCE}" \
  "${HARNESS_TRITON_BASELINE}" "${VENDORED_SPARSE_ATTN}" \
  > "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256"
install -D -m 0644 "${HARNESS_INIT}" "${SOURCE_SNAPSHOT}/harness/__init__.py"
install -D -m 0644 "${HARNESS_DATA}" "${SOURCE_SNAPSHOT}/harness/data.py"
install -D -m 0644 "${HARNESS_REFERENCE}" "${SOURCE_SNAPSHOT}/harness/reference.py"
install -D -m 0644 "${HARNESS_TRITON_BASELINE}" "${SOURCE_SNAPSHOT}/harness/triton_baseline.py"
install -D -m 0644 "${VENDORED_SPARSE_ATTN}" "${SOURCE_SNAPSHOT}/vllm_msa_ref/sparse_attn.py"
find "${SOURCE_SNAPSHOT}" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${ARTIFACT_DIR}/reviewed-c2-source-job${RUN_ID}.sha256"

"${PYTHON}" - "${WHEEL}" "${PROVENANCE}" "${C2_EXPECTED_WHEEL_SHA}" \
  "${C2_EXPECTED_PLUGIN_SHA}" "${C2_EXPECTED_PACKAGER_SHA}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

wheel, provenance = map(Path, sys.argv[1:3])
wheel_sha, plugin_sha, packager_sha = sys.argv[3:]
p = json.loads(provenance.read_text(encoding="utf-8"))
assert p["schema"] == "c2-native-plugin-overlay-provenance-v1"
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == wheel_sha
assert p["derived_wheel"]["sha256"] == wheel_sha
assert p["baseline_wheel"]["sha256"] == "91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06"
assert p["plugin_archive_member"] == "vllm/_native_c2_msa_decode_plugin.abi3.so"
assert p["plugin_expected_sha256"] == plugin_sha
assert p["packager"]["sha256"] == packager_sha
assert p["baseline_stable_member"]["byte_identical"] is True
assert p["baseline_stable_member"]["baseline_sha256"] == p["baseline_stable_member"]["derived_sha256"] == "cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442"
assert p["member_set"]["new_members"] == [
    "vllm/_native_c2_msa_decode_plugin.abi3.so",
    "vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py",
]
assert p["member_set"]["removed_members"] == []
assert p["derived_record_full_verification"]["all_payload_entries_validated"] is True
assert p["derived_record_full_verification"]["record_self_row_valid"] is True
PY

"${UV}" pip install --python "${PYTHON}" --target "${RUNTIME}" --no-deps "${WHEEL}"

"${PYTHON}" - "${RUNTIME}" "${WHEEL}" "${PROVENANCE}" "${INSTALL_AUDIT}" \
  "${C2_EXPECTED_PLUGIN_SHA}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import zipfile

runtime, wheel, provenance_path, output = map(Path, sys.argv[1:5])
plugin_sha = sys.argv[5]
p = json.loads(provenance_path.read_text(encoding="utf-8"))
root = runtime / "vllm"
stable = root / "_C_stable_libtorch.abi3.so"
plugin = root / "_native_c2_msa_decode_plugin.abi3.so"
adapter = root / "models/minimax_m3/nvidia/msa_native_c2_decode.py"
assert all(path.is_file() for path in (stable, plugin, adapter))
with zipfile.ZipFile(wheel, "r") as archive:
    assert stable.read_bytes() == archive.read("vllm/_C_stable_libtorch.abi3.so")
    assert plugin.read_bytes() == archive.read("vllm/_native_c2_msa_decode_plugin.abi3.so")
    assert adapter.read_bytes() == archive.read("vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py")
assert hashlib.sha256(stable.read_bytes()).hexdigest() == "cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442"
assert hashlib.sha256(plugin.read_bytes()).hexdigest() == plugin_sha
payload = {
    "schema": "c2-native-c2-v5-lifecycle-installed-wheel-v1",
    "runtime": str(runtime),
    "stable_sha256": hashlib.sha256(stable.read_bytes()).hexdigest(),
    "plugin_sha256": hashlib.sha256(plugin.read_bytes()).hexdigest(),
    "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
    "record_full_verification_from_provenance": p["derived_record_full_verification"],
    "all_gates_pass": True,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${RUNTIME}:${C2_ROOT}" \
  timeout --foreground --preserve-status --kill-after=10s 1800s \
  "${PYTHON}" "${HARNESS}" --c2-root "${C2_ROOT}" --vllm-root "${RUNTIME}" \
  --base-harness "${BASE_HARNESS}" --output "${RESULT}" --seed 20260830 \
  --steady-iterations 1000 --graph-replays 100 --memory-bound-bytes 8388608

"${PYTHON}" - "${RESULT}" <<'PY'
import json
import sys
from pathlib import Path

p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert p["schema"] == "c2-native-c2-v5-production-lifecycle-v2"
assert p["all_gates_pass"] is True
expected_gates = {
    "fresh_adapter_eight_thread_first_load",
    "real_forward_reaches_dispatcher_and_native_cuda_kernel",
    "steady_1000_real_forward_lifecycle",
    "cuda_graph_capture_replay_and_static_query_update",
    "native_support_predicate_matrix",
    "dynamic_native_rejection_without_fallback_claim",
    "native_forward_sequence_boundaries",
    "dispatcher_surface_persists_through_lifecycle",
}
assert set(p["gates"]) == expected_gates and all(p["gates"].values())
assert p["arguments"] == {
    "graph_replays_per_static_query": 100,
    "memory_bound_bytes": 8388608,
    "seed": 20260830,
    "steady_iterations": 1000,
}
assert p["first_load"]["concurrent_loader"]["thread_count"] == 8
assert p["first_load"]["concurrent_loader"]["load_library_call_count"] == 1
assert p["steady_lifecycle"]["iterations"] == 1000
assert p["cuda_graph"]["original_query"]["replays"]["replays"] == 100
assert p["cuda_graph"]["updated_query"]["replays"]["replays"] == 100
assert [row["sequence_length"] for row in p["sequence_length_boundaries"]["rows"]] == [2048, 2049, 4095, 4096]
assert all(row["pass"] for row in p["sequence_length_boundaries"]["rows"])
assert all(
    row["one_token_short_negative_control"]["pass"]
    for row in p["sequence_length_boundaries"]["rows"]
)
assert p["dynamic_rejection_gate"]["mode"] == "selection_only"
assert all(row["pass"] for row in p["dynamic_rejection_gate"]["rows"])
assert "full model weights" in p["boundary"]["not_covered"]
assert "unsupported native shape actually executes Triton" in p["boundary"]["not_covered"]
PY

snapshot POST
apps_after=$(compute_apps) || { echo "unable to query post-run compute apps" >&2; exit 1; }
[[ -z "${apps_after}" ]]
sha256sum "${RESULT}" "${INSTALL_AUDIT}" \
  "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256" \
  "${ARTIFACT_DIR}/reviewed-c2-source-job${RUN_ID}.sha256" \
  > "${ARTIFACT_DIR}/outputs-job${RUN_ID}.sha256"
