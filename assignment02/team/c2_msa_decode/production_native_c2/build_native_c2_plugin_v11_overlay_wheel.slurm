#!/usr/bin/env bash
# Build an experimental, local overlay wheel from the AOT-validated v11 DSO.
# This is a repackaging/verification gate only: it is neither an upstream
# change nor a release artifact.  The base d4 wheel and its stable extension
# remain immutable; only the reviewed native plugin and dispatch overlay are
# allowed to differ.
#
# Submit only with the reviewed script digest, for example:
#   sbatch --export=ALL,C2_EXPECTED_RUNTIME_SCRIPT_SHA=<sha256> \
#     build_native_c2_plugin_v11_overlay_wheel.slurm

#SBATCH --job-name=c2-native-plugin-v11-wheel
#SBATCH --partition=cpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v11-wheel-artifacts-20260830/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONOPTIMIZE=0
export PYTHONPATH=

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v11-wheel-artifacts-20260830
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID:?must run under Slurm}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
PACKAGER=/home/lcpu/85117379/build_native_c2_plugin_overlay_wheel_20260829.py
BASE_WHEEL=/home/lcpu/85117379/vllm-d4-wheel/vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl
EXPECTED_BASELINE_SHA=91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06
EXPECTED_STABLE_MEMBER_SHA=cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442
EXPECTED_PLUGIN_ARCHIVE_MEMBER=vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_ADAPTER_MEMBER=vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py
EXPECTED_D4_COMMIT=d4da0c55af3aa231b6209bf77871f3ed36eab0d2
EXPECTED_V11_SOURCE_SHA=0e82b278b7aa44a034a96d2ddd19946a27928d95a9f9d03e8e1fe9f30680c5b4
EXPECTED_V11_PLUGIN_SHA=501b9c56a29a81f779a35a34b2808b4122af7dcb23e8108d8c590306be43d957
EXPECTED_PACKAGER_SHA=84f607ec46f4535a5d5e3506631d4eeb9a044f9e8d48f4aa0eec57984ca2866e
EXPECTED_ADAPTER_SHA=6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412

# Immutable AOT inputs from the accepted v11 build.  Pinning the three text
# artifacts by content guards against a later replacement that preserves only
# a filename; the provenance file also re-verifies all original AOT inputs.
EXPECTED_V11_CHECKOUT=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v11-q-fragment-reuse-20260830-job12825
EXPECTED_V11_PLUGIN_LIBRARY=/home/lcpu/85117379/c2-native-plugin-v11-aot-artifacts-20260830/job12825/vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_V11_BUILD_PLUGIN=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v11-q-fragment-reuse-build-20260830-job12825/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_V11_PLUGIN_SOURCE=${EXPECTED_V11_CHECKOUT}/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
AOT_ARTIFACT_DIR=/home/lcpu/85117379/c2-native-plugin-v11-aot-artifacts-20260830/job12825
AOT_FINAL_STATUS=${AOT_ARTIFACT_DIR}/final-status-job12825.txt
AOT_FINAL_STATUS_SIDECAR=${AOT_ARTIFACT_DIR}/final-status-job12825.sha256
AOT_OUTPUTS=${AOT_ARTIFACT_DIR}/outputs-job12825.sha256
AOT_RESOURCE_GATE=${AOT_ARTIFACT_DIR}/plugin-resource-gate-job12825.txt
AOT_PROVENANCE=${AOT_ARTIFACT_DIR}/plugin-v11-provenance-job12825.sha256
EXPECTED_AOT_FINAL_STATUS_SHA=69bd4be581ea5a6d519753c6bf746083ad1b393e6add3fcaf6e93c0e9b2653d3
EXPECTED_AOT_FINAL_STATUS_SIDECAR_SHA=153b2a792dcef3d8834805a66a6b626b291b6522c151d75b54b1cbdbec7881d1
EXPECTED_AOT_OUTPUTS_SHA=f60b3307a1a343cb0d7be9107526b410351b4793f8f8ce6bf71b9e2c93984f75
EXPECTED_AOT_RESOURCE_GATE_SHA=f82d6890d14be7cfb1df54ec12fdff0b2de3096204591b554c46eb02862d5ab4
EXPECTED_AOT_PROVENANCE_SHA=6d96a7437616c672c8640592ac99da0b722d0d6f4e7d8f78b18a3a6652b59128

# Packaging is allowed only after the separately hardened correctness run and
# the accepted same-GPU v9->v11 >3% incremental gate.  These immutable
# prerequisites prevent a merely compilable DSO from being promoted.
DIRECTED_ARTIFACT_DIR=/home/lcpu/85117379/c2-native-plugin-v11-directed-artifacts-20260830/job12904
DIRECTED_RESULT=${DIRECTED_ARTIFACT_DIR}/v11-q-fragment-reuse-directed-job12904.json
DIRECTED_FINAL_STATUS=${DIRECTED_ARTIFACT_DIR}/final-status-job12904.txt
DIRECTED_FINAL_STATUS_SIDECAR=${DIRECTED_ARTIFACT_DIR}/final-status-job12904.sha256
DIRECTED_OUTPUTS=${DIRECTED_ARTIFACT_DIR}/outputs-job12904.sha256
DIRECTED_RUN_MANIFEST=${DIRECTED_ARTIFACT_DIR}/run-manifest-job12904.json
EXPECTED_DIRECTED_RESULT_SHA=7567626bc80950303bde34bf3e9e5062be2338bb927c19a4184b4245dde58f93
EXPECTED_DIRECTED_FINAL_STATUS_SHA=a8d67a71b4f74b5dbc236e0cbeb0322a39ad9d7f8ed2219bfd0f637faeb3e521
EXPECTED_DIRECTED_FINAL_STATUS_SIDECAR_SHA=f7665b3e4814bfd581e6b58a6ab6f3431f4fa3d1667ce22bfc10973ab436c13e
EXPECTED_DIRECTED_OUTPUTS_SHA=9b3d7fb69065ba5923131d5b00cb0bb1357fc8d0376cb1e85d1b2363793c6edf
EXPECTED_DIRECTED_RUN_MANIFEST_SHA=1af137f33adcb3ab52a2fce09f5926afc5d8bccc2c008af473b2d6cdcad640c6

STRESS_ARTIFACT_DIR=/home/lcpu/85117379/c2-native-plugin-v11-stress-3pct-artifacts-20260830/job12905
STRESS_DECISION=${STRESS_ARTIFACT_DIR}/v11-q-fragment-reuse-vs-v9-3pct-incremental-decision-job12905.json
STRESS_AGGREGATE=${STRESS_ARTIFACT_DIR}/v11-q-fragment-reuse-vs-v9-3pct-incremental-aggregate-job12905.json
STRESS_FINAL_STATUS=${STRESS_ARTIFACT_DIR}/final-status-job12905.txt
STRESS_FINAL_STATUS_SIDECAR=${STRESS_ARTIFACT_DIR}/final-status-job12905.sha256
STRESS_OUTPUTS=${STRESS_ARTIFACT_DIR}/outputs-job12905.sha256
EXPECTED_STRESS_DECISION_SHA=ae34d33010be4f71bc9bfbbd2a0f8857e04c6071099314f538c6b406735b1569
EXPECTED_STRESS_AGGREGATE_SHA=85b9489cc36155a18c239729f4fcea7b746c67c7c3bed8b10944c73ddd7635b9
EXPECTED_STRESS_FINAL_STATUS_SHA=003b99492f9ff2a1a1f61fba9b142f05145519a916f1978a344d143638a109bc
EXPECTED_STRESS_FINAL_STATUS_SIDECAR_SHA=b39c5100667149b259e4fb227872591b95482fda8a21c77b53540f0a92a7fa62
EXPECTED_STRESS_OUTPUTS_SHA=34366fad9ca87f3b034c899ca7ed421534a4e25b15a987dd9d851c6d5b88fd98

: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?reviewed v11 wheel-script SHA-256 required}"
[[ "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" =~ ^[0-9a-f]{64}$ ]]
for digest in "${EXPECTED_BASELINE_SHA}" "${EXPECTED_STABLE_MEMBER_SHA}" \
  "${EXPECTED_V11_SOURCE_SHA}" "${EXPECTED_V11_PLUGIN_SHA}" \
  "${EXPECTED_PACKAGER_SHA}" "${EXPECTED_ADAPTER_SHA}" \
  "${EXPECTED_AOT_FINAL_STATUS_SHA}" "${EXPECTED_AOT_RESOURCE_GATE_SHA}" \
  "${EXPECTED_AOT_FINAL_STATUS_SIDECAR_SHA}" "${EXPECTED_AOT_OUTPUTS_SHA}" \
  "${EXPECTED_AOT_PROVENANCE_SHA}" \
  "${EXPECTED_DIRECTED_RESULT_SHA}" "${EXPECTED_DIRECTED_FINAL_STATUS_SHA}" \
  "${EXPECTED_DIRECTED_FINAL_STATUS_SIDECAR_SHA}" "${EXPECTED_DIRECTED_OUTPUTS_SHA}" \
  "${EXPECTED_DIRECTED_RUN_MANIFEST_SHA}" "${EXPECTED_STRESS_DECISION_SHA}" \
  "${EXPECTED_STRESS_AGGREGATE_SHA}" "${EXPECTED_STRESS_FINAL_STATUS_SHA}" \
  "${EXPECTED_STRESS_FINAL_STATUS_SIDECAR_SHA}" "${EXPECTED_STRESS_OUTPUTS_SHA}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]
done

# sbatch places an immutable copy of this script in the slurmd job spool.  Do
# not trust a submit-directory path after queueing: verify the executed spool
# copy against the reviewed digest and keep both identities in the manifest.
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SUBMITTED_SCRIPT=/home/lcpu/85117379/build_native_c2_plugin_v11_overlay_wheel_20260830.slurm
[[ -f "${SCRIPT_PATH}" && -f "${SUBMITTED_SCRIPT}" ]] || {
  echo "missing executed Slurm spool script or reviewed submitted copy" >&2; exit 2;
}
cmp -s "${SCRIPT_PATH}" "${SUBMITTED_SCRIPT}" || {
  echo "executed Slurm spool bytes differ from reviewed submitted copy" >&2; exit 2;
}
[[ ! -e "${ARTIFACT_DIR}" ]] || { echo "refusing to reuse ${ARTIFACT_DIR}" >&2; exit 2; }
mkdir -p "${ARTIFACT_ROOT}" "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1

finalize() {
  local rc=$?
  trap - EXIT
  set +e
  local status_path=${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt
  local manifest_path=${ARTIFACT_DIR}/final-output-manifest-job${SLURM_JOB_ID}.sha256
  write_status() {
    printf 'FINAL_RC=%s UTC=%s\n' "$1" "$(date -u +%FT%TZ)" | tee "${status_path}"
  }
  write_manifest() {
    # The process-substitution tee can append to the driver log until this
    # shell exits, so that live log is deliberately outside the stable final
    # manifest.  Every completed gate artifact and final status remains in it.
    find "${ARTIFACT_DIR}" -maxdepth 1 -type f \
      ! -name 'driver-job*.log' ! -name 'final-output-manifest-job*.sha256' \
      -printf '%p\0' | sort -z | xargs -0r sha256sum > "${manifest_path}"
  }
  if ! write_status "${rc}"; then
    [[ "${rc}" -ne 0 ]] || rc=2
  fi
  if ! write_manifest; then
    if [[ "${rc}" -eq 0 ]]; then
      rc=2
      write_status "${rc}" || true
      write_manifest || true
    fi
  fi
  exit "${rc}"
}
trap finalize EXIT

[[ -f "${BASE_WHEEL}" && -f "${PACKAGER}" && -x "${PYTHON}" ]]
[[ -d "${EXPECTED_V11_CHECKOUT}" && -f "${EXPECTED_V11_PLUGIN_LIBRARY}" \
   && -f "${EXPECTED_V11_BUILD_PLUGIN}" && -f "${EXPECTED_V11_PLUGIN_SOURCE}" ]]
[[ -f "${AOT_FINAL_STATUS}" && -f "${AOT_FINAL_STATUS_SIDECAR}" \
   && -f "${AOT_OUTPUTS}" && -f "${AOT_RESOURCE_GATE}" && -f "${AOT_PROVENANCE}" ]]
[[ -f "${DIRECTED_RESULT}" && -f "${DIRECTED_FINAL_STATUS}" \
   && -f "${DIRECTED_FINAL_STATUS_SIDECAR}" && -f "${DIRECTED_OUTPUTS}" \
   && -f "${DIRECTED_RUN_MANIFEST}" ]]
[[ -f "${STRESS_DECISION}" && -f "${STRESS_AGGREGATE}" \
   && -f "${STRESS_FINAL_STATUS}" && -f "${STRESS_FINAL_STATUS_SIDECAR}" \
   && -f "${STRESS_OUTPUTS}" ]]
"${PYTHON}" -c 'raise SystemExit(0 if __debug__ else 1)'
"${PYTHON}" -m py_compile "${PACKAGER}"
git -C "${EXPECTED_V11_CHECKOUT}" rev-parse HEAD
[[ "$(git -C "${EXPECTED_V11_CHECKOUT}" rev-parse HEAD)" == "${EXPECTED_D4_COMMIT}" ]]
git -C "${EXPECTED_V11_CHECKOUT}" diff --check
[[ "$(readlink -f "${EXPECTED_V11_PLUGIN_SOURCE}")" == \
   "$(readlink -f "${EXPECTED_V11_CHECKOUT}/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu")" ]]

# The AOT build attestation has to be fully successful and retain the v11
# q-fragment-reuse resource gate before its DSO may enter an overlay wheel.
grep -qxE 'BODY_RC=0 TEE_RC=0 MANIFEST_RC=0 FINAL_RC=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${AOT_FINAL_STATUS}"
grep -qx 'REG=127' "${AOT_RESOURCE_GATE}"
grep -qx 'STACK=0' "${AOT_RESOURCE_GATE}"
grep -qx 'SHARED=30880' "${AOT_RESOURCE_GATE}"
grep -qx 'LOCAL=0' "${AOT_RESOURCE_GATE}"
grep -qxE 'FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 POST_CHECKED=1 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${DIRECTED_FINAL_STATUS}"
grep -qxE 'FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${STRESS_FINAL_STATUS}"
printf '%s  %s\n' \
  "${EXPECTED_BASELINE_SHA}" "${BASE_WHEEL}" \
  "${EXPECTED_V11_PLUGIN_SHA}" "${EXPECTED_V11_PLUGIN_LIBRARY}" \
  "${EXPECTED_V11_PLUGIN_SHA}" "${EXPECTED_V11_BUILD_PLUGIN}" \
  "${EXPECTED_V11_SOURCE_SHA}" "${EXPECTED_V11_PLUGIN_SOURCE}" \
  "${EXPECTED_PACKAGER_SHA}" "${PACKAGER}" \
  "${EXPECTED_ADAPTER_SHA}" "${EXPECTED_V11_CHECKOUT}/${EXPECTED_ADAPTER_MEMBER}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SUBMITTED_SCRIPT}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" \
  "${EXPECTED_AOT_FINAL_STATUS_SHA}" "${AOT_FINAL_STATUS}" \
  "${EXPECTED_AOT_FINAL_STATUS_SIDECAR_SHA}" "${AOT_FINAL_STATUS_SIDECAR}" \
  "${EXPECTED_AOT_OUTPUTS_SHA}" "${AOT_OUTPUTS}" \
  "${EXPECTED_AOT_RESOURCE_GATE_SHA}" "${AOT_RESOURCE_GATE}" \
  "${EXPECTED_DIRECTED_RESULT_SHA}" "${DIRECTED_RESULT}" \
  "${EXPECTED_DIRECTED_FINAL_STATUS_SHA}" "${DIRECTED_FINAL_STATUS}" \
  "${EXPECTED_DIRECTED_FINAL_STATUS_SIDECAR_SHA}" "${DIRECTED_FINAL_STATUS_SIDECAR}" \
  "${EXPECTED_DIRECTED_OUTPUTS_SHA}" "${DIRECTED_OUTPUTS}" \
  "${EXPECTED_DIRECTED_RUN_MANIFEST_SHA}" "${DIRECTED_RUN_MANIFEST}" \
  "${EXPECTED_STRESS_DECISION_SHA}" "${STRESS_DECISION}" \
  "${EXPECTED_STRESS_AGGREGATE_SHA}" "${STRESS_AGGREGATE}" \
  "${EXPECTED_STRESS_FINAL_STATUS_SHA}" "${STRESS_FINAL_STATUS}" \
  "${EXPECTED_STRESS_FINAL_STATUS_SIDECAR_SHA}" "${STRESS_FINAL_STATUS_SIDECAR}" \
  "${EXPECTED_STRESS_OUTPUTS_SHA}" "${STRESS_OUTPUTS}" \
  > "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
printf '%s  %s\n' "${EXPECTED_AOT_PROVENANCE_SHA}" "${AOT_PROVENANCE}" | sha256sum -c -
# This verifies the AOT source, both DSO paths, and every frozen AOT build
# input named in the original provenance file before repackaging starts.
sha256sum -c "${AOT_FINAL_STATUS_SIDECAR}"
sha256sum -c "${AOT_OUTPUTS}"
sha256sum -c "${AOT_PROVENANCE}"
sha256sum -c "${DIRECTED_FINAL_STATUS_SIDECAR}"
sha256sum -c "${DIRECTED_OUTPUTS}"
sha256sum -c "${STRESS_FINAL_STATUS_SIDECAR}"
sha256sum -c "${STRESS_OUTPUTS}"
sha256sum "${BASE_WHEEL}" "${EXPECTED_V11_PLUGIN_LIBRARY}" "${EXPECTED_V11_BUILD_PLUGIN}" \
  "${EXPECTED_V11_PLUGIN_SOURCE}" "${PACKAGER}" \
  "${EXPECTED_V11_CHECKOUT}/${EXPECTED_ADAPTER_MEMBER}" "${SUBMITTED_SCRIPT}" "${SCRIPT_PATH}" \
  "${AOT_FINAL_STATUS}" "${AOT_FINAL_STATUS_SIDECAR}" "${AOT_OUTPUTS}" \
  "${AOT_RESOURCE_GATE}" "${AOT_PROVENANCE}" \
  "${DIRECTED_RESULT}" "${DIRECTED_FINAL_STATUS}" "${DIRECTED_FINAL_STATUS_SIDECAR}" \
  "${DIRECTED_OUTPUTS}" "${DIRECTED_RUN_MANIFEST}" "${STRESS_DECISION}" \
  "${STRESS_AGGREGATE}" "${STRESS_FINAL_STATUS}" "${STRESS_FINAL_STATUS_SIDECAR}" \
  "${STRESS_OUTPUTS}" \
  > "${ARTIFACT_DIR}/all-inputs-job${SLURM_JOB_ID}.sha256"

"${PYTHON}" - "${ARTIFACT_DIR}/promotion-prerequisites-job${SLURM_JOB_ID}.json" \
  "${DIRECTED_RESULT}" "${STRESS_DECISION}" "${STRESS_AGGREGATE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out, directed_path, decision_path, aggregate_path = map(Path, sys.argv[1:])
directed = json.loads(directed_path.read_text(encoding="utf-8"))
decision = json.loads(decision_path.read_text(encoding="utf-8"))
aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

assert directed["all_gates_pass"] is True
assert directed["no_monkeypatch"] is True
assert directed["profiler"]["exactly_one_native_kernel_event"] is True
assert directed["profiler"]["native_dispatcher_hit"] is True
assert decision["accepted"] is True and decision["issues"] == []
assert decision["all_correctness_and_resource_identity_gates_pass"] is True
assert decision["aggregate_sha256"] == digest(aggregate_path)
performance = decision["performance"]
assert performance["paired_median_improvement_fraction"] > performance["required_improvement_fraction"] == 0.03
assert performance["bootstrap_95pct_lower_bound_fraction"] > performance["required_bootstrap_lower_bound_fraction"] == 0.0
assert aggregate["all_correctness_and_comparability_gates_pass"] is True
assert aggregate["issues"] == []
assert len(set(aggregate["gpu_uuids"])) == 1
assert aggregate["allocated_gpu_uuid"] == aggregate["gpu_uuids"][0]

payload = {
    "schema": "c2-native-plugin-v11-wheel-promotion-prerequisites-v1",
    "directed_job_id": 12904,
    "stress_job_id": 12905,
    "directed_all_gates_pass": True,
    "stress_accepted": True,
    "paired_median_improvement_fraction": performance["paired_median_improvement_fraction"],
    "bootstrap_95pct_lower_bound_fraction": performance["bootstrap_95pct_lower_bound_fraction"],
    "one_physical_gpu_uuid": aggregate["allocated_gpu_uuid"],
    "inputs_sha256": {
        "directed_result": digest(directed_path),
        "stress_decision": digest(decision_path),
        "stress_aggregate": digest(aggregate_path),
    },
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"${PYTHON}" - "${ARTIFACT_DIR}/aot-attestation-job${SLURM_JOB_ID}.json" \
  "${AOT_FINAL_STATUS}" "${AOT_RESOURCE_GATE}" "${AOT_PROVENANCE}" \
  "${EXPECTED_V11_PLUGIN_LIBRARY}" "${EXPECTED_V11_BUILD_PLUGIN}" \
  "${EXPECTED_V11_PLUGIN_SOURCE}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

out, final_status, resource_gate, provenance, plugin, build_plugin, source = map(Path, sys.argv[1:])
expected_plugin = "501b9c56a29a81f779a35a34b2808b4122af7dcb23e8108d8c590306be43d957"
expected_source = "0e82b278b7aa44a034a96d2ddd19946a27928d95a9f9d03e8e1fe9f30680c5b4"
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert re.fullmatch(
    r"BODY_RC=0 TEE_RC=0 MANIFEST_RC=0 FINAL_RC=0 UTC=\d{4}-\d{2}-\d{2}T[\d:]+Z\n",
    final_status.read_text(encoding="utf-8"),
)
resource = resource_gate.read_text(encoding="utf-8").splitlines()
assert {"REG=127", "STACK=0", "SHARED=30880", "LOCAL=0"}.issubset(resource)
records = {}
for line in provenance.read_text(encoding="utf-8").splitlines():
    fields = line.split(maxsplit=1)
    assert len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]), line
    records[fields[1]] = fields[0]
assert records[str(plugin)] == expected_plugin
assert records[str(build_plugin)] == expected_plugin
assert records[str(source)] == expected_source
assert digest(plugin) == digest(build_plugin) == expected_plugin
assert digest(source) == expected_source
payload = {
    "schema": "c2-native-plugin-v11-aot-attestation-v1",
    "aot_job_id": 12825,
    "aot_final_rc": 0,
    "resource_gate": {"REG": 127, "STACK": 0, "SHARED": 30880, "LOCAL": 0},
    "v11_plugin_sha256": expected_plugin,
    "v11_source_sha256": expected_source,
    "aot_files_sha256": {
        "final_status": digest(final_status),
        "resource_gate": digest(resource_gate),
        "provenance": digest(provenance),
    },
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"${PYTHON}" "${PACKAGER}" \
  --base-wheel "${BASE_WHEEL}" \
  --plugin-checkout "${EXPECTED_V11_CHECKOUT}" \
  --plugin-library "${EXPECTED_V11_PLUGIN_LIBRARY}" \
  --plugin-archive-member "${EXPECTED_PLUGIN_ARCHIVE_MEMBER}" \
  --expected-plugin-sha256 "${EXPECTED_V11_PLUGIN_SHA}" \
  --artifact-dir "${ARTIFACT_DIR}" \
  | tee "${ARTIFACT_DIR}/result-job${SLURM_JOB_ID}.json"

# Re-verify the produced archive independently of the packager's provenance:
# all RECORD payload entries, exact member deltas, each unchanged baseline byte,
# the frozen stable DSO, and every reviewed dispatch/member digest must agree.
"${PYTHON}" - "${ARTIFACT_DIR}" "${BASE_WHEEL}" "${EXPECTED_V11_PLUGIN_LIBRARY}" \
  "${AOT_PROVENANCE}" "${SUBMITTED_SCRIPT}" "${SCRIPT_PATH}" \
  "${ARTIFACT_DIR}/wheel-driver-verification-job${SLURM_JOB_ID}.json" <<'PY'
import base64
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
import zipfile

artifact_dir, baseline, plugin, aot_provenance, script, spool, out = map(Path, sys.argv[1:])
expected_plugin = "501b9c56a29a81f779a35a34b2808b4122af7dcb23e8108d8c590306be43d957"
expected_stable = "cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442"
plugin_member = "vllm/_native_c2_msa_decode_plugin.abi3.so"
adapter_member = "vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py"
dispatch = {
    "vllm/_custom_ops.py": "b57dfa819424b89bed8bf39924d5a4a3d2ebf327b39f3ae7c2e9e65b61c2398d",
    "vllm/config/attention.py": "caf68c29d8d36324d252a66f0811a25b26d11da9860732359076fbb76625b5ae",
    "vllm/v1/attention/backends/registry.py": "cd539941417fd5faf6672676de0f8ae15c9adb2a915b47cdd508e154c34faea4",
    adapter_member: "6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412",
    "vllm/models/minimax_m3/nvidia/sparse_attention_msa.py": "07d00372ca1995aa96159a0b9f8949b1b40f817a63f49224e88aafac1b722200",
}
def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
def record_check(archive: zipfile.ZipFile, names: list[str], record: str) -> dict:
    rows = list(csv.reader(io.StringIO(archive.read(record).decode("utf-8"))))
    assert rows and all(len(row) == 3 for row in rows)
    index = {row[0]: row for row in rows}
    assert len(index) == len(rows)
    payload = {name for name in names if not name.endswith("/")} - {record}
    assert set(index) - {record} == payload
    assert index[record] == [record, "", ""]
    for name in payload:
        data = archive.read(name)
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        assert index[name] == [name, f"sha256={encoded}", str(len(data))]
    return {"record_member": record, "payload_count": len(payload), "all_payload_entries_validated": True}
def record_name(names: list[str]) -> str:
    found = [name for name in names if name.endswith(".dist-info/RECORD")]
    assert len(found) == 1
    return found[0]
wheel = artifact_dir / "vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl"
provenance = json.loads((artifact_dir / "c2-native-plugin-overlay-provenance.json").read_text(encoding="utf-8"))
manifest = json.loads((artifact_dir / "c2-native-plugin-overlay-manifest.json").read_text(encoding="utf-8"))
assert wheel.is_file() and zipfile.is_zipfile(wheel)
assert digest(wheel.read_bytes()) == provenance["derived_wheel"]["sha256"] == manifest["derived_wheel_sha256"]
assert provenance["baseline_wheel"]["sha256"] == "91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06"
assert provenance["packager"]["sha256"] == manifest["packager_sha256"] == "84f607ec46f4535a5d5e3506631d4eeb9a044f9e8d48f4aa0eec57984ca2866e"
assert provenance["member_set"]["new_members"] == [plugin_member, adapter_member]
assert provenance["member_set"]["removed_members"] == []
assert provenance["baseline_stable_member"]["byte_identical"] is True
assert provenance["baseline_stable_member"]["baseline_sha256"] == provenance["baseline_stable_member"]["derived_sha256"] == expected_stable
with zipfile.ZipFile(baseline) as base, zipfile.ZipFile(wheel) as derived:
    assert base.testzip() is None and derived.testzip() is None
    base_names, derived_names = base.namelist(), derived.namelist()
    assert len(base_names) == len(set(base_names)) and len(derived_names) == len(set(derived_names))
    base_record, derived_record = record_name(base_names), record_name(derived_names)
    base_record_check = record_check(base, base_names, base_record)
    derived_record_check = record_check(derived, derived_names, derived_record)
    assert set(derived_names) == set(base_names) | {plugin_member, adapter_member}
    approved_changes = set(dispatch) | {base_record}
    unchanged = []
    for name in base_names:
        if name.endswith("/") or name in approved_changes:
            continue
        assert base.read(name) == derived.read(name), name
        unchanged.append(name)
    stable_base = base.read("vllm/_C_stable_libtorch.abi3.so")
    stable_derived = derived.read("vllm/_C_stable_libtorch.abi3.so")
    assert stable_base == stable_derived and digest(stable_derived) == expected_stable
    assert derived.read(plugin_member) == plugin.read_bytes()
    assert digest(derived.read(plugin_member)) == expected_plugin
    for name, expected in dispatch.items():
        assert digest(derived.read(name)) == expected
assert aot_provenance.is_file() and script.is_file() and spool.is_file()
payload = {
    "schema": "c2-native-plugin-v11-overlay-wheel-driver-verification-v1",
    "wheel": str(wheel),
    "wheel_sha256": digest(wheel.read_bytes()),
    "plugin_sha256": expected_plugin,
    "stable_member_sha256": expected_stable,
    "stable_member_byte_identical": True,
    "only_expected_new_members": [plugin_member, adapter_member],
    "record": {"baseline": base_record_check, "derived": derived_record_check},
    "unchanged_baseline_payload_count": len(unchanged),
    "script_spool_same_bytes": script.read_bytes() == spool.read_bytes(),
    "all_gates_pass": True,
}
assert payload["script_spool_same_bytes"]
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

wheel=$("${PYTHON}" - "${ARTIFACT_DIR}/c2-native-plugin-overlay-provenance.json" <<'PY'
import json
import sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(p["derived_wheel"]["filename"])
PY
)
sha256sum "${ARTIFACT_DIR}/c2-native-plugin-overlay-provenance.json" \
  "${ARTIFACT_DIR}/c2-native-plugin-overlay-manifest.json" "${ARTIFACT_DIR}/${wheel}" \
  "${ARTIFACT_DIR}/aot-attestation-job${SLURM_JOB_ID}.json" \
  "${ARTIFACT_DIR}/promotion-prerequisites-job${SLURM_JOB_ID}.json" \
  "${ARTIFACT_DIR}/wheel-driver-verification-job${SLURM_JOB_ID}.json" \
  "${ARTIFACT_DIR}/result-job${SLURM_JOB_ID}.json" \
  > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
