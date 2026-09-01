#!/usr/bin/env bash
# Build an experimental, local overlay wheel from the AOT-validated v6 DSO.
# This is a repackaging/verification gate only: it is neither an upstream
# change nor a release artifact.  The base d4 wheel and its stable extension
# remain immutable; only the reviewed native plugin and dispatch overlay are
# allowed to differ.
#
# Submit only with the reviewed script digest, for example:
#   sbatch --export=ALL,C2_EXPECTED_RUNTIME_SCRIPT_SHA=<sha256> \
#     build_native_c2_plugin_v6_overlay_wheel.slurm

#SBATCH --job-name=c2-native-plugin-v6-wheel
#SBATCH --partition=cpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v6-wheel-artifacts-20260830/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONOPTIMIZE=0
export PYTHONPATH=

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v6-wheel-artifacts-20260830
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID:?must run under Slurm}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
PACKAGER=/home/lcpu/85117379/build_native_c2_plugin_overlay_wheel_20260829.py
BASE_WHEEL=/home/lcpu/85117379/vllm-d4-wheel/vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl
EXPECTED_BASELINE_SHA=91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06
EXPECTED_STABLE_MEMBER_SHA=cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442
EXPECTED_PLUGIN_ARCHIVE_MEMBER=vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_ADAPTER_MEMBER=vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py
EXPECTED_D4_COMMIT=d4da0c55af3aa231b6209bf77871f3ed36eab0d2
EXPECTED_V6_SOURCE_SHA=c7f05928fd2ba3a0e29d3154f2bf56849e9d7ad6fd3eddc24b0890722e117f02
EXPECTED_V6_PLUGIN_SHA=0d5492bee11b2a785d5f16681614b0a1fc6c74ddb33351d44f50b124580e21a4
EXPECTED_PACKAGER_SHA=84f607ec46f4535a5d5e3506631d4eeb9a044f9e8d48f4aa0eec57984ca2866e
EXPECTED_ADAPTER_SHA=6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412

# Immutable AOT inputs from the accepted v6 build.  Pinning the three text
# artifacts by content guards against a later replacement that preserves only
# a filename; the provenance file also re-verifies all original AOT inputs.
EXPECTED_V6_CHECKOUT=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v6-register-numerator-20260830-job12278
EXPECTED_V6_PLUGIN_LIBRARY=/home/lcpu/85117379/c2-native-plugin-v6-aot-artifacts-20260830/job12278/vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_V6_BUILD_PLUGIN=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v6-register-numerator-build-20260830-job12278/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_V6_PLUGIN_SOURCE=${EXPECTED_V6_CHECKOUT}/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
AOT_ARTIFACT_DIR=/home/lcpu/85117379/c2-native-plugin-v6-aot-artifacts-20260830/job12278
AOT_FINAL_STATUS=${AOT_ARTIFACT_DIR}/final-status-job12278.txt
AOT_RESOURCE_GATE=${AOT_ARTIFACT_DIR}/plugin-resource-gate-job12278.txt
AOT_PROVENANCE=${AOT_ARTIFACT_DIR}/plugin-v6-provenance-job12278.sha256
EXPECTED_AOT_FINAL_STATUS_SHA=898d77659852171987f11ce22ee29de5b1683cc463b80a81308f009a4a62e032
EXPECTED_AOT_RESOURCE_GATE_SHA=2f5d0898888446b79808b02c709dd2f0b33309679629cdc335c263482a965560
EXPECTED_AOT_PROVENANCE_SHA=e2067f586f338139283fdf54b9cd6cc2b57ac5a2d189f04bfa3dfc732e0839d8

: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?reviewed v6 wheel-script SHA-256 required}"
[[ "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" =~ ^[0-9a-f]{64}$ ]]
for digest in "${EXPECTED_BASELINE_SHA}" "${EXPECTED_STABLE_MEMBER_SHA}" \
  "${EXPECTED_V6_SOURCE_SHA}" "${EXPECTED_V6_PLUGIN_SHA}" \
  "${EXPECTED_PACKAGER_SHA}" "${EXPECTED_ADAPTER_SHA}" \
  "${EXPECTED_AOT_FINAL_STATUS_SHA}" "${EXPECTED_AOT_RESOURCE_GATE_SHA}" \
  "${EXPECTED_AOT_PROVENANCE_SHA}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]
done

# sbatch places an immutable copy of this script in the slurmd job spool.  Do
# not trust a submit-directory path after queueing: verify the executed spool
# copy against the reviewed digest and keep both identities in the manifest.
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SUBMITTED_SCRIPT=/home/lcpu/85117379/build_native_c2_plugin_v6_overlay_wheel_20260830.slurm
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
[[ -d "${EXPECTED_V6_CHECKOUT}" && -f "${EXPECTED_V6_PLUGIN_LIBRARY}" \
   && -f "${EXPECTED_V6_BUILD_PLUGIN}" && -f "${EXPECTED_V6_PLUGIN_SOURCE}" ]]
[[ -f "${AOT_FINAL_STATUS}" && -f "${AOT_RESOURCE_GATE}" && -f "${AOT_PROVENANCE}" ]]
"${PYTHON}" -c 'raise SystemExit(0 if __debug__ else 1)'
"${PYTHON}" -m py_compile "${PACKAGER}"
git -C "${EXPECTED_V6_CHECKOUT}" rev-parse HEAD
[[ "$(git -C "${EXPECTED_V6_CHECKOUT}" rev-parse HEAD)" == "${EXPECTED_D4_COMMIT}" ]]
git -C "${EXPECTED_V6_CHECKOUT}" diff --check
[[ "$(readlink -f "${EXPECTED_V6_PLUGIN_SOURCE}")" == \
   "$(readlink -f "${EXPECTED_V6_CHECKOUT}/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu")" ]]

# The AOT build attestation has to be fully successful and retain the v6
# register-numerator resource gate before its DSO may enter an overlay wheel.
grep -qxE 'FINAL_RC=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${AOT_FINAL_STATUS}"
grep -qx 'REG=64' "${AOT_RESOURCE_GATE}"
grep -qx 'STACK=0' "${AOT_RESOURCE_GATE}"
grep -qx 'SHARED=30560' "${AOT_RESOURCE_GATE}"
grep -qx 'LOCAL=0' "${AOT_RESOURCE_GATE}"
printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${EXPECTED_BASELINE_SHA}" "${BASE_WHEEL}" \
  "${EXPECTED_V6_PLUGIN_SHA}" "${EXPECTED_V6_PLUGIN_LIBRARY}" \
  "${EXPECTED_V6_PLUGIN_SHA}" "${EXPECTED_V6_BUILD_PLUGIN}" \
  "${EXPECTED_V6_SOURCE_SHA}" "${EXPECTED_V6_PLUGIN_SOURCE}" \
  "${EXPECTED_PACKAGER_SHA}" "${PACKAGER}" \
  "${EXPECTED_ADAPTER_SHA}" "${EXPECTED_V6_CHECKOUT}/${EXPECTED_ADAPTER_MEMBER}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SUBMITTED_SCRIPT}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" \
  "${EXPECTED_AOT_FINAL_STATUS_SHA}" "${AOT_FINAL_STATUS}" \
  "${EXPECTED_AOT_RESOURCE_GATE_SHA}" "${AOT_RESOURCE_GATE}" \
  > "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
printf '%s  %s\n' "${EXPECTED_AOT_PROVENANCE_SHA}" "${AOT_PROVENANCE}" | sha256sum -c -
# This verifies the AOT source, both DSO paths, and every frozen AOT build
# input named in the original provenance file before repackaging starts.
sha256sum -c "${AOT_PROVENANCE}"
sha256sum "${BASE_WHEEL}" "${EXPECTED_V6_PLUGIN_LIBRARY}" "${EXPECTED_V6_BUILD_PLUGIN}" \
  "${EXPECTED_V6_PLUGIN_SOURCE}" "${PACKAGER}" \
  "${EXPECTED_V6_CHECKOUT}/${EXPECTED_ADAPTER_MEMBER}" "${SUBMITTED_SCRIPT}" "${SCRIPT_PATH}" \
  "${AOT_FINAL_STATUS}" "${AOT_RESOURCE_GATE}" "${AOT_PROVENANCE}" \
  > "${ARTIFACT_DIR}/all-inputs-job${SLURM_JOB_ID}.sha256"

"${PYTHON}" - "${ARTIFACT_DIR}/aot-attestation-job${SLURM_JOB_ID}.json" \
  "${AOT_FINAL_STATUS}" "${AOT_RESOURCE_GATE}" "${AOT_PROVENANCE}" \
  "${EXPECTED_V6_PLUGIN_LIBRARY}" "${EXPECTED_V6_BUILD_PLUGIN}" \
  "${EXPECTED_V6_PLUGIN_SOURCE}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

out, final_status, resource_gate, provenance, plugin, build_plugin, source = map(Path, sys.argv[1:])
expected_plugin = "0d5492bee11b2a785d5f16681614b0a1fc6c74ddb33351d44f50b124580e21a4"
expected_source = "c7f05928fd2ba3a0e29d3154f2bf56849e9d7ad6fd3eddc24b0890722e117f02"
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert re.fullmatch(r"FINAL_RC=0 UTC=\d{4}-\d{2}-\d{2}T[\d:]+Z\n", final_status.read_text(encoding="utf-8"))
resource = resource_gate.read_text(encoding="utf-8").splitlines()
assert {"REG=64", "STACK=0", "SHARED=30560", "LOCAL=0"}.issubset(resource)
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
    "schema": "c2-native-plugin-v6-aot-attestation-v1",
    "aot_job_id": 12278,
    "aot_final_rc": 0,
    "resource_gate": {"REG": 64, "STACK": 0, "SHARED": 30560, "LOCAL": 0},
    "v6_plugin_sha256": expected_plugin,
    "v6_source_sha256": expected_source,
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
  --plugin-checkout "${EXPECTED_V6_CHECKOUT}" \
  --plugin-library "${EXPECTED_V6_PLUGIN_LIBRARY}" \
  --plugin-archive-member "${EXPECTED_PLUGIN_ARCHIVE_MEMBER}" \
  --expected-plugin-sha256 "${EXPECTED_V6_PLUGIN_SHA}" \
  --artifact-dir "${ARTIFACT_DIR}" \
  | tee "${ARTIFACT_DIR}/result-job${SLURM_JOB_ID}.json"

# Re-verify the produced archive independently of the packager's provenance:
# all RECORD payload entries, exact member deltas, each unchanged baseline byte,
# the frozen stable DSO, and every reviewed dispatch/member digest must agree.
"${PYTHON}" - "${ARTIFACT_DIR}" "${BASE_WHEEL}" "${EXPECTED_V6_PLUGIN_LIBRARY}" \
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
expected_plugin = "0d5492bee11b2a785d5f16681614b0a1fc6c74ddb33351d44f50b124580e21a4"
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
    "schema": "c2-native-plugin-v6-overlay-wheel-driver-verification-v1",
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
  "${ARTIFACT_DIR}/wheel-driver-verification-job${SLURM_JOB_ID}.json" \
  "${ARTIFACT_DIR}/result-job${SLURM_JOB_ID}.json" \
  > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
