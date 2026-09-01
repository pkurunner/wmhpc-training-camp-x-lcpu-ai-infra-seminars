#!/usr/bin/env bash
# Repack a v4 plugin in an isolated directory.  The reviewed generic packager
# preserves vllm/_C_stable_libtorch.abi3.so byte-for-byte and adds only the
# independent plugin plus the five v1-contract Python dispatch members.

#SBATCH --job-name=c2-native-plugin-v4-wheel
#SBATCH --partition=cpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v4-wheel-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONOPTIMIZE=0
export PYTHONPATH=

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v4-wheel-artifacts-20260829
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
PACKAGER=/home/lcpu/85117379/build_native_c2_plugin_overlay_wheel_20260829.py
BASE_WHEEL=/home/lcpu/85117379/vllm-d4-wheel/vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl
EXPECTED_BASELINE_SHA=91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06
EXPECTED_STABLE_MEMBER_SHA=cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442
EXPECTED_PLUGIN_ARCHIVE_MEMBER=vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_V4_SOURCE_SHA=96c0af14ed68261ada954688c1379d4a9d8a541f84997ef39eb12f373ea7f4c9
EXPECTED_PACKAGER_SHA=84f607ec46f4535a5d5e3506631d4eeb9a044f9e8d48f4aa0eec57984ca2866e

: "${C2_PLUGIN_CHECKOUT:?submit with exact v4 plugin checkout}"
: "${C2_PLUGIN_LIBRARY:?submit with independent v4 plugin shared library}"
: "${C2_PLUGIN_SOURCE:?submit with v4 patched native_c2_decode.cu}"
: "${C2_PLUGIN_ARCHIVE_MEMBER:?submit with exact vllm plugin member}"
: "${C2_EXPECTED_PLUGIN_SHA:?submit with reviewed v4 plugin SHA-256}"
[[ "${C2_EXPECTED_PLUGIN_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ "${EXPECTED_PACKAGER_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ "${C2_PLUGIN_ARCHIVE_MEMBER}" == "${EXPECTED_PLUGIN_ARCHIVE_MEMBER}" ]]
[[ "${C2_PLUGIN_CHECKOUT}" == /* && "${C2_PLUGIN_LIBRARY}" == /* && "${C2_PLUGIN_SOURCE}" == /* ]]
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

[[ -f "${BASE_WHEEL}" && -f "${PACKAGER}" && -x "${PYTHON}" ]]
[[ -d "${C2_PLUGIN_CHECKOUT}" && -f "${C2_PLUGIN_LIBRARY}" && -f "${C2_PLUGIN_SOURCE}" ]]
[[ "$(readlink -f "${C2_PLUGIN_SOURCE}")" == \
   "$(readlink -f "${C2_PLUGIN_CHECKOUT}/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu")" ]]
"${PYTHON}" -c 'raise SystemExit(0 if __debug__ else 1)'
"${PYTHON}" -m py_compile "${PACKAGER}"
git -C "${C2_PLUGIN_CHECKOUT}" rev-parse HEAD
git -C "${C2_PLUGIN_CHECKOUT}" diff --check
printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${EXPECTED_BASELINE_SHA}" "${BASE_WHEEL}" \
  "${C2_EXPECTED_PLUGIN_SHA}" "${C2_PLUGIN_LIBRARY}" \
  "${EXPECTED_V4_SOURCE_SHA}" "${C2_PLUGIN_SOURCE}" \
  "${EXPECTED_PACKAGER_SHA}" "${PACKAGER}" | sha256sum -c -
sha256sum "${BASE_WHEEL}" "${C2_PLUGIN_LIBRARY}" "${C2_PLUGIN_SOURCE}" "${PACKAGER}" \
  | tee "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"

"${PYTHON}" "${PACKAGER}" \
  --base-wheel "${BASE_WHEEL}" \
  --plugin-checkout "${C2_PLUGIN_CHECKOUT}" \
  --plugin-library "${C2_PLUGIN_LIBRARY}" \
  --plugin-archive-member "${C2_PLUGIN_ARCHIVE_MEMBER}" \
  --expected-plugin-sha256 "${C2_EXPECTED_PLUGIN_SHA}" \
  --artifact-dir "${ARTIFACT_DIR}" \
  | tee "${ARTIFACT_DIR}/result-job${SLURM_JOB_ID}.json"

"${PYTHON}" - "${ARTIFACT_DIR}" "${BASE_WHEEL}" "${C2_EXPECTED_PLUGIN_SHA}" \
  "${C2_PLUGIN_ARCHIVE_MEMBER}" "${EXPECTED_PACKAGER_SHA}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import zipfile

artifact_dir, baseline_wheel = map(Path, sys.argv[1:3])
expected_plugin_sha, plugin_member, expected_packager_sha = sys.argv[3:]
adapter_member = "vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py"
provenance = json.loads((artifact_dir / "c2-native-plugin-overlay-provenance.json").read_text(encoding="utf-8"))
manifest = json.loads((artifact_dir / "c2-native-plugin-overlay-manifest.json").read_text(encoding="utf-8"))
assert provenance["schema"] == "c2-native-plugin-overlay-provenance-v1"
assert manifest["schema"] == "c2-native-plugin-overlay-manifest-v1"
assert provenance["baseline_wheel"]["sha256"] == "91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06"
assert provenance["plugin_archive_member"] == plugin_member
assert provenance["plugin_expected_sha256"] == expected_plugin_sha
assert provenance["packager"]["sha256"] == expected_packager_sha
assert manifest["packager_sha256"] == expected_packager_sha
assert provenance["member_set"]["new_members"] == [plugin_member, adapter_member]
assert provenance["member_set"]["removed_members"] == []
assert provenance["derived_record_full_verification"]["all_payload_entries_validated"] is True
assert provenance["derived_record_full_verification"]["record_self_row_valid"] is True
assert provenance["unchanged_baseline_members"]["unapproved_changed_members"] == []
stable = provenance["baseline_stable_member"]
assert stable["byte_identical"] is True
assert stable["baseline_sha256"] == stable["derived_sha256"] == "cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442"
wheel = artifact_dir / provenance["derived_wheel"]["filename"]
wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
assert wheel_hash == provenance["derived_wheel"]["sha256"] == manifest["derived_wheel_sha256"]
members = {item["archive_member"]: item for item in provenance["overlay_members"]}
assert members[plugin_member]["kind"] == "added_native_c2_plugin"
assert members[plugin_member]["action"] == "add"
assert members[plugin_member]["zip_sha256"] == expected_plugin_sha
assert all(item["source_matches_zip"] for item in members.values())
expected_dispatch = {
    "vllm/_custom_ops.py": "b57dfa819424b89bed8bf39924d5a4a3d2ebf327b39f3ae7c2e9e65b61c2398d",
    "vllm/config/attention.py": "caf68c29d8d36324d252a66f0811a25b26d11da9860732359076fbb76625b5ae",
    "vllm/v1/attention/backends/registry.py": "cd539941417fd5faf6672676de0f8ae15c9adb2a915b47cdd508e154c34faea4",
    "vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py": "6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412",
    "vllm/models/minimax_m3/nvidia/sparse_attention_msa.py": "07d00372ca1995aa96159a0b9f8949b1b40f817a63f49224e88aafac1b722200",
}
assert provenance["expected_dispatch_member_sha256"] == expected_dispatch
assert manifest["expected_dispatch_member_sha256"] == expected_dispatch
for member, expected_hash in expected_dispatch.items():
    expected_kind = "added_python_dispatch" if member == adapter_member else "replaced_python_dispatch"
    expected_action = "add" if member == adapter_member else "replace"
    assert members[member]["kind"] == expected_kind
    assert members[member]["action"] == expected_action
    assert members[member]["source_sha256"] == expected_hash
    assert members[member]["expected_post_patch_sha256"] == expected_hash
    assert members[member]["zip_sha256"] == expected_hash
with zipfile.ZipFile(baseline_wheel, "r") as base, zipfile.ZipFile(wheel, "r") as derived:
    assert adapter_member not in base.namelist()
    assert adapter_member in derived.namelist()
    assert base.read("vllm/_C_stable_libtorch.abi3.so") == derived.read("vllm/_C_stable_libtorch.abi3.so")
print(json.dumps({"schema": "c2-native-plugin-v4-overlay-wheel-driver-v1", "wheel": str(wheel),
                  "wheel_sha256": wheel_hash, "plugin_sha256": expected_plugin_sha,
                  "stable_member_sha256": stable["baseline_sha256"], "all_gates_pass": True}, sort_keys=True))
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
  > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"


