#!/usr/bin/env bash
# Build an installable, auditable native-C2 overlay wheel without touching the
# shared vLLM virtual environment.  This file only submits CPU-side packaging.
#SBATCH --job-name=c2-native-wheel
#SBATCH --partition=cpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/lcpu/85117379/slurm-c2-native-wheel-%j.log

set -euo pipefail

ARTIFACT_DIR=/home/lcpu/85117379/c2-native-wheel-artifacts-20260829
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
PACKAGER=/home/lcpu/85117379/build_native_c2_overlay_wheel_20260829.py
BASE_WHEEL=/home/lcpu/85117379/vllm-d4-wheel/vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl
DERIVED_CHECKOUT=/home/lcpu/85117379/vllm-d4-native-c2-20260829
AOT_LIBRARY=/home/lcpu/85117379/vllm-d4-native-c2-build-20260829/_C_stable_libtorch.abi3.so

mkdir -p "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1
trap 'rc=$?; printf "FINAL_RC=%s UTC=%s\\n" "${rc}" "$(date -u +%FT%TZ)"' EXIT

printf 'UTC=%s\n' "$(date -u +%FT%TZ)"
printf 'SLURM_JOB_ID=%s HOST=%s CPUS=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "${SLURM_CPUS_PER_TASK}"
printf 'BASE_WHEEL=%s\nDERIVED_CHECKOUT=%s\nAOT_LIBRARY=%s\nPACKAGER=%s\n' \
  "${BASE_WHEEL}" "${DERIVED_CHECKOUT}" "${AOT_LIBRARY}" "${PACKAGER}"

[[ -f "${BASE_WHEEL}" && -f "${AOT_LIBRARY}" && -f "${PACKAGER}" && -x "${PYTHON}" ]]
[[ -d "${DERIVED_CHECKOUT}" ]]
"${PYTHON}" --version
"${PYTHON}" -m py_compile "${PACKAGER}"
git -C "${DERIVED_CHECKOUT}" rev-parse HEAD
git -C "${DERIVED_CHECKOUT}" diff --check
sha256sum "${BASE_WHEEL}" "${AOT_LIBRARY}" "${PACKAGER}" \
  | tee "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"

"${PYTHON}" "${PACKAGER}" \
  --base-wheel "${BASE_WHEEL}" \
  --derived-checkout "${DERIVED_CHECKOUT}" \
  --aot-library "${AOT_LIBRARY}" \
  --artifact-dir "${ARTIFACT_DIR}" \
  | tee "${ARTIFACT_DIR}/result-job${SLURM_JOB_ID}.json"

"${PYTHON}" - "${ARTIFACT_DIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

artifact_dir = Path(sys.argv[1])
provenance_path = artifact_dir / "c2-native-overlay-provenance.json"
manifest_path = artifact_dir / "c2-native-overlay-manifest.json"
provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert provenance["schema"] == "c2-native-overlay-provenance-v1"
assert manifest["schema"] == "c2-native-overlay-manifest-v1"
assert provenance["experimental_overlay_same_distribution_version"] is True
assert manifest["experimental_overlay_same_distribution_version"] is True
assert provenance["derived_record_full_verification"]["all_payload_entries_validated"] is True
assert provenance["derived_record_full_verification"]["record_self_row_valid"] is True
assert provenance["unchanged_members"]["unapproved_changed_members"] == []
wheel = artifact_dir / provenance["derived_wheel"]["filename"]
assert wheel.is_file()
wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
assert wheel_hash == provenance["derived_wheel"]["sha256"]
assert wheel_hash == manifest["derived_wheel_sha256"]
assert all(entry["source_matches_zip"] for entry in provenance["overlay_members"])
print(json.dumps({
    "schema": "c2-native-overlay-wheel-driver-verification-v1",
    "wheel": str(wheel),
    "wheel_sha256": wheel_hash,
    "overlay_member_count": len(provenance["overlay_members"]),
    "record_row_count": provenance["derived_record_full_verification"]["record_row_count"],
}, sort_keys=True))
PY

sha256sum \
  "${ARTIFACT_DIR}/c2-native-overlay-provenance.json" \
  "${ARTIFACT_DIR}/c2-native-overlay-manifest.json" \
  "${ARTIFACT_DIR}/$(basename "${BASE_WHEEL}")" \
  > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
