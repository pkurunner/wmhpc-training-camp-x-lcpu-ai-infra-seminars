#!/usr/bin/env bash
# Fresh-target integration and observation gate for the v6 register-resident
# numerator overlay.  This script deliberately makes two different statements:
#
# * every v6 full-backend seed is a new Python process which imports only the
#   job-scoped overlay target plus a staged ``harness`` support package, and
#   compares the real MiniMaxM3SparseMSAImpl.forward route against Triton in
#   that same fresh process; and
# * the promoted v5 fresh result is a frozen, cross-job directional reference.
#   The script never loads v5 and v6 DSOs together, and never calls the latter
#   an in-process A/B comparison.
#
# A Triton parity miss is reported but is intentionally not an integration
# failure.  The hard exit gate is wheel identity, clean installation, loader
# surface preservation, eight-seed real-backend correctness, and the companion
# eight-seed direct bitwise result for this exact v6 DSO.

#SBATCH --job-name=c2-native-plugin-v6-fresh
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v6-fresh-backend-artifacts-20260830/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONOPTIMIZE=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=
export PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v6-fresh-backend-artifacts-20260830
RUN_ID=${C2_RUN_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID is required unless C2_RUN_ID is supplied}}
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "unsafe C2 run identifier: ${RUN_ID}" >&2
  exit 2
}
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${RUN_ID}
RUNTIME_ROOT=/home/lcpu/85117379/c2-native-plugin-v6-fresh-runtime-20260830
RUNTIME=${RUNTIME_ROOT}/job${RUN_ID}
SUPPORT=${RUNTIME}/c2_support
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
UV=/usr/local/bin/uv
BASELINE_VLLM=/home/lcpu/85117379/venv-vllm-d4da0c5/lib/python3.12/site-packages/vllm
C2_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02/team/c2_msa_decode
HARNESS=/home/lcpu/85117379/native_c2_full_backend_bench_20260829.py
HARNESS_COPY=${ARTIFACT_DIR}/native_c2_full_backend_bench.py
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")

EXPECTED_HARNESS_SHA=7883edc25df48e3b69a9f4948775a1baafd3ba0b1bd17edcfeac6644aa7b4762
EXPECTED_BASELINE_WHEEL_SHA=91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06
EXPECTED_STABLE_MEMBER_SHA=cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442
EXPECTED_PACKAGER_SHA=84f607ec46f4535a5d5e3506631d4eeb9a044f9e8d48f4aa0eec57984ca2866e
EXPECTED_WHEEL_FILENAME=vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl
EXPECTED_PLUGIN_MEMBER=vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_ADAPTER_MEMBER=vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py
EXPECTED_LOADER_MODULE=vllm.models.minimax_m3.nvidia.msa_native_c2_decode
EXPECTED_LOADER_CALLABLE=_load_native_c2_plugin_once
EXPECTED_V6_AOT_DSO=/home/lcpu/85117379/c2-native-plugin-v6-aot-artifacts-20260830/job12278/vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_V6_AOT_DSO_SHA=0d5492bee11b2a785d5f16681614b0a1fc6c74ddb33351d44f50b124580e21a4
EXPECTED_V5_FRESH_RESULT=/home/lcpu/85117379/c2-native-plugin-v5-wheel-runtime-artifacts-20260829/job11775/plugin-v5-wheel-full-backend-job11775.json
EXPECTED_V5_FRESH_NATIVE_MEDIAN_MS=0.11718399822711945
EXPECTED_SUPPORT_MANIFEST_SHA=dc9f1252285e82e30887b5124bc1e9b895dfeaed49d522aa8ebfb5a00da67243
SEEDS=(17 23 42 2024 314159 20260801 20260815 20260829)

BASELINE_OPS=${ARTIFACT_DIR}/baseline-stable-ops-job${RUN_ID}.json
INSTALLED_MEMBERS=${ARTIFACT_DIR}/installed-members-job${RUN_ID}.json
INSTALLED_PRE=${ARTIFACT_DIR}/installed-pre-plugin-ops-job${RUN_ID}.json
INSTALLED_POST=${ARTIFACT_DIR}/installed-post-plugin-ops-job${RUN_ID}.json
SUPPORT_SOURCE_MANIFEST=${ARTIFACT_DIR}/support-source-job${RUN_ID}.sha256
SUPPORT_INSTALLED_MANIFEST=${ARTIFACT_DIR}/support-installed-job${RUN_ID}.sha256
UUID_MANIFEST=${ARTIFACT_DIR}/seed-gpu-uuids-job${RUN_ID}.csv
SUMMARY=${ARTIFACT_DIR}/plugin-v6-fresh-backend-summary-job${RUN_ID}.json
DECISION=${ARTIFACT_DIR}/plugin-v6-fresh-backend-decision-job${RUN_ID}.json
GPU_PRE=${ARTIFACT_DIR}/gpu-pre-job${RUN_ID}.csv
GPU_POST=${ARTIFACT_DIR}/gpu-post-job${RUN_ID}.csv
GPU_PRE_APPS=${ARTIFACT_DIR}/gpu-pre-compute-apps-job${RUN_ID}.csv
GPU_POST_APPS=${ARTIFACT_DIR}/gpu-post-compute-apps-job${RUN_ID}.csv

: "${C2_WHEEL_ABSPATH:?absolute job-specific v6 overlay wheel is required}"
: "${C2_PROVENANCE_ABSPATH:?absolute matching v6 overlay provenance is required}"
: "${C2_EXPECTED_WHEEL_SHA:?reviewed v6 wheel SHA-256 is required}"
: "${C2_EXPECTED_PROVENANCE_SHA:?reviewed v6 provenance SHA-256 is required}"
: "${C2_EXPECTED_PLUGIN_SHA:?reviewed v6 job12278 DSO SHA-256 is required}"
: "${C2_EXPECTED_PACKAGER_SHA:?reviewed overlay packager SHA-256 is required}"
: "${C2_EXPECTED_HARNESS_SHA:?reviewed real-backend harness SHA-256 is required}"
: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?reviewed fresh-backend script SHA-256 is required}"
: "${C2_V6_DIRECT_STRESS_AGGREGATE:?passed eight-seed v6 direct aggregate JSON is required for bitwise evidence}"
: "${C2_EXPECTED_V6_DIRECT_STRESS_AGGREGATE_SHA:?reviewed direct aggregate SHA-256 is required}"
: "${C2_V6_DIRECT_BITWISE_RESULT:?candidate-v6 eight-seed direct worker JSON is required for explicit bitwise evidence}"
: "${C2_EXPECTED_V6_DIRECT_BITWISE_RESULT_SHA:?reviewed direct bitwise worker SHA-256 is required}"
: "${C2_V5_FRESH_RESULT:?frozen promoted-v5 fresh backend JSON is required for boundary recording}"
: "${C2_EXPECTED_V5_FRESH_SHA:?reviewed promoted-v5 fresh backend JSON SHA-256 is required}"

for digest in "${C2_EXPECTED_WHEEL_SHA}" "${C2_EXPECTED_PROVENANCE_SHA}" \
  "${C2_EXPECTED_PLUGIN_SHA}" "${C2_EXPECTED_PACKAGER_SHA}" \
  "${C2_EXPECTED_HARNESS_SHA}" "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" \
  "${C2_EXPECTED_V6_DIRECT_STRESS_AGGREGATE_SHA}" "${C2_EXPECTED_V6_DIRECT_BITWISE_RESULT_SHA}" \
  "${C2_EXPECTED_V5_FRESH_SHA}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid SHA-256: ${digest}" >&2; exit 2; }
done
[[ "${C2_EXPECTED_PLUGIN_SHA}" == "${EXPECTED_V6_AOT_DSO_SHA}" ]]
[[ "${C2_EXPECTED_PACKAGER_SHA}" == "${EXPECTED_PACKAGER_SHA}" ]]
[[ "${C2_EXPECTED_HARNESS_SHA}" == "${EXPECTED_HARNESS_SHA}" ]]
[[ "${C2_V5_FRESH_RESULT}" == "${EXPECTED_V5_FRESH_RESULT}" ]]
[[ "${C2_WHEEL_ABSPATH}" == /* && "${C2_PROVENANCE_ABSPATH}" == /* ]]
WHEEL=$(readlink -f "${C2_WHEEL_ABSPATH}")
PROVENANCE=$(readlink -f "${C2_PROVENANCE_ABSPATH}")
V6_DIRECT=$(readlink -f "${C2_V6_DIRECT_STRESS_AGGREGATE}")
V6_BITWISE=$(readlink -f "${C2_V6_DIRECT_BITWISE_RESULT}")
V5_FRESH=$(readlink -f "${C2_V5_FRESH_RESULT}")
[[ "$(basename "${WHEEL}")" == "${EXPECTED_WHEEL_FILENAME}" ]]
[[ "$(basename "${PROVENANCE}")" == c2-native-plugin-overlay-provenance.json ]]
[[ "$(dirname "${WHEEL}")" == "$(dirname "${PROVENANCE}")" ]]
[[ ! -e "${ARTIFACT_DIR}" && ! -e "${RUNTIME}" ]] || {
  echo "refusing to reuse v6 runtime/artifact path: ${RUN_ID}" >&2; exit 2;
}
[[ "${RUNTIME}" != "${C2_ROOT}" && "${RUNTIME}" != "${BASELINE_VLLM}" ]]
mkdir -p "${ARTIFACT_ROOT}" "${RUNTIME_ROOT}" "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${RUN_ID}.log") 2>&1
finalize() {
  local rc=$?
  trap - EXIT
  set +e
  printf 'FINAL_RC=%s UTC=%s RUN_ID=%s\n' "${rc}" "$(date -u +%FT%TZ)" "${RUN_ID}" | \
    tee "${ARTIFACT_DIR}/final-status-job${RUN_ID}.txt"
  exit "${rc}"
}
trap finalize EXIT

allocated_gpu_selector() {
  local selector=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}
  [[ "${selector}" != *,* ]] || {
    echo "expected one CUDA_VISIBLE_DEVICES entry, got: ${selector}" >&2
    return 1
  }
  [[ "${selector}" =~ ^[0-9]+$ || "${selector}" =~ ^GPU-[0-9A-Fa-f-]+$ ]] || {
    echo "unsupported CUDA_VISIBLE_DEVICES entry: ${selector}" >&2
    return 1
  }
  printf '%s\n' "${selector}"
}
compute_apps() {
  local selector
  selector=$(allocated_gpu_selector)
  nvidia-smi --id "${selector}" --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}
visible_gpu_uuid() {
  local selector uuid
  local -a uuids
  selector=$(allocated_gpu_selector)
  mapfile -t uuids < <(nvidia-smi --id "${selector}" --query-gpu=uuid --format=csv,noheader,nounits)
  [[ ${#uuids[@]} -eq 1 ]] || { printf 'expected exactly one allocated GPU, got %s\n' "${#uuids[@]}" >&2; return 1; }
  uuid=${uuids[0]}
  [[ "${uuid}" =~ ^GPU- ]] || { echo "invalid GPU UUID: ${uuid}" >&2; return 1; }
  printf '%s\n' "${uuid}"
}
snapshot() {
  local stage=$1 gpu_file=$2 apps_file=$3 selector
  selector=$(allocated_gpu_selector)
  printf '\n===== %s UTC=%s =====\n' "${stage}" "$(date -u +%FT%TZ)"
  nvidia-smi --id "${selector}" --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap \
    --format=csv,noheader,nounits | tee "${gpu_file}"
  compute_apps | tee "${apps_file}"
}

snapshot PRE "${GPU_PRE}" "${GPU_PRE_APPS}"
gpu_line=$(<"${GPU_PRE}")
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* ]]
EXPECTED_GPU_UUID=$(visible_gpu_uuid)
[[ -z "$(<"${GPU_PRE_APPS}")" ]] || { echo 'GPU is not clean before fresh validation' >&2; exit 1; }
[[ -f "${WHEEL}" && -f "${PROVENANCE}" && -f "${HARNESS}" && -f "${SCRIPT_PATH}" ]]
[[ -f "${V6_DIRECT}" && -f "${V6_BITWISE}" && -f "${V5_FRESH}" && -f "${EXPECTED_V6_AOT_DSO}" ]]
[[ -d "${C2_ROOT}/harness" && ! -e "${C2_ROOT}/vllm" ]]
[[ -d "${C2_ROOT}/vllm_msa_ref" ]]
[[ -d "${BASELINE_VLLM}" && -x "${UV}" && -x "${PYTHON}" ]]
"${UV}" --version
"${PYTHON}" -c 'raise SystemExit(0 if __debug__ else 1)'
[[ "$(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == 1 ]]
printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${C2_EXPECTED_WHEEL_SHA}" "${WHEEL}" \
  "${C2_EXPECTED_PROVENANCE_SHA}" "${PROVENANCE}" \
  "${C2_EXPECTED_PLUGIN_SHA}" "${EXPECTED_V6_AOT_DSO}" \
  "${C2_EXPECTED_HARNESS_SHA}" "${HARNESS}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" \
  "${C2_EXPECTED_V6_DIRECT_STRESS_AGGREGATE_SHA}" "${V6_DIRECT}" \
  "${C2_EXPECTED_V6_DIRECT_BITWISE_RESULT_SHA}" "${V6_BITWISE}" \
  "${C2_EXPECTED_V5_FRESH_SHA}" "${V5_FRESH}" \
  "${EXPECTED_BASELINE_WHEEL_SHA}" /home/lcpu/85117379/vllm-d4-wheel/"${EXPECTED_WHEEL_FILENAME}" \
  "${C2_EXPECTED_PACKAGER_SHA}" /home/lcpu/85117379/build_native_c2_plugin_overlay_wheel_20260829.py \
  > "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256"
printf '%s  %s\n' \
  'df9dd0b01c3e91afeb1a490785fab1a0554a16f001e118005192f55f891358a8' 'harness/__init__.py' \
  '38faefb89b38e3bc26b135b762a6bbbd813dcc695423982b423a806fe41c11d6' 'harness/data.py' \
  '712c1632367700137392d2c43bd5c6962130d945f01f1f585cc6338788ce53cc' 'harness/reference.py' \
  '6b0a40463b4f1c05ca53649d6689377b4f755d1efb58b10700dad05705b53f71' 'harness/triton_baseline.py' \
  '5db5d5e692044e54bb57418cbb7dcf5c9cc94f789eaccf128eec6fa8df86d967' 'vllm_msa_ref/sparse_attn.py' \
  > "${SUPPORT_SOURCE_MANIFEST}"
[[ "$(sha256sum "${SUPPORT_SOURCE_MANIFEST}" | awk '{print $1}')" == "${EXPECTED_SUPPORT_MANIFEST_SHA}" ]]
(cd "${C2_ROOT}" && sha256sum -c "${SUPPORT_SOURCE_MANIFEST}")
sha256sum "${WHEEL}" "${PROVENANCE}" "${EXPECTED_V6_AOT_DSO}" "${HARNESS}" "${SCRIPT_PATH}" \
  "${V6_DIRECT}" "${V6_BITWISE}" "${V5_FRESH}" "${SUPPORT_SOURCE_MANIFEST}" \
  >> "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256"

# The accompanying direct aggregate is deliberately only a bitwise/ABI
# companion.  Its promotion outcome is not a prerequisite: an otherwise valid
# v6 may be slower than v5 or Triton and must still finish this integration job.
"${PYTHON}" - "${V6_DIRECT}" "${V6_BITWISE}" "${EXPECTED_V6_AOT_DSO_SHA}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
bitwise = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_dso = sys.argv[3]
assert payload["schema"] == "c2-native-plugin-v6-vs-v5-fixed-fixture-aggregate-v1"
assert payload["all_correctness_and_comparability_gates_pass"] is True
checksums = payload["input_checksums"]
assert checksums["candidate_v6_job12278_plugin"] == expected_dso
assert checksums["reference_v5_job11760_plugin"] != expected_dso
rounds = payload["round_order"]
assert rounds == ["reference_v5_a", "candidate_v6_b1", "candidate_v6_b2", "reference_v5_a2"]
assert payload["process_isolation"]["same_python_process"] is False
assert len(payload["gpu_uuids"]) == 4 and len(set(payload["gpu_uuids"])) == 1
for series in (payload["reference_seed_mean_ms"], payload["candidate_seed_mean_ms"]):
    assert len(series) == 8 and all(math.isfinite(float(x)) and float(x) > 0 for x in series)
assert bitwise["schema"] == "c2-native-plugin-v6-fixed-fixture-worker-v1"
assert bitwise["role"] in {"candidate_v6_b1", "candidate_v6_b2"}
assert bitwise["all_gates_pass"] is True
assert bitwise["operator_library_sha256"] == expected_dso
assert len(bitwise["seeds"]) == 8
for record in bitwise["seeds"]:
    assert record["all_gates_pass"] is True
    assert record["correctness"]["allclose"] is True
    assert record["caller_output"]["pointer_unchanged"] is True
    assert record["caller_output"]["timing_pointer_unchanged"] is True
    assert record["stability"]["bitwise_repeatable"] is True
PY

"${UV}" pip install --python "${PYTHON}" --target "${RUNTIME}" --no-deps "${WHEEL}"
[[ ! -L "${RUNTIME}" ]]
[[ -z "$(find "${RUNTIME}" -type l -print -quit)" ]] || { echo 'fresh target contains a symlink' >&2; exit 2; }
install -D -m 0644 "${HARNESS}" "${HARNESS_COPY}"

# Verify both the wheel RECORD and every installed wheel payload independently;
# provenance's own RECORD flags are necessary but not accepted as sufficient.
# An installer legitimately rewrites RECORD and adds five audited entries, so
# every original payload except RECORD must remain byte-identical while the
# installed RECORD is validated as a second, complete integrity manifest.
"${PYTHON}" - "${RUNTIME}" "${WHEEL}" "${PROVENANCE}" "${INSTALLED_MEMBERS}" \
  "${C2_EXPECTED_PLUGIN_SHA}" "${EXPECTED_V6_AOT_DSO}" "${C2_EXPECTED_PACKAGER_SHA}" <<'PY'
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
import zipfile

runtime, wheel, provenance_path, output = map(Path, sys.argv[1:5])
plugin_sha, expected_source_path, expected_packager_sha = sys.argv[5:]
plugin_member = "vllm/_native_c2_msa_decode_plugin.abi3.so"
adapter_member = "vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py"
stable_member = "vllm/_C_stable_libtorch.abi3.so"
expected_dispatch = {
    "vllm/_custom_ops.py": "b57dfa819424b89bed8bf39924d5a4a3d2ebf327b39f3ae7c2e9e65b61c2398d",
    "vllm/config/attention.py": "caf68c29d8d36324d252a66f0811a25b26d11da9860732359076fbb76625b5ae",
    adapter_member: "6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412",
    "vllm/models/minimax_m3/nvidia/sparse_attention_msa.py": "07d00372ca1995aa96159a0b9f8949b1b40f817a63f49224e88aafac1b722200",
    "vllm/v1/attention/backends/registry.py": "cd539941417fd5faf6672676de0f8ae15c9adb2a915b47cdd508e154c34faea4",
}

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def validate_record(z: zipfile.ZipFile) -> tuple[str, list[str]]:
    records = [name for name in z.namelist() if name.endswith(".dist-info/RECORD")]
    assert len(records) == 1, records
    record_member = records[0]
    rows = list(csv.reader(io.StringIO(z.read(record_member).decode("utf-8"))))
    names = [row[0] for row in rows]
    assert len(rows) == len(names) == len(set(names))
    payload = sorted(name for name in z.namelist() if not name.endswith("/"))
    assert set(names) == set(payload)
    for row in rows:
        assert len(row) == 3
        name, encoded, size = row
        if name == record_member:
            assert encoded == size == ""
            continue
        algorithm, encoded_digest = encoded.split("=", 1)
        assert algorithm == "sha256"
        raw = z.read(name)
        expected = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        assert encoded_digest == expected and int(size) == len(raw), name
    return record_member, payload

def validate_installed_record(
    runtime: Path, record_member: str, wheel_payload: list[str]
) -> tuple[list[str], list[dict[str, str | int]]]:
    dist_info = record_member.rsplit("/", 1)[0]
    installer_added = {
        "bin/vllm",
        f"{dist_info}/INSTALLER",
        f"{dist_info}/REQUESTED",
        f"{dist_info}/direct_url.json",
        f"{dist_info}/uv_cache.json",
    }
    rows = list(csv.reader((runtime / record_member).read_text(encoding="utf-8").splitlines()))
    names = [row[0] for row in rows]
    assert len(rows) == len(names) == len(set(names))
    assert set(names) == set(wheel_payload) | installer_added
    verified = []
    root = runtime.resolve()
    for row in rows:
        assert len(row) == 3
        name, encoded, size = row
        path = (runtime / name).resolve(strict=True)
        assert path.is_relative_to(root) and path.is_file(), name
        if name == record_member:
            assert encoded == size == ""
        else:
            algorithm, encoded_digest = encoded.split("=", 1)
            assert algorithm == "sha256"
            raw = path.read_bytes()
            expected = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
            assert encoded_digest == expected and int(size) == len(raw), name
        verified.append({"member": name, "path": str(path), "size": path.stat().st_size})
    return sorted(installer_added), verified

p = json.loads(provenance_path.read_text(encoding="utf-8"))
assert p["schema"] == "c2-native-plugin-overlay-provenance-v1"
assert p["baseline_wheel"]["sha256"] == "91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06"
assert Path(p["derived_wheel"]["path"]).resolve() == wheel.resolve()
assert p["plugin_archive_member"] == plugin_member
assert p["plugin_expected_sha256"] == plugin_sha
assert p["packager"]["sha256"] == expected_packager_sha
assert p["member_set"]["new_members"] == [plugin_member, adapter_member]
assert p["member_set"]["removed_members"] == []
assert p["unchanged_baseline_members"]["unapproved_changed_members"] == []
for key in ("base_record_full_verification", "derived_record_full_verification"):
    assert p[key]["all_payload_entries_validated"] is True
    assert p[key]["record_self_row_valid"] is True
stable = p["baseline_stable_member"]
assert stable["archive_member"] == stable_member
assert stable["byte_identical"] is True
assert stable["baseline_sha256"] == stable["derived_sha256"] == "cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442"
members = {row["archive_member"]: row for row in p["overlay_members"]}
assert set(members) == {plugin_member, *expected_dispatch}
plugin = members[plugin_member]
assert plugin["action"] == "add" and plugin["kind"] == "added_native_c2_plugin"
assert plugin["source_matches_zip"] is True
assert Path(plugin["source_path"]).resolve() == Path(expected_source_path).resolve()
assert plugin["source_sha256"] == plugin["zip_sha256"] == plugin_sha
assert digest(Path(expected_source_path).read_bytes()) == plugin_sha
assert p["expected_dispatch_member_sha256"] == expected_dispatch
for member, expected in expected_dispatch.items():
    row = members[member]
    assert row["source_matches_zip"] is True
    assert row["source_sha256"] == row["zip_sha256"] == expected
    assert row["expected_post_patch_sha256"] == expected
    assert row["action"] == ("add" if member == adapter_member else "replace")

assert digest(wheel.read_bytes()) == p["derived_wheel"]["sha256"]
with zipfile.ZipFile(wheel, "r") as z:
    record_member, payload = validate_record(z)
    installer_added, installed_record_rows = validate_installed_record(runtime, record_member, payload)
    installed = []
    for member in payload:
        path = runtime / member
        assert path.is_file(), path
        if member != record_member:
            assert path.read_bytes() == z.read(member), member
        installed.append({"archive_member": member, "path": str(path), "sha256": digest(path.read_bytes())})
    assert digest(z.read(plugin_member)) == plugin_sha
    assert (runtime / stable_member).read_bytes() == z.read(stable_member)
    assert (runtime / stable_member).read_bytes() == Path(runtime / stable_member).read_bytes()
actual_files = {str(path.relative_to(runtime)) for path in runtime.rglob("*") if path.is_file()}
installed_record_members = {row["member"] for row in installed_record_rows}
assert actual_files == installed_record_members | {".lock"}
assert (runtime / ".lock").is_file() and not (runtime / ".lock").is_symlink()
output.write_text(json.dumps({
    "schema": "c2-native-plugin-v6-installed-members-v1",
    "runtime": str(runtime), "wheel_record_member": record_member,
    "wheel_record_full_verification": True,
    "installed_record_full_verification": True,
    "wheel_payload_except_installer_rewritten_record_byte_identical": True,
    "installer_added_record_members": installer_added,
    "unrecorded_installer_files": [".lock"],
    "plugin_sha256": plugin_sha, "plugin_source_path": expected_source_path,
    "stable_member_sha256": stable["baseline_sha256"],
    "wheel_payload_member_count": len(installed), "wheel_payload_members": installed,
    "installed_record_member_count": len(installed_record_rows),
    "all_gates_pass": True,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

# Stage only the five reviewed support files after the wheel-install audit, so
# they cannot be mistaken for installer output or replace an installed module.
mkdir -p "${SUPPORT}"
while read -r digest member; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ && -n "${member}" ]]
  source=${C2_ROOT}/${member}
  [[ -f "${source}" && ! -L "${source}" ]]
  install -D -m 0444 "${source}" "${SUPPORT}/${member}"
done < "${SUPPORT_SOURCE_MANIFEST}"
cp "${SUPPORT_SOURCE_MANIFEST}" "${SUPPORT_INSTALLED_MANIFEST}"
(cd "${SUPPORT}" && sha256sum -c "${SUPPORT_INSTALLED_MANIFEST}")
[[ -z "$(find "${SUPPORT}" -type l -print -quit)" ]]
[[ -z "$(find "${SUPPORT}" ! -type d ! -type f -print -quit)" ]]
expected_support_members=$(awk '{print $2}' "${SUPPORT_SOURCE_MANIFEST}" | sort)
installed_support_members=$(find "${SUPPORT}" -type f -printf '%P\n' | sort)
[[ "${installed_support_members}" == "${expected_support_members}" ]]
[[ ! -e "${SUPPORT}/vllm" ]]
chmod -R a-w "${SUPPORT}"

# Immutable base surface: this separate process has no overlay target in its
# import path.  It is the only permitted pre-loader baseline.
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= \
  "${PYTHON}" - "${BASELINE_VLLM}" "${BASELINE_OPS}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import torch
import vllm
import vllm._C_stable_libtorch as stable
from vllm import _custom_ops
from vllm.config.attention import MiniMaxM3MSADecodeBackend
from vllm.platforms import current_platform

def key(schema):
    return f"{schema.name}.{schema.overload_name}" if schema.overload_name else schema.name
def table(op):
    rows = {}
    for line in torch._C._dispatch_dump_table(op).splitlines():
        name, sep, registration = line.partition(": ")
        if not sep:
            continue
        left = registration.rfind("[")
        assert left >= 0 and registration.endswith("]"), (op, line)
        rows[name] = registration[left:]
    return rows
def snapshot():
    ops = sorted(op for op in torch._C._dispatch_get_all_op_names() if op.startswith("_C::"))
    schemas = {}
    for schema in torch._C._jit_get_all_schemas():
        if schema.name.startswith("_C::"):
            schemas.setdefault(key(schema), []).append(str(schema))
    schemas = {name: sorted(values) for name, values in sorted(schemas.items())}
    assert set(ops) == set(schemas)
    dispatch = {op: table(op) for op in ops}
    assert all(dispatch.values())
    return ops, schemas, dispatch

expected, output = map(Path, sys.argv[1:])
root = Path(vllm.__file__).resolve().parent
assert root == expected.resolve(), (root, expected)
assert Path(stable.__file__).resolve() == root / "_C_stable_libtorch.abi3.so"
assert Path(_custom_ops.__file__).resolve() == root / "_custom_ops.py"
assert torch.__version__ == "2.13.0+cu130" and torch.version.cuda == "13.0"
assert hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest() == "cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442"
ops, schemas, dispatch = snapshot()
assert "_C::native_c2_msa_decode" not in ops
output.write_text(json.dumps({"schema": "c2-native-plugin-v6-baseline-dispatch-surface-v1",
    "vllm_root": str(root), "stable_sha256": hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest(),
    "torch_version": torch.__version__, "ops": ops, "schemas": schemas,
    "dispatch_key_registration_kinds": dispatch,
    "normal_imports_before_snapshot": ["vllm._custom_ops", "vllm.config.attention", "vllm.platforms"]},
    indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

# Fresh target pre/post-loader surface.  This command intentionally has no
# checkout or support directory on PYTHONPATH, so its loader cannot borrow a
# source-tree adapter or a preinstalled DSO.
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${RUNTIME}" \
  "${PYTHON}" - "${RUNTIME}" "${WHEEL}" "${BASELINE_OPS}" "${INSTALLED_PRE}" "${INSTALLED_POST}" <<'PY'
import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
import zipfile
import torch
import vllm
import vllm._C_stable_libtorch as stable

plugin_member = "vllm/_native_c2_msa_decode_plugin.abi3.so"
adapter_member = "vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py"
loader_name = "vllm.models.minimax_m3.nvidia.msa_native_c2_decode"
callable_name = "_load_native_c2_plugin_once"
def key(schema):
    return f"{schema.name}.{schema.overload_name}" if schema.overload_name else schema.name
def table(op):
    rows = {}
    for line in torch._C._dispatch_dump_table(op).splitlines():
        name, sep, registration = line.partition(": ")
        if not sep:
            continue
        left = registration.rfind("[")
        assert left >= 0 and registration.endswith("]"), (op, line)
        rows[name] = registration[left:]
    return rows
def snapshot():
    ops = sorted(op for op in torch._C._dispatch_get_all_op_names() if op.startswith("_C::"))
    schemas = {}
    for schema in torch._C._jit_get_all_schemas():
        if schema.name.startswith("_C::"):
            schemas.setdefault(key(schema), []).append(str(schema))
    schemas = {name: sorted(values) for name, values in sorted(schemas.items())}
    assert set(ops) == set(schemas)
    dispatch = {op: table(op) for op in ops}
    assert all(dispatch.values())
    return ops, schemas, dispatch

runtime, wheel, baseline_path, pre_path, post_path = map(Path, sys.argv[1:])
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
root = Path(vllm.__file__).resolve().parent
assert root == runtime.resolve() / "vllm"
assert importlib.metadata.distribution("vllm").locate_file("vllm").resolve() == root
assert Path(stable.__file__).resolve() == root / "_C_stable_libtorch.abi3.so"
assert hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest() == "cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442"
with zipfile.ZipFile(wheel, "r") as z:
    assert Path(stable.__file__).read_bytes() == z.read("vllm/_C_stable_libtorch.abi3.so")
    assert (root / plugin_member.removeprefix("vllm/")).read_bytes() == z.read(plugin_member)
module = importlib.import_module(loader_name)
adapter = root / adapter_member.removeprefix("vllm/")
assert Path(module.__file__).resolve() == adapter
with zipfile.ZipFile(wheel, "r") as z:
    assert adapter.read_bytes() == z.read(adapter_member)
pre_ops, pre_schemas, pre_dispatch = snapshot()
assert pre_ops == baseline["ops"] and pre_schemas == baseline["schemas"]
assert pre_dispatch == baseline["dispatch_key_registration_kinds"]
assert "_C::native_c2_msa_decode" not in pre_ops
pre_path.write_text(json.dumps({"schema": "c2-native-plugin-v6-installed-preloader-surface-v1",
    "vllm_root": str(root), "adapter_imported_before_snapshot_without_calling_loader": True,
    "matches_immutable_baseline": True, "ops": pre_ops, "schemas": pre_schemas,
    "dispatch_key_registration_kinds": pre_dispatch}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
loader = getattr(module, callable_name)
assert callable(loader)
first = loader()
post_ops, post_schemas, post_dispatch = snapshot()
base_ops = set(baseline["ops"])
added, missing = sorted(set(post_ops) - base_ops), sorted(base_ops - set(post_ops))
assert added == ["_C::native_c2_msa_decode"] and not missing
assert all(post_schemas[op] == baseline["schemas"][op] for op in base_ops)
assert all(post_dispatch[op] == baseline["dispatch_key_registration_kinds"][op] for op in base_ops)
assert torch._C._dispatch_has_kernel_for_dispatch_key("_C::native_c2_msa_decode", "CUDA")
second = loader()
assert second is True and snapshot() == (post_ops, post_schemas, post_dispatch)
plugin = root / plugin_member.removeprefix("vllm/")
post_path.write_text(json.dumps({"schema": "c2-native-plugin-v6-installed-postloader-surface-v1",
    "plugin_path": str(plugin), "plugin_sha256": hashlib.sha256(plugin.read_bytes()).hexdigest(),
    "loader": {"module": loader_name, "callable": callable_name, "style": "noargs",
               "first_return_is_true": first is True, "second_return_is_true": second is True,
               "second_call_snapshot_unchanged": True}, "added_ops": added,
    "missing_baseline_ops": missing, "native_c2_has_cuda_kernel": True,
    "all_baseline_op_names_schemas_and_dispatch_surfaces_preserved": True},
    indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

printf 'seed,before_uuid,after_uuid,result,trace\n' > "${UUID_MANIFEST}"
for seed in "${SEEDS[@]}"; do
  result=${ARTIFACT_DIR}/plugin-v6-wheel-full-backend-seed${seed}-job${RUN_ID}.json
  trace=${ARTIFACT_DIR}/plugin-v6-wheel-full-backend-seed${seed}-job${RUN_ID}.chrome.json
  before=$(visible_gpu_uuid)
  [[ "${before}" == "${EXPECTED_GPU_UUID}" ]]
  # New interpreter for every seed: it loads only the installed v6 target and
  # uses its own in-process frozen Triton baseline through the base harness's
  # ABBA protocol.  SUPPORT has harness/ only, never a vllm checkout.
  PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${RUNTIME}:${SUPPORT}" \
    timeout --foreground --preserve-status --kill-after=10s 360s \
    "${PYTHON}" "${HARNESS_COPY}" --c2-root "${SUPPORT}" --vllm-root "${RUNTIME}" \
      --output "${result}" --trace "${trace}" --seed "${seed}" \
      --timing-warmup 10 --timing-repetitions 50
  after=$(visible_gpu_uuid)
  [[ "${after}" == "${EXPECTED_GPU_UUID}" ]]
  printf '%s,%s,%s,%s,%s\n' "${seed}" "${before}" "${after}" "${result}" "${trace}" >> "${UUID_MANIFEST}"
done

# Complete integration is a hard gate.  Performance is retained as a
# measurement, including the Triton ratio, but it is never converted into a
# parity requirement for FINAL_RC.
"${PYTHON}" - "${SUMMARY}" "${DECISION}" "${UUID_MANIFEST}" "${RUNTIME}" \
  "${V5_FRESH}" "${EXPECTED_V5_FRESH_NATIVE_MEDIAN_MS}" "${EXPECTED_GPU_UUID}" \
  "${C2_EXPECTED_PLUGIN_SHA}" "${C2_EXPECTED_V5_FRESH_SHA}" \
  "${ARTIFACT_DIR}" "${SEEDS[@]}" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

(summary_path, decision_path, uuid_path, runtime, v5_path, expected_v5_text,
 expected_uuid, v6_dso_sha, expected_v5_sha, artifact_dir, *seed_text) = sys.argv[1:]
summary_path, decision_path, uuid_path, runtime, v5_path, artifact_dir = map(
    Path, (summary_path, decision_path, uuid_path, runtime, v5_path, artifact_dir)
)
seeds = [int(value) for value in seed_text]
assert len(seeds) == 8 and len(set(seeds)) == 8
assert hashlib.sha256(v5_path.read_bytes()).hexdigest() == expected_v5_sha
rows = list(csv.DictReader(uuid_path.open(encoding="utf-8", newline="")))
assert [int(row["seed"]) for row in rows] == seeds
assert all(row["before_uuid"] == row["after_uuid"] == expected_uuid for row in rows)

results = []
native_ms, triton_ms, ratios = [], [], []
contract_shape = None
for seed in seeds:
    path = artifact_dir / f"plugin-v6-wheel-full-backend-seed{seed}-job{artifact_dir.name.removeprefix('job')}.json"
    trace = artifact_dir / f"plugin-v6-wheel-full-backend-seed{seed}-job{artifact_dir.name.removeprefix('job')}.chrome.json"
    p = json.loads(path.read_text(encoding="utf-8"))
    assert p["schema"] == "c2-native-c2-full-vllm-backend-v2"
    assert p["seed"] == seed and p["all_gates_pass"] is True
    assert p["correctness"]["pass"] is True
    assert p["correctness"]["atol"] == 1.0e-4 and p["correctness"]["rtol"] == 1.0e-3
    assert "independent FP32" in p["correctness"]["oracle"]
    assert p["triton_correctness"]["pass"] is True
    assert p["caller_output"]["pointer_unchanged"] is True
    assert p["profiling"]["checks"]["one_cpu_dispatcher_event"] is True
    assert p["profiling"]["checks"]["one_cuda_native_kernel_event"] is True
    assert trace.is_file() and trace.stat().st_size > 0
    timing = p["timing"]
    assert timing["pass"] is True
    assert "same BF16 query" in timing["comparison_input_semantics"]
    assert "ABBA=native,triton,triton,native" in timing["protocol"]
    assert timing["warmup"] == 10 and timing["repetitions"] == 50
    assert timing["native"]["sample_count"] == timing["triton"]["sample_count"] == 100
    n, t = float(timing["native"]["median_ms"]), float(timing["triton"]["median_ms"])
    assert math.isfinite(n) and math.isfinite(t) and n > 0 and t > 0
    assert Path(p["environment"]["vllm_import_root"]).resolve() == runtime / "vllm"
    shape = {key: p["data_contract"][key] for key in ("batch", "query_heads", "kv_heads", "head_dim", "page_size", "topk")}
    if contract_shape is None:
        contract_shape = shape
    assert shape == contract_shape
    native_ms.append(n); triton_ms.append(t); ratios.append(n / t)
    results.append({"seed": seed, "result": str(path), "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "trace": str(trace), "native_median_ms": n, "triton_median_ms": t,
                    "native_over_triton_latency_ratio": n / t})

v5 = json.loads(v5_path.read_text(encoding="utf-8"))
assert v5["schema"] == "c2-native-c2-full-vllm-backend-v2"
assert v5["all_gates_pass"] is True and v5["seed"] == 20260829
assert float(v5["timing"]["native"]["median_ms"]) == float(expected_v5_text)
v6_matching = next(row for row in results if row["seed"] == 20260829)
v6_matching_payload = json.loads(Path(v6_matching["result"]).read_text(encoding="utf-8"))
assert v5["boundary"] == v6_matching_payload["boundary"]
assert v5["data_contract"] == v6_matching_payload["data_contract"]
v5_native = float(v5["timing"]["native"]["median_ms"])
v6_native = float(v6_matching["native_median_ms"])
summary = {
    "schema": "c2-native-plugin-v6-fresh-backend-summary-v1",
    "all_integration_and_correctness_gates_pass": True,
    "seeds": results,
    "seed_count": len(results),
    "one_physical_gpu_uuid": expected_uuid,
    "fresh_process_per_seed": True,
    "native_median_of_seed_medians_ms": statistics.median(native_ms),
    "triton_median_of_seed_medians_ms": statistics.median(triton_ms),
    "native_over_triton_ratio_of_seed_medians": statistics.median(ratios),
    "triton_parity_achieved": max(ratios) <= 1.0,
    "triton_parity_is_hard_gate": False,
    "v6_dso_sha256": v6_dso_sha,
    "v6_vs_promoted_v5_fresh_boundary": {
        "comparison_kind": "directional cross-job fresh-process observation only",
        "same_python_process": False,
        "same_slurm_job": False,
        "dual_dso_loaded": False,
        "reason": "v5 and v6 register the same _C schema/CUDA implementation; loading both would make dispatcher ownership ambiguous",
        "matching_seed": 20260829,
        "contract_equal": True,
        "promoted_v5_native_median_ms": v5_native,
        "v6_native_median_ms": v6_native,
        "directional_improvement_fraction": 1.0 - v6_native / v5_native,
        "not_a_paired_promotion_decision": True,
    },
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
decision = {
    "schema": "c2-native-plugin-v6-fresh-backend-decision-v1",
    "accepted_for_fresh_integration": True,
    "final_rc": 0,
    "hard_gates": {"wheel_identity": True, "record_and_install": True, "stable_dso_preserved": True,
                   "loader_surface": True, "eight_seed_real_backend": True, "direct_bitwise_companion": True,
                   "clean_b300_before_after": True},
    "performance": {"reported": True, "triton_parity_required": False,
                    "triton_parity_achieved": summary["triton_parity_achieved"]},
    "v5_comparison_boundary": summary["v6_vs_promoted_v5_fresh_boundary"],
}
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

snapshot POST "${GPU_POST}" "${GPU_POST_APPS}"
[[ "$(visible_gpu_uuid)" == "${EXPECTED_GPU_UUID}" ]]
[[ -z "$(<"${GPU_POST_APPS}")" ]] || { echo 'GPU is not clean after fresh validation' >&2; exit 1; }
sha256sum "${SUMMARY}" "${DECISION}" "${INSTALLED_MEMBERS}" "${BASELINE_OPS}" \
  "${INSTALLED_PRE}" "${INSTALLED_POST}" "${UUID_MANIFEST}" "${GPU_PRE}" "${GPU_POST}" \
  "${GPU_PRE_APPS}" "${GPU_POST_APPS}" "${SUPPORT_SOURCE_MANIFEST}" "${SUPPORT_INSTALLED_MANIFEST}" \
  "${HARNESS_COPY}" "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256" \
  ${ARTIFACT_DIR}/plugin-v6-wheel-full-backend-seed*-job${RUN_ID}.json \
  ${ARTIFACT_DIR}/plugin-v6-wheel-full-backend-seed*-job${RUN_ID}.chrome.json \
  > "${ARTIFACT_DIR}/outputs-job${RUN_ID}.sha256"
exit 0
