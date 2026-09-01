#!/usr/bin/env bash
# Directed correctness/liveness gate for a reviewed v12 Q-row-padding DSO.
# This consumes an explicit, immutable AOT plugin only; it makes no performance
# claim and must pass before a v12 candidate can enter the v9-v12 ABBA timing.
# The v6 directed harness is deliberately frozen: its head/dimension mapping
# and output schema are validation inputs, not v12 implementation artifacts.
#SBATCH --job-name=c2-native-plugin-v12-directed
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/lcpu/85117379/slurm-c2-native-plugin-v12-directed-%j.log

set -Eeuo pipefail
normalize_preflight_rc() {
  local original_rc=$? normalized_rc
  trap - EXIT
  case "${original_rc}" in
    0|2) normalized_rc=${original_rc} ;;
    *) normalized_rc=2 ;;
  esac
  exit "${normalized_rc}"
}
trap normalize_preflight_rc EXIT
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONPATH= PYTHONDONTWRITEBYTECODE=1

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v12-directed-artifacts-20260830
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
BASE_HARNESS=/home/lcpu/85117379/native_c2_operator_bench_20260829.py
V5_DIRECTED_HARNESS=/home/lcpu/85117379/native_c2_v5_softmax_directed_20260829.py
DIRECTED_HARNESS=/home/lcpu/85117379/native_c2_v6_register_numerator_directed_20260830.py
EXPECTED_BASE_HARNESS_SHA=0b51fede1e6258eea0b535acba2a86e371b284e9655352156cdeb3d310d90c28
EXPECTED_V5_DIRECTED_HARNESS_SHA=43ad54c5ad72cc40bf8b8a6e17592fa16cacc1d1e62f5f8c42ac2a4d3206cdd1
: "${SLURM_JOB_ID:?must run under Slurm}"
: "${C2_V12_CANDIDATE_PLUGIN:?set the reviewed v12 plugin absolute path}"
: "${C2_EXPECTED_V12_CANDIDATE_SHA:?set the reviewed v12 plugin SHA-256}"
: "${C2_V12_CANDIDATE_SOURCE:?set the derived v12 source absolute path}"
: "${C2_EXPECTED_V12_SOURCE_SHA:?set the reviewed v12 source SHA-256}"
: "${C2_V12_CANDIDATE_RESOURCE_GATE:?set the reviewed v12 resource gate}"
: "${C2_EXPECTED_V12_RESOURCE_GATE_SHA:?set the reviewed v12 resource-gate SHA-256}"
: "${C2_V12_CANDIDATE_PROVENANCE:?set the reviewed v12 provenance manifest}"
: "${C2_EXPECTED_V12_PROVENANCE_SHA:?set the reviewed v12 provenance-manifest SHA-256}"
: "${C2_V12_AOT_FINAL_STATUS:?set the v12 AOT final-status path}"
: "${C2_EXPECTED_V12_AOT_FINAL_STATUS_SHA:?set the v12 AOT final-status SHA-256}"
: "${C2_V12_AOT_FINAL_STATUS_SIDECAR:?set the v12 AOT final-status sidecar path}"
: "${C2_EXPECTED_V12_AOT_FINAL_STATUS_SIDECAR_SHA:?set the v12 AOT final-status sidecar SHA-256}"
: "${C2_V12_AOT_OUTPUTS_MANIFEST:?set the v12 AOT outputs manifest path}"
: "${C2_EXPECTED_V12_AOT_OUTPUTS_MANIFEST_SHA:?set the v12 AOT outputs-manifest SHA-256}"
: "${C2_EXPECTED_V6_DIRECTED_HARNESS_SHA:?set the frozen v6 harness SHA-256}"
: "${C2_EXPECTED_V12_DIRECTED_SCRIPT_SHA:?set the reviewed v12 Slurm script SHA-256}"
[[ "${C2_V12_CANDIDATE_PLUGIN}" == /* && "${C2_V12_CANDIDATE_SOURCE}" == /* && \
   "${C2_V12_CANDIDATE_RESOURCE_GATE}" == /* && "${C2_V12_CANDIDATE_PROVENANCE}" == /* && \
   "${C2_V12_AOT_FINAL_STATUS}" == /* && "${C2_V12_AOT_FINAL_STATUS_SIDECAR}" == /* && \
   "${C2_V12_AOT_OUTPUTS_MANIFEST}" == /* ]]
[[ "${C2_EXPECTED_V12_CANDIDATE_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ "${C2_EXPECTED_V12_SOURCE_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ "${C2_EXPECTED_V12_RESOURCE_GATE_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ "${C2_EXPECTED_V12_PROVENANCE_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ "${C2_EXPECTED_V12_AOT_FINAL_STATUS_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ "${C2_EXPECTED_V12_AOT_FINAL_STATUS_SIDECAR_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ "${C2_EXPECTED_V12_AOT_OUTPUTS_MANIFEST_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ "${C2_EXPECTED_V6_DIRECTED_HARNESS_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ "${C2_EXPECTED_V12_DIRECTED_SCRIPT_SHA}" =~ ^[0-9a-f]{64}$ ]]
RUN_ID=${C2_RUN_ID:-${SLURM_JOB_ID}}
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${RUN_ID}
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
[[ "${SCRIPT_PATH}" == /* ]]
[[ -d "${ARTIFACT_ROOT}" ]] || { echo 'artifact root missing' >&2; exit 2; }
[[ -z "$(find "${ARTIFACT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
  echo 'artifact root must be empty before v12 directed submission' >&2; exit 2;
}
[[ ! -e "${ARTIFACT_DIR}" ]] || { echo 'run-scoped output already exists' >&2; exit 2; }
[[ -f "${C2_V12_CANDIDATE_PLUGIN}" && -f "${C2_V12_CANDIDATE_SOURCE}" && \
   -f "${C2_V12_CANDIDATE_RESOURCE_GATE}" && -f "${C2_V12_CANDIDATE_PROVENANCE}" && \
   -f "${C2_V12_AOT_FINAL_STATUS}" && -f "${C2_V12_AOT_FINAL_STATUS_SIDECAR}" && \
   -f "${C2_V12_AOT_OUTPUTS_MANIFEST}" ]] || {
  echo 'candidate or AOT-evidence input missing' >&2; exit 2;
}
[[ -f "${BASE_HARNESS}" && -f "${V5_DIRECTED_HARNESS}" && -f "${DIRECTED_HARNESS}" ]] || {
  echo 'frozen directed-harness input missing' >&2; exit 2;
}
mkdir -p "${ARTIFACT_DIR}" || exit 2
SUBMITTED_SCRIPT_COPY=${ARTIFACT_DIR}/submitted-script-job${SLURM_JOB_ID}.slurm
FINAL_STATUS=${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt
FINAL_STATUS_SIDECAR=${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.sha256
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1

export PATH=/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin
# Keep py_compile/import cache writes under the immutable run artifact rather
# than beside any frozen harness input.
export PYTHONPYCACHEPREFIX="${ARTIFACT_DIR}/pycache"
EXPECTED_GPU_UUID=
post_checked=0
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
visible_gpu_uuid() {
  local selector uuid
  local -a uuids
  selector=$(allocated_gpu_selector)
  mapfile -t uuids < <(nvidia-smi --id "${selector}" --query-gpu=uuid --format=csv,noheader,nounits)
  [[ ${#uuids[@]} -eq 1 ]] || {
    printf 'expected one allocated GPU UUID, got %s\n' "${#uuids[@]}" >&2
    return 1
  }
  uuid=${uuids[0]}
  [[ "${uuid}" =~ ^GPU- ]] || { echo "invalid GPU UUID: ${uuid}" >&2; return 1; }
  printf '%s\n' "${uuid}"
}
compute_apps() {
  local selector
  selector=$(allocated_gpu_selector)
  nvidia-smi --id "${selector}" --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}
snapshot() {
  local label=$1 selector output
  selector=$(allocated_gpu_selector) || return 1
  output="${ARTIFACT_DIR}/gpu-${label,,}-job${SLURM_JOB_ID}.txt"
  {
    printf '===== %s UTC=%s =====\n' "${label}" "$(date -u +%FT%TZ)"
    nvidia-smi --id "${selector}" --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap \
      --format=csv,noheader,nounits || return 1
    printf '%s\n' 'COMPUTE_APPS'
    compute_apps || return 1
  } > "${output}" || return 1
  cat -- "${output}"
}
ensure_clean_post() {
  snapshot POST || { echo 'unable to snapshot post-run allocated GPU' >&2; return 1; }
  local apps_after
  apps_after=$(compute_apps) || { echo 'unable to query post-run compute apps' >&2; return 1; }
  [[ -z "${apps_after}" ]] || { echo "post-run compute apps remain: ${apps_after}" >&2; return 1; }
  post_checked=1
}
finalize() {
  local original_rc=$? final_rc finalizer_error=0 apps_after='' apps_rc uuid_rc post_uuid='' snapshot_rc
  final_rc=${original_rc}
  trap - EXIT
  set +e
  case "${final_rc}" in
    0|2) ;;
    *) final_rc=2; finalizer_error=1 ;;
  esac
  snapshot FINAL_POST
  snapshot_rc=$?
  [[ "${snapshot_rc}" -eq 0 ]] || finalizer_error=1
  apps_after=$(compute_apps)
  apps_rc=$?
  [[ "${apps_rc}" -eq 0 && -z "${apps_after}" ]] || finalizer_error=1
  post_uuid=$(visible_gpu_uuid)
  uuid_rc=$?
  [[ "${uuid_rc}" -eq 0 && -n "${EXPECTED_GPU_UUID}" && "${post_uuid}" == "${EXPECTED_GPU_UUID}" ]] || finalizer_error=1
  [[ "${finalizer_error}" -eq 0 ]] || final_rc=2
  printf 'ORIGINAL_RC=%s POST_SNAPSHOT_RC=%s APPS_QUERY_RC=%s APPS_EMPTY=%s UUID_QUERY_RC=%s EXPECTED_UUID=%s POST_UUID=%s\n' \
    "${original_rc}" "${snapshot_rc}" "${apps_rc}" \
    "$([[ -z "${apps_after}" ]] && echo true || echo false)" "${uuid_rc}" \
    "${EXPECTED_GPU_UUID:-unset}" "${post_uuid:-unset}" \
    > "${ARTIFACT_DIR}/finalizer-diagnostics-job${SLURM_JOB_ID}.txt" || {
      final_rc=2
      finalizer_error=1
    }
  printf 'FINAL_RC=%s ORIGINAL_RC=%s FINALIZER_ERROR=%s POST_CHECKED=%s UTC=%s\n' \
    "${final_rc}" "${original_rc}" "${finalizer_error}" "${post_checked}" "$(date -u +%FT%TZ)" \
    > "${FINAL_STATUS}" || {
      final_rc=2
      finalizer_error=1
      printf 'FINAL_RC=%s ORIGINAL_RC=%s FINALIZER_ERROR=%s POST_CHECKED=%s UTC=%s\n' \
        "${final_rc}" "${original_rc}" "${finalizer_error}" "${post_checked}" "$(date -u +%FT%TZ)" \
        > "${FINAL_STATUS}" || true
    }
  if ! sha256sum "${ARTIFACT_DIR}/finalizer-diagnostics-job${SLURM_JOB_ID}.txt" \
      "${FINAL_STATUS}" > "${FINAL_STATUS_SIDECAR}"; then
    final_rc=2
    finalizer_error=1
    printf 'FINAL_RC=%s ORIGINAL_RC=%s FINALIZER_ERROR=%s POST_CHECKED=%s UTC=%s\n' \
      "${final_rc}" "${original_rc}" "${finalizer_error}" "${post_checked}" "$(date -u +%FT%TZ)" \
      > "${FINAL_STATUS}" || true
    sha256sum "${ARTIFACT_DIR}/finalizer-diagnostics-job${SLURM_JOB_ID}.txt" \
      "${FINAL_STATUS}" > "${FINAL_STATUS_SIDECAR}" || true
  fi
  exit "${final_rc}"
}
trap finalize EXIT

snapshot PRE || exit 2
gpu_line=$(nvidia-smi --id "$(allocated_gpu_selector)" --query-gpu=name,compute_cap --format=csv,noheader,nounits) || exit 2
apps_before=$(compute_apps) || { echo 'unable to query pre-run compute apps' >&2; exit 2; }
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* && -z "${apps_before}" ]] || {
  echo 'requires a clean B300 capability-10.3 GPU' >&2; exit 2;
}
EXPECTED_GPU_UUID=$(visible_gpu_uuid) || { echo 'unable to capture allocated GPU UUID' >&2; exit 2; }

printf '%s  %s\n' \
  "${C2_EXPECTED_V12_CANDIDATE_SHA}" "${C2_V12_CANDIDATE_PLUGIN}" \
  "${C2_EXPECTED_V12_SOURCE_SHA}" "${C2_V12_CANDIDATE_SOURCE}" \
  "${C2_EXPECTED_V12_RESOURCE_GATE_SHA}" "${C2_V12_CANDIDATE_RESOURCE_GATE}" \
  "${C2_EXPECTED_V12_PROVENANCE_SHA}" "${C2_V12_CANDIDATE_PROVENANCE}" \
  "${C2_EXPECTED_V12_AOT_FINAL_STATUS_SHA}" "${C2_V12_AOT_FINAL_STATUS}" \
  "${C2_EXPECTED_V12_AOT_FINAL_STATUS_SIDECAR_SHA}" "${C2_V12_AOT_FINAL_STATUS_SIDECAR}" \
  "${C2_EXPECTED_V12_AOT_OUTPUTS_MANIFEST_SHA}" "${C2_V12_AOT_OUTPUTS_MANIFEST}" \
  "${EXPECTED_BASE_HARNESS_SHA}" "${BASE_HARNESS}" \
  "${EXPECTED_V5_DIRECTED_HARNESS_SHA}" "${V5_DIRECTED_HARNESS}" \
  "${C2_EXPECTED_V6_DIRECTED_HARNESS_SHA}" "${DIRECTED_HARNESS}" \
  "${C2_EXPECTED_V12_DIRECTED_SCRIPT_SHA}" "${SCRIPT_PATH}" | sha256sum -c - || exit 2
cp -- "${SCRIPT_PATH}" "${SUBMITTED_SCRIPT_COPY}" || exit 2
chmod a-w "${SUBMITTED_SCRIPT_COPY}" || exit 2
printf '%s  %s\n' "${C2_EXPECTED_V12_DIRECTED_SCRIPT_SHA}" "${SUBMITTED_SCRIPT_COPY}" | sha256sum -c - || exit 2
RESOURCE_LINE=$(grep -m 1 -E 'REG:[[:space:]]*[0-9]+.*STACK:[[:space:]]*[0-9]+.*SHARED:[[:space:]]*[0-9]+.*LOCAL:[[:space:]]*[0-9]+' "${C2_V12_CANDIDATE_RESOURCE_GATE}")
[[ -n "${RESOURCE_LINE}" ]] || { echo 'candidate native-kernel resource line missing' >&2; exit 2; }
read_resource() { sed -nE "s/.*$1:[[:space:]]*([0-9]+).*/\\1/p" <<< "${RESOURCE_LINE}"; }
REG=$(read_resource REG); STACK=$(read_resource STACK); SHARED=$(read_resource SHARED); LOCAL=$(read_resource LOCAL)
[[ "${REG}" =~ ^[0-9]+$ && "${STACK}" =~ ^[0-9]+$ && "${SHARED}" =~ ^[0-9]+$ && "${LOCAL}" =~ ^[0-9]+$ ]] || exit 2
(( REG <= 127 && STACK == 0 && LOCAL == 0 && SHARED == 31136 )) || {
  echo "candidate resource gate failed: REG=${REG} STACK=${STACK} SHARED=${SHARED} LOCAL=${LOCAL}" >&2
  exit 2
}
grep -Fx "${C2_EXPECTED_V12_CANDIDATE_SHA}  ${C2_V12_CANDIDATE_PLUGIN}" "${C2_V12_CANDIDATE_PROVENANCE}" || exit 2
if ! "${PYTHON}" - "${C2_V12_CANDIDATE_PLUGIN}" "${C2_V12_CANDIDATE_SOURCE}" \
  "${C2_V12_CANDIDATE_RESOURCE_GATE}" "${C2_V12_CANDIDATE_PROVENANCE}" \
  "${C2_V12_AOT_FINAL_STATUS}" "${C2_V12_AOT_FINAL_STATUS_SIDECAR}" \
  "${C2_V12_AOT_OUTPUTS_MANIFEST}" <<'PY'
import hashlib
import re
import sys
from pathlib import Path

(plugin, source, resource_gate, provenance, final_status, sidecar,
 outputs_manifest) = [Path(value).resolve() for value in sys.argv[1:]]

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def parse_manifest(path: Path) -> dict[Path, str]:
    records: dict[Path, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        expected, separator, recorded = raw_line.partition("  ")
        assert separator and re.fullmatch(r"[0-9a-f]{64}", expected), raw_line
        target = Path(recorded)
        assert target.is_absolute(), (path, target)
        target = target.resolve()
        assert target.is_file() and digest(target) == expected, (path, target)
        assert target not in records, (path, target)
        records[target] = expected
    assert records, path
    return records

match = re.fullmatch(r"final-status-job([0-9]+)\.txt", final_status.name)
assert match, final_status
job_id = match.group(1)
aot_parent = final_status.parent
assert aot_parent.name == f"job{job_id}", aot_parent
assert sidecar == aot_parent / f"final-status-job{job_id}.sha256"
assert outputs_manifest == aot_parent / f"outputs-job{job_id}.sha256"
assert resource_gate == aot_parent / f"plugin-resource-gate-job{job_id}.txt"
assert provenance == aot_parent / f"plugin-v12-provenance-job{job_id}.sha256"
assert source == aot_parent / "native_c2_decode.v12-q-row-padding.cu"
assert plugin == aot_parent / "vllm/_native_c2_msa_decode_plugin.abi3.so"

status = dict(token.split("=", 1) for token in final_status.read_text(encoding="utf-8").split() if "=" in token)
assert all(status.get(key) == "0" for key in ("BODY_RC", "TEE_RC", "MANIFEST_RC", "FINAL_RC")), status
sidecar_records = parse_manifest(sidecar)
assert final_status in sidecar_records, sidecar
outputs_records = parse_manifest(outputs_manifest)
for required in (plugin, source, resource_gate, provenance):
    assert required in outputs_records, (outputs_manifest, required)
provenance_records = parse_manifest(provenance)
assert provenance_records.get(plugin) == digest(plugin), (provenance, plugin)
# AOT deliberately archives a job-scoped source copy while provenance records
# the exact build-worktree source path.  Bind the two by immutable bytes rather
# than incorrectly requiring those distinct paths to be identical.
source_sha = digest(source)
provenance_sources = [
    path for path, recorded_sha in provenance_records.items()
    if recorded_sha == source_sha and path.name == "native_c2_decode.cu"
]
assert len(provenance_sources) == 1, (provenance, source_sha, provenance_sources)
print(f"AOT_EVIDENCE_BOUND job={job_id} parent={aot_parent}")
PY
then
  echo 'v12 AOT result/status/sidecar/outputs evidence is not one valid job bundle' >&2
  exit 2
fi
"${PYTHON}" -m py_compile "${BASE_HARNESS}" "${V5_DIRECTED_HARNESS}" "${DIRECTED_HARNESS}" || exit 2
if ! "${PYTHON}" - "${DIRECTED_HARNESS}" <<'PY'
import ast
import sys
from pathlib import Path

path = Path(sys.argv[1])
ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("AST_OK", path)
PY
then
  echo 'frozen v6 directed harness AST validation failed' >&2
  exit 2
fi

RUN_MANIFEST=${ARTIFACT_DIR}/run-manifest-job${SLURM_JOB_ID}.json
if ! "${PYTHON}" - "${RUN_MANIFEST}" \
  "${C2_V12_CANDIDATE_PLUGIN}" "${C2_V12_CANDIDATE_SOURCE}" \
  "${C2_V12_CANDIDATE_RESOURCE_GATE}" "${C2_V12_CANDIDATE_PROVENANCE}" \
  "${C2_V12_AOT_FINAL_STATUS}" "${C2_V12_AOT_FINAL_STATUS_SIDECAR}" \
  "${C2_V12_AOT_OUTPUTS_MANIFEST}" \
  "${BASE_HARNESS}" "${V5_DIRECTED_HARNESS}" "${DIRECTED_HARNESS}" "${SUBMITTED_SCRIPT_COPY}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
labels = (
    "v12_q_row_padding_plugin", "v12_q_row_padding_source", "v12_q_row_padding_resource_gate",
    "v12_q_row_padding_provenance", "v12_aot_final_status", "v12_aot_final_status_sidecar",
    "v12_aot_outputs_manifest", "base_harness", "v5_directed_harness",
    "frozen_v6_directed_harness", "v12_q_row_padding_directed_script",
)
paths = [Path(value).resolve() for value in sys.argv[2:]]
assert len(labels) == len(paths)
entries = []
for label, path in zip(labels, paths, strict=True):
    assert path.is_absolute() and path.is_file(), (label, path)
    entries.append({
        "label": label,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    })
manifest_path.write_text(
    json.dumps(
        {
            "schema": "c2-native-c2-v12-q-row-padding-directed-input-manifest-v1",
            "candidate": "v12-q-row-padding",
            "frozen_harness_schema": "c2-native-c2-v6-register-numerator-directed-v1",
            "inputs": entries,
        },
        allow_nan=False, indent=2, sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY
then
  echo 'v12 directed input manifest generation failed' >&2
  exit 2
fi
INPUT_MANIFEST=${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256
sha256sum "${C2_V12_CANDIDATE_PLUGIN}" "${C2_V12_CANDIDATE_SOURCE}" \
  "${C2_V12_CANDIDATE_RESOURCE_GATE}" "${C2_V12_CANDIDATE_PROVENANCE}" \
  "${C2_V12_AOT_FINAL_STATUS}" "${C2_V12_AOT_FINAL_STATUS_SIDECAR}" "${C2_V12_AOT_OUTPUTS_MANIFEST}" \
  "${BASE_HARNESS}" "${V5_DIRECTED_HARNESS}" "${DIRECTED_HARNESS}" "${SUBMITTED_SCRIPT_COPY}" \
  "${RUN_MANIFEST}" "${ARTIFACT_DIR}/gpu-pre-job${SLURM_JOB_ID}.txt" > "${INPUT_MANIFEST}"

RESULT=${ARTIFACT_DIR}/v12-q-row-padding-directed-job${SLURM_JOB_ID}.json
timeout --foreground --preserve-status --kill-after=10s 600s \
  "${PYTHON}" "${DIRECTED_HARNESS}" --library "${C2_V12_CANDIDATE_PLUGIN}" \
  --base-harness "${BASE_HARNESS}" --v5-directed-harness "${V5_DIRECTED_HARNESS}" \
  --sync-repeats 4 --max-iteration-seconds 15 --atol 1e-4 --rtol 1e-3 > "${RESULT}" || {
    rc=$?; echo "v12 directed harness failed rc=${rc}" >&2; exit 2;
  }

if ! "${PYTHON}" - "${RESULT}" "${C2_V12_CANDIDATE_PLUGIN}" <<'PY'
import json
import math
import sys
from pathlib import Path

result_path, library = map(Path, sys.argv[1:])
payload = json.loads(result_path.read_text(encoding="utf-8"))
assert payload["schema"] == "c2-native-c2-v6-register-numerator-directed-v1"
assert payload["all_gates_pass"] is True
assert payload["no_monkeypatch"] is True
assert payload["dispatch"] == "direct_torch.ops._C.native_c2_msa_decode"
assert Path(payload["operator_library"]).resolve() == library.resolve()
assert payload["contract"]["producer_ctas"] == 4
assert payload["contract"]["pages_per_producer"] == 4
assert payload["contract"]["selected_pages"] == 16
assert payload["contract"]["page_size"] == 128

def assert_case(case: dict) -> None:
    assert case["all_gates_pass"] is True
    assert case["caller_output"]["pointer_unchanged"] is True
    assert case["caller_output"]["return_is_none"] is True
    assert case["stability"]["bitwise_repeatable"] is True
    assert case["oracle"]["finite_output"] is True
    assert case["oracle"]["reference_fp64_fp32_agree"] is True
    assert case["oracle"]["fp64"]["allclose"] is True
    assert case["oracle"]["fp32"]["allclose"] is True
    elapsed = case["sync_watchdog"]["elapsed_seconds"]
    assert len(elapsed) == 4
    assert all(math.isfinite(float(value)) and 0.0 <= float(value) <= 15.0 for value in elapsed)

assert_case(payload["head_dim_encoding"])
encoding = payload["head_dim_encoding"]["head_dim_encoding"]
assert encoding["dim_code_count"] == 128
assert encoding["head_signature_count"] == 64
assert len(encoding["head_signature_offsets"]) == 64
assert len(set(encoding["head_signature_offsets"])) == 64
assert len(encoding["target_tokens"]) == 64
assert encoding["exact_location_match"] is True
assert encoding["nonzero_mismatch_count"] == 0
assert encoding["max_abs_to_exact_expected"] == 0.0

assert payload["rank_directed_case_count"] == 4
for expected_rank, case in enumerate(payload["rank_directed"]):
    assert_case(case)
    assert case["producer_rank"] == expected_rank
    assert case["selected_page_indices"] == list(range(4 * expected_rank, 4 * expected_rank + 4))
    assert case["rank_expected_value"]["allclose"] is True

assert payload["softmax_directed_case_count"] == 2
expected_softmax = (("all-valid-online-rescale", 2048, False), ("mixed-causal-tail-37", 37, True))
for case, (label, sequence_length, mixed_tail) in zip(payload["softmax_directed"], expected_softmax, strict=True):
    assert_case(case)
    assert case["case"] == label
    assert case["sequence_length"] == sequence_length
    assert case["mixed_causal_subgroup"] is mixed_tail

assert_case(payload["all_invalid"])
assert payload["all_invalid"]["case"] == "all-invalid"
assert payload["all_invalid"]["all_zero"] is True
assert payload["all_invalid"]["oracle"]["fp64"]["max_abs"] == 0.0
assert payload["all_invalid"]["oracle"]["fp32"]["max_abs"] == 0.0

profiler = payload["profiler"]
assert profiler["native_dispatcher_hit"] is True
assert profiler["native_kernel_hit"] is True
assert profiler["exactly_one_dispatcher_event"] is True
assert profiler["exactly_one_native_kernel_event"] is True
assert profiler["dispatcher_events"] == [
    {"key": "_C::native_c2_msa_decode", "count": 1}
]
assert len(profiler["kernel_events"]) == 1
assert profiler["kernel_events"][0]["count"] == 1
assert profiler["caller_output_pointer_unchanged"] is True
assert profiler["return_is_none"] is True
assert profiler["finite_output"] is True
PY
then
  echo 'v12 directed result validation failed' >&2
  exit 2
fi

ensure_clean_post || exit 2
sha256sum -c "${INPUT_MANIFEST}" || exit 2
OUTPUT_MANIFEST=${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256
sha256sum "${RESULT}" "${RUN_MANIFEST}" "${INPUT_MANIFEST}" \
  "${ARTIFACT_DIR}/gpu-pre-job${SLURM_JOB_ID}.txt" \
  "${ARTIFACT_DIR}/gpu-post-job${SLURM_JOB_ID}.txt" \
  > "${OUTPUT_MANIFEST}" || exit 2
sha256sum -c "${OUTPUT_MANIFEST}" || exit 2


