#!/usr/bin/env bash
# Direct-plugin v9 strict->3% performance gate.  It freezes the accepted v7
# V-prefetch DSO from job12484 and compares the parallel-merge candidate in one
# Slurm allocation, on one clean B300 GPU, against one immutable fixture set.
#
# The four A-B-B-A rounds deliberately use isolated Python workers.  Both DSO
# files use the same STABLE_TORCH_LIBRARY_FRAGMENT schema and CUDA impl for
# _C::native_c2_msa_decode, so loading both in one interpreter would make the
# active implementation ambiguous.  The process boundary is therefore a
# correctness-preserving dispatcher boundary, not a claim of a single-Python-
# process comparison.  The fixture SHA-256, GPU UUID, environment, harness SHA
# and measurement contract are checked equal across all four workers.
# This is an incremental v7->v9 comparison.  It does not reinterpret an older
# v6->v7 decision; both the baseline and candidate are identity-pinned.

#SBATCH --job-name=c2-native-v9-3pct-confirm
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/slurm-c2-native-plugin-v9-stress-3pct-%j.log

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

# Retry 1 follows a pre-measurement provenance-path rejection in job12772.
# Keep that failed evidence immutable and write this run into a fresh root.
ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v9-stress-3pct-retry1-artifacts-20260830
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
BASE_HARNESS=/home/lcpu/85117379/native_c2_operator_bench_20260829.py
EXPECTED_BASE_HARNESS_SHA=0b51fede1e6258eea0b535acba2a86e371b284e9655352156cdeb3d310d90c28
# v7 is a frozen reference; v9 inputs are supplied by the immediately
# preceding AOT and directed gates and are verified below by SHA-256.
EXPECTED_REFERENCE_LIBRARY=/home/lcpu/85117379/c2-native-plugin-v7-aot-artifacts-20260830/job12484/vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_REFERENCE_SHA=0873af3cf48ef756f9945fc6e711a03dd1208fcc6d4ce20a8041cf9a587e97db
EXPECTED_REFERENCE_SOURCE=/home/lcpu/85117379/c2-native-plugin-v7-aot-artifacts-20260830/job12484/native_c2_decode.v7-v-prefetch.cu
EXPECTED_REFERENCE_SOURCE_SHA=1d638ce66c3b17b2b83298fa98f44770b14b94154c0f56cf5ff84d508a12e9a3
SEEDS=17,23,42,2024,314159,20260801,20260815,20260829

: "${C2_REFERENCE_PLUGIN:?submit the frozen v7 job12484 plugin path}"
: "${C2_EXPECTED_REFERENCE_SHA:?submit the frozen v7 job12484 SHA-256}"
: "${C2_CANDIDATE_PLUGIN:?submit the v9 candidate plugin path}"
: "${C2_CANDIDATE_SOURCE:?submit the v9 candidate native_c2_decode.cu path}"
: "${C2_EXPECTED_CANDIDATE_SHA:?submit the reviewed v9 plugin SHA-256}"
: "${C2_EXPECTED_CANDIDATE_SOURCE_SHA:?submit the reviewed v9 source SHA-256}"
: "${C2_CANDIDATE_RESOURCE_GATE:?submit the reviewed v9 resource-gate path}"
: "${C2_EXPECTED_CANDIDATE_RESOURCE_GATE_SHA:?submit the reviewed v9 resource-gate SHA-256}"
: "${C2_CANDIDATE_PROVENANCE:?submit the reviewed v9 provenance-manifest path}"
: "${C2_EXPECTED_CANDIDATE_PROVENANCE_SHA:?submit the reviewed v9 provenance-manifest SHA-256}"
: "${C2_DIRECTED_RESULT:?submit the passing v9 directed result path}"
: "${C2_EXPECTED_DIRECTED_RESULT_SHA:?submit the passing v9 directed result SHA-256}"
: "${C2_DIRECTED_FINAL_STATUS:?submit the passing v9 directed final-status path}"
: "${C2_EXPECTED_DIRECTED_FINAL_STATUS_SHA:?submit the passing v9 directed final-status SHA-256}"
: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?submit the reviewed runtime-script SHA-256}"
for digest in "${C2_EXPECTED_REFERENCE_SHA}" "${C2_EXPECTED_CANDIDATE_SHA}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${EXPECTED_REFERENCE_SHA}" \
  "${EXPECTED_REFERENCE_SOURCE_SHA}" "${C2_EXPECTED_CANDIDATE_SOURCE_SHA}" \
  "${C2_EXPECTED_CANDIDATE_RESOURCE_GATE_SHA}" "${C2_EXPECTED_CANDIDATE_PROVENANCE_SHA}" \
  "${C2_EXPECTED_DIRECTED_RESULT_SHA}" "${C2_EXPECTED_DIRECTED_FINAL_STATUS_SHA}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid SHA-256: ${digest}" >&2; exit 2; }
done
[[ "${C2_REFERENCE_PLUGIN}" == "${EXPECTED_REFERENCE_LIBRARY}" ]] || {
  echo 'reference plugin path is not the frozen v7 job12484 DSO' >&2; exit 2;
}
[[ "${C2_EXPECTED_REFERENCE_SHA}" == "${EXPECTED_REFERENCE_SHA}" ]] || {
  echo 'reference plugin SHA is not the frozen v7 job12484 SHA' >&2; exit 2;
}
[[ "${C2_CANDIDATE_PLUGIN}" == /* && "${C2_CANDIDATE_SOURCE}" == /* && \
   "${C2_CANDIDATE_RESOURCE_GATE}" == /* && "${C2_CANDIDATE_PROVENANCE}" == /* && \
   "${C2_DIRECTED_RESULT}" == /* && "${C2_DIRECTED_FINAL_STATUS}" == /* ]] || {
  echo 'candidate and directed-evidence inputs must be absolute paths' >&2; exit 2;
}
[[ -d "${ARTIFACT_ROOT}" ]] || { echo 'artifact root missing' >&2; exit 2; }
[[ -z "$(find "${ARTIFACT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
  echo 'artifact root must be empty before v9 stress submission' >&2; exit 2;
}
[[ ! -e "${ARTIFACT_DIR}" ]] || { echo 'job output reused' >&2; exit 2; }
mkdir -p "${ARTIFACT_DIR}" || exit 2
export PYTHONPYCACHEPREFIX="${ARTIFACT_DIR}/pycache"
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}") || exit 2
SUBMITTED_SCRIPT_COPY=${ARTIFACT_DIR}/submitted-script-job${SLURM_JOB_ID}.slurm
FINAL_STATUS=${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt
FINAL_STATUS_SIDECAR=${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.sha256
[[ "${SCRIPT_PATH}" == /* ]] || { echo 'script path must be absolute' >&2; exit 2; }
export PATH=/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin
EXPECTED_GPU_UUID=
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
  selector=$(allocated_gpu_selector) || return 1
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
  selector=$(allocated_gpu_selector) || return 1
  nvidia-smi --id "${selector}" --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}
snapshot() {
  local selector
  selector=$(allocated_gpu_selector) || return 1
  printf '\n===== %s UTC=%s =====\n' "$1" "$(date -u +%FT%TZ)"
  nvidia-smi --id "${selector}" --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap \
    --format=csv,noheader,nounits || return 1
  printf '%s\n' 'COMPUTE_APPS'
  compute_apps || return 1
}
sha256_of() { sha256sum "$1" | awk '{print $1}'; }
finalize() {
  local original_rc=$? final_rc finalizer_error=0 apps_after='' apps_rc uuid_rc post_uuid=''
  local -a pipeline_rc
  final_rc=${original_rc}
  trap - EXIT
  set +e
  case "${final_rc}" in
    0|2|3) ;;
    *) final_rc=2; finalizer_error=1 ;;
  esac
  snapshot FINAL_POST | tee "${ARTIFACT_DIR}/gpu-final_post-job${SLURM_JOB_ID}.txt"
  pipeline_rc=("${PIPESTATUS[@]}")
  [[ "${pipeline_rc[0]}" -eq 0 && "${pipeline_rc[1]}" -eq 0 ]] || finalizer_error=1
  apps_after=$(compute_apps)
  apps_rc=$?
  [[ "${apps_rc}" -eq 0 && -z "${apps_after}" ]] || finalizer_error=1
  post_uuid=$(visible_gpu_uuid)
  uuid_rc=$?
  [[ "${uuid_rc}" -eq 0 && -n "${EXPECTED_GPU_UUID}" && "${post_uuid}" == "${EXPECTED_GPU_UUID}" ]] || finalizer_error=1
  [[ "${finalizer_error}" -eq 0 ]] || final_rc=2
  printf 'ORIGINAL_RC=%s POST_PIPELINE_RC=%s,%s APPS_QUERY_RC=%s APPS_EMPTY=%s UUID_QUERY_RC=%s EXPECTED_UUID=%s POST_UUID=%s\n' \
    "${original_rc}" "${pipeline_rc[0]:-missing}" "${pipeline_rc[1]:-missing}" "${apps_rc}" \
    "$([[ -z "${apps_after}" ]] && echo true || echo false)" "${uuid_rc}" \
    "${EXPECTED_GPU_UUID:-unset}" "${post_uuid:-unset}" \
    > "${ARTIFACT_DIR}/finalizer-diagnostics-job${SLURM_JOB_ID}.txt" || {
      final_rc=2
      finalizer_error=1
    }
  printf 'FINAL_RC=%s ORIGINAL_RC=%s FINALIZER_ERROR=%s UTC=%s\n' \
    "${final_rc}" "${original_rc}" "${finalizer_error}" "$(date -u +%FT%TZ)" \
    > "${FINAL_STATUS}" || {
      final_rc=2
      finalizer_error=1
      printf 'FINAL_RC=%s ORIGINAL_RC=%s FINALIZER_ERROR=%s UTC=%s\n' \
        "${final_rc}" "${original_rc}" "${finalizer_error}" "$(date -u +%FT%TZ)" \
        > "${FINAL_STATUS}" || true
    }
  if ! sha256sum "${ARTIFACT_DIR}/finalizer-diagnostics-job${SLURM_JOB_ID}.txt" \
      "${FINAL_STATUS}" > "${FINAL_STATUS_SIDECAR}"; then
    final_rc=2
    finalizer_error=1
    printf 'FINAL_RC=%s ORIGINAL_RC=%s FINALIZER_ERROR=%s UTC=%s\n' \
      "${final_rc}" "${original_rc}" "${finalizer_error}" "$(date -u +%FT%TZ)" \
      > "${FINAL_STATUS}" || true
    sha256sum "${ARTIFACT_DIR}/finalizer-diagnostics-job${SLURM_JOB_ID}.txt" \
      "${FINAL_STATUS}" > "${FINAL_STATUS_SIDECAR}" || true
  fi
  exit "${final_rc}"
}
trap finalize EXIT

exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1
printf '%s  %s\n' "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" | sha256sum -c - || exit 2
cp -- "${SCRIPT_PATH}" "${SUBMITTED_SCRIPT_COPY}" || exit 2
chmod a-w "${SUBMITTED_SCRIPT_COPY}" || exit 2
printf '%s  %s\n' "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SUBMITTED_SCRIPT_COPY}" | sha256sum -c - || exit 2

snapshot PRE | tee "${ARTIFACT_DIR}/gpu-pre-job${SLURM_JOB_ID}.txt" || exit 2
gpu_line=$(nvidia-smi --id "$(allocated_gpu_selector)" --query-gpu=name,compute_cap --format=csv,noheader,nounits) || exit 2
apps_before=$(compute_apps) || { echo 'unable to query pre-run compute apps' >&2; exit 2; }
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* ]] || {
  echo 'requires a B300 capability-10.3 GPU' >&2; exit 2;
}
[[ -z "${apps_before}" ]] || { echo 'GPU is not clean before measurement' >&2; exit 2; }
EXPECTED_GPU_UUID=$(visible_gpu_uuid) || { echo 'unable to capture allocated GPU UUID' >&2; exit 2; }
[[ -f "${C2_REFERENCE_PLUGIN}" && -f "${C2_CANDIDATE_PLUGIN}" && -f "${C2_CANDIDATE_SOURCE}" ]] || exit 2
[[ -f "${BASE_HARNESS}" && -f "${C2_CANDIDATE_RESOURCE_GATE}" && -f "${C2_CANDIDATE_PROVENANCE}" ]] || exit 2
[[ -f "${C2_DIRECTED_RESULT}" && -f "${C2_DIRECTED_FINAL_STATUS}" ]] || exit 2
printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${EXPECTED_REFERENCE_SHA}" "${C2_REFERENCE_PLUGIN}" \
  "${C2_EXPECTED_CANDIDATE_SHA}" "${C2_CANDIDATE_PLUGIN}" \
  "${EXPECTED_REFERENCE_SOURCE_SHA}" "${EXPECTED_REFERENCE_SOURCE}" \
  "${C2_EXPECTED_CANDIDATE_SOURCE_SHA}" "${C2_CANDIDATE_SOURCE}" \
  "${C2_EXPECTED_CANDIDATE_RESOURCE_GATE_SHA}" "${C2_CANDIDATE_RESOURCE_GATE}" \
  "${C2_EXPECTED_CANDIDATE_PROVENANCE_SHA}" "${C2_CANDIDATE_PROVENANCE}" | sha256sum -c - || exit 2
printf '%s  %s\n%s  %s\n' \
  "${C2_EXPECTED_DIRECTED_RESULT_SHA}" "${C2_DIRECTED_RESULT}" \
  "${C2_EXPECTED_DIRECTED_FINAL_STATUS_SHA}" "${C2_DIRECTED_FINAL_STATUS}" | sha256sum -c - || exit 2
[[ "$(sha256_of "${BASE_HARNESS}")" == "${EXPECTED_BASE_HARNESS_SHA}" ]] || exit 2
[[ "$(sha256_of "${C2_REFERENCE_PLUGIN}")" != "$(sha256_of "${C2_CANDIDATE_PLUGIN}")" ]] || exit 2

# Re-read the candidate AOT resource proof instead of assuming an AOT job was
# sufficient.  It must prove the exact v9 shared/register budget used by the
# candidate DSO before timing can begin.
RESOURCE_LINE=$(grep -m 1 -E 'REG:[[:space:]]*[0-9]+.*STACK:[[:space:]]*[0-9]+.*SHARED:[[:space:]]*[0-9]+.*LOCAL:[[:space:]]*[0-9]+' "${C2_CANDIDATE_RESOURCE_GATE}")
[[ -n "${RESOURCE_LINE}" ]] || { echo 'candidate native-kernel resource line missing' >&2; exit 2; }
read_resource() { sed -nE "s/.*$1:[[:space:]]*([0-9]+).*/\\1/p" <<< "${RESOURCE_LINE}"; }
REG=$(read_resource REG); STACK=$(read_resource STACK); SHARED=$(read_resource SHARED); LOCAL=$(read_resource LOCAL)
[[ "${REG}" =~ ^[0-9]+$ && "${STACK}" =~ ^[0-9]+$ && "${SHARED}" =~ ^[0-9]+$ && "${LOCAL}" =~ ^[0-9]+$ ]] || exit 2
(( STACK == 0 && LOCAL == 0 && SHARED == 30880 )) || {
  echo "candidate resource gate failed: REG=${REG} STACK=${STACK} SHARED=${SHARED} LOCAL=${LOCAL}" >&2
  exit 2
}
grep -Fx "${C2_EXPECTED_CANDIDATE_SHA}  ${C2_CANDIDATE_PLUGIN}" "${C2_CANDIDATE_PROVENANCE}" || exit 2
grep -Fx "${C2_EXPECTED_CANDIDATE_SOURCE_SHA}  ${C2_CANDIDATE_SOURCE}" "${C2_CANDIDATE_PROVENANCE}" || exit 2
if ! "${PYTHON}" - "${C2_DIRECTED_RESULT}" "${C2_DIRECTED_FINAL_STATUS}" \
  "${C2_CANDIDATE_PLUGIN}" <<'PY'
import json
import sys
from pathlib import Path

result_path, status_path, candidate_path = map(Path, sys.argv[1:])
payload = json.loads(result_path.read_text(encoding="utf-8"))
assert payload["schema"] == "c2-native-c2-v6-register-numerator-directed-v1"
assert payload["all_gates_pass"] is True
assert Path(payload["operator_library"]).resolve() == candidate_path.resolve()
assert payload["rank_directed_case_count"] == 4
assert all(case["all_gates_pass"] is True for case in payload["rank_directed"])
assert payload["softmax_directed_case_count"] == 2
assert all(case["all_gates_pass"] is True for case in payload["softmax_directed"])
assert payload["all_invalid"]["all_gates_pass"] is True
assert payload["profiler"]["exactly_one_dispatcher_event"] is True
assert payload["profiler"]["exactly_one_native_kernel_event"] is True
status = dict(token.split("=", 1) for token in status_path.read_text(encoding="utf-8").split() if "=" in token)
assert status["FINAL_RC"] == "0"
assert status["ORIGINAL_RC"] == "0"
assert status["FINALIZER_ERROR"] == "0"
assert status["POST_CHECKED"] == "1"
PY
then
  echo 'fresh v9 directed evidence validation failed' >&2
  exit 2
fi

FIXTURE_WRITER=${ARTIFACT_DIR}/write_fixed_fixtures.py
FIXTURE_WORKER=${ARTIFACT_DIR}/run_fixed_fixture_worker.py
FIXTURE=${ARTIFACT_DIR}/fixtures-8-seeds.pt
FIXTURE_META=${ARTIFACT_DIR}/fixtures-8-seeds-meta.json
cat > "${FIXTURE_WRITER}" <<'PY'
#!/usr/bin/env python3
"""Create one immutable CPU fixture payload from the frozen base generator."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("c2_v9_fixture_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("Contract", "_make_inputs", "_validate_args"):
        if not hasattr(module, name):
            raise RuntimeError(f"base harness lacks {name}")
    return module


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(repr(tuple(cpu.shape)).encode("ascii"))
    digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-harness", type=Path, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(part) for part in args.seeds.split(",")]
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("expected exactly eight unique fixed seeds")
    base = load_base(args.base_harness)
    contract = base._validate_args(type("Args", (), {
        "num_physical_pages": 64, "max_logical_pages": 32,
        "scale": 1.0 / (128.0 ** 0.5), "q_scale": 0.25,
        "k_scale": 0.25, "v_scale": 0.5, "atol": 1.0e-4, "rtol": 1.0e-3,
    })())
    rows = []
    for seed in seeds:
        tensors = tuple(item.detach().cpu().contiguous() for item in base._make_inputs(contract, seed))
        rows.append({"seed": seed, "inputs": tensors,
                     "tensor_sha256": [tensor_sha256(item) for item in tensors]})
    payload = {
        "schema": "c2-native-v9-parallel-merge-fixed-fixtures-v1", "seeds": seeds,
        "contract": {"num_physical_pages": contract.num_physical_pages,
                     "max_logical_pages": contract.max_logical_pages,
                     "scale": contract.scale, "q_scale": contract.q_scale,
                     "k_scale": contract.k_scale, "v_scale": contract.v_scale},
        "rows": rows,
    }
    torch.save(payload, args.fixture)
    meta = {
        "schema": payload["schema"], "seeds": seeds,
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "base_harness_sha256": hashlib.sha256(args.base_harness.read_bytes()).hexdigest(),
        "contract": payload["contract"],
        "per_seed_tensor_sha256": {str(row["seed"]): row["tensor_sha256"] for row in rows},
    }
    args.meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
PY
cat > "${FIXTURE_WORKER}" <<'PY'
#!/usr/bin/env python3
"""One direct-plugin C2 worker over a pre-generated immutable fixture set."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("c2_v9_worker_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = ("Contract", "_BATCH", "_QUERY_HEADS", "_HEAD_DIM", "_oracle")
    for name in required:
        if not hasattr(module, name):
            raise RuntimeError(f"base harness lacks {name}")
    return module


def visible_gpu_uuid() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if len(entries) == 1 and entries[0].startswith("GPU-"):
        return entries[0]
    if len(entries) == 1 and entries[0].isdigit():
        return subprocess.check_output(
            ["nvidia-smi", "--id", entries[0], "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    props = torch.cuda.get_device_properties(0)
    uuid = getattr(props, "uuid", None)
    if isinstance(uuid, bytes):
        uuid = uuid.decode("ascii")
    if isinstance(uuid, str) and uuid.startswith("GPU-"):
        return uuid
    raise RuntimeError("cannot establish the allocated physical GPU UUID fail-closed")


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def call(op: Any, output: torch.Tensor, inputs: tuple[torch.Tensor, ...], contract: Any):
    return op(output, *inputs, contract.scale, contract.q_scale, contract.k_scale, contract.v_scale)


def profile_one(op: Any, inputs: tuple[torch.Tensor, ...], contract: Any, shape: tuple[int, ...]) -> dict[str, Any]:
    output = torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    pointer = output.data_ptr()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        returned = call(op, output, inputs, contract)
        torch.cuda.synchronize()
    events = profiler.key_averages()
    dispatcher = [event for event in events if str(event.key) == "_C::native_c2_msa_decode"]
    kernels = [event for event in events if "native_c2_msa_decode_kernel" in str(event.key)]
    checks = {
        "one_dispatcher_event": len(dispatcher) == 1 and int(getattr(dispatcher[0], "count", 0)) == 1,
        "one_native_cuda_kernel_event": len(kernels) == 1 and int(getattr(kernels[0], "count", 0)) == 1,
        "caller_pointer_unchanged": output.data_ptr() == pointer,
        "return_is_none": returned is None,
    }
    return {
        "pass": all(checks.values()), "checks": checks,
        "dispatcher_events": [{"key": str(event.key), "count": int(getattr(event, "count", 0))} for event in dispatcher],
        "cuda_kernel_events": [{"key": str(event.key), "count": int(getattr(event, "count", 0))} for event in kernels],
    }


def exercise(base: Any, op: Any, inputs: tuple[torch.Tensor, ...], contract: Any, seed: int) -> dict[str, Any]:
    shape = (base._BATCH, base._QUERY_HEADS, base._HEAD_DIM)
    output = torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    pointer_before = output.data_ptr()
    returned = call(op, output, inputs, contract)
    torch.cuda.synchronize()
    reference = base._oracle(*inputs, contract, torch.float32)
    reference_fp64 = base._oracle(*inputs, contract, torch.float64)
    actual = output.to(torch.float32)
    actual_fp64 = output.to(torch.float64)
    difference = (actual - reference).abs()
    fp64_difference = (actual_fp64 - reference_fp64).abs()
    denominator = reference.abs().clamp_min(torch.finfo(torch.float32).eps)
    finite = bool(torch.isfinite(actual).all().item())
    allclose = bool(torch.allclose(actual, reference, atol=1.0e-4, rtol=1.0e-3))
    fp64_allclose = bool(torch.allclose(actual_fp64, reference_fp64, atol=1.0e-4, rtol=1.0e-3))
    reference_fp64_fp32_agree = bool(torch.allclose(reference_fp64, reference.to(torch.float64), atol=1.0e-4, rtol=1.0e-3))
    repeat_outputs = [output.clone()]
    repeat_pointers = output.data_ptr() == pointer_before
    repeat_returns = returned is None
    for _ in range(3):
        repeated = torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16)
        repeated_pointer = repeated.data_ptr()
        repeated_return = call(op, repeated, inputs, contract)
        repeat_pointers = repeat_pointers and repeated.data_ptr() == repeated_pointer
        repeat_returns = repeat_returns and repeated_return is None
        repeat_outputs.append(repeated)
    torch.cuda.synchronize()
    bitwise = all(torch.equal(repeat_outputs[0], other) for other in repeat_outputs[1:])
    warmup_output = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    for _ in range(30):
        if call(op, warmup_output, inputs, contract) is not None:
            raise RuntimeError("operator returned a value during warmup")
    torch.cuda.synchronize()
    timing_output = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    timing_pointer = timing_output.data_ptr()
    pairs = []
    for _ in range(200):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        timing_return = call(op, timing_output, inputs, contract)
        end.record()
        if timing_return is not None:
            raise RuntimeError("operator returned a value during timing")
        pairs.append((start, end))
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in pairs]
    profile = profile_one(op, inputs, contract, shape)
    timing_finite = bool(torch.isfinite(timing_output).all().item())
    all_gates = bool(finite and allclose and fp64_allclose and reference_fp64_fp32_agree
                     and output.data_ptr() == pointer_before and returned is None
                     and repeat_pointers and repeat_returns and bitwise
                     and timing_output.data_ptr() == timing_pointer and timing_finite and profile["pass"])
    return {
        "seed": seed, "all_gates_pass": all_gates,
        "correctness": {"oracle_dtype": "float32", "atol": 1.0e-4, "rtol": 1.0e-3,
                        "finite_output": finite, "allclose": allclose,
                        "max_abs": float(difference.max().item()),
                        "mean_abs": float(difference.mean().item()),
                        "max_rel": float((difference / denominator).max().item()),
                        "fp64_allclose": fp64_allclose,
                        "fp64_max_abs": float(fp64_difference.max().item()),
                        "reference_fp64_fp32_agree": reference_fp64_fp32_agree},
        "caller_output": {"pointer_before": pointer_before, "pointer_after": output.data_ptr(),
                          "pointer_unchanged": output.data_ptr() == pointer_before,
                          "return_is_none": returned is None,
                          "timing_pointer_unchanged": timing_output.data_ptr() == timing_pointer,
                          "timing_output_finite": timing_finite},
        "stability": {"repeat_count": 4, "bitwise_repeatable": bitwise,
                      "repeat_pointers_unchanged": repeat_pointers,
                      "repeat_returns_none": repeat_returns},
        "profiling": profile,
        "latency": {"mean_ms": statistics.fmean(samples), "min_ms": min(samples),
                    "p50_ms": percentile(samples, 0.50), "p95_ms": percentile(samples, 0.95),
                    "max_ms": max(samples), "iterations": len(samples),
                    "method": "one CUDA-event pair per direct call; warmup/oracle/profiler excluded"},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device must be visible")
    if torch.cuda.get_device_capability(0) != (10, 3):
        raise RuntimeError(f"B300 capability required, got {torch.cuda.get_device_capability(0)}")
    if not args.library.is_absolute() or not args.base_harness.is_absolute() or not args.fixture.is_absolute():
        raise ValueError("library, base harness, and fixture must be absolute paths")
    if sha256(args.fixture) != args.fixture_sha256:
        raise RuntimeError("fixture file SHA-256 drifted")
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    if fixture.get("schema") != "c2-native-v9-parallel-merge-fixed-fixtures-v1":
        raise RuntimeError("unexpected fixture schema")
    seeds = [17, 23, 42, 2024, 314159, 20260801, 20260815, 20260829]
    if fixture.get("seeds") != seeds or len(fixture.get("rows", [])) != len(seeds):
        raise RuntimeError("fixture seed contract drifted")
    base = load_base(args.base_harness)
    contract = base.Contract(**fixture["contract"])
    torch.ops.load_library(str(args.library))
    if not (hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "native_c2_msa_decode")):
        raise RuntimeError("plugin did not register _C::native_c2_msa_decode")
    if not torch._C._dispatch_has_kernel_for_dispatch_key("_C::native_c2_msa_decode", "CUDA"):
        raise RuntimeError("plugin lacks CUDA dispatch implementation")
    op = torch.ops._C.native_c2_msa_decode
    records = []
    for expected_seed, row in zip(seeds, fixture["rows"], strict=True):
        if row.get("seed") != expected_seed:
            raise RuntimeError("fixture row order drifted")
        inputs = tuple(tensor.to(device="cuda").contiguous() for tensor in row["inputs"])
        records.append(exercise(base, op, inputs, contract, expected_seed) | {"fixture_tensor_sha256": row["tensor_sha256"]})
    gpu_uuid = visible_gpu_uuid()
    return {
        "schema": "c2-native-plugin-v9-parallel-merge-fixed-fixture-worker-v1", "role": args.role,
        "all_gates_pass": all(record["all_gates_pass"] for record in records),
        "operator_library": str(args.library.resolve()), "operator_library_sha256": sha256(args.library),
        "base_harness": str(args.base_harness.resolve()), "base_harness_sha256": sha256(args.base_harness),
        "worker_harness": str(Path(__file__).resolve()), "worker_harness_sha256": sha256(Path(__file__)),
        "fixture": str(args.fixture.resolve()), "fixture_sha256": args.fixture_sha256,
        "fixture_schema": fixture["schema"], "fixture_seeds": seeds,
        "contract": fixture["contract"], "correctness_gate": {"oracle_dtype": "float32", "atol": 1.0e-4, "rtol": 1.0e-3},
        "measurement": {"warmup_per_seed": 30, "iterations_per_seed": 200, "stability_repeats": 4},
        "dispatch": "direct_torch.ops._C.native_c2_msa_decode", "no_monkeypatch": True,
        "device": {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
                   "gpu_uuid": gpu_uuid, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "seeds": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--base-harness", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--role", required=True)
    args = parser.parse_args()
    try:
        payload = run(args)
    except Exception as error:
        payload = {"schema": "c2-native-plugin-v9-parallel-merge-fixed-fixture-worker-v1", "role": args.role,
                   "all_gates_pass": False, "error": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc()}
        print(json.dumps(payload, allow_nan=False, sort_keys=True))
        return 1
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0 if payload["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
PY
chmod 0644 "${FIXTURE_WRITER}" "${FIXTURE_WORKER}"
"${PYTHON}" -m py_compile "${FIXTURE_WRITER}" "${FIXTURE_WORKER}" || exit 2
"${PYTHON}" "${FIXTURE_WRITER}" --base-harness "${BASE_HARNESS}" --seeds "${SEEDS}" \
  --fixture "${FIXTURE}" --meta "${FIXTURE_META}" | tee "${ARTIFACT_DIR}/fixture-writer-job${SLURM_JOB_ID}.json" || exit 2
FIXTURE_SHA=$(sha256_of "${FIXTURE}")
META_FIXTURE_SHA=$("${PYTHON}" - "${FIXTURE_META}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["fixture_sha256"])
PY
) || exit 2
[[ "${FIXTURE_SHA}" == "${META_FIXTURE_SHA}" ]] || {
  echo 'fixture SHA disagrees with fixture metadata' >&2; exit 2;
}

printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${EXPECTED_REFERENCE_SHA}" "${C2_REFERENCE_PLUGIN}" \
  "${C2_EXPECTED_CANDIDATE_SHA}" "${C2_CANDIDATE_PLUGIN}" \
  "${EXPECTED_REFERENCE_SOURCE_SHA}" "${EXPECTED_REFERENCE_SOURCE}" \
  "${C2_EXPECTED_CANDIDATE_SOURCE_SHA}" "${C2_CANDIDATE_SOURCE}" \
  "${EXPECTED_BASE_HARNESS_SHA}" "${BASE_HARNESS}" \
  "${FIXTURE_SHA}" "${FIXTURE}" \
  "$(sha256_of "${FIXTURE_META}")" "${FIXTURE_META}" \
  "$(sha256_of "${FIXTURE_WRITER}")" "${FIXTURE_WRITER}" \
  "$(sha256_of "${FIXTURE_WORKER}")" "${FIXTURE_WORKER}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SUBMITTED_SCRIPT_COPY}" \
  > "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" || exit 2
sha256sum "${C2_CANDIDATE_RESOURCE_GATE}" "${C2_CANDIDATE_PROVENANCE}" \
  >> "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
printf '%s  %s\n%s  %s\n' \
  "${C2_EXPECTED_DIRECTED_RESULT_SHA}" "${C2_DIRECTED_RESULT}" \
  "${C2_EXPECTED_DIRECTED_FINAL_STATUS_SHA}" "${C2_DIRECTED_FINAL_STATUS}" \
  >> "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" || exit 2
"${PYTHON}" -c 'import platform, torch; print({"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda})' || exit 2
ldd "${C2_REFERENCE_PLUGIN}" | tee "${ARTIFACT_DIR}/reference-v7-plugin-ldd-job${SLURM_JOB_ID}.txt" || exit 2
ldd "${C2_CANDIDATE_PLUGIN}" | tee "${ARTIFACT_DIR}/candidate-v9-parallel-merge-plugin-ldd-job${SLURM_JOB_ID}.txt" || exit 2

REF_A=${ARTIFACT_DIR}/reference-v7-a-round-job${SLURM_JOB_ID}.json
CAN_B1=${ARTIFACT_DIR}/candidate-v9-parallel-merge-b1-round-job${SLURM_JOB_ID}.json
CAN_B2=${ARTIFACT_DIR}/candidate-v9-parallel-merge-b2-round-job${SLURM_JOB_ID}.json
REF_A2=${ARTIFACT_DIR}/reference-v7-a2-round-job${SLURM_JOB_ID}.json
RUN_STATUS=${ARTIFACT_DIR}/worker-exit-status-job${SLURM_JOB_ID}.json
AGGREGATE=${ARTIFACT_DIR}/v9-parallel-merge-vs-v7-3pct-confirmation-aggregate-job${SLURM_JOB_ID}.json
DECISION=${ARTIFACT_DIR}/v9-parallel-merge-vs-v7-3pct-significance-decision-job${SLURM_JOB_ID}.json
run_worker() {
  local role=$1
  local library=$2
  local output=$3
  timeout --foreground --preserve-status --kill-after=10s 600s "${PYTHON}" "${FIXTURE_WORKER}" \
    --role "${role}" --library "${library}" --base-harness "${BASE_HARNESS}" \
    --fixture "${FIXTURE}" --fixture-sha256 "${FIXTURE_SHA}" > "${output}"
}
set +e
run_worker reference_v7_a "${C2_REFERENCE_PLUGIN}" "${REF_A}"; REF_A_RC=$?
run_worker candidate_v9_parallel_merge_b1 "${C2_CANDIDATE_PLUGIN}" "${CAN_B1}"; CAN_B1_RC=$?
run_worker candidate_v9_parallel_merge_b2 "${C2_CANDIDATE_PLUGIN}" "${CAN_B2}"; CAN_B2_RC=$?
run_worker reference_v7_a2 "${C2_REFERENCE_PLUGIN}" "${REF_A2}"; REF_A2_RC=$?
set -e
"${PYTHON}" - "${RUN_STATUS}" "${REF_A_RC}" "${CAN_B1_RC}" "${CAN_B2_RC}" "${REF_A2_RC}" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
labels = ("reference_v7_a", "candidate_v9_parallel_merge_b1", "candidate_v9_parallel_merge_b2", "reference_v7_a2")
codes = [int(value) for value in sys.argv[2:]]
out.write_text(json.dumps({"schema": "c2-native-plugin-v9-parallel-merge-worker-exit-status-v1", "exit_status": dict(zip(labels, codes, strict=True))}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

set +e
"${PYTHON}" - "${REF_A}" "${CAN_B1}" "${CAN_B2}" "${REF_A2}" "${RUN_STATUS}" \
  "${AGGREGATE}" "${DECISION}" "${C2_REFERENCE_PLUGIN}" "${C2_CANDIDATE_PLUGIN}" \
  "${C2_CANDIDATE_SOURCE}" "${BASE_HARNESS}" "${FIXTURE}" "${FIXTURE_META}" \
  "${FIXTURE_WORKER}" "${SUBMITTED_SCRIPT_COPY}" "${C2_CANDIDATE_RESOURCE_GATE}" "${C2_CANDIDATE_PROVENANCE}" \
  "${C2_DIRECTED_RESULT}" "${C2_DIRECTED_FINAL_STATUS}" "${EXPECTED_GPU_UUID}" \
  "${C2_EXPECTED_REFERENCE_SHA}" "${C2_EXPECTED_CANDIDATE_SHA}" \
  "${C2_EXPECTED_CANDIDATE_SOURCE_SHA}" "${C2_EXPECTED_DIRECTED_RESULT_SHA}" \
  "${C2_EXPECTED_DIRECTED_FINAL_STATUS_SHA}" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

(ref_a_path, can_b1_path, can_b2_path, ref_a2_path, status_path, aggregate_path,
 decision_path, ref_library, can_library, can_source, base_harness, fixture,
 fixture_meta, worker_harness, script_path, resource_gate, provenance,
 directed_result_path, directed_status_path) = map(Path, sys.argv[1:-6])
(expected_gpu_uuid, EXPECTED_REF_SHA, EXPECTED_CANDIDATE_SHA, EXPECTED_SOURCE_SHA,
 EXPECTED_DIRECTED_RESULT_SHA, EXPECTED_DIRECTED_STATUS_SHA) = sys.argv[-6:]
assert expected_gpu_uuid.startswith("GPU-"), expected_gpu_uuid
SEEDS = [17, 23, 42, 2024, 314159, 20260801, 20260815, 20260829]

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path: Path, issues: list[str], label: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        issues.append(f"{label}: unreadable JSON: {type(error).__name__}: {error}")
        return None

def median(values: list[float]) -> float:
    return statistics.median(values)

issues: list[str] = []
statuses = load_json(status_path, issues, "worker exit status")
directed = load_json(directed_result_path, issues, "fresh v9 directed result")
if not isinstance(directed, dict) or directed.get("all_gates_pass") is not True:
    issues.append("fresh v9 directed gate did not pass")
else:
    if (directed.get("rank_directed_case_count") != 4
            or directed.get("softmax_directed_case_count") != 2
            or directed.get("all_invalid", {}).get("all_gates_pass") is not True
            or directed.get("profiler", {}).get("exactly_one_dispatcher_event") is not True
            or directed.get("profiler", {}).get("exactly_one_native_kernel_event") is not True):
        issues.append("fresh v9 directed gate semantic contract drift")
try:
    directed_status = dict(token.split("=", 1) for token in directed_status_path.read_text(encoding="utf-8").split() if "=" in token)
except Exception as error:
    issues.append(f"fresh v9 directed final status unreadable: {type(error).__name__}: {error}")
    directed_status = {}
if any(directed_status.get(key) != value for key, value in {
        "FINAL_RC": "0", "ORIGINAL_RC": "0", "FINALIZER_ERROR": "0", "POST_CHECKED": "1"}.items()):
    issues.append("fresh v9 directed final status did not pass")
round_specs = [("reference_v7_a", ref_a_path, ref_library),
               ("candidate_v9_parallel_merge_b1", can_b1_path, can_library),
               ("candidate_v9_parallel_merge_b2", can_b2_path, can_library),
               ("reference_v7_a2", ref_a2_path, ref_library)]
rounds: dict[str, dict] = {}
for label, path, library in round_specs:
    payload = load_json(path, issues, label)
    if payload is None:
        continue
    rounds[label] = payload
    if payload.get("schema") != "c2-native-plugin-v9-parallel-merge-fixed-fixture-worker-v1":
        issues.append(f"{label}: schema")
    if payload.get("role") != label or payload.get("all_gates_pass") is not True:
        issues.append(f"{label}: worker gate")
    if Path(payload.get("operator_library", ".")).resolve() != library.resolve():
        issues.append(f"{label}: library path")
    if payload.get("operator_library_sha256") != digest(library):
        issues.append(f"{label}: library SHA")
    if payload.get("fixture_sha256") != digest(fixture):
        issues.append(f"{label}: fixture SHA")
    if payload.get("fixture_seeds") != SEEDS or payload.get("dispatch") != "direct_torch.ops._C.native_c2_msa_decode":
        issues.append(f"{label}: fixture/dispatch contract")
    if payload.get("correctness_gate") != {"oracle_dtype": "float32", "atol": 1.0e-4, "rtol": 1.0e-3}:
        issues.append(f"{label}: FP32 oracle contract")
    records = payload.get("seeds", [])
    if len(records) != len(SEEDS) or [record.get("seed") for record in records] != SEEDS:
        issues.append(f"{label}: seed list")
        continue
    for record in records:
        correctness = record.get("correctness", {})
        caller = record.get("caller_output", {})
        stability = record.get("stability", {})
        profiling = record.get("profiling", {})
        latency = record.get("latency", {})
        if not (record.get("all_gates_pass") is True and correctness.get("allclose") is True
                and correctness.get("fp64_allclose") is True and correctness.get("reference_fp64_fp32_agree") is True
                and correctness.get("finite_output") is True and caller.get("pointer_unchanged") is True
                and caller.get("timing_pointer_unchanged") is True and caller.get("return_is_none") is True
                and stability.get("bitwise_repeatable") is True and stability.get("repeat_pointers_unchanged") is True
                and stability.get("repeat_returns_none") is True and profiling.get("pass") is True):
            issues.append(f"{label}: per-seed correctness/stability/profiler gate seed={record.get('seed')}")
        mean = latency.get("mean_ms")
        if not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean <= 0.0:
            issues.append(f"{label}: nonpositive latency seed={record.get('seed')}")

if not isinstance(statuses, dict) or any(statuses.get("exit_status", {}).get(label) != 0 for label, _, _ in round_specs):
    issues.append("one or more isolated workers returned nonzero")
if set(rounds) == {label for label, _, _ in round_specs}:
    baseline = {key: rounds["reference_v7_a"].get(key) for key in ("base_harness_sha256", "worker_harness_sha256", "fixture_sha256", "fixture", "fixture_schema", "fixture_seeds", "contract", "correctness_gate", "measurement", "environment")}
    gpu_uuids = []
    for label, payload in rounds.items():
        if {key: payload.get(key) for key in baseline} != baseline:
            issues.append(f"{label}: shared fixture/harness/environment contract drift")
        uuid = payload.get("device", {}).get("gpu_uuid")
        if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
            issues.append(f"{label}: missing physical GPU UUID")
        elif uuid != expected_gpu_uuid:
            issues.append(f"{label}: allocated GPU UUID drift")
        gpu_uuids.append(uuid)
    if len(set(gpu_uuids)) != 1:
        issues.append("workers did not use one physical GPU UUID")
else:
    gpu_uuids = []

reference_seed_means: list[float] = []
candidate_seed_means: list[float] = []
paired_improvements: list[float] = []
if not issues:
    for index in range(len(SEEDS)):
        ref = (float(rounds["reference_v7_a"]["seeds"][index]["latency"]["mean_ms"])
               + float(rounds["reference_v7_a2"]["seeds"][index]["latency"]["mean_ms"])) / 2.0
        candidate = (float(rounds["candidate_v9_parallel_merge_b1"]["seeds"][index]["latency"]["mean_ms"])
                     + float(rounds["candidate_v9_parallel_merge_b2"]["seeds"][index]["latency"]["mean_ms"])) / 2.0
        reference_seed_means.append(ref)
        candidate_seed_means.append(candidate)
        paired_improvements.append(1.0 - candidate / ref)
    rng = random.Random(20260830)
    bootstrap = []
    for _ in range(10000):
        indices = [rng.randrange(len(SEEDS)) for _ in SEEDS]
        bootstrap.append(1.0 - median([candidate_seed_means[i] for i in indices])
                         / median([reference_seed_means[i] for i in indices]))
    bootstrap.sort()
    bootstrap_lower = bootstrap[math.floor(0.025 * (len(bootstrap) - 1))]
    reference_median = median(reference_seed_means)
    candidate_median = median(candidate_seed_means)
    improvement = 1.0 - candidate_median / reference_median
else:
    bootstrap_lower = None
    reference_median = None
    candidate_median = None
    improvement = None

inputs = {
    "reference_v7_job12484_plugin": digest(ref_library),
    "candidate_v9_plugin": digest(can_library),
    "candidate_v9_parallel_merge_source": digest(can_source),
    "base_harness": digest(base_harness), "fixed_fixture": digest(fixture),
    "fixture_metadata": digest(fixture_meta), "worker_harness": digest(worker_harness),
    "driver_script": digest(script_path), "candidate_resource_gate": digest(resource_gate),
    "candidate_provenance": digest(provenance),
    "fresh_v9_directed_result": digest(directed_result_path),
    "fresh_v9_directed_final_status": digest(directed_status_path),
}
if inputs["reference_v7_job12484_plugin"] != EXPECTED_REF_SHA:
    issues.append("frozen v7 reference DSO SHA drift")
if inputs["candidate_v9_plugin"] != EXPECTED_CANDIDATE_SHA:
    issues.append("candidate v9 plugin SHA drift")
if inputs["candidate_v9_parallel_merge_source"] != EXPECTED_SOURCE_SHA:
    issues.append("candidate v9 parallel-merge source SHA drift")
if inputs["fresh_v9_directed_result"] != EXPECTED_DIRECTED_RESULT_SHA:
    issues.append("fresh v9 directed result SHA drift")
if inputs["fresh_v9_directed_final_status"] != EXPECTED_DIRECTED_STATUS_SHA:
    issues.append("fresh v9 directed final-status SHA drift")
if inputs["reference_v7_job12484_plugin"] == inputs["candidate_v9_plugin"]:
    issues.append("candidate DSO equals reference DSO")

correct_and_comparable = not issues
performance_gate = bool(correct_and_comparable and improvement is not None and improvement > 0.03
                        and bootstrap_lower is not None and bootstrap_lower >= 0.0)
accepted = performance_gate
aggregate = {
    "schema": "c2-native-plugin-v9-parallel-merge-vs-v7-3pct-confirmation-aggregate-v1",
    "round_order": [label for label, _, _ in round_specs],
    "process_isolation": {"same_slurm_step": True, "same_python_process": False,
                          "reason": "both DSOs register the same _C schema/CUDA impl; separate workers prevent ambiguous repeated registration"},
    "worker_exit_status": statuses, "issues": issues, "all_correctness_and_comparability_gates_pass": correct_and_comparable,
    "allocated_gpu_uuid": expected_gpu_uuid, "gpu_uuids": gpu_uuids, "input_checksums": inputs,
    "reference_seed_mean_ms": reference_seed_means, "candidate_seed_mean_ms": candidate_seed_means,
    "paired_seed_improvement_fraction": paired_improvements,
    "reference_paired_median_ms": reference_median, "candidate_paired_median_ms": candidate_median,
    "paired_median_improvement_fraction": improvement,
    "bootstrap": {"method": "10,000 deterministic paired-seed resamples; statistic=1-median(candidate)/median(reference)",
                   "seed": 20260830, "confidence": 0.95, "lower_bound_fraction": bootstrap_lower},
    "threshold_provenance": "User-defined strict >3% significance criterion for the current four-hour optimization continuation; this is an incremental v7-to-v9 gate.",
}
aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
decision = {
    "schema": "c2-native-plugin-v9-parallel-merge-vs-v7-3pct-significance-decision-v1",
    "accepted": accepted, "all_correctness_and_resource_identity_gates_pass": correct_and_comparable,
    "performance": {"paired_median_improvement_fraction": improvement, "required_improvement_fraction": 0.03,
                    "required_improvement_relation": "strictly greater than",
                    "bootstrap_95pct_lower_bound_fraction": bootstrap_lower,
                    "required_bootstrap_lower_bound_fraction": 0.0},
    "comparison_boundary": aggregate["process_isolation"], "one_physical_gpu_uuid": gpu_uuids[0] if len(set(gpu_uuids)) == 1 else None,
    "input_checksums": inputs, "aggregate_sha256": digest(aggregate_path), "issues": issues,
    "rejection_policy": "If any identity/correctness/comparability gate fails return 2. If valid but performance gates fail return 3; retain the frozen v7 reference and do not package v9.",
    "cross_gpu_claim": "none; this is a same-physical-GPU comparison only.",
    "threshold_provenance": aggregate["threshold_provenance"],
}
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(decision, sort_keys=True))
raise SystemExit(0 if accepted else (2 if not correct_and_comparable else 3))
PY
decision_rc=$?
set -e
snapshot POST | tee "${ARTIFACT_DIR}/gpu-post-job${SLURM_JOB_ID}.txt" || exit 2
apps_after=$(compute_apps) || { echo 'unable to query post-run compute apps' >&2; exit 2; }
[[ -z "${apps_after}" ]] || { echo 'GPU is not clean after measurement' >&2; exit 2; }
sha256sum -c "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" || exit 2
sha256sum "${REF_A}" "${CAN_B1}" "${CAN_B2}" "${REF_A2}" "${RUN_STATUS}" \
  "${AGGREGATE}" "${DECISION}" "${FIXTURE_META}" "${ARTIFACT_DIR}/fixture-writer-job${SLURM_JOB_ID}.json" \
  "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" \
  "${ARTIFACT_DIR}/reference-v7-plugin-ldd-job${SLURM_JOB_ID}.txt" \
  "${ARTIFACT_DIR}/candidate-v9-parallel-merge-plugin-ldd-job${SLURM_JOB_ID}.txt" \
  "${ARTIFACT_DIR}/gpu-pre-job${SLURM_JOB_ID}.txt" "${ARTIFACT_DIR}/gpu-post-job${SLURM_JOB_ID}.txt" \
  > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256" || exit 2
sha256sum -c "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256" || exit 2
exit "${decision_rc}"
