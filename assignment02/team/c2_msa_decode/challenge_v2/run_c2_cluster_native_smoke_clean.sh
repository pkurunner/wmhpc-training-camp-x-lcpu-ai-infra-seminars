#!/usr/bin/env bash
# Authorized clean runner for the native CUDA cluster-communication prerequisite.
# It never submits work to Slurm; it must run inside an existing private B300 allocation.

set -Eeuo pipefail

if [[ "${C2_CLUSTER_NATIVE_SMOKE_AUTHORIZED:-}" != "1" || "${1:-}" != "--authorized-by-parent" ]]; then
    printf '%s\n' 'Refusing native cluster smoke without explicit coordinator authorization.' >&2
    exit 64
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Refusing native cluster smoke outside a Slurm allocation (SLURM_JOB_ID is unset).' >&2
    exit 64
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
c2_root="$(cd "${script_dir}/.." && pwd)"
assignment_root="$(cd "${c2_root}/../.." && pwd)"
python_bin="${C2_PYTHON_BIN:-${assignment_root}/.venv/bin/python}"
nvcc_candidate="${C2_NVCC_BIN:-$(command -v nvcc || true)}"
nvcc_bin=""
if [[ -n "${nvcc_candidate}" ]]; then
    nvcc_bin="$(readlink -f "${nvcc_candidate}")"
fi
source_path="${script_dir}/c2_cluster_native_smoke.cu"
out_dir="${C2_CLUSTER_NATIVE_OUT_DIR:-${c2_root}/experiment_logs/c2_cluster_native_smoke}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
audit_log="${out_dir}/c2_cluster_native_smoke_clean_${stamp}.log"
compile_log="${out_dir}/c2_cluster_native_smoke_compile_${stamp}.log"
run_log="${out_dir}/c2_cluster_native_smoke_run_${stamp}.log"
raw_json="${out_dir}/c2_cluster_native_smoke_raw_${stamp}.json"
final_json="${out_dir}/c2_cluster_native_smoke_clean_${stamp}.json"
binary_path="${out_dir}/c2_cluster_native_smoke_${stamp}"
FINAL_RC=0

mkdir -p "${out_dir}"
exec > >(tee -a "${audit_log}") 2>&1

compute_apps() {
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}

require_b300() {
    local label="$1" rows name capability
    rows="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits 2>/dev/null)" || {
        printf 'ABORT: nvidia-smi B300 identity query failed at %s.\n' "${label}" >&2
        return 74
    }
    [[ -n "${rows//[[:space:]]/}" ]] || { printf 'ABORT: no visible GPU at %s.\n' "${label}" >&2; return 74; }
    while IFS=',' read -r name capability; do
        name="${name#"${name%%[![:space:]]*}"}"
        name="${name%"${name##*[![:space:]]}"}"
        capability="${capability//[[:space:]]/}"
        [[ "${name}" == *B300* && "${capability}" == "10.3" ]] || {
            printf 'ABORT: expected B300 capability 10.3 at %s; got name=%q capability=%q.\n' \
                "${label}" "${name}" "${capability}" >&2
            return 75
        }
    done <<<"${rows}"
}

require_empty_gpu() {
    local label="$1" apps memory_used value
    apps="$(compute_apps)" || { printf 'ABORT: nvidia-smi app query failed at %s.\n' "${label}" >&2; return 74; }
    [[ -z "${apps}" ]] || { printf 'ABORT: CUDA compute app(s) visible at %s:\n%s\n' "${label}" "${apps}" >&2; return 73; }
    memory_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" || {
        printf 'ABORT: nvidia-smi memory query failed at %s.\n' "${label}" >&2
        return 74
    }
    while IFS= read -r value; do
        value="${value//[[:space:]]/}"
        [[ "${value}" =~ ^[0-9]+$ && "${value}" -eq 0 ]] || {
            printf 'ABORT: GPU memory is not exactly zero at %s: %s MiB.\n' "${label}" "${value}" >&2
            return 73
        }
    done <<<"${memory_used}"
}

snapshot() {
    local label="$1"
    printf '\n===== %s UTC %s (Slurm job %s) =====\n' "${label}" "$(date -u +%FT%TZ)" "${SLURM_JOB_ID}"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap \
        --format=csv,noheader,nounits || true
    printf '%s\n' '-- compute apps --'
    compute_apps || true
    printf '%s\n' '-- source SHA256 --'
    sha256sum "${source_path}" "${script_dir}/run_c2_cluster_native_smoke_clean.sh" || true
}

on_exit() {
    local rc=$?
    local post_rc=0
    trap - EXIT
    set +e
    snapshot POST
    require_empty_gpu POST || post_rc=$?
    if [[ "${rc}" -eq 0 && "${post_rc}" -ne 0 ]]; then
        rc="${post_rc}"
    fi
    FINAL_RC="${rc}"
    printf '\n===== FINAL_RC=%s =====\n' "${FINAL_RC}"
    exit "${FINAL_RC}"
}
trap on_exit EXIT

[[ -x "${python_bin}" ]] || { printf 'Missing selected Python: %s\n' "${python_bin}" >&2; exit 65; }
[[ -n "${nvcc_bin}" && -x "${nvcc_bin}" ]] || { printf 'Missing nvcc: %s\n' "${nvcc_bin}" >&2; exit 65; }
[[ -f "${source_path}" ]] || { printf 'Missing CUDA source: %s\n' "${source_path}" >&2; exit 65; }
nvcc_help="$("${nvcc_bin}" --help)"
if ! grep -q "sm_103a" <<<"${nvcc_help}"; then
    printf '%s\n' 'nvcc does not advertise sm_103a; refusing to compile a non-B300 binary.' >&2
    exit 65
fi
source_sha_pre="$(sha256sum "${source_path}" | awk '{print $1}')"
runner_sha_pre="$(sha256sum "${script_dir}/run_c2_cluster_native_smoke_clean.sh" | awk '{print $1}')"
nvcc_version="$("${nvcc_bin}" --version)"
compile_flags='-std=c++17 -O3 -arch=sm_103a'

snapshot PRE
require_b300 PRE
require_empty_gpu PRE
gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
[[ "${gpu_uuid}" == GPU-* ]] || { printf 'Invalid GPU UUID: %s\n' "${gpu_uuid}" >&2; exit 75; }

printf '\n===== compile native CUDA smoke =====\n'
"${nvcc_bin}" -std=c++17 -O3 -arch=sm_103a "${source_path}" -o "${binary_path}" >"${compile_log}" 2>&1
printf 'nvcc completed: %s\n' "${compile_log}"

printf '\n===== run native CUDA smoke (120 second watchdog) =====\n'
set +e
timeout --preserve-status 120s "${binary_path}" >"${raw_json}" 2>"${run_log}"
run_rc=$?
set -e
if [[ "${run_rc}" -ne 0 ]]; then
    printf 'Native CUDA smoke failed or timed out (rc=%s); raw=%s stderr=%s\n' "${run_rc}" "${raw_json}" "${run_log}" >&2
    exit "${run_rc}"
fi

source_sha_post="$(sha256sum "${source_path}" | awk '{print $1}')"
runner_sha_post="$(sha256sum "${script_dir}/run_c2_cluster_native_smoke_clean.sh" | awk '{print $1}')"
[[ "${source_sha_pre}" == "${source_sha_post}" ]] || { printf '%s\n' 'CUDA source changed during audit.' >&2; exit 66; }
[[ "${runner_sha_pre}" == "${runner_sha_post}" ]] || { printf '%s\n' 'Runner changed during audit.' >&2; exit 66; }
binary_sha="$(sha256sum "${binary_path}" | awk '{print $1}')"
triton_version="$("${python_bin}" -c 'import triton; print(triton.__version__)')"
"${python_bin}" - "${raw_json}" "${final_json}" "${source_sha_pre}" "${runner_sha_pre}" \
    "${binary_sha}" "${triton_version}" "${nvcc_version}" "${compile_flags}" \
    "${SLURM_JOB_ID}" "${gpu_uuid}" <<'PY'
import json
import math
from pathlib import Path
import sys

if sys.flags.optimize != 0:
    raise RuntimeError("secondary gate requires Python assertions; optimized mode is forbidden")

(
    raw_path,
    final_path,
    source_sha,
    runner_sha,
    binary_sha,
    triton_version,
    nvcc_version,
    compile_flags,
    slurm_job_id,
    gpu_uuid,
) = sys.argv[1:]
payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
assert payload["schema"] == "c2-cluster-native-smoke-v1"
assert payload["status"] == "pass", payload
assert payload["boundary"] == "cluster communication prerequisite only; producers are synthetic"
assert payload["num_ctas"] == 4 and payload["clusters"] >= 4
assert payload["hgroup"] == 16 and payload["head_dim"] == 128
assert payload["caller_owned_output"] is True
assert payload["global_seed_inputs"] is True
assert payload["global_inter_cta_scratch"] is False
assert payload["remote_shared_api"] == "cooperative_groups::cluster_group::map_shared_rank"
assert payload["partial_dtype"] == "bfloat16"
assert payload["sync_api"] == "cooperative_groups::cluster_group::sync"
assert payload["mbarrier_phase"].startswith("pending/not implemented: this prototype uses no mbarrier;")
assert payload["capability"] == [10, 3] and "B300" in payload["device"]
assert payload["cluster_launch_supported"] is True
assert payload["finite"] is True and payload["sentinel_clean"] is True and payload["allclose"] is True
assert len(payload["seeds"]) >= 2
for row in payload["seeds"]:
    assert row["finite"] is True and row["sentinel_clean"] is True and row["allclose"] is True
    assert math.isfinite(float(row["max_abs"])) and math.isfinite(float(row["max_rel"]))
for key in ("max_abs", "max_rel"):
    assert math.isfinite(float(payload[key]))
assert math.isclose(float(payload["tolerance"]["rtol"]), 1e-3, rel_tol=0.0, abs_tol=1e-9)
assert math.isclose(float(payload["tolerance"]["atol"]), 1e-3, rel_tol=0.0, abs_tol=1e-9)
payload["triton_version"] = triton_version
payload["build"] = {
    "nvcc_version": nvcc_version,
    "compile_flags": compile_flags,
    "binary_sha256": binary_sha,
}
payload["execution"] = {
    "slurm_job_id": int(slurm_job_id),
    "gpu_uuid": gpu_uuid,
}
payload["source_sha256"] = {
    "challenge_v2/c2_cluster_native_smoke.cu": source_sha,
    "challenge_v2/run_c2_cluster_native_smoke_clean.sh": runner_sha,
}
Path(final_path).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "secondary_gate": "pass",
    "json": final_path,
    "max_abs": payload["max_abs"],
    "max_rel": payload["max_rel"],
    "mbarrier_phase": payload["mbarrier_phase"],
}, ensure_ascii=False, sort_keys=True))
PY

require_empty_gpu AFTER_RUN
printf 'Native CUDA cluster smoke completed: %s\n' "${final_json}"
