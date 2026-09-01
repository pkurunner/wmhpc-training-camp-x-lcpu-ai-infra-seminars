#!/usr/bin/env bash
# Authorized clean runner for dynamic missing-arrival mbarrier fault injection.
# It never submits work to Slurm; invoke only inside a private B300 allocation.

set -Eeuo pipefail

if [[ "${C2_CLUSTER_ATTENTION_MBARRIER_FAULT_AUTHORIZED:-}" != "1" || "${1:-}" != "--authorized-by-parent" ]]; then
    printf '%s\n' 'Refusing mbarrier fault injection without explicit coordinator authorization.' >&2
    exit 64
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Refusing mbarrier fault injection outside a Slurm allocation (SLURM_JOB_ID is unset).' >&2
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
source_path="${script_dir}/c2_cluster_attention_mbarrier_fault_smoke.cu"
runner_path="${script_dir}/run_c2_cluster_attention_mbarrier_fault_smoke_clean.sh"
out_dir="${C2_CLUSTER_ATTENTION_MBARRIER_FAULT_OUT_DIR:-${c2_root}/experiment_logs/c2_cluster_attention_mbarrier_fault_smoke}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
audit_log="${out_dir}/c2_cluster_attention_mbarrier_fault_smoke_clean_${stamp}.log"
compile_log="${out_dir}/c2_cluster_attention_mbarrier_fault_smoke_compile_${stamp}.log"
run_log="${out_dir}/c2_cluster_attention_mbarrier_fault_smoke_run_${stamp}.log"
raw_json="${out_dir}/c2_cluster_attention_mbarrier_fault_smoke_raw_${stamp}.json"
final_json="${out_dir}/c2_cluster_attention_mbarrier_fault_smoke_clean_${stamp}.json"
binary_path="${out_dir}/c2_cluster_attention_mbarrier_fault_smoke_${stamp}"
ptx_path="${out_dir}/c2_cluster_attention_mbarrier_fault_smoke_${stamp}.ptx"
sass_path="${out_dir}/c2_cluster_attention_mbarrier_fault_smoke_${stamp}.sass"

mkdir -p "${out_dir}"
exec > >(tee -a "${audit_log}") 2>&1

compute_apps() {
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}

require_b300() {
    local label="$1" rows name capability
    rows="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits 2>/dev/null)" || {
        printf 'ABORT: nvidia-smi B300 identity query failed at %s.\n' "${label}" >&2; return 74;
    }
    [[ -n "${rows//[[:space:]]/}" ]] || { printf 'ABORT: no visible GPU at %s.\n' "${label}" >&2; return 74; }
    while IFS=',' read -r name capability; do
        name="${name#"${name%%[![:space:]]*}"}"
        name="${name%"${name##*[![:space:]]}"}"
        capability="${capability//[[:space:]]/}"
        [[ "${name}" == *B300* && "${capability}" == "10.3" ]] || {
            printf 'ABORT: expected B300 capability 10.3 at %s; got name=%q capability=%q.\n' \
                "${label}" "${name}" "${capability}" >&2; return 75;
        }
    done <<<"${rows}"
}

require_empty_gpu() {
    local label="$1" apps memory_used value
    apps="$(compute_apps)" || { printf 'ABORT: nvidia-smi app query failed at %s.\n' "${label}" >&2; return 74; }
    [[ -z "${apps}" ]] || { printf 'ABORT: CUDA compute app(s) visible at %s:\n%s\n' "${label}" "${apps}" >&2; return 73; }
    memory_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" || {
        printf 'ABORT: nvidia-smi memory query failed at %s.\n' "${label}" >&2; return 74;
    }
    while IFS= read -r value; do
        value="${value//[[:space:]]/}"
        [[ "${value}" =~ ^[0-9]+$ && "${value}" -eq 0 ]] || {
            printf 'ABORT: GPU memory is not exactly zero at %s: %s MiB.\n' "${label}" "${value}" >&2; return 73;
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
    sha256sum "${source_path}" "${runner_path}" || true
}

on_exit() {
    local rc=$? post_rc=0
    trap - EXIT
    set +e
    snapshot POST
    require_empty_gpu POST || post_rc=$?
    if [[ "${rc}" -eq 0 && "${post_rc}" -ne 0 ]]; then rc="${post_rc}"; fi
    printf '\n===== FINAL_RC=%s =====\n' "${rc}"
    exit "${rc}"
}
trap on_exit EXIT

[[ -x "${python_bin}" ]] || { printf 'Missing selected Python: %s\n' "${python_bin}" >&2; exit 65; }
[[ -n "${nvcc_bin}" && -x "${nvcc_bin}" ]] || { printf 'Missing nvcc: %s\n' "${nvcc_bin}" >&2; exit 65; }
cuobjdump_bin="$(dirname "${nvcc_bin}")/cuobjdump"
[[ -x "${cuobjdump_bin}" ]] || { printf 'Missing cuobjdump paired with nvcc: %s\n' "${cuobjdump_bin}" >&2; exit 65; }
[[ -f "${source_path}" ]] || { printf 'Missing CUDA source: %s\n' "${source_path}" >&2; exit 65; }
command -v timeout >/dev/null || { printf '%s\n' 'Missing timeout watchdog utility.' >&2; exit 65; }
export PYTHONOPTIMIZE=0
[[ "$("${python_bin}" -c 'print(int(__debug__))')" == "1" ]] || {
    printf '%s\n' 'Python assertions are disabled; refusing an unverifiable fault test.' >&2; exit 65;
}
nvcc_help="$("${nvcc_bin}" --help)"
grep -q 'sm_103a' <<<"${nvcc_help}" || { printf '%s\n' 'nvcc does not advertise sm_103a.' >&2; exit 65; }

source_sha_pre="$(sha256sum "${source_path}" | awk '{print $1}')"
runner_sha_pre="$(sha256sum "${runner_path}" | awk '{print $1}')"
nvcc_version="$("${nvcc_bin}" --version)"
compile_flags='-std=c++17 -O3 -arch=sm_103a'

snapshot PRE
require_b300 PRE
require_empty_gpu PRE
mapfile -t gpu_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
[[ "${#gpu_uuids[@]}" -eq 1 && "${gpu_uuids[0]}" == GPU-* ]] || {
    printf 'Expected exactly one allocated GPU UUID; got %q\n' "${gpu_uuids[*]:-}" >&2; exit 75;
}
gpu_uuid="${gpu_uuids[0]}"

printf '\n===== compile dynamic C=2 missing-arrival mbarrier fault path =====\n'
"${nvcc_bin}" -std=c++17 -O3 -arch=sm_103a "${source_path}" -o "${binary_path}" >"${compile_log}" 2>&1
"${nvcc_bin}" -std=c++17 -O3 -arch=sm_103a -ptx "${source_path}" -o "${ptx_path}" >>"${compile_log}" 2>&1
grep -Fq 'mbarrier.arrive.release.cluster.shared::cluster.b64' "${ptx_path}" || {
    printf '%s\n' 'PTX evidence is missing remote DSM mbarrier release-arrive.' >&2; exit 67;
}
grep -Fq 'mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64' "${ptx_path}" || {
    printf '%s\n' 'PTX evidence is missing bounded local acquire cluster mbarrier parity wait.' >&2; exit 67;
}
"${cuobjdump_bin}" --dump-sass "${binary_path}" >"${sass_path}"
grep -Eq 'CGAERRBAR|UCGABAR' "${sass_path}" || {
    printf '%s\n' 'SASS evidence is missing the convergent cluster-barrier lifecycle instruction.' >&2; exit 67;
}
grep -Fq 'SYNCS.ARRIVE.TRANS64.RED.A1T0' "${sass_path}" || {
    printf '%s\n' 'SASS evidence is missing executable remote mbarrier arrival.' >&2; exit 67;
}
grep -Fq 'SYNCS.PHASECHK.TRANS64.TRYWAIT' "${sass_path}" || {
    printf '%s\n' 'SASS evidence is missing executable mbarrier bounded phase wait.' >&2; exit 67;
}

printf '\n===== run dynamic missing-arrival fault path (45 second watchdog only) =====\n'
set +e
timeout --preserve-status --kill-after=5s 45s "${binary_path}" >"${raw_json}" 2>"${run_log}"
run_rc=$?
set -e
if [[ "${run_rc}" -ne 0 ]]; then
    printf 'Fault path failed, hit CUDA error, or watchdog expired (rc=%s); raw=%s stderr=%s\n' \
        "${run_rc}" "${raw_json}" "${run_log}" >&2
    exit "${run_rc}"
fi

source_sha_post="$(sha256sum "${source_path}" | awk '{print $1}')"
runner_sha_post="$(sha256sum "${runner_path}" | awk '{print $1}')"
[[ "${source_sha_pre}" == "${source_sha_post}" ]] || { printf '%s\n' 'CUDA source changed during audit.' >&2; exit 66; }
[[ "${runner_sha_pre}" == "${runner_sha_post}" ]] || { printf '%s\n' 'Runner changed during audit.' >&2; exit 66; }
binary_sha="$(sha256sum "${binary_path}" | awk '{print $1}')"
ptx_sha="$(sha256sum "${ptx_path}" | awk '{print $1}')"
sass_sha="$(sha256sum "${sass_path}" | awk '{print $1}')"

"${python_bin}" - "${raw_json}" "${final_json}" "${source_sha_pre}" "${runner_sha_pre}" \
    "${binary_sha}" "${ptx_sha}" "${sass_sha}" "${nvcc_version}" "${compile_flags}" \
    "${SLURM_JOB_ID}" "${gpu_uuid}" <<'PY'
import json
import math
from pathlib import Path
import sys

if sys.flags.optimize != 0:
    raise RuntimeError("secondary gate requires Python assertions; optimized mode is forbidden")

(
    raw_path, final_path, source_sha, runner_sha, binary_sha, ptx_sha, sass_sha,
    nvcc_version, compile_flags, slurm_job_id, gpu_uuid,
) = sys.argv[1:]
payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
assert payload["schema"] == "c2-cluster-attention-mbarrier-fault-smoke-v1"
assert payload["status"] == "pass", payload
assert payload["boundary"] == (
    "dynamic missing-arrival fault injection only; not an attention correctness or performance candidate"
)
fault = payload["fault_injection"]
assert fault == {
    "omitted_role": 1, "arrival_role": 0, "consumer_role": 2,
    "expected_arrivals": 2, "actual_remote_arrivals": 1,
    "wait_parity": 0, "max_polls": 1 << 20,
}
assert payload["mbarrier_phase"] == (
    "rank-1 deliberately omits its required remote DSM release-arrive; rank-2 bounded parity-0 acquire wait must expire"
)
assert payload["producer_ready_sync"] == (
    "rank-0 only: cuda::ptx::mbarrier_arrive(sem_release, scope_cluster, space_cluster, remote DSM); "
    "rank-1 intentionally does not arrive"
)
assert payload["wait_sync"] == (
    "rank-2: mbarrier_try_wait_parity(sem_acquire, scope_cluster, local shared, parity=0), bounded"
)
assert payload["init_sync"] == (
    "cooperative_groups::cluster_group::sync: rank-2 mbarrier initialization and cluster residency only"
)
assert payload["lifetime_sync"] == (
    "cooperative_groups::cluster_group::sync: every CTA reaches a final lifetime barrier after fault handling"
)
assert payload["sync_api"] == "cooperative_groups::cluster_group::sync (init + final lifetime only)"
assert payload["remote_shared_api"] == "cooperative_groups::cluster_group::map_shared_rank"
assert payload["num_ctas"] == 4 and payload["clusters"] >= 4 and payload["threads_per_block"] == 256
assert payload["output_elements_per_cluster"] == 16 * 128
assert payload["capability"] == [10, 3] and "B300" in payload["device"]
assert payload["cluster_launch_supported"] is True
assert payload["resource_model"]["static_shared_bytes"] > 0
assert payload["resource_model"]["static_shared_fits"] is True
assert payload["event_timing_scope"] == "single fault kernel liveness bound only; not a performance metric"
assert math.isclose(float(payload["fault_sentinel"]), -12352.0, rel_tol=0.0, abs_tol=0.0)
assert math.isclose(float(payload["fault_kernel_upper_bound_ms"]), 5000.0, rel_tol=0.0, abs_tol=0.0)
expected_seeds = {17, 2026}
assert {int(row["seed"]) for row in payload["seeds"]} == expected_seeds
for row in payload["seeds"]:
    seed = int(row["seed"])
    expected_status = 0x4D420100 + (seed & 0xff)
    assert int(row["expected_status"]) == expected_status
    assert row["wait_not_ready"] is True
    assert row["sentinel_complete"] is True
    assert row["kernel_within_bound"] is True
    elapsed_ms = float(row["fault_kernel_elapsed_ms"])
    assert math.isfinite(elapsed_ms) and 0.0 <= elapsed_ms < 5000.0
    assert len(row["clusters"]) == int(payload["clusters"])
    assert {int(cluster["cluster"]) for cluster in row["clusters"]} == set(range(int(payload["clusters"])))
    for cluster in row["clusters"]:
        assert int(cluster["fault_status"]) == expected_status
        assert cluster["status_expected"] is True

payload["build"] = {
    "nvcc_version": nvcc_version,
    "compile_flags": compile_flags,
    "binary_sha256": binary_sha,
    "ptx_sha256": ptx_sha,
    "sass_sha256": sass_sha,
    "mbarrier_ptx_evidence": {
        "remote_release_arrive": "mbarrier.arrive.release.cluster.shared::cluster.b64",
        "local_acquire_wait": "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64",
    },
    "mbarrier_sass_evidence": {
        "remote_release_arrive": "SYNCS.ARRIVE.TRANS64.RED.A1T0",
        "local_acquire_wait": "SYNCS.PHASECHK.TRANS64.TRYWAIT",
    },
}
payload["execution"] = {"slurm_job_id": slurm_job_id, "gpu_uuid": gpu_uuid}
payload["source_sha256"] = {
    "challenge_v2/c2_cluster_attention_mbarrier_fault_smoke.cu": source_sha,
    "challenge_v2/run_c2_cluster_attention_mbarrier_fault_smoke_clean.sh": runner_sha,
}
Path(final_path).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "secondary_gate": "pass", "json": final_path,
    "fault_status": "all clusters expired", "watchdog_used_as_control_flow": False,
}, ensure_ascii=False, sort_keys=True))
PY

require_empty_gpu AFTER_RUN
printf 'Dynamic C=2 mbarrier missing-arrival fault path completed: %s\n' "${final_json}"
