#!/usr/bin/env bash
# Minimal Nsight Compute comparison for the frozen B=1 prepared stage-3/stage-5
# kernels. This wrapper profiles exactly one matching decode launch per process
# and never allocates Slurm resources itself.
set -Eeuo pipefail

if [[ "${C2_PREPARED_STAGE5_NCU_AUTHORIZED:-}" != "1" || "${1:-}" != "--authorized-by-parent" ]]; then
    printf '%s\n' 'Refusing NCU run without explicit coordinator authorization.' >&2
    exit 64
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Refusing NCU run outside a Slurm allocation.' >&2
    exit 64
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
c2_root="$(cd "${script_dir}/.." && pwd)"
assignment_root="$(cd "${c2_root}/../.." && pwd)"
python_bin="${C2_PYTHON_BIN:-${assignment_root}/.venv/bin/python}"
ncu_bin="${NCU_BIN:-ncu}"
out_dir="${C2_PREPARED_STAGE5_NCU_OUT_DIR:-${c2_root}/experiment_logs/prepared_stage5_ncu}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
audit_log="${out_dir}/c2_prepared_stage5_ncu_b1_audit_${stamp}.log"
stage3_report="${out_dir}/c2_prepared_stage3_b1_${stamp}.ncu-rep"
stage5_report="${out_dir}/c2_prepared_stage5_b1_${stamp}.ncu-rep"
stage3_csv="${out_dir}/c2_prepared_stage3_b1_${stamp}.csv"
stage5_csv="${out_dir}/c2_prepared_stage5_b1_${stamp}.csv"
stage3_driver_json="${out_dir}/c2_prepared_stage5_ncu_driver_stage3_b1_${stamp}.json"
stage5_driver_json="${out_dir}/c2_prepared_stage5_ncu_driver_stage5_b1_${stamp}.json"
evidence_manifest="${out_dir}/c2_prepared_stage5_ncu_b1_${stamp}.sha256"

[[ -x "${python_bin}" ]] || { printf 'Missing Python: %s\n' "${python_bin}" >&2; exit 65; }
command -v "${ncu_bin}" >/dev/null || { printf 'Missing NCU: %s\n' "${ncu_bin}" >&2; exit 65; }
command -v timeout >/dev/null || { printf '%s\n' 'Missing timeout(1).' >&2; exit 65; }
export PYTHONOPTIMIZE=0
[[ "$("${python_bin}" -c 'print(int(__debug__))')" == "1" ]] || {
    printf '%s\n' 'Python assertions are disabled.' >&2
    exit 65
}

mkdir -p "${out_dir}"
exec > >(tee -a "${audit_log}") 2>&1

compute_apps() {
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}

require_b300() {
    local label="$1" gpu_rows name capability
    gpu_rows="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits 2>/dev/null)" || return 74
    [[ -n "${gpu_rows//[[:space:]]/}" ]] || return 74
    while IFS=',' read -r name capability; do
        name="${name#"${name%%[![:space:]]*}"}"
        name="${name%"${name##*[![:space:]]}"}"
        capability="${capability//[[:space:]]/}"
        if [[ "${name}" != *B300* || "${capability}" != "10.3" ]]; then
            printf 'ABORT: expected B300 capability 10.3 at %s; got %q / %q.\n' \
                "${label}" "${name}" "${capability}" >&2
            return 75
        fi
    done <<<"${gpu_rows}"
}

require_empty_gpu() {
    local label="$1" apps memory_used value
    apps="$(compute_apps)" || return 74
    if [[ -n "${apps}" ]]; then
        printf 'ABORT: compute app(s) visible at %s:\n%s\n' "${label}" "${apps}" >&2
        return 73
    fi
    memory_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" || return 74
    [[ -n "${memory_used//[[:space:]]/}" ]] || return 74
    while IFS= read -r value; do
        value="${value//[[:space:]]/}"
        if [[ ! "${value}" =~ ^[0-9]+$ || "${value}" -ne 0 ]]; then
            printf 'ABORT: GPU memory is not zero at %s: %s MiB.\n' "${label}" "${value}" >&2
            return 73
        fi
    done <<<"${memory_used}"
}

source_snapshot() {
    sha256sum \
        "${script_dir}/run_prepared_stage5_ncu_b1_clean.sh" \
        "${script_dir}/prepared_stage_abba_cli.py" \
        "${script_dir}/prepared_tuned.py" \
        "${script_dir}/cli.py" \
        "${c2_root}/challenge/prepared_decode.py" \
        "${c2_root}/harness/data.py" \
        "${c2_root}/harness/reference.py" \
        "${c2_root}/harness/triton_baseline.py" \
        "${c2_root}/vllm_msa_ref/sparse_attn.py"
}

snapshot() {
    printf '\n===== %s UTC %s (Slurm job %s) =====\n' "$1" "$(date -u +%FT%TZ)" "${SLURM_JOB_ID}"
    nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader,nounits || true
    printf '%s\n' '-- compute apps --'
    compute_apps || true
    printf '%s\n' '-- source SHA256 --'
    source_snapshot
}

finish() {
    local rc=$? post_rc=0 check_rc=0
    trap - EXIT
    snapshot POST || { check_rc=$?; post_rc="${check_rc}"; }
    require_b300 POST || { check_rc=$?; post_rc="${check_rc}"; }
    require_empty_gpu POST || {
        check_rc=$?
        [[ "${post_rc}" -ne 0 ]] || post_rc="${check_rc}"
    }
    if [[ "${rc}" -eq 0 && "${post_rc}" -ne 0 ]]; then rc="${post_rc}"; fi
    printf 'STAGE3_REPORT=%s\nSTAGE5_REPORT=%s\nSTAGE3_CSV=%s\nSTAGE5_CSV=%s\nEVIDENCE_MANIFEST=%s\nFINAL_RC=%s\n' \
        "${stage3_report}" "${stage5_report}" "${stage3_csv}" "${stage5_csv}" \
        "${evidence_manifest}" "${rc}"
    exit "${rc}"
}
trap finish EXIT

validate_profile_json() {
    local json_path="$1" label="$2" launch_skip="$3"
    "${python_bin}" - "${json_path}" "${c2_root}" "${label}" "${launch_skip}" <<'PY'
import hashlib, json, math, sys
from pathlib import Path

path, root_text, label, launch_skip_text = sys.argv[1:]
root = Path(root_text).resolve()
payload = json.load(open(path, encoding="utf-8"))
assert payload["schema"] == "c2-prepared-stage5-abba-v1"
assert "B300" in payload["environment"]["device"]
assert payload["environment"]["capability"] == [10, 3]
profiling = payload["profiling_context"]
assert profiling == {
    "tool": "NVIDIA Nsight Compute",
    "profiled_role": label,
    "kernel_regex": ".*_gqa_sparse_decode_kernel.*",
    "matching_launch_skip": int(launch_skip_text),
    "matching_launch_count": 1,
    "timing_valid_for_benchmark": False,
    "timing_note": "NCU replay/instrumentation invalidates all timing fields in this driver JSON",
}
assert len(payload["results"]) == 1
row = payload["results"][0]
assert row["batch"] == 1 and row["storage"] == "bf16" and row["status"] == "pass"
assert row["control"]["config"]["decode_num_stages"] == 3
assert row["candidate"]["config"]["decode_num_stages"] == 5
assert row["control"]["metadata"]["merge_bypassed"] is True
assert row["candidate"]["metadata"]["merge_bypassed"] is True
for side in ("control", "candidate"):
    correctness = row[side]["correctness"]
    assert correctness["finite"] is True
    assert math.isfinite(float(correctness["max_abs"]))
for relative, declared in payload["source_sha256"].items():
    source = (root / relative).resolve()
    assert source.is_relative_to(root) and source.is_file(), relative
    assert hashlib.sha256(source.read_bytes()).hexdigest() == declared, relative
print(json.dumps({"batch": 1, "control_max_abs": row["control"]["correctness"]["max_abs"],
                  "candidate_max_abs": row["candidate"]["correctness"]["max_abs"]}, sort_keys=True))
PY
}

annotate_profile_json() {
    local json_path="$1" label="$2" launch_skip="$3"
    "${python_bin}" - "${json_path}" "${label}" "${launch_skip}" <<'PY'
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1])
label = sys.argv[2]
launch_skip = int(sys.argv[3])
payload = json.loads(path.read_text(encoding="utf-8"))
assert "profiling_context" not in payload
payload["profiling_context"] = {
    "tool": "NVIDIA Nsight Compute",
    "profiled_role": label,
    "kernel_regex": ".*_gqa_sparse_decode_kernel.*",
    "matching_launch_skip": launch_skip,
    "matching_launch_count": 1,
    "timing_valid_for_benchmark": False,
    "timing_note": "NCU replay/instrumentation invalidates all timing fields in this driver JSON",
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

profile_one() {
    local label="$1" launch_skip="$2" report="$3" csv="$4" json_path="$5" bundle_hash
    printf '\n===== NCU %s (matching launch skip=%s) =====\n' "${label}" "${launch_skip}"
    require_empty_gpu "before ${label}"
    (
        cd "${c2_root}"
        PYTHONPATH=. timeout --signal=TERM --kill-after=30s 360s "${ncu_bin}" \
            --target-processes all \
            --kernel-name-base function \
            --kernel-name 'regex:.*_gqa_sparse_decode_kernel.*' \
            --launch-skip "${launch_skip}" \
            --launch-count 1 \
            --clock-control none \
            --section SpeedOfLight \
            --section ComputeWorkloadAnalysis \
            --section MemoryWorkloadAnalysis \
            --section Occupancy \
            --section SchedulerStats \
            --force-overwrite \
            --export "${report}" \
            "${python_bin}" -m challenge_v2.prepared_stage_abba_cli \
                --batch 1 --seed 20260828 --max-seq-len 4096 --output "${json_path}"
    )
    [[ -s "${report}" && -s "${json_path}" ]] || {
        printf 'ABORT: missing NCU report or driver JSON for %s.\n' "${label}" >&2
        return 76
    }
    "${ncu_bin}" --import "${report}" --csv --page details > "${csv}"
    [[ -s "${csv}" ]] || { printf 'ABORT: empty NCU CSV for %s.\n' "${label}" >&2; return 76; }
    grep -q '_gqa_sparse_decode_kernel' "${csv}" || {
        printf 'ABORT: NCU report for %s did not capture the decode kernel.\n' "${label}" >&2
        return 76
    }
    annotate_profile_json "${json_path}" "${label}" "${launch_skip}"
    validate_profile_json "${json_path}" "${label}" "${launch_skip}"
    bundle_hash="${report%.ncu-rep}.bundle.sha256"
    sha256sum "${report}" "${csv}" "${json_path}" | tee "${bundle_hash}"
    require_empty_gpu "after ${label}"
}

snapshot PRE
require_b300 PRE
require_empty_gpu PRE
printf 'NCU_VERSION=%s\n' "$("${ncu_bin}" --version | tail -1)"
printf '%s\n' '-- selected NCU sections --'
"${ncu_bin}" --list-sections | grep -E '^(SpeedOfLight|ComputeWorkloadAnalysis|MemoryWorkloadAnalysis|Occupancy|SchedulerStats)[[:space:]]'

# The stage-3 correctness call is the first matching decode launch. The
# stage-5 correctness call is the second; both precede ABBA timing.
profile_one stage3 0 "${stage3_report}" "${stage3_csv}" "${stage3_driver_json}"
profile_one stage5 1 "${stage5_report}" "${stage5_csv}" "${stage5_driver_json}"

sha256sum \
    "${stage3_report}" "${stage3_csv}" "${stage3_driver_json}" \
    "${stage5_report}" "${stage5_csv}" "${stage5_driver_json}" > "${evidence_manifest}"
printf '%s\n' '-- final NCU evidence SHA256 --'
cat "${evidence_manifest}"

printf '\nNCU B=1 stage-3/stage-5 audit completed.\n'
