#!/usr/bin/env bash
# Frozen FP8 B=1/4/8/16 x two-seed AB/BA policy evidence wrapper.
#
# It never allocates a GPU.  A coordinator must invoke it in one already
# isolated B300 Slurm allocation.  Only a fully validated rc=3 (the strict
# performance policy missed) is allowed to continue to another context.

set -Eeuo pipefail

if [[ "${C2_PREPARED_STAGE5_FP8_ABBA_AUTHORIZED:-}" != "1" || "${1:-}" != "--authorized-by-parent" ]]; then
    printf '%s\n' 'Refusing FP8 policy performance run without explicit coordinator authorization.' >&2
    exit 64
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Refusing FP8 policy performance run outside a Slurm allocation (SLURM_JOB_ID is unset).' >&2
    exit 64
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
c2_root="$(cd "${script_dir}/.." && pwd)"
assignment_root="$(cd "${c2_root}/../.." && pwd)"
python_bin="${C2_PYTHON_BIN:-${assignment_root}/.venv/bin/python}"
out_dir="${C2_PREPARED_STAGE5_FP8_ABBA_OUT_DIR:-${c2_root}/experiment_logs/prepared_stage5_fp8_abba}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
readonly storages=(fp8-scalar fp8-token)
readonly batches=(1 4 8 16)
readonly base_seeds=(20260828 20260829)
audit_log="${out_dir}/c2_prepared_stage5_fp8_abba_scalar_token_b1_b4_b8_b16_base_seed20260828_20260829_clean_audit_${stamp}.log"
readonly tracked_sources=(
    "${script_dir}/prepared_stage_fp8_abba_cli.py"
    "${script_dir}/run_prepared_stage5_fp8_abba_clean.sh"
    "${script_dir}/prepared_tuned.py"
    "${c2_root}/challenge/prepared_decode.py"
    "${c2_root}/harness/data.py"
    "${c2_root}/harness/reference.py"
    "${c2_root}/harness/triton_baseline.py"
    "${c2_root}/vllm_msa_ref/sparse_attn.py"
)

if [[ ! -x "${python_bin}" ]]; then
    printf 'Missing explicitly selected Python: %s\n' "${python_bin}" >&2
    exit 65
fi
command -v timeout >/dev/null || { printf '%s\n' 'Missing timeout watchdog utility.' >&2; exit 65; }
# The per-JSON validator deliberately uses Python assertions; do not permit an
# inherited optimized mode to silently erase them.
export PYTHONOPTIMIZE=0
if [[ "$("${python_bin}" -c 'print(int(__debug__))')" != "1" ]]; then
    printf '%s\n' 'Python assertions are disabled; refusing an unverifiable FP8 audit.' >&2
    exit 65
fi

mkdir -p "${out_dir}"
exec > >(tee -a "${audit_log}") 2>&1

compute_apps() {
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}

require_b300() {
    local label="$1" gpu_rows name capability
    if ! gpu_rows="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits 2>/dev/null)"; then
        printf 'ABORT: nvidia-smi B300 identity query failed at %s.\n' "${label}" >&2
        return 74
    fi
    if [[ -z "${gpu_rows//[[:space:]]/}" ]]; then
        printf 'ABORT: no allocated GPU is visible at %s.\n' "${label}" >&2
        return 74
    fi
    while IFS=',' read -r name capability; do
        name="${name#"${name%%[![:space:]]*}"}"
        name="${name%"${name##*[![:space:]]}"}"
        capability="${capability//[[:space:]]/}"
        if [[ "${name}" != *B300* || "${capability}" != "10.3" ]]; then
            printf 'ABORT: expected B300 capability 10.3 at %s; got name=%q capability=%q.\n' \
                "${label}" "${name}" "${capability}" >&2
            return 75
        fi
    done <<<"${gpu_rows}"
}

require_empty_gpu() {
    local label="$1" apps memory_used value
    if ! apps="$(compute_apps)"; then
        printf 'ABORT: nvidia-smi compute-app query failed at %s.\n' "${label}" >&2
        return 74
    fi
    if [[ -n "${apps}" ]]; then
        printf 'ABORT: CUDA compute app(s) visible at %s:\n%s\n' "${label}" "${apps}" >&2
        return 73
    fi
    if ! memory_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"; then
        printf 'ABORT: nvidia-smi memory query failed at %s.\n' "${label}" >&2
        return 74
    fi
    if [[ -z "${memory_used//[[:space:]]/}" ]]; then
        printf 'ABORT: nvidia-smi memory query returned no GPU at %s.\n' "${label}" >&2
        return 74
    fi
    while IFS= read -r value; do
        value="${value//[[:space:]]/}"
        if [[ ! "${value}" =~ ^[0-9]+$ || "${value}" -ne 0 ]]; then
            printf 'ABORT: GPU memory is not exactly zero at %s: %s MiB.\n' "${label}" "${value}" >&2
            return 73
        fi
    done <<<"${memory_used}"
}

snapshot() {
    printf '\n===== %s UTC %s (Slurm job %s) =====\n' "$1" "$(date -u +%FT%TZ)" "${SLURM_JOB_ID}"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader,nounits
    printf '%s\n' '-- compute apps --'
    compute_apps || true
    printf '%s\n' '-- source and runner SHA256 --'
    sha256sum "${tracked_sources[@]}"
}

post_snapshot_done=0
on_exit() {
    local rc=$? post_rc=0
    trap - EXIT
    set +e
    if [[ "${post_snapshot_done}" -eq 0 ]]; then
        snapshot POST_ON_EXIT
        require_b300 POST_ON_EXIT || post_rc=$?
        require_empty_gpu POST_ON_EXIT || post_rc=$?
        if [[ "${rc}" -eq 0 && "${post_rc}" -ne 0 ]]; then
            rc="${post_rc}"
        fi
    fi
    printf '\n===== FINAL_RC=%s =====\n' "${rc}"
    exit "${rc}"
}
trap on_exit EXIT

validate_json() {
    local json_path="$1" storage="$2" batch="$3" base_seed="$4"
    "${python_bin}" - "${json_path}" "${storage}" "${batch}" "${base_seed}" "${c2_root}" <<'PY'
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

path_text, storage, batch_text, base_seed_text, c2_root_text = sys.argv[1:]
path = Path(path_text)
batch = int(batch_text)
base_seed = int(base_seed_text)
c2_root = Path(c2_root_text).resolve()
with path.open(encoding="utf-8") as handle:
    payload = json.load(handle)

selected_chunks = {
    "fp8-scalar": {"1": 16, "4": 4, "8": 8, "16": 4},
    "fp8-token": {"1": 4, "4": 16, "8": 16, "16": 4},
}
assert storage in selected_chunks and batch in (1, 4, 8, 16)
chunks = selected_chunks[storage][str(batch)]
control = {
    "num_topk_chunks": chunks, "decode_num_warps": 4, "merge_num_warps": 4,
    "decode_num_stages": 3, "merge_num_stages": 3, "pdl_mode": "auto",
    "decode_maxnreg": None, "merge_maxnreg": None,
}
candidate = {**control, "decode_num_stages": 5}
assert payload["schema"] == "c2-prepared-stage5-fp8-abba-v1"
environment = payload["environment"]
assert "B300" in environment["device"], environment
assert environment["capability"] == [10, 3], environment
scope = payload["scope"]
assert scope["boundary"] == "complete prepared sparse-decode dispatch: decode plus required merge; not pure decode-only and not model/server end-to-end"
assert scope["cross_mode_comparison"] == "prohibited: each FP8 mode is evaluated only against its same-mode stage-3 control"
frozen = payload["frozen_configuration"]
assert frozen["storage"] == storage and frozen["batch"] == batch and frozen["base_seed"] == base_seed
assert frozen["selected_num_topk_chunks"] == chunks
assert frozen["selected_chunks_by_storage_and_batch"] == selected_chunks
assert frozen["control"] == control and frozen["candidate"] == candidate
assert frozen["changed_field"] == "decode_num_stages"
assert frozen["selection_rule"] == "C mapping and launch configuration were specified before this FP8 AB/BA run; not reselected from event samples"
assert frozen["strict_10_percent_policy"] == "reported after validation only; it does not select or retune a configuration"
assert {key: value for key, value in control.items() if key != "decode_num_stages"} == {
    key: value for key, value in candidate.items() if key != "decode_num_stages"
}

contract = payload["fairness_contract"]
for key in (
    "same_problem_object_per_context", "same_problem_buffers_per_context",
    "same_scale_tensor_objects_per_context", "caller_owned_independent_outputs",
    "persistent_workspace_outside_timing", "same_selected_chunks_per_context",
    "full_prepared_decode_and_merge_per_timed_call", "single_complete_dispatch_per_cuda_event",
    "AB_BA_interleaved", "raw_event_samples_recorded",
):
    assert contract[key] is True, key
assert contract["oracle"] == "independent harness.reference dense FP32 selected-page causal attention"
assert contract["tolerance"] == {"rtol": 0.03, "atol": 0.03}

expected_sources = {
    "challenge_v2/prepared_stage_fp8_abba_cli.py",
    "challenge_v2/run_prepared_stage5_fp8_abba_clean.sh",
    "challenge_v2/prepared_tuned.py",
    "challenge/prepared_decode.py",
    "harness/data.py",
    "harness/reference.py",
    "harness/triton_baseline.py",
    "vllm_msa_ref/sparse_attn.py",
}
declared_sources = payload["source_sha256"]
assert set(declared_sources) == expected_sources, set(declared_sources)
for relative, declared_sha in declared_sources.items():
    source = (c2_root / relative).resolve()
    assert source.is_relative_to(c2_root) and source.is_file(), relative
    assert hashlib.sha256(source.read_bytes()).hexdigest() == declared_sha, relative

assert len(payload["results"]) == 1
row = payload["results"][0]
assert row["status"] == "pass", row.get("error")
assert row["storage"] == storage and row["batch"] == batch
assert row["base_seed"] == base_seed and row["seed"] == base_seed + batch
problem = row["problem"]
assert problem["q_shape"] == [batch, 64, 128]
assert problem["kv_cache_shape"] == [batch * 32, 4, 128, 256]
assert str(problem["kv_cache_dtype"]).startswith("torch.float8"), problem["kv_cache_dtype"]
assert problem["max_seq_len"] == 4096 and problem["decode_query_len"] == 1 and problem["num_kv_heads"] == 4
assert row["selected_num_topk_chunks"] == chunks > 1

scale = row["scale"]
assert scale["storage_mode"] == storage
assert scale["k_scale"]["dtype"] == scale["v_scale"]["dtype"] == "torch.float32"
assert scale["k_scale"]["contiguous"] is True and scale["v_scale"]["contiguous"] is True
if storage == "fp8-scalar":
    assert scale["abi_mode"] == "scalar" and scale["kv_scale_mode"] == 1
    assert scale["dequantization"] == "FP8 cache value multiplied by its scalar K or V scale"
    for key in ("k_scale", "v_scale"):
        descriptor = scale[key]
        assert descriptor["shape"] == [] and descriptor["stride"] == [] and descriptor["numel"] == 1
        assert math.isfinite(float(descriptor["value"])) and float(descriptor["value"]) > 0
else:
    assert scale["abi_mode"] == "per_token_head" and scale["kv_scale_mode"] == 2
    assert scale["dequantization"] == "FP8 cache value multiplied by scale[kv_head, physical_page*128 + token]"
    expected_shape = [4, batch * 4096]
    expected_stride = [batch * 4096, 1]
    for key in ("k_scale", "v_scale"):
        descriptor = scale[key]
        assert descriptor["shape"] == expected_shape and descriptor["stride"] == expected_stride
        assert descriptor["numel"] == 4 * batch * 4096

dispatch = row["dispatch"]
assert dispatch == {
    "boundary": "complete prepared dispatch: decode plus required merge",
    "decode_executed": True, "merge_required": True, "merge_executed": True,
    "merge_bypassed": False, "kernels_per_timed_runner_call": 2,
}
shared = row["shared_inputs"]
assert shared["same_problem_object"] is True
assert shared["same_scale_tensor_objects"] is True
assert shared["same_problem_buffers"] is True
assert all(int(pointer) > 0 for pointer in shared["shared_input_data_ptrs"].values())
outputs = row["outputs"]
assert outputs["caller_owned_independent"] is True
assert int(outputs["control_data_ptr"]) > 0 and int(outputs["candidate_data_ptr"]) > 0
assert outputs["control_data_ptr"] != outputs["candidate_data_ptr"]
assert outputs["shape"] == [batch, 64, 128] and outputs["dtype"] == "torch.bfloat16"
assert row["control"]["config"] == control and row["candidate"]["config"] == candidate
control_metadata, candidate_metadata = row["control"]["metadata"], row["candidate"]["metadata"]
assert control_metadata["num_topk_chunks"] == candidate_metadata["num_topk_chunks"] == chunks
assert control_metadata["merge_bypassed"] is False and candidate_metadata["merge_bypassed"] is False
assert control_metadata["decode_num_stages"] == 3 and candidate_metadata["decode_num_stages"] == 5
assert control_metadata["merge_num_stages"] == candidate_metadata["merge_num_stages"] == 3
assert {key: value for key, value in control_metadata.items() if key != "decode_num_stages"} == {
    key: value for key, value in candidate_metadata.items() if key != "decode_num_stages"
}
for runner in ("control", "candidate"):
    correctness = row[runner]["correctness"]
    assert correctness["finite"] is True
    assert math.isfinite(float(correctness["max_abs"])) and math.isfinite(float(correctness["mean_abs"]))

timing = row["timing"]
assert timing["warmup_each"] == 30 and timing["abba_pairs"] == 101
assert timing["samples_per_runner"] == 202 and timing["AB_BA_interleaved"] is True

def summarize_us(values):
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    assert count > 0
    return {
        "p10_us": ordered[max(0, math.ceil(0.10 * count) - 1)],
        "median_us": float(statistics.median(ordered)),
        "p90_us": ordered[min(count - 1, math.ceil(0.90 * count) - 1)],
    }

raw = timing["raw_samples_us"]
for runner in ("prepared_stage3_control", "prepared_stage5_candidate"):
    ab = [float(value) for value in raw[runner]["AB"]]
    ba = [float(value) for value in raw[runner]["BA"]]
    assert len(ab) == len(ba) == 101
    assert all(value > 0 and math.isfinite(value) for value in [*ab, *ba])
    recomputed = {
        "all": summarize_us([*ab, *ba]),
        "when_launch_order_is_AB": summarize_us(ab),
        "when_launch_order_is_BA": summarize_us(ba),
    }
    for partition, expected_count in (("all", 202), ("when_launch_order_is_AB", 101), ("when_launch_order_is_BA", 101)):
        series = timing[runner][partition]
        assert expected_count == (len(ab) + len(ba) if partition == "all" else len(ab))
        for stat in ("p10_us", "median_us", "p90_us"):
            assert float(series[stat]) > 0 and math.isfinite(float(series[stat]))
            assert math.isclose(float(series[stat]), recomputed[partition][stat], rel_tol=0.0, abs_tol=1e-12)
        assert float(series["p10_us"]) <= float(series["median_us"]) <= float(series["p90_us"])

control_us = float(timing["prepared_stage3_control"]["all"]["median_us"])
candidate_us = float(timing["prepared_stage5_candidate"]["all"]["median_us"])
speedup = float(row["speedup_stage5_vs_stage3"])
assert control_us > 0 and candidate_us > 0
assert math.isclose(speedup, control_us / candidate_us, rel_tol=0.0, abs_tol=1e-9)
assert bool(row["strict_10_percent_target_met"]) == (speedup >= 1.10)
assert bool(payload["all_contexts_strict_10_percent"]) == bool(row["strict_10_percent_target_met"])
print(json.dumps({
    "json": str(path), "storage": storage, "batch": batch, "base_seed": base_seed,
    "row_seed": row["seed"], "selected_num_topk_chunks": chunks,
    "control_us": control_us, "candidate_us": candidate_us,
    "speedup_stage5_vs_stage3": speedup,
    "strict_10_percent_target_met": bool(row["strict_10_percent_target_met"]),
}, ensure_ascii=False, sort_keys=True))
PY
}

selected_chunks() {
    local storage="$1" batch="$2"
    case "${storage}:${batch}" in
        fp8-scalar:1) echo 16 ;;
        fp8-scalar:4) echo 4 ;;
        fp8-scalar:8) echo 8 ;;
        fp8-scalar:16) echo 4 ;;
        fp8-token:1) echo 4 ;;
        fp8-token:4|fp8-token:8) echo 16 ;;
        fp8-token:16) echo 4 ;;
        *) printf 'invalid frozen FP8 storage/batch context: %s:%s\n' "${storage}" "${batch}" >&2; return 66 ;;
    esac
}

run_one() {
    local storage="$1" batch="$2" base_seed="$3"
    local chunks
    chunks="$(selected_chunks "${storage}" "${batch}")" || return $?
    local tag="${storage}_b${batch}_base_seed${base_seed}"
    local json_path="${out_dir}/c2_prepared_stage5_fp8_abba_${tag}_full_prepared_clean_${stamp}.json"
    local stdout_path="${out_dir}/c2_prepared_stage5_fp8_abba_${tag}_full_prepared_clean_${stamp}.stdout.log"
    printf '\n===== frozen FP8 full-prepared AB/BA storage=%s B=%s base_seed=%s row_seed=%s C=%s =====\n' \
        "${storage}" "${batch}" "${base_seed}" "$((base_seed + batch))" \
        "${chunks}"
    require_empty_gpu "before storage=${storage}, B=${batch}, base_seed=${base_seed}" || return $?

    local cli_rc=0
    set +e
    (
        cd "${c2_root}"
        PYTHONPATH=. timeout --preserve-status 900s "${python_bin}" -m challenge_v2.prepared_stage_fp8_abba_cli \
            --storage "${storage}" --batch "${batch}" --seed "${base_seed}" --max-seq-len 4096 \
            --require-strict-10 --output "${json_path}"
    ) >"${stdout_path}" 2>&1
    cli_rc=$?
    set -e
    if [[ "${cli_rc}" -ne 0 && "${cli_rc}" -ne 3 ]]; then
        printf 'ABORT: FP8 ABBA CLI failed or timed out (rc=%s): %s\n' "${cli_rc}" "${stdout_path}" >&2
        return "${cli_rc}"
    fi
    # rc=3 remains a valid outcome only after every correctness and provenance
    # assertion below has accepted the emitted JSON.
    validate_json "${json_path}" "${storage}" "${batch}" "${base_seed}" || return $?
    require_empty_gpu "after storage=${storage}, B=${batch}, base_seed=${base_seed}" || return $?
    return "${cli_rc}"
}

source_manifest_pre="$(sha256sum "${tracked_sources[@]}")"
snapshot PRE
require_b300 PRE
require_empty_gpu PRE
mapfile -t gpu_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
[[ "${#gpu_uuids[@]}" -eq 1 && "${gpu_uuids[0]}" == GPU-* ]] || {
    printf 'Expected exactly one allocated GPU UUID; got %q\n' "${gpu_uuids[*]:-}" >&2
    exit 75
}
printf 'AUDIT_EXECUTION slurm_job_id=%s gpu_uuid=%s\n' "${SLURM_JOB_ID}" "${gpu_uuids[0]}"
strict_failure=0
for storage in "${storages[@]}"; do
    for base_seed in "${base_seeds[@]}"; do
        for batch in "${batches[@]}"; do
            if run_one "${storage}" "${batch}" "${base_seed}"; then
                :
            else
                rc=$?
                if [[ "${rc}" -eq 3 ]]; then
                    strict_failure=1
                    printf 'STRICT-10 GATE FAILED for storage=%s B=%s base_seed=%s; retaining validated evidence and continuing.\n' \
                        "${storage}" "${batch}" "${base_seed}" >&2
                else
                    exit "${rc}"
                fi
            fi
        done
    done
done
source_manifest_post="$(sha256sum "${tracked_sources[@]}")"
[[ "${source_manifest_pre}" == "${source_manifest_post}" ]] || {
    printf '%s\n' 'Tracked FP8 source or runner changed during the audit.' >&2
    exit 66
}
snapshot POST
require_b300 POST
require_empty_gpu POST
post_snapshot_done=1
printf '\nPrepared stage-5 FP8 B=1/4/8/16 two-seed full-dispatch AB/BA audit completed: %s\n' "${audit_log}"
if [[ "${strict_failure}" -ne 0 ]]; then
    exit 3
fi
