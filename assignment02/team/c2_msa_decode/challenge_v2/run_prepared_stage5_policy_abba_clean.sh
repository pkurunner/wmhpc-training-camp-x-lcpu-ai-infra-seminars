#!/usr/bin/env bash
# Frozen B=4/8, two-base-seed policy confirmation for prepared BF16 C=1.
#
# This wrapper does not allocate Slurm resources.  A coordinator must invoke it
# inside an already isolated B300 allocation.  It deliberately runs all four
# valid AB/BA comparisons even when an individual comparison returns rc=3
# (the predeclared performance threshold was missed).

set -Eeuo pipefail

if [[ "${C2_PREPARED_STAGE5_POLICY_ABBA_AUTHORIZED:-}" != "1" || "${1:-}" != "--authorized-by-parent" ]]; then
    printf '%s\n' 'Refusing policy performance run without explicit coordinator authorization.' >&2
    exit 64
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Refusing policy performance run outside a Slurm allocation (SLURM_JOB_ID is unset).' >&2
    exit 64
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
c2_root="$(cd "${script_dir}/.." && pwd)"
assignment_root="$(cd "${c2_root}/../.." && pwd)"
python_bin="${C2_PYTHON_BIN:-${assignment_root}/.venv/bin/python}"
out_dir="${C2_PREPARED_STAGE5_POLICY_ABBA_OUT_DIR:-${c2_root}/experiment_logs/prepared_stage5_policy_abba}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
# These are intentionally not configurable: the policy evidence is defined by
# B=4,8 and two independently declared base seeds, not by a post-hoc sweep.
readonly batches=(4 8)
readonly base_seeds=(20260828 20260829)
audit_log="${out_dir}/c2_prepared_stage5_policy_abba_b4_b8_base_seed20260828_20260829_clean_audit_${stamp}.log"

if [[ ! -x "${python_bin}" ]]; then
    printf 'Missing explicitly selected Python: %s\n' "${python_bin}" >&2
    exit 65
fi
command -v timeout >/dev/null || { printf '%s\n' 'Missing timeout watchdog utility.' >&2; exit 65; }
# The JSON validator uses assertions.  Prove that optimization cannot silently
# remove them before accepting any evidence.
export PYTHONOPTIMIZE=0
if [[ "$("${python_bin}" -c 'print(int(__debug__))')" != "1" ]]; then
    printf '%s\n' 'Python assertions are disabled; refusing an unverifiable audit.' >&2
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
    printf '%s\n' '-- source SHA256 --'
    sha256sum \
        "${script_dir}/prepared_stage_abba_cli.py" \
        "${script_dir}/prepared_tuned.py" \
        "${script_dir}/cli.py" \
        "${script_dir}/run_prepared_stage5_abba_clean.sh" \
        "${script_dir}/run_prepared_stage5_policy_abba_clean.sh" \
        "${c2_root}/challenge/prepared_decode.py" \
        "${c2_root}/harness/data.py" \
        "${c2_root}/harness/reference.py" \
        "${c2_root}/harness/triton_baseline.py" \
        "${c2_root}/vllm_msa_ref/sparse_attn.py"
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
    local json_path="$1" batch="$2" base_seed="$3"
    "${python_bin}" - "${json_path}" "${batch}" "${base_seed}" "${c2_root}" <<'PY'
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

path_text, batch_text, base_seed_text, c2_root_text = sys.argv[1:]
path = Path(path_text)
batch = int(batch_text)
base_seed = int(base_seed_text)
c2_root = Path(c2_root_text).resolve()
with path.open(encoding="utf-8") as handle:
    payload = json.load(handle)

control = {
    "num_topk_chunks": 1, "decode_num_warps": 4, "merge_num_warps": 4,
    "decode_num_stages": 3, "merge_num_stages": 3, "pdl_mode": "auto",
    "decode_maxnreg": None, "merge_maxnreg": None,
}
candidate = {**control, "decode_num_stages": 5}
assert payload["schema"] == "c2-prepared-stage5-abba-v1"
environment = payload["environment"]
assert "B300" in environment["device"], environment
assert environment["capability"] == [10, 3], environment
frozen = payload["frozen_configuration"]
assert frozen["control"] == control
assert frozen["candidate"] == candidate
assert frozen["changed_field"] == "decode_num_stages"
assert frozen["bf16_c1_only"] is True
assert frozen["selection_rule"] == "specified before this AB/BA run; not reselected from its event samples"
assert {key: value for key, value in candidate.items() if key != "decode_num_stages"} == {
    key: value for key, value in control.items() if key != "decode_num_stages"
}
assert control["decode_num_stages"] == 3 and candidate["decode_num_stages"] == 5

contract = payload["fairness_contract"]
for key in (
    "same_problem_instance_per_batch", "same_input_seed_per_batch", "caller_owned_output",
    "persistent_workspace_outside_timing", "merge_bypassed", "single_call_per_cuda_event",
    "AB_BA_interleaved", "raw_event_samples_recorded",
):
    assert contract[key] is True, key
assert contract["selected_chunks"] == 1
assert contract["oracle"] == "independent harness.reference dense FP32 selected-page causal attention"
assert contract["tolerance"] == {"rtol": 0.03, "atol": 0.03}

expected_sources = {
    "challenge_v2/prepared_stage_abba_cli.py",
    "challenge_v2/prepared_tuned.py",
    "challenge_v2/cli.py",
    "challenge_v2/run_prepared_stage5_abba_clean.sh",
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
    actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert actual_sha == declared_sha, (relative, declared_sha, actual_sha)

assert len(payload["results"]) == 1
row = payload["results"][0]
assert row["batch"] == batch and row["storage"] == "bf16"
assert row["seed"] == base_seed + batch, (row["seed"], base_seed, batch)
assert row["problem"]["max_seq_len"] == 4096
assert row["status"] == "pass", row.get("error")
assert row["control"]["config"] == control
assert row["candidate"]["config"] == candidate
for runner in ("control", "candidate"):
    assert row[runner]["metadata"]["num_topk_chunks"] == 1
    assert row[runner]["metadata"]["merge_bypassed"] is True
    correctness = row[runner]["correctness"]
    assert correctness["finite"] is True
    assert math.isfinite(float(correctness["max_abs"]))
    assert math.isfinite(float(correctness["mean_abs"]))

timing = row["timing"]
assert timing["warmup_each"] == 30
assert timing["abba_pairs"] == 101
assert timing["samples_per_runner"] == 202
assert timing["AB_BA_interleaved"] is True

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
    assert len(ab) == 101 and len(ba) == 101
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
            assert float(series[stat]) > 0 and math.isfinite(float(series[stat])), (runner, partition, stat)
            assert math.isclose(float(series[stat]), recomputed[partition][stat], rel_tol=0.0, abs_tol=1e-12), (runner, partition, stat)
    assert timing[runner]["when_launch_order_is_AB"]["p10_us"] <= timing[runner]["when_launch_order_is_AB"]["median_us"] <= timing[runner]["when_launch_order_is_AB"]["p90_us"]
    assert timing[runner]["when_launch_order_is_BA"]["p10_us"] <= timing[runner]["when_launch_order_is_BA"]["median_us"] <= timing[runner]["when_launch_order_is_BA"]["p90_us"]

control_us = float(timing["prepared_stage3_control"]["all"]["median_us"])
candidate_us = float(timing["prepared_stage5_candidate"]["all"]["median_us"])
declared_speedup = float(row["speedup_stage5_vs_stage3"])
assert control_us > 0 and candidate_us > 0
assert abs(declared_speedup - control_us / candidate_us) <= 1e-9
assert bool(row["strict_10_percent_target_met"]) == (declared_speedup >= 1.10)
assert bool(payload["all_contexts_strict_10_percent"]) == bool(row["strict_10_percent_target_met"])
print(json.dumps({
    "json": str(path), "batch": batch, "base_seed": base_seed, "row_seed": row["seed"],
    "control_us": control_us, "candidate_us": candidate_us,
    "speedup_stage5_vs_stage3": declared_speedup,
    "strict_10_percent_target_met": bool(row["strict_10_percent_target_met"]),
    "control_ab_median_us": float(timing["prepared_stage3_control"]["when_launch_order_is_AB"]["median_us"]),
    "control_ba_median_us": float(timing["prepared_stage3_control"]["when_launch_order_is_BA"]["median_us"]),
    "candidate_ab_median_us": float(timing["prepared_stage5_candidate"]["when_launch_order_is_AB"]["median_us"]),
    "candidate_ba_median_us": float(timing["prepared_stage5_candidate"]["when_launch_order_is_BA"]["median_us"]),
}, ensure_ascii=False, sort_keys=True))
PY
}

run_one() {
    local batch="$1" base_seed="$2"
    local tag="b${batch}_base_seed${base_seed}"
    local json_path="${out_dir}/c2_prepared_stage5_policy_abba_${tag}_bf16_c1_clean_${stamp}.json"
    local stdout_path="${out_dir}/c2_prepared_stage5_policy_abba_${tag}_bf16_c1_clean_${stamp}.stdout.log"
    printf '\n===== frozen prepared policy AB/BA B=%s base_seed=%s row_seed=%s =====\n' \
        "${batch}" "${base_seed}" "$((base_seed + batch))"
    require_empty_gpu "before B=${batch}, base_seed=${base_seed}" || return $?

    local cli_rc=0
    set +e
    (
        cd "${c2_root}"
        PYTHONPATH=. timeout --preserve-status 900s "${python_bin}" -m challenge_v2.prepared_stage_abba_cli \
            --batch "${batch}" --seed "${base_seed}" --max-seq-len 4096 \
            --require-strict-10 --output "${json_path}"
    ) >"${stdout_path}" 2>&1
    cli_rc=$?
    set -e
    if [[ "${cli_rc}" -ne 0 && "${cli_rc}" -ne 3 ]]; then
        printf 'ABORT: policy ABBA CLI failed before valid JSON (rc=%s): %s\n' "${cli_rc}" "${stdout_path}" >&2
        return "${cli_rc}"
    fi
    # Every rc=3 still requires complete, independent correctness and evidence
    # validation before it is allowed to count as a policy observation.
    validate_json "${json_path}" "${batch}" "${base_seed}" || return $?
    require_empty_gpu "after B=${batch}, base_seed=${base_seed}" || return $?
    return "${cli_rc}"
}

snapshot PRE
require_b300 PRE
require_empty_gpu PRE
strict_failure=0
for base_seed in "${base_seeds[@]}"; do
    for batch in "${batches[@]}"; do
        if run_one "${batch}" "${base_seed}"; then
            :
        else
            rc=$?
            if [[ "${rc}" -eq 3 ]]; then
                strict_failure=1
                printf 'STRICT-10 GATE FAILED for B=%s base_seed=%s; retaining validated evidence and continuing.\n' \
                    "${batch}" "${base_seed}" >&2
            else
                exit "${rc}"
            fi
        fi
    done
done
snapshot POST
require_b300 POST
require_empty_gpu POST
post_snapshot_done=1
printf '\nPrepared stage-5 B=4/8 two-seed policy AB/BA audit completed: %s\n' "${audit_log}"
if [[ "${strict_failure}" -ne 0 ]]; then
    exit 3
fi
