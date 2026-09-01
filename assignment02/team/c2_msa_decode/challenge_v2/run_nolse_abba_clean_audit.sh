#!/usr/bin/env bash
# Explicitly authorized, fixed-configuration AB/BA audit for C2 BF16 C=1.
# It does not tune: every batch defaults to the frozen S=1/warp4/stage3/PDL-off
# configuration; explicit environment overrides are recorded and validated.
# It must run inside a pre-approved exclusive Slurm
# allocation; this wrapper only audits that allocation and never allocates one.

set -Eeuo pipefail

if [[ "${C2_CLEAN_AUDIT_AUTHORIZED:-}" != "1" || "${1:-}" != "--authorized-by-parent" ]]; then
    printf '%s\n' 'Refusing performance run without explicit coordinator authorization.' >&2
    exit 64
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
c2_root="$(cd "${script_dir}/.." && pwd)"
assignment_root="$(cd "${c2_root}/../.." && pwd)"
python_bin="${C2_PYTHON_BIN:-${assignment_root}/.venv/bin/python}"
out_dir="${c2_root}/experiment_logs/optimization_v2"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
audit_log="${out_dir}/c2_nolse_abba_clean_audit_${stamp}.log"
audit_batches="${C2_ABBA_AUDIT_BATCHES:-1,4,8,16}"
candidate_shards="${C2_ABBA_SHARDS:-1}"
candidate_warps="${C2_ABBA_WARPS:-4}"
candidate_stages="${C2_ABBA_STAGES:-3}"
candidate_pdl="${C2_ABBA_PDL:-off}"
candidate_maxnreg="${C2_ABBA_MAXNREG:-none}"

if [[ ! -x "${python_bin}" ]]; then
    printf 'Missing explicitly selected Python: %s\n' "${python_bin}" >&2
    exit 65
fi
case "${audit_batches}" in
    1|4|8|16|1,4,8,16) ;;
    *) printf 'C2_ABBA_AUDIT_BATCHES must be 1,4,8,16 (or one member); got %q\n' "${audit_batches}" >&2; exit 66 ;;
esac
case "${candidate_shards}" in 1|2|4) ;; *) printf 'Invalid C2_ABBA_SHARDS=%q\n' "${candidate_shards}" >&2; exit 66;; esac
case "${candidate_warps}" in 1|2|4|8) ;; *) printf 'Invalid C2_ABBA_WARPS=%q\n' "${candidate_warps}" >&2; exit 66;; esac
case "${candidate_stages}" in 1|2|3|4|5|6) ;; *) printf 'Invalid C2_ABBA_STAGES=%q\n' "${candidate_stages}" >&2; exit 66;; esac
case "${candidate_pdl}" in auto|on|off) ;; *) printf 'Invalid C2_ABBA_PDL=%q\n' "${candidate_pdl}" >&2; exit 66;; esac
case "${candidate_maxnreg}" in none|64|96|128|160) ;; *) printf 'Invalid C2_ABBA_MAXNREG=%q\n' "${candidate_maxnreg}" >&2; exit 66;; esac

mkdir -p "${out_dir}"
exec > >(tee -a "${audit_log}") 2>&1

compute_apps() {
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}
require_empty_compute_apps() {
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
    while IFS= read -r value; do
        value="${value//[[:space:]]/}"
        if [[ ! "${value}" =~ ^[0-9]+$ || "${value}" -ne 0 ]]; then
            printf 'ABORT: GPU memory is not exactly zero at %s: %s MiB.\n' "${label}" "${value}" >&2
            return 73
        fi
    done <<<"${memory_used}"
    if [[ -z "${memory_used//[[:space:]]/}" ]]; then
        printf 'ABORT: nvidia-smi memory query returned no GPU at %s.\n' "${label}" >&2
        return 74
    fi
}
snapshot() {
    printf '\n===== %s UTC %s =====\n' "$1" "$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader,nounits
    printf '%s\n' '-- compute apps --'
    compute_apps || true
    printf '%s\n' '-- source SHA256 --'
    sha256sum \
        "${script_dir}/c1_no_lse.py" \
        "${script_dir}/c1_no_lse_abba_cli.py" \
        "${script_dir}/c1_no_lse_cli.py" \
        "${script_dir}/cli.py" \
        "${script_dir}/prepared_tuned.py" \
        "${c2_root}/challenge/prepared_decode.py" \
        "${c2_root}/harness/data.py" \
        "${c2_root}/harness/reference.py" \
        "${script_dir}/run_nolse_abba_clean_audit.sh"
}

validate_json() {
    local json_path="$1" batch="$2"
    "${python_bin}" - "${json_path}" "${batch}" "${candidate_shards}" "${candidate_warps}" "${candidate_stages}" "${candidate_pdl}" "${candidate_maxnreg}" "${c2_root}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

path, batch_text, shards_text, warps_text, stages_text, pdl, maxnreg_text, c2_root_text = sys.argv[1:]
batch = int(batch_text)
maxnreg = None if maxnreg_text == "none" else int(maxnreg_text)
c2_root = Path(c2_root_text).resolve()
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload["schema"] == "c2-c1-online-softmax-no-lse-abba-v1"
frozen = payload["frozen_configuration"]
assert frozen["bf16_c1_only"] is True
assert frozen["candidate"] == {"gqa_shards": int(shards_text), "num_warps": int(warps_text),
                               "num_stages": int(stages_text), "pdl_mode": pdl, "maxnreg": maxnreg}
assert frozen["control"] == {"num_topk_chunks": 1, "decode_num_warps": 4, "merge_num_warps": 4,
                             "decode_num_stages": 3, "merge_num_stages": 3, "pdl_mode": "auto",
                             "decode_maxnreg": None, "merge_maxnreg": None}
contract = payload["fairness_contract"]
assert contract["selected_chunks"] == 1
for key in ("same_input_seed_per_batch", "caller_owned_output", "persistent_workspace_outside_timing",
            "no_merge", "single_call_per_cuda_event", "AB_BA_interleaved"):
    assert contract[key] is True, key
source_sha256 = payload["source_sha256"]
for required in ("challenge_v2/c1_no_lse.py", "challenge_v2/c1_no_lse_abba_cli.py",
                 "challenge_v2/prepared_tuned.py", "challenge/prepared_decode.py", "harness/reference.py"):
    assert required in source_sha256, required
for relative, declared_sha in source_sha256.items():
    source = (c2_root / relative).resolve()
    assert source.is_relative_to(c2_root), relative
    assert source.is_file(), relative
    actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert actual_sha == declared_sha, (relative, declared_sha, actual_sha)
assert len(payload["results"]) == 1
row = payload["results"][0]
assert row["batch"] == batch and row["storage"] == "bf16"
assert row["status"] == "pass", row.get("error")
assert row["control"]["correctness"]["finite"] is True
assert row["candidate"]["correctness"]["finite"] is True
timing = row["timing"]
# v1.0 records the invariant in the fairness contract; newer output duplicates
# it in timing.  Accept either representation while still requiring it.
assert timing.get("AB_BA_interleaved", contract["AB_BA_interleaved"]) is True
assert timing["samples_per_runner"] >= 202
assert timing["abba_pairs"] >= 101
control_us = float(timing["current_prepared_control"]["all"]["median_us"])
candidate_us = float(timing["c1_online_softmax_no_lse"]["all"]["median_us"])
declared = float(row["speedup_vs_current_prepared_control"])
assert control_us > 0 and candidate_us > 0 and math.isfinite(control_us) and math.isfinite(candidate_us)
assert abs(declared - control_us / candidate_us) <= 1e-9
assert bool(row["strict_10_percent_target_met"]) == (declared >= 1.10)
assert bool(payload["all_contexts_strict_10_percent"]) == bool(row["strict_10_percent_target_met"])
print(json.dumps({"json": path, "batch": batch, "control_us": control_us, "candidate_us": candidate_us,
                  "speedup": declared, "strict_10_percent_target_met": bool(row["strict_10_percent_target_met"]),
                  "control_ab_median_us": float(timing["current_prepared_control"]["when_launch_order_is_AB"]["median_us"]),
                  "control_ba_median_us": float(timing["current_prepared_control"]["when_launch_order_is_BA"]["median_us"]),
                  "candidate_ab_median_us": float(timing["c1_online_softmax_no_lse"]["when_launch_order_is_AB"]["median_us"]),
                  "candidate_ba_median_us": float(timing["c1_online_softmax_no_lse"]["when_launch_order_is_BA"]["median_us"])},
                 ensure_ascii=False, sort_keys=True))
PY
}

run_batch() {
    local batch="$1"
    local json_path="${out_dir}/c2_nolse_abba_b${batch}_bf16_c1_clean_${stamp}.json"
    local stdout_path="${out_dir}/c2_nolse_abba_b${batch}_bf16_c1_clean_${stamp}.stdout.log"
    printf '\n===== fixed AB/BA batch B=%s =====\n' "${batch}"
    local cli_rc=0
    set +e
    (
        cd "${c2_root}"
        PYTHONPATH=. "${python_bin}" -m challenge_v2.c1_no_lse_abba_cli \
            --batch "${batch}" --warmup 30 --pairs 101 --seed 20260819 \
            --shards "${candidate_shards}" --warps "${candidate_warps}" --stages "${candidate_stages}" \
            --pdl "${candidate_pdl}" --maxnreg "${candidate_maxnreg}" \
            --require-strict-10 --output "${json_path}"
    ) >"${stdout_path}" 2>&1
    cli_rc=$?
    set -e
    if [[ "${cli_rc}" -ne 0 && "${cli_rc}" -ne 3 ]]; then
        printf 'ABORT: AB/BA CLI failed before valid JSON (rc=%s): %s\n' "${cli_rc}" "${stdout_path}" >&2
        return "${cli_rc}"
    fi
    # run_batch is invoked as an `if` condition below. Bash consequently disables
    # errexit inside this function, so each evidence gate must propagate failure
    # explicitly instead of relying on `set -e`.
    validate_json "${json_path}" "${batch}" || return $?
    require_empty_compute_apps "after B=${batch}" || return $?
    return "${cli_rc}"
}

snapshot PRE
require_empty_compute_apps PRE
strict_failure=0
IFS=',' read -r -a requested_batches <<<"${audit_batches}"
for batch in "${requested_batches[@]}"; do
    if run_batch "${batch}"; then
        :
    else
        rc=$?
        if [[ "${rc}" -eq 3 ]]; then
            strict_failure=1
            printf 'STRICT-10 GATE FAILED for B=%s; preserving complete JSON and continuing only to collect fixed-config evidence.\n' "${batch}" >&2
        else
            exit "${rc}"
        fi
    fi
done
snapshot POST
require_empty_compute_apps POST
printf '\nC2 no-LSE fixed AB/BA audit completed: %s\n' "${audit_log}"
if [[ "${strict_failure}" -ne 0 ]]; then
    exit 3
fi
