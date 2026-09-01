#!/usr/bin/env bash
# One-shot, explicitly authorized clean audit for the C=1 no-LSE candidate.
#
# This script deliberately refuses to run until the parent coordinator has
# granted a clean, exclusive B300 window.  It is intended to be executed *in*
# the allocated Slurm job (for example through `srun --jobid=... --overlap`),
# not to allocate a GPU itself.  It makes no claim of a clean timing result if
# any unrelated CUDA compute process is visible before, between, or after the
# four targeted sweeps.

set -Eeuo pipefail

if [[ "${C2_CLEAN_AUDIT_AUTHORIZED:-}" != "1" || "${1:-}" != "--authorized-by-parent" ]]; then
    cat >&2 <<'USAGE'
Refusing to start a performance run without coordinator authorization.

After the parent coordinator has verified the exclusive B300 window, run:
  C2_CLEAN_AUDIT_AUTHORIZED=1 ./challenge_v2/run_nolse_clean_audit.sh --authorized-by-parent
USAGE
    exit 64
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
c2_root="$(cd "${script_dir}/.." && pwd)"
assignment_root="$(cd "${c2_root}/../.." && pwd)"
# B300 uses the assignment-local venv.  The isolated 5090 mirror deliberately
# contains only the source tree, so its already-provisioned CUDA 13 venv is
# supplied explicitly through C2_PYTHON_BIN.  Never silently fall back to an
# arbitrary shell `python`, because that would make the measurement identity
# ambiguous.
python_bin="${C2_PYTHON_BIN:-${assignment_root}/.venv/bin/python}"
out_dir="${c2_root}/experiment_logs/optimization_v2"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
audit_log="${out_dir}/c2_nolse_targeted_clean_audit_${stamp}.log"
audit_batches="${C2_CLEAN_AUDIT_BATCHES:-4,8}"

case "${audit_batches}" in
    4|8|4,8|8,4) ;;
    *)
        printf 'C2_CLEAN_AUDIT_BATCHES must be one of 4, 8, 4,8, or 8,4; got %q\n' "${audit_batches}" >&2
        exit 66
        ;;
esac

if [[ ! -x "${python_bin}" ]]; then
    printf 'Missing required assignment virtualenv: %s\n' "${python_bin}" >&2
    exit 65
fi

mkdir -p "${out_dir}"
exec > >(tee -a "${audit_log}") 2>&1

compute_apps() {
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}

record_snapshot() {
    local label="$1"
    printf '\n===== %s UTC %s =====\n' "${label}" "$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader,nounits
    printf '%s\n' '-- compute apps --'
    compute_apps || true
    printf '%s\n' '-- source SHA256 --'
    sha256sum \
        "${script_dir}/c1_no_lse.py" \
        "${script_dir}/c1_no_lse_cli.py" \
        "${script_dir}/cli.py" \
        "${script_dir}/prepared_tuned.py" \
        "${c2_root}/challenge/prepared_decode.py" \
        "${c2_root}/harness/data.py" \
        "${c2_root}/harness/reference.py" \
        "${script_dir}/run_nolse_clean_audit.sh"
}

require_empty_compute_apps() {
    local label="$1"
    local apps memory_used value
    if ! apps="$(compute_apps)"; then
        printf 'ABORT: nvidia-smi compute-app query failed at %s.\n' "${label}" >&2
        return 74
    fi
    if [[ -n "${apps}" ]]; then
        printf 'ABORT: visible CUDA compute app(s) at %s; timing evidence is invalid:\n%s\n' "${label}" "${apps}" >&2
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

validate_json() {
    local json_path="$1"
    local batch="$2"
    local shards="$3"
    local stages="$4"
    local pdl="$5"
    "${python_bin}" - "${json_path}" "${batch}" "${shards}" "${stages}" "${pdl}" "${c2_root}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

path, batch_text, shards_text, stages_text, pdl, c2_root_text = sys.argv[1:]
batch, shards, stages = map(int, (batch_text, shards_text, stages_text))
c2_root = Path(c2_root_text).resolve()
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload["schema"] == "c2-c1-online-softmax-no-lse-v1"
assert payload["fairness_contract"]["same_input_seed_per_batch"] is True
assert payload["fairness_contract"]["caller_owned_output"] is True
assert payload["fairness_contract"]["persistent_workspace_outside_timing"] is True
assert payload["fairness_contract"]["selected_chunks"] == 1
assert payload["fairness_contract"]["no_merge"] is True
assert payload["fairness_contract"]["no_lse_workspace_or_store"] is True
assert payload["fairness_contract"]["single_call_per_cuda_event"] is True
source_sha256 = payload["source_sha256"]
for required in ("challenge_v2/c1_no_lse.py", "challenge_v2/c1_no_lse_cli.py",
                 "challenge_v2/prepared_tuned.py", "challenge/prepared_decode.py",
                 "harness/reference.py"):
    assert required in source_sha256, required
for relative, declared_sha in source_sha256.items():
    source = (c2_root / relative).resolve()
    assert source.is_relative_to(c2_root), relative
    assert source.is_file(), relative
    actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert actual_sha == declared_sha, (relative, declared_sha, actual_sha)
assert len(payload["results"]) == 1
context = payload["results"][0]
assert context["batch"] == batch and context["storage"] == "bf16"
rows = context["candidates"]
assert len(rows) == 11, len(rows)  # one current-prepared control plus 10 targeted candidates
control = rows[0]
assert control["implementation"] == "current_prepared_control"
assert control["status"] == "pass"
assert control["correctness"]["finite"] is True
control_us = float(control["timing"]["median_us"])
assert control_us > 0 and math.isfinite(control_us)
observed = set()
passing = []
for row in rows[1:]:
    assert row["implementation"] == "c1_online_softmax_no_lse"
    config = row["config"]
    assert config["gqa_shards"] == shards
    assert config["num_stages"] == stages
    assert config["pdl_mode"] == pdl
    assert config["num_warps"] in (2, 4)
    assert config["maxnreg"] in (None, 64, 96, 128, 160)
    observed.add((config["num_warps"], config["maxnreg"]))
    if row["status"] == "pass":
        assert row["correctness"]["finite"] is True
        candidate_us = float(row["timing"]["median_us"])
        declared = float(row["speedup_vs_current_prepared_control"])
        assert candidate_us > 0 and math.isfinite(candidate_us)
        assert abs(declared - control_us / candidate_us) <= 1e-9
        passing.append(row)
assert observed == {(warps, maxnreg) for warps in (2, 4) for maxnreg in (None, 64, 96, 128, 160)}
assert passing, "every no-LSE candidate was rejected"
winner = min(passing, key=lambda row: float(row["timing"]["median_us"]))
summary = context["summary"]
assert summary["status"] == "pass"
assert abs(float(summary["control_median_us"]) - control_us) <= 1e-9
assert summary["winner_config"] == winner["config"]
assert abs(float(summary["no_lse_winner_median_us"]) - float(winner["timing"]["median_us"])) <= 1e-9
assert abs(float(summary["no_lse_winner_speedup"]) - float(winner["speedup_vs_current_prepared_control"])) <= 1e-9
assert bool(summary["strict_10_percent_target_met"]) == (float(summary["no_lse_winner_speedup"]) >= 1.10)
print(json.dumps({
    "json": path,
    "batch": batch,
    "targeted_structure": {"shards": shards, "stages": stages, "pdl": pdl},
    "control_us": control_us,
    "winner_us": float(winner["timing"]["median_us"]),
    "speedup": float(winner["speedup_vs_current_prepared_control"]),
    "strict_10_percent_target_met": bool(summary["strict_10_percent_target_met"]),
    "passing_candidates": len(passing),
}, ensure_ascii=False, sort_keys=True))
PY
}

run_case() {
    local batch="$1"
    local shards="$2"
    local stages="$3"
    local pdl="$4"
    local tag="b${batch}_s${shards}_s${stages}_pdl${pdl}_regwarp_clean_${stamp}"
    local json_path="${out_dir}/c2_nolse_${tag}.json"
    local stdout_path="${out_dir}/c2_nolse_${tag}.stdout.log"
    printf '\n===== targeted case B=%s, shards=%s, stages=%s, PDL=%s =====\n' "${batch}" "${shards}" "${stages}" "${pdl}"
    local cli_rc=0
    set +e
    (
        cd "${c2_root}"
        PYTHONPATH=. "${python_bin}" -m challenge_v2.c1_no_lse_cli \
            --batch "${batch}" --storage-mode bf16 \
            --shards "${shards}" --stages "${stages}" --pdl-modes "${pdl}" \
            --maxnregs none,64,96,128,160 --warps 2,4 \
            --warmup 30 --repetitions 101 --seed 20260819 --require-strict-10 \
            --output "${json_path}"
    ) >"${stdout_path}" 2>&1
    cli_rc=$?
    set -e
    if [[ "${cli_rc}" -ne 0 && "${cli_rc}" -ne 3 ]]; then
        printf 'ABORT: candidate CLI failed before writing a valid audit JSON (rc=%s): %s\n' "${cli_rc}" "${stdout_path}" >&2
        return "${cli_rc}"
    fi
    # run_case is called as an `if` condition, which disables Bash errexit in
    # the whole function. Propagate every evidence gate explicitly.
    validate_json "${json_path}" "${batch}" "${shards}" "${stages}" "${pdl}" || return $?
    require_empty_compute_apps "after ${tag}" || return $?
    # CLI RC=3 means the JSON is structurally valid but its required strict
    # target did not hold.  Propagate it to the collector; do not disguise it
    # as a successful audit case.
    return "${cli_rc}"
}

record_snapshot "PRE"
require_empty_compute_apps "PRE"

strict_failure=0
collect_case() {
    if run_case "$@"; then
        return 0
    else
        # Capture the conditional command's status in the `else` branch.
        # Reading `$?` after a completed `if ... fi` would instead observe the
        # compound conditional's success status and accidentally turn a
        # strict-gate failure into RC=0.
        local rc=$?
    fi
    if [[ "${rc}" -eq 3 ]]; then
        strict_failure=1
        printf 'STRICT-10 GATE FAILED for this structurally valid case; continuing only to collect the remaining targeted evidence.\n' >&2
        return 0
    fi
    return "${rc}"
}

IFS=',' read -r -a requested_batches <<<"${audit_batches}"
for batch in "${requested_batches[@]}"; do
    collect_case "${batch}" 1 5 on
    collect_case "${batch}" 2 4 off
done

record_snapshot "POST"
require_empty_compute_apps "POST"
printf '\nC2 no-LSE targeted clean audit completed: %s\n' "${audit_log}"
if [[ "${strict_failure}" -ne 0 ]]; then
    printf 'C2 no-LSE targeted audit contains at least one strict-10 failure; returning rc=3.\n' >&2
    exit 3
fi
