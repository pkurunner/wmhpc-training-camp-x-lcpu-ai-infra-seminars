#!/usr/bin/env bash
# Authorized AB/BA runner for the native C=2 cluster.sync versus DSM-mbarrier
# synchronization microbenchmark.  This script never submits a Slurm job: it
# may run only in one coordinator-authorized, empty B300 allocation.

set -Eeuo pipefail

if [[ "${C2_CLUSTER_ATTENTION_SYNC_ABBA_AUTHORIZED:-}" != "1" || "${1:-}" != "--authorized-by-parent" ]]; then
    printf '%s\n' 'Refusing native C=2 synchronization AB/BA benchmark without explicit coordinator authorization.' >&2
    exit 64
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Refusing native C=2 synchronization AB/BA benchmark outside a Slurm allocation.' >&2
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
source_path="${script_dir}/c2_cluster_attention_sync_abba.cu"
candidate_source_path="${script_dir}/c2_cluster_attention_mbarrier_smoke.cu"
runner_path="${script_dir}/run_c2_cluster_attention_sync_abba_clean.sh"
audited_candidate_sha256='6d69cecbda9baf93b19bfba3162a4c33e42298bee43a9235de98e2acd369372f'
out_dir="${C2_CLUSTER_ATTENTION_SYNC_ABBA_OUT_DIR:-${c2_root}/experiment_logs/c2_cluster_attention_sync_abba}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
audit_log="${out_dir}/c2_cluster_attention_sync_abba_clean_${stamp}.log"
compile_log="${out_dir}/c2_cluster_attention_sync_abba_compile_${stamp}.log"
run_log="${out_dir}/c2_cluster_attention_sync_abba_run_${stamp}.log"
raw_json="${out_dir}/c2_cluster_attention_sync_abba_raw_${stamp}.json"
final_json="${out_dir}/c2_cluster_attention_sync_abba_clean_${stamp}.json"
binary_path="${out_dir}/c2_cluster_attention_sync_abba_${stamp}"
ptx_path="${out_dir}/c2_cluster_attention_sync_abba_${stamp}.ptx"
sass_path="${out_dir}/c2_cluster_attention_sync_abba_${stamp}.sass"
control_ptx_path="${out_dir}/c2_cluster_attention_sync_abba_control_${stamp}.ptx"
candidate_ptx_path="${out_dir}/c2_cluster_attention_sync_abba_candidate_${stamp}.ptx"
control_sass_path="${out_dir}/c2_cluster_attention_sync_abba_control_${stamp}.sass"
candidate_sass_path="${out_dir}/c2_cluster_attention_sync_abba_candidate_${stamp}.sass"

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
    [[ -n "${memory_used//[[:space:]]/}" ]] || { printf 'ABORT: no memory rows at %s.\n' "${label}" >&2; return 74; }
    while IFS= read -r value; do
        value="${value//[[:space:]]/}"
        [[ "${value}" =~ ^[0-9]+$ && "${value}" -eq 0 ]] || {
            printf 'ABORT: GPU memory is not exactly zero at %s: %s MiB.\n' "${label}" "${value}" >&2
            return 73
        }
    done <<<"${memory_used}"
}

require_one_gpu_uuid() {
    local -a uuids=()
    mapfile -t uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
    [[ "${#uuids[@]}" -eq 1 && "${uuids[0]}" == GPU-* ]] || {
        printf 'ABORT: expected exactly one allocated GPU UUID; got %q.\n' "${uuids[*]:-}" >&2
        return 75
    }
    printf '%s\n' "${uuids[0]}"
}

snapshot() {
    local label="$1"
    printf '\n===== %s UTC %s (Slurm job %s) =====\n' "${label}" "$(date -u +%FT%TZ)" "${SLURM_JOB_ID}"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap \
        --format=csv,noheader,nounits || true
    printf '%s\n' '-- compute apps --'
    compute_apps || true
    printf '%s\n' '-- tracked source SHA256 --'
    sha256sum "${source_path}" "${candidate_source_path}" "${runner_path}" || true
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
        require_one_gpu_uuid >/dev/null || post_rc=$?
        if [[ "${rc}" -eq 0 && "${post_rc}" -ne 0 ]]; then
            rc="${post_rc}"
        fi
    fi
    printf '\n===== FINAL_RC=%s =====\n' "${rc}"
    exit "${rc}"
}
trap on_exit EXIT

[[ -x "${python_bin}" ]] || { printf 'Missing selected Python: %s\n' "${python_bin}" >&2; exit 65; }
[[ -n "${nvcc_bin}" && -x "${nvcc_bin}" ]] || { printf 'Missing nvcc: %s\n' "${nvcc_bin}" >&2; exit 65; }
cuobjdump_bin="$(dirname "${nvcc_bin}")/cuobjdump"
[[ -x "${cuobjdump_bin}" ]] || { printf 'Missing cuobjdump paired with nvcc: %s\n' "${cuobjdump_bin}" >&2; exit 65; }
[[ -f "${source_path}" && -f "${candidate_source_path}" ]] || { printf '%s\n' 'Missing AB/BA CUDA source or imported candidate source.' >&2; exit 65; }
command -v timeout >/dev/null || { printf '%s\n' 'Missing timeout watchdog utility.' >&2; exit 65; }
export PYTHONOPTIMIZE=0
[[ "$("${python_bin}" -c 'print(int(__debug__))')" == "1" ]] || {
    printf '%s\n' 'Python assertions are disabled; refusing an unverifiable synchronization audit.' >&2
    exit 65
}
nvcc_help="$("${nvcc_bin}" --help)"
grep -q 'sm_103a' <<<"${nvcc_help}" || { printf '%s\n' 'nvcc does not advertise sm_103a.' >&2; exit 65; }

source_sha_pre="$(sha256sum "${source_path}" | awk '{print $1}')"
candidate_source_sha_pre="$(sha256sum "${candidate_source_path}" | awk '{print $1}')"
runner_sha_pre="$(sha256sum "${runner_path}" | awk '{print $1}')"
[[ "${candidate_source_sha_pre}" == "${audited_candidate_sha256}" ]] || {
    printf 'Imported candidate SHA is not the independently audited job-10731 source: expected=%s actual=%s\n' \
        "${audited_candidate_sha256}" "${candidate_source_sha_pre}" >&2
    exit 66
}
nvcc_version="$("${nvcc_bin}" --version)"
compile_flags='-std=c++17 -O3 -arch=sm_103a'

snapshot PRE
require_b300 PRE
require_empty_gpu PRE
gpu_uuid="$(require_one_gpu_uuid)"

printf '\n===== compile native C=2 cluster.sync versus DSM mbarrier AB/BA =====\n'
"${nvcc_bin}" -std=c++17 -O3 -arch=sm_103a "${source_path}" -o "${binary_path}" >"${compile_log}" 2>&1
"${nvcc_bin}" -std=c++17 -O3 -arch=sm_103a -ptx "${source_path}" -o "${ptx_path}" >>"${compile_log}" 2>&1
"${cuobjdump_bin}" --dump-sass "${binary_path}" >"${sass_path}"
"${python_bin}" - "${ptx_path}" "${sass_path}" "${control_ptx_path}" "${candidate_ptx_path}" \
    "${control_sass_path}" "${candidate_sass_path}" <<'PY'
from pathlib import Path
import json
import sys

ptx_path, sass_path, control_ptx_path, candidate_ptx_path, control_sass_path, candidate_sass_path = sys.argv[1:]
ptx_lines = Path(ptx_path).read_text(encoding="utf-8").splitlines(keepends=True)
sass_lines = Path(sass_path).read_text(encoding="utf-8").splitlines(keepends=True)

def extract(lines, marker, needle):
    starts = [index for index, line in enumerate(lines) if marker in line and needle in line]
    assert len(starts) == 1, (marker, needle, starts)
    start = starts[0]
    end = next((index for index in range(start + 1, len(lines)) if marker in lines[index]), len(lines))
    return "".join(lines[start:end])

control_ptx = extract(ptx_lines, ".entry ", "cluster_attention_cluster_sync_kernel")
candidate_ptx = extract(ptx_lines, ".entry ", "cluster_attention_mbarrier_kernel")
control_sass = extract(sass_lines, "Function : ", "cluster_attention_cluster_sync_kernel")
candidate_sass = extract(sass_lines, "Function : ", "cluster_attention_mbarrier_kernel")

Path(control_ptx_path).write_text(control_ptx, encoding="utf-8")
Path(candidate_ptx_path).write_text(candidate_ptx, encoding="utf-8")
Path(control_sass_path).write_text(control_sass, encoding="utf-8")
Path(candidate_sass_path).write_text(candidate_sass, encoding="utf-8")

evidence = {
    "control": {
        "ptx_cluster_arrive": control_ptx.count("barrier.cluster.arrive"),
        "ptx_cluster_wait": control_ptx.count("barrier.cluster.wait"),
        "ptx_mbarrier_release_arrive": control_ptx.count("mbarrier.arrive.release.cluster.shared::cluster.b64"),
        "ptx_mbarrier_acquire_wait": control_ptx.count("mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"),
        "sass_cluster_arrive": control_sass.count("UCGABAR_ARV"),
        "sass_cluster_wait": control_sass.count("UCGABAR_WAIT"),
        "sass_mbarrier_release_arrive": control_sass.count("SYNCS.ARRIVE.TRANS64.RED.A1T0"),
        "sass_mbarrier_acquire_wait": control_sass.count("SYNCS.PHASECHK.TRANS64.TRYWAIT"),
    },
    "candidate": {
        "ptx_cluster_arrive": candidate_ptx.count("barrier.cluster.arrive"),
        "ptx_cluster_wait": candidate_ptx.count("barrier.cluster.wait"),
        "ptx_mbarrier_init": candidate_ptx.count("mbarrier.init.shared.b64"),
        "ptx_mbarrier_release_arrive": candidate_ptx.count("mbarrier.arrive.release.cluster.shared::cluster.b64"),
        "ptx_mbarrier_acquire_wait": candidate_ptx.count("mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"),
        "sass_cluster_arrive": candidate_sass.count("UCGABAR_ARV"),
        "sass_cluster_wait": candidate_sass.count("UCGABAR_WAIT"),
        "sass_mbarrier_init": candidate_sass.count("SYNCS.EXCH.64"),
        "sass_mbarrier_release_arrive": candidate_sass.count("SYNCS.ARRIVE.TRANS64.RED.A1T0"),
        "sass_mbarrier_acquire_wait": candidate_sass.count("SYNCS.PHASECHK.TRANS64.TRYWAIT"),
    },
}
assert evidence["control"] == {
    "ptx_cluster_arrive": 3, "ptx_cluster_wait": 3,
    "ptx_mbarrier_release_arrive": 0, "ptx_mbarrier_acquire_wait": 0,
    "sass_cluster_arrive": 3, "sass_cluster_wait": 3,
    "sass_mbarrier_release_arrive": 0, "sass_mbarrier_acquire_wait": 0,
}, evidence
assert evidence["candidate"] == {
    "ptx_cluster_arrive": 2, "ptx_cluster_wait": 2, "ptx_mbarrier_init": 1,
    "ptx_mbarrier_release_arrive": 1, "ptx_mbarrier_acquire_wait": 1,
    "sass_cluster_arrive": 2, "sass_cluster_wait": 2, "sass_mbarrier_init": 1,
    "sass_mbarrier_release_arrive": 1, "sass_mbarrier_acquire_wait": 1,
}, evidence
print(json.dumps({"symbol_scoped_instruction_gate": "pass", "evidence": evidence}, sort_keys=True))
PY

printf '\n===== run C=2 synchronization AB/BA (120 second watchdog) =====\n'
set +e
timeout --preserve-status --kill-after=5s 120s "${binary_path}" >"${raw_json}" 2>"${run_log}"
run_rc=$?
set -e
if [[ "${run_rc}" -ne 0 ]]; then
    printf 'C=2 synchronization AB/BA failed or timed out (rc=%s); raw=%s stderr=%s\n' \
        "${run_rc}" "${raw_json}" "${run_log}" >&2
    exit "${run_rc}"
fi

source_sha_post="$(sha256sum "${source_path}" | awk '{print $1}')"
candidate_source_sha_post="$(sha256sum "${candidate_source_path}" | awk '{print $1}')"
runner_sha_post="$(sha256sum "${runner_path}" | awk '{print $1}')"
[[ "${source_sha_pre}" == "${source_sha_post}" ]] || { printf '%s\n' 'AB/BA CUDA source changed during audit.' >&2; exit 66; }
[[ "${candidate_source_sha_pre}" == "${candidate_source_sha_post}" ]] || { printf '%s\n' 'Imported mbarrier candidate source changed during audit.' >&2; exit 66; }
[[ "${runner_sha_pre}" == "${runner_sha_post}" ]] || { printf '%s\n' 'AB/BA runner changed during audit.' >&2; exit 66; }
binary_sha="$(sha256sum "${binary_path}" | awk '{print $1}')"
ptx_sha="$(sha256sum "${ptx_path}" | awk '{print $1}')"
sass_sha="$(sha256sum "${sass_path}" | awk '{print $1}')"

"${python_bin}" - "${raw_json}" "${final_json}" "${source_sha_pre}" "${candidate_source_sha_pre}" \
    "${runner_sha_pre}" "${binary_sha}" "${ptx_sha}" "${sass_sha}" "${nvcc_version}" \
    "${compile_flags}" "${SLURM_JOB_ID}" "${gpu_uuid}" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

if sys.flags.optimize != 0:
    raise RuntimeError("secondary gate requires enabled Python assertions")

(
    raw_path, final_path, source_sha, candidate_source_sha, runner_sha,
    binary_sha, ptx_sha, sass_sha, nvcc_version, compile_flags,
    slurm_job_id, gpu_uuid,
) = sys.argv[1:]
payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
assert payload["schema"] == "c2-cluster-attention-sync-abba-v1"
assert payload["status"] == "pass", payload
assert payload["boundary"] == (
    "scalar native C=2 correctness-prototype synchronization-cost signal only; "
    "not a production fusion, throughput result, or vLLM/model/server speedup"
)
assert payload["timing_seed"] == 2026
assert payload["shape"] == {
    "B": 1, "Hkv": 4, "Hq": 64, "G": 16, "D": 128,
    "page_size": 128, "selected_pages": 16, "logical_pages": 32,
}
assert payload["cluster_layout"] == {
    "num_ctas": 4, "clusters": 4, "selected_pages_per_producer": 8, "threads_per_block": 256,
}
environment = payload["environment"]
assert "B300" in environment["device"] and environment["capability"] == [10, 3]
assert environment["cluster_launch_supported"] is True
assert payload["input_contract"] == {
    "input_indirection": "topk_idx -> block_table -> physical KV page",
    "block_table_abi": "[B,max_blocks], shared by all KV heads",
    "adversarial_unselected_visible_pages": True,
    "adversarial_causal_tail": True,
    "validated_before_oracle_or_gpu": True,
}
contract = payload["fairness_contract"]
for key in (
    "same_real_selected_causal_attention", "same_launch_shape", "same_input_device_buffers",
    "caller_owned_independent_outputs", "persistent_device_buffers_outside_timing",
    "single_kernel_launch_per_cuda_event_sample", "ABBA_interleaved",
    "initialization_copies_and_oracle_outside_timing",
):
    assert contract[key] is True, key
assert contract["changed_field"] == "producer-ready synchronization protocol only"
assert contract["timed_launch_validation_scope"] == (
    "pre-timing two-seed checks plus post-timing final-state recheck; intermediate timed outputs not inspected"
)
synchronization = payload["synchronization"]
assert synchronization["control_data_ready"] == (
    "cooperative_groups::cluster_group::sync after both producers publish CTA-local BF16 partials"
)
assert synchronization["candidate_data_ready"] == (
    "two remote DSM mbarrier.arrive.release.cluster calls followed by rank-2 local "
    "mbarrier.try_wait.parity.acquire.cluster"
)
assert synchronization["candidate_mbarrier_expected_arrivals"] == 2
assert synchronization["candidate_mbarrier_wait_parity"] == 0
assert synchronization["candidate_mbarrier_max_polls"] == (1 << 24)
assert synchronization["shared_lifetime_sync"] == "cluster.sync in both arms after rank-2 DSM reads"
dtype = payload["dtype_contract"]
assert dtype["producer_partial"] == dtype["caller_output"] == "bfloat16"
assert dtype["oracle_accumulator"] == "float64"
assert dtype["oracle"] == "independent two-pass natural-exp direct selected-page causal attention"
assert math.isclose(float(dtype["tolerance"]["rtol"]), 0.005, rel_tol=0.0, abs_tol=1e-9)
assert math.isclose(float(dtype["tolerance"]["atol"]), 0.0005, rel_tol=0.0, abs_tol=1e-9)
resources = payload["resource_model"]
assert resources["interpretation"] == (
    "register/local-memory differences are disclosed protocol implementation cost; only static shared bytes are matched"
)
assert resources["static_shared_equal"] is True
for arm in ("control", "candidate"):
    assert int(resources[arm]["static_shared_bytes"]) > 0
    assert int(resources[arm]["num_regs"]) > 0
    assert int(resources[arm]["local_bytes"]) >= 0
assert resources["control"]["static_shared_bytes"] == resources["candidate"]["static_shared_bytes"]

expected_scenarios = {
    17: {"sequence_length": 2049, "adversarial_unselected_visible_pages": 4},
    2026: {"sequence_length": 3969, "adversarial_unselected_visible_pages": 64},
}
assert {int(row["seed"]) for row in payload["correctness"]} == set(expected_scenarios)
for row in payload["correctness"]:
    expected = expected_scenarios[int(row["seed"])]
    assert int(row["sequence_length"]) == expected["sequence_length"]
    assert int(row["adversarial_unselected_visible_pages"]) == expected["adversarial_unselected_visible_pages"]
    assert int(row["adversarial_masked_tokens"]) == 4 * 127
    assert row["hierarchy_valid"] is True
    for arm in ("control", "candidate"):
        correctness = row[arm]
        assert correctness["oracle_finite"] is True and correctness["finite"] is True
        assert correctness["sentinel_clean"] is True and correctness["allclose"] is True
        assert math.isfinite(float(correctness["max_abs"])) and math.isfinite(float(correctness["max_rel"]))
    assert row["cross_arm_bf16_bitwise_equal"] is True
post_timing = payload["post_timing_correctness"]
assert post_timing["seed"] == 2026 and post_timing["hierarchy_valid"] is True
for arm in ("control", "candidate"):
    correctness = post_timing[arm]
    assert correctness["oracle_finite"] is True and correctness["finite"] is True
    assert correctness["sentinel_clean"] is True and correctness["allclose"] is True
    assert math.isfinite(float(correctness["max_abs"])) and math.isfinite(float(correctness["max_rel"]))
assert post_timing["cross_arm_bf16_bitwise_equal"] is True

timing = payload["timing"]
assert timing["protocol"] == "warmup_each_then_101_control_candidate_candidate_control_ABBA_pairs"
assert timing["warmup_each"] >= 20 and timing["abba_pairs"] == 101 and timing["samples_per_arm"] == 202

def summarize(values):
    ordered = sorted(float(value) for value in values)
    assert ordered and all(value > 0 and math.isfinite(value) for value in ordered)
    count = len(ordered)
    return {
        "p10_us": ordered[max(0, math.ceil(0.10 * count) - 1)],
        "median_us": float(statistics.median(ordered)),
        "p90_us": ordered[min(count - 1, math.ceil(0.90 * count) - 1)],
    }

raw = timing["raw_samples_us"]
for arm in ("cluster_sync_control", "remote_dsm_mbarrier_candidate"):
    ab = [float(value) for value in raw[arm]["AB"]]
    ba = [float(value) for value in raw[arm]["BA"]]
    assert len(ab) == len(ba) == 101
    recomputed = {
        "all": summarize([*ab, *ba]),
        "when_launch_order_is_AB": summarize(ab),
        "when_launch_order_is_BA": summarize(ba),
    }
    for partition, expected_count in (("all", 202), ("when_launch_order_is_AB", 101), ("when_launch_order_is_BA", 101)):
        actual = timing[arm][partition]
        for field, expected in recomputed[partition].items():
            assert math.isclose(float(actual[field]), expected, rel_tol=0.0, abs_tol=1e-6), (arm, partition, field)
        assert float(actual["p10_us"]) <= float(actual["median_us"]) <= float(actual["p90_us"])
        assert expected_count == (len(ab) + len(ba) if partition == "all" else len(ab))

control_median = float(timing["cluster_sync_control"]["all"]["median_us"])
candidate_median = float(timing["remote_dsm_mbarrier_candidate"]["all"]["median_us"])
speedup = float(timing["speedup_control_over_candidate"])
assert control_median > 0 and candidate_median > 0
assert math.isclose(speedup, control_median / candidate_median, rel_tol=0.0, abs_tol=1e-6)
assert bool(timing["strict_10_percent_target_met"]) == (speedup >= 1.10)

payload["build"] = {
    "nvcc_version": nvcc_version,
    "compile_flags": compile_flags,
    "binary_sha256": binary_sha,
    "ptx_sha256": ptx_sha,
    "sass_sha256": sass_sha,
    "symbol_scoped_instruction_evidence": {
        "control": {
            "ptx_cluster_arrive": 3, "ptx_cluster_wait": 3,
            "ptx_mbarrier_release_arrive": 0, "ptx_mbarrier_acquire_wait": 0,
            "sass_cluster_arrive": 3, "sass_cluster_wait": 3,
            "sass_mbarrier_release_arrive": 0, "sass_mbarrier_acquire_wait": 0,
        },
        "candidate": {
            "ptx_cluster_arrive": 2, "ptx_cluster_wait": 2, "ptx_mbarrier_init": 1,
            "ptx_mbarrier_release_arrive": 1, "ptx_mbarrier_acquire_wait": 1,
            "sass_cluster_arrive": 2, "sass_cluster_wait": 2, "sass_mbarrier_init": 1,
            "sass_mbarrier_release_arrive": 1, "sass_mbarrier_acquire_wait": 1,
        },
    },
}
payload["execution"] = {"slurm_job_id": slurm_job_id, "gpu_uuid": gpu_uuid}
payload["source_sha256"] = {
    "challenge_v2/c2_cluster_attention_sync_abba.cu": source_sha,
    "challenge_v2/c2_cluster_attention_mbarrier_smoke.cu": candidate_source_sha,
    "challenge_v2/run_c2_cluster_attention_sync_abba_clean.sh": runner_sha,
}
assert candidate_source_sha == "6d69cecbda9baf93b19bfba3162a4c33e42298bee43a9235de98e2acd369372f"
Path(final_path).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "secondary_gate": "pass", "json": final_path,
    "control_median_us": control_median, "candidate_median_us": candidate_median,
    "speedup_control_over_candidate": speedup,
}, ensure_ascii=False, sort_keys=True))
PY

require_b300 POST
require_empty_gpu POST
post_uuid="$(require_one_gpu_uuid)"
[[ "${post_uuid}" == "${gpu_uuid}" ]] || { printf 'GPU UUID changed during audit: %s -> %s\n' "${gpu_uuid}" "${post_uuid}" >&2; exit 75; }
post_snapshot_done=1
snapshot POST
printf 'Native C=2 synchronization AB/BA completed: %s\n' "${final_json}"
