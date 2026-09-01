#!/usr/bin/env bash
# Slurm-only, double-authorized audit runner for the B=1 C=2 warp-control
# versus WMMA-QK-producer AB/BA experiment.  This script deliberately does not
# submit work; its parent must supply one empty B300 allocation.

set -Eeuo pipefail

if [[ "${C2_CLUSTER_ATTENTION_TC_QK_ABBA_AUTHORIZED:-}" != "1" || "${1:-}" != "--authorized-by-parent" ]]; then
    printf '%s\n' 'Refusing TC-QK AB/BA benchmark without both authorization tokens.' >&2
    exit 64
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Refusing TC-QK AB/BA benchmark outside a Slurm allocation.' >&2
    exit 64
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
c2_root="$(cd "${script_dir}/.." && pwd)"
assignment_root="$(cd "${c2_root}/../.." && pwd)"
python_bin="${C2_PYTHON_BIN:-${assignment_root}/.venv/bin/python}"
nvcc_candidate="${C2_NVCC_BIN:-$(command -v nvcc || true)}"
nvcc_bin=""; [[ -n "${nvcc_candidate}" ]] && nvcc_bin="$(readlink -f "${nvcc_candidate}")"
cuobjdump_bin="$(dirname "${nvcc_bin:-/missing/nvcc}")/cuobjdump"
source_path="${script_dir}/c2_cluster_attention_tc_qk_abba.cu"
warp_import_path="${script_dir}/c2_cluster_attention_warp_producer_abba.cu"
scalar_import_path="${script_dir}/c2_cluster_attention_mbarrier_smoke.cu"
runner_path="${script_dir}/run_c2_cluster_attention_tc_qk_abba_clean.sh"
audited_warp_sha256='24938b464a5b179a7c0e6f2450dd72b231635c73e7b46ea6c5a3fac85357444a'
audited_scalar_sha256='6d69cecbda9baf93b19bfba3162a4c33e42298bee43a9235de98e2acd369372f'
out_dir="${C2_CLUSTER_ATTENTION_TC_QK_ABBA_OUT_DIR:-${c2_root}/experiment_logs/c2_cluster_attention_tc_qk_abba}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)_job${SLURM_JOB_ID}"
audit_log="${out_dir}/c2_cluster_attention_tc_qk_abba_clean_${stamp}.log"
compile_log="${out_dir}/c2_cluster_attention_tc_qk_abba_compile_${stamp}.log"
run_log="${out_dir}/c2_cluster_attention_tc_qk_abba_run_${stamp}.log"
raw_json="${out_dir}/c2_cluster_attention_tc_qk_abba_raw_${stamp}.json"
final_json="${out_dir}/c2_cluster_attention_tc_qk_abba_clean_${stamp}.json"
instruction_json="${out_dir}/c2_cluster_attention_tc_qk_abba_instruction_${stamp}.json"
binary_path="${out_dir}/c2_cluster_attention_tc_qk_abba_${stamp}"
ptx_path="${out_dir}/c2_cluster_attention_tc_qk_abba_${stamp}.ptx"
sass_path="${out_dir}/c2_cluster_attention_tc_qk_abba_${stamp}.sass"
control_ptx_path="${out_dir}/c2_cluster_attention_tc_qk_abba_control_${stamp}.ptx"
tc_ptx_path="${out_dir}/c2_cluster_attention_tc_qk_abba_tc_qk_${stamp}.ptx"
control_sass_path="${out_dir}/c2_cluster_attention_tc_qk_abba_control_${stamp}.sass"
tc_sass_path="${out_dir}/c2_cluster_attention_tc_qk_abba_tc_qk_${stamp}.sass"

mkdir -p "${out_dir}"
exec > >(tee -a "${audit_log}") 2>&1

compute_apps() { nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'; }
require_b300() {
    local label="$1" rows name capability
    rows="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits 2>/dev/null)" || { printf 'ABORT: B300 identity query failed at %s.\n' "${label}" >&2; return 74; }
    [[ -n "${rows//[[:space:]]/}" ]] || { printf 'ABORT: no visible GPU at %s.\n' "${label}" >&2; return 74; }
    while IFS=',' read -r name capability; do
        name="${name#"${name%%[![:space:]]*}"}"; name="${name%"${name##*[![:space:]]}"}"
        capability="${capability//[[:space:]]/}"
        [[ "${name}" == *B300* && "${capability}" == '10.3' ]] || { printf 'ABORT: expected B300 CC 10.3 at %s; got name=%q capability=%q.\n' "${label}" "${name}" "${capability}" >&2; return 75; }
    done <<<"${rows}"
}
require_empty_gpu() {
    local label="$1" apps rows value
    apps="$(compute_apps)" || { printf 'ABORT: app query failed at %s.\n' "${label}" >&2; return 74; }
    [[ -z "${apps}" ]] || { printf 'ABORT: compute app(s) visible at %s:\n%s\n' "${label}" "${apps}" >&2; return 73; }
    rows="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" || { printf 'ABORT: memory query failed at %s.\n' "${label}" >&2; return 74; }
    [[ -n "${rows//[[:space:]]/}" ]] || { printf 'ABORT: no memory rows at %s.\n' "${label}" >&2; return 74; }
    while IFS= read -r value; do
        value="${value//[[:space:]]/}"
        [[ "${value}" =~ ^[0-9]+$ && "${value}" -eq 0 ]] || { printf 'ABORT: GPU memory at %s is %s MiB, not zero.\n' "${label}" "${value}" >&2; return 73; }
    done <<<"${rows}"
}
require_one_uuid() {
    local -a uuids=()
    mapfile -t uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
    [[ "${#uuids[@]}" -eq 1 && "${uuids[0]}" == GPU-* ]] || { printf 'ABORT: expected exactly one GPU UUID, got %q.\n' "${uuids[*]:-}" >&2; return 75; }
    printf '%s\n' "${uuids[0]}"
}
snapshot() {
    local label="$1"
    printf '\n===== %s UTC %s (Slurm job %s) =====\n' "${label}" "$(date -u +%FT%TZ)" "${SLURM_JOB_ID}"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap --format=csv,noheader,nounits || true
    printf '%s\n' '-- compute apps --'; compute_apps || true
    printf '%s\n' '-- source/import/runner SHA256 --'; sha256sum "${source_path}" "${warp_import_path}" "${scalar_import_path}" "${runner_path}" || true
}

post_snapshot_done=0
on_exit() {
    local rc=$? post_rc=0
    trap - EXIT; set +e
    if [[ "${post_snapshot_done}" -eq 0 ]]; then
        snapshot POST_ON_EXIT
        require_b300 POST_ON_EXIT || post_rc=$?
        require_empty_gpu POST_ON_EXIT || post_rc=$?
        require_one_uuid >/dev/null || post_rc=$?
        [[ "${rc}" -ne 0 || "${post_rc}" -eq 0 ]] || rc="${post_rc}"
    fi
    printf '\n===== FINAL_RC=%s =====\n' "${rc}"
    exit "${rc}"
}
trap on_exit EXIT

[[ -x "${python_bin}" ]] || { printf 'Missing selected Python: %s\n' "${python_bin}" >&2; exit 65; }
[[ -n "${nvcc_bin}" && -x "${nvcc_bin}" && -x "${cuobjdump_bin}" ]] || { printf '%s\n' 'Missing paired nvcc/cuobjdump.' >&2; exit 65; }
export PATH="$(dirname "${nvcc_bin}"):${PATH}"
[[ -f "${source_path}" && -f "${warp_import_path}" && -f "${scalar_import_path}" ]] || { printf '%s\n' 'Missing source or imported sources.' >&2; exit 65; }
command -v timeout >/dev/null || { printf '%s\n' 'Missing timeout watchdog.' >&2; exit 65; }
cuda_include_dir=""
for candidate in "$(cd "$(dirname "${nvcc_bin}")/.." && pwd)/targets/x86_64-linux/include" \
                 /usr/local/cuda/targets/x86_64-linux/include; do
    if [[ -f "${candidate}/cuda_runtime.h" ]]; then cuda_include_dir="${candidate}"; break; fi
done
[[ -n "${cuda_include_dir}" ]] || { printf '%s\n' 'Could not locate cuda_runtime.h for selected nvcc.' >&2; exit 65; }
cuda_cccl_dir="${cuda_include_dir}/cccl"
[[ -f "${cuda_cccl_dir}/cuda/std/type_traits" ]] || { printf '%s\n' 'Could not locate CUDA C++ core headers for selected nvcc.' >&2; exit 65; }
cuda_include_flag=("-I${cuda_include_dir}" "-I${cuda_cccl_dir}")
export PYTHONOPTIMIZE=0
[[ "$("${python_bin}" -c 'print(int(__debug__))')" == 1 ]] || { printf '%s\n' 'Python assertions are disabled.' >&2; exit 65; }
grep -q 'sm_103a' <<<"$("${nvcc_bin}" --help)" || { printf '%s\n' 'nvcc does not advertise sm_103a.' >&2; exit 65; }

source_sha_pre="$(sha256sum "${source_path}" | awk '{print $1}')"
warp_sha_pre="$(sha256sum "${warp_import_path}" | awk '{print $1}')"
scalar_sha_pre="$(sha256sum "${scalar_import_path}" | awk '{print $1}')"
runner_sha_pre="$(sha256sum "${runner_path}" | awk '{print $1}')"
[[ "${warp_sha_pre}" == "${audited_warp_sha256}" ]] || { printf 'Imported warp source SHA mismatch: %s\n' "${warp_sha_pre}" >&2; exit 66; }
[[ "${scalar_sha_pre}" == "${audited_scalar_sha256}" ]] || { printf 'Imported scalar source SHA mismatch: %s\n' "${scalar_sha_pre}" >&2; exit 66; }
nvcc_version="$("${nvcc_bin}" --version)"; compile_flags="-std=c++17 -O3 -arch=sm_103a -I${cuda_include_dir} -I${cuda_cccl_dir}"

snapshot PRE
require_b300 PRE; require_empty_gpu PRE; gpu_uuid="$(require_one_uuid)"

printf '\n===== compile warp control versus WMMA-QK candidate =====\n'
"${nvcc_bin}" -std=c++17 -O3 -arch=sm_103a "${cuda_include_flag[@]}" "${source_path}" -o "${binary_path}" >"${compile_log}" 2>&1
"${nvcc_bin}" -std=c++17 -O3 -arch=sm_103a "${cuda_include_flag[@]}" -ptx "${source_path}" -o "${ptx_path}" >>"${compile_log}" 2>&1
"${cuobjdump_bin}" --dump-sass "${binary_path}" >"${sass_path}"

"${python_bin}" - "${ptx_path}" "${sass_path}" "${control_ptx_path}" "${tc_ptx_path}" "${control_sass_path}" "${tc_sass_path}" "${instruction_json}" <<'PY'
import json
import re
import sys
from pathlib import Path

(ptx_path, sass_path, control_ptx_path, tc_ptx_path, control_sass_path, tc_sass_path, evidence_path) = sys.argv[1:]
ptx_lines = Path(ptx_path).read_text(encoding="utf-8").splitlines(keepends=True)
sass_lines = Path(sass_path).read_text(encoding="utf-8").splitlines(keepends=True)

def extract(lines, start_pattern, target):
    starts = [i for i, line in enumerate(lines) if start_pattern in line and target in line]
    assert len(starts) == 1, (start_pattern, target, starts)
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if start_pattern in lines[i]), len(lines))
    return "".join(lines[start:end])

control_ptx = extract(ptx_lines, ".entry ", "cluster_attention_mbarrier_warp_producer_kernel")
tc_ptx = extract(ptx_lines, ".entry ", "cluster_attention_mbarrier_warp_producer_tc_qk_kernel")
control_sass = extract(sass_lines, "Function : ", "cluster_attention_mbarrier_warp_producer_kernel")
tc_sass = extract(sass_lines, "Function : ", "cluster_attention_mbarrier_warp_producer_tc_qk_kernel")
for path, contents in ((control_ptx_path, control_ptx), (tc_ptx_path, tc_ptx),
                       (control_sass_path, control_sass), (tc_sass_path, tc_sass)):
    Path(path).write_text(contents, encoding="utf-8")

def protocol_counts(ptx, sass):
    result = {
        "ptx_mbarrier_init": ptx.count("mbarrier.init.shared.b64"),
        "ptx_mbarrier_release_arrive": ptx.count("mbarrier.arrive.release.cluster.shared::cluster.b64"),
        "ptx_mbarrier_acquire_wait": ptx.count("mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"),
        "ptx_cluster_arrive": ptx.count("barrier.cluster.arrive"),
        "ptx_cluster_wait": ptx.count("barrier.cluster.wait"),
        "sass_mbarrier_init": sass.count("SYNCS.EXCH.64"),
        "sass_mbarrier_release_arrive": sass.count("SYNCS.ARRIVE.TRANS64.RED.A1T0"),
        "sass_mbarrier_acquire_wait": sass.count("SYNCS.PHASECHK.TRANS64.TRYWAIT"),
        "sass_cluster_arrive": sass.count("UCGABAR_ARV"),
        "sass_cluster_wait": sass.count("UCGABAR_WAIT"),
    }
    for key in ("ptx_mbarrier_init", "ptx_mbarrier_release_arrive", "ptx_mbarrier_acquire_wait",
                "sass_mbarrier_init", "sass_mbarrier_release_arrive", "sass_mbarrier_acquire_wait"):
        assert result[key] == 1, (key, result)
    for key in ("ptx_cluster_arrive", "ptx_cluster_wait", "sass_cluster_arrive", "sass_cluster_wait"):
        assert result[key] == 2, (key, result)
    return result

control = protocol_counts(control_ptx, control_sass)
tc = protocol_counts(tc_ptx, tc_sass)
control["ptx_bf16_mma_sync"] = len(re.findall(r"\bmma\.sync(?:\.aligned)?\.[^\n]*\.bf16(?:\.bf16)?", control_ptx))
tc["ptx_bf16_mma_sync"] = len(re.findall(r"\bmma\.sync(?:\.aligned)?\.[^\n]*\.bf16(?:\.bf16)?", tc_ptx))
control["sass_hmma_16816_f32_bf16"] = control_sass.count("HMMA.16816.F32.BF16")
tc["sass_hmma_16816_f32_bf16"] = tc_sass.count("HMMA.16816.F32.BF16")
assert control["ptx_bf16_mma_sync"] == 0 and control["sass_hmma_16816_f32_bf16"] == 0, control
assert tc["ptx_bf16_mma_sync"] > 0, tc
assert tc["sass_hmma_16816_f32_bf16"] > 0, tc
Path(evidence_path).write_text(json.dumps({"symbol_scoped_instruction_gate": "pass", "control": control, "tc_qk": tc}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"symbol_scoped_instruction_gate": "pass", "control": control, "tc_qk": tc}, sort_keys=True))
PY

printf '\n===== run control versus WMMA-QK AB/BA (120 second watchdog) =====\n'
set +e
timeout --preserve-status --kill-after=5s 120s "${binary_path}" >"${raw_json}" 2>"${run_log}"
run_rc=$?
set -e
if [[ "${run_rc}" -ne 0 ]]; then
    printf 'TC-QK AB/BA failed or timed out (rc=%s), raw=%s stderr=%s\n' "${run_rc}" "${raw_json}" "${run_log}" >&2
    exit "${run_rc}"
fi

source_sha_post="$(sha256sum "${source_path}" | awk '{print $1}')"
warp_sha_post="$(sha256sum "${warp_import_path}" | awk '{print $1}')"
scalar_sha_post="$(sha256sum "${scalar_import_path}" | awk '{print $1}')"
runner_sha_post="$(sha256sum "${runner_path}" | awk '{print $1}')"
[[ "${source_sha_pre}" == "${source_sha_post}" && "${warp_sha_pre}" == "${warp_sha_post}" && "${scalar_sha_pre}" == "${scalar_sha_post}" && "${runner_sha_pre}" == "${runner_sha_post}" ]] || { printf '%s\n' 'Source, import, or runner changed during audit.' >&2; exit 66; }
binary_sha="$(sha256sum "${binary_path}" | awk '{print $1}')"; ptx_sha="$(sha256sum "${ptx_path}" | awk '{print $1}')"; sass_sha="$(sha256sum "${sass_path}" | awk '{print $1}')"

"${python_bin}" - "${raw_json}" "${final_json}" "${instruction_json}" "${source_sha_pre}" "${warp_sha_pre}" "${scalar_sha_pre}" "${runner_sha_pre}" "${binary_sha}" "${ptx_sha}" "${sass_sha}" "${nvcc_version}" "${compile_flags}" "${SLURM_JOB_ID}" "${gpu_uuid}" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

assert sys.flags.optimize == 0
(raw_path, final_path, instruction_path, source_sha, warp_sha, scalar_sha, runner_sha, binary_sha, ptx_sha, sass_sha,
 nvcc_version, compile_flags, slurm_job_id, gpu_uuid) = sys.argv[1:]
payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
instruction = json.loads(Path(instruction_path).read_text(encoding="utf-8"))
assert payload["schema"] == "c2-cluster-attention-tc-qk-abba-v1" and payload["status"] == "pass"
assert payload["timing_seed"] == 2026
assert payload["shape"] == {"B": 1, "Hkv": 4, "Hq": 64, "G": 16, "D": 128, "page_size": 128, "selected_pages": 16, "logical_pages": 32}
assert payload["cluster_layout"] == {"num_ctas": 4, "clusters": 4, "selected_pages_per_producer": 8, "threads_per_block": 256}
contract = payload["producer_contract"]
for key in ("same_remote_dsm_mbarrier_protocol", "same_rank2_merge_output_abi_and_lifetime_sync", "same_launch_shape",
            "same_real_selected_causal_attention", "persistent_device_buffers_outside_timing", "caller_owned_independent_outputs",
            "single_kernel_launch_per_cuda_event_sample", "ABBA_interleaved", "initialization_copies_and_oracle_outside_timing",
            "post_timing_fresh_sentinel_reset_and_relaunch"):
    assert contract[key] is True, key
assert contract["changed_field"] == "rank-0/1 producer QK data plane only"
assert contract["timed_launch_validation_scope"] == "pre-timing two-seed checks plus post-timing sentinel reset and fresh untimed control/TC-QK relaunch; intermediate timed outputs not inspected"
assert payload["synchronization"]["mbarrier_expected_arrivals"] == 2 and payload["synchronization"]["mbarrier_wait_parity"] == 0
assert payload["environment"]["capability"] == [10, 3] and "B300" in payload["environment"]["device"] and payload["environment"]["cluster_launch_supported"] is True
dtype = payload["dtype_contract"]
assert dtype["qk"] == "WMMA BF16 m16n16k16 with FP32 accumulator" and dtype["oracle_accumulator"] == "float64"
assert math.isclose(float(dtype["tolerance"]["rtol"]), 5e-3, rel_tol=0.0, abs_tol=1e-9)
assert math.isclose(float(dtype["tolerance"]["atol"]), 5e-4, rel_tol=0.0, abs_tol=1e-9)
resources = payload["resource_model"]
assert resources["static_shared_equal"] is False and resources["candidate_adds_q_and_score_shared"] is True
assert int(resources["control"]["static_shared_bytes"]) > 0 and int(resources["tc_qk"]["static_shared_bytes"]) > int(resources["control"]["static_shared_bytes"])
assert int(resources["control"]["num_regs"]) > 0 and int(resources["tc_qk"]["num_regs"]) > 0 and int(resources["tc_qk"]["local_bytes"]) == 0
expected = {17: (2049, 4), 2026: (3969, 64)}
assert {int(row["seed"]) for row in payload["correctness"]} == set(expected)
for row in payload["correctness"]:
    seq, unselected = expected[int(row["seed"])]
    assert int(row["sequence_length"]) == seq and int(row["adversarial_unselected_visible_pages"]) == unselected
    assert int(row["adversarial_masked_tokens"]) == 4 * 127 and row["hierarchy_valid"] is True
    for arm in ("control", "tc_qk"):
        check = row[arm]
        assert check["oracle_finite"] is True and check["finite"] is True and check["sentinel_clean"] is True and check["allclose"] is True
        assert math.isfinite(float(check["max_abs"])) and math.isfinite(float(check["max_rel"]))
post = payload["post_timing_correctness"]
assert post["seed"] == 2026 and post["hierarchy_valid"] is True
for arm in ("control", "tc_qk"):
    check = post[arm]
    assert check["oracle_finite"] is True and check["finite"] is True and check["sentinel_clean"] is True and check["allclose"] is True
for value in post["cross_arm_diagnostic"].values(): assert isinstance(value, bool) or math.isfinite(float(value))
for arm in ("control", "tc_qk"):
    assert len(payload["timing"]["raw_samples_us"][arm]["AB"]) == 101
    assert len(payload["timing"]["raw_samples_us"][arm]["BA"]) == 101
def summary(values):
    values = sorted(float(v) for v in values); assert values and all(math.isfinite(v) and v > 0 for v in values)
    n = len(values)
    return {"p10_us": values[max(0, math.ceil(.10*n)-1)], "median_us": float(statistics.median(values)), "p90_us": values[min(n-1, math.ceil(.90*n)-1)]}
timing = payload["timing"]
assert timing["protocol"] == "warmup_each_then_101_control_tc_tc_control_ABBA_pairs" and timing["warmup_each"] == 20 and timing["abba_pairs"] == 101 and timing["samples_per_arm"] == 202
for arm in ("control", "tc_qk"):
    for partition, values in (("all", [*timing["raw_samples_us"][arm]["AB"], *timing["raw_samples_us"][arm]["BA"]]),
                              ("when_launch_order_is_AB", timing["raw_samples_us"][arm]["AB"]),
                              ("when_launch_order_is_BA", timing["raw_samples_us"][arm]["BA"])):
        expected_stats = summary(values); actual = timing[arm][partition]
        for key, value in expected_stats.items(): assert math.isclose(float(actual[key]), value, rel_tol=0.0, abs_tol=1e-6), (arm, partition, key)
control_median = float(timing["control"]["all"]["median_us"]); tc_median = float(timing["tc_qk"]["all"]["median_us"])
speedup = control_median / tc_median
assert math.isclose(float(timing["speedup_control_over_tc_qk"]), speedup, rel_tol=0.0, abs_tol=1e-6)
ab_speedup = float(timing["control"]["when_launch_order_is_AB"]["median_us"]) / float(timing["tc_qk"]["when_launch_order_is_AB"]["median_us"])
ba_speedup = float(timing["control"]["when_launch_order_is_BA"]["median_us"]) / float(timing["tc_qk"]["when_launch_order_is_BA"]["median_us"])
gate = timing["promotion_gate"]
assert gate["combined_control_over_tc_qk_at_least_1_10"] == (speedup >= 1.10)
assert gate["AB_control_over_tc_qk_greater_than_1_05"] == (ab_speedup > 1.05)
assert gate["BA_control_over_tc_qk_greater_than_1_05"] == (ba_speedup > 1.05)
assert gate["tc_qk_local_size_bytes_zero"] is True and gate["all_correct"] is True
assert gate["promoted"] == (speedup >= 1.10 and ab_speedup > 1.05 and ba_speedup > 1.05)
assert warp_sha == "24938b464a5b179a7c0e6f2450dd72b231635c73e7b46ea6c5a3fac85357444a"
assert scalar_sha == "6d69cecbda9baf93b19bfba3162a4c33e42298bee43a9235de98e2acd369372f"
assert instruction["symbol_scoped_instruction_gate"] == "pass" and instruction["control"]["ptx_bf16_mma_sync"] == 0 and instruction["control"]["sass_hmma_16816_f32_bf16"] == 0
assert instruction["tc_qk"]["ptx_bf16_mma_sync"] > 0 and instruction["tc_qk"]["sass_hmma_16816_f32_bf16"] > 0
payload["build"] = {"nvcc_version": nvcc_version, "compile_flags": compile_flags, "binary_sha256": binary_sha, "ptx_sha256": ptx_sha, "sass_sha256": sass_sha, "symbol_scoped_instruction_gate": instruction}
payload["execution"] = {"slurm_job_id": slurm_job_id, "gpu_uuid": gpu_uuid}
payload["source_sha256"] = {"challenge_v2/c2_cluster_attention_tc_qk_abba.cu": source_sha, "challenge_v2/c2_cluster_attention_warp_producer_abba.cu": warp_sha, "challenge_v2/c2_cluster_attention_mbarrier_smoke.cu": scalar_sha, "challenge_v2/run_c2_cluster_attention_tc_qk_abba_clean.sh": runner_sha}
Path(final_path).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"secondary_gate": "pass", "json": final_path, "speedup_control_over_tc_qk": speedup, "promotion": gate["promoted"]}, sort_keys=True))
PY

require_b300 POST; require_empty_gpu POST
post_uuid="$(require_one_uuid)"; [[ "${post_uuid}" == "${gpu_uuid}" ]] || { printf 'GPU UUID changed: %s -> %s\n' "${gpu_uuid}" "${post_uuid}" >&2; exit 75; }
post_snapshot_done=1
snapshot POST
printf 'B=1 C=2 warp-control versus WMMA-QK AB/BA completed: %s\n' "${final_json}"
