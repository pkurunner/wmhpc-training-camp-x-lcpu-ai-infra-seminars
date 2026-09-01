#!/usr/bin/env bash
# Authorized AB/BA runner for the scalar versus warp-cooperative C=2 producer
# experiment.  It never submits work and may run only inside one explicitly
# authorized, otherwise-empty B300 Slurm allocation.

set -Eeuo pipefail

if [[ "${C2_CLUSTER_ATTENTION_WARP_PRODUCER_ABBA_AUTHORIZED:-}" != "1" || "${1:-}" != "--authorized-by-parent" ]]; then
    printf '%s\n' 'Refusing native C=2 warp-producer AB/BA benchmark without both authorization tokens.' >&2
    exit 64
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Refusing native C=2 warp-producer AB/BA benchmark outside a Slurm allocation.' >&2
    exit 64
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
c2_root="$(cd "${script_dir}/.." && pwd)"
assignment_root="$(cd "${c2_root}/../.." && pwd)"
python_bin="${C2_PYTHON_BIN:-${assignment_root}/.venv/bin/python}"
nvcc_candidate="${C2_NVCC_BIN:-$(command -v nvcc || true)}"
nvcc_bin=""
[[ -n "${nvcc_candidate}" ]] && nvcc_bin="$(readlink -f "${nvcc_candidate}")"
cuobjdump_bin="$(dirname "${nvcc_bin:-/missing/nvcc}")/cuobjdump"
source_path="${script_dir}/c2_cluster_attention_warp_producer_abba.cu"
imported_path="${script_dir}/c2_cluster_attention_mbarrier_smoke.cu"
runner_path="${script_dir}/run_c2_cluster_attention_warp_producer_abba_clean.sh"
audited_import_sha256='6d69cecbda9baf93b19bfba3162a4c33e42298bee43a9235de98e2acd369372f'
out_dir="${C2_CLUSTER_ATTENTION_WARP_PRODUCER_ABBA_OUT_DIR:-${c2_root}/experiment_logs/c2_cluster_attention_warp_producer_abba}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)_job${SLURM_JOB_ID}"
audit_log="${out_dir}/c2_cluster_attention_warp_producer_abba_clean_${stamp}.log"
compile_log="${out_dir}/c2_cluster_attention_warp_producer_abba_compile_${stamp}.log"
run_log="${out_dir}/c2_cluster_attention_warp_producer_abba_run_${stamp}.log"
raw_json="${out_dir}/c2_cluster_attention_warp_producer_abba_raw_${stamp}.json"
final_json="${out_dir}/c2_cluster_attention_warp_producer_abba_clean_${stamp}.json"
binary_path="${out_dir}/c2_cluster_attention_warp_producer_abba_${stamp}"
ptx_path="${out_dir}/c2_cluster_attention_warp_producer_abba_${stamp}.ptx"
sass_path="${out_dir}/c2_cluster_attention_warp_producer_abba_${stamp}.sass"
scalar_ptx_path="${out_dir}/c2_cluster_attention_warp_producer_abba_scalar_${stamp}.ptx"
warp_ptx_path="${out_dir}/c2_cluster_attention_warp_producer_abba_warp_${stamp}.ptx"
scalar_sass_path="${out_dir}/c2_cluster_attention_warp_producer_abba_scalar_${stamp}.sass"
warp_sass_path="${out_dir}/c2_cluster_attention_warp_producer_abba_warp_${stamp}.sass"

mkdir -p "${out_dir}"
exec > >(tee -a "${audit_log}") 2>&1

compute_apps() {
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}
require_b300() {
    local label="$1" rows name capability
    rows="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits 2>/dev/null)" || {
        printf 'ABORT: B300 identity query failed at %s.\n' "${label}" >&2; return 74; }
    [[ -n "${rows//[[:space:]]/}" ]] || { printf 'ABORT: no visible GPU at %s.\n' "${label}" >&2; return 74; }
    while IFS=',' read -r name capability; do
        name="${name#"${name%%[![:space:]]*}"}"; name="${name%"${name##*[![:space:]]}"}"
        capability="${capability//[[:space:]]/}"
        [[ "${name}" == *B300* && "${capability}" == '10.3' ]] || {
            printf 'ABORT: expected B300 CC 10.3 at %s; got name=%q capability=%q.\n' "${label}" "${name}" "${capability}" >&2; return 75; }
    done <<<"${rows}"
}
require_empty_gpu() {
    local label="$1" apps rows value
    apps="$(compute_apps)" || { printf 'ABORT: app query failed at %s.\n' "${label}" >&2; return 74; }
    [[ -z "${apps}" ]] || { printf 'ABORT: compute app(s) visible at %s:\n%s\n' "${label}" "${apps}" >&2; return 73; }
    rows="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" || {
        printf 'ABORT: memory query failed at %s.\n' "${label}" >&2; return 74; }
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
    printf '%s\n' '-- source/import/runner SHA256 --'; sha256sum "${source_path}" "${imported_path}" "${runner_path}" || true
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
        require_one_uuid >/dev/null || post_rc=$?
        [[ "${rc}" -ne 0 || "${post_rc}" -eq 0 ]] || rc="${post_rc}"
    fi
    printf '\n===== FINAL_RC=%s =====\n' "${rc}"
    exit "${rc}"
}
trap on_exit EXIT

[[ -x "${python_bin}" ]] || { printf 'Missing selected Python: %s\n' "${python_bin}" >&2; exit 65; }
[[ -n "${nvcc_bin}" && -x "${nvcc_bin}" && -x "${cuobjdump_bin}" ]] || { printf '%s\n' 'Missing paired nvcc/cuobjdump.' >&2; exit 65; }
[[ -f "${source_path}" && -f "${imported_path}" ]] || { printf '%s\n' 'Missing source or audited imported source.' >&2; exit 65; }
command -v timeout >/dev/null || { printf '%s\n' 'Missing timeout watchdog.' >&2; exit 65; }
export PYTHONOPTIMIZE=0
[[ "$("${python_bin}" -c 'print(int(__debug__))')" == 1 ]] || { printf '%s\n' 'Python assertions are disabled.' >&2; exit 65; }
grep -q 'sm_103a' <<<"$("${nvcc_bin}" --help)" || { printf '%s\n' 'nvcc does not advertise sm_103a.' >&2; exit 65; }

source_sha_pre="$(sha256sum "${source_path}" | awk '{print $1}')"
imported_sha_pre="$(sha256sum "${imported_path}" | awk '{print $1}')"
runner_sha_pre="$(sha256sum "${runner_path}" | awk '{print $1}')"
[[ "${imported_sha_pre}" == "${audited_import_sha256}" ]] || { printf 'Imported scalar source SHA mismatch: %s\n' "${imported_sha_pre}" >&2; exit 66; }
nvcc_version="$("${nvcc_bin}" --version)"
compile_flags='-std=c++17 -O3 -arch=sm_103a'

snapshot PRE
require_b300 PRE; require_empty_gpu PRE; gpu_uuid="$(require_one_uuid)"

printf '\n===== compile scalar versus warp producer AB/BA =====\n'
"${nvcc_bin}" -std=c++17 -O3 -arch=sm_103a "${source_path}" -o "${binary_path}" >"${compile_log}" 2>&1
"${nvcc_bin}" -std=c++17 -O3 -arch=sm_103a -ptx "${source_path}" -o "${ptx_path}" >>"${compile_log}" 2>&1
"${cuobjdump_bin}" --dump-sass "${binary_path}" >"${sass_path}"

"${python_bin}" - "${ptx_path}" "${sass_path}" "${scalar_ptx_path}" "${warp_ptx_path}" "${scalar_sass_path}" "${warp_sass_path}" <<'PY'
import json
from pathlib import Path
import sys

ptx_path, sass_path, scalar_ptx_path, warp_ptx_path, scalar_sass_path, warp_sass_path = sys.argv[1:]
ptx = Path(ptx_path).read_text(encoding="utf-8").splitlines(keepends=True)
sass = Path(sass_path).read_text(encoding="utf-8").splitlines(keepends=True)

def extract(lines, marker, needle):
    starts = [i for i, line in enumerate(lines) if marker in line and needle in line]
    assert len(starts) == 1, (marker, needle, starts)
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if marker in lines[i]), len(lines))
    return "".join(lines[start:end])

scalar_ptx = extract(ptx, ".entry ", "cluster_attention_mbarrier_kernel")
warp_ptx = extract(ptx, ".entry ", "cluster_attention_mbarrier_warp_producer_kernel")
scalar_sass = extract(sass, "Function : ", "cluster_attention_mbarrier_kernel")
warp_sass = extract(sass, "Function : ", "cluster_attention_mbarrier_warp_producer_kernel")
for path, contents in ((scalar_ptx_path, scalar_ptx), (warp_ptx_path, warp_ptx),
                       (scalar_sass_path, scalar_sass), (warp_sass_path, warp_sass)):
    Path(path).write_text(contents, encoding="utf-8")

def proof(ptx_text, sass_text, require_shuffle):
    result = {
        "ptx_mbarrier_init": ptx_text.count("mbarrier.init.shared.b64"),
        "ptx_mbarrier_release_arrive": ptx_text.count("mbarrier.arrive.release.cluster.shared::cluster.b64"),
        "ptx_mbarrier_acquire_wait": ptx_text.count("mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"),
        "ptx_cluster_arrive": ptx_text.count("barrier.cluster.arrive"),
        "ptx_cluster_wait": ptx_text.count("barrier.cluster.wait"),
        "sass_mbarrier_init": sass_text.count("SYNCS.EXCH.64"),
        "sass_mbarrier_release_arrive": sass_text.count("SYNCS.ARRIVE.TRANS64.RED.A1T0"),
        "sass_mbarrier_acquire_wait": sass_text.count("SYNCS.PHASECHK.TRANS64.TRYWAIT"),
        "sass_cluster_arrive": sass_text.count("UCGABAR_ARV"),
        "sass_cluster_wait": sass_text.count("UCGABAR_WAIT"),
        "ptx_shuffle_down": ptx_text.count("shfl.sync.down.b32"),
        "ptx_shuffle_index": ptx_text.count("shfl.sync.idx.b32"),
        "sass_shuffle_down": sass_text.count("SHFL.DOWN"),
        "sass_shuffle_index": sass_text.count("SHFL.IDX"),
    }
    for key in ("ptx_mbarrier_init", "ptx_mbarrier_release_arrive", "ptx_mbarrier_acquire_wait",
                "sass_mbarrier_init", "sass_mbarrier_release_arrive", "sass_mbarrier_acquire_wait"):
        assert result[key] == 1, (key, result)
    # One cluster sync compiles to a paired arrive/wait.  The fixed protocol has
    # precisely two source-level cluster.sync calls (initialization and lifetime).
    for key in ("ptx_cluster_arrive", "ptx_cluster_wait", "sass_cluster_arrive", "sass_cluster_wait"):
        assert result[key] == 2, (key, result)
    if require_shuffle:
        for key in ("ptx_shuffle_down", "ptx_shuffle_index", "sass_shuffle_down", "sass_shuffle_index"):
            assert result[key] >= 1, (key, result)
    return result

evidence = {"scalar": proof(scalar_ptx, scalar_sass, False), "warp": proof(warp_ptx, warp_sass, True)}
print(json.dumps({"symbol_scoped_instruction_gate": "pass", "evidence": evidence}, sort_keys=True))
PY

printf '\n===== run scalar versus warp producer AB/BA (120 second watchdog) =====\n'
set +e
timeout --preserve-status --kill-after=5s 120s "${binary_path}" >"${raw_json}" 2>"${run_log}"
run_rc=$?
set -e
if [[ "${run_rc}" -ne 0 ]]; then
    printf 'Producer AB/BA failed or timed out (rc=%s), raw=%s stderr=%s\n' "${run_rc}" "${raw_json}" "${run_log}" >&2
    exit "${run_rc}"
fi

source_sha_post="$(sha256sum "${source_path}" | awk '{print $1}')"
imported_sha_post="$(sha256sum "${imported_path}" | awk '{print $1}')"
runner_sha_post="$(sha256sum "${runner_path}" | awk '{print $1}')"
[[ "${source_sha_pre}" == "${source_sha_post}" && "${imported_sha_pre}" == "${imported_sha_post}" && "${runner_sha_pre}" == "${runner_sha_post}" ]] || {
    printf '%s\n' 'Source, import, or runner changed during audit.' >&2; exit 66; }
binary_sha="$(sha256sum "${binary_path}" | awk '{print $1}')"
ptx_sha="$(sha256sum "${ptx_path}" | awk '{print $1}')"
sass_sha="$(sha256sum "${sass_path}" | awk '{print $1}')"

"${python_bin}" - "${raw_json}" "${final_json}" "${source_sha_pre}" "${imported_sha_pre}" "${runner_sha_pre}" "${binary_sha}" "${ptx_sha}" "${sass_sha}" "${nvcc_version}" "${compile_flags}" "${SLURM_JOB_ID}" "${gpu_uuid}" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

assert sys.flags.optimize == 0
(raw_path, final_path, source_sha, imported_sha, runner_sha, binary_sha, ptx_sha, sass_sha,
 nvcc_version, compile_flags, slurm_job_id, gpu_uuid) = sys.argv[1:]
payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
assert payload["schema"] == "c2-cluster-attention-warp-producer-abba-v1"
assert payload["status"] == "pass", payload
assert payload["timing_seed"] == 2026
assert payload["shape"] == {"B": 1, "Hkv": 4, "Hq": 64, "G": 16, "D": 128, "page_size": 128, "selected_pages": 16, "logical_pages": 32}
assert payload["cluster_layout"] == {"num_ctas": 4, "clusters": 4, "selected_pages_per_producer": 8, "threads_per_block": 256}
contract = payload["producer_contract"]
for key in ("same_remote_dsm_mbarrier_protocol", "same_shared_layout_and_output_abi", "same_launch_shape",
            "same_real_selected_causal_attention", "persistent_device_buffers_outside_timing",
            "caller_owned_independent_outputs", "single_kernel_launch_per_cuda_event_sample", "ABBA_interleaved",
            "initialization_copies_and_oracle_outside_timing"):
    assert contract[key] is True, key
assert contract["changed_field"] == "rank-0/1 producer compute mapping only"
assert contract["timed_launch_validation_scope"] == "pre-timing two-seed checks plus post-timing final-state recheck; intermediate timed outputs not inspected"
assert payload["synchronization"]["mbarrier_expected_arrivals"] == 2
assert payload["synchronization"]["mbarrier_wait_parity"] == 0
assert payload["synchronization"]["mbarrier_max_polls"] == 1 << 24
assert payload["environment"]["capability"] == [10, 3] and "B300" in payload["environment"]["device"]
assert payload["environment"]["cluster_launch_supported"] is True
dtype = payload["dtype_contract"]
assert dtype["producer_partial"] == dtype["caller_output"] == "bfloat16" and dtype["oracle_accumulator"] == "float64"
assert math.isclose(float(dtype["tolerance"]["rtol"]), 5e-3, rel_tol=0.0, abs_tol=1e-9)
assert math.isclose(float(dtype["tolerance"]["atol"]), 5e-4, rel_tol=0.0, abs_tol=1e-9)
resources = payload["resource_model"]
assert resources["static_shared_equal"] is True
for arm in ("scalar", "warp"):
    assert int(resources[arm]["static_shared_bytes"]) > 0
    assert int(resources[arm]["num_regs"]) > 0 and int(resources[arm]["local_bytes"]) >= 0
assert resources["scalar"]["static_shared_bytes"] == resources["warp"]["static_shared_bytes"]
expected = {17: (2049, 4), 2026: (3969, 64)}
assert {int(row["seed"]) for row in payload["correctness"]} == set(expected)
for row in payload["correctness"]:
    sequence, unselected = expected[int(row["seed"])]
    assert int(row["sequence_length"]) == sequence and int(row["adversarial_unselected_visible_pages"]) == unselected
    assert int(row["adversarial_masked_tokens"]) == 4 * 127 and row["hierarchy_valid"] is True
    for arm in ("scalar", "warp"):
        check = row[arm]
        assert check["oracle_finite"] is True and check["finite"] is True and check["sentinel_clean"] is True and check["allclose"] is True
        assert math.isfinite(float(check["max_abs"])) and math.isfinite(float(check["max_rel"]))
    cross = row["cross_arm"]
    assert math.isfinite(float(cross["max_abs"])) and math.isfinite(float(cross["max_rel"]))
post = payload["post_timing_correctness"]
assert post["seed"] == 2026 and post["hierarchy_valid"] is True
for arm in ("scalar", "warp"):
    check = post[arm]
    assert check["oracle_finite"] is True and check["finite"] is True and check["sentinel_clean"] is True and check["allclose"] is True
for value in post["cross_arm"].values():
    assert isinstance(value, bool) or math.isfinite(float(value))

def summarize(values):
    values = sorted(float(value) for value in values)
    assert values and all(math.isfinite(value) and value > 0 for value in values)
    count = len(values)
    return {"p10_us": values[max(0, math.ceil(.10 * count) - 1)],
            "median_us": float(statistics.median(values)),
            "p90_us": values[min(count - 1, math.ceil(.90 * count) - 1)]}

timing = payload["timing"]
assert timing["protocol"] == "warmup_each_then_101_scalar_warp_warp_scalar_ABBA_pairs"
assert timing["warmup_each"] == 20 and timing["abba_pairs"] == 101 and timing["samples_per_arm"] == 202
for arm in ("scalar", "warp"):
    ab = [float(value) for value in timing["raw_samples_us"][arm]["AB"]]
    ba = [float(value) for value in timing["raw_samples_us"][arm]["BA"]]
    assert len(ab) == len(ba) == 101
    for partition, values in (("all", [*ab, *ba]), ("when_launch_order_is_AB", ab), ("when_launch_order_is_BA", ba)):
        expected_stats = summarize(values); actual_stats = timing[arm][partition]
        for key, value in expected_stats.items():
            assert math.isclose(float(actual_stats[key]), value, rel_tol=0.0, abs_tol=1e-6), (arm, partition, key)
        assert float(actual_stats["p10_us"]) <= float(actual_stats["median_us"]) <= float(actual_stats["p90_us"])
scalar_median = float(timing["scalar"]["all"]["median_us"]); warp_median = float(timing["warp"]["all"]["median_us"])
speedup = scalar_median / warp_median
assert math.isclose(float(timing["speedup_scalar_over_warp"]), speedup, rel_tol=0.0, abs_tol=1e-6)
ab_speedup = float(timing["scalar"]["when_launch_order_is_AB"]["median_us"]) / float(timing["warp"]["when_launch_order_is_AB"]["median_us"])
ba_speedup = float(timing["scalar"]["when_launch_order_is_BA"]["median_us"]) / float(timing["warp"]["when_launch_order_is_BA"]["median_us"])
assert math.isclose(float(timing["speedup_by_partition"]["AB"]), ab_speedup, rel_tol=0.0, abs_tol=1e-6)
assert math.isclose(float(timing["speedup_by_partition"]["BA"]), ba_speedup, rel_tol=0.0, abs_tol=1e-6)
gate = timing["promotion_gate"]
assert gate["combined_scalar_over_warp_at_least_1_10"] == (speedup >= 1.10)
assert gate["AB_scalar_over_warp_greater_than_1_05"] == (ab_speedup > 1.05)
assert gate["BA_scalar_over_warp_greater_than_1_05"] == (ba_speedup > 1.05)
assert gate["all_correct"] is True
assert gate["promoted"] == (speedup >= 1.10 and ab_speedup > 1.05 and ba_speedup > 1.05 and int(resources["warp"]["local_bytes"]) == 0)
assert imported_sha == "6d69cecbda9baf93b19bfba3162a4c33e42298bee43a9235de98e2acd369372f"
payload["build"] = {"nvcc_version": nvcc_version, "compile_flags": compile_flags, "binary_sha256": binary_sha,
                    "ptx_sha256": ptx_sha, "sass_sha256": sass_sha,
                    "symbol_scoped_instruction_gate": "scalar and warp mbarrier plus two cluster-barrier pairs; warp shuffle PTX/SASS present"}
payload["execution"] = {"slurm_job_id": slurm_job_id, "gpu_uuid": gpu_uuid}
payload["source_sha256"] = {
    "challenge_v2/c2_cluster_attention_warp_producer_abba.cu": source_sha,
    "challenge_v2/c2_cluster_attention_mbarrier_smoke.cu": imported_sha,
    "challenge_v2/run_c2_cluster_attention_warp_producer_abba_clean.sh": runner_sha,
}
Path(final_path).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"secondary_gate": "pass", "json": final_path, "speedup_scalar_over_warp": speedup,
                  "promotion": gate["promoted"]}, sort_keys=True))
PY

require_b300 POST; require_empty_gpu POST
post_uuid="$(require_one_uuid)"; [[ "${post_uuid}" == "${gpu_uuid}" ]] || { printf 'GPU UUID changed: %s -> %s\n' "${gpu_uuid}" "${post_uuid}" >&2; exit 75; }
post_snapshot_done=1
snapshot POST
printf 'Native C=2 scalar versus warp-producer AB/BA completed: %s\n' "${final_json}"
