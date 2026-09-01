#!/usr/bin/env bash
# Authorized runner for 4-CTA (idle r3) versus real 3-CTA cluster.sync ABBA.
# This script never submits Slurm work.  It may run only inside one explicitly
# authorized, empty B300 allocation.

set -Eeuo pipefail

if [[ "${C2_CLUSTER_ATTENTION_TOPOLOGY_ABBA_AUTHORIZED:-}" != "1" || "${1:-}" != "--authorized-by-parent" ]]; then
    printf '%s\n' 'Refusing topology ABBA without explicit coordinator authorization.' >&2
    exit 64
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Refusing topology ABBA outside a Slurm allocation.' >&2
    exit 64
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
c2_root="$(cd "${script_dir}/.." && pwd)"
assignment_root="$(cd "${c2_root}/../.." && pwd)"
python_bin="${C2_PYTHON_BIN:-${assignment_root}/.venv/bin/python}"
nvcc_candidate="${C2_NVCC_BIN:-$(command -v nvcc || true)}"
nvcc_bin=""; [[ -n "${nvcc_candidate}" ]] && nvcc_bin="$(readlink -f "${nvcc_candidate}")"
source_path="${script_dir}/c2_cluster_attention_topology_abba.cu"
imported_source_path="${script_dir}/c2_cluster_attention_mbarrier_smoke.cu"
runner_path="${script_dir}/run_c2_cluster_attention_topology_abba_clean.sh"
audited_imported_sha256='6d69cecbda9baf93b19bfba3162a4c33e42298bee43a9235de98e2acd369372f'
out_dir="${C2_CLUSTER_ATTENTION_TOPOLOGY_ABBA_OUT_DIR:-${c2_root}/experiment_logs/c2_cluster_attention_topology_abba}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)_job${SLURM_JOB_ID}"
audit_log="${out_dir}/c2_cluster_attention_topology_abba_clean_${stamp}.log"
compile_log="${out_dir}/c2_cluster_attention_topology_abba_compile_${stamp}.log"
run_log="${out_dir}/c2_cluster_attention_topology_abba_run_${stamp}.log"
raw_json="${out_dir}/c2_cluster_attention_topology_abba_raw_${stamp}.json"
final_json="${out_dir}/c2_cluster_attention_topology_abba_clean_${stamp}.json"
binary_path="${out_dir}/c2_cluster_attention_topology_abba_${stamp}"
ptx_path="${out_dir}/c2_cluster_attention_topology_abba_${stamp}.ptx"
sass_path="${out_dir}/c2_cluster_attention_topology_abba_${stamp}.sass"
topology4_ptx_path="${out_dir}/c2_cluster_attention_topology_abba_topology4_${stamp}.ptx"
topology3_ptx_path="${out_dir}/c2_cluster_attention_topology_abba_topology3_${stamp}.ptx"
topology4_sass_path="${out_dir}/c2_cluster_attention_topology_abba_topology4_${stamp}.sass"
topology3_sass_path="${out_dir}/c2_cluster_attention_topology_abba_topology3_${stamp}.sass"
mkdir -p "${out_dir}"
exec > >(tee -a "${audit_log}") 2>&1

compute_apps() { nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'; }
require_b300() {
    local label="$1" rows name capability
    rows="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits 2>/dev/null)" || { printf 'ABORT: B300 query failed at %s.\n' "${label}" >&2; return 74; }
    [[ -n "${rows//[[:space:]]/}" ]] || { printf 'ABORT: no GPU at %s.\n' "${label}" >&2; return 74; }
    while IFS=',' read -r name capability; do
        name="${name#"${name%%[![:space:]]*}"}"; name="${name%"${name##*[![:space:]]}"}"; capability="${capability//[[:space:]]/}"
        [[ "${name}" == *B300* && "${capability}" == '10.3' ]] || { printf 'ABORT: expected one B300 CC10.3 at %s; got %q / %q.\n' "${label}" "${name}" "${capability}" >&2; return 75; }
    done <<<"${rows}"
}
require_empty_gpu() {
    local label="$1" apps memory_rows used
    apps="$(compute_apps)" || { printf 'ABORT: app query failed at %s.\n' "${label}" >&2; return 74; }
    [[ -z "${apps}" ]] || { printf 'ABORT: compute apps at %s:\n%s\n' "${label}" "${apps}" >&2; return 73; }
    memory_rows="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" || { printf 'ABORT: memory query failed at %s.\n' "${label}" >&2; return 74; }
    [[ -n "${memory_rows//[[:space:]]/}" ]] || { printf 'ABORT: no GPU memory rows at %s.\n' "${label}" >&2; return 74; }
    while IFS= read -r used; do
        used="${used//[[:space:]]/}"; [[ "${used}" =~ ^[0-9]+$ && "${used}" -eq 0 ]] || { printf 'ABORT: nonzero GPU memory at %s: %s MiB.\n' "${label}" "${used}" >&2; return 73; }
    done <<<"${memory_rows}"
}
require_one_gpu_uuid() {
    local -a uuids=(); mapfile -t uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
    [[ "${#uuids[@]}" -eq 1 && "${uuids[0]}" == GPU-* ]] || { printf 'ABORT: expected exactly one GPU UUID; got %q.\n' "${uuids[*]:-}" >&2; return 75; }
    printf '%s\n' "${uuids[0]}"
}
snapshot() {
    local label="$1"
    printf '\n===== %s UTC %s (Slurm job %s) =====\n' "${label}" "$(date -u +%FT%TZ)" "${SLURM_JOB_ID}"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap --format=csv,noheader,nounits || true
    printf '%s\n' '-- compute apps --'; compute_apps || true
    printf '%s\n' '-- tracked source SHA256 --'; sha256sum "${source_path}" "${imported_source_path}" "${runner_path}" || true
}
post_snapshot_done=0
on_exit() {
    local rc=$? post_rc=0; trap - EXIT; set +e
    if [[ "${post_snapshot_done}" -eq 0 ]]; then
        snapshot POST_ON_EXIT; require_b300 POST_ON_EXIT || post_rc=$?; require_empty_gpu POST_ON_EXIT || post_rc=$?; require_one_gpu_uuid >/dev/null || post_rc=$?
        [[ "${rc}" -ne 0 || "${post_rc}" -eq 0 ]] || rc="${post_rc}"
    fi
    printf '\n===== FINAL_RC=%s =====\n' "${rc}"; exit "${rc}"
}
trap on_exit EXIT

[[ -x "${python_bin}" ]] || { printf 'Missing selected Python: %s\n' "${python_bin}" >&2; exit 65; }
[[ -n "${nvcc_bin}" && -x "${nvcc_bin}" ]] || { printf 'Missing nvcc: %s\n' "${nvcc_bin}" >&2; exit 65; }
cuobjdump_bin="$(dirname "${nvcc_bin}")/cuobjdump"; [[ -x "${cuobjdump_bin}" ]] || { printf 'Missing cuobjdump: %s\n' "${cuobjdump_bin}" >&2; exit 65; }
[[ -f "${source_path}" && -f "${imported_source_path}" ]] || { printf '%s\n' 'Missing topology source or audited import.' >&2; exit 65; }
command -v timeout >/dev/null || { printf '%s\n' 'Missing timeout watchdog.' >&2; exit 65; }
cuda_include_dir=""
for candidate in "$(cd "$(dirname "${nvcc_bin}")/.." && pwd)/targets/x86_64-linux/include" \
                 /usr/local/cuda/targets/x86_64-linux/include; do
    if [[ -f "${candidate}/cuda_runtime.h" ]]; then cuda_include_dir="${candidate}"; break; fi
done
[[ -n "${cuda_include_dir}" ]] || { printf '%s\n' 'Could not locate cuda_runtime.h for selected nvcc.' >&2; exit 65; }
cuda_include_flag=("-I${cuda_include_dir}")
export PYTHONOPTIMIZE=0
[[ "$("${python_bin}" -c 'print(int(__debug__))')" == 1 ]] || { printf '%s\n' 'Python assertions disabled.' >&2; exit 65; }
grep -q sm_103a <<<"$("${nvcc_bin}" --help)" || { printf '%s\n' 'nvcc lacks sm_103a.' >&2; exit 65; }

source_sha_pre="$(sha256sum "${source_path}" | awk '{print $1}')"
imported_sha_pre="$(sha256sum "${imported_source_path}" | awk '{print $1}')"
runner_sha_pre="$(sha256sum "${runner_path}" | awk '{print $1}')"
[[ "${imported_sha_pre}" == "${audited_imported_sha256}" ]] || { printf 'Audited imported source SHA mismatch: expected=%s actual=%s\n' "${audited_imported_sha256}" "${imported_sha_pre}" >&2; exit 66; }
nvcc_version="$("${nvcc_bin}" --version)"; compile_flags="-std=c++17 -O3 -arch=sm_103a -I${cuda_include_dir}"
snapshot PRE; require_b300 PRE; require_empty_gpu PRE; gpu_uuid="$(require_one_gpu_uuid)"

printf '\n===== compile native C=2 topology ABBA =====\n'
"${nvcc_bin}" -std=c++17 -O3 -arch=sm_103a "${cuda_include_flag[@]}" "${source_path}" -o "${binary_path}" >"${compile_log}" 2>&1
"${nvcc_bin}" -std=c++17 -O3 -arch=sm_103a "${cuda_include_flag[@]}" -ptx "${source_path}" -o "${ptx_path}" >>"${compile_log}" 2>&1
"${cuobjdump_bin}" --dump-sass "${binary_path}" >"${sass_path}"
"${python_bin}" - "${ptx_path}" "${sass_path}" "${topology4_ptx_path}" "${topology3_ptx_path}" "${topology4_sass_path}" "${topology3_sass_path}" <<'PY'
import json
import sys
from pathlib import Path

ptx_path, sass_path, t4_ptx_path, t3_ptx_path, t4_sass_path, t3_sass_path = sys.argv[1:]
ptx = Path(ptx_path).read_text(encoding="utf-8").splitlines(keepends=True)
sass = Path(sass_path).read_text(encoding="utf-8").splitlines(keepends=True)
def extract(lines, marker, symbol):
    starts = [i for i, line in enumerate(lines) if marker in line and symbol in line]
    assert len(starts) == 1, (marker, symbol, starts)
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if marker in lines[i]), len(lines))
    return "".join(lines[start:end])
t4_ptx = extract(ptx, ".entry ", "cluster_attention_topology4_cluster_sync_kernel")
t3_ptx = extract(ptx, ".entry ", "cluster_attention_topology3_cluster_sync_kernel")
t4_sass = extract(sass, "Function : ", "cluster_attention_topology4_cluster_sync_kernel")
t3_sass = extract(sass, "Function : ", "cluster_attention_topology3_cluster_sync_kernel")
Path(t4_ptx_path).write_text(t4_ptx, encoding="utf-8"); Path(t3_ptx_path).write_text(t3_ptx, encoding="utf-8")
Path(t4_sass_path).write_text(t4_sass, encoding="utf-8"); Path(t3_sass_path).write_text(t3_sass, encoding="utf-8")
def evidence(ptx_text, sass_text):
    return {
        "ptx_cluster_arrive": ptx_text.count("barrier.cluster.arrive"),
        "ptx_cluster_wait": ptx_text.count("barrier.cluster.wait"),
        "ptx_any_mbarrier": ptx_text.count("mbarrier."),
        "sass_cluster_arrive": sass_text.count("UCGABAR_ARV"),
        "sass_cluster_wait": sass_text.count("UCGABAR_WAIT"),
        "sass_any_syncs": sass_text.count("SYNCS."),
    }
out = {"topology4": evidence(t4_ptx, t4_sass), "topology3": evidence(t3_ptx, t3_sass)}
expected = {"ptx_cluster_arrive": 3, "ptx_cluster_wait": 3, "ptx_any_mbarrier": 0, "sass_cluster_arrive": 3, "sass_cluster_wait": 3, "sass_any_syncs": 0}
assert out["topology4"] == expected, out
assert out["topology3"] == expected, out
print(json.dumps({"symbol_scoped_instruction_gate": "pass", "evidence": out}, sort_keys=True))
PY

printf '\n===== run topology ABBA (120 second watchdog) =====\n'
set +e; timeout --preserve-status --kill-after=5s 120s "${binary_path}" >"${raw_json}" 2>"${run_log}"; run_rc=$?; set -e
[[ "${run_rc}" -eq 0 ]] || { printf 'Topology ABBA failed or timed out: rc=%s raw=%s stderr=%s\n' "${run_rc}" "${raw_json}" "${run_log}" >&2; exit "${run_rc}"; }
source_sha_post="$(sha256sum "${source_path}" | awk '{print $1}')"; imported_sha_post="$(sha256sum "${imported_source_path}" | awk '{print $1}')"; runner_sha_post="$(sha256sum "${runner_path}" | awk '{print $1}')"
[[ "${source_sha_pre}" == "${source_sha_post}" && "${imported_sha_pre}" == "${imported_sha_post}" && "${runner_sha_pre}" == "${runner_sha_post}" ]] || { printf '%s\n' 'Tracked source changed during audit.' >&2; exit 66; }
binary_sha="$(sha256sum "${binary_path}" | awk '{print $1}')"; ptx_sha="$(sha256sum "${ptx_path}" | awk '{print $1}')"; sass_sha="$(sha256sum "${sass_path}" | awk '{print $1}')"

"${python_bin}" - "${raw_json}" "${final_json}" "${source_sha_pre}" "${imported_sha_pre}" "${runner_sha_pre}" "${binary_sha}" "${ptx_sha}" "${sass_sha}" "${nvcc_version}" "${compile_flags}" "${SLURM_JOB_ID}" "${gpu_uuid}" <<'PY'
import json, math, statistics, sys
from pathlib import Path
assert sys.flags.optimize == 0
(raw_path, final_path, source_sha, imported_sha, runner_sha, binary_sha, ptx_sha, sass_sha, nvcc_version, compile_flags, job_id, gpu_uuid) = sys.argv[1:]
p = json.loads(Path(raw_path).read_text(encoding="utf-8"))
assert p["schema"] == "c2-cluster-attention-topology-abba-v1" and p["status"] == "pass", p
assert p["boundary"] == (
    "scalar native C=2 complete 4-CTA-versus-3-CTA topology-implementation cost signal, including required "
    "block-to-KV-head mapping; not a pure idle-rank hardware cost or a production/model/server speedup"
)
assert p["timing_seed"] == 2026
assert p["shape"] == {"B": 1, "Hkv": 4, "Hq": 64, "G": 16, "D": 128, "page_size": 128, "selected_pages": 16, "logical_pages": 32}
layout = p["cluster_layout"]
assert layout["topology4"] == {"ctas_per_cluster": 4, "clusters": 4, "idle_rank": 3, "grid_ctas": 16}
assert layout["topology3"] == {"ctas_per_cluster": 3, "clusters": 4, "roles": "rank0/rank1 producers plus rank2 merge only", "grid_ctas": 12}
assert layout["selected_pages_per_producer"] == 8 and layout["threads_per_block"] == 256
assert p["input_contract"] == {"input_indirection": "topk_idx -> block_table -> physical KV page", "block_table_abi": "[B,max_blocks], shared by all KV heads", "adversarial_unselected_visible_pages": True, "adversarial_causal_tail": True, "validated_before_oracle_or_gpu": True}
assert p["synchronization"] == {"data_ready": "cooperative_groups::cluster_group::sync after both producers publish CTA-local BF16 partials", "initialization_sync": "cluster.sync in both arms", "shared_lifetime_sync": "cluster.sync in both arms after rank-2 DSM reads", "mbarrier_used_by_tested_symbols": False}
contract = p["fairness_contract"]
for key in ("same_real_selected_causal_attention", "same_cluster_sync_data_ready_protocol", "same_input_device_buffers", "caller_owned_independent_outputs", "persistent_device_buffers_outside_timing", "single_kernel_launch_per_cuda_event_sample", "ABBA_interleaved", "initialization_copies_and_oracle_outside_timing"):
    assert contract[key] is True, key
assert contract["changed_field"] == "complete cluster topology implementation: 4 CTA including idle rank 3 versus real 3 CTA"
assert contract["attribution_limit"] == (
    "ClusterCtas changes both clusterDim/grid and required blockIdx-to-KV-head mapping; result is not a pure "
    "idle-rank hardware cost"
)
assert contract["timed_launch_validation_scope"] == "pre-timing two-seed checks plus post-timing final-state recheck; intermediate timed outputs not inspected"
dtype = p["dtype_contract"]
assert dtype["producer_partial"] == dtype["caller_output"] == "bfloat16" and dtype["oracle_accumulator"] == "float64"
assert dtype["oracle"] == "independent two-pass natural-exp direct selected-page causal attention"
assert math.isclose(float(dtype["tolerance"]["rtol"]), .005, rel_tol=0.0, abs_tol=1e-9) and math.isclose(float(dtype["tolerance"]["atol"]), .0005, rel_tol=0.0, abs_tol=1e-9)
env = p["environment"]; assert "B300" in env["device"] and env["capability"] == [10, 3] and env["cluster_launch_supported"] is True
resources = p["resource_model"]
assert resources["interpretation"] == "register/local-memory differences are disclosed topology implementation cost; only static shared bytes are matched"
assert resources["static_shared_equal"] is True
for arm in ("topology4", "topology3"):
    assert int(resources[arm]["static_shared_bytes"]) > 0 and int(resources[arm]["num_regs"]) > 0 and int(resources[arm]["local_bytes"]) >= 0
assert resources["topology4"]["static_shared_bytes"] == resources["topology3"]["static_shared_bytes"]
expected = {17: (2049, 4), 2026: (3969, 64)}
assert {int(row["seed"]) for row in p["correctness"]} == set(expected)
def check_arm(arm):
    assert arm["oracle_finite"] is True and arm["finite"] is True and arm["sentinel_clean"] is True and arm["allclose"] is True
    assert math.isfinite(float(arm["max_abs"])) and math.isfinite(float(arm["max_rel"]))
for row in p["correctness"]:
    assert (int(row["sequence_length"]), int(row["adversarial_unselected_visible_pages"])) == expected[int(row["seed"])]
    assert row["hierarchy_valid"] is True and int(row["adversarial_masked_tokens"]) == 4 * 127
    check_arm(row["topology4"]); check_arm(row["topology3"]); assert row["cross_arm_bf16_bitwise_equal"] is True
post = p["post_timing_correctness"]; assert post["seed"] == 2026 and post["hierarchy_valid"] is True
check_arm(post["topology4"]); check_arm(post["topology3"]); assert post["cross_arm_bf16_bitwise_equal"] is True
timing = p["timing"]
assert timing["protocol"] == "warmup_each_then_101_topology4_topology3_topology3_topology4_ABBA_pairs" and timing["warmup_each"] >= 20 and timing["abba_pairs"] == 101 and timing["samples_per_arm"] == 202
def summarize(v):
    v = sorted(float(x) for x in v); assert v and all(math.isfinite(x) and x > 0 for x in v); n = len(v)
    return {"p10_us": v[max(0, math.ceil(.10*n)-1)], "median_us": float(statistics.median(v)), "p90_us": v[min(n-1, math.ceil(.90*n)-1)]}
for arm in ("topology4", "topology3"):
    ab = timing["raw_samples_us"][arm]["AB"]; ba = timing["raw_samples_us"][arm]["BA"]; assert len(ab) == len(ba) == 101
    for bucket, vals in (("all", [*ab, *ba]), ("when_launch_order_is_AB", ab), ("when_launch_order_is_BA", ba)):
        got, want = timing[arm][bucket], summarize(vals)
        for key, val in want.items(): assert math.isclose(float(got[key]), val, rel_tol=0.0, abs_tol=1e-6), (arm, bucket, key, got[key], val)
        assert float(got["p10_us"]) <= float(got["median_us"]) <= float(got["p90_us"])
t4, t3 = float(timing["topology4"]["all"]["median_us"]), float(timing["topology3"]["all"]["median_us"])
s_topo = float(timing["S_topo"]); assert math.isclose(s_topo, t4/t3, rel_tol=0.0, abs_tol=1e-6)
assert math.isclose(float(timing["S_topo_t4_over_t3"]), s_topo, rel_tol=0.0, abs_tol=1e-6)
gate = timing["promotion_gate"]
ab_ratio = float(timing["topology4"]["when_launch_order_is_AB"]["median_us"]) / float(timing["topology3"]["when_launch_order_is_AB"]["median_us"])
ba_ratio = float(timing["topology4"]["when_launch_order_is_BA"]["median_us"]) / float(timing["topology3"]["when_launch_order_is_BA"]["median_us"])
assert math.isclose(float(gate["combined_median_threshold"]), 1.05, rel_tol=0.0, abs_tol=1e-9)
assert math.isclose(float(gate["AB_partition_t4_over_t3"]), ab_ratio, rel_tol=0.0, abs_tol=1e-6) and math.isclose(float(gate["BA_partition_t4_over_t3"]), ba_ratio, rel_tol=0.0, abs_tol=1e-6)
assert bool(gate["combined_median_met"]) == (s_topo >= 1.05) and bool(gate["both_partitions_t4_over_t3_gt_1"]) == (ab_ratio > 1 and ba_ratio > 1)
assert bool(gate["promote_topology_optimization"]) == (s_topo >= 1.05 and ab_ratio > 1 and ba_ratio > 1) and gate["otherwise"] == "freeze scalar topology tuning"
p["build"] = {"nvcc_version": nvcc_version, "compile_flags": compile_flags, "binary_sha256": binary_sha, "ptx_sha256": ptx_sha, "sass_sha256": sass_sha, "symbol_scoped_instruction_evidence": {arm: {"ptx_cluster_arrive": 3, "ptx_cluster_wait": 3, "ptx_any_mbarrier": 0, "sass_cluster_arrive": 3, "sass_cluster_wait": 3, "sass_any_syncs": 0} for arm in ("topology4", "topology3")}}
p["execution"] = {"slurm_job_id": job_id, "gpu_uuid": gpu_uuid}
p["source_sha256"] = {"challenge_v2/c2_cluster_attention_topology_abba.cu": source_sha, "challenge_v2/c2_cluster_attention_mbarrier_smoke.cu": imported_sha, "challenge_v2/run_c2_cluster_attention_topology_abba_clean.sh": runner_sha}
assert imported_sha == "6d69cecbda9baf93b19bfba3162a4c33e42298bee43a9235de98e2acd369372f"
Path(final_path).write_text(json.dumps(p, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"secondary_gate": "pass", "json": final_path, "S_topo_t4_over_t3": s_topo, "promote_topology_optimization": bool(gate["promote_topology_optimization"])}, sort_keys=True))
PY
require_b300 POST; require_empty_gpu POST; post_uuid="$(require_one_gpu_uuid)"; [[ "${post_uuid}" == "${gpu_uuid}" ]] || { printf 'GPU UUID changed: %s -> %s\n' "${gpu_uuid}" "${post_uuid}" >&2; exit 75; }
post_snapshot_done=1; snapshot POST
printf 'Native C=2 topology ABBA completed: %s\n' "${final_json}"
