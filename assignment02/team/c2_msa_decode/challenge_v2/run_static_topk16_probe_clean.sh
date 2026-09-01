#!/usr/bin/env bash
# Authorized clean single-candidate gate for static-topk16 BF16 C=1.
set -Eeuo pipefail

if [[ "${C2_CLEAN_AUDIT_AUTHORIZED:-}" != "1" || "${1:-}" != "--authorized-by-parent" ]]; then
    printf '%s\n' 'Refusing performance run without explicit coordinator authorization.' >&2
    exit 64
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
c2_root="$(cd "${script_dir}/.." && pwd)"
assignment_root="$(cd "${c2_root}/../.." && pwd)"
python_bin="${C2_PYTHON_BIN:-${assignment_root}/.venv/bin/python}"
batch="${C2_STATIC_PROBE_BATCH:-4}"
case "${batch}" in 1|4|8|16) ;; *) printf 'Invalid C2_STATIC_PROBE_BATCH=%q\n' "${batch}" >&2; exit 66;; esac
[[ -x "${python_bin}" ]] || { printf 'Missing Python: %s\n' "${python_bin}" >&2; exit 65; }
out_dir="${c2_root}/experiment_logs/optimization_v2"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
audit_log="${out_dir}/c2_static_topk16_probe_b${batch}_clean_${stamp}.log"
json_path="${out_dir}/c2_static_topk16_probe_b${batch}_clean_${stamp}.json"
mkdir -p "${out_dir}"
exec > >(tee -a "${audit_log}") 2>&1

apps() { nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'; }
check_empty() { local v; v="$(apps || true)"; [[ -z "${v}" ]] || { printf 'ABORT external compute app(s):\n%s\n' "${v}" >&2; exit 73; }; }
snapshot() {
    printf '\n===== %s UTC %s =====\n' "$1" "$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
    printf '%s\n' '-- compute apps --'; apps || true
    printf '%s\n' '-- source SHA256 --'
    sha256sum "${script_dir}/c1_static_topk16.py" "${script_dir}/c1_static_topk16_probe_cli.py" \
              "${script_dir}/c1_no_lse.py" "${script_dir}/cli.py" "${script_dir}/prepared_tuned.py" \
              "${c2_root}/challenge/prepared_decode.py" "${c2_root}/harness/data.py" "${c2_root}/harness/reference.py" \
              "${script_dir}/run_static_topk16_probe_clean.sh"
}

snapshot PRE
check_empty
(
  cd "${c2_root}"
  PYTHONPATH=. "${python_bin}" -m challenge_v2.c1_static_topk16_probe_cli \
    --batch "${batch}" --warmup 30 --repetitions 101 --seed 20260819 --output "${json_path}"
)
"${python_bin}" - "${json_path}" "${batch}" <<'PY'
import json, math, sys
p=json.load(open(sys.argv[1], encoding="utf-8")); b=int(sys.argv[2])
assert p["schema"] == "c2-static-topk16-probe-v1" and p["batch"] == b
assert p["fairness_contract"]["storage"] == "bf16" and p["fairness_contract"]["selected_chunks"] == 1
assert [r["implementation"] for r in p["rows"]] == ["current_prepared_control", "static_topk16_bf16"]
assert all(r["status"] == "pass" and r["correctness"]["finite"] for r in p["rows"])
c, n=p["rows"]
cu=float(c["timing"]["median_us"]); nu=float(n["timing"]["median_us"]); sp=float(n["speedup_vs_current_prepared_control"])
assert cu>0 and nu>0 and math.isfinite(cu) and math.isfinite(nu) and abs(sp-cu/nu)<=1e-9
print(json.dumps({"batch":b,"control_us":cu,"static_topk16_us":nu,"speedup":sp,
                  "strict_10_percent_target_met":bool(p["summary"]["strict_10_percent_target_met"])}))
PY
check_empty
snapshot POST
check_empty
printf 'Static-topk16 clean probe completed: %s\n' "${audit_log}"
