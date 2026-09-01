#!/usr/bin/env bash
# Package the complete v14/v15 continuation evidence without modifying the
# job-scoped source artifacts.  The resulting archive deliberately retains the
# v14 pre-measurement harness failure alongside both valid RC=3 decisions.

set -Eeuo pipefail
umask 077

: "${C2_EXPECTED_ARCHIVE_SCRIPT_SHA:?set the reviewed archive-script SHA-256}"
[[ "${C2_EXPECTED_ARCHIVE_SCRIPT_SHA}" =~ ^[0-9a-f]{64}$ ]]

BASE=/home/lcpu/85117379
META=${BASE}/c2-native-v14-v15-continuation-metadata-20260831
MANIFEST=${BASE}/c2-native-v14-v15-continuation-evidence-20260831.manifest.sha256
ARCHIVE=${BASE}/c2-native-v14-v15-continuation-evidence-20260831.tar.gz
SIDECAR=${ARCHIVE}.sha256
SCRIPT=$(readlink -f -- "${BASH_SOURCE[0]}")

[[ -d "${BASE}" && ! -L "${BASE}" && "$(readlink -f -- "${BASE}")" == "${BASE}" &&
   "$(stat -c %u -- "${BASE}")" == "$(id -u)" ]]
for output in "${META}" "${MANIFEST}" "${ARCHIVE}" "${SIDECAR}"; do
  [[ ! -e "${output}" && ! -L "${output}" ]] || {
    printf 'refusing to reuse archive output: %s\n' "${output}" >&2
    exit 2
  }
done

published=0
cleanup_partial_outputs() {
  local original_rc=$?
  trap - EXIT
  set +e
  if (( original_rc != 0 && published == 0 )); then
    if [[ -d "${META}" && ! -L "${META}" &&
          "$(readlink -f -- "${META}")" == "${META}" &&
          "$(stat -c %u -- "${META}")" == "$(id -u)" ]]; then
      rm -rf -- "${META}"
    fi
    for partial in "${MANIFEST}" "${ARCHIVE}" "${SIDECAR}"; do
      [[ ! -e "${partial}" && ! -L "${partial}" ]] ||
        [[ -f "${partial}" && ! -L "${partial}" ]] || continue
      rm -f -- "${partial}"
    done
  fi
  exit "${original_rc}"
}
trap cleanup_partial_outputs EXIT

roots=(
  "${BASE}/c2-native-plugin-v14-kv-stage-padding-aot-artifacts-20260831"
  "${BASE}/c2-native-plugin-v14-kv-stage-padding-directed-artifacts-20260831"
  "${BASE}/c2-native-plugin-v14-kv-stage-padding-stress-3pct-artifacts-20260831"
  "${BASE}/c2-native-plugin-v14-kv-stage-padding-stress-3pct-retry1-artifacts-20260831"
  "${BASE}/c2-native-plugin-v15-q-stage-stride144-aot-artifacts-20260831"
  "${BASE}/c2-native-plugin-v15-q-stage-stride144-directed-artifacts-20260831"
  "${BASE}/c2-native-plugin-v15-q-stage-stride144-stress-3pct-artifacts-20260831"
)
files=(
  "${BASE}/slurm-c2-native-plugin-v14-aot-13487.log"
  "${BASE}/slurm-c2-native-plugin-v14-directed-13513.log"
  "${BASE}/slurm-c2-native-plugin-v14-stress-3pct-13518.log"
  "${BASE}/slurm-c2-native-plugin-v14-stress-3pct-13539.log"
  "${BASE}/slurm-c2-native-plugin-v15-aot-13564.log"
  "${BASE}/slurm-c2-native-plugin-v15-directed-13575.log"
  "${BASE}/slurm-c2-native-plugin-v15-stress-3pct-13576.log"
  "${BASE}/native_c2_decode.plugin-v1.cu"
  "${BASE}/native_c2_plugin_schema_20260829.patch"
  "${BASE}/exact_d4_native_c2_plugin_cmake_20260829.patch"
  "${BASE}/exact_d4_python_dispatch_20260829.patch"
  "${BASE}/exact_d4_native_c2_plugin_python_loader_20260829.patch"
  "${BASE}/native_c2_operator_bench_20260829.py"
  "${BASE}/native_c2_v5_softmax_directed_20260829.py"
  "${BASE}/native_c2_v6_register_numerator_directed_20260830.py"
  "${BASE}/native_c2_v14_kv_stage_padding_20260831.patch"
  "${BASE}/build_native_c2_plugin_v14_kv_stage_padding_aot.slurm"
  "${BASE}/validate_native_c2_plugin_v14_kv_stage_padding_directed.slurm"
  "${BASE}/validate_native_c2_plugin_v14_kv_stage_padding_stress_perf_3pct.slurm"
  "${BASE}/validate_native_c2_plugin_v14_kv_stage_padding_stress_perf_3pct_retry1.slurm"
  "${BASE}/native_c2_v15_q_stage_stride144_20260831.patch"
  "${BASE}/build_native_c2_plugin_v15_q_stage_stride144_aot.slurm"
  "${BASE}/validate_native_c2_plugin_v15_q_stage_stride144_directed.slurm"
  "${BASE}/validate_native_c2_plugin_v15_q_stage_stride144_stress_perf_3pct.slurm"
  "${BASE}/c2-native-plugin-v12-aot-artifacts-20260830/job12983/native_c2_decode.v12-q-row-padding.cu"
  "${BASE}/c2-native-plugin-v12-aot-artifacts-20260830/job12983/vllm/_native_c2_msa_decode_plugin.abi3.so"
)
expected_job_dirs=(job13487 job13513 job13518 job13539 job13564 job13575 job13576)
job_ids=(13487 13513 13518 13539 13564 13575 13576)

for index in "${!roots[@]}"; do
  root=${roots[index]}
  [[ -d "${root}" && ! -L "${root}" && "$(readlink -f -- "${root}")" == "${root}" ]]
  mapfile -t top_entries < <(find "${root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)
  [[ ${#top_entries[@]} -eq 1 && "${top_entries[0]}" == "${expected_job_dirs[index]}" ]]
done
for file in "${files[@]}" "${SCRIPT}"; do
  [[ -f "${file}" && ! -L "${file}" ]]
done
[[ -z "$(find "${roots[@]}" -type l -print -quit)" ]] || {
  echo 'symlink found in a selected job-scoped evidence root' >&2
  exit 2
}
[[ -z "$(find "${roots[@]}" ! -type d ! -type f -print -quit)" ]] || {
  echo 'non-regular filesystem object found in a selected evidence root' >&2
  exit 2
}
printf '%s  %s\n' "${C2_EXPECTED_ARCHIVE_SCRIPT_SHA}" "${SCRIPT}" | sha256sum -c -

# Controller records are useful but not a prerequisite for preserving durable
# job-local evidence.  Capture them before creating output; if MinJobAge has
# expired one, retain an explicit boundary instead of making the archive
# irretryably fail after META creation.
scontrol_records=()
for job_id in "${job_ids[@]}"; do
  if record=$(scontrol show job -o "${job_id}" 2>/dev/null); then
    [[ "${record}" == JobId="${job_id}"\ * ]]
    scontrol_records+=("${record}")
  else
    scontrol_records+=("SCHEDULER_RECORD_UNAVAILABLE JobId=${job_id} SOURCE=scontrol DURABLE_JOB_LOCAL_EVIDENCE_IS_AUTHORITATIVE")
  fi
done

mkdir -m 0700 -- "${META}"
install -m 0444 -- "${SCRIPT}" "${META}/submitted-archive-script.sh"
for index in "${!job_ids[@]}"; do
  printf '%s\n' "${scontrol_records[index]}" > "${META}/scontrol-job${job_ids[index]}.txt"
done

cat > "${META}/claims-boundary.txt" <<'EOF'
v14 job13487 AOT and job13513 directed passed.  job13518 stopped before
fixture generation or timing because of an inline-Python argument-unpack bug.
The independently reviewed retry job13539 was a valid RC=3 rejection:
v12->v14 paired-median improvement 0.0240452555 with bootstrap LCB
0.0229466423, so the predeclared strict point >0.03 gate did not pass.

v15 job13564 AOT and job13575 directed passed.  job13576 was a valid RC=3
rejection: v12->v15 paired-median improvement -0.0009713217 with bootstrap
LCB -0.0011928154.  Both decisions retain v12.  These independent percentages
must not be added to or multiplied with any historical version comparison.
EOF

root_rel=()
for root in "${roots[@]}"; do root_rel+=("${root#${BASE}/}"); done
file_rel=()
for file in "${files[@]}"; do file_rel+=("${file#${BASE}/}"); done
meta_rel=${META#${BASE}/}

(
  cd "${BASE}"
  {
    find "${root_rel[@]}" "${meta_rel}" -type f -print0
    printf '%s\0' "${file_rel[@]}"
  } | sort -zu | xargs -0 sha256sum
) > "${MANIFEST}"
[[ -s "${MANIFEST}" ]]
(cd "${BASE}" && sha256sum -c "$(basename "${MANIFEST}")" >/dev/null)

tar -C "${BASE}" --format=posix --sort=name --numeric-owner --owner=0 --group=0 \
  -czf "${ARCHIVE}" "${root_rel[@]}" "${file_rel[@]}" "${meta_rel}" "$(basename "${MANIFEST}")"
[[ -s "${ARCHIVE}" ]]

[[ -z "$(tar -tzf "${ARCHIVE}" | grep -E '(^/|(^|/)\.\.(/|$))' || true)" ]]
[[ -z "$(tar -tzf "${ARCHIVE}" | sort | uniq -d)" ]]
[[ -z "$(comm -23 \
  <(sed -nE 's/^[0-9a-f]{64}  //p' "${MANIFEST}" | sort -u) \
  <(tar -tzf "${ARCHIVE}" | sed 's#^\./##' | sort -u))" ]]
records=$(wc -l < "${MANIFEST}")
regular_members=$(tar -tvzf "${ARCHIVE}" | awk 'substr($1,1,1)=="-" {count++} END {print count+0}')
(( regular_members == records + 1 ))

sha256sum "${ARCHIVE}" "${MANIFEST}" > "${SIDECAR}"
sha256sum -c "${SIDECAR}"

members=$(tar -tzf "${ARCHIVE}" | wc -l)
published=1
printf 'ARCHIVE=%s\nMANIFEST=%s\nSIDECAR=%s\nRECORDS=%s\nMEMBERS=%s\n' \
  "${ARCHIVE}" "${MANIFEST}" "${SIDECAR}" "${records}" "${members}"
