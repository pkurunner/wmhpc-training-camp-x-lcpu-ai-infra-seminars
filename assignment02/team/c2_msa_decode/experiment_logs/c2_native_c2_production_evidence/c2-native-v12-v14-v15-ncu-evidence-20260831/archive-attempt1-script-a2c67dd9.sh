#!/usr/bin/env bash
# Freeze one successful v12/v14/v15 NCU mechanism job after it is quiescent.
# NCU counters are mechanism evidence only; this archive never creates or
# changes a benchmark-performance decision.

set -Eeuo pipefail
umask 077

: "${C2_NCU_JOB_ID:?set the completed NCU Slurm job id}"
: "${C2_EXPECTED_ARCHIVE_SCRIPT_SHA:?set reviewed archive script SHA-256}"
: "${C2_EXPECTED_PROFILE_SCRIPT_SHA:?set reviewed profile script SHA-256}"
[[ "${C2_NCU_JOB_ID}" =~ ^[0-9]+$ ]]
[[ "${C2_EXPECTED_ARCHIVE_SCRIPT_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ "${C2_EXPECTED_PROFILE_SCRIPT_SHA}" =~ ^[0-9a-f]{64}$ ]]

BASE=/home/lcpu/85117379
ROOT=${BASE}/c2-native-plugin-v12-v14-v15-ncu-artifacts-20260831
JOB_DIR=${ROOT}/job${C2_NCU_JOB_ID}
SLURM_LOG=${ROOT}/slurm-${C2_NCU_JOB_ID}.log
PROFILE_SCRIPT=${BASE}/profile_native_c2_v12_v14_v15_ncu_20260831.slurm
EXPECTED_ARCHIVE_SCRIPT=${BASE}/archive_native_c2_v12_v14_v15_ncu_20260831.sh
ARCHIVE_SCRIPT=$(readlink -f -- "${BASH_SOURCE[0]}")
OUTPUTS=${JOB_DIR}/outputs-manifest-job${C2_NCU_JOB_ID}.sha256
FINAL=${JOB_DIR}/final-status-job${C2_NCU_JOB_ID}.txt
FINAL_SIDECAR=${JOB_DIR}/final-status-job${C2_NCU_JOB_ID}.sha256
RESULT=${JOB_DIR}/v12-v14-v15-ncu-mechanism-job${C2_NCU_JOB_ID}.json
MANIFEST=${BASE}/c2-native-v12-v14-v15-ncu-job${C2_NCU_JOB_ID}-evidence-20260831.manifest.sha256
ARCHIVE=${BASE}/c2-native-v12-v14-v15-ncu-job${C2_NCU_JOB_ID}-evidence-20260831.tar.gz
SIDECAR=${ARCHIVE}.sha256
LOCK=${BASE}/.c2-native-v12-v14-v15-ncu-job${C2_NCU_JOB_ID}-archive-20260831.lock
MANIFEST_TMP=${MANIFEST}.tmp.${BASHPID}
ARCHIVE_TMP=${ARCHIVE}.tmp.${BASHPID}
SIDECAR_TMP=${SIDECAR}.tmp.${BASHPID}

lock_acquired=0
published=0
manifest_linked=0
archive_linked=0
sidecar_linked=0
cleanup_outputs_and_lock() {
  local original_rc=$?
  trap - EXIT
  set +e
  if (( original_rc != 0 && published == 0 )); then
    if (( sidecar_linked == 1 )) && [[ -f "${SIDECAR}" && -f "${SIDECAR_TMP}" && "${SIDECAR}" -ef "${SIDECAR_TMP}" ]]; then
      rm -f -- "${SIDECAR}"
    fi
    if (( archive_linked == 1 )) && [[ -f "${ARCHIVE}" && -f "${ARCHIVE_TMP}" && "${ARCHIVE}" -ef "${ARCHIVE_TMP}" ]]; then
      rm -f -- "${ARCHIVE}"
    fi
    if (( manifest_linked == 1 )) && [[ -f "${MANIFEST}" && -f "${MANIFEST_TMP}" && "${MANIFEST}" -ef "${MANIFEST_TMP}" ]]; then
      rm -f -- "${MANIFEST}"
    fi
  fi
  for temporary in "${MANIFEST_TMP}" "${ARCHIVE_TMP}" "${SIDECAR_TMP}"; do
    [[ ! -e "${temporary}" && ! -L "${temporary}" ]] ||
      [[ -f "${temporary}" && ! -L "${temporary}" ]] || continue
    rm -f -- "${temporary}"
  done
  if (( lock_acquired == 1 )); then
    rmdir -- "${LOCK}"
  fi
  exit "${original_rc}"
}

[[ -d "${BASE}" && ! -L "${BASE}" && "$(readlink -f -- "${BASE}")" == "${BASE}" &&
   "$(stat -c %u -- "${BASE}")" == "$(id -u)" ]]
[[ "${BASH_SOURCE[0]}" == "${EXPECTED_ARCHIVE_SCRIPT}" &&
   "${ARCHIVE_SCRIPT}" == "${EXPECTED_ARCHIVE_SCRIPT}" ]]
[[ -d "${ROOT}" && ! -L "${ROOT}" && "$(readlink -f -- "${ROOT}")" == "${ROOT}" ]]
[[ -d "${JOB_DIR}" && ! -L "${JOB_DIR}" && "$(readlink -f -- "${JOB_DIR}")" == "${JOB_DIR}" ]]
for input in "${SLURM_LOG}" "${PROFILE_SCRIPT}" "${ARCHIVE_SCRIPT}" \
             "${OUTPUTS}" "${FINAL}" "${FINAL_SIDECAR}" "${RESULT}"; do
  [[ -f "${input}" && ! -L "${input}" ]]
done
mkdir -m 0700 -- "${LOCK}"
lock_acquired=1
trap cleanup_outputs_and_lock EXIT
for output in "${MANIFEST}" "${ARCHIVE}" "${SIDECAR}"; do
  [[ ! -e "${output}" && ! -L "${output}" ]] || {
    printf 'refusing to reuse archive output: %s\n' "${output}" >&2
    exit 2
  }
done
for temporary in "${MANIFEST_TMP}" "${ARCHIVE_TMP}" "${SIDECAR_TMP}"; do
  [[ ! -e "${temporary}" && ! -L "${temporary}" ]]
done
[[ -z "$(find "${JOB_DIR}" -mindepth 1 -maxdepth 1 ! -type f -print -quit)" ]]

printf '%s  %s\n' \
  "${C2_EXPECTED_ARCHIVE_SCRIPT_SHA}" "${ARCHIVE_SCRIPT}" \
  "${C2_EXPECTED_PROFILE_SCRIPT_SHA}" "${PROFILE_SCRIPT}" | sha256sum -c -
[[ -z "$(squeue -h -j "${C2_NCU_JOB_ID}" -o '%i')" ]]
scheduler_state=$(scontrol show job -o "${C2_NCU_JOB_ID}")
[[ -n "${scheduler_state}" ]]
scheduler_state=" ${scheduler_state} "
[[ "${scheduler_state}" == *" JobId=${C2_NCU_JOB_ID} "* &&
   "${scheduler_state}" == *" JobState=COMPLETED "* &&
   "${scheduler_state}" == *" ExitCode=0:0 "* ]]

sha256sum -c "${OUTPUTS}" >/dev/null
sha256sum -c "${FINAL_SIDECAR}" >/dev/null
grep -Eq '^FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 TEE_RC=0 RUNTIME_CLEANUP_RC=0 MANIFEST_RC=0 EXPECTED_UUID=GPU-[0-9A-Fa-f-]+ POST_UUID=GPU-[0-9A-Fa-f-]+ POST_APPS_EMPTY=true UTC=' "${FINAL}"

python3 - "${RESULT}" "${FINAL}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
status = Path(sys.argv[2]).read_text().strip()
assert result.get("schema") == "c2-native-v12-v14-v15-ncu-mechanism-v1"
assert result.get("all_collection_and_comparison_gates_pass") is True
assert result.get("timing_valid_for_benchmark") is False
assert result.get("duration_metric_requested") is False
contract = result["same_collection_contract"]
assert contract["seed"] == 20260829
assert contract["separate_python_processes"] is True
assert contract["single_native_kernel_action_each"] is True
assert contract["counter_catalog_size"] == 15
uuid = contract["same_physical_gpu_uuid"]
assert re.fullmatch(r"GPU-[0-9A-Fa-f-]+", uuid)
assert f"EXPECTED_UUID={uuid}" in status and f"POST_UUID={uuid}" in status
absolute = result["absolute_counters"]
assert set(absolute) == {"v12", "v14", "v15"}
keys = set(absolute["v12"])
assert len(keys) == 15 and all(set(absolute[v]) == keys for v in absolute)
for key in keys:
    units = {absolute[v][key]["unit"] for v in absolute}
    assert len(units) == 1
    for version in absolute:
        value = absolute[version][key]["value"]
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(float(value)) and float(value) >= 0
ratios = result["counter_ratios"]
assert set(ratios) == {"v14_over_v12", "v15_over_v12"}
for row in ratios.values():
    assert set(row) == keys
    assert all(value is None or math.isfinite(float(value)) and float(value) >= 0
               for value in row.values())
context = result["historical_context"]
assert len(context) == 3
assert [row["accepted"] for row in context] == [True, False, False]
assert all(row["correctness_identity_gates_pass"] is True for row in context)
PY

(
  cd "${BASE}"
  {
    find "$(basename "${ROOT}")/job${C2_NCU_JOB_ID}" -mindepth 1 -maxdepth 1 -type f -print0
    printf '%s\0' \
      "$(basename "${ROOT}")/slurm-${C2_NCU_JOB_ID}.log" \
      "$(basename "${PROFILE_SCRIPT}")" \
      "$(basename "${ARCHIVE_SCRIPT}")"
  } | sort -z | xargs -0 -r sha256sum
) > "${MANIFEST_TMP}"
records=$(wc -l < "${MANIFEST_TMP}")
(( records >= 20 ))
(cd "${BASE}" && sha256sum -c "${MANIFEST_TMP}" >/dev/null)
ln -- "${MANIFEST_TMP}" "${MANIFEST}"
manifest_linked=1

{
  sed -nE 's/^[0-9a-f]{64}  //p' "${MANIFEST_TMP}"
  basename "${MANIFEST}"
} | tar -C "${BASE}" --format=posix --sort=name --numeric-owner --owner=0 --group=0 \
  --no-recursion -czf "${ARCHIVE_TMP}" --files-from=-
[[ -s "${ARCHIVE_TMP}" ]]
[[ -z "$(tar -tzf "${ARCHIVE_TMP}" | grep -E '(^/|(^|/)\.\.(/|$))' || true)" ]]
[[ -z "$(tar -tzf "${ARCHIVE_TMP}" | sort | uniq -d)" ]]
members=$(tar -tzf "${ARCHIVE_TMP}" | wc -l)
regular_members=$(tar -tvzf "${ARCHIVE_TMP}" | awk 'substr($1,1,1)=="-" {count++} END {print count+0}')
(( members == records + 1 && regular_members == records + 1 ))

archive_sha=$(sha256sum "${ARCHIVE_TMP}" | awk '{print $1}')
manifest_sha=$(sha256sum "${MANIFEST_TMP}" | awk '{print $1}')
printf '%s  %s\n%s  %s\n' "${archive_sha}" "${ARCHIVE}" "${manifest_sha}" "${MANIFEST}" > "${SIDECAR_TMP}"
ln -- "${ARCHIVE_TMP}" "${ARCHIVE}"
archive_linked=1
ln -- "${SIDECAR_TMP}" "${SIDECAR}"
sidecar_linked=1
sha256sum -c "${SIDECAR}"
published=1
printf 'ARCHIVE=%s\nMANIFEST=%s\nSIDECAR=%s\nRECORDS=%s\nMEMBERS=%s\n' \
  "${ARCHIVE}" "${MANIFEST}" "${SIDECAR}" "${records}" "${members}"
