#!/usr/bin/env bash
# Freeze the successful, read-only v12/v16 NCU mechanism collection.  NCU
# counters explain a mechanism only; this archive neither measures nor changes
# any performance-acceptance decision.
#
# Run only after copying this reviewed file to the exact remote path below:
#   C2_EXPECTED_ARCHIVE_SCRIPT_SHA=<this-file-sha256> \
#     bash /home/lcpu/85117379/archive_native_c2_v12_v16_ncu_20260831.sh

set -Eeuo pipefail
umask 077

: "${C2_EXPECTED_ARCHIVE_SCRIPT_SHA:?set the reviewed archive script SHA-256}"
[[ "${C2_EXPECTED_ARCHIVE_SCRIPT_SHA}" =~ ^[0-9a-f]{64}$ ]]

BASE=/home/lcpu/85117379
JOB_ID=13868
ROOT=${BASE}/c2-native-plugin-v12-v16-ncu-artifacts-20260831
JOB_DIR=${ROOT}/job${JOB_ID}
SLURM_LOG=${ROOT}/slurm-${JOB_ID}.log
PROFILE_SCRIPT=${BASE}/profile_native_c2_v12_v16_ncu_20260831.slurm
EXPECTED_ARCHIVE_SCRIPT=${BASE}/archive_native_c2_v12_v16_ncu_20260831.sh
ARCHIVE_SCRIPT=$(readlink -f -- "${BASH_SOURCE[0]}")
OUTPUTS=${JOB_DIR}/outputs-manifest-job${JOB_ID}.sha256
FINAL=${JOB_DIR}/final-status-job${JOB_ID}.txt
FINAL_SIDECAR=${JOB_DIR}/final-status-job${JOB_ID}.sha256
RESULT=${JOB_DIR}/v12-v16-ncu-mechanism-job${JOB_ID}.json

EXPECTED_PROFILE_SCRIPT_SHA=ab179715d1b218263b66ed100c298574eecc3874cb359ed6bb0d293d9b72cf62
EXPECTED_RESULT_SHA=37cae5eabac16afbaf097fd3be66cc1a4c6c509e724de0877dc460acfdcbe4aa
EXPECTED_OUTPUTS_SHA=bb11fd954e5e9d917ae2029a5619778fde2fe4360ee8fd63e51f92883cf7897c
EXPECTED_FINAL_SHA=0a10cfac5b82706528d0de572dcaef81e8d0fb0ace381a6e85042b3e9f0ddfca
EXPECTED_FINAL_SIDECAR_SHA=1aba96eaefa68879052b9c63f4aa0d9fb772e0bdaae817c54ee3a8f36cc8d8eb

MANIFEST=${BASE}/c2-native-v12-v16-ncu-job${JOB_ID}-evidence-20260831.manifest.sha256
ARCHIVE=${BASE}/c2-native-v12-v16-ncu-job${JOB_ID}-evidence-20260831.tar.gz
SIDECAR=${ARCHIVE}.sha256
LOCK=${BASE}/.c2-native-v12-v16-ncu-job${JOB_ID}-archive-20260831.lock
MANIFEST_TMP=${MANIFEST}.tmp.${BASHPID}
ARCHIVE_TMP=${ARCHIVE}.tmp.${BASHPID}
SIDECAR_TMP=${SIDECAR}.tmp.${BASHPID}
EXPECTED_MEMBERS_TMP=${ARCHIVE}.expected-members.tmp.${BASHPID}
ACTUAL_MEMBERS_TMP=${ARCHIVE}.actual-members.tmp.${BASHPID}
ARCHIVE_MEMBER_HASHES_TMP=${ARCHIVE}.member-hashes.tmp.${BASHPID}

lock_acquired=0
published=0
manifest_linked=0
archive_linked=0
sidecar_linked=0
cleanup_outputs_and_lock() {
  local original_rc=$?
  trap - EXIT
  set +e
  # A published sidecar is the commit marker.  Before that point, remove only
  # hard links that this invocation itself created; never replace or remove a
  # pre-existing evidence artifact.
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
  for temporary in "${MANIFEST_TMP}" "${ARCHIVE_TMP}" "${SIDECAR_TMP}" "${EXPECTED_MEMBERS_TMP}" "${ACTUAL_MEMBERS_TMP}" "${ARCHIVE_MEMBER_HASHES_TMP}"; do
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
[[ -d "${ROOT}" && ! -L "${ROOT}" && "$(readlink -f -- "${ROOT}")" == "${ROOT}" &&
   "$(stat -c %u -- "${ROOT}")" == "$(id -u)" ]]
[[ -d "${JOB_DIR}" && ! -L "${JOB_DIR}" && "$(readlink -f -- "${JOB_DIR}")" == "${JOB_DIR}" &&
   "$(stat -c %u -- "${JOB_DIR}")" == "$(id -u)" ]]
for input in "${SLURM_LOG}" "${PROFILE_SCRIPT}" "${ARCHIVE_SCRIPT}" \
             "${OUTPUTS}" "${FINAL}" "${FINAL_SIDECAR}" "${RESULT}"; do
  [[ -f "${input}" && ! -L "${input}" ]]
done

# The job directory is deliberately flat: exactly the 22 completed-job
# regular files, no nested directory, symlink, FIFO, device, or dangling link.
[[ -z "$(find "${JOB_DIR}" -mindepth 1 ! -type f -print -quit)" ]]
[[ -z "$(find "${JOB_DIR}" -mindepth 1 -type l -print -quit)" ]]
job_file_count=$(find "${JOB_DIR}" -mindepth 1 -maxdepth 1 -type f -print | wc -l)
(( job_file_count == 22 ))

mkdir -m 0700 -- "${LOCK}"
lock_acquired=1
trap cleanup_outputs_and_lock EXIT
for output in "${MANIFEST}" "${ARCHIVE}" "${SIDECAR}"; do
  [[ ! -e "${output}" && ! -L "${output}" ]] || {
    printf 'refusing to reuse archive output: %s\n' "${output}" >&2
    exit 2
  }
done
for temporary in "${MANIFEST_TMP}" "${ARCHIVE_TMP}" "${SIDECAR_TMP}" "${EXPECTED_MEMBERS_TMP}" "${ACTUAL_MEMBERS_TMP}" "${ARCHIVE_MEMBER_HASHES_TMP}"; do
  [[ ! -e "${temporary}" && ! -L "${temporary}" ]]
done

printf '%s  %s\n' \
  "${C2_EXPECTED_ARCHIVE_SCRIPT_SHA}" "${ARCHIVE_SCRIPT}" \
  "${EXPECTED_PROFILE_SCRIPT_SHA}" "${PROFILE_SCRIPT}" \
  "${EXPECTED_RESULT_SHA}" "${RESULT}" \
  "${EXPECTED_OUTPUTS_SHA}" "${OUTPUTS}" \
  "${EXPECTED_FINAL_SHA}" "${FINAL}" \
  "${EXPECTED_FINAL_SIDECAR_SHA}" "${FINAL_SIDECAR}" | sha256sum -c -

if queue_state=$(squeue -h -j "${JOB_ID}" -o '%i' 2>&1); then
  [[ -z "${queue_state}" ]]
else
  [[ "${queue_state}" == 'slurm_load_jobs error: Invalid job id specified' ]]
fi
if scheduler_state=$(scontrol show job -o "${JOB_ID}" 2>&1); then
  scheduler_state=" ${scheduler_state} "
  [[ "${scheduler_state}" == *" JobId=${JOB_ID} "* &&
     "${scheduler_state}" == *" JobState=COMPLETED "* &&
     "${scheduler_state}" == *" ExitCode=0:0 "* ]]
else
  # A controller may purge a completed job before slurmdbd is reachable.  The
  # immutable final-status, output manifests, result schema, and GPU gates
  # below still have to pass before any evidence is made visible.
  [[ "${scheduler_state}" == 'slurm_load_jobs error: Invalid job id specified' ]]
fi

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
assert result.get("schema") == "c2-native-v12-v16-ncu-mechanism-v1"
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
assert set(absolute) == {"v12", "v16"}
keys = set(absolute["v12"])
assert len(keys) == 15 and all(set(absolute[version]) == keys for version in absolute)
for key in keys:
    units = {absolute[version][key]["unit"] for version in absolute}
    assert len(units) == 1
    for version in absolute:
        value = absolute[version][key]["value"]
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(float(value)) and float(value) >= 0
ratios = result["counter_ratios"]
assert set(ratios) == {"v16_over_v12"}
assert set(ratios["v16_over_v12"]) == keys
assert all(value is None or math.isfinite(float(value)) and float(value) >= 0
           for value in ratios["v16_over_v12"].values())
context = result["historical_context"]
assert len(context) == 2
assert [row["accepted"] for row in context] == [True, True]
assert all(row["correctness_identity_gates_pass"] is True for row in context)
PY

# The manifest records every archived regular member, in a deterministic order.
# It deliberately stays external to the tarball, so there is an exact one-to-one
# mapping between each manifest record and one safe tar regular-file member.
(
  cd "${BASE}"
  {
    find "$(basename "${ROOT}")/job${JOB_ID}" -mindepth 1 -maxdepth 1 -type f -print0
    printf '%s\0' \
      "$(basename "${ROOT}")/slurm-${JOB_ID}.log" \
      "$(basename "${PROFILE_SCRIPT}")" \
      "$(basename "${ARCHIVE_SCRIPT}")"
  } | LC_ALL=C sort -z | xargs -0 -r sha256sum
) > "${MANIFEST_TMP}"
records=$(wc -l < "${MANIFEST_TMP}")
(( records == 25 ))
(cd "${BASE}" && sha256sum -c "${MANIFEST_TMP}" >/dev/null)
sed -nE 's/^[0-9a-f]{64}  //p' "${MANIFEST_TMP}" > "${EXPECTED_MEMBERS_TMP}"
[[ $(wc -l < "${EXPECTED_MEMBERS_TMP}") -eq "${records}" ]]
[[ -z "$(LC_ALL=C sort "${EXPECTED_MEMBERS_TMP}" | uniq -d)" ]]

tar -C "${BASE}" --format=posix --sort=name --numeric-owner --owner=0 --group=0 \
  --no-recursion -czf "${ARCHIVE_TMP}" --files-from="${EXPECTED_MEMBERS_TMP}"
[[ -s "${ARCHIVE_TMP}" ]]
tar -tzf "${ARCHIVE_TMP}" > "${ACTUAL_MEMBERS_TMP}"
[[ -z "$(grep -E '(^/|(^|/)\.\.(/|$))' "${ACTUAL_MEMBERS_TMP}" || true)" ]]
[[ $(wc -l < "${ACTUAL_MEMBERS_TMP}") -eq "${records}" ]]
[[ -z "$(LC_ALL=C sort "${ACTUAL_MEMBERS_TMP}" | uniq -d)" ]]
cmp <(LC_ALL=C sort "${EXPECTED_MEMBERS_TMP}") <(LC_ALL=C sort "${ACTUAL_MEMBERS_TMP}")
tar -tvzf "${ARCHIVE_TMP}" | awk -v expected="${records}" '
  BEGIN { ok = 1 }
  { ++count; if (substr($1, 1, 1) != "-") ok = 0 }
  END { exit !(count == expected && ok) }
'
# Re-hash each decompressed member in manifest order.  This is deliberately
# stronger than checking member names/counts: every tar payload byte must map
# to exactly one already verified manifest record.
while IFS= read -r member; do
  member_sha=$(tar -xOf "${ARCHIVE_TMP}" -- "${member}" | sha256sum | awk '{print $1}')
  printf '%s  %s\n' "${member_sha}" "${member}"
done < "${EXPECTED_MEMBERS_TMP}" > "${ARCHIVE_MEMBER_HASHES_TMP}"
cmp "${MANIFEST_TMP}" "${ARCHIVE_MEMBER_HASHES_TMP}"

archive_sha=$(sha256sum "${ARCHIVE_TMP}" | awk '{print $1}')
manifest_sha=$(sha256sum "${MANIFEST_TMP}" | awk '{print $1}')
printf '%s  %s\n%s  %s\n' "${archive_sha}" "${ARCHIVE}" "${manifest_sha}" "${MANIFEST}" > "${SIDECAR_TMP}"

# All content and cross-checks above complete before the first visible output.
# Linking only into absent, fixed destinations prevents overwrites; the sidecar
# is the final commit marker consumers must require.
ln -- "${MANIFEST_TMP}" "${MANIFEST}"
manifest_linked=1
ln -- "${ARCHIVE_TMP}" "${ARCHIVE}"
archive_linked=1
ln -- "${SIDECAR_TMP}" "${SIDECAR}"
sidecar_linked=1
sha256sum -c "${SIDECAR}" >/dev/null
published=1
printf 'ARCHIVE=%s\nMANIFEST=%s\nSIDECAR=%s\nJOB_FILES=%s\nRECORDS=%s\nMEMBERS=%s\n' \
  "${ARCHIVE}" "${MANIFEST}" "${SIDECAR}" "${job_file_count}" "${records}" "${records}"
