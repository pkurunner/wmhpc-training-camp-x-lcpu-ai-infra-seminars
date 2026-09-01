#!/usr/bin/env bash
# Freeze an exact, quiescent, locally re-verifiable B300 continuation evidence set.
set -Eeuo pipefail

HOME_ROOT=/home/lcpu/85117379
RECOVERY_LOCK=${HOME_ROOT}/.c2-native-v11-hardened-recovery-20260830.lock
MANIFEST_NAME=c2-native-v7-v11-continuation-hardened-evidence-20260830.sha256
ARCHIVE_NAME=c2-native-v7-v11-continuation-hardened-evidence-20260830.tar
ARCHIVE_SIDECAR_NAME=${ARCHIVE_NAME}.sha256
INVENTORY_NAME=c2-native-v7-v11-continuation-hardened-root-inventory-20260830.txt
STAGING=$(mktemp -d "${HOME_ROOT}/.c2-native-v7-v11-hardened.XXXXXX")
LIST_FILE=${STAGING}/files.nul
EXPECTED_MEMBERS=${STAGING}/expected-members.txt
ACTUAL_MEMBERS=${STAGING}/actual-members.txt
MANIFEST_STAGE=${STAGING}/${MANIFEST_NAME}
ARCHIVE_STAGE=${STAGING}/${ARCHIVE_NAME}
SIDECAR_STAGE=${STAGING}/${ARCHIVE_SIDECAR_NAME}
INVENTORY_STAGE=${STAGING}/${INVENTORY_NAME}

cleanup() {
  rm -f -- "${LIST_FILE}" "${EXPECTED_MEMBERS}" "${ACTUAL_MEMBERS}" \
    "${MANIFEST_STAGE}" "${ARCHIVE_STAGE}" "${SIDECAR_STAGE}" "${INVENTORY_STAGE}"
  rmdir -- "${STAGING}" 2>/dev/null || true
}
trap cleanup EXIT

command -v flock >/dev/null
exec 9>"${RECOVERY_LOCK}"
flock -n 9 || { echo 'v11 recovery/archive lock is already held' >&2; exit 2; }

ROOTS=(
  c2-native-plugin-v7-stress-3pct-artifacts-20260830
  c2-native-plugin-v9-aot-artifacts-20260830
  c2-native-plugin-v9-directed-artifacts-20260830
  c2-native-plugin-v9-stress-3pct-artifacts-20260830
  c2-native-plugin-v9-stress-3pct-retry1-artifacts-20260830
  c2-native-plugin-v10-aot-artifacts-20260830
  c2-native-plugin-v10-directed-artifacts-20260830
  c2-native-plugin-v10-stress-3pct-artifacts-20260830
  c2-native-plugin-v9b-aot-artifacts-20260830
  c2-native-plugin-v9b-directed-artifacts-20260830
  c2-native-plugin-v9b-stress-3pct-artifacts-20260830
  c2-native-plugin-v11-aot-artifacts-20260830
  c2-native-plugin-v11-aot-prehardening-job12793-artifacts-20260830
  c2-native-plugin-v11-directed-failure-audit-job12829-artifacts-20260830
  c2-native-plugin-v11-directed-artifacts-20260830
  c2-native-plugin-v11-stress-3pct-artifacts-20260830
)

EXTRAS=(
  build_native_c2_plugin_v11_q_fragment_reuse_aot.slurm
  validate_native_c2_plugin_v11_q_fragment_reuse_directed.slurm
  validate_native_c2_plugin_v11_q_fragment_reuse_stress_perf_3pct.slurm
  native_c2_v11_q_fragment_reuse_20260830.patch
  continue_native_c2_v11_hardened_20260830.sh
  archive_native_c2_v7_v11_continuation_hardened_20260830.sh
  c2-native-plugin-v11-aot-job-id-20260830.txt
  c2-native-plugin-v11-directed-job-id-20260830.txt
  c2-native-plugin-v11-stress-3pct-job-id-20260830.txt
  slurm-c2-native-plugin-v9b-stress-3pct-12795.log
)

FINAL_STATUSES=(
  c2-native-plugin-v7-stress-3pct-artifacts-20260830/job12599/final-status-job12599.txt
  c2-native-plugin-v9-aot-artifacts-20260830/job12701/final-status-job12701.txt
  c2-native-plugin-v9-directed-artifacts-20260830/job12767/final-status-job12767.txt
  c2-native-plugin-v9-stress-3pct-artifacts-20260830/job12772/final-status-job12772.txt
  c2-native-plugin-v9-stress-3pct-retry1-artifacts-20260830/job12776/final-status-job12776.txt
  c2-native-plugin-v10-aot-artifacts-20260830/job12702/final-status-job12702.txt
  c2-native-plugin-v10-directed-artifacts-20260830/job12783/final-status-job12783.txt
  c2-native-plugin-v10-stress-3pct-artifacts-20260830/job12784/final-status-job12784.txt
  c2-native-plugin-v9b-aot-artifacts-20260830/job12790/final-status-job12790.txt
  c2-native-plugin-v9b-directed-artifacts-20260830/job12794/final-status-job12794.txt
  c2-native-plugin-v11-aot-prehardening-job12793-artifacts-20260830/job12793/final-status-job12793.txt
  c2-native-plugin-v11-aot-artifacts-20260830/job12825/final-status-job12825.txt
  c2-native-plugin-v11-directed-failure-audit-job12829-artifacts-20260830/job12829/final-status-job12829.txt
  c2-native-plugin-v11-directed-artifacts-20260830/job12904/final-status-job12904.txt
  c2-native-plugin-v11-stress-3pct-artifacts-20260830/job12905/final-status-job12905.txt
)

JOBS=(12599 12701 12702 12767 12772 12776 12783 12784 12790 12793 12794 12795 12825 12829 12904 12905)

cd "${HOME_ROOT}"
[[ ! -e "${MANIFEST_NAME}" && ! -e "${ARCHIVE_NAME}" && ! -e "${ARCHIVE_SIDECAR_NAME}" ]]
for root in "${ROOTS[@]}"; do
  [[ -d "${root}" && ! -L "${root}" ]] || {
    echo "evidence root is missing, not a directory, or a symlink: ${root}" >&2
    exit 2
  }
done
for path in "${EXTRAS[@]}" "${FINAL_STATUSES[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || {
    echo "required evidence file is missing, non-regular, or a symlink: ${path}" >&2
    exit 2
  }
done

queue_ids=$(squeue -h -u 85117379 -o %i | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; /^[[:space:]]*$/d')
for job in "${JOBS[@]}"; do
  if grep -Fxq "${job}" <<< "${queue_ids}"; then
    echo "evidence job is still active: ${job}" >&2
    exit 2
  fi
done

special_path=$(find "${ROOTS[@]}" -mindepth 1 ! -type d ! -type f -print -quit)
[[ -z "${special_path}" ]] || { echo "unsupported non-regular evidence member: ${special_path}" >&2; exit 2; }

{
  for root in "${ROOTS[@]}"; do
    find "${root}" -type f -print0
  done
  printf '%s\0' "${EXTRAS[@]}"
} | sort -z -u > "${LIST_FILE}"

mapfile -d '' -t files < "${LIST_FILE}"
(( ${#files[@]} > 0 ))
for path in "${files[@]}"; do
  [[ "${path}" != *$'\n'* && -f "${path}" && ! -L "${path}" ]]
done

{
  printf 'generated_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'root=%s\n' "${ROOTS[@]}"
  printf 'extra=%s\n' "${EXTRAS[@]}"
  printf 'job=%s\n' "${JOBS[@]}"
  printf 'regular_file_count=%s\n' "${#files[@]}"
} > "${INVENTORY_STAGE}"

sha256sum -- "${files[@]}" > "${MANIFEST_STAGE}"
sha256sum -c --quiet "${MANIFEST_STAGE}"

tar --create --file="${ARCHIVE_STAGE}" --no-recursion --directory="${HOME_ROOT}" \
  --null --files-from="${LIST_FILE}" --directory="${STAGING}" \
  "${MANIFEST_NAME}" "${INVENTORY_NAME}"

tr '\0' '\n' < "${LIST_FILE}" > "${EXPECTED_MEMBERS}"
printf '%s\n%s\n' "${MANIFEST_NAME}" "${INVENTORY_NAME}" >> "${EXPECTED_MEMBERS}"
LC_ALL=C sort -u -o "${EXPECTED_MEMBERS}" "${EXPECTED_MEMBERS}"
tar -tf "${ARCHIVE_STAGE}" | LC_ALL=C sort -u > "${ACTUAL_MEMBERS}"
cmp -- "${EXPECTED_MEMBERS}" "${ACTUAL_MEMBERS}"
sha256sum -c --quiet "${MANIFEST_STAGE}"

archive_sha=$(sha256sum -- "${ARCHIVE_STAGE}" | awk '{print $1}')
printf '%s  %s\n' "${archive_sha}" "${ARCHIVE_NAME}" > "${SIDECAR_STAGE}"

# Hard links publish without clobber; the sidecar is the final commit marker.
ln -- "${ARCHIVE_STAGE}" "${HOME_ROOT}/${ARCHIVE_NAME}"
ln -- "${MANIFEST_STAGE}" "${HOME_ROOT}/${MANIFEST_NAME}"
ln -- "${SIDECAR_STAGE}" "${HOME_ROOT}/${ARCHIVE_SIDECAR_NAME}"

printf 'FILES=%s\n' "${#files[@]}"
du -h -- "${HOME_ROOT}/${ARCHIVE_NAME}"
cat -- "${HOME_ROOT}/${ARCHIVE_SIDECAR_NAME}"
