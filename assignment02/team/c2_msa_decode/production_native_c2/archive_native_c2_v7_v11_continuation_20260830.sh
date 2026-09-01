#!/usr/bin/env bash
# Freeze the complete B300 evidence roots for the >3% optimization continuation.
set -Eeuo pipefail

HOME_ROOT=/home/lcpu/85117379
MANIFEST_NAME=c2-native-v7-v11-continuation-evidence-20260830.sha256
ARCHIVE_NAME=c2-native-v7-v11-continuation-evidence-20260830.tar
ARCHIVE_SIDECAR_NAME=${ARCHIVE_NAME}.sha256
LIST_FILE=$(mktemp "${HOME_ROOT}/.c2-native-v7-v11-files.XXXXXX")
MANIFEST_TMP=$(mktemp "${HOME_ROOT}/.${MANIFEST_NAME}.XXXXXX")
ARCHIVE_TMP=$(mktemp "${HOME_ROOT}/.${ARCHIVE_NAME}.XXXXXX")

cleanup() {
  rm -f -- "${LIST_FILE}" "${MANIFEST_TMP}" "${ARCHIVE_TMP}"
}
trap cleanup EXIT

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
  archive_native_c2_v7_v11_continuation_20260830.sh
  c2-native-plugin-v11-aot-job-id-20260830.txt
  c2-native-plugin-v11-directed-job-id-20260830.txt
  c2-native-plugin-v11-stress-3pct-job-id-20260830.txt
)

cd "${HOME_ROOT}"
[[ ! -e "${MANIFEST_NAME}" && ! -e "${ARCHIVE_NAME}" && ! -e "${ARCHIVE_SIDECAR_NAME}" ]]
for path in "${ROOTS[@]}" "${EXTRAS[@]}"; do
  [[ -e "${path}" ]] || { echo "required evidence path missing: ${path}" >&2; exit 2; }
done

{
  for root in "${ROOTS[@]}"; do
    find "${root}" -type f -print0
  done
  printf '%s\0' "${EXTRAS[@]}"
} | sort -z -u > "${LIST_FILE}"

mapfile -d '' -t files < "${LIST_FILE}"
(( ${#files[@]} > 0 ))
sha256sum -- "${files[@]}" > "${MANIFEST_TMP}"
sha256sum -c --quiet "${MANIFEST_TMP}"
mv -T -- "${MANIFEST_TMP}" "${MANIFEST_NAME}"

tar -cf "${ARCHIVE_TMP}" -- "${ROOTS[@]}" "${EXTRAS[@]}" "${MANIFEST_NAME}"
sha256sum -c --quiet "${MANIFEST_NAME}"
mv -T -- "${ARCHIVE_TMP}" "${ARCHIVE_NAME}"
sha256sum -- "${ARCHIVE_NAME}" > "${ARCHIVE_SIDECAR_NAME}"

printf 'FILES=%s\n' "${#files[@]}"
du -h -- "${ARCHIVE_NAME}"
cat -- "${ARCHIVE_SIDECAR_NAME}"
