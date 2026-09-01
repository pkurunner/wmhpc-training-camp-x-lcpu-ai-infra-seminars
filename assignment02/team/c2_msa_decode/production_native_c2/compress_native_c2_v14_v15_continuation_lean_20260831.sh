#!/usr/bin/env bash
# Create a locally transferable lean copy of the complete v14/v15 evidence.
# Only the two immutable 8-seed fixture payloads are omitted; their SHA-256
# records remain in both the full manifest and an explicit exclusion list.

set -Eeuo pipefail
umask 077

: "${C2_EXPECTED_LEAN_SCRIPT_SHA:?set reviewed compressor script SHA-256}"
[[ "${C2_EXPECTED_LEAN_SCRIPT_SHA}" =~ ^[0-9a-f]{64}$ ]]

BASE=/home/lcpu/85117379
FULL_ARCHIVE=${BASE}/c2-native-v14-v15-continuation-evidence-20260831.tar.gz
FULL_MANIFEST=${BASE}/c2-native-v14-v15-continuation-evidence-20260831.manifest.sha256
FULL_SIDECAR=${FULL_ARCHIVE}.sha256
EXCLUDED=${BASE}/c2-native-v14-v15-continuation-excluded-fixtures-20260831.sha256
LEAN_MANIFEST=${BASE}/c2-native-v14-v15-continuation-lean-evidence-20260831.manifest.sha256
LEAN_ARCHIVE=${BASE}/c2-native-v14-v15-continuation-lean-evidence-20260831.tar.gz
LEAN_SIDECAR=${LEAN_ARCHIVE}.sha256
SCRIPT=$(readlink -f -- "${BASH_SOURCE[0]}")

EXPECTED_FULL_ARCHIVE_SHA=0a5260556ad189ec0b0b9c405fa0b5f10aaa0f84d98ab4ab9fa0ca0d21301458
EXPECTED_FULL_MANIFEST_SHA=7948c961adf92642e29fd8c053399a5a8c3ea311735aaeceba610cede3fb378e
EXPECTED_FULL_SIDECAR_SHA=858194dbe17a2e1a3b169d2304a4c8ce50adb984cee0074648fbbcb57ffd8efc

[[ -d "${BASE}" && ! -L "${BASE}" && "$(readlink -f -- "${BASE}")" == "${BASE}" &&
   "$(stat -c %u -- "${BASE}")" == "$(id -u)" ]]
for input in "${FULL_ARCHIVE}" "${FULL_MANIFEST}" "${FULL_SIDECAR}" "${SCRIPT}"; do
  [[ -f "${input}" && ! -L "${input}" ]]
done
for output in "${EXCLUDED}" "${LEAN_MANIFEST}" "${LEAN_ARCHIVE}" "${LEAN_SIDECAR}"; do
  [[ ! -e "${output}" && ! -L "${output}" ]] || {
    printf 'refusing to reuse lean output: %s\n' "${output}" >&2
    exit 2
  }
done

published=0
cleanup_partial_outputs() {
  local original_rc=$?
  trap - EXIT
  set +e
  if (( original_rc != 0 && published == 0 )); then
    for partial in "${EXCLUDED}" "${LEAN_MANIFEST}" "${LEAN_ARCHIVE}" "${LEAN_SIDECAR}"; do
      [[ ! -e "${partial}" && ! -L "${partial}" ]] ||
        [[ -f "${partial}" && ! -L "${partial}" ]] || continue
      rm -f -- "${partial}"
    done
  fi
  exit "${original_rc}"
}
trap cleanup_partial_outputs EXIT

printf '%s  %s\n' "${C2_EXPECTED_LEAN_SCRIPT_SHA}" "${SCRIPT}" | sha256sum -c -
printf '%s  %s\n' \
  "${EXPECTED_FULL_ARCHIVE_SHA}" "${FULL_ARCHIVE}" \
  "${EXPECTED_FULL_MANIFEST_SHA}" "${FULL_MANIFEST}" \
  "${EXPECTED_FULL_SIDECAR_SHA}" "${FULL_SIDECAR}" | sha256sum -c -
(cd "${BASE}" && sha256sum -c "$(basename "${FULL_SIDECAR}")")
[[ $(wc -l < "${FULL_MANIFEST}") -eq 159 ]]

grep '/fixtures-8-seeds\.pt$' "${FULL_MANIFEST}" > "${EXCLUDED}"
[[ $(wc -l < "${EXCLUDED}") -eq 2 ]]
grep -F 'c2-native-plugin-v14-kv-stage-padding-stress-3pct-retry1-artifacts-20260831/job13539/fixtures-8-seeds.pt' "${EXCLUDED}" >/dev/null
grep -F 'c2-native-plugin-v15-q-stage-stride144-stress-3pct-artifacts-20260831/job13576/fixtures-8-seeds.pt' "${EXCLUDED}" >/dev/null

(
  cd "${BASE}"
  awk '!/\/fixtures-8-seeds\.pt$/' "$(basename "${FULL_MANIFEST}")"
  sha256sum "$(basename "${FULL_MANIFEST}")" "$(basename "${FULL_SIDECAR}")" \
    "$(basename "${EXCLUDED}")" "$(basename "${SCRIPT}")"
) > "${LEAN_MANIFEST}"
[[ $(wc -l < "${LEAN_MANIFEST}") -eq 161 ]]
(cd "${BASE}" && sha256sum -c "$(basename "${LEAN_MANIFEST}")" >/dev/null)

while IFS= read -r path; do
  [[ "${path}" != /* && "${path}" != '..' && "${path}" != ../* && "${path}" != */../* ]]
  [[ -f "${BASE}/${path}" && ! -L "${BASE}/${path}" ]]
done < <(sed -nE 's/^[0-9a-f]{64}  //p' "${LEAN_MANIFEST}")

{
  sed -nE 's/^[0-9a-f]{64}  //p' "${LEAN_MANIFEST}"
  basename "${LEAN_MANIFEST}"
} | tar -C "${BASE}" --format=posix --sort=name --numeric-owner --owner=0 --group=0 \
  --no-recursion -czf "${LEAN_ARCHIVE}" --files-from=-
[[ -s "${LEAN_ARCHIVE}" ]]
[[ -z "$(tar -tzf "${LEAN_ARCHIVE}" | grep -E '(^/|(^|/)\.\.(/|$))' || true)" ]]
[[ -z "$(tar -tzf "${LEAN_ARCHIVE}" | sort | uniq -d)" ]]
records=$(wc -l < "${LEAN_MANIFEST}")
members=$(tar -tzf "${LEAN_ARCHIVE}" | wc -l)
regular_members=$(tar -tvzf "${LEAN_ARCHIVE}" | awk 'substr($1,1,1)=="-" {count++} END {print count+0}')
(( members == records + 1 && regular_members == records + 1 ))

sha256sum "${LEAN_ARCHIVE}" "${LEAN_MANIFEST}" "${EXCLUDED}" "${FULL_MANIFEST}" "${FULL_SIDECAR}" > "${LEAN_SIDECAR}"
sha256sum -c "${LEAN_SIDECAR}"
published=1
printf 'LEAN_ARCHIVE=%s\nLEAN_MANIFEST=%s\nEXCLUDED=%s\nLEAN_SIDECAR=%s\nRECORDS=%s\nMEMBERS=%s\n' \
  "${LEAN_ARCHIVE}" "${LEAN_MANIFEST}" "${EXCLUDED}" "${LEAN_SIDECAR}" "${records}" "${members}"
