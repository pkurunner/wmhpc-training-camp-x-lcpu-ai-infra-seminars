#!/usr/bin/env bash
# Derive a small exact-set local-transfer archive from the already frozen full tar.
set -Eeuo pipefail

HOME_ROOT=/home/lcpu/85117379
RECOVERY_LOCK=${HOME_ROOT}/.c2-native-v11-hardened-recovery-20260830.lock
FULL_ARCHIVE=c2-native-v7-v11-continuation-hardened-evidence-20260830.tar
FULL_SIDECAR=${FULL_ARCHIVE}.sha256
FULL_MANIFEST=c2-native-v7-v11-continuation-hardened-evidence-20260830.sha256
LEAN_ARCHIVE=c2-native-v7-v11-continuation-hardened-lean-evidence-20260830.tar
LEAN_SIDECAR=${LEAN_ARCHIVE}.sha256
LEAN_MANIFEST=c2-native-v7-v11-continuation-hardened-lean-evidence-20260830.sha256
LEAN_INVENTORY=c2-native-v7-v11-continuation-hardened-lean-inventory-20260830.txt
EXCLUDED_FIXTURES=c2-native-v7-v11-continuation-excluded-fixtures-20260830.sha256
LEAN_SCRIPT=archive_native_c2_v7_v11_continuation_lean_from_hardened_20260830.sh
STAGING=$(mktemp -d "${HOME_ROOT}/.c2-native-v7-v11-lean.XXXXXX")
EXTRACTED=${STAGING}/extracted
FILES_NUL=${STAGING}/files.nul
EXPECTED=${STAGING}/expected.txt
ACTUAL=${STAGING}/actual.txt

cleanup() {
  [[ "${STAGING}" == "${HOME_ROOT}"/.c2-native-v7-v11-lean.* ]]
  rm -rf -- "${STAGING}"
}
trap cleanup EXIT

command -v flock >/dev/null
exec 9>"${RECOVERY_LOCK}"
flock -n 9 || { echo 'v11 recovery/archive lock is already held' >&2; exit 2; }

cd "${HOME_ROOT}"
[[ -f "${FULL_ARCHIVE}" && ! -L "${FULL_ARCHIVE}" ]]
[[ -f "${FULL_SIDECAR}" && ! -L "${FULL_SIDECAR}" ]]
[[ -f "${FULL_MANIFEST}" && ! -L "${FULL_MANIFEST}" ]]
[[ -f "${LEAN_SCRIPT}" && ! -L "${LEAN_SCRIPT}" ]]
[[ ! -e "${LEAN_ARCHIVE}" && ! -e "${LEAN_SIDECAR}" && ! -e "${LEAN_MANIFEST}" ]]
sha256sum -c --quiet "${FULL_SIDECAR}"

mkdir -- "${EXTRACTED}"
tar -xf "${FULL_ARCHIVE}" -C "${EXTRACTED}" --exclude='*/fixtures-8-seeds.pt'
cmp -- "${FULL_MANIFEST}" "${EXTRACTED}/${FULL_MANIFEST}"
grep -E 'fixtures-8-seeds\.pt$' "${FULL_MANIFEST}" > "${STAGING}/${EXCLUDED_FIXTURES}"
[[ "$(wc -l < "${STAGING}/${EXCLUDED_FIXTURES}")" -eq 4 ]]
while IFS= read -r line; do
  fixture=${line#*  }
  [[ "${fixture}" != "${line}" && ! -e "${EXTRACTED}/${fixture}" ]]
done < "${STAGING}/${EXCLUDED_FIXTURES}"

special=$(find "${EXTRACTED}" -mindepth 1 ! -type d ! -type f -print -quit)
[[ -z "${special}" ]] || { echo "unsupported extracted member: ${special}" >&2; exit 2; }
(cd "${EXTRACTED}" && find . -type f -printf '%P\0' | sort -z -u > "${FILES_NUL}")
mapfile -d '' -t files < "${FILES_NUL}"
(( ${#files[@]} > 0 ))
for path in "${files[@]}"; do
  [[ "${path}" != *$'\n'* && -f "${EXTRACTED}/${path}" && ! -L "${EXTRACTED}/${path}" ]]
done

(cd "${EXTRACTED}" && sha256sum -- "${files[@]}" > "${STAGING}/${LEAN_MANIFEST}")
(cd "${EXTRACTED}" && sha256sum -c --quiet "${STAGING}/${LEAN_MANIFEST}")
{
  printf 'generated_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'source_full_archive=%s\n' "${FULL_ARCHIVE}"
  printf 'source_full_archive_sha256=%s\n' "$(sha256sum "${FULL_ARCHIVE}" | awk '{print $1}')"
  printf 'included_regular_files=%s\n' "${#files[@]}"
  printf 'excluded_fixture_count=4\n'
} > "${STAGING}/${LEAN_INVENTORY}"
cp -- "${LEAN_SCRIPT}" "${STAGING}/${LEAN_SCRIPT}"
chmod a-w "${STAGING}/${LEAN_SCRIPT}"

tar --create --file="${STAGING}/${LEAN_ARCHIVE}" --no-recursion \
  --directory="${EXTRACTED}" --null --files-from="${FILES_NUL}" \
  --directory="${STAGING}" "${LEAN_MANIFEST}" "${LEAN_INVENTORY}" "${EXCLUDED_FIXTURES}" "${LEAN_SCRIPT}"

tr '\0' '\n' < "${FILES_NUL}" > "${EXPECTED}"
printf '%s\n%s\n%s\n%s\n' "${LEAN_MANIFEST}" "${LEAN_INVENTORY}" "${EXCLUDED_FIXTURES}" "${LEAN_SCRIPT}" >> "${EXPECTED}"
LC_ALL=C sort -u -o "${EXPECTED}" "${EXPECTED}"
tar -tf "${STAGING}/${LEAN_ARCHIVE}" | LC_ALL=C sort -u > "${ACTUAL}"
cmp -- "${EXPECTED}" "${ACTUAL}"
(cd "${EXTRACTED}" && sha256sum -c --quiet "${STAGING}/${LEAN_MANIFEST}")

lean_sha=$(sha256sum "${STAGING}/${LEAN_ARCHIVE}" | awk '{print $1}')
printf '%s  %s\n' "${lean_sha}" "${LEAN_ARCHIVE}" > "${STAGING}/${LEAN_SIDECAR}"
ln -- "${STAGING}/${LEAN_ARCHIVE}" "${HOME_ROOT}/${LEAN_ARCHIVE}"
ln -- "${STAGING}/${LEAN_MANIFEST}" "${HOME_ROOT}/${LEAN_MANIFEST}"
ln -- "${STAGING}/${LEAN_SIDECAR}" "${HOME_ROOT}/${LEAN_SIDECAR}"

printf 'LEAN_FILES=%s\n' "${#files[@]}"
du -h "${HOME_ROOT}/${LEAN_ARCHIVE}"
cat "${HOME_ROOT}/${LEAN_SIDECAR}"
