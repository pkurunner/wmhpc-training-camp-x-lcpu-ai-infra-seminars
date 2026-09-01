#!/usr/bin/env bash
# Deterministically compress the verified lean tar for an unreliable SSH link.
set -Eeuo pipefail

HOME_ROOT=/home/lcpu/85117379
RECOVERY_LOCK=${HOME_ROOT}/.c2-native-v11-hardened-recovery-20260830.lock
LEAN_TAR=c2-native-v7-v11-continuation-hardened-lean-evidence-20260830.tar
LEAN_TAR_SIDECAR=${LEAN_TAR}.sha256
LEAN_GZIP=${LEAN_TAR}.gz
LEAN_GZIP_SIDECAR=${LEAN_GZIP}.sha256
TMP_GZIP=$(mktemp "${HOME_ROOT}/.${LEAN_GZIP}.XXXXXX")
TMP_SIDECAR=$(mktemp "${HOME_ROOT}/.${LEAN_GZIP_SIDECAR}.XXXXXX")

cleanup() {
  rm -f -- "${TMP_GZIP}" "${TMP_SIDECAR}"
}
trap cleanup EXIT

command -v flock >/dev/null
exec 9>"${RECOVERY_LOCK}"
flock -n 9 || { echo 'v11 recovery/archive lock is already held' >&2; exit 2; }

cd "${HOME_ROOT}"
[[ -f "${LEAN_TAR}" && ! -L "${LEAN_TAR}" ]]
[[ -f "${LEAN_TAR_SIDECAR}" && ! -L "${LEAN_TAR_SIDECAR}" ]]
[[ ! -e "${LEAN_GZIP}" && ! -e "${LEAN_GZIP_SIDECAR}" ]]
sha256sum -c --quiet "${LEAN_TAR_SIDECAR}"
expected_tar_sha=$(awk 'NR == 1 {print $1}' "${LEAN_TAR_SIDECAR}")
[[ "${expected_tar_sha}" =~ ^[0-9a-f]{64}$ && "$(wc -l < "${LEAN_TAR_SIDECAR}")" -eq 1 ]]

gzip -n -9 -c -- "${LEAN_TAR}" > "${TMP_GZIP}"
gzip -t -- "${TMP_GZIP}"
[[ "$(gzip -dc -- "${TMP_GZIP}" | sha256sum | awk '{print $1}')" == "${expected_tar_sha}" ]]
gzip_sha=$(sha256sum -- "${TMP_GZIP}" | awk '{print $1}')
printf '%s  %s\n' "${gzip_sha}" "${LEAN_GZIP}" > "${TMP_SIDECAR}"

ln -- "${TMP_GZIP}" "${HOME_ROOT}/${LEAN_GZIP}"
ln -- "${TMP_SIDECAR}" "${HOME_ROOT}/${LEAN_GZIP_SIDECAR}"
du -h "${HOME_ROOT}/${LEAN_GZIP}"
cat "${HOME_ROOT}/${LEAN_GZIP_SIDECAR}"
