#!/usr/bin/env bash
# Idempotent bridge from the two fixed AOT jobs to the first (v9) GPU gate.
# It is intentionally a one-shot login-side transaction: no result is inferred
# from Slurm state alone, and every consumed artifact is re-hashed first.

set -Eeuo pipefail

HOME_ROOT=/home/lcpu/85117379
V9_JOB=12701
V10_JOB=12702
V9_ART=${HOME_ROOT}/c2-native-plugin-v9-aot-artifacts-20260830/job${V9_JOB}
V10_ART=${HOME_ROOT}/c2-native-plugin-v10-aot-artifacts-20260830/job${V10_JOB}
V9_SRC=${HOME_ROOT}/vllm-d4-native-c2-plugin-v9-parallel-merge-20260830-job${V9_JOB}/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
V10_SRC=${HOME_ROOT}/vllm-d4-native-c2-plugin-v10-k-prefetch-20260830-job${V10_JOB}/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
READY_LOG=${HOME_ROOT}/c2-v9-v10-aot-ready-20260830.txt
V9_DIRECTED_LOG=${HOME_ROOT}/c2-v9-directed-submission-20260830.txt
V9_DIRECTED_ROOT=${HOME_ROOT}/c2-native-plugin-v9-directed-artifacts-20260830

digest() { sha256sum "$1" | awk '{print $1}'; }
fresh_manifest_matches() {
  local artifact_dir=$1 recorded_manifest=$2 fresh_manifest
  fresh_manifest=$(mktemp -p "${HOME_ROOT}" c2-aot-fresh-manifest.XXXXXX)
  if ! find "${artifact_dir}" -type f \
      ! -name "$(basename "${recorded_manifest}")" \
      ! -name 'final-status-job*.txt' \
      ! -name 'final-status-job*.sha256' \
      -print0 | sort -z | xargs -0 -r sha256sum > "${fresh_manifest}"; then
    rm -f -- "${fresh_manifest}"
    return 1
  fi
  if [[ ! -s "${fresh_manifest}" ]] || \
      ! sha256sum -c "${fresh_manifest}" >/dev/null || \
      ! cmp -s "${fresh_manifest}" "${recorded_manifest}"; then
    rm -f -- "${fresh_manifest}"
    return 1
  fi
  rm -f -- "${fresh_manifest}"
}

# slurmdbd is currently unavailable on the login node, so sacct cannot be the
# sole readiness oracle.  A running/pending job remains visible to squeue; a
# finished job must additionally provide a fully closed artifact/status chain.
active_jobs=$(squeue -h -j "${V9_JOB},${V10_JOB}" -o '%i|%T')
if [[ -n "${active_jobs}" ]]; then
  printf 'AOT jobs are not both finished:\n%s\n' "${active_jobs}" >&2
  exit 4
fi

v9_status=${V9_ART}/final-status-job${V9_JOB}.txt
v10_status=${V10_ART}/final-status-job${V10_JOB}.txt
v9_status_sidecar=${V9_ART}/final-status-job${V9_JOB}.sha256
v10_status_sidecar=${V10_ART}/final-status-job${V10_JOB}.sha256
v9_outputs=${V9_ART}/outputs-job${V9_JOB}.sha256
v10_outputs=${V10_ART}/outputs-job${V10_JOB}.sha256

sha256sum -c "${v9_status_sidecar}"
sha256sum -c "${v10_status_sidecar}"
sha256sum -c "${v9_outputs}"
sha256sum -c "${v10_outputs}"
grep -Eq '^BODY_RC=0 TEE_RC=0 MANIFEST_RC=0 FINAL_RC=0 ' "${v9_status}"
grep -Eq '^BODY_RC=0 TEE_RC=0 MANIFEST_RC=0 FINAL_RC=0 ' "${v10_status}"
fresh_manifest_matches "${V9_ART}" "${v9_outputs}"
fresh_manifest_matches "${V10_ART}" "${v10_outputs}"

v9_plugin=${V9_ART}/vllm/_native_c2_msa_decode_plugin.abi3.so
v10_plugin=${V10_ART}/vllm/_native_c2_msa_decode_plugin.abi3.so
v9_resource=${V9_ART}/plugin-resource-gate-job${V9_JOB}.txt
v10_resource=${V10_ART}/plugin-resource-gate-job${V10_JOB}.txt
v9_provenance=${V9_ART}/plugin-v9-provenance-job${V9_JOB}.sha256
v10_provenance=${V10_ART}/plugin-v10-provenance-job${V10_JOB}.sha256

[[ "$(digest "${V9_SRC}")" == 9956b6b659c8867a00e4651a2482a063e2a1f5f361ef62be93528b97970806ed ]]
[[ "$(digest "${V10_SRC}")" == 0477c79e85750fb8a7006e4dc68bd7cab81c65103f0e55cd8c6d7c2e25a44ce6 ]]
grep -Eq 'STACK:[[:space:]]*0.*SHARED:[[:space:]]*30880.*LOCAL:[[:space:]]*0' "${v9_resource}"
grep -Eq 'STACK:[[:space:]]*0.*SHARED:[[:space:]]*46944.*LOCAL:[[:space:]]*0' "${v10_resource}"
# The provenance manifests intentionally include the Slurm spool copy of the
# submitted script.  That path is ephemeral after job cleanup, so re-running
# the whole provenance manifest would reject a sound durable artifact.  The
# manifest files themselves are covered by the AOT outputs manifest; downstream
# gates below additionally require the exact durable DSO and source entries.
grep -Fx "$(digest "${v9_plugin}")  ${v9_plugin}" "${v9_provenance}"
grep -Fx "9956b6b659c8867a00e4651a2482a063e2a1f5f361ef62be93528b97970806ed  ${V9_SRC}" "${v9_provenance}"
grep -Fx "$(digest "${v10_plugin}")  ${v10_plugin}" "${v10_provenance}"
grep -Fx "0477c79e85750fb8a7006e4dc68bd7cab81c65103f0e55cd8c6d7c2e25a44ce6  ${V10_SRC}" "${v10_provenance}"

v9_plugin_sha=$(digest "${v9_plugin}")
v10_plugin_sha=$(digest "${v10_plugin}")
v9_resource_sha=$(digest "${v9_resource}")
v10_resource_sha=$(digest "${v10_resource}")
v9_provenance_sha=$(digest "${v9_provenance}")
v10_provenance_sha=$(digest "${v10_provenance}")

{
  printf 'READY_UTC=%s\n' "$(date -u +%FT%TZ)"
  printf 'V9_AOT_JOB=%s\nV9_PLUGIN=%s\nV9_PLUGIN_SHA=%s\nV9_SOURCE=%s\n' \
    "${V9_JOB}" "${v9_plugin}" "${v9_plugin_sha}" "${V9_SRC}"
  printf 'V9_RESOURCE=%s\nV9_RESOURCE_SHA=%s\nV9_PROVENANCE=%s\nV9_PROVENANCE_SHA=%s\n' \
    "${v9_resource}" "${v9_resource_sha}" "${v9_provenance}" "${v9_provenance_sha}"
  printf 'V10_AOT_JOB=%s\nV10_PLUGIN=%s\nV10_PLUGIN_SHA=%s\nV10_SOURCE=%s\n' \
    "${V10_JOB}" "${v10_plugin}" "${v10_plugin_sha}" "${V10_SRC}"
  printf 'V10_RESOURCE=%s\nV10_RESOURCE_SHA=%s\nV10_PROVENANCE=%s\nV10_PROVENANCE_SHA=%s\n' \
    "${v10_resource}" "${v10_resource_sha}" "${v10_provenance}" "${v10_provenance_sha}"
} > "${READY_LOG}"

[[ ! -e "${V9_DIRECTED_LOG}" ]] || {
  cat "${V9_DIRECTED_LOG}"
  exit 0
}
mkdir -p "${V9_DIRECTED_ROOT}"
[[ -z "$(find "${V9_DIRECTED_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]

v9_directed_job=$(sbatch --parsable \
  --export=ALL,C2_V9_CANDIDATE_PLUGIN="${v9_plugin}",C2_EXPECTED_V9_CANDIDATE_SHA="${v9_plugin_sha}",C2_V9_CANDIDATE_SOURCE="${V9_SRC}",C2_EXPECTED_V9_SOURCE_SHA=9956b6b659c8867a00e4651a2482a063e2a1f5f361ef62be93528b97970806ed,C2_V9_CANDIDATE_RESOURCE_GATE="${v9_resource}",C2_EXPECTED_V9_RESOURCE_GATE_SHA="${v9_resource_sha}",C2_V9_CANDIDATE_PROVENANCE="${v9_provenance}",C2_EXPECTED_V9_PROVENANCE_SHA="${v9_provenance_sha}",C2_EXPECTED_V6_DIRECTED_HARNESS_SHA=c38ec0bc7ffb9d85567d3cbbf4c2991077eb6cb3a52874778ddac01d107577a5,C2_EXPECTED_V9_DIRECTED_SCRIPT_SHA=cabe571d068813f2a4da4ef4f2b81b7b4e11681ca3dacbcfe8fb5fa52807b238 \
  "${HOME_ROOT}/validate_native_c2_plugin_v9_parallel_merge_directed.slurm")
printf 'V9_DIRECTED_JOB=%s\nSUBMIT_UTC=%s\n' "${v9_directed_job}" "$(date -u +%FT%TZ)" | tee "${V9_DIRECTED_LOG}"
