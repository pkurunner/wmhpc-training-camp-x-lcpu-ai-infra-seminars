#!/usr/bin/env bash
# One-shot, identity-pinned submission of the two independent CPU AOT builds.
# Keeping both submissions in one login-shell transaction makes recovery from
# an unreliable SSH login plane deterministic and avoids duplicate jobs.

set -Eeuo pipefail

HOME_ROOT=/home/lcpu/85117379
SUBMISSION_LOG=${HOME_ROOT}/c2-v9-v10-aot-submissions-20260830.txt
V9_ROOT=${HOME_ROOT}/c2-native-plugin-v9-aot-artifacts-20260830
V10_ROOT=${HOME_ROOT}/c2-native-plugin-v10-aot-artifacts-20260830

[[ ! -e "${SUBMISSION_LOG}" ]] || {
  echo "submission log already exists: ${SUBMISSION_LOG}" >&2
  exit 2
}

sha256sum -c - <<'SHA256'
e7f84215631bc68859dda1cc478fca6301d1b1bb95b9f6a1135728b7caa8d092  /home/lcpu/85117379/native_c2_v9_parallel_merge_20260830.patch
9956b6b659c8867a00e4651a2482a063e2a1f5f361ef62be93528b97970806ed  /home/lcpu/85117379/native_c2_decode.v9-parallel-merge.cu
9bffbfd95cf0e25249509977a779e90641463b41637bedce943fc271cb6cd84b  /home/lcpu/85117379/build_native_c2_plugin_v9_parallel_merge_aot.slurm
fb9aca9b1232b94239e528863757d1ccab763b0c2e615bbd1b06237e754cd93c  /home/lcpu/85117379/validate_native_c2_plugin_v9_parallel_merge_directed.slurm
0c984b10746b57aaf14694233ecd7a51fb843044e8447b6f3681435e75b5b7ca  /home/lcpu/85117379/validate_native_c2_plugin_v9_parallel_merge_stress_perf_3pct.slurm
ea99b88122ca7f11b9ddfaf9781122eee2df5333f19613d345b7f86491f8385c  /home/lcpu/85117379/native_c2_v10_k_prefetch_20260830.patch
0477c79e85750fb8a7006e4dc68bd7cab81c65103f0e55cd8c6d7c2e25a44ce6  /home/lcpu/85117379/native_c2_decode.v10-k-prefetch.cu
a5ff03ac285f03ddda4c04a31143ef582c4a941fe41de3e33823cf3eb7becc42  /home/lcpu/85117379/native_c2_v10_k_prefetch_directed.py
6110e60a1a6305620d08dfe2fbd9ac48c77b17f110e4bc4954eae78a375c0ac9  /home/lcpu/85117379/build_native_c2_plugin_v10_k_prefetch_aot.slurm
7fcb890b503cb0caa03c7f399dbf514301158403a096fbb93f4a473794584835  /home/lcpu/85117379/validate_native_c2_plugin_v10_k_prefetch_directed.slurm
777d22e7645cc8ba227334fc6bc6a7149db83f9de33405d487cd7b515b7e082f  /home/lcpu/85117379/validate_native_c2_plugin_v10_k_prefetch_stress_perf_3pct.slurm
SHA256

bash -n \
  "${HOME_ROOT}/build_native_c2_plugin_v9_parallel_merge_aot.slurm" \
  "${HOME_ROOT}/validate_native_c2_plugin_v9_parallel_merge_directed.slurm" \
  "${HOME_ROOT}/validate_native_c2_plugin_v9_parallel_merge_stress_perf_3pct.slurm" \
  "${HOME_ROOT}/build_native_c2_plugin_v10_k_prefetch_aot.slurm" \
  "${HOME_ROOT}/validate_native_c2_plugin_v10_k_prefetch_directed.slurm" \
  "${HOME_ROOT}/validate_native_c2_plugin_v10_k_prefetch_stress_perf_3pct.slurm"

for root in "${V9_ROOT}" "${V10_ROOT}"; do
  mkdir -p "${root}"
  [[ -z "$(find "${root}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    echo "artifact root is not empty: ${root}" >&2
    exit 2
  }
done

{
  printf 'START_UTC=%s\n' "$(date -u +%FT%TZ)"
  sha256sum "$0"
} > "${SUBMISSION_LOG}"

v9_job=$(sbatch --parsable \
  --export=ALL,C2_EXPECTED_SCRIPT_SHA=9bffbfd95cf0e25249509977a779e90641463b41637bedce943fc271cb6cd84b \
  "${HOME_ROOT}/build_native_c2_plugin_v9_parallel_merge_aot.slurm")
printf 'V9_AOT_JOB=%s\n' "${v9_job}" | tee -a "${SUBMISSION_LOG}"

v10_job=$(sbatch --parsable \
  --export=ALL,C2_EXPECTED_SCRIPT_SHA=6110e60a1a6305620d08dfe2fbd9ac48c77b17f110e4bc4954eae78a375c0ac9 \
  "${HOME_ROOT}/build_native_c2_plugin_v10_k_prefetch_aot.slurm")
printf 'V10_AOT_JOB=%s\n' "${v10_job}" | tee -a "${SUBMISSION_LOG}"
printf 'END_UTC=%s\n' "$(date -u +%FT%TZ)" | tee -a "${SUBMISSION_LOG}"
