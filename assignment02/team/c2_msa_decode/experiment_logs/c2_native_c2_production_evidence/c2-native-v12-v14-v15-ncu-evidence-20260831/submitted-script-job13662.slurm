#!/usr/bin/env bash
# Read-only, three-arm NCU mechanism collection for the retained v12 candidate
# and the two valid-but-rejected stride experiments (v14, v15).  This program
# has no timing score or acceptance rule: NCU instrumentation is not a timing
# source.  Review first; do not submit until C2_EXPECTED_RUNTIME_SCRIPT_SHA is
# replaced with this file's verified SHA-256.
#
# sbatch --export=ALL,C2_EXPECTED_RUNTIME_SCRIPT_SHA=<reviewed-sha256> \
#   profile_native_c2_v12_v14_v15_ncu_20260831.slurm

#SBATCH --job-name=c2-native-v12-v14-v15-ncu
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v12-v14-v15-ncu-artifacts-20260831/slurm-%j.log

set -Eeuo pipefail
umask 077
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONPATH=
export PYTHONDONTWRITEBYTECODE=1
export PATH=/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin

BASE=/home/lcpu/85117379
ARTIFACT_ROOT=${BASE}/c2-native-plugin-v12-v14-v15-ncu-artifacts-20260831
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID:?must run under Slurm}
RUNTIME=/tmp/c2-native-plugin-v12-v14-v15-ncu-job${SLURM_JOB_ID}
PYTHON=${BASE}/venv-vllm-d4da0c5/bin/python
BASE_HARNESS=${BASE}/native_c2_operator_bench_20260829.py
SUBMITTED_SCRIPT=${BASE}/profile_native_c2_v12_v14_v15_ncu_20260831.slurm

# The v12 paths/hashes are independently recoverable from the preserved
# job12983/job12984 manifest; these are intentionally not guessed at submit.
V12_AOT=${BASE}/c2-native-plugin-v12-aot-artifacts-20260830/job12983
V12_DIRECTED=${BASE}/c2-native-plugin-v12-directed-artifacts-20260830/job12984
V12_STRESS=${BASE}/c2-native-plugin-v12-stress-3pct-artifacts-20260830/job12985
V12_LIBRARY=${V12_AOT}/vllm/_native_c2_msa_decode_plugin.abi3.so
V12_SOURCE=${V12_AOT}/native_c2_decode.v12-q-row-padding.cu
V12_RESOURCE=${V12_AOT}/plugin-resource-gate-job12983.txt
V12_PROVENANCE=${V12_AOT}/plugin-v12-provenance-job12983.sha256
V12_AOT_FINAL=${V12_AOT}/final-status-job12983.txt
V12_AOT_SIDECAR=${V12_AOT}/final-status-job12983.sha256
V12_AOT_OUTPUTS=${V12_AOT}/outputs-job12983.sha256
V12_AOT_SUBMITTED=${V12_AOT}/submitted-script-job12983.slurm
V12_DIRECTED_RESULT=${V12_DIRECTED}/v12-q-row-padding-directed-job12984.json
V12_DIRECTED_FINAL=${V12_DIRECTED}/final-status-job12984.txt
V12_DIRECTED_SIDECAR=${V12_DIRECTED}/final-status-job12984.sha256
V12_DIRECTED_OUTPUTS=${V12_DIRECTED}/outputs-job12984.sha256
V12_DIRECTED_RUN=${V12_DIRECTED}/run-manifest-job12984.json
V12_DIRECTED_SUBMITTED=${V12_DIRECTED}/submitted-script-job12984.slurm
V12_DECISION=${V12_STRESS}/v12-q-row-padding-vs-v11-3pct-incremental-decision-job12985.json
V12_AGGREGATE=${V12_STRESS}/v12-q-row-padding-vs-v11-3pct-incremental-aggregate-job12985.json
V12_STRESS_FINAL=${V12_STRESS}/final-status-job12985.txt
V12_STRESS_SIDECAR=${V12_STRESS}/final-status-job12985.sha256
V12_STRESS_OUTPUTS=${V12_STRESS}/outputs-job12985.sha256

V14_AOT=${BASE}/c2-native-plugin-v14-kv-stage-padding-aot-artifacts-20260831/job13487
V14_DIRECTED=${BASE}/c2-native-plugin-v14-kv-stage-padding-directed-artifacts-20260831/job13513
V14_STRESS=${BASE}/c2-native-plugin-v14-kv-stage-padding-stress-3pct-retry1-artifacts-20260831/job13539
V14_LIBRARY=${V14_AOT}/vllm/_native_c2_msa_decode_plugin.abi3.so
V14_SOURCE=${V14_AOT}/native_c2_decode.v14-kv-stage-padding.cu
V14_RESOURCE=${V14_AOT}/plugin-resource-gate-job13487.txt
V14_PROVENANCE=${V14_AOT}/plugin-v14-provenance-job13487.sha256
V14_AOT_FINAL=${V14_AOT}/final-status-job13487.txt
V14_AOT_SIDECAR=${V14_AOT}/final-status-job13487.sha256
V14_AOT_OUTPUTS=${V14_AOT}/outputs-job13487.sha256
V14_AOT_SUBMITTED=${V14_AOT}/submitted-script-job13487.slurm
V14_DIRECTED_RESULT=${V14_DIRECTED}/v14-kv-stage-padding-directed-job13513.json
V14_DIRECTED_FINAL=${V14_DIRECTED}/final-status-job13513.txt
V14_DIRECTED_SIDECAR=${V14_DIRECTED}/final-status-job13513.sha256
V14_DIRECTED_OUTPUTS=${V14_DIRECTED}/outputs-job13513.sha256
V14_DIRECTED_RUN=${V14_DIRECTED}/run-manifest-job13513.json
V14_DIRECTED_SUBMITTED=${V14_DIRECTED}/submitted-script-job13513.slurm
V14_DECISION=${V14_STRESS}/decision-job13539.json
V14_AGGREGATE=${V14_STRESS}/aggregate-job13539.json
V14_STRESS_FINAL=${V14_STRESS}/final-status-job13539.txt
V14_STRESS_SIDECAR=${V14_STRESS}/final-status-job13539.sha256
V14_STRESS_OUTPUTS=${V14_STRESS}/outputs-job13539.sha256

V15_AOT=${BASE}/c2-native-plugin-v15-q-stage-stride144-aot-artifacts-20260831/job13564
V15_DIRECTED=${BASE}/c2-native-plugin-v15-q-stage-stride144-directed-artifacts-20260831/job13575
V15_STRESS=${BASE}/c2-native-plugin-v15-q-stage-stride144-stress-3pct-artifacts-20260831/job13576
V15_LIBRARY=${V15_AOT}/vllm/_native_c2_msa_decode_plugin.abi3.so
V15_SOURCE=${V15_AOT}/native_c2_decode.v15-q-stage-stride144.cu
V15_RESOURCE=${V15_AOT}/plugin-resource-gate-job13564.txt
V15_PROVENANCE=${V15_AOT}/plugin-v15-provenance-job13564.sha256
V15_AOT_FINAL=${V15_AOT}/final-status-job13564.txt
V15_AOT_SIDECAR=${V15_AOT}/final-status-job13564.sha256
V15_AOT_OUTPUTS=${V15_AOT}/outputs-job13564.sha256
V15_AOT_SUBMITTED=${V15_AOT}/submitted-script-job13564.slurm
V15_DIRECTED_RESULT=${V15_DIRECTED}/v15-q-stage-stride144-directed-job13575.json
V15_DIRECTED_FINAL=${V15_DIRECTED}/final-status-job13575.txt
V15_DIRECTED_SIDECAR=${V15_DIRECTED}/final-status-job13575.sha256
V15_DIRECTED_OUTPUTS=${V15_DIRECTED}/outputs-job13575.sha256
V15_DIRECTED_RUN=${V15_DIRECTED}/run-manifest-job13575.json
V15_DIRECTED_SUBMITTED=${V15_DIRECTED}/submitted-script-job13575.slurm
V15_DECISION=${V15_STRESS}/decision-job13576.json
V15_AGGREGATE=${V15_STRESS}/aggregate-job13576.json
V15_STRESS_FINAL=${V15_STRESS}/final-status-job13576.txt
V15_STRESS_SIDECAR=${V15_STRESS}/final-status-job13576.sha256
V15_STRESS_OUTPUTS=${V15_STRESS}/outputs-job13576.sha256

EXPECTED_BASE_SHA=0b51fede1e6258eea0b535acba2a86e371b284e9655352156cdeb3d310d90c28
EXPECTED_V12_LIBRARY_SHA=064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6
EXPECTED_V12_SOURCE_SHA=535d90b856ed1062aa7b8a105eb2c5f236c450826e65496e646d0d5a27eb8aaf
EXPECTED_V12_RESOURCE_SHA=c4a6cd69617ce82e99459fcd7a975a37fbfad497e51b36a6ec09945ffa97d6c0
EXPECTED_V12_PROVENANCE_SHA=65bb886cece4a49b7ef7d7d4cab00588e4cc1f9238d2aa45dd5cf36f6952d2a2
EXPECTED_V12_AOT_FINAL_SHA=b5fd3d82083165415ba20c37cf574e898ef0209d1a7844866c7491cc438b0f18
EXPECTED_V12_AOT_SIDECAR_SHA=912b28b5f81610b7cc04457c54a26165d4e321cfe709bdc79489a3b102041f67
EXPECTED_V12_AOT_OUTPUTS_SHA=ca1f37f00f9ce78c0d88bb3a90e8288bbffd83e94315a378b36a9c7998d99419
EXPECTED_V12_AOT_SUBMITTED_SHA=64a40eaaf436363038d58eef38fdd96d9a4fe25470d4d2ab43030a51a5f166a7
EXPECTED_V12_DIRECTED_RESULT_SHA=240b683c1b6921981cc814084697ff945a7714de1bace5956f74a09ea38765c8
EXPECTED_V12_DIRECTED_FINAL_SHA=16026037d3916cf0dc741317e35ca258536ab3a8012f2671a0886e2c49ee2758
EXPECTED_V12_DIRECTED_SIDECAR_SHA=9da4e355841c9448683cd82d3281112753e2accdf519b7fdad05b60f8a3bbc5a
EXPECTED_V12_DIRECTED_OUTPUTS_SHA=0b1b512ef6bf0c8a5b461561f6fd4b5b768e4922f75c0285ce8bac37b793bf7a
EXPECTED_V12_DIRECTED_RUN_SHA=d07f6028ed6ff0815cb0019a3df30235edc3a3b918cded23b413feaaf94bacde
EXPECTED_V12_DIRECTED_SUBMITTED_SHA=8ab0dcdea71d9e2d864afdafaf0f16703593cdccecfa0e18cfe0a2aecd8c4a3a
EXPECTED_V12_DECISION_SHA=829505fce32c737f596b372819add7f7701dc793c87d58f429e61ba7806862d9
EXPECTED_V12_AGGREGATE_SHA=0e9b4da6ea78c812c66feab0d4c669e68d0dc9a65630d13fbbc37c2e66295142
EXPECTED_V12_STRESS_FINAL_SHA=4e8c40bb6784b111dced540a2b977ff383d2a5cb41796a65711ef609bf409ca1
EXPECTED_V12_STRESS_SIDECAR_SHA=042449479aa483c2b3f1b345e04c89d0c711322b2ee4b82014efd4a40a54468e
EXPECTED_V12_STRESS_OUTPUTS_SHA=799ee63d1dd815d381fcc866687da97bf52177994a490df7ab8b5f07e5bf78cc

EXPECTED_V14_LIBRARY_SHA=7e49201d9b08cfdcc028c94b3cefbc93f03417812caa51113813dfae1f0ca132
EXPECTED_V14_SOURCE_SHA=1f53ce18eb29565564c945bcfebb973192f7068d38f31d432e7a8a3046a9556d
EXPECTED_V14_RESOURCE_SHA=cd46e367bec60eb21cf5cf867ba4d0fcb968983c5589436a8f4843b4af354a4e
EXPECTED_V14_PROVENANCE_SHA=8adece2157d7ab569c5b84eecc201ce7a657f5b7ae5ce29ef19e1446714f310e
EXPECTED_V14_AOT_FINAL_SHA=52332b8164c285712cc1ec65daa980fefa1a6a4ef6c98066b94bfa622734aa9c
EXPECTED_V14_AOT_SIDECAR_SHA=540c7c2a3c8d6f5c021b05de52f2cef9a56956f642446900013d560115946a4c
EXPECTED_V14_AOT_OUTPUTS_SHA=b656e85c8aa5f24bad894242b4a65e8cdf972b36619352ca589018fa718cb4c0
EXPECTED_V14_AOT_SUBMITTED_SHA=770461213278f538f6d9ac7d9bc53dcd21dfd268471fc2bb70f025ce80f05a0e
EXPECTED_V14_DIRECTED_RESULT_SHA=10d3c3eb7547203ebf9b5daa5220378c5edb35100cce26019547e0980375eb11
EXPECTED_V14_DIRECTED_FINAL_SHA=bfee7a9e466d400362847385f0e9b04857ec4697708deb16c7f2f6e0e0c93958
EXPECTED_V14_DIRECTED_SIDECAR_SHA=8f2325307d1f14c1d63278f1ebe08cd2033f2112008ea02a19d692233d973677
EXPECTED_V14_DIRECTED_OUTPUTS_SHA=75725cfe7cbc774bc183017a32108f0ad0af5a0632f1294625e904f1dd4f3bd3
EXPECTED_V14_DIRECTED_RUN_SHA=27421e355b1e37a9052bcb65c5eefd70f91ea7ae98fb01cece2a3becac7de8f3
EXPECTED_V14_DIRECTED_SUBMITTED_SHA=960d05d5f0df63ba83a7420e5cfb2c9cae1c5c37f6e9a6c0278f7bee69281639
EXPECTED_V14_DECISION_SHA=726b150e8ae17cdac6ab47606942fa441092b51a23209520cd1ce9fcfd194d23
EXPECTED_V14_AGGREGATE_SHA=c79c6de05741bba52706788744d171270a7f35307ede795e1ceb399e1e20962e
EXPECTED_V14_STRESS_FINAL_SHA=3febddc26804e0400cae769067e824b9b1a4a290c4c34d486a4b826e5032f8fe
EXPECTED_V14_STRESS_SIDECAR_SHA=90707698946efaa187a0602db30baebc6a1d03cfc5a823708828343c45889954
EXPECTED_V14_STRESS_OUTPUTS_SHA=f2a81b331f411f040a7376e62dffbf40b414bde703cfbaf310d8bab7dd648a5e

EXPECTED_V15_LIBRARY_SHA=0c5ac4a1c890cac886fd41989746bce2740619965dee3e85bf8d85e05dd2f97d
EXPECTED_V15_SOURCE_SHA=b71fc21d4cffbe73757d501849eb9048009f32333d96beb0ae1452b49a9912af
EXPECTED_V15_RESOURCE_SHA=7d51f0c9da5bc4fc55ca649598293adab2f4c961d817893038e17a98784d8404
EXPECTED_V15_PROVENANCE_SHA=5675c3b19638da5b0464f6bafc60d280ccf28afc0f919f6ef801cf422fbf18be
EXPECTED_V15_AOT_FINAL_SHA=973bfa1c46db72fa29de41e36ddc6306151f467758162a32b2167934029761ef
EXPECTED_V15_AOT_SIDECAR_SHA=341595740e0a1783d398afb87ffcc3c10812014516fe41b0cd8f4f8bc4cc2c61
EXPECTED_V15_AOT_OUTPUTS_SHA=8de6a0eeaa2080c9409291116bd0820aa9f5a7e7c898206a11a1ba652c7a50f6
EXPECTED_V15_AOT_SUBMITTED_SHA=7d899ae1cb9854049b1c9b79d806b700233255de5d41c45fe05789a0866886d0
EXPECTED_V15_DIRECTED_RESULT_SHA=8698990ba4db30a5755df404630fcf5a99033b4cf9b82f4eb9314439d647a07a
EXPECTED_V15_DIRECTED_FINAL_SHA=87237d0c4e8ba576ccc086c1dee7c761c4008553147e63ae4664060ae8bc3349
EXPECTED_V15_DIRECTED_SIDECAR_SHA=c2edb44caaa7c7322973ecc904e287c464453f0b3edb289c82264d47ce099e29
EXPECTED_V15_DIRECTED_OUTPUTS_SHA=5229c00f8b0459ae04f2d34cf73480ce0254cee6517650be01902490d243476f
EXPECTED_V15_DIRECTED_RUN_SHA=d4cb3a419b3898ea514239736c55c3a65e37ef3f9d378c24cfe12e2384cb385f
EXPECTED_V15_DIRECTED_SUBMITTED_SHA=3f56f14af72abb4cde74beb34a25d498fd78fbeb9432a4a44b61c9345f2768d8
EXPECTED_V15_DECISION_SHA=43f79d07814914a8fcbc5f7f2dc24979fd1e4c8d14d5f949d533760132f20011
EXPECTED_V15_AGGREGATE_SHA=368fcbf5db6d14fff6e54a34e31b1930efbffd529b39cc2656d6a52c82cb72eb
EXPECTED_V15_STRESS_FINAL_SHA=be4688511462973cd55ec024c11796bcab803b4683f9457fc10b8bf8222bbd5c
EXPECTED_V15_STRESS_SIDECAR_SHA=b1e6afe5037659e89abe556434422b291f6cf6459a975efd4b2dd36468f8b057
EXPECTED_V15_STRESS_OUTPUTS_SHA=1d284a58af14f9078197b556e17e37eada67627ca66bbc3f7255130f1de34e6f

: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?reviewed NCU script SHA-256 required}"
[[ ${C2_EXPECTED_RUNTIME_SCRIPT_SHA} =~ ^[0-9a-f]{64}$ ]]

# Exactly fifteen collection counters.  Duration is deliberately absent.
metrics=(
  smsp__cycles_elapsed.sum dram__bytes_read.sum dram__bytes_write.sum lts__t_bytes.sum
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active smsp__inst_executed_pipe_tensor.sum
  smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct
  smsp__average_warps_issue_stalled_wait_per_issue_active.pct
  smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
  smsp__warps_active.avg.per_cycle_active sm__warps_active.avg.pct_of_peak_sustained_active
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
)
(( ${#metrics[@]} == 15 )) && (( $(printf '%s\n' "${metrics[@]}" | sort -u | wc -l) == 15 ))
metrics_csv=$(IFS=,; printf '%s' "${metrics[*]}")

sha256_of() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3
  [[ "$path" == /* && -f "$path" && ! -L "$path" && "$expected" =~ ^[0-9a-f]{64}$ ]] || { echo "bad ${label}" >&2; exit 2; }
  [[ $(sha256_of "$path") == "$expected" ]] || { echo "digest drift: ${label}" >&2; exit 2; }
}
selector() {
  local x=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES required}
  [[ "$x" != *,* && ( "$x" =~ ^[0-9]+$ || "$x" =~ ^GPU-[0-9A-Fa-f-]+$ ) ]] || return 1
  printf '%s\n' "$x"
}
gpu_uuid() {
  local x; x=$(selector)
  mapfile -t uuids < <(nvidia-smi --id "$x" --query-gpu=uuid --format=csv,noheader,nounits)
  [[ ${#uuids[@]} -eq 1 && ${uuids[0]} =~ ^GPU- ]] || return 1
  printf '%s\n' "${uuids[0]}"
}
compute_apps() { nvidia-smi --id "$(selector)" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'; }
snapshot() {
  local tag=$1 out=${ARTIFACT_DIR}/gpu-${tag}-job${SLURM_JOB_ID}.txt
  { printf '===== %s UTC=%s =====\n' "$tag" "$(date -u +%FT%TZ)"
    nvidia-smi --id "$(selector)" --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap --format=csv,noheader,nounits
    printf 'COMPUTE_APPS\n'; compute_apps
  } > "$out"
  cat "$out"
}
safe_runtime() {
  [[ "$RUNTIME" == "/tmp/c2-native-plugin-v12-v14-v15-ncu-job${SLURM_JOB_ID}" && ! -e "$RUNTIME" && ! -L "$RUNTIME" ]]
  mkdir -m 0700 -- "$RUNTIME"
  [[ -d "$RUNTIME" && ! -L "$RUNTIME" && "$(readlink -f -- "$RUNTIME")" == "$RUNTIME" && "$(stat -c %u -- "$RUNTIME")" == "$(id -u)" ]]
}
cleanup_runtime() {
  [[ -d "$RUNTIME" && ! -L "$RUNTIME" && "$(readlink -f -- "$RUNTIME")" == "$RUNTIME" && "$(stat -c %u -- "$RUNTIME")" == "$(id -u)" ]] || return 1
  rm -rf -- "$RUNTIME"
  [[ ! -e "$RUNTIME" && ! -L "$RUNTIME" ]]
}
finish() {
  local original_rc=$? rc finalizer_error=0 cleanup_rc=0 manifest_rc=0 tee_rc=0 apps='' post_uuid='' manifest output_list
  rc=$original_rc
  trap - EXIT; set +e
  [[ $original_rc -eq 0 ]] || rc=2
  printf 'BODY_RC=%s UTC=%s FINALIZING=1\n' "$original_rc" "$(date -u +%FT%TZ)"
  exec 1>&3 2>&4
  exec 3>&- 4>&-
  wait "$DRIVER_TEE_PID"; tee_rc=$?
  [[ $tee_rc -eq 0 ]] || finalizer_error=1
  snapshot post || finalizer_error=1
  apps=$(compute_apps) || finalizer_error=1
  [[ -z "$apps" ]] || finalizer_error=1
  post_uuid=$(gpu_uuid) || finalizer_error=1
  [[ -n ${EXPECTED_GPU_UUID:-} && "$post_uuid" == "${EXPECTED_GPU_UUID:-}" ]] || finalizer_error=1
  cleanup_runtime || { cleanup_rc=1; finalizer_error=1; }
  manifest=${ARTIFACT_DIR}/outputs-manifest-job${SLURM_JOB_ID}.sha256
  output_list=${ARTIFACT_DIR}/outputs-list-job${SLURM_JOB_ID}.nul
  if ! (set -o pipefail; find "$ARTIFACT_DIR" -maxdepth 1 -type f ! -name "$(basename "$manifest")" ! -name 'final-status-job*.txt' ! -name 'final-status-job*.sha256' ! -name "$(basename "$output_list")" -print0 | sort -z > "$output_list"); then
    manifest_rc=1
  elif [[ ! -s "$output_list" ]]; then
    manifest_rc=1
  elif ! xargs -0 -r sha256sum < "$output_list" > "$manifest"; then
    manifest_rc=1
  elif ! sha256sum -c "$manifest" >/dev/null; then
    manifest_rc=1
  fi
  rm -f -- "$output_list" || manifest_rc=1
  [[ $finalizer_error -eq 0 && $cleanup_rc -eq 0 && $manifest_rc -eq 0 ]] || rc=2
  printf 'FINAL_RC=%s ORIGINAL_RC=%s FINALIZER_ERROR=%s TEE_RC=%s RUNTIME_CLEANUP_RC=%s MANIFEST_RC=%s EXPECTED_UUID=%s POST_UUID=%s POST_APPS_EMPTY=%s UTC=%s\n' \
    "$rc" "$original_rc" "$finalizer_error" "$tee_rc" "$cleanup_rc" "$manifest_rc" "${EXPECTED_GPU_UUID:-unset}" "${post_uuid:-unset}" "$([[ -z "$apps" ]] && echo true || echo false)" "$(date -u +%FT%TZ)" > "${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt"
  sha256sum "${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt" "${ARTIFACT_DIR}/gpu-post-job${SLURM_JOB_ID}.txt" > "${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.sha256" && sha256sum -c "${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.sha256" >/dev/null || rc=2
  exit "$rc"
}

[[ ! -e "$ARTIFACT_DIR" && ! -L "$ARTIFACT_DIR" ]]
[[ ! -e "$ARTIFACT_ROOT" || ( -d "$ARTIFACT_ROOT" && ! -L "$ARTIFACT_ROOT" && "$(readlink -f -- "$ARTIFACT_ROOT")" == "$ARTIFACT_ROOT" ) ]]
mkdir -p -m 0700 -- "$ARTIFACT_ROOT" "$ARTIFACT_DIR"
safe_runtime
export TMPDIR=$RUNTIME TMP=$RUNTIME TEMP=$RUNTIME PYTHONPYCACHEPREFIX=$RUNTIME/pycache
exec 3>&1 4>&2
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1
DRIVER_TEE_PID=$!
trap finish EXIT
SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")
require_hash "$SCRIPT_PATH" "$C2_EXPECTED_RUNTIME_SCRIPT_SHA" runtime-script
require_hash "$SUBMITTED_SCRIPT" "$C2_EXPECTED_RUNTIME_SCRIPT_SHA" submitted-script
cmp -s "$SCRIPT_PATH" "$SUBMITTED_SCRIPT"

snapshot pre
gpu_line=$(nvidia-smi --id "$(selector)" --query-gpu=name,compute_cap --format=csv,noheader,nounits)
EXPECTED_GPU_UUID=$(gpu_uuid)
apps_before=$(compute_apps)
[[ "$gpu_line" == *B300* && "$gpu_line" == *10.3* && "$EXPECTED_GPU_UUID" =~ ^GPU- && -z "$apps_before" ]]
[[ -x "$PYTHON" && -f "$BASE_HARNESS" ]]
require_hash "$BASE_HARNESS" "$EXPECTED_BASE_SHA" base-harness

# Bind all production bytes before any Python or NCU invocation.
pin() { require_hash "$2" "$3" "$1"; printf '%s  %s\n' "$3" "$2" >> "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"; }
: > "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
pin base-harness "$BASE_HARNESS" "$EXPECTED_BASE_SHA"
pin v12-library "$V12_LIBRARY" "$EXPECTED_V12_LIBRARY_SHA"; pin v12-source "$V12_SOURCE" "$EXPECTED_V12_SOURCE_SHA"; pin v12-resource "$V12_RESOURCE" "$EXPECTED_V12_RESOURCE_SHA"; pin v12-provenance "$V12_PROVENANCE" "$EXPECTED_V12_PROVENANCE_SHA"; pin v12-aot-final "$V12_AOT_FINAL" "$EXPECTED_V12_AOT_FINAL_SHA"; pin v12-aot-sidecar "$V12_AOT_SIDECAR" "$EXPECTED_V12_AOT_SIDECAR_SHA"; pin v12-aot-outputs "$V12_AOT_OUTPUTS" "$EXPECTED_V12_AOT_OUTPUTS_SHA"; pin v12-aot-submitted "$V12_AOT_SUBMITTED" "$EXPECTED_V12_AOT_SUBMITTED_SHA"; pin v12-directed-result "$V12_DIRECTED_RESULT" "$EXPECTED_V12_DIRECTED_RESULT_SHA"; pin v12-directed-final "$V12_DIRECTED_FINAL" "$EXPECTED_V12_DIRECTED_FINAL_SHA"; pin v12-directed-sidecar "$V12_DIRECTED_SIDECAR" "$EXPECTED_V12_DIRECTED_SIDECAR_SHA"; pin v12-directed-outputs "$V12_DIRECTED_OUTPUTS" "$EXPECTED_V12_DIRECTED_OUTPUTS_SHA"; pin v12-directed-run "$V12_DIRECTED_RUN" "$EXPECTED_V12_DIRECTED_RUN_SHA"; pin v12-directed-submitted "$V12_DIRECTED_SUBMITTED" "$EXPECTED_V12_DIRECTED_SUBMITTED_SHA"; pin v12-decision "$V12_DECISION" "$EXPECTED_V12_DECISION_SHA"; pin v12-aggregate "$V12_AGGREGATE" "$EXPECTED_V12_AGGREGATE_SHA"; pin v12-stress-final "$V12_STRESS_FINAL" "$EXPECTED_V12_STRESS_FINAL_SHA"; pin v12-stress-sidecar "$V12_STRESS_SIDECAR" "$EXPECTED_V12_STRESS_SIDECAR_SHA"; pin v12-stress-outputs "$V12_STRESS_OUTPUTS" "$EXPECTED_V12_STRESS_OUTPUTS_SHA"
pin v14-library "$V14_LIBRARY" "$EXPECTED_V14_LIBRARY_SHA"; pin v14-source "$V14_SOURCE" "$EXPECTED_V14_SOURCE_SHA"; pin v14-resource "$V14_RESOURCE" "$EXPECTED_V14_RESOURCE_SHA"; pin v14-provenance "$V14_PROVENANCE" "$EXPECTED_V14_PROVENANCE_SHA"; pin v14-aot-final "$V14_AOT_FINAL" "$EXPECTED_V14_AOT_FINAL_SHA"; pin v14-aot-sidecar "$V14_AOT_SIDECAR" "$EXPECTED_V14_AOT_SIDECAR_SHA"; pin v14-aot-outputs "$V14_AOT_OUTPUTS" "$EXPECTED_V14_AOT_OUTPUTS_SHA"; pin v14-aot-submitted "$V14_AOT_SUBMITTED" "$EXPECTED_V14_AOT_SUBMITTED_SHA"; pin v14-directed-result "$V14_DIRECTED_RESULT" "$EXPECTED_V14_DIRECTED_RESULT_SHA"; pin v14-directed-final "$V14_DIRECTED_FINAL" "$EXPECTED_V14_DIRECTED_FINAL_SHA"; pin v14-directed-sidecar "$V14_DIRECTED_SIDECAR" "$EXPECTED_V14_DIRECTED_SIDECAR_SHA"; pin v14-directed-outputs "$V14_DIRECTED_OUTPUTS" "$EXPECTED_V14_DIRECTED_OUTPUTS_SHA"; pin v14-directed-run "$V14_DIRECTED_RUN" "$EXPECTED_V14_DIRECTED_RUN_SHA"; pin v14-directed-submitted "$V14_DIRECTED_SUBMITTED" "$EXPECTED_V14_DIRECTED_SUBMITTED_SHA"; pin v14-decision "$V14_DECISION" "$EXPECTED_V14_DECISION_SHA"; pin v14-aggregate "$V14_AGGREGATE" "$EXPECTED_V14_AGGREGATE_SHA"; pin v14-stress-final "$V14_STRESS_FINAL" "$EXPECTED_V14_STRESS_FINAL_SHA"; pin v14-stress-sidecar "$V14_STRESS_SIDECAR" "$EXPECTED_V14_STRESS_SIDECAR_SHA"; pin v14-stress-outputs "$V14_STRESS_OUTPUTS" "$EXPECTED_V14_STRESS_OUTPUTS_SHA"
pin v15-library "$V15_LIBRARY" "$EXPECTED_V15_LIBRARY_SHA"; pin v15-source "$V15_SOURCE" "$EXPECTED_V15_SOURCE_SHA"; pin v15-resource "$V15_RESOURCE" "$EXPECTED_V15_RESOURCE_SHA"; pin v15-provenance "$V15_PROVENANCE" "$EXPECTED_V15_PROVENANCE_SHA"; pin v15-aot-final "$V15_AOT_FINAL" "$EXPECTED_V15_AOT_FINAL_SHA"; pin v15-aot-sidecar "$V15_AOT_SIDECAR" "$EXPECTED_V15_AOT_SIDECAR_SHA"; pin v15-aot-outputs "$V15_AOT_OUTPUTS" "$EXPECTED_V15_AOT_OUTPUTS_SHA"; pin v15-aot-submitted "$V15_AOT_SUBMITTED" "$EXPECTED_V15_AOT_SUBMITTED_SHA"; pin v15-directed-result "$V15_DIRECTED_RESULT" "$EXPECTED_V15_DIRECTED_RESULT_SHA"; pin v15-directed-final "$V15_DIRECTED_FINAL" "$EXPECTED_V15_DIRECTED_FINAL_SHA"; pin v15-directed-sidecar "$V15_DIRECTED_SIDECAR" "$EXPECTED_V15_DIRECTED_SIDECAR_SHA"; pin v15-directed-outputs "$V15_DIRECTED_OUTPUTS" "$EXPECTED_V15_DIRECTED_OUTPUTS_SHA"; pin v15-directed-run "$V15_DIRECTED_RUN" "$EXPECTED_V15_DIRECTED_RUN_SHA"; pin v15-directed-submitted "$V15_DIRECTED_SUBMITTED" "$EXPECTED_V15_DIRECTED_SUBMITTED_SHA"; pin v15-decision "$V15_DECISION" "$EXPECTED_V15_DECISION_SHA"; pin v15-aggregate "$V15_AGGREGATE" "$EXPECTED_V15_AGGREGATE_SHA"; pin v15-stress-final "$V15_STRESS_FINAL" "$EXPECTED_V15_STRESS_FINAL_SHA"; pin v15-stress-sidecar "$V15_STRESS_SIDECAR" "$EXPECTED_V15_STRESS_SIDECAR_SHA"; pin v15-stress-outputs "$V15_STRESS_OUTPUTS" "$EXPECTED_V15_STRESS_OUTPUTS_SHA"
pin runtime-script "$SCRIPT_PATH" "$C2_EXPECTED_RUNTIME_SCRIPT_SHA"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"

"$PYTHON" - "$V12_LIBRARY" "$V12_SOURCE" "$V12_RESOURCE" "$V12_PROVENANCE" "$V12_AOT_FINAL" "$V12_AOT_SIDECAR" "$V12_AOT_OUTPUTS" "$V12_AOT_SUBMITTED" "$V12_DIRECTED_RESULT" "$V12_DIRECTED_FINAL" "$V12_DIRECTED_SIDECAR" "$V12_DIRECTED_OUTPUTS" "$V12_DIRECTED_RUN" "$V12_DIRECTED_SUBMITTED" "$V12_DECISION" "$V12_AGGREGATE" "$V12_STRESS_FINAL" "$V12_STRESS_SIDECAR" "$V12_STRESS_OUTPUTS" "$V14_LIBRARY" "$V14_SOURCE" "$V14_RESOURCE" "$V14_PROVENANCE" "$V14_AOT_FINAL" "$V14_AOT_SIDECAR" "$V14_AOT_OUTPUTS" "$V14_AOT_SUBMITTED" "$V14_DIRECTED_RESULT" "$V14_DIRECTED_FINAL" "$V14_DIRECTED_SIDECAR" "$V14_DIRECTED_OUTPUTS" "$V14_DIRECTED_RUN" "$V14_DIRECTED_SUBMITTED" "$V14_DECISION" "$V14_AGGREGATE" "$V14_STRESS_FINAL" "$V14_STRESS_SIDECAR" "$V14_STRESS_OUTPUTS" "$V15_LIBRARY" "$V15_SOURCE" "$V15_RESOURCE" "$V15_PROVENANCE" "$V15_AOT_FINAL" "$V15_AOT_SIDECAR" "$V15_AOT_OUTPUTS" "$V15_AOT_SUBMITTED" "$V15_DIRECTED_RESULT" "$V15_DIRECTED_FINAL" "$V15_DIRECTED_SIDECAR" "$V15_DIRECTED_OUTPUTS" "$V15_DIRECTED_RUN" "$V15_DIRECTED_SUBMITTED" "$V15_DECISION" "$V15_AGGREGATE" "$V15_STRESS_FINAL" "$V15_STRESS_SIDECAR" "$V15_STRESS_OUTPUTS" <<'PY'
import hashlib,json,sys
from pathlib import Path
xs=list(map(Path,sys.argv[1:])); assert len(xs)==57
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def status(path, expected):
 d=dict(x.split('=',1) for x in path.read_text().split() if '=' in x); assert d.get('FINAL_RC')==str(expected)
def manifest(path):
 out={}
 for line in path.read_text().splitlines():
  h,sep,name=line.partition('  '); assert sep and len(h)==64; out[Path(name).resolve()]=h
 return out
for offset,version,resource in ((0,'v12',(127,31136)),(19,'v14',(118,33184)),(38,'v15',(127,31392))):
 lib,src,res,prov,af,aside,aout,asub,dres,df,dside,dout,drun,dsub,dec,agg,sf,sside,sout=xs[offset:offset+19]
 r=res.read_text(); assert all(f'{k}={v}' in r for k,v in (('REG',resource[0]),('STACK',0),('SHARED',resource[1]),('LOCAL',0)))
 pm=manifest(prov)
 assert pm.get(lib.resolve())==digest(lib), f'{version}: provenance lacks exact DSO path/digest'
 assert any(h==digest(src) for h in pm.values()), f'{version}: provenance lacks source digest (path may be build checkout)'
 status(af,0); assert manifest(aside).get(af.resolve())==digest(af); am=manifest(aout); assert all(am.get(p.resolve())==digest(p) for p in (lib,src,res,prov))
 status(df,0); assert manifest(dside).get(df.resolve())==digest(df); dm=manifest(dout); assert all(dm.get(p.resolve())==digest(p) for p in (dres,drun))
 result=json.loads(dres.read_text()); assert result.get('all_gates_pass') is True and result.get('dispatch')=='direct_torch.ops._C.native_c2_msa_decode' and Path(result.get('operator_library')).resolve()==lib.resolve()
 status(sf,0 if version=='v12' else 3); assert manifest(sside).get(sf.resolve())==digest(sf)
 d=json.loads(dec.read_text()); a=json.loads(agg.read_text()); assert d.get('aggregate_sha256')==digest(agg) and d.get('all_correctness_and_resource_identity_gates_pass') is True
 assert d.get('accepted') is (version=='v12') and a.get('process_isolation',{}).get('same_python_process') is False
 def finite_vector(name, positive):
  values=a.get(name); assert isinstance(values,list) and len(values)==8, f'{version}: {name} must contain eight entries'
  assert all(isinstance(x,(int,float)) and not isinstance(x,bool) and __import__('math').isfinite(float(x)) and (float(x)>0 if positive else True) for x in values), f'{version}: invalid {name}'
 finite_vector('candidate_seed_mean_ms',True); finite_vector('reference_seed_mean_ms',True); finite_vector('paired_seed_improvement_fraction',False)
 if version!='v12':
  seeds=a.get('seeds'); assert isinstance(seeds,list) and len(seeds)==8 and all(isinstance(x,int) and not isinstance(x,bool) for x in seeds) and len(set(seeds))==8, f'{version}: seeds must be eight unique integer entries'
PY

command -v ncu; ncu --version | tee "${ARTIFACT_DIR}/ncu-version-job${SLURM_JOB_ID}.txt"
METRIC_CATALOG=${ARTIFACT_DIR}/metric-catalog-job${SLURM_JOB_ID}.txt
grep_args=(); for metric in "${metrics[@]}"; do grep_args+=(--regexp="$metric"); done
timeout --signal=TERM --kill-after=15s 180s ncu --query-metrics --query-metrics-mode all | grep --fixed-strings "${grep_args[@]}" > "$METRIC_CATALOG"
"$PYTHON" - "$METRIC_CATALOG" "${metrics[@]}" <<'PY'
import re,sys
s=open(sys.argv[1],encoding='utf-8',errors='replace').read(); available=set(re.findall(r'\b[A-Za-z][A-Za-z0-9_]*(?:__[A-Za-z0-9_]+)+(?:\.[A-Za-z0-9_]+)+\b',s)); wanted=sys.argv[2:]
assert len(wanted)==15 and len(set(wanted))==15 and not [x for x in wanted if x not in available]
PY

TARGET_RUNNER=${ARTIFACT_DIR}/ncu_target_runner.py
cat > "$TARGET_RUNNER" <<'PY'
import argparse,hashlib,importlib.util,json,os,platform,subprocess,sys,traceback
from pathlib import Path
import torch
def h(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def base(p):
 s=importlib.util.spec_from_file_location('c2_ncu_harness',p); assert s and s.loader; m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m)
 for n in ('_BATCH','_QUERY_HEADS','_HEAD_DIM','_make_inputs','_oracle','_validate_args'): assert hasattr(m,n),n
 return m
def th(t):
 t=t.detach().contiguous().cpu(); x=hashlib.sha256(); x.update(str(t.dtype).encode()); x.update(repr(tuple(t.shape)).encode()); x.update(t.view(torch.uint8).numpy().tobytes()); return x.hexdigest()
def uuid():
 v=os.environ.get('CUDA_VISIBLE_DEVICES','').split(','); assert len(v)==1 and v[0]
 r=subprocess.check_output(['nvidia-smi','--id',v[0],'--query-gpu=uuid','--format=csv,noheader,nounits'],text=True).strip(); assert r.startswith('GPU-'); return r
def call(op,out,ins,c): return op(out,*ins,c.scale,c.q_scale,c.k_scale,c.v_scale)
def record(a):
 out={'schema':'c2-native-v12-v14-v15-ncu-target-v1','version':a.version,'mode':a.mode,'all_gates_pass':False}
 try:
  assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(10,3)
  assert Path(a.library).is_absolute() and h(a.library)==a.library_sha256 and h(a.base_harness)==a.base_harness_sha256
  b=base(a.base_harness); c=b._validate_args(type('Args',(),{'num_physical_pages':64,'max_logical_pages':32,'scale':1/(128**.5),'q_scale':.25,'k_scale':.25,'v_scale':.5,'atol':1e-4,'rtol':1e-3})())
  torch.ops.load_library(a.library); assert torch._C._dispatch_has_kernel_for_dispatch_key('_C::native_c2_msa_decode','CUDA')
  ins=b._make_inputs(c,a.seed); y=torch.full((b._BATCH,b._QUERY_HEADS,b._HEAD_DIM),float('nan'),device='cuda',dtype=torch.bfloat16); p=y.data_ptr(); ret=call(torch.ops._C.native_c2_msa_decode,y,ins,c); torch.cuda.synchronize()
  checks={'return_is_none':ret is None,'pointer_unchanged':y.data_ptr()==p,'finite_output':bool(torch.isfinite(y).all().item())}
  out.update({'operator_library':str(Path(a.library).resolve()),'operator_library_sha256':h(a.library),'base_harness_sha256':h(a.base_harness),'seed':a.seed,'input_tensor_sha256':[th(x) for x in ins],'contract':{'batch':b._BATCH,'head_dim':b._HEAD_DIM,'kv_heads':4,'q_heads':b._QUERY_HEADS,'page_size':128,'topk':16,'num_physical_pages':64,'max_logical_pages':32,'scale':c.scale,'q_scale':c.q_scale,'k_scale':c.k_scale,'v_scale':c.v_scale},'caller_output':checks,'device':{'name':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),'gpu_uuid':uuid()},'environment':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda},'dispatch':'direct_torch.ops._C.native_c2_msa_decode','no_monkeypatch':True})
  out['input_manifest_sha256']=hashlib.sha256(''.join(out['input_tensor_sha256']).encode()).hexdigest()
  if a.mode=='ncu': out['logical_dispatch_actions']=1; out['all_gates_pass']=all(checks.values()); print(json.dumps(out,sort_keys=True)); return
  ref=b._oracle(*ins,c,torch.float64); actual=y.to(torch.float64); z=torch.full_like(y,float('nan')); q=z.data_ptr()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof: rr=call(torch.ops._C.native_c2_msa_decode,z,ins,c); torch.cuda.synchronize()
  ev=prof.key_averages(); ds=[e for e in ev if str(e.key)=='_C::native_c2_msa_decode']; ks=[e for e in ev if 'native_c2_msa_decode_kernel' in str(e.key)]
  out['correctness']={'allclose':bool(torch.allclose(actual,ref,atol=1e-4,rtol=1e-3)),'oracle_dtype':'float64'}; out['profiler']={'exactly_one_dispatcher_event':len(ds)==1 and ds[0].count==1,'exactly_one_native_kernel_event':len(ks)==1 and ks[0].count==1,'return_is_none':rr is None,'pointer_unchanged':z.data_ptr()==q}; out['all_gates_pass']=all(checks.values()) and out['correctness']['allclose'] and all(out['profiler'].values()); print(json.dumps(out,sort_keys=True))
 except Exception as e: out['error']=str(e); out['traceback']=traceback.format_exc(); print(json.dumps(out,sort_keys=True)); raise
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--mode',choices=('validate','ncu'),required=True); p.add_argument('--version',choices=('v12','v14','v15'),required=True); p.add_argument('--library',type=Path,required=True); p.add_argument('--library-sha256',required=True); p.add_argument('--base-harness',type=Path,required=True); p.add_argument('--base-harness-sha256',required=True); p.add_argument('--seed',type=int,required=True); record(p.parse_args())
PY
chmod 0444 "$TARGET_RUNNER"; "$PYTHON" -m py_compile "$TARGET_RUNNER"
run_preflight() { timeout --foreground --preserve-status --kill-after=15s 300s "$PYTHON" "$TARGET_RUNNER" --mode validate --version "$1" --library "$2" --library-sha256 "$3" --base-harness "$BASE_HARNESS" --base-harness-sha256 "$EXPECTED_BASE_SHA" --seed 20260829 > "$4" 2>&1; }
V12_PREFLIGHT=${ARTIFACT_DIR}/v12-preflight-job${SLURM_JOB_ID}.json; V14_PREFLIGHT=${ARTIFACT_DIR}/v14-preflight-job${SLURM_JOB_ID}.json; V15_PREFLIGHT=${ARTIFACT_DIR}/v15-preflight-job${SLURM_JOB_ID}.json
run_preflight v12 "$V12_LIBRARY" "$EXPECTED_V12_LIBRARY_SHA" "$V12_PREFLIGHT"
[[ -z $(compute_apps) ]]
run_preflight v14 "$V14_LIBRARY" "$EXPECTED_V14_LIBRARY_SHA" "$V14_PREFLIGHT"
[[ -z $(compute_apps) ]]
run_preflight v15 "$V15_LIBRARY" "$EXPECTED_V15_LIBRARY_SHA" "$V15_PREFLIGHT"
[[ -z $(compute_apps) ]]
"$PYTHON" - "$EXPECTED_GPU_UUID" "$V12_PREFLIGHT" "$V14_PREFLIGHT" "$V15_PREFLIGHT" <<'PY'
import json,sys
from pathlib import Path
u=sys.argv[1]; rows=[]
for version,path in zip(('v12','v14','v15'),map(Path,sys.argv[2:])):
 xs=[json.loads(x) for x in path.read_text().splitlines() if x.startswith('{')]; assert len(xs)==1; x=xs[0]; assert x.get('version')==version and x.get('mode')=='validate' and x.get('all_gates_pass') is True and x['device']['gpu_uuid']==u; rows.append(x)
for k in ('input_tensor_sha256','input_manifest_sha256','contract','base_harness_sha256'): assert len({json.dumps(x[k],sort_keys=True) for x in rows})==1
PY

profile_one() { timeout --signal=TERM --kill-after=30s 300s ncu --target-processes all --kernel-name-base function --kernel-name 'regex:.*native_c2_msa_decode_kernel.*' --launch-skip 0 --launch-count 1 --clock-control none --metrics "$metrics_csv" --force-overwrite --export "$3" "$PYTHON" "$TARGET_RUNNER" --mode ncu --version "$1" --library "$2" --library-sha256 "$4" --base-harness "$BASE_HARNESS" --base-harness-sha256 "$EXPECTED_BASE_SHA" --seed 20260829 > "$5" 2>&1; [[ -s "$3" && -s "$5" ]]; }
V12_REPORT=${ARTIFACT_DIR}/native-c2-v12-job${SLURM_JOB_ID}.ncu-rep; V14_REPORT=${ARTIFACT_DIR}/native-c2-v14-job${SLURM_JOB_ID}.ncu-rep; V15_REPORT=${ARTIFACT_DIR}/native-c2-v15-job${SLURM_JOB_ID}.ncu-rep
V12_TARGET=${ARTIFACT_DIR}/v12-ncu-target-job${SLURM_JOB_ID}.log; V14_TARGET=${ARTIFACT_DIR}/v14-ncu-target-job${SLURM_JOB_ID}.log; V15_TARGET=${ARTIFACT_DIR}/v15-ncu-target-job${SLURM_JOB_ID}.log
# Strictly serialized: identical schema registrations stay in separate Python processes.
profile_one v12 "$V12_LIBRARY" "$V12_REPORT" "$EXPECTED_V12_LIBRARY_SHA" "$V12_TARGET"; [[ -z $(compute_apps) ]]
profile_one v14 "$V14_LIBRARY" "$V14_REPORT" "$EXPECTED_V14_LIBRARY_SHA" "$V14_TARGET"; [[ -z $(compute_apps) ]]
profile_one v15 "$V15_LIBRARY" "$V15_REPORT" "$EXPECTED_V15_LIBRARY_SHA" "$V15_TARGET"; [[ -z $(compute_apps) ]]
V12_RAW=${ARTIFACT_DIR}/native-c2-v12-job${SLURM_JOB_ID}.raw.csv; V14_RAW=${ARTIFACT_DIR}/native-c2-v14-job${SLURM_JOB_ID}.raw.csv; V15_RAW=${ARTIFACT_DIR}/native-c2-v15-job${SLURM_JOB_ID}.raw.csv
for pair in "$V12_REPORT:$V12_RAW" "$V14_REPORT:$V14_RAW" "$V15_REPORT:$V15_RAW"; do report=${pair%%:*}; raw=${pair#*:}; timeout --signal=TERM --kill-after=15s 120s ncu --import "$report" --csv --page raw --print-units base --print-fp > "$raw"; [[ -s "$raw" ]]; done

parse_one() {
  "$PYTHON" - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "${metrics[@]}" <<'PY'
import csv,json,math,sys
from collections import Counter
from pathlib import Path
raw,target,out,version,ereg,eshared,library,*wanted=sys.argv[1:]; ereg=float(ereg); eshared=float(eshared)
js=[json.loads(x) for x in Path(target).read_text(encoding='utf-8',errors='replace').splitlines() if x.startswith('{')]; assert len(js)==1; t=js[0]
assert t.get('schema')=='c2-native-v12-v14-v15-ncu-target-v1' and t.get('mode')=='ncu' and t.get('version')==version and t.get('all_gates_pass') is True and t.get('logical_dispatch_actions')==1 and Path(t.get('operator_library')).resolve()==Path(library).resolve()
rows=list(csv.reader(Path(raw).open(encoding='utf-8',errors='replace',newline=''))); launch=('launch__cluster_size','launch__cluster_max_potential_size','launch__cluster_max_active','launch__occupancy_cluster_pct','launch__occupancy_cluster_gpu_pct','launch__registers_per_thread','launch__shared_mem_per_block','launch__barrier_count')
i=next((j for j,r in enumerate(rows) if 'ID' in r and 'Kernel Name' in r and all(m in r for m in wanted)),None); assert i is not None
h=rows[i]; assert all(Counter(h)[x]==1 for x in ('ID','Kernel Name',*wanted,*launch)); idx={x:h.index(x) for x in ('ID','Kernel Name',*wanted,*launch)}; last=max(idx.values()); assert i+1<len(rows) and len(rows[i+1])>last
hits=[r for r in rows[i+2:] if len(r)>last and 'native_c2_msa_decode_kernel' in r[idx['Kernel Name']]]; assert len(hits)==1; row=hits[0]
def n(k):
 x=float(row[idx[k]].replace(',','')); assert math.isfinite(x) and x>=0; return x
vals={m:{'value':n(m),'unit':rows[i+1][idx[m]].strip()} for m in wanted}; meta={m:n(m) for m in launch}
assert vals['smsp__cycles_elapsed.sum']['value']>0 and vals['smsp__inst_executed_pipe_tensor.sum']['value']>0
assert meta['launch__registers_per_thread']==ereg and meta['launch__shared_mem_per_block']==eshared and meta['launch__cluster_size']==4 and meta['launch__barrier_count']==1
Path(out).write_text(json.dumps({'schema':'c2-native-v12-v14-v15-ncu-counters-v1','version':version,'all_collection_gates_pass':True,'kernel_name':row[idx['Kernel Name']],'action_id':row[idx['ID']].strip(),'matching_kernel_rows':1,'target':t,'metrics':vals,'launch_metadata':meta,'timing_valid_for_benchmark':False,'duration_metric_requested':False},indent=2,sort_keys=True)+'\n')
PY
}
V12_SUMMARY=${ARTIFACT_DIR}/native-c2-v12-ncu-job${SLURM_JOB_ID}.json; V14_SUMMARY=${ARTIFACT_DIR}/native-c2-v14-ncu-job${SLURM_JOB_ID}.json; V15_SUMMARY=${ARTIFACT_DIR}/native-c2-v15-ncu-job${SLURM_JOB_ID}.json
parse_one "$V12_RAW" "$V12_TARGET" "$V12_SUMMARY" v12 127 31136 "$V12_LIBRARY"
parse_one "$V14_RAW" "$V14_TARGET" "$V14_SUMMARY" v14 118 33184 "$V14_LIBRARY"
parse_one "$V15_RAW" "$V15_TARGET" "$V15_SUMMARY" v15 127 31392 "$V15_LIBRARY"

RESULT=${ARTIFACT_DIR}/v12-v14-v15-ncu-mechanism-job${SLURM_JOB_ID}.json
"$PYTHON" - "$RESULT" "$EXPECTED_GPU_UUID" "$V12_SUMMARY" "$V14_SUMMARY" "$V15_SUMMARY" "$V12_DECISION" "$V12_AGGREGATE" "$V12_STRESS_FINAL" "$V14_DECISION" "$V14_AGGREGATE" "$V14_STRESS_FINAL" "$V15_DECISION" "$V15_AGGREGATE" "$V15_STRESS_FINAL" <<'PY'
import hashlib,json,math,sys
from pathlib import Path
out=Path(sys.argv[1]); uuid=sys.argv[2]; summaries=list(map(Path,sys.argv[3:6])); history=list(map(Path,sys.argv[6:]))
rows=[json.loads(p.read_text()) for p in summaries]
assert [r['version'] for r in rows]==['v12','v14','v15'] and all(r.get('all_collection_gates_pass') is True and r.get('matching_kernel_rows')==1 and r.get('timing_valid_for_benchmark') is False and r.get('duration_metric_requested') is False for r in rows)
for k in ('input_tensor_sha256','input_manifest_sha256','contract','base_harness_sha256'):
 assert len({json.dumps(r['target'][k],sort_keys=True) for r in rows})==1
assert all(r['target']['device']['gpu_uuid']==uuid for r in rows)
metric_keys=set(rows[0]['metrics']); assert len(metric_keys)==15 and all(set(r['metrics'])==metric_keys for r in rows)
for key in metric_keys:
 assert len({r['metrics'][key]['unit'] for r in rows})==1, f'unit mismatch: {key}'
def ratio(a,b): return None if b==0 else a/b
ratios={}
for version,row in zip(('v14','v15'),rows[1:]):
 ratios[f'{version}_over_v12']={k:ratio(float(row['metrics'][k]['value']),float(rows[0]['metrics'][k]['value'])) for k in sorted(rows[0]['metrics'])}
assert all(v is None or math.isfinite(v) and v>=0 for table in ratios.values() for v in table.values())
ctx=[]
for decision,aggregate,final in zip(history[0::3],history[1::3],history[2::3],strict=True):
 d=json.loads(decision.read_text()); a=json.loads(aggregate.read_text()); f=final.read_text(); assert d['aggregate_sha256']==hashlib.sha256(aggregate.read_bytes()).hexdigest() and 'FINAL_RC=' in f
 ctx.append({'decision':str(decision.resolve()),'decision_sha256':hashlib.sha256(decision.read_bytes()).hexdigest(),'aggregate':str(aggregate.resolve()),'aggregate_sha256':hashlib.sha256(aggregate.read_bytes()).hexdigest(),'final_status':str(final.resolve()),'final_status_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'accepted':d['accepted'],'correctness_identity_gates_pass':d['all_correctness_and_resource_identity_gates_pass']})
out.write_text(json.dumps({'schema':'c2-native-v12-v14-v15-ncu-mechanism-v1','all_collection_and_comparison_gates_pass':True,'timing_valid_for_benchmark':False,'duration_metric_requested':False,'same_collection_contract':{'seed':20260829,'same_input_manifest_sha256':rows[0]['target']['input_manifest_sha256'],'same_harness_sha256':rows[0]['target']['base_harness_sha256'],'same_physical_gpu_uuid':uuid,'separate_python_processes':True,'single_native_kernel_action_each':True,'kernel_filter':'regex:.*native_c2_msa_decode_kernel.*','counter_catalog_size':15},'resources':{'v12':{'registers':127,'stack':0,'local':0,'shared_bytes':31136},'v14':{'registers':118,'stack':0,'local':0,'shared_bytes':33184},'v15':{'registers':127,'stack':0,'local':0,'shared_bytes':31392}},'absolute_counters':{r['version']:r['metrics'] for r in rows},'counter_ratios':ratios,'historical_context':ctx},indent=2,sort_keys=True)+'\n')
PY
sha256sum "$V12_PREFLIGHT" "$V14_PREFLIGHT" "$V15_PREFLIGHT" "$V12_REPORT" "$V14_REPORT" "$V15_REPORT" "$V12_RAW" "$V14_RAW" "$V15_RAW" "$V12_TARGET" "$V14_TARGET" "$V15_TARGET" "$V12_SUMMARY" "$V14_SUMMARY" "$V15_SUMMARY" "$RESULT" "$TARGET_RUNNER" "$METRIC_CATALOG" "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" > "${ARTIFACT_DIR}/outputs-before-post-job${SLURM_JOB_ID}.sha256"
