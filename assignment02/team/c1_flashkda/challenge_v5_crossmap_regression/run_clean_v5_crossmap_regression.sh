#!/usr/bin/env bash
# Canonical clean-allocation entrypoint for the v5 cross-map regression.
set -Eeuo pipefail

die() { echo "v5-crossmap: $*" >&2; exit 96; }
sha_of() { sha256sum "$1" | awk '{print $1}'; }
check_sha() { [[ -f "$1" && "$(sha_of "$1")" == "$2" ]] || { echo "v5-crossmap: SHA gate failed for $3" >&2; return 1; }; }
clean_gpu() {
  local phase="$1" line apps index uuid name capability memory
  line="$(nvidia-smi --query-gpu=index,uuid,name,compute_cap,memory.used --format=csv,noheader,nounits)" || return 1
  [[ "$(wc -l <<<"$line")" -eq 1 ]] || return 1
  IFS=',' read -r index uuid name capability memory <<<"$line"
  [[ "${memory// /}" == "0" ]] || return 1
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | sed '/No running compute processes found/d;/^[[:space:]]*$/d')" || return 1
  [[ -z "$apps" ]] || return 1
  echo "V5_CROSSMAP_${phase}_CLEAN_GPU uuid=${uuid// /} memory_mib=0"
}
[[ $# -eq 1 && "$1" == "--authorized-by-parent" ]] || die "requires exactly --authorized-by-parent"
[[ "${C1_V5_CROSSMAP_GPU_AUTHORIZED:-}" == "1" ]] || die "parent GPU authorization missing"

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
[[ -n "${CANONICAL_PROTOCOL_SHELL:-}" ]] || die "CANONICAL_PROTOCOL_SHELL is required"
[[ "$SELF" == "$(readlink -f "$CANONICAL_PROTOCOL_SHELL")" ]] || die "shell was not executed from its canonical path"
[[ -n "${EXPECTED_PROTOCOL_SHELL_SHA256:-}" ]] || die "external shell SHA is required"
[[ "$(sha256sum "$SELF" | awk '{print $1}')" == "$EXPECTED_PROTOCOL_SHELL_SHA256" ]] || die "canonical shell SHA mismatch"

for required in A02_ROOT PATCHED_ROOT REFERENCE_ROOT FLA_ROOT PYTHON_BIN ALLOCATION_ID LABEL \
  C1_PINNED_REFERENCE_HELPER_PATH C1_PINNED_REFERENCE_HELPER_SHA256 \
  C1_V5_CROSSMAP_RUNNER_SHA256 C1_V5_CROSSMAP_ANALYZER_SHA256 \
  C1_V5_CROSSMAP_AUTO_DISPATCH_SHA256 C1_V5_CROSSMAP_FLA_BACKEND_SHA256; do
  [[ -n "${!required:-}" ]] || die "$required is required"
done
[[ "${ALLOCATION_ID}" == "A1" || "${ALLOCATION_ID}" == "A2" ]] || die "ALLOCATION_ID must be A1 or A2"
[[ "${LABEL}" =~ ^[A-Za-z0-9._-]+$ ]] || die "LABEL contains unsafe characters"
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || die "must run inside a positive-decimal Slurm job"
[[ "$C1_PINNED_REFERENCE_HELPER_PATH" == "/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so" ]] || die "helper path drift"
[[ "$C1_PINNED_REFERENCE_HELPER_SHA256" == "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f" ]] || die "helper SHA declaration drift"
[[ "$(sha256sum "$C1_PINNED_REFERENCE_HELPER_PATH" | awk '{print $1}')" == "$C1_PINNED_REFERENCE_HELPER_SHA256" ]] || die "helper bytes drift"
[[ "$C1_V5_CROSSMAP_AUTO_DISPATCH_SHA256" == "9633179c6a5e8ce61b1857a102f81074c6bdfdc63163e13980f40c24d6583c29" ]] || die "auto_dispatch SHA declaration drift"
[[ "$C1_V5_CROSSMAP_FLA_BACKEND_SHA256" == "152ac4ba897c1887963553d592456b7f988c89105e6f6cb507a44757058d5fd1" ]] || die "fla_backend SHA declaration drift"

DIR="$A02_ROOT/team/c1_flashkda/challenge_v5_crossmap_regression"
RUNNER="$DIR/run_v5_crossmap_regression.py"
ANALYZER="$DIR/analyze_v5_crossmap_regression.py"
AUTO="$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py"
BACKEND="$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py"
[[ "$SELF" == "$(readlink -f "$DIR/run_clean_v5_crossmap_regression.sh")" ]] || die "canonical directory mismatch"
[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable"
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "$C1_V5_CROSSMAP_RUNNER_SHA256" ]] || die "runner external SHA mismatch"
[[ "$(sha256sum "$ANALYZER" | awk '{print $1}')" == "$C1_V5_CROSSMAP_ANALYZER_SHA256" ]] || die "analyzer external SHA mismatch"
[[ "$(sha256sum "$AUTO" | awk '{print $1}')" == "$C1_V5_CROSSMAP_AUTO_DISPATCH_SHA256" ]] || die "auto_dispatch bytes drift"
[[ "$(sha256sum "$BACKEND" | awk '{print $1}')" == "$C1_V5_CROSSMAP_FLA_BACKEND_SHA256" ]] || die "fla_backend bytes drift"

attest_sources() {
  check_sha "$RUNNER" "$C1_V5_CROSSMAP_RUNNER_SHA256" runner &&
  check_sha "$ANALYZER" "$C1_V5_CROSSMAP_ANALYZER_SHA256" analyzer &&
  check_sha "$SELF" "$EXPECTED_PROTOCOL_SHELL_SHA256" canonical_shell &&
  check_sha "$AUTO" "9633179c6a5e8ce61b1857a102f81074c6bdfdc63163e13980f40c24d6583c29" auto_dispatch &&
  check_sha "$BACKEND" "152ac4ba897c1887963553d592456b7f988c89105e6f6cb507a44757058d5fd1" fla_backend &&
  check_sha "$C1_PINNED_REFERENCE_HELPER_PATH" "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f" pinned_sigmoid_ext &&
  check_sha "$A02_ROOT/team/c1_flashkda/challenge_varlen_dispatch/run_varlen_fla_handoff_candidate.py" "e07481e7e66551f1f9eaafa7f1fb9db9ef193f1694ff831ee0a07802c8ffbe14" varlen_helper &&
  check_sha "$A02_ROOT/team/c1_flashkda/challenge_tail8191_production_freeze/run_tail8191_production_freeze.py" "f4144f5fbdd61396ff907c6290b767b5570e04d19087f8332f9db10e56e7b1dc" tail_helper &&
  check_sha "$A02_ROOT/team/c1_flashkda/challenge_seqcount_dispatch/run_seqcount_dispatch.py" "4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f" shared_seqcount &&
  check_sha "$A02_ROOT/team/c1_flashkda/challenge_varlen_dispatch/run_varlen_dispatch_confirmation.py" "9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b" confirmation &&
  check_sha "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52" harness &&
  check_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/varlen_metadata.py" "f89a97ba284c7a24a3df54efca7bb60eb70f9cbda659200bd6cb3e8dfdaf4ccd" varlen_metadata &&
  check_sha "$REFERENCE_ROOT/tests/torch_ref.py" "bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5" pinned_torch_ref &&
  check_sha "$FLA_ROOT/fla/__init__.py" "b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d" fla_init &&
  check_sha "$FLA_ROOT/fla/ops/backends/__init__.py" "a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635" fla_backends_init &&
  check_sha "$FLA_ROOT/fla/ops/kda/__init__.py" "24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb" fla_kda_init &&
  check_sha "$FLA_ROOT/fla/ops/kda/backends/__init__.py" "86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797" fla_kda_backends_init &&
  check_sha "$FLA_ROOT/fla/ops/kda/backends/flash_kda.py" "0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2" fla_flash_backend &&
  check_sha "$FLA_ROOT/fla/ops/kda/chunk.py" "a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8" fla_public_chunk
}
attest_sources || die "pre-workload source/helper ledger drift"
final_attestation() {
  local rc="$?" check_rc=0
  set +e
  attest_sources || check_rc=1
  clean_gpu FINAL || check_rc=1
  trap - EXIT
  if [[ "$rc" -ne 0 || "$check_rc" -ne 0 ]]; then exit 97; fi
}
trap final_attestation EXIT

if [[ "$ALLOCATION_ID" == "A1" ]]; then
  [[ -z "${A1_AUDIT:-}" && -z "${EXPECTED_A1_AUDIT_SHA256:-}" ]] || die "A1 must not receive A1 prerequisite"
else
  [[ -n "${A1_AUDIT:-}" && -n "${EXPECTED_A1_AUDIT_SHA256:-}" ]] || die "A2 requires externally recorded A1 audit path/SHA"
  [[ -f "$A1_AUDIT" ]] || die "A1 audit is missing"
  [[ "$(sha256sum "$A1_AUDIT" | awk '{print $1}')" == "$EXPECTED_A1_AUDIT_SHA256" ]] || die "A1 audit SHA mismatch"
fi

clean_gpu PRE || die "GPU must be 0 MiB with no compute process before workload"

export PYTHONPATH="$PATCHED_ROOT:$FLA_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export C1_V5_CROSSMAP_CLEAN_GPU=1 C1_B300_FLASH_KDA=1 FLA_FLASH_KDA=1 C1_B300_VARLEN_CPU_DESCRIPTOR=1
RESULTS="$DIR/results"
mkdir -p "$RESULTS"
MAIN0="$RESULTS/c1_v5_crossmap_${LABEL}_${ALLOCATION_ID}_main0.json"
MAIN1="$RESULTS/c1_v5_crossmap_${LABEL}_${ALLOCATION_ID}_main1.json"
AUDIT="$RESULTS/c1_v5_crossmap_${LABEL}_${ALLOCATION_ID}.allocation_audit.json"

for process in 0 1; do
  json_var="MAIN${process}"
  "$PYTHON_BIN" "$RUNNER" --allocation-id "$ALLOCATION_ID" --process-index "$process" \
    --reference-root "$REFERENCE_ROOT" --patched-root "$PATCHED_ROOT" --fla-root "$FLA_ROOT" \
    --analyzer-path "$ANALYZER" --protocol-shell-path "$SELF" --json "${!json_var}"
  clean_gpu "BETWEEN_${process}" || die "GPU did not return to 0 MiB/no-process between fresh PIDs"
done
MAIN0_SHA="$(sha256sum "$MAIN0" | awk '{print $1}')"
MAIN1_SHA="$(sha256sum "$MAIN1" | awk '{print $1}')"

allocation_args=(allocation --allocation-id "$ALLOCATION_ID" --current-slurm-job-id "$SLURM_JOB_ID" \
  --main0 "$MAIN0" --main1 "$MAIN1" --main0-sha256 "$MAIN0_SHA" --main1-sha256 "$MAIN1_SHA" \
  --expected-runner-sha256 "$C1_V5_CROSSMAP_RUNNER_SHA256" \
  --expected-analyzer-sha256 "$C1_V5_CROSSMAP_ANALYZER_SHA256" \
  --expected-protocol-shell-sha256 "$EXPECTED_PROTOCOL_SHELL_SHA256" --json "$AUDIT")
if [[ "$ALLOCATION_ID" == "A2" ]]; then
  allocation_args+=(--a1-audit "$A1_AUDIT" --expected-a1-audit-sha256 "$EXPECTED_A1_AUDIT_SHA256")
fi
"$PYTHON_BIN" "$ANALYZER" "${allocation_args[@]}"
clean_gpu POST || die "GPU did not return to 0 MiB/no-process after allocation audit"
echo "V5_CROSSMAP_AUDIT=$AUDIT"
echo "V5_CROSSMAP_AUDIT_SHA256=$(sha256sum "$AUDIT" | awk '{print $1}')"
