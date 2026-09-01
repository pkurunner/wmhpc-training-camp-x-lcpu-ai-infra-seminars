#!/usr/bin/env bash
# Freeze the completed v16 fresh-process / real-MiniMax-backend observation.
# This is a lean, fail-closed evidence archive: it preserves the submitted
# driver, all eight seed results and traces, identity/oracle evidence and job
# finalization records, but deliberately excludes the reproducible wheel and
# node-local runtime payload.
#
# Run only after copying this reviewed file to the exact remote path below:
#   C2_EXPECTED_ARCHIVE_SCRIPT_SHA=<this-file-sha256> \
#     bash /home/lcpu/85117379/archive_native_c2_v16_fresh_backend_20260831.sh

set -Eeuo pipefail
umask 077

: "${C2_EXPECTED_ARCHIVE_SCRIPT_SHA:?set the reviewed archive script SHA-256}"
[[ "${C2_EXPECTED_ARCHIVE_SCRIPT_SHA}" =~ ^[0-9a-f]{64}$ ]]

BASE=/home/lcpu/85117379
JOB_ID=${C2_FRESH_JOB_ID:-13900}
ROOT=${C2_FRESH_ARTIFACT_ROOT:-${BASE}/c2-native-plugin-v16-k-chunk-lookahead-fresh-backend-artifacts-20260831}
[[ "${JOB_ID}" == 13900 ]]
[[ "${ROOT}" == "${BASE}/c2-native-plugin-v16-k-chunk-lookahead-fresh-backend-artifacts-20260831" ]]

JOB_DIR=${ROOT}/job${JOB_ID}
SLURM_LOG=${ROOT}/slurm-${JOB_ID}.log
FRESH_SCRIPT=${BASE}/validate_native_c2_plugin_v16_k_chunk_lookahead_fresh_backend_8seeds_20260831.slurm
EXPECTED_ARCHIVE_SCRIPT=${BASE}/archive_native_c2_v16_fresh_backend_20260831.sh
ARCHIVE_SCRIPT=$(readlink -f -- "${BASH_SOURCE[0]}")

INPUTS=${JOB_DIR}/inputs-job${JOB_ID}.sha256
OUTPUTS=${JOB_DIR}/outputs-job${JOB_ID}.sha256
FINAL=${JOB_DIR}/final-status-job${JOB_ID}.txt
FINAL_SIDECAR=${JOB_DIR}/final-status-job${JOB_ID}.sha256
SUMMARY=${JOB_DIR}/plugin-v16-fresh-backend-summary-job${JOB_ID}.json
DECISION=${JOB_DIR}/plugin-v16-fresh-backend-decision-job${JOB_ID}.json

EXPECTED_FRESH_SCRIPT_SHA=4f934f8b4d47bd14b13f42c8032d4f69e8d70de54ff710beb4b1dcd68d198223
EXPECTED_HARNESS_SHA=7883edc25df48e3b69a9f4948775a1baafd3ba0b1bd17edcfeac6644aa7b4762
EXPECTED_PLUGIN_SHA=1da5f731da796656759f0e673e3479392b6b8337c054a61aa4ca1fd0afb4edd4
EXPECTED_STABLE_SHA=cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442
EXPECTED_WHEEL_SHA=3947fab41739c98a30a8fd5486b867347b932f3419def3bfbd846db458ba90a9
EXPECTED_STRESS_DECISION_SHA=3763ce31dbda5482896aa00c0192af1aca0ba8e322a4e555b52a6c7ad491bed9
EXPECTED_STRESS_AGGREGATE_SHA=95c456509add154bf11747a78b3b5203cc6580ad08bccf31cc562b3815172984
EXPECTED_LIFECYCLE_RESULT_SHA=b3c45988ad20daf352f598f74a8be287f20e417b12b1e8e6b18aee62ad991ade
EXPECTED_LIFECYCLE_ATTESTATION_SHA=f8860613c63950ef741c5ccb2e0f6ad413d798fb2ba3d65f2f92f685ed3020c9

ARCHIVE=${BASE}/c2-native-v16-k-chunk-lookahead-fresh-backend-job${JOB_ID}-lean-evidence-20260831.tar.gz
MANIFEST=${BASE}/c2-native-v16-k-chunk-lookahead-fresh-backend-job${JOB_ID}-lean-evidence-20260831.manifest.sha256
SIDECAR=${ARCHIVE}.sha256
LOCK=${BASE}/.c2-native-v16-fresh-backend-job${JOB_ID}-lean-archive-20260831.lock
MANIFEST_TMP=${MANIFEST}.tmp.${BASHPID}
ARCHIVE_TMP=${ARCHIVE}.tmp.${BASHPID}
SIDECAR_TMP=${SIDECAR}.tmp.${BASHPID}
EXPECTED_MEMBERS_TMP=${ARCHIVE}.expected-members.tmp.${BASHPID}
ACTUAL_MEMBERS_TMP=${ARCHIVE}.actual-members.tmp.${BASHPID}
ARCHIVE_MEMBER_HASHES_TMP=${ARCHIVE}.member-hashes.tmp.${BASHPID}
EXPECTED_JOB_NAMES_TMP=${ARCHIVE}.expected-job-names.tmp.${BASHPID}
ACTUAL_JOB_NAMES_TMP=${ARCHIVE}.actual-job-names.tmp.${BASHPID}

lock_acquired=0
published=0
manifest_linked=0
archive_linked=0
sidecar_linked=0
cleanup_outputs_and_lock() {
  local original_rc=$?
  trap - EXIT
  set +e
  # The sidecar is the publication commit marker.  Before that marker, unlink
  # only hard links created by this invocation; never overwrite old evidence.
  if (( original_rc != 0 && published == 0 )); then
    if (( sidecar_linked == 1 )) && [[ -f "${SIDECAR}" && -f "${SIDECAR_TMP}" && "${SIDECAR}" -ef "${SIDECAR_TMP}" ]]; then rm -f -- "${SIDECAR}"; fi
    if (( archive_linked == 1 )) && [[ -f "${ARCHIVE}" && -f "${ARCHIVE_TMP}" && "${ARCHIVE}" -ef "${ARCHIVE_TMP}" ]]; then rm -f -- "${ARCHIVE}"; fi
    if (( manifest_linked == 1 )) && [[ -f "${MANIFEST}" && -f "${MANIFEST_TMP}" && "${MANIFEST}" -ef "${MANIFEST_TMP}" ]]; then rm -f -- "${MANIFEST}"; fi
  fi
  for temporary in "${MANIFEST_TMP}" "${ARCHIVE_TMP}" "${SIDECAR_TMP}" "${EXPECTED_MEMBERS_TMP}" "${ACTUAL_MEMBERS_TMP}" "${ARCHIVE_MEMBER_HASHES_TMP}" "${EXPECTED_JOB_NAMES_TMP}" "${ACTUAL_JOB_NAMES_TMP}"; do
    [[ ! -e "${temporary}" && ! -L "${temporary}" ]] || [[ -f "${temporary}" && ! -L "${temporary}" ]] || continue
    rm -f -- "${temporary}"
  done
  if (( lock_acquired == 1 )); then rmdir -- "${LOCK}"; fi
  exit "${original_rc}"
}

require_regular_canonical() {
  local path=$1
  [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]]
}
require_directory_canonical() {
  local path=$1
  [[ -d "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]]
}

require_directory_canonical "${BASE}"
[[ "$(stat -c %u -- "${BASE}")" == "$(id -u)" ]]
require_directory_canonical "${ROOT}"
require_directory_canonical "${JOB_DIR}"
[[ "$(stat -c %u -- "${ROOT}")" == "$(id -u)" && "$(stat -c %u -- "${JOB_DIR}")" == "$(id -u)" ]]
[[ "${BASH_SOURCE[0]}" == "${EXPECTED_ARCHIVE_SCRIPT}" && "${ARCHIVE_SCRIPT}" == "${EXPECTED_ARCHIVE_SCRIPT}" ]]
for input in "${SLURM_LOG}" "${FRESH_SCRIPT}" "${ARCHIVE_SCRIPT}" "${INPUTS}" "${OUTPUTS}" "${FINAL}" "${FINAL_SIDECAR}" "${SUMMARY}" "${DECISION}"; do
  require_regular_canonical "${input}"
done

# Acquire the lock before creating, opening, or even checking any temporary
# file.  A competing invocation therefore fails at mkdir without writing an
# untracked file under BASE.
mkdir -m 0700 -- "${LOCK}"
lock_acquired=1
trap cleanup_outputs_and_lock EXIT
for output in "${MANIFEST}" "${ARCHIVE}" "${SIDECAR}"; do
  [[ ! -e "${output}" && ! -L "${output}" ]] || { printf 'refusing to reuse archive output: %s\n' "${output}" >&2; exit 2; }
done
for temporary in "${MANIFEST_TMP}" "${ARCHIVE_TMP}" "${SIDECAR_TMP}" "${EXPECTED_MEMBERS_TMP}" "${ACTUAL_MEMBERS_TMP}" "${ARCHIVE_MEMBER_HASHES_TMP}" "${EXPECTED_JOB_NAMES_TMP}" "${ACTUAL_JOB_NAMES_TMP}"; do
  [[ ! -e "${temporary}" && ! -L "${temporary}" ]]
done
# noclobber turns the initial create into an O_EXCL-style fail-closed open on
# Bash.  It protects against a symlink or pre-existing file appearing between
# the absence check and creation.  Subsequent writes use only these exact,
# canonical, owner-owned regular files while this invocation owns the lock.
set -o noclobber
for temporary in "${MANIFEST_TMP}" "${ARCHIVE_TMP}" "${SIDECAR_TMP}" "${EXPECTED_MEMBERS_TMP}" "${ACTUAL_MEMBERS_TMP}" "${ARCHIVE_MEMBER_HASHES_TMP}" "${EXPECTED_JOB_NAMES_TMP}" "${ACTUAL_JOB_NAMES_TMP}"; do
  : > "${temporary}"
done
set +o noclobber
for temporary in "${MANIFEST_TMP}" "${ARCHIVE_TMP}" "${SIDECAR_TMP}" "${EXPECTED_MEMBERS_TMP}" "${ACTUAL_MEMBERS_TMP}" "${ARCHIVE_MEMBER_HASHES_TMP}" "${EXPECTED_JOB_NAMES_TMP}" "${ACTUAL_JOB_NAMES_TMP}"; do
  require_regular_canonical "${temporary}"
  [[ "$(stat -c %u -- "${temporary}")" == "$(id -u)" ]]
done

# The job emits a flat, fixed set of files.  Refuse a nested payload, a link,
# a special file, or an unexpected filename instead of silently archiving it.
cat > "${EXPECTED_JOB_NAMES_TMP}" <<EOF
baseline-dispatch-surface-job${JOB_ID}.json
driver-job${JOB_ID}.log
final-status-job${JOB_ID}.sha256
final-status-job${JOB_ID}.txt
gpu-identity-job${JOB_ID}.txt
gpu-final-post-apps-job${JOB_ID}.txt
gpu-final-post-job${JOB_ID}.txt
gpu-post-apps-job${JOB_ID}.txt
gpu-post-job${JOB_ID}.txt
gpu-pre-apps-job${JOB_ID}.txt
gpu-pre-job${JOB_ID}.txt
inputs-job${JOB_ID}.sha256
installed-members-job${JOB_ID}.json
installed-postloader-surface-job${JOB_ID}.json
installed-preloader-surface-job${JOB_ID}.json
native_c2_full_backend_bench.py
outputs-job${JOB_ID}.sha256
plugin-v16-fresh-backend-decision-job${JOB_ID}.json
plugin-v16-fresh-backend-summary-job${JOB_ID}.json
runtime-location-job${JOB_ID}.txt
seed-gpu-uuid-job${JOB_ID}.csv
support-installed-job${JOB_ID}.sha256
support-source-job${JOB_ID}.sha256
EOF
for seed in 17 23 42 2024 314159 20260801 20260815 20260829; do
  printf 'plugin-v16-wheel-full-backend-seed%s-job%s.chrome.json\nplugin-v16-wheel-full-backend-seed%s-job%s.json\n' "${seed}" "${JOB_ID}" "${seed}" "${JOB_ID}" >> "${EXPECTED_JOB_NAMES_TMP}"
done
LC_ALL=C sort -o "${EXPECTED_JOB_NAMES_TMP}" "${EXPECTED_JOB_NAMES_TMP}"
[[ $(wc -l < "${EXPECTED_JOB_NAMES_TMP}") -eq 39 && -z "$(uniq -d "${EXPECTED_JOB_NAMES_TMP}")" ]]
[[ -z "$(find "${JOB_DIR}" -mindepth 1 ! -type f -print -quit)" && -z "$(find "${JOB_DIR}" -mindepth 1 -type l -print -quit)" ]]
find "${JOB_DIR}" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort > "${ACTUAL_JOB_NAMES_TMP}"
cmp "${EXPECTED_JOB_NAMES_TMP}" "${ACTUAL_JOB_NAMES_TMP}"

printf '%s  %s\n%s  %s\n' "${C2_EXPECTED_ARCHIVE_SCRIPT_SHA}" "${ARCHIVE_SCRIPT}" "${EXPECTED_FRESH_SCRIPT_SHA}" "${FRESH_SCRIPT}" | sha256sum -c -
sha256sum -c "${OUTPUTS}" >/dev/null
sha256sum -c "${FINAL_SIDECAR}" >/dev/null
grep -qxE 'FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 TEE_RC=0 MANIFEST_RC=0 CLEANUP_RC=0 POST_UUID=GPU-[0-9A-Fa-f-]+ UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${FINAL}"

if queue_state=$(squeue -h -j "${JOB_ID}" -o '%i' 2>&1); then
  [[ -z "${queue_state}" ]]
else
  [[ "${queue_state}" == 'slurm_load_jobs error: Invalid job id specified' ]]
fi
if scheduler_state=$(scontrol show job -o "${JOB_ID}" 2>&1); then
  scheduler_state=" ${scheduler_state} "
  [[ "${scheduler_state}" == *" JobId=${JOB_ID} "* && "${scheduler_state}" == *' JobState=COMPLETED '* && "${scheduler_state}" == *' ExitCode=0:0 '* ]]
else
  # A purged controller record is acceptable only because every persistent
  # result, final-status sidecar, output hash and semantic gate is checked.
  [[ "${scheduler_state}" == 'slurm_load_jobs error: Invalid job id specified' ]]
fi

python3 - "${JOB_DIR}" "${JOB_ID}" "${INPUTS}" "${OUTPUTS}" "${FINAL}" "${SUMMARY}" "${DECISION}" "${FRESH_SCRIPT}" "${EXPECTED_FRESH_SCRIPT_SHA}" "${EXPECTED_HARNESS_SHA}" "${EXPECTED_PLUGIN_SHA}" "${EXPECTED_STABLE_SHA}" "${EXPECTED_WHEEL_SHA}" "${EXPECTED_STRESS_DECISION_SHA}" "${EXPECTED_STRESS_AGGREGATE_SHA}" "${EXPECTED_LIFECYCLE_RESULT_SHA}" "${EXPECTED_LIFECYCLE_ATTESTATION_SHA}" <<'PY'
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

(job, job_id, inputs, outputs, final, summary_path, decision_path, fresh_script,
 fresh_sha, harness_sha, plugin_sha, stable_sha, wheel_sha, stress_decision_sha,
 stress_aggregate_sha, lifecycle_sha, lifecycle_attestation_sha) = sys.argv[1:]
job = Path(job); inputs = Path(inputs); outputs = Path(outputs); final = Path(final)
summary_path = Path(summary_path); decision_path = Path(decision_path); fresh_script = Path(fresh_script)
seeds = [17, 23, 42, 2024, 314159, 20260801, 20260815, 20260829]
H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert H(fresh_script) == fresh_sha
assert H(job / 'native_c2_full_backend_bench.py') == harness_sha

def read_hash_manifest(path):
    rows = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        assert re.fullmatch(r'[0-9a-f]{64}', digest) and name and name not in rows
        rows[name] = digest
    return rows

output_rows = read_hash_manifest(outputs)
expected_output_names = {
    'baseline-dispatch-surface-job%s.json' % job_id,
    'gpu-final-post-apps-job%s.txt' % job_id,
    'gpu-final-post-job%s.txt' % job_id,
    'gpu-identity-job%s.txt' % job_id,
    'gpu-post-apps-job%s.txt' % job_id,
    'gpu-post-job%s.txt' % job_id,
    'gpu-pre-apps-job%s.txt' % job_id,
    'gpu-pre-job%s.txt' % job_id,
    'inputs-job%s.sha256' % job_id,
    'installed-members-job%s.json' % job_id,
    'installed-postloader-surface-job%s.json' % job_id,
    'installed-preloader-surface-job%s.json' % job_id,
    'native_c2_full_backend_bench.py',
    'plugin-v16-fresh-backend-decision-job%s.json' % job_id,
    'plugin-v16-fresh-backend-summary-job%s.json' % job_id,
    'runtime-location-job%s.txt' % job_id,
    'seed-gpu-uuid-job%s.csv' % job_id,
    'support-installed-job%s.sha256' % job_id,
    'support-source-job%s.sha256' % job_id,
}
for seed in seeds:
    expected_output_names |= {
        'plugin-v16-wheel-full-backend-seed%s-job%s.json' % (seed, job_id),
        'plugin-v16-wheel-full-backend-seed%s-job%s.chrome.json' % (seed, job_id),
    }
assert len(expected_output_names) == 35
assert set(Path(name).name for name in output_rows) == expected_output_names
for name, digest in output_rows.items():
    path = Path(name)
    assert path.parent == job and path.is_file() and not path.is_symlink() and H(path) == digest

status = final.read_text(encoding='utf-8').strip()
m = re.fullmatch(r'FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 TEE_RC=0 MANIFEST_RC=0 CLEANUP_RC=0 POST_UUID=(GPU-[0-9A-Fa-f-]+) UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z', status)
assert m
uuid = m.group(1)

input_rows = read_hash_manifest(inputs)
input_hashes = Counter(input_rows.values())
for required in (harness_sha, plugin_sha, wheel_sha, stress_decision_sha,
                 stress_aggregate_sha, lifecycle_sha, lifecycle_attestation_sha):
    assert input_hashes[required] == 1
assert input_hashes[fresh_sha] == 2
submitted = '/home/lcpu/85117379/validate_native_c2_plugin_v16_k_chunk_lookahead_fresh_backend_8seeds_20260831.slurm'
assert input_rows[submitted] == fresh_sha

runtime_location = (job / ('runtime-location-job%s.txt' % job_id)).read_text(encoding='utf-8').splitlines()
assert runtime_location[1:] == ['node_local_install=true', 'ephemeral=true']
assert runtime_location[0].startswith('runtime_parent=') and runtime_location[0] != 'runtime_parent='

baseline = json.loads((job / ('baseline-dispatch-surface-job%s.json' % job_id)).read_text())
pre = json.loads((job / ('installed-preloader-surface-job%s.json' % job_id)).read_text())
post = json.loads((job / ('installed-postloader-surface-job%s.json' % job_id)).read_text())
installed = json.loads((job / ('installed-members-job%s.json' % job_id)).read_text())
assert baseline['schema'] == 'c2-native-plugin-v16-k-chunk-lookahead-fresh-baseline-dispatch-surface-v1'
assert '_C::native_c2_msa_decode' not in baseline['ops'] and baseline['stable_sha256'] == stable_sha
assert pre['schema'] == 'c2-native-plugin-v16-k-chunk-lookahead-fresh-installed-preloader-surface-v1' and pre['matches_immutable_baseline'] is True and pre['ops'] == baseline['ops']
assert post['schema'] == 'c2-native-plugin-v16-k-chunk-lookahead-fresh-installed-postloader-surface-v1'
assert post['plugin_sha256'] == plugin_sha and post['added_ops'] == ['_C::native_c2_msa_decode'] and post['native_c2_has_cuda_kernel'] is True and post['all_baseline_op_names_preserved'] is True
assert installed['schema'] == 'c2-native-plugin-v16-k-chunk-lookahead-fresh-installed-members-v1'
assert installed['plugin_sha256'] == plugin_sha and installed['stable_member_sha256'] == stable_sha and installed['all_gates_pass'] is True

for name in ('gpu-pre-apps-job%s.txt' % job_id, 'gpu-post-apps-job%s.txt' % job_id, 'gpu-final-post-apps-job%s.txt' % job_id):
    assert (job / name).read_bytes() == b''
assert 'B300' in (job / ('gpu-pre-job%s.txt' % job_id)).read_text(encoding='utf-8')
final_post = (job / ('gpu-final-post-job%s.txt' % job_id)).read_text(encoding='utf-8')
assert 'FINAL_POST UTC=' in final_post and 'B300' in final_post and uuid in final_post
identity = (job / ('gpu-identity-job%s.txt' % job_id)).read_text(encoding='utf-8').splitlines()
assert identity == ['PRE_UUID=%s' % uuid, 'POST_UUID=%s' % uuid]

uuid_rows = list(csv.DictReader((job / ('seed-gpu-uuid-job%s.csv' % job_id)).open(encoding='utf-8', newline='')))
assert [int(row['seed']) for row in uuid_rows] == seeds
assert all(row['before_uuid'] == row['after_uuid'] == uuid for row in uuid_rows)

summary = json.loads(summary_path.read_text())
decision = json.loads(decision_path.read_text())
assert summary['schema'] == 'c2-native-plugin-v16-k-chunk-lookahead-fresh-backend-summary-v1'
assert summary['all_integration_and_correctness_gates_pass'] is True and summary['seed_count'] == len(seeds)
assert summary['one_physical_gpu_uuid'] == uuid and summary['fresh_process_per_seed'] is True
assert summary['v16_dso_sha256'] == plugin_sha and summary['triton_parity_is_hard_gate'] is False
boundary = summary['v16_vs_v12_fresh_wheel_comparator']
assert boundary['available'] is False and boundary['comparison_kind'] == 'none' and boundary['fresh_native_over_triton_is_not_v16_vs_v12_promotion'] is True
rows = summary['seeds']
assert [row['seed'] for row in rows] == seeds
native = []; triton = []; ratios = []
for row, seed, uuid_row in zip(rows, seeds, uuid_rows):
    result = job / ('plugin-v16-wheel-full-backend-seed%s-job%s.json' % (seed, job_id))
    trace = job / ('plugin-v16-wheel-full-backend-seed%s-job%s.chrome.json' % (seed, job_id))
    assert Path(row['result']) == result and Path(row['trace']) == trace and uuid_row['result'] == str(result) and uuid_row['trace'] == str(trace)
    assert row['result_sha256'] == H(result) and trace.stat().st_size > 0
    one = json.loads(result.read_text())
    assert one['schema'] == 'c2-native-c2-full-vllm-backend-v2' and one['seed'] == seed and one['all_gates_pass'] is True
    assert one['correctness']['pass'] is True and one['triton_correctness']['pass'] is True
    assert one['correctness']['oracle'].startswith('independent FP32 ') and one['triton_correctness']['oracle'].startswith('independent FP32 ')
    assert one['caller_output']['pointer_unchanged'] is True
    assert one['profiling']['checks']['one_cpu_dispatcher_event'] is True and one['profiling']['checks']['one_cuda_native_kernel_event'] is True
    timing = one['timing']
    assert timing['pass'] is True and 'ABBA=native,triton,triton,native' in timing['protocol'] and timing['warmup'] == 10 and timing['repetitions'] == 50
    assert timing['native']['sample_count'] == timing['triton']['sample_count'] == 100
    assert all(timing['pre_timing_checks'][k] is True for k in ('distinct_output_buffers', 'native_hard_gates', 'triton_builder', 'triton_correctness', 'triton_static_selection'))
    n = float(timing['native']['median_ms']); t = float(timing['triton']['median_ms'])
    assert math.isfinite(n) and math.isfinite(t) and n > 0 and t > 0
    assert row['native_median_ms'] == n and row['triton_median_ms'] == t and row['native_over_triton_latency_ratio'] == n / t
    native.append(n); triton.append(t); ratios.append(n / t)
assert summary['native_median_of_seed_medians_ms'] == statistics.median(native)
assert summary['triton_median_of_seed_medians_ms'] == statistics.median(triton)
assert summary['native_over_triton_ratio_of_seed_medians'] == statistics.median(ratios)
assert summary['triton_parity_achieved'] == (max(ratios) <= 1.0)

assert decision['schema'] == 'c2-native-plugin-v16-k-chunk-lookahead-fresh-backend-decision-v1'
assert decision['accepted_for_fresh_integration'] is True and decision['final_rc'] == 0
assert set(decision['hard_gates']) == {'wheel_identity', 'record_and_install', 'stable_dso_preserved', 'loader_surface', 'eight_seed_real_backend', 'direct_v16_vs_v12_bitwise_and_promotion_evidence', 'lifecycle_evidence', 'clean_b300_before_after'}
assert all(value is True for value in decision['hard_gates'].values())
assert decision['performance']['reported'] is True and decision['performance']['triton_parity_required'] is False and decision['performance']['triton_parity_achieved'] == summary['triton_parity_achieved']
assert decision['v16_vs_v12_fresh_wheel_boundary'] == boundary
PY

# The external manifest has one record for every safe tar regular member.
# It intentionally does not include itself, yielding a byte-for-byte
# manifest-to-tar bijection that a local downloader can repeat without state.
(
  cd "${BASE}"
  {
    find "$(basename "${ROOT}")/job${JOB_ID}" -mindepth 1 -maxdepth 1 -type f -print0
    printf '%s\0' "$(basename "${ROOT}")/slurm-${JOB_ID}.log" "$(basename "${FRESH_SCRIPT}")" "$(basename "${ARCHIVE_SCRIPT}")"
  } | LC_ALL=C sort -z | xargs -0 -r sha256sum
) > "${MANIFEST_TMP}"
records=$(wc -l < "${MANIFEST_TMP}")
(( records == 42 ))
(cd "${BASE}" && sha256sum -c "${MANIFEST_TMP}" >/dev/null)
sed -nE 's/^[0-9a-f]{64}  //p' "${MANIFEST_TMP}" > "${EXPECTED_MEMBERS_TMP}"
[[ $(wc -l < "${EXPECTED_MEMBERS_TMP}") -eq "${records}" && -z "$(LC_ALL=C sort "${EXPECTED_MEMBERS_TMP}" | uniq -d)" ]]

tar -C "${BASE}" --format=posix --sort=name --numeric-owner --owner=0 --group=0 --no-recursion -czf "${ARCHIVE_TMP}" --files-from="${EXPECTED_MEMBERS_TMP}"
[[ -s "${ARCHIVE_TMP}" ]]
tar -tzf "${ARCHIVE_TMP}" > "${ACTUAL_MEMBERS_TMP}"
[[ -z "$(grep -E '(^/|(^|/)\.\.(/|$))' "${ACTUAL_MEMBERS_TMP}" || true)" && $(wc -l < "${ACTUAL_MEMBERS_TMP}") -eq "${records}" && -z "$(LC_ALL=C sort "${ACTUAL_MEMBERS_TMP}" | uniq -d)" ]]
cmp <(LC_ALL=C sort "${EXPECTED_MEMBERS_TMP}") <(LC_ALL=C sort "${ACTUAL_MEMBERS_TMP}")
tar -tvzf "${ARCHIVE_TMP}" | awk -v expected="${records}" 'BEGIN { ok=1 } { ++count; if (substr($1,1,1)!="-") ok=0 } END { exit !(count==expected && ok) }'
while IFS= read -r member; do
  member_sha=$(tar -xOf "${ARCHIVE_TMP}" -- "${member}" | sha256sum | awk '{print $1}')
  printf '%s  %s\n' "${member_sha}" "${member}"
done < "${EXPECTED_MEMBERS_TMP}" > "${ARCHIVE_MEMBER_HASHES_TMP}"
cmp "${MANIFEST_TMP}" "${ARCHIVE_MEMBER_HASHES_TMP}"

archive_sha=$(sha256sum "${ARCHIVE_TMP}" | awk '{print $1}')
manifest_sha=$(sha256sum "${MANIFEST_TMP}" | awk '{print $1}')
printf '%s  %s\n%s  %s\n' "${archive_sha}" "${ARCHIVE}" "${manifest_sha}" "${MANIFEST}" > "${SIDECAR_TMP}"
ln -- "${MANIFEST_TMP}" "${MANIFEST}"; manifest_linked=1
ln -- "${ARCHIVE_TMP}" "${ARCHIVE}"; archive_linked=1
ln -- "${SIDECAR_TMP}" "${SIDECAR}"; sidecar_linked=1
sha256sum -c "${SIDECAR}" >/dev/null
published=1
printf 'ARCHIVE=%s\nMANIFEST=%s\nSIDECAR=%s\nJOB_FILES=39\nRECORDS=%s\nMEMBERS=%s\n' "${ARCHIVE}" "${MANIFEST}" "${SIDECAR}" "${records}" "${records}"
