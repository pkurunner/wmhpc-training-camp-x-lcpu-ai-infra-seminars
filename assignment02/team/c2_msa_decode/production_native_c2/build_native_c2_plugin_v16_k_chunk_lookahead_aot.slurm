#!/usr/bin/env bash
# Build the v16 intra-page K-chunk raw-FP8 lookahead candidate as a standalone C2-plugin DSO.
# Restore the immutable v12 kernel source, then apply this one reviewed patch
# with fuzz=0.  This script does not validate semantics; directed validation
# is a later, separately frozen gate.

#SBATCH --job-name=c2-native-plugin-v16-aot
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/slurm-c2-native-plugin-v16-aot-%j.log

set -Eeuo pipefail
umask 077
: "$C2_EXPECTED_SCRIPT_SHA"
[[ "$C2_EXPECTED_SCRIPT_SHA" =~ ^[0-9a-f]{64}$ ]]

BASE=/home/lcpu/85117379/vllm-d4da0c5
RUNTIME=/tmp/c2-native-plugin-v16-k-chunk-lookahead-aot-job$SLURM_JOB_ID
SRC=$RUNTIME/src
BUILD_DIR=$RUNTIME/build
DEPS=$RUNTIME/deps
ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v16-k-chunk-lookahead-aot-artifacts-20260831
ARTIFACT_DIR=$ARTIFACT_ROOT/job$SLURM_JOB_ID
TOOLS=/home/lcpu/85117379/c2-native-tools-20260829
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
CUTLASS=/home/lcpu/85117379/cutlass-v4.4.2-c2-20260829
MSA=/home/lcpu/85117379/msa-official
SCRIPT_PATH=$(readlink -f "$BASH_SOURCE")
NATIVE_V1_SOURCE=/home/lcpu/85117379/native_c2_decode.plugin-v1.cu
SCHEMA_PATCH=/home/lcpu/85117379/native_c2_plugin_schema_20260829.patch
CMAKE_PATCH=/home/lcpu/85117379/exact_d4_native_c2_plugin_cmake_20260829.patch
DISPATCH_PATCH=/home/lcpu/85117379/exact_d4_python_dispatch_20260829.patch
LOADER_PATCH=/home/lcpu/85117379/exact_d4_native_c2_plugin_python_loader_20260829.patch
V12_BASELINE_SOURCE=/home/lcpu/85117379/c2-native-plugin-v12-aot-artifacts-20260830/job12983/native_c2_decode.v12-q-row-padding.cu
V16_PATCH=/home/lcpu/85117379/native_c2_v16_k_chunk_lookahead_20260831.patch

EXPECTED_BASE_COMMIT=d4da0c55af3aa231b6209bf77871f3ed36eab0d2
EXPECTED_CUTLASS_COMMIT=da5e086dab31d63815acafdac9a9c5893b1c69e2
EXPECTED_MSA_COMMIT=80434d7f67877c6570ca19cac444b84bc9855dac
EXPECTED_MSA_CUTLASS_COMMIT=eb61c911471867a5fd2466bfd8f29306cea6ebf8
EXPECTED_NATIVE_V1_SOURCE_SHA=f67495586759b3b8f718dbbf60627c499f917842f2ed6ecf89d2344f1c82bc4b
EXPECTED_SCHEMA_PATCH_SHA=53e7cd4b09a8999010442934c863427c30b019fc549d7f0cf0f3d16ccc1e2e6e
EXPECTED_CMAKE_PATCH_SHA=5b145f36ce5f8b12dae71ce02e77d65545a29eefcb5a6045f591348bc325ac35
EXPECTED_DISPATCH_PATCH_SHA=1ca224ec4f94adf8ee97580a0e08321552224555570784c308c0edcda34bcd91
EXPECTED_LOADER_PATCH_SHA=9cb3a22216048083d7a4802ba700fa07fd892436151871dc6e909c77239adb33
EXPECTED_PLUGIN_V1_SOURCE_SHA=8fee086f8ba4f5b2710347025211f9733ce64c61b04e44e710016d59b336f406
EXPECTED_PLUGIN_CMAKE_SHA=6662b0400eb04c590b56c15346848b4a26c45d86e796a69e1bcb4a0d91523734
EXPECTED_PLUGIN_ADAPTER_SHA=6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412
EXPECTED_V12_BASELINE_SOURCE_SHA=535d90b856ed1062aa7b8a105eb2c5f236c450826e65496e646d0d5a27eb8aaf
EXPECTED_V16_PATCH_SHA=586d15cfcac5c9a1bbe1efa94e89e9031f090eb19a376ec817f10599ba8e11be
EXPECTED_V16_SOURCE_SHA=a01b78c89205a8ded07d65c82cdcf9d2eb5b39df4fc45a2fd992d92200b84180

[[ -d "$ARTIFACT_ROOT" && ! -L "$ARTIFACT_ROOT" &&
   "$(readlink -f -- "$ARTIFACT_ROOT")" == "$ARTIFACT_ROOT" &&
   "$(stat -c %u -- "$ARTIFACT_ROOT")" == "$(id -u)" ]] || {
  echo 'artifact root missing, linked, noncanonical, or not owned by this user' >&2; exit 2;
}
[[ -z "$(find "$ARTIFACT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
  echo 'artifact root must be empty before v16 AOT submission' >&2; exit 2;
}
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]
[[ "$RUNTIME" == "/tmp/c2-native-plugin-v16-k-chunk-lookahead-aot-job$SLURM_JOB_ID" ]]
[[ ! -e "$RUNTIME" && ! -L "$RUNTIME" && ! -e "$ARTIFACT_DIR" ]] || {
  echo 'job-scoped output already exists' >&2; exit 2;
}
mkdir -m 0700 -- "$ARTIFACT_DIR"
DRIVER_LOG=$ARTIFACT_DIR/build-driver-job$SLURM_JOB_ID.log
OUTPUTS_MANIFEST=$ARTIFACT_DIR/outputs-job$SLURM_JOB_ID.sha256
FINAL_STATUS=$ARTIFACT_DIR/final-status-job$SLURM_JOB_ID.txt
FINAL_STATUS_SIDECAR=$ARTIFACT_DIR/final-status-job$SLURM_JOB_ID.sha256
SUBMITTED_SCRIPT_COPY=$ARTIFACT_DIR/submitted-script-job$SLURM_JOB_ID.slurm

cleanup_runtime() {
  local runtime=$1
  [[ "$runtime" == "/tmp/c2-native-plugin-v16-k-chunk-lookahead-aot-job$SLURM_JOB_ID" ]] || return 1
  [[ ! -e "$runtime" && ! -L "$runtime" ]] && return 0
  [[ -d "$runtime" && ! -L "$runtime" &&
     "$(readlink -f -- "$runtime")" == "$runtime" &&
     "$(stat -c %u -- "$runtime")" == "$(id -u)" ]] || return 1
  rm -rf -- "$runtime" || return 1
  [[ ! -e "$runtime" && ! -L "$runtime" ]]
}
setup_finalize() {
  local body_rc=$? cleanup_rc=0 manifest_rc=0
  local setup_diag=$ARTIFACT_DIR/setup-finalizer-diagnostics-job$SLURM_JOB_ID.txt
  trap - EXIT
  set +e
  cleanup_runtime "$RUNTIME" || cleanup_rc=$?
  printf 'SETUP_FINALIZER=1 BODY_RC=%s CLEANUP_RC=%s UTC=%s\n' \
    "$body_rc" "$cleanup_rc" "$(date -u +%FT%TZ)" > "$setup_diag"
  if [[ -f "$SUBMITTED_SCRIPT_COPY" ]]; then
    sha256sum "$SUBMITTED_SCRIPT_COPY" "$setup_diag" > "$OUTPUTS_MANIFEST" &&
      sha256sum -c "$OUTPUTS_MANIFEST" >/dev/null || manifest_rc=1
  else
    sha256sum "$setup_diag" > "$OUTPUTS_MANIFEST" &&
      sha256sum -c "$OUTPUTS_MANIFEST" >/dev/null || manifest_rc=1
  fi
  printf 'BODY_RC=%s TEE_RC=0 MANIFEST_RC=%s CLEANUP_RC=%s SETUP_FAILURE=1 FINAL_RC=2 UTC=%s\n' \
    "$body_rc" "$manifest_rc" "$cleanup_rc" "$(date -u +%FT%TZ)" > "$FINAL_STATUS"
  sha256sum "$FINAL_STATUS" > "$FINAL_STATUS_SIDECAR"
  exit 2
}
trap setup_finalize EXIT

for input in "$SCRIPT_PATH" "$NATIVE_V1_SOURCE" "$SCHEMA_PATCH" "$CMAKE_PATCH" \
  "$DISPATCH_PATCH" "$LOADER_PATCH" "$V12_BASELINE_SOURCE" "$V16_PATCH"; do
  [[ -f "$input" && ! -L "$input" ]] || {
    echo "missing or linked input: $input" >&2; exit 2;
  }
done
mkdir -m 0700 -- "$RUNTIME"
[[ -d "$RUNTIME" && ! -L "$RUNTIME" &&
   "$(readlink -f -- "$RUNTIME")" == "$RUNTIME" &&
   "$(stat -c %u -- "$RUNTIME")" == "$(id -u)" ]]
mkdir -m 0700 -- "$DEPS"
printf '%s  %s\n' "$C2_EXPECTED_SCRIPT_SHA" "$SCRIPT_PATH" | sha256sum -c -
cp -- "$SCRIPT_PATH" "$SUBMITTED_SCRIPT_COPY"
chmod a-w "$SUBMITTED_SCRIPT_COPY"
printf '%s  %s\n' "$C2_EXPECTED_SCRIPT_SHA" "$SUBMITTED_SCRIPT_COPY" | sha256sum -c -
exec 3>&1 4>&2
exec > >(tee "$DRIVER_LOG") 2>&1
DRIVER_TEE_PID=$!

finalize() {
  local body_rc=$? tee_rc=0 cleanup_rc=0 manifest_rc=0 final_rc=0
  local output_list=$ARTIFACT_DIR/outputs-list-job$SLURM_JOB_ID.nul
  trap - EXIT
  set +e
  printf 'BODY_RC=%s UTC=%s FINALIZING=1\n' "$body_rc" "$(date -u +%FT%TZ)"
  exec 1>&3 2>&4
  exec 3>&- 4>&-
  wait "$DRIVER_TEE_PID"; tee_rc=$?
  cleanup_runtime "$RUNTIME" || cleanup_rc=$?
  if ! (set -o pipefail; find "$ARTIFACT_DIR" -type f \
      ! -name "$(basename "$OUTPUTS_MANIFEST")" \
      ! -name "$(basename "$FINAL_STATUS")" \
      ! -name "$(basename "$FINAL_STATUS_SIDECAR")" \
      ! -name "$(basename "$output_list")" -print0 | sort -z > "$output_list"); then
    manifest_rc=1
  elif [[ ! -s "$output_list" ]]; then
    manifest_rc=1
  elif ! xargs -0 -r sha256sum < "$output_list" > "$OUTPUTS_MANIFEST"; then
    manifest_rc=1
  elif ! sha256sum -c "$OUTPUTS_MANIFEST" >/dev/null; then
    manifest_rc=1
  fi
  rm -f -- "$output_list" || manifest_rc=1
  final_rc=$body_rc
  (( final_rc != 0 || (tee_rc == 0 && cleanup_rc == 0 && manifest_rc == 0) )) || final_rc=1
  printf 'BODY_RC=%s TEE_RC=%s CLEANUP_RC=%s MANIFEST_RC=%s FINAL_RC=%s UTC=%s\n' \
    "$body_rc" "$tee_rc" "$cleanup_rc" "$manifest_rc" "$final_rc" \
    "$(date -u +%FT%TZ)" > "$FINAL_STATUS"
  sha256sum "$FINAL_STATUS" > "$FINAL_STATUS_SIDECAR" || final_rc=1
  exit "$final_rc"
}
trap finalize EXIT

export PATH="$TOOLS/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin"
export TORCH_CUDA_ARCH_LIST=10.3f
export VLLM_CUTLASS_SRC_DIR="$CUTLASS"
export FMHA_SM100_SRC_DIR="$MSA"
sha256_of() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local actual
  actual=$(sha256_of "$1")
  printf '%s SHA256=%s path=%s\n' "$3" "$actual" "$1"
  [[ "$actual" == "$2" ]] || { echo "$3 hash mismatch" >&2; exit 2; }
}

printf 'UTC=%s JOB=%s HOST=%s CPUS=%s\n' "$(date -u +%FT%TZ)" "$SLURM_JOB_ID" "$(hostname)" "$SLURM_CPUS_PER_TASK"
[[ "$(git -C "$BASE" rev-parse HEAD)" == "$EXPECTED_BASE_COMMIT" &&
   -z "$(git -C "$BASE" status --porcelain)" ]] || { echo 'exact d4 base not clean' >&2; exit 2; }
[[ "$(git -C "$CUTLASS" rev-parse HEAD)" == "$EXPECTED_CUTLASS_COMMIT" &&
   -z "$(git -C "$CUTLASS" status --porcelain)" ]] || { echo 'CUTLASS dependency identity/cleanliness mismatch' >&2; exit 2; }
[[ "$(git -C "$MSA" rev-parse HEAD)" == "$EXPECTED_MSA_COMMIT" &&
   -z "$(git -C "$MSA" status --porcelain)" ]] || { echo 'MSA dependency identity/cleanliness mismatch' >&2; exit 2; }
[[ "$(git -C "$MSA/python/fmha_sm100/cutlass" rev-parse HEAD)" == "$EXPECTED_MSA_CUTLASS_COMMIT" &&
   -z "$(git -C "$MSA/python/fmha_sm100/cutlass" status --porcelain)" ]] || { echo 'nested MSA CUTLASS identity/cleanliness mismatch' >&2; exit 2; }
printf '%s\n' "$EXPECTED_BASE_COMMIT" > "$ARTIFACT_DIR/base-commit-job$SLURM_JOB_ID.txt"
require_hash "$SCRIPT_PATH" "$C2_EXPECTED_SCRIPT_SHA" v16-aot-submission-script
require_hash "$SUBMITTED_SCRIPT_COPY" "$C2_EXPECTED_SCRIPT_SHA" v16-aot-durable-submitted-script
require_hash "$NATIVE_V1_SOURCE" "$EXPECTED_NATIVE_V1_SOURCE_SHA" native-v1-source
require_hash "$SCHEMA_PATCH" "$EXPECTED_SCHEMA_PATCH_SHA" schema-patch
require_hash "$CMAKE_PATCH" "$EXPECTED_CMAKE_PATCH_SHA" cmake-patch
require_hash "$DISPATCH_PATCH" "$EXPECTED_DISPATCH_PATCH_SHA" dispatch-patch
require_hash "$LOADER_PATCH" "$EXPECTED_LOADER_PATCH_SHA" loader-patch
require_hash "$V12_BASELINE_SOURCE" "$EXPECTED_V12_BASELINE_SOURCE_SHA" accepted-v12-q-row-padding-source
require_hash "$V16_PATCH" "$EXPECTED_V16_PATCH_SHA" v16-k-chunk-lookahead-patch
{
  printf 'BASE_COMMIT=%s\n' "$EXPECTED_BASE_COMMIT"
  printf 'AOT_SCRIPT_EXPECTED_SHA256=%s\n' "$C2_EXPECTED_SCRIPT_SHA"
  git -C "$CUTLASS" rev-parse HEAD | sed 's/^/CUTLASS_COMMIT=/'
  git -C "$MSA" rev-parse HEAD | sed 's/^/MSA_COMMIT=/'
  git -C "$MSA/python/fmha_sm100/cutlass" rev-parse HEAD | sed 's/^/MSA_CUTLASS_COMMIT=/'
  sha256sum "$SUBMITTED_SCRIPT_COPY" "$NATIVE_V1_SOURCE" "$SCHEMA_PATCH" "$CMAKE_PATCH" \
    "$DISPATCH_PATCH" "$LOADER_PATCH" "$V12_BASELINE_SOURCE" "$V16_PATCH"
} > "$ARTIFACT_DIR/input-provenance-job$SLURM_JOB_ID.txt"
"$PYTHON" -c 'import torch; print("TORCH", torch.__version__, torch.version.cuda)'
cmake --version; ninja --version; nvcc --version

git clone --no-hardlinks "$BASE" "$SRC"
git -C "$SRC" checkout --detach "$EXPECTED_BASE_COMMIT"
[[ -z "$(git -C "$SRC" status --porcelain)" ]]
TARGET_SOURCE=$SRC/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
install -D -m 0644 "$NATIVE_V1_SOURCE" "$TARGET_SOURCE"
require_hash "$TARGET_SOURCE" "$EXPECTED_NATIVE_V1_SOURCE_SHA" copied-native-v1-source
for patch_file in "$SCHEMA_PATCH" "$CMAKE_PATCH" "$DISPATCH_PATCH"; do
  patch --dry-run --batch --fuzz=0 -d "$SRC" -p1 < "$patch_file"
  patch --batch --fuzz=0 -d "$SRC" -p1 < "$patch_file"
done
patch --dry-run --batch --fuzz=0 -d "$SRC" -p1 < "$LOADER_PATCH"
patch --batch --fuzz=0 -d "$SRC" -p1 < "$LOADER_PATCH"
require_hash "$TARGET_SOURCE" "$EXPECTED_PLUGIN_V1_SOURCE_SHA" plugin-v1-source-before-frozen-v12
require_hash "$SRC/CMakeLists.txt" "$EXPECTED_PLUGIN_CMAKE_SHA" plugin-cmake
require_hash "$SRC/vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py" "$EXPECTED_PLUGIN_ADAPTER_SHA" plugin-adapter
"$PYTHON" -m py_compile "$SRC/vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py"

# Historical v9--v13 patch chains are deliberately absent: the kernel is
# exactly v12 plus the one v16 intra-page K-chunk lookahead delta.
install -m 0644 "$V12_BASELINE_SOURCE" "$TARGET_SOURCE"
require_hash "$TARGET_SOURCE" "$EXPECTED_V12_BASELINE_SOURCE_SHA" restored-accepted-v12-source
patch --dry-run --batch --fuzz=0 -d "$SRC" -p1 < "$V16_PATCH" | tee "$ARTIFACT_DIR/v16-patch-dry-run-job$SLURM_JOB_ID.log"
patch --batch --fuzz=0 -d "$SRC" -p1 < "$V16_PATCH" | tee "$ARTIFACT_DIR/v16-patch-apply-job$SLURM_JOB_ID.log"
require_hash "$TARGET_SOURCE" "$EXPECTED_V16_SOURCE_SHA" post-v16-k-chunk-lookahead-source

# Structural drift alarms; directed correctness is a separate gate.
grep -F '__nv_fp8_e4m3 key_prefetch[kTile / 2];' "$TARGET_SOURCE"
grep -F 'if (k_offset == 0) {' "$TARGET_SOURCE"
grep -F 'if (k_offset + kTile < kHeadDim) {' "$TARGET_SOURCE"
grep -F '(k_offset + kTile) + dim;' "$TARGET_SOURCE"
! grep -qE 'next_(logical|physical)_page|prefetch_next_k|k_prefetch\[' "$TARGET_SOURCE"
! grep -q 'have_prefetched_key' "$TARGET_SOURCE"
[[ $(grep -Ec 'wmma::load_matrix_sync\((k_fragment|value_fragment), &fp8_stage\[warp\]\[0\]\[0\], kTile\);' "$TARGET_SOURCE") == 2 ]] || {
  echo 'v16 must retain exactly the two v12 K/V stage WMMA loads' >&2; exit 2;
}
"$PYTHON" - "$TARGET_SOURCE" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")
key = source.index("__nv_fp8_e4m3 key_prefetch[kTile / 2];")
scope_end = source.index(
    "\n      }\n      wmma::store_matrix_sync(&score_tiles", key
)
value = source.index("__nv_fp8_e4m3 value_prefetch[kTile / 2];", scope_end)
assert key < scope_end < value
PY
git -C "$SRC" diff --binary > "$ARTIFACT_DIR/derived-plugin-v16-job$SLURM_JOB_ID.patch"
DURABLE_CANDIDATE_SOURCE=$ARTIFACT_DIR/native_c2_decode.v16-k-chunk-lookahead.cu
cp --preserve=mode,timestamps "$TARGET_SOURCE" "$DURABLE_CANDIDATE_SOURCE"
require_hash "$DURABLE_CANDIDATE_SOURCE" "$EXPECTED_V16_SOURCE_SHA" durable-v16-candidate-source
DURABLE_BUILD_SOURCE=$ARTIFACT_DIR/build-source/native_c2_decode.cu
install -D -m 0444 "$TARGET_SOURCE" "$DURABLE_BUILD_SOURCE"
require_hash "$DURABLE_BUILD_SOURCE" "$EXPECTED_V16_SOURCE_SHA" durable-v16-build-source

cmake -S "$SRC" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DVLLM_TARGET_DEVICE=cuda -DVLLM_PYTHON_EXECUTABLE="$PYTHON" \
  -DFETCHCONTENT_BASE_DIR="$DEPS" -DCMAKE_INSTALL_PREFIX="$ARTIFACT_DIR/install" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DNVCC_THREADS=2 \
  -DCMAKE_JOB_POOL_COMPILE:STRING=compile -DCMAKE_JOB_POOLS:STRING=compile=32 \
  2>&1 | tee "$ARTIFACT_DIR/configure-job$SLURM_JOB_ID.log"
grep -F 'CUDA target architectures: 10.3f' "$ARTIFACT_DIR/configure-job$SLURM_JOB_ID.log"
grep -F 'CUDA supported target architectures: 10.0' "$ARTIFACT_DIR/configure-job$SLURM_JOB_ID.log"
grep -F 'Building standalone native C2 plugin for archs: 10.0' "$ARTIFACT_DIR/configure-job$SLURM_JOB_ID.log"
"$PYTHON" - "$BUILD_DIR/compile_commands.json" "$ARTIFACT_DIR/native-c2-compile-command-job$SLURM_JOB_ID.json" "$TARGET_SOURCE" <<'PY'
import json
import sys
from pathlib import Path
database_path, output_path, source_path = map(Path, sys.argv[1:])
database = json.loads(database_path.read_text(encoding="utf-8"))
matches = [entry for entry in database if Path(entry["file"]).resolve() == source_path.resolve()]
assert len(matches) == 1, (source_path, len(matches))
output_path.write_text(json.dumps(matches[0], indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
grep -F -- '-gencode arch=compute_100,code=sm_100' "$ARTIFACT_DIR/native-c2-compile-command-job$SLURM_JOB_ID.json"
cmake --build "$BUILD_DIR" --target _native_c2_msa_decode_plugin -- -j64 -v 2>&1 | tee "$ARTIFACT_DIR/compile-job$SLURM_JOB_ID.log"
PLUGIN=$(find "$BUILD_DIR" -type f -name '_native_c2_msa_decode_plugin.abi3.so' -print -quit)
[[ -n "$PLUGIN" && $(find "$BUILD_DIR" -type f -name '_native_c2_msa_decode_plugin.abi3.so' -print | wc -l) == 1 ]] || {
  echo 'expected exactly one DSO' >&2; exit 2;
}
install -D -m 0755 "$PLUGIN" "$ARTIFACT_DIR/vllm/_native_c2_msa_decode_plugin.abi3.so"
[[ "$(sha256_of "$PLUGIN")" == "$(sha256_of "$ARTIFACT_DIR/vllm/_native_c2_msa_decode_plugin.abi3.so")" ]] || {
  echo 'durable candidate plugin differs from build output' >&2; exit 2;
}
grep -F -- '-gencode arch=compute_100,code=sm_100' "$ARTIFACT_DIR/compile-job$SLURM_JOB_ID.log"
cuobjdump --dump-resource-usage "$PLUGIN" | tee "$ARTIFACT_DIR/plugin-resource-usage-job$SLURM_JOB_ID.txt"
grep -A1 'native_c2_msa_decode_kernel' "$ARTIFACT_DIR/plugin-resource-usage-job$SLURM_JOB_ID.txt" | tee "$ARTIFACT_DIR/plugin-resource-proof-job$SLURM_JOB_ID.txt"
RESOURCE_LINE=$(grep -m 1 -E 'REG:[[:space:]]*[0-9]+.*STACK:[[:space:]]*[0-9]+.*SHARED:[[:space:]]*[0-9]+.*LOCAL:[[:space:]]*[0-9]+' "$ARTIFACT_DIR/plugin-resource-proof-job$SLURM_JOB_ID.txt")
[[ -n "$RESOURCE_LINE" ]] || { echo 'native kernel resource line missing' >&2; exit 2; }
read_resource() { sed -nE "s/.*$1:[[:space:]]*([0-9]+).*/\\1/p" <<< "$RESOURCE_LINE"; }
REG=$(read_resource REG); STACK=$(read_resource STACK); SHARED=$(read_resource SHARED); LOCAL=$(read_resource LOCAL)
[[ "$REG" =~ ^[0-9]+$ && "$STACK" =~ ^[0-9]+$ && "$SHARED" =~ ^[0-9]+$ && "$LOCAL" =~ ^[0-9]+$ ]] || {
  echo "unparseable resource line: $RESOURCE_LINE" >&2; exit 2;
}
printf 'RESOURCE_LINE=%s\nREG=%s\nSTACK=%s\nSHARED=%s\nLOCAL=%s\n' "$RESOURCE_LINE" "$REG" "$STACK" "$SHARED" "$LOCAL" | tee "$ARTIFACT_DIR/plugin-resource-gate-job$SLURM_JOB_ID.txt"
# v16 adds only lane-private registers: the v12 shared-memory layout must be
# unchanged at 31,136 B.  Any other value proves an unintended staging/layout
# change and fails closed before directed correctness or timing can run.
(( REG <= 128 && STACK == 0 && SHARED == 31136 && LOCAL == 0 )) || {
  echo "v16 stop-loss resource gate failed: REG=$REG STACK=$STACK SHARED=$SHARED LOCAL=$LOCAL" >&2; exit 2;
}
nm -D --defined-only "$PLUGIN" | tee "$ARTIFACT_DIR/plugin-symbols-job$SLURM_JOB_ID.txt"
! grep -q ' PyInit_' "$ARTIFACT_DIR/plugin-symbols-job$SLURM_JOB_ID.txt"
cuobjdump --list-elf "$PLUGIN" | tee "$ARTIFACT_DIR/plugin-cuobjdump-elf-job$SLURM_JOB_ID.txt"
[[ $(grep -Ec '^ELF file[[:space:]]+[0-9]+:.*\.cubin$' "$ARTIFACT_DIR/plugin-cuobjdump-elf-job$SLURM_JOB_ID.txt") == 1 &&
   $(grep -Ec '^ELF file[[:space:]]+[0-9]+:.*\.sm_100\.cubin$' "$ARTIFACT_DIR/plugin-cuobjdump-elf-job$SLURM_JOB_ID.txt") == 1 ]] || {
  echo 'expected exactly one sm_100 cubin' >&2; exit 2;
}
! grep -Eqi 'sm_10(3|3f)|compute_10(3|3f)|\.ptx($|[^[:alnum:]_])' "$ARTIFACT_DIR/plugin-cuobjdump-elf-job$SLURM_JOB_ID.txt"
cuobjdump --list-ptx "$PLUGIN" | tee "$ARTIFACT_DIR/plugin-cuobjdump-ptx-job$SLURM_JOB_ID.txt"
[[ ! -s "$ARTIFACT_DIR/plugin-cuobjdump-ptx-job$SLURM_JOB_ID.txt" ]] || { echo 'PTX fallback forbidden' >&2; exit 2; }
cuobjdump --dump-sass "$PLUGIN" | tee "$ARTIFACT_DIR/plugin-sass-job$SLURM_JOB_ID.sass"
grep -m 80 -E 'native_c2_msa_decode_kernel|HMMA|MMA' "$ARTIFACT_DIR/plugin-sass-job$SLURM_JOB_ID.sass" | tee "$ARTIFACT_DIR/plugin-sass-proof-job$SLURM_JOB_ID.txt"
grep -q native_c2_msa_decode_kernel "$ARTIFACT_DIR/plugin-sass-proof-job$SLURM_JOB_ID.txt"
grep -Eq 'HMMA|MMA' "$ARTIFACT_DIR/plugin-sass-proof-job$SLURM_JOB_ID.txt"
sha256sum "$SUBMITTED_SCRIPT_COPY" "$NATIVE_V1_SOURCE" "$SCHEMA_PATCH" "$CMAKE_PATCH" \
  "$DISPATCH_PATCH" "$LOADER_PATCH" "$V12_BASELINE_SOURCE" "$V16_PATCH" \
  "$DURABLE_CANDIDATE_SOURCE" "$DURABLE_BUILD_SOURCE" \
  "$ARTIFACT_DIR/vllm/_native_c2_msa_decode_plugin.abi3.so" > "$ARTIFACT_DIR/plugin-v16-provenance-job$SLURM_JOB_ID.sha256"
ldd "$PLUGIN" | tee "$ARTIFACT_DIR/plugin-ldd-job$SLURM_JOB_ID.txt"
