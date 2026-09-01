#!/usr/bin/env bash
# Fresh-target validation for the independent native-C2 plugin overlay wheel.
#
# Required submit-time identity inputs:
#   C2_WHEEL_ABSPATH                 absolute path to the job-specific overlay wheel
#   C2_PROVENANCE_ABSPATH            absolute path to its matching provenance JSON
#   C2_EXPECTED_WHEEL_SHA            SHA-256 of the derived overlay wheel
#   C2_EXPECTED_PROVENANCE_SHA       SHA-256 of its provenance JSON
#   C2_EXPECTED_HARNESS_SHA          SHA-256 of native_c2_full_backend_bench.py
#   C2_EXPECTED_PLUGIN_SHA           SHA-256 of the independently built plugin .so
#   C2_EXPECTED_PACKAGER_SHA         reviewed SHA-256 of the packager embedded in provenance
#   C2_EXPECTED_RUNTIME_SCRIPT_SHA   reviewed SHA-256 of this runtime script
#
# The plugin archive path and production load API are deliberately fixed, not
# caller-provided.  The pre-loader audit proves the baseline dispatcher/schema
# surface survived installation; only then is the exact production loader
# imported and invoked twice in the same process.
#SBATCH --job-name=c2-native-plugin-wheel-run
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-wheel-runtime-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONOPTIMIZE=0
export PYTHONPATH=
export PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-wheel-runtime-artifacts-20260829
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
UV=/usr/local/bin/uv
BASELINE_VLLM=/home/lcpu/85117379/venv-vllm-d4da0c5/lib/python3.12/site-packages/vllm
C2_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02/team/c2_msa_decode
HARNESS=/home/lcpu/85117379/native_c2_full_backend_bench_20260829.py
RUNTIME_ROOT=/home/lcpu/85117379/c2-native-plugin-wheel-runtime-20260829
RUNTIME=${RUNTIME_ROOT}/job${SLURM_JOB_ID}
RESULT="${ARTIFACT_DIR}/plugin-wheel-full-backend.json"
TRACE="${ARTIFACT_DIR}/plugin-wheel-full-backend.chrome.json"
BASELINE_OPS="${ARTIFACT_DIR}/baseline-stable-ops.json"
INSTALLED_PRE="${ARTIFACT_DIR}/installed-pre-plugin-ops.json"
INSTALLED_POST="${ARTIFACT_DIR}/installed-post-plugin-ops.json"
INSTALLED_MEMBERS="${ARTIFACT_DIR}/installed-members.json"
EXPECTED_BASELINE_WHEEL_SHA=91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06
EXPECTED_STABLE_MEMBER_SHA=cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442
EXPECTED_PLUGIN_ARCHIVE_MEMBER=vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_LOADER_MODULE=vllm.models.minimax_m3.nvidia.msa_native_c2_decode
EXPECTED_LOADER_CALLABLE=_load_native_c2_plugin_once
EXPECTED_ADAPTER_MEMBER=vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py
EXPECTED_WHEEL_FILENAME=vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")

: "${C2_WHEEL_ABSPATH:?submit with an absolute job-specific overlay wheel path}"
: "${C2_PROVENANCE_ABSPATH:?submit with an absolute matching provenance path}"
: "${C2_EXPECTED_WHEEL_SHA:?submit with reviewed derived wheel SHA-256}"
: "${C2_EXPECTED_PROVENANCE_SHA:?submit with reviewed provenance SHA-256}"
: "${C2_EXPECTED_HARNESS_SHA:?submit with reviewed strict v2 harness SHA-256}"
: "${C2_EXPECTED_PLUGIN_SHA:?submit with reviewed independent plugin SHA-256}"
: "${C2_EXPECTED_PACKAGER_SHA:?submit with reviewed packager SHA-256}"
: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?submit with reviewed runtime-script SHA-256}"
[[ "${C2_WHEEL_ABSPATH}" == /* && "${C2_PROVENANCE_ABSPATH}" == /* ]]
WHEEL=$(readlink -f "${C2_WHEEL_ABSPATH}")
PROVENANCE=$(readlink -f "${C2_PROVENANCE_ABSPATH}")
for c2_digest in "${C2_EXPECTED_WHEEL_SHA}" "${C2_EXPECTED_PROVENANCE_SHA}" \
  "${C2_EXPECTED_HARNESS_SHA}" "${C2_EXPECTED_PLUGIN_SHA}" \
  "${C2_EXPECTED_PACKAGER_SHA}" "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}"; do
  [[ "${c2_digest}" =~ ^[0-9a-f]{64}$ ]]
done
[[ "$(basename "${WHEEL}")" == "${EXPECTED_WHEEL_FILENAME}" ]]
[[ "$(basename "${PROVENANCE}")" == c2-native-plugin-overlay-provenance.json ]]
[[ "$(dirname "${WHEEL}")" == "$(dirname "${PROVENANCE}")" ]]

mkdir -p "${ARTIFACT_ROOT}" "${RUNTIME_ROOT}"
[[ ! -e "${ARTIFACT_DIR}" && ! -e "${RUNTIME}" ]] || {
  echo "refusing to reuse job-specific artifact/runtime directory" >&2
  exit 2
}
mkdir -p "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1
finalize() {
  local rc=$?
  trap - EXIT
  set +e
  printf 'FINAL_RC=%s UTC=%s\n' "${rc}" "$(date -u +%FT%TZ)"
  printf 'FINAL_RC=%s UTC=%s\n' "${rc}" "$(date -u +%FT%TZ)" \
    > "${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt"
  exit "${rc}"
}
trap finalize EXIT

compute_apps() {
  local raw
  raw=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null) || return 1
  printf '%s\n' "${raw}" | sed '/^[[:space:]]*$/d'
}

snapshot() {
  local label=$1
  printf '\n===== %s UTC=%s =====\n' "${label}" "$(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap \
    --format=csv,noheader,nounits
  printf '%s\n' '-- compute apps --'
  compute_apps || true
}

snapshot PRE
gpu_line=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* ]]
apps_before=$(compute_apps) || { echo "failed to query pre-run compute apps" >&2; exit 1; }
[[ -z "${apps_before}" ]]
[[ -f "${WHEEL}" && -f "${PROVENANCE}" && -f "${HARNESS}" && -f "${SCRIPT_PATH}" ]]
[[ -d "${C2_ROOT}" && -d "${BASELINE_VLLM}" && -x "${UV}" && -x "${PYTHON}" ]]
[[ ! -e "${RUNTIME}" ]]
"${UV}" --version
"${PYTHON}" -c 'raise SystemExit(0 if __debug__ else 1)'
printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${C2_EXPECTED_WHEEL_SHA}" "${WHEEL}" \
  "${C2_EXPECTED_PROVENANCE_SHA}" "${PROVENANCE}" \
  "${C2_EXPECTED_HARNESS_SHA}" "${HARNESS}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" | sha256sum -c -
# The plugin is not a standalone external input in this job.  Its identity is
# checked against the immutable wheel/provenance below, rather than hashing a
# mutable build path.
sha256sum "${WHEEL}" "${PROVENANCE}" "${HARNESS}" "${SCRIPT_PATH}" \
  | tee "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"

"${UV}" pip install --python "${PYTHON}" --target "${RUNTIME}" --no-deps "${WHEEL}"

"${PYTHON}" - "${RUNTIME}" "${WHEEL}" "${PROVENANCE}" "${INSTALLED_MEMBERS}" \
  "${C2_EXPECTED_PLUGIN_SHA}" "${EXPECTED_PLUGIN_ARCHIVE_MEMBER}" \
  "${C2_EXPECTED_PACKAGER_SHA}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import zipfile

runtime, wheel, provenance_path, output = map(Path, sys.argv[1:5])
expected_plugin_sha, plugin_member, expected_packager_sha = sys.argv[5:]
adapter_member = "vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py"
stable_member = "vllm/_C_stable_libtorch.abi3.so"
expected_wheel_sha = "91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06"
expected_stable_sha = "cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442"
assert plugin_member == "vllm/_native_c2_msa_decode_plugin.abi3.so", plugin_member
provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
assert provenance["schema"] == "c2-native-plugin-overlay-provenance-v1"
assert Path(provenance["derived_wheel"]["path"]).resolve() == wheel.resolve()
assert provenance["baseline_wheel"]["sha256"] == expected_wheel_sha
assert provenance["plugin_archive_member"] == plugin_member
assert provenance["plugin_expected_sha256"] == expected_plugin_sha
assert provenance["packager"]["sha256"] == expected_packager_sha
assert provenance["member_set"]["new_members"] == [plugin_member, adapter_member]
assert provenance["member_set"]["removed_members"] == []
assert provenance["derived_record_full_verification"]["all_payload_entries_validated"] is True
assert provenance["derived_record_full_verification"]["record_self_row_valid"] is True
stable = provenance["baseline_stable_member"]
assert stable["byte_identical"] is True
assert stable["baseline_sha256"] == stable["derived_sha256"] == expected_stable_sha
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == provenance["derived_wheel"]["sha256"]
members = {item["archive_member"]: item for item in provenance["overlay_members"]}
assert members[plugin_member]["zip_sha256"] == expected_plugin_sha
assert members[plugin_member]["kind"] == "added_native_c2_plugin"
assert members[plugin_member]["action"] == "add"
assert all(item["source_matches_zip"] for item in members.values())
expected_dispatch = {
    "vllm/_custom_ops.py": "b57dfa819424b89bed8bf39924d5a4a3d2ebf327b39f3ae7c2e9e65b61c2398d",
    "vllm/config/attention.py": "caf68c29d8d36324d252a66f0811a25b26d11da9860732359076fbb76625b5ae",
    "vllm/v1/attention/backends/registry.py": "cd539941417fd5faf6672676de0f8ae15c9adb2a915b47cdd508e154c34faea4",
    "vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py": "6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412",
    "vllm/models/minimax_m3/nvidia/sparse_attention_msa.py": "07d00372ca1995aa96159a0b9f8949b1b40f817a63f49224e88aafac1b722200",
}
assert provenance["expected_dispatch_member_sha256"] == expected_dispatch
for member, expected_hash in expected_dispatch.items():
    expected_kind = (
        "added_python_dispatch"
        if member == adapter_member
        else "replaced_python_dispatch"
    )
    expected_action = "add" if member == adapter_member else "replace"
    assert members[member]["kind"] == expected_kind
    assert members[member]["action"] == expected_action
    assert members[member]["source_sha256"] == expected_hash
    assert members[member]["expected_post_patch_sha256"] == expected_hash
    assert members[member]["zip_sha256"] == expected_hash
checks = []
for member, metadata in members.items():
    installed = runtime / member
    assert installed.is_file(), installed
    digest = hashlib.sha256(installed.read_bytes()).hexdigest()
    assert digest == metadata["zip_sha256"], member
    checks.append({"archive_member": member, "installed_path": str(installed),
                   "sha256": digest, "size": installed.stat().st_size})
with zipfile.ZipFile(wheel, "r") as archive:
    wheel_stable = archive.read(stable_member)
    wheel_plugin = archive.read(plugin_member)
assert hashlib.sha256(wheel_stable).hexdigest() == expected_stable_sha
assert hashlib.sha256(wheel_plugin).hexdigest() == expected_plugin_sha
installed_stable = runtime / stable_member
assert installed_stable.is_file()
assert installed_stable.read_bytes() == wheel_stable
assert hashlib.sha256(installed_stable.read_bytes()).hexdigest() == expected_stable_sha
output.write_text(json.dumps({
    "schema": "c2-native-plugin-wheel-installed-members-v1",
    "runtime": str(runtime),
    "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "plugin_archive_member": plugin_member,
    "plugin_sha256": expected_plugin_sha,
    "baseline_stable_member_sha256": expected_stable_sha,
    "installed_members": checks,
    "all_gates_pass": True,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(output)
PY

# Baseline capture is a clean process with no overlay path.  It performs the
# same normal Python imports as the adapter (including _custom_ops, whose import
# installs reviewed Meta kernels), then records every _C schema and dispatch
# registration kind.  The plugin loader is never present in this process.
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= \
  "${PYTHON}" - "${BASELINE_VLLM}" "${BASELINE_OPS}" <<'PY'
import json
from pathlib import Path
import sys
import torch
import vllm
import vllm._C_stable_libtorch as stable
from vllm import _custom_ops as baseline_custom_ops
from vllm.config.attention import MiniMaxM3MSADecodeBackend
from vllm.platforms import current_platform

def schema_key(schema):
    return f"{schema.name}.{schema.overload_name}" if schema.overload_name else schema.name

def dispatch_surface(op):
    rows = {}
    for line in torch._C._dispatch_dump_table(op).splitlines():
        key, sep, registration = line.partition(": ")
        if not sep:
            continue
        left = registration.rfind("[")
        assert left >= 0 and registration.endswith("]"), (op, line)
        rows[key] = registration[left:]
    return rows

required = ("_dispatch_get_all_op_names", "_dispatch_dump_table", "_jit_get_all_schemas",
            "_dispatch_has_kernel_for_dispatch_key")
assert all(callable(getattr(torch._C, name, None)) for name in required)
expected_root, output = map(Path, sys.argv[1:])
root = Path(vllm.__file__).resolve().parent
assert root == expected_root.resolve(), (root, expected_root)
assert Path(stable.__file__).resolve() == root / "_C_stable_libtorch.abi3.so"
assert Path(baseline_custom_ops.__file__).resolve() == root / "_custom_ops.py"
assert torch.__version__ == "2.13.0+cu130"
assert torch.version.cuda == "13.0"
ops = sorted(name for name in torch._C._dispatch_get_all_op_names() if name.startswith("_C::"))
schemas = {}
for schema in torch._C._jit_get_all_schemas():
    if schema.name.startswith("_C::"):
        schemas.setdefault(schema_key(schema), []).append(str(schema))
schemas = {key: sorted(value) for key, value in sorted(schemas.items())}
assert set(ops) == set(schemas), (set(ops) - set(schemas), set(schemas) - set(ops))
dispatch = {op: dispatch_surface(op) for op in ops}
assert all(dispatch.values())
output.write_text(json.dumps({
    "schema": "c2-native-plugin-baseline-dispatch-surface-v1",
    "vllm_root": str(root), "stable_library": str(Path(stable.__file__).resolve()),
    "stable_sha256": __import__("hashlib").sha256(Path(stable.__file__).read_bytes()).hexdigest(),
    "torch_version": torch.__version__, "ops": ops, "op_count": len(ops),
    "normal_imports_before_snapshot": [
        "vllm._custom_ops", "vllm.config.attention", "vllm.platforms"
    ],
    "schemas": schemas, "dispatch_key_registration_kinds": dispatch,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

# A second clean process imports the installed adapter without calling its
# loader, snapshots the comparable normal-import state, then invokes the fixed
# production loader twice.  No caller-provided loader API or plugin path can
# alter this.
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${RUNTIME}:${C2_ROOT}" \
  "${PYTHON}" - "${RUNTIME}" "${WHEEL}" "${BASELINE_OPS}" "${INSTALLED_PRE}" \
  "${INSTALLED_POST}" "${EXPECTED_PLUGIN_ARCHIVE_MEMBER}" \
  "${EXPECTED_LOADER_MODULE}" "${EXPECTED_LOADER_CALLABLE}" \
  "${EXPECTED_ADAPTER_MEMBER}" <<'PY'
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile
import torch
import vllm
import vllm._C_stable_libtorch as stable

def schema_key(schema):
    return f"{schema.name}.{schema.overload_name}" if schema.overload_name else schema.name

def dispatch_surface(op):
    rows = {}
    for line in torch._C._dispatch_dump_table(op).splitlines():
        key, sep, registration = line.partition(": ")
        if not sep:
            continue
        left = registration.rfind("[")
        assert left >= 0 and registration.endswith("]"), (op, line)
        rows[key] = registration[left:]
    return rows

def snapshot():
    ops = sorted(name for name in torch._C._dispatch_get_all_op_names() if name.startswith("_C::"))
    schemas = {}
    for schema in torch._C._jit_get_all_schemas():
        if schema.name.startswith("_C::"):
            schemas.setdefault(schema_key(schema), []).append(str(schema))
    schemas = {key: sorted(value) for key, value in sorted(schemas.items())}
    assert set(ops) == set(schemas), (set(ops) - set(schemas), set(schemas) - set(ops))
    dispatch = {op: dispatch_surface(op) for op in ops}
    assert all(dispatch.values())
    return ops, schemas, dispatch

required = ("_dispatch_get_all_op_names", "_dispatch_dump_table", "_jit_get_all_schemas",
            "_dispatch_has_kernel_for_dispatch_key")
assert all(callable(getattr(torch._C, name, None)) for name in required)
(runtime, wheel, baseline_path, pre_output, post_output) = map(Path, sys.argv[1:6])
plugin_member, loader_module_name, loader_callable_name, adapter_member = sys.argv[6:]
assert plugin_member == "vllm/_native_c2_msa_decode_plugin.abi3.so", plugin_member
assert loader_module_name == "vllm.models.minimax_m3.nvidia.msa_native_c2_decode"
assert loader_callable_name == "_load_native_c2_plugin_once"
assert adapter_member == "vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py"
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
root = Path(vllm.__file__).resolve().parent
assert root == runtime.resolve() / "vllm", (root, runtime)
assert Path(stable.__file__).resolve() == root / "_C_stable_libtorch.abi3.so"
assert importlib.metadata.distribution("vllm").locate_file("vllm").resolve() == root
assert torch.__version__ == baseline["torch_version"]
assert hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest() == (
    "cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442")
with zipfile.ZipFile(wheel, "r") as archive:
    assert Path(stable.__file__).read_bytes() == archive.read("vllm/_C_stable_libtorch.abi3.so")
    plugin_path = root / plugin_member.removeprefix("vllm/")
    assert plugin_path.is_file()
    assert plugin_path.read_bytes() == archive.read(plugin_member)

loader_module = importlib.import_module(loader_module_name)
adapter_path = root / adapter_member.removeprefix("vllm/")
assert Path(loader_module.__file__).resolve() == adapter_path
with zipfile.ZipFile(wheel, "r") as archive:
    assert adapter_path.read_bytes() == archive.read(adapter_member)

pre_ops, pre_schemas, pre_dispatch = snapshot()
assert pre_ops == baseline["ops"], (set(pre_ops) ^ set(baseline["ops"]))
assert pre_schemas == baseline["schemas"]
assert pre_dispatch == baseline["dispatch_key_registration_kinds"]
assert "_C::native_c2_msa_decode" not in pre_ops
pre_output.write_text(json.dumps({
    "schema": "c2-native-plugin-installed-preloader-surface-v1",
    "vllm_root": str(root), "stable_library": str(Path(stable.__file__).resolve()),
    "stable_sha256": hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest(),
    "adapter_imported_before_snapshot_without_calling_loader": True,
    "ops": pre_ops, "schemas": pre_schemas,
    "dispatch_key_registration_kinds": pre_dispatch,
    "matches_immutable_baseline": True,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

loader = getattr(loader_module, loader_callable_name)
assert callable(loader), (loader_module_name, loader_callable_name)
first_loader_result = loader()
assert first_loader_result is True

post_ops, post_schemas, post_dispatch = snapshot()
baseline_ops = set(baseline["ops"])
added = sorted(set(post_ops) - baseline_ops)
missing = sorted(baseline_ops - set(post_ops))
assert not missing, missing
assert added == ["_C::native_c2_msa_decode"], added
schema_changes = {op: {"baseline": baseline["schemas"][op], "installed": post_schemas.get(op)}
                  for op in sorted(baseline_ops)
                  if post_schemas.get(op) != baseline["schemas"][op]}
dispatch_changes = {op: {
    "baseline": baseline["dispatch_key_registration_kinds"][op],
    "installed": post_dispatch.get(op),
} for op in sorted(baseline_ops)
                     if post_dispatch.get(op) != baseline["dispatch_key_registration_kinds"][op]}
assert not schema_changes, schema_changes
assert not dispatch_changes, dispatch_changes
assert set(post_schemas) - baseline_ops == {"_C::native_c2_msa_decode"}
assert torch._C._dispatch_has_kernel_for_dispatch_key("_C::native_c2_msa_decode", "CUDA")

# The production loader must be idempotent in the same interpreter: a second
# call returns exactly True and cannot mutate any dispatcher/schema surface.
second_loader_result = loader()
assert second_loader_result is True
second_ops, second_schemas, second_dispatch = snapshot()
assert second_ops == post_ops
assert second_schemas == post_schemas
assert second_dispatch == post_dispatch
post_output.write_text(json.dumps({
    "schema": "c2-native-plugin-installed-postloader-surface-v1",
    "plugin_archive_member": plugin_member,
    "plugin_path": str(plugin_path),
    "plugin_sha256": hashlib.sha256(plugin_path.read_bytes()).hexdigest(),
    "loader": {
        "module": loader_module_name,
        "module_file": str(Path(loader_module.__file__).resolve()),
        "module_file_matches_wheel_adapter": True,
        "adapter_member": adapter_member,
        "adapter_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "callable": loader_callable_name,
        "style": "noargs",
        "first_return_is_true": first_loader_result is True,
        "second_return_is_true": second_loader_result is True,
        "second_call_snapshot_unchanged": True,
    },
    "missing_baseline_ops": missing, "added_ops": added,
    "added_schemas": {op: post_schemas[op] for op in added},
    "baseline_schema_changes": schema_changes,
    "baseline_dispatch_key_registration_kind_changes": dispatch_changes,
    "native_c2_has_cuda_kernel": True,
    "all_baseline_op_names_schemas_and_dispatch_surfaces_preserved": True,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${RUNTIME}:${C2_ROOT}" \
  timeout --foreground --preserve-status --kill-after=10s 1200s \
  "${PYTHON}" "${HARNESS}" \
  --c2-root "${C2_ROOT}" \
  --vllm-root "${RUNTIME}" \
  --output "${RESULT}" \
  --trace "${TRACE}" \
  --seed 20260829 \
  --timing-warmup 10 \
  --timing-repetitions 50

"${PYTHON}" - "${RESULT}" "${TRACE}" <<'PY'
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["schema"] == "c2-native-c2-full-vllm-backend-v2"
assert payload["all_gates_pass"] is True
assert payload["correctness"]["pass"] is True
assert payload["triton_correctness"]["pass"] is True
assert payload["profiling"]["pass"] is True
assert payload["timing"]["pass"] is True
assert "same BF16 query" in payload["timing"]["comparison_input_semantics"]
assert "ABBA=native,triton,triton,native" in payload["timing"]["protocol"]
assert payload["timing"]["warmup"] == 10
assert payload["timing"]["repetitions"] == 50
assert len(payload["timing"]["raw_cycles_ms"]) == 50
assert all(sorted(row) == ["native_a_0", "native_a_3", "triton_b_1", "triton_b_2"]
           for row in payload["timing"]["raw_cycles_ms"])
assert payload["timing"]["native"]["sample_count"] == 100
assert payload["timing"]["triton"]["sample_count"] == 100
ratio = float(payload["timing"]["native_over_triton_median_latency_ratio"])
assert math.isfinite(ratio) and ratio > 0.0
trace = Path(sys.argv[2])
assert trace.is_file() and trace.stat().st_size > 0
PY

snapshot POST
apps_after=$(compute_apps) || { echo "failed to query post-run compute apps" >&2; exit 1; }
[[ -z "${apps_after}" ]]
printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${C2_EXPECTED_WHEEL_SHA}" "${WHEEL}" \
  "${C2_EXPECTED_PROVENANCE_SHA}" "${PROVENANCE}" \
  "${C2_EXPECTED_HARNESS_SHA}" "${HARNESS}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" | sha256sum -c -
sha256sum "${RESULT}" "${TRACE}" "${INSTALLED_MEMBERS}" "${BASELINE_OPS}" \
  "${INSTALLED_PRE}" "${INSTALLED_POST}" \
  "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" \
  > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
