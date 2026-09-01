#!/usr/bin/env bash
# Fresh-target promotion gate for the four-producer v4 plugin wheel, using the
# promoted v2 wheel-runtime result as its performance reference.  The immutable
# original wheel remains the installation/dispatcher baseline.
#
# The v4 plugin is accepted only when all of the following hold:
#   1. independently built v4 direct stress result passed FP64/multi-seed/
#      bitwise/caller-pointer gates;
#   2. the fresh overlay install preserves the exact baseline stable DSO and
#      baseline dispatcher surface before loading the plugin;
#   3. the real MiniMaxM3SparseMSAImpl.forward harness passes its same-query
#      ABBA, oracle, caller-pointer and exactly-one dispatcher/kernel gates;
#   4. v4 native median latency is <= 95% of the comparable v2 result.
#
# On any failure this job exits nonzero after writing its decision where
# possible.  It does not alter v2's checkout, DSO, wheel, or artifact paths.

#SBATCH --job-name=c2-native-plugin-v4-wheel-run
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v4-wheel-runtime-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONOPTIMIZE=0
export PYTHONPATH=
export PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v4-wheel-runtime-artifacts-20260829
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
UV=/usr/local/bin/uv
BASELINE_VLLM=/home/lcpu/85117379/venv-vllm-d4da0c5/lib/python3.12/site-packages/vllm
C2_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02/team/c2_msa_decode
HARNESS=/home/lcpu/85117379/native_c2_full_backend_bench_20260829.py
RUNTIME_ROOT=/home/lcpu/85117379/c2-native-plugin-v4-wheel-runtime-20260829
RUNTIME=${RUNTIME_ROOT}/job${SLURM_JOB_ID}
RESULT=${ARTIFACT_DIR}/plugin-v4-wheel-full-backend-job${SLURM_JOB_ID}.json
TRACE=${ARTIFACT_DIR}/plugin-v4-wheel-full-backend-job${SLURM_JOB_ID}.chrome.json
BASELINE_OPS=${ARTIFACT_DIR}/baseline-stable-ops-job${SLURM_JOB_ID}.json
INSTALLED_PRE=${ARTIFACT_DIR}/installed-pre-plugin-ops-job${SLURM_JOB_ID}.json
INSTALLED_POST=${ARTIFACT_DIR}/installed-post-plugin-ops-job${SLURM_JOB_ID}.json
INSTALLED_MEMBERS=${ARTIFACT_DIR}/installed-members-job${SLURM_JOB_ID}.json
DECISION=${ARTIFACT_DIR}/v4-vs-v2-decision-job${SLURM_JOB_ID}.json
EXPECTED_HARNESS_SHA=7883edc25df48e3b69a9f4948775a1baafd3ba0b1bd17edcfeac6644aa7b4762
EXPECTED_BASELINE_WHEEL_SHA=91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06
EXPECTED_STABLE_MEMBER_SHA=cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442
EXPECTED_PLUGIN_ARCHIVE_MEMBER=vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_LOADER_MODULE=vllm.models.minimax_m3.nvidia.msa_native_c2_decode
EXPECTED_LOADER_CALLABLE=_load_native_c2_plugin_once
EXPECTED_ADAPTER_MEMBER=vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py
EXPECTED_WHEEL_FILENAME=vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl
EXPECTED_V2_FULL_BACKEND_RESULT=/home/lcpu/85117379/c2-native-plugin-v2-wheel-runtime-artifacts-20260829/job11468/plugin-v2-wheel-full-backend-job11468.json
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")

: "${C2_WHEEL_ABSPATH:?submit with an absolute job-specific v4 overlay wheel path}"
: "${C2_PROVENANCE_ABSPATH:?submit with an absolute matching provenance path}"
: "${C2_EXPECTED_WHEEL_SHA:?submit with reviewed v4 wheel SHA-256}"
: "${C2_EXPECTED_PROVENANCE_SHA:?submit with reviewed v4 provenance SHA-256}"
: "${C2_EXPECTED_HARNESS_SHA:?submit with reviewed strict v4 harness SHA-256}"
: "${C2_EXPECTED_PLUGIN_SHA:?submit with reviewed v4 plugin SHA-256}"
: "${C2_EXPECTED_PACKAGER_SHA:?submit with reviewed packager SHA-256}"
: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?submit with reviewed runtime-script SHA-256}"
: "${C2_V4_STRESS_RESULT:?submit with passed v4 direct-stress result JSON}"
: "${C2_EXPECTED_V4_STRESS_SHA:?submit with v4 direct-stress JSON SHA-256}"
: "${C2_V4_STRESS_DECISION:?submit with passed v4 direct-stress decision JSON}"
: "${C2_EXPECTED_V4_STRESS_DECISION_SHA:?submit with v4 direct-stress decision SHA-256}"
: "${C2_V2_FULL_BACKEND_RESULT:?submit with comparable passed v2 full-backend JSON}"
: "${C2_EXPECTED_V2_FULL_BACKEND_SHA:?submit with v2 full-backend JSON SHA-256}"
[[ "${C2_WHEEL_ABSPATH}" == /* && "${C2_PROVENANCE_ABSPATH}" == /* ]]
WHEEL=$(readlink -f "${C2_WHEEL_ABSPATH}")
PROVENANCE=$(readlink -f "${C2_PROVENANCE_ABSPATH}")
for digest in "${C2_EXPECTED_WHEEL_SHA}" "${C2_EXPECTED_PROVENANCE_SHA}" \
  "${C2_EXPECTED_HARNESS_SHA}" "${C2_EXPECTED_PLUGIN_SHA}" \
  "${C2_EXPECTED_PACKAGER_SHA}" "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" \
  "${C2_EXPECTED_V4_STRESS_SHA}" "${C2_EXPECTED_V4_STRESS_DECISION_SHA}" \
  "${C2_EXPECTED_V2_FULL_BACKEND_SHA}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]
done
[[ "${C2_V2_FULL_BACKEND_RESULT}" == "${EXPECTED_V2_FULL_BACKEND_RESULT}" ]]
[[ "$(basename "${WHEEL}")" == "${EXPECTED_WHEEL_FILENAME}" ]]
[[ "$(basename "${PROVENANCE}")" == c2-native-plugin-overlay-provenance.json ]]
[[ "$(dirname "${WHEEL}")" == "$(dirname "${PROVENANCE}")" ]]
[[ ! -e "${ARTIFACT_DIR}" && ! -e "${RUNTIME}" ]] || {
  echo "refusing to reuse v4 runtime/artifact path" >&2; exit 2;
}
mkdir -p "${ARTIFACT_ROOT}" "${RUNTIME_ROOT}"
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
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}
snapshot() {
  printf '\n===== %s UTC=%s =====\n' "$1" "$(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap \
    --format=csv,noheader,nounits
  compute_apps || true
}

snapshot PRE
gpu_line=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* ]]
apps_before=$(compute_apps) || { echo "unable to query pre-run compute apps" >&2; exit 1; }
[[ -z "${apps_before}" ]]
[[ -f "${WHEEL}" && -f "${PROVENANCE}" && -f "${HARNESS}" && -f "${SCRIPT_PATH}" ]]
[[ -f "${C2_V4_STRESS_RESULT}" && -f "${C2_V4_STRESS_DECISION}" && -f "${C2_V2_FULL_BACKEND_RESULT}" ]]
[[ -d "${C2_ROOT}" && -d "${BASELINE_VLLM}" && -x "${UV}" && -x "${PYTHON}" ]]
"${UV}" --version
"${PYTHON}" -c 'raise SystemExit(0 if __debug__ else 1)'
printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${C2_EXPECTED_WHEEL_SHA}" "${WHEEL}" \
  "${C2_EXPECTED_PROVENANCE_SHA}" "${PROVENANCE}" \
  "${C2_EXPECTED_HARNESS_SHA}" "${HARNESS}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" \
  "${C2_EXPECTED_V4_STRESS_SHA}" "${C2_V4_STRESS_RESULT}" \
  "${C2_EXPECTED_V4_STRESS_DECISION_SHA}" "${C2_V4_STRESS_DECISION}" \
  "${C2_EXPECTED_V2_FULL_BACKEND_SHA}" "${C2_V2_FULL_BACKEND_RESULT}" | sha256sum -c -
sha256sum "${WHEEL}" "${PROVENANCE}" "${HARNESS}" "${SCRIPT_PATH}" \
  "${C2_V4_STRESS_RESULT}" "${C2_V4_STRESS_DECISION}" "${C2_V2_FULL_BACKEND_RESULT}" \
  | tee "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"

# Reject before installation if the direct v4 gate did not pass its strict
# job11464 comparison or did not produce the FP64/multi-seed contract.
"${PYTHON}" - "${C2_V4_STRESS_RESULT}" "${C2_V4_STRESS_DECISION}" \
  "${C2_EXPECTED_PLUGIN_SHA}" <<'PY'
import json
import math
import sys
from pathlib import Path

p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
decision = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_plugin_sha = sys.argv[3]
assert p["schema"] == "c2-native-c2-stress-perf-v1"
assert p["all_gates_pass"] is True and p["no_monkeypatch"] is True
assert p["correctness_gate"] == {"atol": 1e-4, "oracle_dtype": "float64", "rtol": 1e-3}
assert p["seed_count"] == 8
assert Path(p["operator_library"]).is_absolute()
assert __import__("hashlib").sha256(Path(p["operator_library"]).read_bytes()).hexdigest() == expected_plugin_sha
for row in p["seeds"]:
    assert row["all_gates_pass"] is True
    assert row["correctness"]["allclose"] is True
    assert row["caller_output"]["pointer_unchanged"] is True
    assert row["caller_output"]["timing_pointer_unchanged"] is True
    assert row["stability"]["bitwise_repeatable"] is True
    assert math.isfinite(float(row["latency"]["mean_ms"]))
assert decision["schema"] == "c2-native-plugin-v4-vs-v2-direct-stress-promotion-v1"
assert decision["reference"]["job"] == 11464
assert decision["required_improvement_fraction"] == 0.05
assert decision["all_correctness_and_comparability_gates_pass"] is True
assert decision["accepted"] is True
PY

"${UV}" pip install --python "${PYTHON}" --target "${RUNTIME}" --no-deps "${WHEEL}"

# Audit every overlaid payload and prove installed stable bytes are the base
# wheel's bytes.  This runs before importing a C2 loader.
"${PYTHON}" - "${RUNTIME}" "${WHEEL}" "${PROVENANCE}" "${INSTALLED_MEMBERS}" \
  "${C2_EXPECTED_PLUGIN_SHA}" "${EXPECTED_PLUGIN_ARCHIVE_MEMBER}" \
  "${C2_EXPECTED_PACKAGER_SHA}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import zipfile

runtime, wheel, provenance_path, output = map(Path, sys.argv[1:5])
plugin_sha, plugin_member, expected_packager_sha = sys.argv[5:]
adapter_member = "vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py"
stable_member = "vllm/_C_stable_libtorch.abi3.so"
p = json.loads(provenance_path.read_text(encoding="utf-8"))
assert p["schema"] == "c2-native-plugin-overlay-provenance-v1"
assert p["baseline_wheel"]["sha256"] == "91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06"
assert Path(p["derived_wheel"]["path"]).resolve() == wheel.resolve()
assert p["plugin_archive_member"] == plugin_member
assert p["plugin_expected_sha256"] == plugin_sha
assert p["packager"]["sha256"] == expected_packager_sha
assert p["member_set"]["new_members"] == [plugin_member, adapter_member]
assert p["member_set"]["removed_members"] == []
assert p["derived_record_full_verification"]["all_payload_entries_validated"] is True
assert p["derived_record_full_verification"]["record_self_row_valid"] is True
stable = p["baseline_stable_member"]
assert stable["byte_identical"] is True
assert stable["baseline_sha256"] == stable["derived_sha256"] == "cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442"
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == p["derived_wheel"]["sha256"]
members = {row["archive_member"]: row for row in p["overlay_members"]}
assert members[plugin_member]["zip_sha256"] == plugin_sha
assert members[plugin_member]["kind"] == "added_native_c2_plugin"
assert members[plugin_member]["action"] == "add"
expected_dispatch = {
    "vllm/_custom_ops.py": "b57dfa819424b89bed8bf39924d5a4a3d2ebf327b39f3ae7c2e9e65b61c2398d",
    "vllm/config/attention.py": "caf68c29d8d36324d252a66f0811a25b26d11da9860732359076fbb76625b5ae",
    "vllm/v1/attention/backends/registry.py": "cd539941417fd5faf6672676de0f8ae15c9adb2a915b47cdd508e154c34faea4",
    "vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py": "6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412",
    "vllm/models/minimax_m3/nvidia/sparse_attention_msa.py": "07d00372ca1995aa96159a0b9f8949b1b40f817a63f49224e88aafac1b722200",
}
assert p["expected_dispatch_member_sha256"] == expected_dispatch
for member, expected_hash in expected_dispatch.items():
    expected_kind = "added_python_dispatch" if member == adapter_member else "replaced_python_dispatch"
    expected_action = "add" if member == adapter_member else "replace"
    assert members[member]["kind"] == expected_kind
    assert members[member]["action"] == expected_action
    assert members[member]["source_sha256"] == expected_hash
    assert members[member]["expected_post_patch_sha256"] == expected_hash
    assert members[member]["zip_sha256"] == expected_hash
installed = []
for member, row in members.items():
    path = runtime / member
    assert path.is_file(), path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == row["zip_sha256"], member
    installed.append({"archive_member": member, "path": str(path), "sha256": digest})
with zipfile.ZipFile(wheel, "r") as z:
    assert (runtime / stable_member).read_bytes() == z.read(stable_member)
    assert hashlib.sha256(z.read(plugin_member)).hexdigest() == plugin_sha
output.write_text(json.dumps({"schema": "c2-native-plugin-v4-installed-members-v1",
    "runtime": str(runtime), "plugin_sha256": plugin_sha,
    "stable_member_sha256": stable["baseline_sha256"], "installed_members": installed,
    "all_gates_pass": True}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

# Capture the ordinary immutable base process separately.  Its complete _C
# schema/dispatch surface is the pre-loader reference for the fresh install.
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= \
  "${PYTHON}" - "${BASELINE_VLLM}" "${BASELINE_OPS}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
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
expected, output = map(Path, sys.argv[1:])
root = Path(vllm.__file__).resolve().parent
assert root == expected.resolve(), (root, expected)
assert Path(stable.__file__).resolve() == root / "_C_stable_libtorch.abi3.so"
assert Path(baseline_custom_ops.__file__).resolve() == root / "_custom_ops.py"
assert torch.__version__ == "2.13.0+cu130"
assert torch.version.cuda == "13.0"

def snap():
    ops = sorted(x for x in torch._C._dispatch_get_all_op_names() if x.startswith("_C::"))
    schemas = {}
    for s in torch._C._jit_get_all_schemas():
        if s.name.startswith("_C::"):
            schemas.setdefault(schema_key(s), []).append(str(s))
    schemas = {k: sorted(v) for k, v in sorted(schemas.items())}
    assert set(ops) == set(schemas)
    dispatch = {op: dispatch_surface(op) for op in ops}
    assert all(dispatch.values())
    return ops, schemas, dispatch

assert hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest() == "cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442"
ops, schemas, dispatch = snap()
output.write_text(json.dumps({"schema": "c2-native-plugin-v4-baseline-dispatch-surface-v1",
    "vllm_root": str(root), "stable_library": str(Path(stable.__file__).resolve()),
    "stable_sha256": hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest(),
    "torch_version": torch.__version__, "ops": ops, "op_count": len(ops),
    "normal_imports_before_snapshot": ["vllm._custom_ops", "vllm.config.attention", "vllm.platforms"],
    "schemas": schemas, "dispatch_key_registration_kinds": dispatch},
    indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

# In a fresh target process, capture the target before/after invoking exactly
# the production no-argument loader.  The only permitted surface addition is
# the C2 schema with a CUDA registration.
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${RUNTIME}:${C2_ROOT}" \
  "${PYTHON}" - "${RUNTIME}" "${WHEEL}" "${BASELINE_OPS}" "${INSTALLED_PRE}" \
  "${INSTALLED_POST}" "${EXPECTED_PLUGIN_ARCHIVE_MEMBER}" \
  "${EXPECTED_LOADER_MODULE}" "${EXPECTED_LOADER_CALLABLE}" \
  "${EXPECTED_ADAPTER_MEMBER}" <<'PY'
import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
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

def snap():
    ops = sorted(x for x in torch._C._dispatch_get_all_op_names() if x.startswith("_C::"))
    schemas = {}
    for s in torch._C._jit_get_all_schemas():
        if s.name.startswith("_C::"):
            schemas.setdefault(schema_key(s), []).append(str(s))
    schemas = {k: sorted(v) for k, v in sorted(schemas.items())}
    assert set(ops) == set(schemas)
    dispatch = {op: dispatch_surface(op) for op in ops}
    assert all(dispatch.values())
    return ops, schemas, dispatch

required = ("_dispatch_get_all_op_names", "_dispatch_dump_table", "_jit_get_all_schemas",
            "_dispatch_has_kernel_for_dispatch_key")
assert all(callable(getattr(torch._C, name, None)) for name in required)
(runtime, wheel, baseline_path, pre_path, post_path) = map(Path, sys.argv[1:6])
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
assert hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest() == "cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442"
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
pre_ops, pre_schemas, pre_dispatch = snap()
assert pre_ops == baseline["ops"]
assert pre_schemas == baseline["schemas"]
assert pre_dispatch == baseline["dispatch_key_registration_kinds"]
assert "_C::native_c2_msa_decode" not in pre_ops
pre_path.write_text(json.dumps({"schema": "c2-native-plugin-v4-installed-preloader-surface-v1",
    "vllm_root": str(root), "stable_library": str(Path(stable.__file__).resolve()),
    "stable_sha256": hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest(),
    "adapter_imported_before_snapshot_without_calling_loader": True,
    "matches_immutable_baseline": True, "ops": pre_ops, "schemas": pre_schemas,
    "dispatch_key_registration_kinds": pre_dispatch}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
loader = getattr(loader_module, loader_callable_name)
assert callable(loader), (loader_module_name, loader_callable_name)
first_loader_result = loader()
assert first_loader_result is True
post_ops, post_schemas, post_dispatch = snap()
base_ops = set(baseline["ops"])
added, missing = sorted(set(post_ops) - base_ops), sorted(base_ops - set(post_ops))
assert added == ["_C::native_c2_msa_decode"] and not missing
assert {op for op in post_schemas if op not in base_ops} == {"_C::native_c2_msa_decode"}
assert all(post_schemas[op] == baseline["schemas"][op] for op in base_ops)
assert all(post_dispatch[op] == baseline["dispatch_key_registration_kinds"][op] for op in base_ops)
assert torch._C._dispatch_has_kernel_for_dispatch_key("_C::native_c2_msa_decode", "CUDA")
second_loader_result = loader()
assert second_loader_result is True
second_ops, second_schemas, second_dispatch = snap()
assert second_ops == post_ops
assert second_schemas == post_schemas
assert second_dispatch == post_dispatch
post_path.write_text(json.dumps({"schema": "c2-native-plugin-v4-installed-postloader-surface-v1",
    "plugin_archive_member": plugin_member, "plugin_path": str(plugin_path),
    "plugin_sha256": hashlib.sha256(plugin_path.read_bytes()).hexdigest(),
    "loader": {"module": loader_module_name, "module_file": str(Path(loader_module.__file__).resolve()),
               "module_file_matches_wheel_adapter": True, "adapter_member": adapter_member,
               "adapter_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
               "callable": loader_callable_name, "style": "noargs",
               "first_return_is_true": first_loader_result is True,
               "second_return_is_true": second_loader_result is True,
               "second_call_snapshot_unchanged": True},
    "added_ops": added, "missing_baseline_ops": missing, "native_c2_has_cuda_kernel": True,
    "all_baseline_op_names_schemas_and_dispatch_surfaces_preserved": True},
    indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${RUNTIME}:${C2_ROOT}" \
  timeout --foreground --preserve-status --kill-after=10s 1200s \
  "${PYTHON}" "${HARNESS}" --c2-root "${C2_ROOT}" --vllm-root "${RUNTIME}" \
  --output "${RESULT}" --trace "${TRACE}" --seed 20260829 \
  --timing-warmup 10 --timing-repetitions 50

"${PYTHON}" - "${RESULT}" "${TRACE}" <<'PY'
import json
import math
import sys
from pathlib import Path

p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert p["schema"] == "c2-native-c2-full-vllm-backend-v2"
assert p["all_gates_pass"] is True
assert p["correctness"]["pass"] is True and p["correctness"]["atol"] == 1e-4 and p["correctness"]["rtol"] == 1e-3
assert p["triton_correctness"]["pass"] is True
assert p["caller_output"]["pointer_unchanged"] is True
assert p["profiling"]["pass"] is True
assert p["profiling"]["checks"]["one_cpu_dispatcher_event"] is True
assert p["profiling"]["checks"]["one_cuda_native_kernel_event"] is True
assert p["timing"]["pass"] is True
assert "same BF16 query" in p["timing"]["comparison_input_semantics"]
assert "ABBA=native,triton,triton,native" in p["timing"]["protocol"]
assert p["timing"]["warmup"] == 10 and p["timing"]["repetitions"] == 50
assert p["timing"]["native"]["sample_count"] == p["timing"]["triton"]["sample_count"] == 100
for side in ("native", "triton"):
    assert math.isfinite(float(p["timing"][side]["median_ms"])) and float(p["timing"][side]["median_ms"]) > 0
trace = Path(sys.argv[2])
assert trace.is_file() and trace.stat().st_size > 0
PY

# Establish comparability before calculating the 5% promotion threshold.  If a
# v2 result lacks a same-contract fair ABBA field, fail closed instead of
# manufacturing a ratio.
set +e
"${PYTHON}" - "${C2_V2_FULL_BACKEND_RESULT}" "${RESULT}" "${DECISION}" <<'PY'
import json
import math
import sys
from pathlib import Path

v2_path, v4_path, decision_path = map(Path, sys.argv[1:])
v2 = json.loads(v2_path.read_text(encoding="utf-8"))
v4 = json.loads(v4_path.read_text(encoding="utf-8"))
for label, p in (("v2", v2), ("v4", v4)):
    assert p["schema"] == "c2-native-c2-full-vllm-backend-v2", label
    assert p["all_gates_pass"] is True, label
    assert p["correctness"]["pass"] is True and p["triton_correctness"]["pass"] is True, label
    assert p["caller_output"]["pointer_unchanged"] is True, label
    assert p["profiling"]["checks"]["one_cpu_dispatcher_event"] is True, label
    assert p["profiling"]["checks"]["one_cuda_native_kernel_event"] is True, label
    timing = p["timing"]
    assert timing["pass"] is True, label
    assert "same BF16 query" in timing["comparison_input_semantics"], label
    assert "ABBA=native,triton,triton,native" in timing["protocol"], label
    assert timing["warmup"] == 10 and timing["repetitions"] == 50, label
    assert timing["native"]["sample_count"] == timing["triton"]["sample_count"] == 100, label
assert v2["boundary"] == v4["boundary"]
assert v2["data_contract"] == v4["data_contract"]
v2_native = float(v2["timing"]["native"]["median_ms"])
v4_native = float(v4["timing"]["native"]["median_ms"])
assert math.isfinite(v2_native) and math.isfinite(v4_native) and v2_native > 0 and v4_native > 0
threshold = v2_native * 0.95
accepted = v4_native <= threshold
decision = {
    "schema": "c2-native-plugin-v4-vs-v2-promotion-v1",
    "v2_native_median_ms": v2_native,
    "v4_native_median_ms": v4_native,
    "required_v4_max_ms": threshold,
    "relative_improvement_fraction": 1.0 - v4_native / v2_native,
    "required_improvement_fraction": 0.05,
    "all_correctness_and_comparability_gates_pass": True,
    "accepted": accepted,
    "rejection_policy": "If accepted is false, retain v2; this script never overwrites v2 artifacts.",
}
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(decision, sort_keys=True))
raise SystemExit(0 if accepted else 3)
PY
decision_rc=$?
set -e

snapshot POST
apps_after=$(compute_apps) || { echo "unable to query post-run compute apps" >&2; exit 1; }
[[ -z "${apps_after}" ]]
sha256sum "${RESULT}" "${TRACE}" "${INSTALLED_MEMBERS}" "${BASELINE_OPS}" \
  "${INSTALLED_PRE}" "${INSTALLED_POST}" "${DECISION}" \
  "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" \
  > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
exit "${decision_rc}"
