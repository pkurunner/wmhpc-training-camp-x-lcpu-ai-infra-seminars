#!/usr/bin/env bash
# Install the experimental overlay wheel into an isolated target directory,
# then re-run the strict real-backend/profiler/ABBA gate from that wheel.
#SBATCH --job-name=c2-native-wheel-run
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-wheel-runtime-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONOPTIMIZE=0
export PYTHONPATH=

ARTIFACT_DIR=/home/lcpu/85117379/c2-native-wheel-runtime-artifacts-20260829
WHEEL_DIR=/home/lcpu/85117379/c2-native-wheel-artifacts-20260829
WHEEL="${WHEEL_DIR}/vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl"
PROVENANCE="${WHEEL_DIR}/c2-native-overlay-provenance.json"
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
UV=/usr/local/bin/uv
BASELINE_VLLM=/home/lcpu/85117379/venv-vllm-d4da0c5/lib/python3.12/site-packages/vllm
C2_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02/team/c2_msa_decode
HARNESS=/home/lcpu/85117379/native_c2_full_backend_bench_20260829.py
RUNTIME_PARENT=/home/lcpu/85117379/c2-native-wheel-runtime-20260829
RUNTIME="${RUNTIME_PARENT}/job${SLURM_JOB_ID}"
RESULT="${ARTIFACT_DIR}/wheel-full-backend-job${SLURM_JOB_ID}.json"
TRACE="${ARTIFACT_DIR}/wheel-full-backend-job${SLURM_JOB_ID}.chrome.json"
BASELINE_OPS="${ARTIFACT_DIR}/baseline-stable-ops-job${SLURM_JOB_ID}.json"
INSTALLED_OPS="${ARTIFACT_DIR}/installed-stable-ops-job${SLURM_JOB_ID}.json"
EXPECTED_WHEEL_SHA=22135a9baf42c9418728a2499f509fa521a5d6be11e0819d34c3c101c28a2c6a
EXPECTED_PROVENANCE_SHA=eb9de7399c56d472e57ad6fa69c0a444ad5886002d26e988e3a2ce85a11b0af1
EXPECTED_HARNESS_SHA=7883edc25df48e3b69a9f4948775a1baafd3ba0b1bd17edcfeac6644aa7b4762

mkdir -p "${ARTIFACT_DIR}" "${RUNTIME_PARENT}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1
trap 'rc=$?; printf "FINAL_RC=%s UTC=%s\n" "${rc}" "$(date -u +%FT%TZ)"' EXIT
export PATH="/home/lcpu/85117379/.local/bin:/usr/local/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin"

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
[[ -f "${WHEEL}" && -f "${PROVENANCE}" && -f "${HARNESS}" ]]
[[ -d "${C2_ROOT}" && -d "${BASELINE_VLLM}" ]]
[[ -x "${UV}" ]]
"${UV}" --version
[[ ! -e "${RUNTIME}" ]]
"${PYTHON}" -c 'raise SystemExit(0 if __debug__ else 1)'

printf '%s  %s\n' \
  "${EXPECTED_WHEEL_SHA}" "${WHEEL}" \
  "${EXPECTED_PROVENANCE_SHA}" "${PROVENANCE}" \
  "${EXPECTED_HARNESS_SHA}" "${HARNESS}" \
  | sha256sum -c -
sha256sum "${WHEEL}" "${PROVENANCE}" "${HARNESS}" \
  | tee "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
"${UV}" pip install --python "${PYTHON}" --target "${RUNTIME}" --no-deps "${WHEEL}"

"${PYTHON}" - "${RUNTIME}" "${PROVENANCE}" "${WHEEL}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

runtime, provenance_path, wheel = map(Path, sys.argv[1:])
provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
assert provenance["schema"] == "c2-native-overlay-provenance-v1"
assert provenance["experimental_overlay_same_distribution_version"] is True
wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
assert wheel_hash == provenance["derived_wheel"]["sha256"]
checks = []
for member in provenance["overlay_members"]:
    installed = runtime / member["archive_member"]
    assert installed.is_file(), installed
    digest = hashlib.sha256(installed.read_bytes()).hexdigest()
    assert digest == member["zip_sha256"], member["archive_member"]
    checks.append({
        "archive_member": member["archive_member"],
        "installed_path": str(installed),
        "sha256": digest,
        "size": installed.stat().st_size,
    })
out = runtime.parent / f"job{runtime.name.removeprefix('job')}-installed-overlay.json"
out.write_text(json.dumps({
    "schema": "c2-native-wheel-installed-overlay-v1",
    "wheel_sha256": wheel_hash,
    "runtime": str(runtime),
    "overlay_members": checks,
    "all_gates_pass": True,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out)
PY

INSTALLED_AUDIT="${RUNTIME_PARENT}/job${SLURM_JOB_ID}-installed-overlay.json"
mv "${INSTALLED_AUDIT}" "${ARTIFACT_DIR}/installed-overlay-job${SLURM_JOB_ID}.json"

# Capture the immutable baseline wheel's stable dispatcher surface in a clean
# process.  Record op names, full schemas, and the dispatch-key/registration
# kind surface (registration source paths are intentionally excluded).
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= \
  "${PYTHON}" - "${BASELINE_VLLM}" "${BASELINE_OPS}" <<'PY'
import json
from pathlib import Path
import sys
import torch
import vllm
import vllm._C_stable_libtorch as stable

required_private_apis = (
    "_dispatch_get_all_op_names",
    "_dispatch_dump_table",
    "_jit_get_all_schemas",
    "_dispatch_has_kernel_for_dispatch_key",
)
assert all(callable(getattr(torch._C, name, None))
           for name in required_private_apis)

def schema_key(schema):
    overload = schema.overload_name
    return f"{schema.name}.{overload}" if overload else schema.name

def dispatch_surface(op):
    rows = {}
    for line in torch._C._dispatch_dump_table(op).splitlines():
        key, sep, registration = line.partition(": ")
        if not sep:
            continue
        left = registration.rfind("[")
        assert left >= 0 and registration.endswith("]"), (op, line)
        kind = registration[left:]
        rows[key] = kind
    return rows

expected = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
root = Path(vllm.__file__).resolve().parent
assert root == expected, (root, expected)
assert Path(stable.__file__).resolve() == expected / "_C_stable_libtorch.abi3.so"
assert torch.__version__ == "2.13.0+cu130"
assert torch.version.cuda == "13.0"
ops = sorted(name for name in torch._C._dispatch_get_all_op_names()
             if name.startswith("_C::"))
assert ops
schemas = {}
for schema in torch._C._jit_get_all_schemas():
    if schema.name.startswith("_C::"):
        schemas.setdefault(schema_key(schema), []).append(str(schema))
schemas = {key: sorted(value) for key, value in sorted(schemas.items())}
assert set(ops) == set(schemas), (set(ops) - set(schemas), set(schemas) - set(ops))
dispatch = {op: dispatch_surface(op) for op in ops}
assert all(dispatch.values())
output.write_text(json.dumps({
    "schema": "c2-native-baseline-stable-dispatch-surface-v2",
    "vllm_root": str(root),
    "stable_library": str(Path(stable.__file__).resolve()),
    "torch_version": torch.__version__,
    "ops": ops,
    "op_count": len(ops),
    "schemas": schemas,
    "dispatch_key_registration_kinds": dispatch,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${RUNTIME}:${C2_ROOT}" \
  "${PYTHON}" - "${RUNTIME}" "${BASELINE_OPS}" "${INSTALLED_OPS}" <<'PY'
import importlib.metadata
import json
from pathlib import Path
import sys
import torch
import vllm
import vllm._C_stable_libtorch as stable

required_private_apis = (
    "_dispatch_get_all_op_names",
    "_dispatch_dump_table",
    "_jit_get_all_schemas",
    "_dispatch_has_kernel_for_dispatch_key",
)
assert all(callable(getattr(torch._C, name, None))
           for name in required_private_apis)

def schema_key(schema):
    overload = schema.overload_name
    return f"{schema.name}.{overload}" if overload else schema.name

def dispatch_surface(op):
    rows = {}
    for line in torch._C._dispatch_dump_table(op).splitlines():
        key, sep, registration = line.partition(": ")
        if not sep:
            continue
        left = registration.rfind("[")
        assert left >= 0 and registration.endswith("]"), (op, line)
        kind = registration[left:]
        rows[key] = kind
    return rows

runtime = Path(sys.argv[1]).resolve()
baseline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
audit_path = Path(sys.argv[3])
root = Path(vllm.__file__).resolve().parent
assert root == runtime / "vllm", (root, runtime)
assert Path(stable.__file__).resolve() == runtime / "vllm/_C_stable_libtorch.abi3.so"
assert importlib.metadata.distribution("vllm").locate_file("vllm").resolve() == root
assert torch.__version__ == baseline["torch_version"], (
    torch.__version__, baseline["torch_version"]
)
assert hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "native_c2_msa_decode")
assert torch._C._dispatch_has_kernel_for_dispatch_key(
    "_C::native_c2_msa_decode", "CUDA")
installed_ops = sorted(name for name in torch._C._dispatch_get_all_op_names()
                       if name.startswith("_C::"))
baseline_ops = set(baseline["ops"])
installed_set = set(installed_ops)
missing = sorted(baseline_ops - installed_set)
added = sorted(installed_set - baseline_ops)
assert not missing, missing
assert added == ["_C::native_c2_msa_decode"], added
installed_schemas = {}
for schema in torch._C._jit_get_all_schemas():
    if schema.name.startswith("_C::"):
        installed_schemas.setdefault(schema_key(schema), []).append(str(schema))
installed_schemas = {
    key: sorted(value) for key, value in sorted(installed_schemas.items())
}
assert set(installed_ops) == set(installed_schemas), (
    set(installed_ops) - set(installed_schemas),
    set(installed_schemas) - set(installed_ops),
)
baseline_schemas = baseline["schemas"]
schema_changes = {
    op: {"baseline": baseline_schemas[op], "installed": installed_schemas.get(op)}
    for op in sorted(baseline_ops)
    if installed_schemas.get(op) != baseline_schemas[op]
}
assert not schema_changes, schema_changes
added_schemas = {
    op: installed_schemas[op] for op in added
}
assert set(added_schemas) == {"_C::native_c2_msa_decode"}
installed_dispatch = {op: dispatch_surface(op) for op in installed_ops}
dispatch_changes = {
    op: {
        "baseline": baseline["dispatch_key_registration_kinds"][op],
        "installed": installed_dispatch.get(op),
    }
    for op in sorted(baseline_ops)
    if installed_dispatch.get(op)
    != baseline["dispatch_key_registration_kinds"][op]
}
assert not dispatch_changes, dispatch_changes
audit_path.write_text(json.dumps({
    "schema": "c2-native-installed-stable-dispatch-surface-v2",
    "baseline_op_count": len(baseline_ops),
    "installed_op_count": len(installed_set),
    "torch_version": torch.__version__,
    "missing_baseline_ops": missing,
    "added_ops": added,
    "added_schemas": added_schemas,
    "baseline_schema_changes": schema_changes,
    "baseline_dispatch_key_registration_kind_changes": dispatch_changes,
    "all_baseline_op_names_schemas_and_dispatch_surfaces_preserved": True,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print({"vllm_root": str(root), "version": importlib.metadata.version("vllm")})
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
printf '%s  %s\n' \
  "${EXPECTED_WHEEL_SHA}" "${WHEEL}" \
  "${EXPECTED_PROVENANCE_SHA}" "${PROVENANCE}" \
  "${EXPECTED_HARNESS_SHA}" "${HARNESS}" \
  | sha256sum -c -
sha256sum "${RESULT}" "${TRACE}" \
  "${ARTIFACT_DIR}/installed-overlay-job${SLURM_JOB_ID}.json" \
  "${BASELINE_OPS}" "${INSTALLED_OPS}" \
  "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" \
  > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
