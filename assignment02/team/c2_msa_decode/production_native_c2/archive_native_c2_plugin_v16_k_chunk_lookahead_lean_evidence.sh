#!/usr/bin/env bash
set -euo pipefail

umask 022

: "${C2_EXPECTED_ARCHIVE_SCRIPT_SHA:?submit the exact independently reviewed archive-script SHA-256}"

HOME_ROOT=/home/lcpu/85117379
SCRIPT_PATH=${HOME_ROOT}/archive_native_c2_plugin_v16_k_chunk_lookahead_lean_evidence.sh
STAGE=${HOME_ROOT}/c2-native-plugin-v16-k-chunk-lookahead-lean-evidence-stage-20260831
ARCHIVE=${HOME_ROOT}/c2-native-plugin-v16-k-chunk-lookahead-lean-evidence-20260831.tar.gz
MANIFEST=${HOME_ROOT}/c2-native-plugin-v16-k-chunk-lookahead-lean-evidence-20260831.manifest.sha256
SIDECAR=${ARCHIVE}.sha256

FULL_ARCHIVE=${HOME_ROOT}/c2-native-plugin-v16-k-chunk-lookahead-evidence-20260831.tar.gz
FULL_MANIFEST=${HOME_ROOT}/c2-native-plugin-v16-k-chunk-lookahead-evidence-20260831.manifest.sha256
FULL_SIDECAR=${FULL_ARCHIVE}.sha256
FIXTURE_MEMBER=c2-native-plugin-v16-k-chunk-lookahead-stress-3pct-artifacts-20260831/job13789/fixtures-8-seeds.pt

WHEEL_REL=c2-native-plugin-v16-k-chunk-lookahead-overlay-wheel-artifacts-20260831/job13832
WHEEL=${HOME_ROOT}/${WHEEL_REL}
WHEEL_PAYLOAD=vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl

LIFECYCLE_REL=c2-native-plugin-v16-k-chunk-lookahead-lifecycle-artifacts-20260831/job13845
LIFECYCLE=${HOME_ROOT}/${LIFECYCLE_REL}

EXPECTED_FULL_ARCHIVE_SHA=784a0d8de98b6ffba3532f55c11e65a3c7c35bb2528b72994f5c4f1496bd104f
EXPECTED_FULL_MANIFEST_SHA=1d84120486a0b0ec613cbb63c696d698de5cb5235bdcd9ff8893febb6564d6c0
EXPECTED_FULL_SIDECAR_SHA=8048f1d14875d4684b1877d737be911986bba39c09d9f2dde0a773d312379c40
EXPECTED_WHEEL_SHA=3947fab41739c98a30a8fd5486b867347b932f3419def3bfbd846db458ba90a9
EXPECTED_WHEEL_OUTPUTS_SHA=af8cdc0ec460a3b04e91dba0fcb3e1b325edc9ea7d7ff6257d245b9593de3d02
EXPECTED_WHEEL_FINAL_OUTPUT_SHA=705b898b3ffb31d4f61ecb9e107cc03d4035eaac7a7447fb68c1dc7e78a105d4
EXPECTED_WHEEL_FINAL_SHA=77ee692757cb613eefe773b9927588789dfddf94c57f7181e122bffaa3c4363b
EXPECTED_WHEEL_DRIVER_SHA=a28e95da73184027cf36720d087a04881e35681eadfe20333cad112b8a5f250c
EXPECTED_LIFECYCLE_OUTPUTS_SHA=5cc19ed7732bc1eeb16f5fc62dae0ca8500717e71b3693b7ffee16cf7a6588b9
EXPECTED_LIFECYCLE_FINAL_SHA=ed69b4477aa54d20c595515c190c78b68aad85c6efd413f9db02da297ac6e884
EXPECTED_LIFECYCLE_SIDECAR_SHA=3f72104cbc6d544536d5403fb3d801e3e61e3e1938db633fde241381dbd1077d
EXPECTED_LIFECYCLE_RESULT_SHA=b3c45988ad20daf352f598f74a8be287f20e417b12b1e8e6b18aee62ad991ade
EXPECTED_WHEEL_SCRIPT_SHA=904092bbc1f94346562e42d0c6032fc9a5285c79b4ee7cf521cc207e7adf2deb
EXPECTED_LIFECYCLE_HARNESS_SHA=36a52047c22296c0b0678093e11a8978a7d0e5eaba967bcc36106cea98b369d8
EXPECTED_LIFECYCLE_SCRIPT_SHA=2182d24b2e5a67860ea373bdabe8f5df15d1189f8b6fc01eb012547b461e65a0

require_regular_canonical() {
  local path=$1
  [[ -f "${path}" ]]
  [[ ! -L "${path}" ]]
  [[ "$(realpath -e "${path}")" == "${path}" ]]
}

require_directory_canonical() {
  local path=$1
  [[ -d "${path}" ]]
  [[ ! -L "${path}" ]]
  [[ "$(realpath -e "${path}")" == "${path}" ]]
}

for path in "${SCRIPT_PATH}" "${FULL_ARCHIVE}" "${FULL_MANIFEST}" "${FULL_SIDECAR}"; do
  require_regular_canonical "${path}"
done
for path in "${WHEEL}" "${LIFECYCLE}"; do
  require_directory_canonical "${path}"
done
require_directory_canonical "${HOME_ROOT}"
[[ "$(stat -c '%U' "${HOME_ROOT}")" == "85117379" ]]

for path in "${STAGE}" "${ARCHIVE}" "${MANIFEST}" "${SIDECAR}"; do
  [[ ! -e "${path}" ]]
  [[ ! -L "${path}" ]]
done

[[ "$(sha256sum "${SCRIPT_PATH}" | awk '{print $1}')" == "${C2_EXPECTED_ARCHIVE_SCRIPT_SHA}" ]]
[[ "$(sha256sum "${FULL_ARCHIVE}" | awk '{print $1}')" == "${EXPECTED_FULL_ARCHIVE_SHA}" ]]
[[ "$(sha256sum "${FULL_MANIFEST}" | awk '{print $1}')" == "${EXPECTED_FULL_MANIFEST_SHA}" ]]
[[ "$(sha256sum "${FULL_SIDECAR}" | awk '{print $1}')" == "${EXPECTED_FULL_SIDECAR_SHA}" ]]
sha256sum -c "${FULL_SIDECAR}"

export FULL_ARCHIVE FULL_MANIFEST FIXTURE_MEMBER
python3 - <<'PY'
import hashlib
import os
from pathlib import Path, PurePosixPath
import tarfile

manifest_path = Path(os.environ["FULL_MANIFEST"])
records = {}
for line in manifest_path.read_text(encoding="utf-8").splitlines():
    digest, name = line.split("  ", 1)
    if len(digest) != 64 or name in records:
        raise SystemExit(f"invalid full-manifest record: {line!r}")
    records[name] = digest
if len(records) != 87:
    raise SystemExit(f"expected 87 full-manifest records, got {len(records)}")

archive_path = Path(os.environ["FULL_ARCHIVE"])
seen = {}
sizes = {}
with tarfile.open(archive_path, "r:gz") as archive:
    members = archive.getmembers()
    if len(members) != 87:
        raise SystemExit(f"expected 87 full-archive members, got {len(members)}")
    for member in members:
        pure = PurePosixPath(member.name)
        if (
            not member.isreg()
            or pure.is_absolute()
            or ".." in pure.parts
            or "\\" in member.name
            or member.name in seen
        ):
            raise SystemExit(f"unsafe or duplicate full-archive member: {member.name!r}")
        stream = archive.extractfile(member)
        if stream is None:
            raise SystemExit(f"unreadable full-archive member: {member.name!r}")
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        seen[member.name] = digest.hexdigest()
        sizes[member.name] = member.size

if seen != records:
    missing = sorted(set(records) - set(seen))
    extra = sorted(set(seen) - set(records))
    mismatched = sorted(name for name in set(seen) & set(records) if seen[name] != records[name])
    raise SystemExit(
        f"full archive/manifest mismatch: missing={missing}, extra={extra}, mismatched={mismatched}"
    )

fixture = os.environ["FIXTURE_MEMBER"]
if fixture not in seen or sizes[fixture] != 68219593:
    raise SystemExit("full replay archive does not retain the exact stress fixture")
print(f"FULL_ARCHIVE_FILES={len(seen)}")
print(f"FIXTURE_SHA256={seen[fixture]}")
print(f"FIXTURE_BYTES={sizes[fixture]}")
PY

for path in \
  "${WHEEL}/outputs-job13832.sha256" \
  "${WHEEL}/final-output-manifest-job13832.sha256" \
  "${WHEEL}/final-status-job13832.txt" \
  "${WHEEL}/driver-job13832.log" \
  "${WHEEL}/${WHEEL_PAYLOAD}" \
  "${LIFECYCLE}/outputs-job13845.sha256" \
  "${LIFECYCLE}/final-status-job13845.txt" \
  "${LIFECYCLE}/final-status-job13845.sha256" \
  "${LIFECYCLE}/native-c2-v16-k-chunk-lookahead-lifecycle-job13845.json"; do
  require_regular_canonical "${path}"
done

[[ "$(sha256sum "${WHEEL}/outputs-job13832.sha256" | awk '{print $1}')" == "${EXPECTED_WHEEL_OUTPUTS_SHA}" ]]
[[ "$(sha256sum "${WHEEL}/final-output-manifest-job13832.sha256" | awk '{print $1}')" == "${EXPECTED_WHEEL_FINAL_OUTPUT_SHA}" ]]
[[ "$(sha256sum "${WHEEL}/final-status-job13832.txt" | awk '{print $1}')" == "${EXPECTED_WHEEL_FINAL_SHA}" ]]
[[ "$(sha256sum "${WHEEL}/driver-job13832.log" | awk '{print $1}')" == "${EXPECTED_WHEEL_DRIVER_SHA}" ]]
[[ "$(sha256sum "${WHEEL}/${WHEEL_PAYLOAD}" | awk '{print $1}')" == "${EXPECTED_WHEEL_SHA}" ]]
sha256sum -c "${WHEEL}/outputs-job13832.sha256"
sha256sum -c "${WHEEL}/final-output-manifest-job13832.sha256"
grep -Eq '^FINAL_RC=0 CLEANUP_RC=0 UTC=' "${WHEEL}/final-status-job13832.txt"

[[ "$(sha256sum "${LIFECYCLE}/outputs-job13845.sha256" | awk '{print $1}')" == "${EXPECTED_LIFECYCLE_OUTPUTS_SHA}" ]]
[[ "$(sha256sum "${LIFECYCLE}/final-status-job13845.txt" | awk '{print $1}')" == "${EXPECTED_LIFECYCLE_FINAL_SHA}" ]]
[[ "$(sha256sum "${LIFECYCLE}/final-status-job13845.sha256" | awk '{print $1}')" == "${EXPECTED_LIFECYCLE_SIDECAR_SHA}" ]]
[[ "$(sha256sum "${LIFECYCLE}/native-c2-v16-k-chunk-lookahead-lifecycle-job13845.json" | awk '{print $1}')" == "${EXPECTED_LIFECYCLE_RESULT_SHA}" ]]
sha256sum -c "${LIFECYCLE}/outputs-job13845.sha256"
sha256sum -c "${LIFECYCLE}/final-status-job13845.sha256"
grep -Eq '^FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 TEE_RC=0 MANIFEST_RC=0 CLEANUP_RC=0 POST_APPS_EMPTY=true ' "${LIFECYCLE}/final-status-job13845.txt"

WHEEL_SCRIPT=${HOME_ROOT}/build_native_c2_plugin_v16_k_chunk_lookahead_overlay_wheel_20260831.slurm
LIFECYCLE_HARNESS=${HOME_ROOT}/native_c2_v16_k_chunk_lookahead_lifecycle_20260831.py
LIFECYCLE_SCRIPT=${HOME_ROOT}/validate_native_c2_plugin_v16_k_chunk_lookahead_lifecycle.slurm
for path in "${WHEEL_SCRIPT}" "${LIFECYCLE_HARNESS}" "${LIFECYCLE_SCRIPT}"; do
  require_regular_canonical "${path}"
done
[[ "$(sha256sum "${WHEEL_SCRIPT}" | awk '{print $1}')" == "${EXPECTED_WHEEL_SCRIPT_SHA}" ]]
[[ "$(sha256sum "${LIFECYCLE_HARNESS}" | awk '{print $1}')" == "${EXPECTED_LIFECYCLE_HARNESS_SHA}" ]]
[[ "$(sha256sum "${LIFECYCLE_SCRIPT}" | awk '{print $1}')" == "${EXPECTED_LIFECYCLE_SCRIPT_SHA}" ]]
grep -Fqx "${EXPECTED_LIFECYCLE_SCRIPT_SHA}  /var/lib/slurm/slurmd/job13845/slurm_script" \
  "${LIFECYCLE}/inputs-job13845.sha256"

mkdir "${STAGE}"
tar -xzf "${FULL_ARCHIVE}" -C "${STAGE}" --exclude="${FIXTURE_MEMBER}"
[[ "$(find "${STAGE}" -type f | wc -l)" -eq 86 ]]
[[ ! -e "${STAGE}/${FIXTURE_MEMBER}" ]]
[[ -z "$(find "${STAGE}" -type l -print -quit)" ]]

mkdir -p "${STAGE}/${WHEEL_REL}"
for name in \
  aot-attestation-job13832.json \
  c2-native-plugin-overlay-manifest.json \
  c2-native-plugin-overlay-provenance.json \
  driver-job13832.log \
  final-output-manifest-job13832.sha256 \
  final-status-job13832.txt \
  inputs-job13832.sha256 \
  outputs-job13832.sha256 \
  promotion-prerequisites-job13832.json \
  result-job13832.json \
  staged-checkout-derivation-job13832.txt \
  wheel-driver-verification-job13832.json; do
  require_regular_canonical "${WHEEL}/${name}"
  install -m 0644 "${WHEEL}/${name}" "${STAGE}/${WHEEL_REL}/${name}"
done
[[ ! -e "${STAGE}/${WHEEL_REL}/${WHEEL_PAYLOAD}" ]]

mkdir -p "${STAGE}/${LIFECYCLE_REL}"
while read -r digest path; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]
  [[ "$(dirname "${path}")" == "${LIFECYCLE}" ]]
  require_regular_canonical "${path}"
  install -m 0644 "${path}" "${STAGE}/${LIFECYCLE_REL}/$(basename "${path}")"
done < "${LIFECYCLE}/outputs-job13845.sha256"
for name in outputs-job13845.sha256 final-status-job13845.txt final-status-job13845.sha256; do
  require_regular_canonical "${LIFECYCLE}/${name}"
  install -m 0644 "${LIFECYCLE}/${name}" "${STAGE}/${LIFECYCLE_REL}/${name}"
done

mkdir "${STAGE}/canonical-inputs-added-after-full-archive"
install -m 0644 "${WHEEL_SCRIPT}" "${STAGE}/canonical-inputs-added-after-full-archive/"
install -m 0644 "${LIFECYCLE_HARNESS}" "${STAGE}/canonical-inputs-added-after-full-archive/"
install -m 0644 "${LIFECYCLE_SCRIPT}" "${STAGE}/canonical-inputs-added-after-full-archive/"
install -m 0644 "${SCRIPT_PATH}" "${STAGE}/canonical-inputs-added-after-full-archive/"

mkdir "${STAGE}/full-replay-archive-control"
install -m 0644 "${FULL_MANIFEST}" "${STAGE}/full-replay-archive-control/"
install -m 0644 "${FULL_SIDECAR}" "${STAGE}/full-replay-archive-control/"

export STAGE WHEEL WHEEL_PAYLOAD EXPECTED_WHEEL_SHA EXPECTED_FULL_ARCHIVE_SHA
export EXPECTED_FULL_MANIFEST_SHA EXPECTED_FULL_SIDECAR_SHA
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


stage = Path(os.environ["STAGE"])
full_manifest = Path(os.environ["FULL_MANIFEST"])
wheel = Path(os.environ["WHEEL"]) / os.environ["WHEEL_PAYLOAD"]
fixture_member = os.environ["FIXTURE_MEMBER"]
records = dict(
    line.split("  ", 1) for line in full_manifest.read_text(encoding="utf-8").splitlines()
)
fixture_sha = next(digest for digest, name in records.items() if name == fixture_member)
policy = {
    "schema": "c2-native-c2-v16-k-chunk-lookahead-lean-evidence-policy-v2",
    "scope": "A controlled lean derivative of the immutable 87-file v16 AOT/directed/stress archive, plus hash-gated wheel and lifecycle metadata.",
    "derivation": {
        "full_archive_sha256": os.environ["EXPECTED_FULL_ARCHIVE_SHA"],
        "full_manifest_sha256": os.environ["EXPECTED_FULL_MANIFEST_SHA"],
        "full_sidecar_sha256": os.environ["EXPECTED_FULL_SIDECAR_SHA"],
        "full_files": 87,
        "full_tar_members": 87,
        "full_archive_contains_stress_fixture": True,
        "full_archive_contains_overlay_wheel": False,
    },
    "lean_exclusions": {
        "stress_fixture": {
            "member": fixture_member,
            "bytes": 68219593,
            "sha256": fixture_sha,
            "retained_in_full_archive": True,
        },
        "overlay_wheel": {
            "path": str(wheel),
            "bytes": wheel.stat().st_size,
            "sha256": digest_file(wheel),
            "retained_in_full_archive": False,
            "remote_canonical_payload_hash_gated": True,
        },
    },
    "included": {
        "success_jobs": [13773, 13786, 13789, 13832, 13845],
        "failure_jobs": [13767, 13783],
        "full_archive_files_except_stress_fixture": 86,
        "wheel_metadata_without_wheel_payload": True,
        "lifecycle_complete_evidence": True,
        "canonical_reviewed_inputs_added_after_full_archive": True,
    },
    "failure_boundary": {
        "job13767": "Durable pre-compilation Git/submodule network failure; not a v16 candidate failure.",
        "job13783": "The original Slurm log is empty and hash-fixed. The dated-harness mismatch was diagnosed by deterministic operator preflight replay, not by that empty log.",
    },
    "lifecycle_input_manifest_note": "The in-job input manifest names the ephemeral Slurm path; identical reviewed script bytes with the same SHA-256 are retained under canonical-inputs-added-after-full-archive.",
}
if wheel.stat().st_size != 313122399:
    raise SystemExit("unexpected wheel payload size")
if policy["lean_exclusions"]["overlay_wheel"]["sha256"] != os.environ["EXPECTED_WHEEL_SHA"]:
    raise SystemExit("wheel payload drifted while writing policy")
(stage / "lean-evidence-policy.json").write_text(
    json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

[[ ! -e "${STAGE}/${FIXTURE_MEMBER}" ]]
[[ -z "$(find "${STAGE}" -type f -name '*.whl' -print -quit)" ]]
[[ -z "$(find "${STAGE}" -type l -print -quit)" ]]

(
  cd "${STAGE}"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "${MANIFEST}"
  sha256sum -c "${MANIFEST}"
)

tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
  -C "$(dirname "${STAGE}")" -czf "${ARCHIVE}" "$(basename "${STAGE}")"
sha256sum "${ARCHIVE}" > "${SIDECAR}"
sha256sum -c "${SIDECAR}"

export ARCHIVE MANIFEST
python3 - <<'PY'
import hashlib
import os
from pathlib import Path, PurePosixPath
import tarfile

archive_path = Path(os.environ["ARCHIVE"])
manifest_path = Path(os.environ["MANIFEST"])
stage_name = Path(os.environ["STAGE"]).name
expected = {}
for line in manifest_path.read_text(encoding="utf-8").splitlines():
    digest, name = line.split("  ", 1)
    expected[name.removeprefix("./")] = digest

actual = {}
with tarfile.open(archive_path, "r:gz") as archive:
    for member in archive.getmembers():
        if member.isdir():
            continue
        pure = PurePosixPath(member.name)
        if not member.isreg() or pure.is_absolute() or ".." in pure.parts or "\\" in member.name:
            raise SystemExit(f"unsafe lean member: {member.name!r}")
        prefix = f"{stage_name}/"
        if not member.name.startswith(prefix):
            raise SystemExit(f"unexpected lean root: {member.name!r}")
        relative = member.name[len(prefix):]
        if relative in actual:
            raise SystemExit(f"duplicate lean member: {relative!r}")
        stream = archive.extractfile(member)
        if stream is None:
            raise SystemExit(f"unreadable lean member: {member.name!r}")
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        actual[relative] = digest.hexdigest()

if actual != expected:
    raise SystemExit("lean archive bytes do not exactly match the lean manifest")
fixture = os.environ["FIXTURE_MEMBER"]
if fixture in actual or any(name.endswith(".whl") for name in actual):
    raise SystemExit("lean archive unexpectedly contains an excluded large payload")
print(f"LEAN_MANIFEST_RECORDS={len(expected)}")
print(f"LEAN_TAR_FILE_MEMBERS={len(actual)}")
PY

printf 'STAGE=%s\n' "${STAGE}"
printf 'ARCHIVE=%s\n' "${ARCHIVE}"
printf 'MANIFEST=%s\n' "${MANIFEST}"
printf 'SIDECAR=%s\n' "${SIDECAR}"
sha256sum "${MANIFEST}" "${ARCHIVE}" "${SIDECAR}"
