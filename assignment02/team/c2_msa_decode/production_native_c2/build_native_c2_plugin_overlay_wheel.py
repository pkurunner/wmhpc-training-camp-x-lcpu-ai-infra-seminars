#!/usr/bin/env python3
"""Package an independent native-C2 plugin into an auditable vLLM overlay wheel.

The experimental wheel deliberately leaves the baseline
``vllm/_C_stable_libtorch.abi3.so`` payload byte-for-byte unchanged.  It adds
one ``vllm/_native_c2_msa_decode_plugin.abi3.so`` plugin and one Python adapter,
replaces only four reviewed Python dispatch files, then recreates and fully
verifies ``RECORD``.
This is a repackaging tool: it never builds vLLM and never changes an installed
environment.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any
import zipfile


EXPECTED_BASELINE_WHEEL = (
    "vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl"
)
EXPECTED_D4_COMMIT = "d4da0c55af3aa231b6209bf77871f3ed36eab0d2"
EXPECTED_BASELINE_SHA256 = (
    "91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06"
)
STABLE_MEMBER = "vllm/_C_stable_libtorch.abi3.so"
PLUGIN_ARCHIVE_MEMBER = "vllm/_native_c2_msa_decode_plugin.abi3.so"
EXPECTED_BASELINE_STABLE_MEMBER_SHA256 = (
    "cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442"
)

# These are the only Python files which may be overlaid.  Four replace members
# from the baseline wheel; the native-C2 adapter and plugin binary are new
# members.  The stable extension is never an overlay source.
DISPATCH_MEMBERS: tuple[str, ...] = (
    "vllm/_custom_ops.py",
    "vllm/config/attention.py",
    "vllm/v1/attention/backends/registry.py",
    "vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py",
    "vllm/models/minimax_m3/nvidia/sparse_attention_msa.py",
)

ADDED_DISPATCH_MEMBERS: tuple[str, ...] = (
    "vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py",
)
REPLACED_DISPATCH_MEMBERS: tuple[str, ...] = tuple(
    member for member in DISPATCH_MEMBERS if member not in ADDED_DISPATCH_MEMBERS
)

# These hashes are the exact post-patch files obtained by applying, in order,
# exact_d4_python_dispatch.patch and
# exact_d4_native_c2_plugin_python_loader.patch to the clean d4 tree.  They
# are an input gate, not merely provenance: packaging aborts before writing a
# wheel if any dispatch source differs.
EXPECTED_DISPATCH_MEMBER_SHA256: dict[str, str] = {
    "vllm/_custom_ops.py": (
        "b57dfa819424b89bed8bf39924d5a4a3d2ebf327b39f3ae7c2e9e65b61c2398d"
    ),
    "vllm/config/attention.py": (
        "caf68c29d8d36324d252a66f0811a25b26d11da9860732359076fbb76625b5ae"
    ),
    "vllm/v1/attention/backends/registry.py": (
        "cd539941417fd5faf6672676de0f8ae15c9adb2a915b47cdd508e154c34faea4"
    ),
    "vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py": (
        "6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412"
    ),
    "vllm/models/minimax_m3/nvidia/sparse_attention_msa.py": (
        "07d00372ca1995aa96159a0b9f8949b1b40f817a63f49224e88aafac1b722200"
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def b64_sha256(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def require_regular_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    return resolved


def is_safe_archive_path(name: str) -> bool:
    candidate = Path(name)
    return bool(name) and "\\" not in name and not candidate.is_absolute() and ".." not in candidate.parts


def require_plugin_archive_member(name: str) -> str:
    if not is_safe_archive_path(name):
        raise ValueError(f"unsafe plugin archive member: {name!r}")
    if name != PLUGIN_ARCHIVE_MEMBER:
        raise ValueError(
            "plugin archive member must be exactly "
            f"{PLUGIN_ARCHIVE_MEMBER!r}; got {name!r}")
    return name


def assert_safe_archive_names(infos: list[zipfile.ZipInfo]) -> None:
    unsafe = [info.filename for info in infos if not is_safe_archive_path(info.filename)]
    if unsafe:
        raise ValueError(f"wheel contains unsafe archive paths: {unsafe[:3]}")
    names = [info.filename for info in infos]
    duplicate = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicate:
        raise ValueError(f"wheel contains duplicate members: {duplicate[:3]}")
    signatures = [
        name for name in names
        if name.endswith(".dist-info/RECORD.jws") or name.endswith(".dist-info/RECORD.p7s")
    ]
    if signatures:
        raise ValueError(f"refusing signed wheel whose signature would be invalidated: {signatures}")


def find_record_member(infos: list[zipfile.ZipInfo]) -> str:
    records = [info.filename for info in infos if info.filename.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise ValueError(f"wheel must contain exactly one dist-info/RECORD, found {records}")
    record = records[0]
    dist_info = record.removesuffix("/RECORD")
    names = {info.filename for info in infos}
    if not {f"{dist_info}/WHEEL", f"{dist_info}/METADATA"}.issubset(names):
        raise ValueError(f"RECORD dist-info directory is incomplete: {dist_info}")
    return record


def read_record_rows(data: bytes, record_name: str) -> list[list[str]]:
    try:
        rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{record_name} is not UTF-8") from exc
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError(f"{record_name} is empty or has malformed rows")
    return rows


def full_record_verification(
    archive: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
    record_name: str,
) -> dict[str, Any]:
    """Verify every RECORD payload row, including names, hashes and sizes."""
    rows = read_record_rows(archive.read(record_name), record_name)
    seen: set[str] = set()
    duplicate_rows: list[str] = []
    row_by_name: dict[str, list[str]] = {}
    for row in rows:
        if row[0] in seen:
            duplicate_rows.append(row[0])
        seen.add(row[0])
        row_by_name[row[0]] = row

    archive_payloads = {info.filename for info in infos if not info.is_dir()} - {record_name}
    record_payloads = set(row_by_name) - {record_name}
    missing = sorted(archive_payloads - record_payloads)
    unexpected = sorted(record_payloads - archive_payloads)
    mismatched: list[str] = []
    for name in sorted(archive_payloads):
        row = row_by_name.get(name)
        if row is None:
            continue
        data = archive.read(name)
        if row[1] != f"sha256={b64_sha256(data)}" or row[2] != str(len(data)):
            mismatched.append(name)
    summary = {
        "record_member": record_name,
        "record_row_count": len(rows),
        "payload_member_count": len(archive_payloads),
        "record_self_row_valid": row_by_name.get(record_name) == [record_name, "", ""],
        "all_payload_entries_validated": not (missing or unexpected or duplicate_rows or mismatched),
        "missing_member_rows": missing,
        "unexpected_member_rows": unexpected,
        "duplicate_record_rows": sorted(set(duplicate_rows)),
        "content_mismatch_members": mismatched,
        "row_name_list_sha256": sha256_bytes(
            "".join(f"{row[0]}\n" for row in rows).encode("utf-8")),
    }
    if not (summary["record_self_row_valid"] and summary["all_payload_entries_validated"]):
        raise ValueError(f"RECORD full verification failed: {json.dumps(summary, sort_keys=True)}")
    return summary


def clone_zipinfo(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    clone = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
    clone.comment = info.comment
    clone.extra = info.extra
    clone.internal_attr = info.internal_attr
    clone.external_attr = info.external_attr
    clone.create_system = info.create_system
    clone.create_version = info.create_version
    clone.extract_version = info.extract_version
    clone.flag_bits = info.flag_bits
    clone.compress_type = info.compress_type
    clone._compresslevel = getattr(info, "_compresslevel", None)  # noqa: SLF001
    return clone


def new_overlay_zipinfo(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    info._compresslevel = 9  # noqa: SLF001
    return info


def run_git(checkout: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", os.fspath(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def checkout_provenance(checkout: Path, expected_commit: str) -> dict[str, Any]:
    commit = run_git(checkout, "rev-parse", "HEAD").decode("ascii").strip()
    if commit != expected_commit:
        raise ValueError(f"plugin checkout commit mismatch: expected {expected_commit}, got {commit}")
    tracked_diff = run_git(checkout, "diff", "--binary", "--no-ext-diff", "HEAD")
    status = run_git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    raw_untracked = run_git(checkout, "ls-files", "--others", "--exclude-standard", "-z")
    untracked: list[dict[str, Any]] = []
    for raw in raw_untracked.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8")
        unresolved = checkout / relative
        if unresolved.is_symlink():
            resolved = unresolved.resolve(strict=True)
            untracked.append({
                "path": relative,
                "type": "symlink",
                "link_target": os.readlink(unresolved),
                "resolved_target": str(resolved),
                "target_sha256": sha256_file(resolved),
                "target_size": resolved.stat().st_size,
            })
        else:
            resolved = unresolved.resolve(strict=True)
            if not resolved.is_relative_to(checkout):
                raise ValueError(f"untracked path escapes plugin checkout: {relative}")
            if resolved.is_file():
                untracked.append({
                    "path": relative,
                    "sha256": sha256_file(resolved),
                    "size": resolved.stat().st_size,
                })
    return {
        "commit": commit,
        "tracked_diff_sha256": sha256_bytes(tracked_diff),
        "tracked_diff_size": len(tracked_diff),
        "status_porcelain_sha256": sha256_bytes(status),
        "status_porcelain_size": len(status),
        "untracked_files": untracked,
    }


def load_overlay_sources(
    checkout: Path,
    plugin_library: Path,
    plugin_member: str,
    expected_plugin_sha256: str,
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    if tuple(EXPECTED_DISPATCH_MEMBER_SHA256) != DISPATCH_MEMBERS:
        raise AssertionError("dispatch member SHA-256 table must exactly match DISPATCH_MEMBERS")
    plugin_data = plugin_library.read_bytes()
    plugin_hash = sha256_bytes(plugin_data)
    if plugin_hash != expected_plugin_sha256:
        raise ValueError(
            "plugin library SHA-256 mismatch: "
            f"expected {expected_plugin_sha256}, got {plugin_hash}")
    overlay_bytes = {plugin_member: plugin_data}
    provenance: list[dict[str, Any]] = [{
        "archive_member": plugin_member,
        "kind": "added_native_c2_plugin",
        "source_path": str(plugin_library),
        "source_sha256": plugin_hash,
        "source_size": len(plugin_data),
    }]
    for archive_member in DISPATCH_MEMBERS:
        source = require_regular_file(checkout / archive_member, f"dispatch source for {archive_member}")
        if not source.is_relative_to(checkout):
            raise ValueError(f"dispatch source escapes plugin checkout: {source}")
        data = source.read_bytes()
        source_hash = sha256_bytes(data)
        expected_hash = EXPECTED_DISPATCH_MEMBER_SHA256[archive_member]
        if source_hash != expected_hash:
            raise ValueError(
                "post-patch dispatch SHA-256 mismatch for "
                f"{archive_member}: expected {expected_hash}, got {source_hash}")
        if archive_member in overlay_bytes:
            raise AssertionError(f"duplicate overlay source: {archive_member}")
        overlay_bytes[archive_member] = data
        provenance.append({
            "archive_member": archive_member,
            "kind": (
                "added_python_dispatch"
                if archive_member in ADDED_DISPATCH_MEMBERS
                else "replaced_python_dispatch"
            ),
            "source_path": str(source),
            "source_sha256": source_hash,
            "expected_post_patch_sha256": expected_hash,
            "source_size": len(data),
        })
    if STABLE_MEMBER in overlay_bytes:
        raise AssertionError("stable extension must never be an overlay source")
    return overlay_bytes, provenance


def build_record(member_data: dict[str, bytes], record_name: str, payload_names: set[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for name in sorted(payload_names):
        data = member_data[name]
        writer.writerow([name, f"sha256={b64_sha256(data)}", str(len(data))])
    writer.writerow([record_name, "", ""])
    return buffer.getvalue().encode("utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def make_overlay_wheel(args: argparse.Namespace) -> dict[str, Path]:
    base_wheel = require_regular_file(Path(args.base_wheel), "baseline wheel")
    if base_wheel.name != args.expected_baseline_wheel_name:
        raise ValueError(
            f"baseline wheel filename mismatch: expected {args.expected_baseline_wheel_name}, "
            f"got {base_wheel.name}")
    base_hash = sha256_file(base_wheel)
    if base_hash != args.expected_baseline_sha256:
        raise ValueError(
            f"baseline wheel SHA-256 mismatch: expected {args.expected_baseline_sha256}, "
            f"got {base_hash}")
    checkout = require_directory(Path(args.plugin_checkout), "plugin checkout")
    plugin_library = require_regular_file(Path(args.plugin_library), "native-C2 plugin library")
    plugin_member = require_plugin_archive_member(args.plugin_archive_member)

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if not artifact_dir.is_dir():
        raise ValueError(f"artifact directory is not a directory: {artifact_dir}")
    output_wheel = (Path(args.output_wheel).expanduser().resolve()
                    if args.output_wheel else artifact_dir / base_wheel.name)
    try:
        output_wheel.relative_to(artifact_dir)
    except ValueError as exc:
        raise ValueError("output wheel must stay inside the artifact directory") from exc
    if output_wheel == base_wheel:
        raise ValueError("output wheel must not overwrite the baseline wheel")
    if output_wheel.name != base_wheel.name:
        raise ValueError("output wheel filename must equal the baseline filename")

    provenance_path = artifact_dir / "c2-native-plugin-overlay-provenance.json"
    manifest_path = artifact_dir / "c2-native-plugin-overlay-manifest.json"
    for output in (output_wheel, provenance_path, manifest_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {output}")

    git_info = checkout_provenance(checkout, args.expected_git_commit)
    overlay_bytes, overlay_provenance = load_overlay_sources(
        checkout, plugin_library, plugin_member, args.expected_plugin_sha256)

    with zipfile.ZipFile(base_wheel, "r") as base_archive:
        if base_archive.testzip() is not None:
            raise ValueError("baseline wheel CRC verification failed")
        base_infos = base_archive.infolist()
        assert_safe_archive_names(base_infos)
        record_name = find_record_member(base_infos)
        base_record_verification = full_record_verification(base_archive, base_infos, record_name)
        base_names = [info.filename for info in base_infos]
        base_name_set = set(base_names)
        if STABLE_MEMBER not in base_name_set:
            raise ValueError(f"baseline wheel lacks required stable extension: {STABLE_MEMBER}")
        missing_dispatch = sorted(set(REPLACED_DISPATCH_MEMBERS) - base_name_set)
        if missing_dispatch:
            raise ValueError(f"baseline wheel lacks reviewed dispatch members: {missing_dispatch}")
        unexpected_added_dispatch = sorted(
            set(ADDED_DISPATCH_MEMBERS) & base_name_set)
        if unexpected_added_dispatch:
            raise ValueError(
                "new dispatch members unexpectedly exist in baseline wheel: "
                f"{unexpected_added_dispatch}")
        if plugin_member in base_name_set:
            raise ValueError(f"plugin member must be newly added, already present: {plugin_member}")

        member_data = {info.filename: base_archive.read(info.filename) for info in base_infos}
        original_data = {info.filename: member_data[info.filename] for info in base_infos if not info.is_dir()}
        original_infos = {info.filename: info for info in base_infos}
        baseline_stable = original_data[STABLE_MEMBER]
        if sha256_bytes(baseline_stable) != EXPECTED_BASELINE_STABLE_MEMBER_SHA256:
            raise ValueError(
                "baseline wheel stable-member SHA-256 mismatch: expected "
                f"{EXPECTED_BASELINE_STABLE_MEMBER_SHA256}, got "
                f"{sha256_bytes(baseline_stable)}")
        member_data.update(overlay_bytes)
        payload_names = {info.filename for info in base_infos if not info.is_dir() and info.filename != record_name}
        payload_names.update(overlay_bytes)
        member_data[record_name] = build_record(member_data, record_name, payload_names)

        ordered_names = [name for name in base_names if name != record_name]
        ordered_names.extend([plugin_member, *ADDED_DISPATCH_MEMBERS])
        ordered_names.append(record_name)
        fd, temp_name = tempfile.mkstemp(prefix=f".{output_wheel.stem}.", suffix=".tmp", dir=artifact_dir)
        os.close(fd)
        temporary_wheel = Path(temp_name)
        try:
            with zipfile.ZipFile(temporary_wheel, "w", allowZip64=True) as result:
                for name in ordered_names:
                    data = member_data[name]
                    info = (new_overlay_zipinfo(name)
                            if name in overlay_bytes or name == record_name
                            else clone_zipinfo(original_infos[name]))
                    result.writestr(info, data)
            os.replace(temporary_wheel, output_wheel)
        except BaseException:
            temporary_wheel.unlink(missing_ok=True)
            raise

    with zipfile.ZipFile(output_wheel, "r") as derived_archive:
        if derived_archive.testzip() is not None:
            raise ValueError("derived wheel CRC verification failed")
        derived_infos = derived_archive.infolist()
        assert_safe_archive_names(derived_infos)
        if find_record_member(derived_infos) != record_name:
            raise ValueError("derived wheel changed the dist-info RECORD path")
        derived_record_verification = full_record_verification(derived_archive, derived_infos, record_name)
        derived_names = [info.filename for info in derived_infos]
        expected_names = base_name_set | {plugin_member, *ADDED_DISPATCH_MEMBERS}
        if set(derived_names) != expected_names or len(derived_names) != len(set(derived_names)):
            raise ValueError(
                "derived wheel member set is not baseline plus the reviewed plugin and adapter")

        allowed_changed = set(REPLACED_DISPATCH_MEMBERS) | {record_name}
        changed_unapproved: list[str] = []
        unchanged_count = 0
        unchanged_size = 0
        unchanged_digest = hashlib.sha256()
        for name, before in original_data.items():
            after = derived_archive.read(name)
            if name not in allowed_changed and after != before:
                changed_unapproved.append(name)
            if name not in allowed_changed and after == before:
                unchanged_count += 1
                unchanged_size += len(before)
                unchanged_digest.update(name.encode("utf-8"))
                unchanged_digest.update(b"\0")
                unchanged_digest.update(hashlib.sha256(before).digest())
                unchanged_digest.update(b"\n")
        if changed_unapproved:
            raise ValueError(f"unapproved baseline member changes: {changed_unapproved[:3]}")

        derived_stable = derived_archive.read(STABLE_MEMBER)
        if derived_stable != baseline_stable:
            raise ValueError("derived wheel changed baseline _C_stable_libtorch bytes")
        if sha256_bytes(derived_stable) != EXPECTED_BASELINE_STABLE_MEMBER_SHA256:
            raise ValueError("derived stable member failed pinned SHA-256 gate")
        stable_record = {
            "archive_member": STABLE_MEMBER,
            "baseline_sha256": sha256_bytes(baseline_stable),
            "baseline_size": len(baseline_stable),
            "derived_sha256": sha256_bytes(derived_stable),
            "derived_size": len(derived_stable),
            "byte_identical": True,
        }

        overlay_by_name = {item["archive_member"]: item for item in overlay_provenance}
        for name, source_data in overlay_bytes.items():
            zip_data = derived_archive.read(name)
            if zip_data != source_data:
                raise ValueError(f"overlay member content mismatch after packing: {name}")
            if name in EXPECTED_DISPATCH_MEMBER_SHA256 and (
                sha256_bytes(zip_data) != EXPECTED_DISPATCH_MEMBER_SHA256[name]
            ):
                raise ValueError(f"derived wheel dispatch SHA-256 mismatch: {name}")
            overlay_by_name[name].update({
                "action": (
                    "add"
                    if name == plugin_member or name in ADDED_DISPATCH_MEMBERS
                    else "replace"
                ),
                "zip_sha256": sha256_bytes(zip_data),
                "zip_size": len(zip_data),
                "source_matches_zip": True,
            })

    derived_hash = sha256_file(output_wheel)
    provenance: dict[str, Any] = {
        "schema": "c2-native-plugin-overlay-provenance-v1",
        "created_utc": utc_now(),
        "experimental_overlay_same_distribution_version": True,
        "baseline_wheel": {
            "path": str(base_wheel), "filename": base_wheel.name,
            "sha256": base_hash, "size": base_wheel.stat().st_size,
        },
        "derived_wheel": {
            "path": str(output_wheel), "filename": output_wheel.name,
            "sha256": derived_hash, "size": output_wheel.stat().st_size,
            "same_filename_as_baseline": output_wheel.name == base_wheel.name,
        },
        "plugin_checkout_base_git": git_info,
        "packager": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "plugin_archive_member": plugin_member,
        "plugin_expected_sha256": args.expected_plugin_sha256,
        "overlay_members": [overlay_by_name[plugin_member]] + [
            overlay_by_name[name] for name in DISPATCH_MEMBERS
        ],
        "expected_dispatch_member_sha256": EXPECTED_DISPATCH_MEMBER_SHA256,
        "baseline_stable_member": stable_record,
        "base_record_full_verification": base_record_verification,
        "derived_record_full_verification": derived_record_verification,
        "unchanged_baseline_members": {
            "definition": "baseline non-directory members excluding four reviewed Python dispatch replacements and RECORD",
            "count": unchanged_count,
            "total_uncompressed_size": unchanged_size,
            "name_and_content_sha256": unchanged_digest.hexdigest(),
            "unapproved_changed_members": changed_unapproved,
        },
        "member_set": {
            "baseline_count": len(base_names),
            "derived_count": len(derived_names),
            "new_members": [plugin_member, *ADDED_DISPATCH_MEMBERS],
            "removed_members": sorted(base_name_set - set(derived_names)),
        },
    }
    manifest: dict[str, Any] = {
        "schema": "c2-native-plugin-overlay-manifest-v1",
        "experimental_overlay_same_distribution_version": True,
        "baseline_wheel_sha256": base_hash,
        "derived_wheel_sha256": derived_hash,
        "derived_wheel_size": output_wheel.stat().st_size,
        "plugin_checkout_commit": git_info["commit"],
        "plugin_archive_member": plugin_member,
        "plugin_sha256": args.expected_plugin_sha256,
        "packager_sha256": provenance["packager"]["sha256"],
        "expected_dispatch_member_sha256": EXPECTED_DISPATCH_MEMBER_SHA256,
        "baseline_stable_member_sha256": stable_record["baseline_sha256"],
        "baseline_stable_member_byte_identical": stable_record["byte_identical"],
        "record_member": record_name,
        "overlay_archive_members": [plugin_member, *DISPATCH_MEMBERS],
        "provenance_file": provenance_path.name,
        "provenance_sha256": sha256_bytes(
            (json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2,
                        allow_nan=False) + "\n").encode("utf-8")),
    }
    write_json(provenance_path, provenance)
    write_json(manifest_path, manifest)
    return {"wheel": output_wheel, "provenance": provenance_path, "manifest": manifest_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-wheel",
        default="/home/lcpu/85117379/vllm-d4-wheel/" + EXPECTED_BASELINE_WHEEL,
    )
    parser.add_argument("--plugin-checkout", required=True,
                        help="exact-d4 checkout containing the five plugin dispatch files")
    parser.add_argument("--plugin-library", required=True,
                        help="already-built independent native C2 plugin shared library")
    parser.add_argument("--plugin-archive-member", required=True,
                        help=f"required exact new wheel member: {PLUGIN_ARCHIVE_MEMBER}")
    parser.add_argument("--expected-plugin-sha256", required=True,
                        help="reviewed SHA-256 of --plugin-library")
    parser.add_argument(
        "--artifact-dir", required=True,
        help="new job-specific directory for this wheel and all audit artifacts",
    )
    parser.add_argument("--output-wheel", default=None)
    parser.add_argument("--expected-baseline-wheel-name", default=EXPECTED_BASELINE_WHEEL)
    parser.add_argument("--expected-git-commit", default=EXPECTED_D4_COMMIT)
    parser.add_argument("--expected-baseline-sha256", default=EXPECTED_BASELINE_SHA256)
    return parser.parse_args()


def main() -> int:
    artifacts = make_overlay_wheel(parse_args())
    print(json.dumps({
        "schema": "c2-native-plugin-overlay-wheel-result-v1",
        "wheel": str(artifacts["wheel"]),
        "provenance": str(artifacts["provenance"]),
        "manifest": str(artifacts["manifest"]),
    }, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
