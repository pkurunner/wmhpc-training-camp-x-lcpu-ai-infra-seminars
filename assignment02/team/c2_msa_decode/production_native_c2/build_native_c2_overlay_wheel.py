#!/usr/bin/env python3
"""Create an auditable native-C2 overlay wheel from the exact d4 baseline.

This tool deliberately does *not* build vLLM or modify an installed environment.
It repacks a supplied baseline wheel, replacing only the declared native-C2
members and regenerating the wheel's RECORD.  The output wheel intentionally
keeps the upstream distribution version and filename; its provenance JSON marks
that as an experimental overlay so it cannot be mistaken for a released wheel.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from typing import Any
import zipfile


EXPECTED_BASELINE_WHEEL = (
    "vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl"
)
EXPECTED_D4_COMMIT = "d4da0c55af3aa231b6209bf77871f3ed36eab0d2"
EXPECTED_BASELINE_SHA256 = (
    "91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06"
)

# The archive paths are intentionally exact.  Adding a file to this table is a
# material packaging change and must be reviewed, rather than being inferred
# from a worktree diff.
OVERLAY_SOURCES: tuple[tuple[str, str], ...] = (
    ("vllm/_C_stable_libtorch.abi3.so", "__AOT_LIBRARY__"),
    ("vllm/_custom_ops.py", "vllm/_custom_ops.py"),
    ("vllm/config/attention.py", "vllm/config/attention.py"),
    ("vllm/v1/attention/backends/registry.py",
     "vllm/v1/attention/backends/registry.py"),
    ("vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py",
     "vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py"),
    ("vllm/models/minimax_m3/nvidia/sparse_attention_msa.py",
     "vllm/models/minimax_m3/nvidia/sparse_attention_msa.py"),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def b64_sha256(data: bytes) -> str:
    """Return the PEP 427 URL-safe, unpadded SHA-256 digest."""
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def require_regular_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    mode = resolved.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    return resolved


def is_safe_archive_path(name: str) -> bool:
    candidate = Path(name)
    return (
        bool(name)
        and "\\" not in name
        and not candidate.is_absolute()
        and ".." not in candidate.parts
    )


def assert_safe_archive_names(infos: list[zipfile.ZipInfo]) -> None:
    unsafe = [info.filename for info in infos if not is_safe_archive_path(info.filename)]
    if unsafe:
        raise ValueError(f"wheel contains unsafe archive paths: {unsafe[:3]}")
    names = [info.filename for info in infos]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"wheel contains duplicate members: {duplicates[:3]}")
    signatures = [
        name for name in names
        if name.endswith(".dist-info/RECORD.jws")
        or name.endswith(".dist-info/RECORD.p7s")
    ]
    if signatures:
        raise ValueError(
            "refusing a signed wheel: signature files would be invalidated "
            f"by overlay packaging: {signatures}")


def find_record_member(infos: list[zipfile.ZipInfo]) -> str:
    records = [info.filename for info in infos if info.filename.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise ValueError(f"wheel must contain exactly one dist-info/RECORD, found {records}")
    record = records[0]
    dist_info = record.removesuffix("/RECORD")
    required = {f"{dist_info}/WHEEL", f"{dist_info}/METADATA"}
    all_names = {info.filename for info in infos}
    if not required.issubset(all_names):
        raise ValueError(f"RECORD dist-info directory is incomplete: {dist_info}")
    return record


def read_record_rows(data: bytes, record_name: str) -> list[list[str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{record_name} is not UTF-8") from exc
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError(f"{record_name} is empty")
    if any(len(row) != 3 for row in rows):
        raise ValueError(f"{record_name} contains a row with a field count other than three")
    return rows


def full_record_verification(
    archive: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
    record_name: str,
) -> dict[str, Any]:
    """Validate every RECORD row against every non-directory archive member."""
    rows = read_record_rows(archive.read(record_name), record_name)
    seen: set[str] = set()
    duplicate_rows: list[str] = []
    row_by_name: dict[str, list[str]] = {}
    for row in rows:
        if row[0] in seen:
            duplicate_rows.append(row[0])
        seen.add(row[0])
        row_by_name[row[0]] = row

    member_infos = {info.filename: info for info in infos if not info.is_dir()}
    expected_payload_names = set(member_infos) - {record_name}
    actual_payload_names = set(row_by_name) - {record_name}
    missing = sorted(expected_payload_names - actual_payload_names)
    unexpected = sorted(actual_payload_names - expected_payload_names)

    failures: list[str] = []
    for name in sorted(expected_payload_names):
        row = row_by_name.get(name)
        if row is None:
            continue
        data = archive.read(name)
        expected_hash = f"sha256={b64_sha256(data)}"
        if row[1] != expected_hash or row[2] != str(len(data)):
            failures.append(name)

    record_row = row_by_name.get(record_name)
    record_self_valid = record_row == [record_name, "", ""]
    row_name_digest = sha256_bytes(
        "".join(f"{row[0]}\n" for row in rows).encode("utf-8"))
    summary = {
        "record_member": record_name,
        "record_row_count": len(rows),
        "payload_member_count": len(expected_payload_names),
        "record_self_row_valid": record_self_valid,
        "all_payload_entries_validated": not (missing or unexpected or failures or duplicate_rows),
        "missing_member_rows": missing,
        "unexpected_member_rows": unexpected,
        "duplicate_record_rows": sorted(set(duplicate_rows)),
        "content_mismatch_members": failures,
        "row_name_list_sha256": row_name_digest,
    }
    if not (summary["record_self_row_valid"] and summary["all_payload_entries_validated"]):
        raise ValueError(
            "RECORD full verification failed: "
            f"{json.dumps(summary, sort_keys=True, allow_nan=False)}")
    return summary


def clone_zipinfo(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    """Keep metadata for non-overlay members while writing their original bytes."""
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
        raise ValueError(
            f"derived checkout commit mismatch: expected {expected_commit}, got {commit}")
    tracked_diff = run_git(checkout, "diff", "--binary", "--no-ext-diff", "HEAD")
    status = run_git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    raw_untracked = run_git(checkout, "ls-files", "--others", "--exclude-standard", "-z")
    untracked_files = [
        item.decode("utf-8") for item in raw_untracked.split(b"\0") if item
    ]
    untracked_detail: list[dict[str, Any]] = []
    for relative in untracked_files:
        unresolved = checkout / relative
        if unresolved.is_symlink():
            target = os.readlink(unresolved)
            resolved_target = unresolved.resolve(strict=True)
            untracked_detail.append({
                "path": relative,
                "type": "symlink",
                "link_target": target,
                "resolved_target": str(resolved_target),
                "target_sha256": sha256_file(resolved_target),
                "target_size": resolved_target.stat().st_size,
            })
            continue
        candidate = unresolved.resolve(strict=True)
        if not candidate.is_relative_to(checkout):
            raise ValueError(f"untracked path escapes derived checkout: {relative}")
        if candidate.is_file():
            untracked_detail.append({
                "path": relative,
                "sha256": sha256_file(candidate),
                "size": candidate.stat().st_size,
            })
    return {
        "commit": commit,
        "tracked_diff_sha256": sha256_bytes(tracked_diff),
        "tracked_diff_size": len(tracked_diff),
        "status_porcelain_sha256": sha256_bytes(status),
        "status_porcelain_size": len(status),
        "untracked_files": untracked_detail,
    }


def load_overlay_sources(checkout: Path, aot_library: Path) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    overlay_bytes: dict[str, bytes] = {}
    provenance: list[dict[str, Any]] = []
    for archive_member, source_spec in OVERLAY_SOURCES:
        source = aot_library if source_spec == "__AOT_LIBRARY__" else (checkout / source_spec)
        source = require_regular_file(source, f"overlay source for {archive_member}")
        if source_spec != "__AOT_LIBRARY__" and not source.is_relative_to(checkout):
            raise ValueError(f"overlay source escapes derived checkout: {source}")
        data = source.read_bytes()
        overlay_bytes[archive_member] = data
        provenance.append({
            "archive_member": archive_member,
            "source_path": str(source),
            "source_sha256": sha256_bytes(data),
            "source_size": len(data),
        })
    if len(overlay_bytes) != len(OVERLAY_SOURCES):
        raise AssertionError("overlay member table contains duplicates")
    return overlay_bytes, provenance


def build_record(
    member_data: dict[str, bytes],
    record_name: str,
    payload_names: set[str],
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for name in sorted(payload_names):
        data = member_data[name]
        writer.writerow([name, f"sha256={b64_sha256(data)}", str(len(data))])
    writer.writerow([record_name, "", ""])
    return buffer.getvalue().encode("utf-8")


def write_json(path: Path, payload: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing artifact without --force: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def make_overlay_wheel(args: argparse.Namespace) -> dict[str, Path]:
    base_wheel = require_regular_file(Path(args.base_wheel), "baseline wheel")
    if base_wheel.name != args.expected_baseline_wheel_name:
        raise ValueError(
            "baseline wheel filename mismatch: "
            f"expected {args.expected_baseline_wheel_name}, got {base_wheel.name}")
    base_hash = sha256_file(base_wheel)
    if base_hash != args.expected_baseline_sha256:
        raise ValueError(
            "baseline wheel SHA-256 mismatch: "
            f"expected {args.expected_baseline_sha256}, got {base_hash}")
    checkout = require_directory(Path(args.derived_checkout), "derived checkout")
    aot_library = require_regular_file(Path(args.aot_library), "AOT stable extension")

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
        raise ValueError(
            "output wheel filename must equal the baseline filename; the "
            "provenance records this as a same-version experimental overlay")

    provenance_path = artifact_dir / "c2-native-overlay-provenance.json"
    manifest_path = artifact_dir / "c2-native-overlay-manifest.json"
    for output in (output_wheel, provenance_path, manifest_path):
        if output.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite existing artifact without --force: {output}")

    git_info = checkout_provenance(checkout, args.expected_git_commit)
    overlay_bytes, overlay_provenance = load_overlay_sources(checkout, aot_library)

    with zipfile.ZipFile(base_wheel, "r") as base_archive:
        if base_archive.testzip() is not None:
            raise ValueError("baseline wheel CRC verification failed")
        base_infos = base_archive.infolist()
        assert_safe_archive_names(base_infos)
        record_name = find_record_member(base_infos)
        base_record_verification = full_record_verification(base_archive, base_infos, record_name)

        base_names = [info.filename for info in base_infos]
        base_name_set = set(base_names)
        member_data: dict[str, bytes] = {
            info.filename: base_archive.read(info.filename)
            for info in base_infos
        }
        original_data = {
            info.filename: member_data[info.filename]
            for info in base_infos if not info.is_dir()
        }
        original_infos = {info.filename: info for info in base_infos}
        member_data.update(overlay_bytes)
        payload_names = {
            info.filename for info in base_infos
            if not info.is_dir() and info.filename != record_name
        }
        payload_names.update(overlay_bytes)
        member_data[record_name] = build_record(
            member_data, record_name, payload_names)

        absent_overlay_names = [name for name in overlay_bytes if name not in base_name_set]
        ordered_names = [name for name in base_names if name != record_name]
        ordered_names.extend(sorted(absent_overlay_names))
        ordered_names.append(record_name)

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{output_wheel.stem}.", suffix=".tmp", dir=artifact_dir)
        os.close(fd)
        temporary_wheel = Path(temp_name)
        try:
            with zipfile.ZipFile(temporary_wheel, "w", allowZip64=True) as result:
                for name in ordered_names:
                    data = member_data[name]
                    if name in overlay_bytes or name == record_name:
                        info = new_overlay_zipinfo(name)
                    else:
                        info = clone_zipinfo(original_infos[name])
                    result.writestr(info, data)
            os.replace(temporary_wheel, output_wheel)
        except BaseException:
            temporary_wheel.unlink(missing_ok=True)
            raise

    # Re-open the result rather than trusting the write path.  This is also the
    # final full RECORD verification used in the auditable provenance artifact.
    with zipfile.ZipFile(output_wheel, "r") as derived_archive:
        if derived_archive.testzip() is not None:
            raise ValueError("derived wheel CRC verification failed")
        derived_infos = derived_archive.infolist()
        assert_safe_archive_names(derived_infos)
        derived_record_name = find_record_member(derived_infos)
        if derived_record_name != record_name:
            raise ValueError("derived wheel changed the dist-info/RECORD path")
        derived_record_verification = full_record_verification(
            derived_archive, derived_infos, record_name)
        derived_names = [info.filename for info in derived_infos]
        expected_names = base_name_set | set(absent_overlay_names)
        if set(derived_names) != expected_names:
            raise ValueError("derived wheel has unexpected member additions or removals")
        if len(derived_names) != len(set(derived_names)):
            raise ValueError("derived wheel has duplicate members")

        changed_unapproved: list[str] = []
        unchanged_member_count = 0
        unchanged_total_size = 0
        unchanged_digest = hashlib.sha256()
        allowed_changed = set(overlay_bytes) | {record_name}
        for name, before in original_data.items():
            after = derived_archive.read(name)
            if name not in allowed_changed and after != before:
                changed_unapproved.append(name)
            if name not in allowed_changed and after == before:
                unchanged_member_count += 1
                unchanged_total_size += len(before)
                unchanged_digest.update(name.encode("utf-8"))
                unchanged_digest.update(b"\0")
                unchanged_digest.update(hashlib.sha256(before).digest())
                unchanged_digest.update(b"\n")
        if changed_unapproved:
            raise ValueError(f"unchanged wheel members were modified: {changed_unapproved[:3]}")

        overlay_by_name = {item["archive_member"]: item for item in overlay_provenance}
        for name, source_data in overlay_bytes.items():
            zip_data = derived_archive.read(name)
            if zip_data != source_data:
                raise ValueError(f"overlay member content mismatch after packing: {name}")
            overlay_by_name[name].update({
                "action": "replace" if name in base_name_set else "add",
                "zip_sha256": sha256_bytes(zip_data),
                "zip_size": len(zip_data),
                "source_matches_zip": True,
            })

    derived_hash = sha256_file(output_wheel)
    provenance: dict[str, Any] = {
        "schema": "c2-native-overlay-provenance-v1",
        "created_utc": utc_now(),
        "experimental_overlay_same_distribution_version": True,
        "baseline_wheel": {
            "path": str(base_wheel),
            "filename": base_wheel.name,
            "sha256": base_hash,
            "size": base_wheel.stat().st_size,
        },
        "derived_wheel": {
            "path": str(output_wheel),
            "filename": output_wheel.name,
            "sha256": derived_hash,
            "size": output_wheel.stat().st_size,
            "same_filename_as_baseline": output_wheel.name == base_wheel.name,
        },
        "derived_checkout_base_git": git_info,
        "overlay_members": [overlay_by_name[name] for name, _ in OVERLAY_SOURCES],
        "base_record_full_verification": base_record_verification,
        "derived_record_full_verification": derived_record_verification,
        "unchanged_members": {
            "definition": "baseline non-directory members excluding declared overlays and RECORD",
            "count": unchanged_member_count,
            "total_uncompressed_size": unchanged_total_size,
            "name_and_content_sha256": unchanged_digest.hexdigest(),
            "unapproved_changed_members": changed_unapproved,
        },
        "member_set": {
            "baseline_count": len(base_names),
            "derived_count": len(derived_names),
            "new_overlay_members": sorted(absent_overlay_names),
            "removed_members": sorted(base_name_set - set(derived_names)),
        },
    }
    manifest: dict[str, Any] = {
        "schema": "c2-native-overlay-manifest-v1",
        "experimental_overlay_same_distribution_version": True,
        "baseline_wheel_sha256": base_hash,
        "derived_wheel_sha256": derived_hash,
        "derived_wheel_size": output_wheel.stat().st_size,
        "derived_checkout_commit": git_info["commit"],
        "derived_checkout_tracked_diff_sha256": git_info["tracked_diff_sha256"],
        "record_member": record_name,
        "overlay_archive_members": [name for name, _ in OVERLAY_SOURCES],
        "provenance_file": provenance_path.name,
        "provenance_sha256": sha256_bytes(
            (json.dumps(
                provenance,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ) + "\n").encode("utf-8")),
    }
    write_json(provenance_path, provenance, args.force)
    write_json(manifest_path, manifest, args.force)
    return {
        "wheel": output_wheel,
        "provenance": provenance_path,
        "manifest": manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-wheel",
        default=("/home/lcpu/85117379/vllm-d4-wheel/" + EXPECTED_BASELINE_WHEEL),
        help="exact d4 baseline wheel to overlay",
    )
    parser.add_argument(
        "--derived-checkout",
        default="/home/lcpu/85117379/vllm-d4-native-c2-20260829",
        help="derived native-C2 checkout at the exact d4 commit",
    )
    parser.add_argument(
        "--aot-library",
        default=("/home/lcpu/85117379/vllm-d4-native-c2-build-20260829/"
                 "_C_stable_libtorch.abi3.so"),
        help="AOT-built stable libtorch extension to place in the wheel",
    )
    parser.add_argument(
        "--artifact-dir",
        default="/home/lcpu/85117379/c2-native-wheel-artifacts-20260829",
        help="independent directory for the derived wheel and audit artifacts",
    )
    parser.add_argument(
        "--output-wheel",
        default=None,
        help="optional output path; it must be contained by --artifact-dir",
    )
    parser.add_argument(
        "--expected-baseline-wheel-name",
        default=EXPECTED_BASELINE_WHEEL,
        help="filename guard for the exact baseline wheel",
    )
    parser.add_argument(
        "--expected-git-commit",
        default=EXPECTED_D4_COMMIT,
        help="commit guard for the derived checkout",
    )
    parser.add_argument(
        "--expected-baseline-sha256",
        default=EXPECTED_BASELINE_SHA256,
        help="SHA-256 guard for the exact baseline wheel",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow replacing previously generated artifacts in --artifact-dir",
    )
    return parser.parse_args()


def main() -> int:
    artifacts = make_overlay_wheel(parse_args())
    print(json.dumps({
        "schema": "c2-native-overlay-wheel-result-v1",
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
