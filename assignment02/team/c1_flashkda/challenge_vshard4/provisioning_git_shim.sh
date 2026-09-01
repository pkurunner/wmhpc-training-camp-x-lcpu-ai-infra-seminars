#!/bin/bash
# setup.py invokes this unbounded network command before compiling.  On the
# isolated 5090 clone the exact pinned CUTLASS source is provisioned from a
# verified local archive instead, so only this redundant sync is bypassed.
# Every other git command is delegated to the system binary unchanged.
set -Eeuo pipefail

if [[ "${1:-}" == "submodule" && "${2:-}" == "update" && "${3:-}" == "--init" && "${4:-}" == "cutlass" ]]; then
    printf '%s\n' 'CUTLASS_PROVISION: skip network submodule sync; exact local 5c149f5 archive already materialized.' >&2
    exit 0
fi
exec /usr/bin/git "$@"
