#!/usr/bin/env bash
# setup.py asks Git to initialize CUTLASS even when the pinned archive is already provisioned.
set -euo pipefail
if [[ "$#" -eq 4 && "$1" == submodule && "$2" == update && "$3" == --init && "$4" == cutlass ]]; then
    echo "CUTLASS already provisioned from pinned local archive; skipping network submodule update"
    exit 0
fi
exec /usr/bin/git "$@"
