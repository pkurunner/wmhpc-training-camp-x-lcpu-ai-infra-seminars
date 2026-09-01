#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -eq 4 && "$1" == submodule && "$2" == update && "$3" == --init && "$4" == cutlass ]]; then
    echo "CUTLASS already provisioned from pinned local source; skipping network submodule update"
    exit 0
fi
exec /usr/bin/git "$@"
