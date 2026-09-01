#!/usr/bin/env python3
"""ProxyCommand helper that waits for a real SSH banner before relaying.

The B300 login TCP port intermittently accepts connections without emitting a
banner.  A normal ProxyJump hands such a half-open connection to the client and
then exhausts ConnectTimeout.  This helper runs on the jump host, retries those
half-open sockets, and relays only the first socket that emitted an SSH banner.
It does not authenticate, inspect SSH payloads, or write remote state.
"""

from __future__ import annotations

import os
import select
import socket
import sys
import time


def connect_with_banner(host: str, port: int, deadline_seconds: float = 540.0) -> tuple[socket.socket, bytes]:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        sock: socket.socket | None = None
        try:
            sock = socket.create_connection((host, port), timeout=10.0)
            sock.settimeout(20.0)
            banner = bytearray()
            while b"\n" not in banner and len(banner) < 1024:
                chunk = sock.recv(1024 - len(banner))
                if not chunk:
                    raise ConnectionError("target closed before SSH banner")
                banner.extend(chunk)
            if bytes(banner).startswith(b"SSH-"):
                sock.settimeout(None)
                return sock, bytes(banner)
        except (OSError, TimeoutError):
            pass
        finally:
            if sock is not None and sock.fileno() >= 0:
                if 'banner' not in locals() or not bytes(banner).startswith(b"SSH-"):
                    sock.close()
        time.sleep(2.0)
    raise TimeoutError("no SSH banner before proxy deadline")


def relay(sock: socket.socket, initial_from_target: bytes) -> None:
    os.write(sys.stdout.fileno(), initial_from_target)
    stdin_open = True
    while True:
        readers: list[object] = [sock]
        if stdin_open:
            readers.append(sys.stdin.fileno())
        ready, _, _ = select.select(readers, [], [], 60.0)
        if not ready:
            continue
        if sock in ready:
            data = sock.recv(65536)
            if not data:
                return
            os.write(sys.stdout.fileno(), data)
        if stdin_open and sys.stdin.fileno() in ready:
            data = os.read(sys.stdin.fileno(), 65536)
            if not data:
                stdin_open = False
                try:
                    sock.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
            else:
                sock.sendall(data)


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} HOST PORT", file=sys.stderr)
        return 2
    sock, banner = connect_with_banner(sys.argv[1], int(sys.argv[2]))
    try:
        relay(sock, banner)
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
