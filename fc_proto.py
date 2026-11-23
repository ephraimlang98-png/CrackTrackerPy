"""Wire protocol between the phone sensor app and the PC lab app.

Every message, in both directions, is one frame:
    [u32 big-endian json_len][u32 big-endian bin_len][json (utf-8)][binary payload]
The JSON always has a "type". Binary payload (JPEG) only for "preview" and "capture".

phone -> pc : hello, metrics, accel, preview, capture, status, error, pong
pc -> phone : set, roi, capture, preview, ping
Times from the phone are SystemClock.elapsedRealtimeNanos (monotonic, ns).
"""
from __future__ import annotations

import json
import socket
import struct

PORT = 8765
HDR = struct.Struct(">II")
MAX_JSON = 1 << 20
MAX_BIN = 64 << 20


def pack(msg: dict, payload: bytes = b"") -> bytes:
    j = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    return HDR.pack(len(j), len(payload)) + j + payload


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def recv(sock: socket.socket) -> tuple[dict, bytes]:
    jl, bl = HDR.unpack(_recv_exact(sock, HDR.size))
    if jl > MAX_JSON or bl > MAX_BIN:
        raise ConnectionError(f"bad frame header ({jl}, {bl})")
    msg = json.loads(_recv_exact(sock, jl).decode("utf-8"))
    payload = _recv_exact(sock, bl) if bl else b""
    return msg, payload
