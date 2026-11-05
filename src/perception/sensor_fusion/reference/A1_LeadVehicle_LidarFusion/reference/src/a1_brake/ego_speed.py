"""Actual ego speed source: a constant (default 0) or the vehicle CAN bus.

The CAN path reads ``USER_right_wheel_info`` (0x201) from the A1 baseline DBC
``/home/ailab/git/A1-BSW/DBC/A1_dbc_fixed.dbc``:

    right_speed : 0|12@1+  (0.1,0) km/h
    left_speed  : 24|12@1+ (0.1,0) km/h

It uses the Linux stdlib SocketCAN socket, so the project .venv needs no
python-can.  This path has NOT been verified on the car; treat it as opt-in.
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time

WHEEL_INFO_ID = 0x201
_CAN_FRAME = struct.Struct("=IB3x8s")


def decode_wheel_info(data: bytes) -> tuple[float, float]:
    """Return (right, left) wheel speed in km/h from a 0x201 payload."""
    word = int.from_bytes(bytes(data[:8]).ljust(8, b"\0"), "little")
    right = (word & 0xFFF) * 0.1
    left = ((word >> 24) & 0xFFF) * 0.1
    return right, left


class ConstantEgoSpeed:
    def __init__(self, speed_mps: float = 0.0):
        if not (math.isfinite(speed_mps) and speed_mps >= 0.0):
            raise ValueError("ego speed must be finite and nonnegative")
        self.speed_mps = speed_mps
        self.source = f"constant {speed_mps:.2f} m/s"

    def start(self):
        return self

    def read(self) -> tuple[float, float | None]:
        """Return (speed m/s, age s or None)."""
        return self.speed_mps, None

    def stop(self):
        pass


class CanEgoSpeed:
    """Background SocketCAN reader; mean of the two wheel speeds."""

    def __init__(self, iface: str = "can0", can_id: int = WHEEL_INFO_ID,
                 stale_s: float = 0.5):
        self.iface = iface
        self.can_id = can_id
        self.stale_s = stale_s
        self.source = f"CAN {iface} 0x{can_id:03X} (unverified)"
        self._lock = threading.Lock()
        self._speed = math.nan
        self._stamp = None
        self._running = False
        self._thread = None
        self._sock = None
        self.error = None

    def start(self):
        sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                        struct.pack("=II", self.can_id, socket.CAN_SFF_MASK))
        sock.settimeout(0.2)
        sock.bind((self.iface,))
        self._sock = sock
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="can-ego-speed", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            try:
                frame = self._sock.recv(_CAN_FRAME.size)
            except socket.timeout:
                continue
            except OSError as exc:
                self.error = str(exc)
                break
            can_id, dlc, data = _CAN_FRAME.unpack(frame)
            if (can_id & socket.CAN_SFF_MASK) != self.can_id or dlc < 6:
                continue
            right, left = decode_wheel_info(data)
            with self._lock:
                self._speed = 0.5 * (right + left) / 3.6
                self._stamp = time.monotonic()

    def read(self) -> tuple[float, float | None]:
        """Return (speed m/s, age s).  NaN when no fresh frame."""
        with self._lock:
            speed, stamp = self._speed, self._stamp
        if stamp is None:
            return math.nan, None
        age = time.monotonic() - stamp
        return (speed if age <= self.stale_s else math.nan), age

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close()
