"""
MORAI UDP binary layouts matching ``lib/morai2`` samples (ctypes for ``lib.network.UDP``).
"""
from __future__ import annotations

import ctypes

try:
    from lib.define.base import Base
except ImportError:
    from define.base import Base


class VehicleInfoPacket(Base):
    """
    ``Vehicle Info.py`` — format ``<qi24s`` + 18× ``f`` (108 bytes).
    """

    _fields_ = [
        ("seconds", ctypes.c_int64),
        ("nanos", ctypes.c_int32),
        ("vehicle_id", ctypes.c_char * 24),
        ("x_loc", ctypes.c_float),
        ("y_loc", ctypes.c_float),
        ("z_loc", ctypes.c_float),
        ("x_rot", ctypes.c_float),
        ("y_rot", ctypes.c_float),
        ("z_rot", ctypes.c_float),
        ("x_vel", ctypes.c_float),
        ("y_vel", ctypes.c_float),
        ("z_vel", ctypes.c_float),
        ("x_acc", ctypes.c_float),
        ("y_acc", ctypes.c_float),
        ("z_acc", ctypes.c_float),
        ("x_ang", ctypes.c_float),
        ("y_ang", ctypes.c_float),
        ("z_ang", ctypes.c_float),
        ("throttle", ctypes.c_float),
        ("brake", ctypes.c_float),
        ("steer_angle", ctypes.c_float),
    ]

    def parsing(self) -> None:
        """Hook for ``UDP.Receiver`` after ``memmove``; optional post-process."""

    def id_string(self) -> str:
        return self.vehicle_id.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")

    def as_dict(self) -> dict:
        """Same keys as the PyQt sample dict (for convenience)."""
        return {
            "seconds": int(self.seconds),
            "nanos": int(self.nanos),
            "id": self.id_string(),
            "x_loc": float(self.x_loc),
            "y_loc": float(self.y_loc),
            "z_loc": float(self.z_loc),
            "x_rot": float(self.x_rot),
            "y_rot": float(self.y_rot),
            "z_rot": float(self.z_rot),
            "x_vel": float(self.x_vel),
            "y_vel": float(self.y_vel),
            "z_vel": float(self.z_vel),
            "x_acc": float(self.x_acc),
            "y_acc": float(self.y_acc),
            "z_acc": float(self.z_acc),
            "x_ang": float(self.x_ang),
            "y_ang": float(self.y_ang),
            "z_ang": float(self.z_ang),
            "throttle": float(self.throttle),
            "brake": float(self.brake),
            "steer_angle": float(self.steer_angle),
        }


class ManualControlPacket(Base):
    """
    ``Manual Control.py`` — ``CONTROL_STRUCT_FORMAT = "<ddd"`` (throttle, brake, steering).
    """

    _fields_ = [
        ("throttle", ctypes.c_double),
        ("brake", ctypes.c_double),
        ("steering", ctypes.c_double),
    ]

    def parsing(self) -> None:
        pass


class TransformFixedPacket(Base):
    """
    ``Transform Control_Fixed.py`` — ``struct.pack('<7f', ...)``:
    x, y, z, roll, pitch, yaw, steer (after GUI axis remap).
    """

    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("roll", ctypes.c_float),
        ("pitch", ctypes.c_float),
        ("yaw", ctypes.c_float),
        ("steer_angle", ctypes.c_float),
    ]

    def parsing(self) -> None:
        pass

    def set_from_gui_rpy(
        self,
        x: float,
        y: float,
        z: float,
        roll_input: float,
        pitch_input: float,
        yaw_input: float,
        steer_angle: float,
    ) -> None:
        """
        Match sample spinbox order: Roll, Pitch, Yaw → wire roll/pitch/yaw remap.
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.roll = float(pitch_input)
        self.pitch = float(yaw_input)
        self.yaw = float(roll_input)
        self.steer_angle = float(steer_angle)
