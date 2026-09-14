"""
Thin wrappers around ``lib.network.UDP.Receiver`` / ``Sender`` for morai2 vehicle packets.
"""
from __future__ import annotations

try:
    from lib.network.UDP import Receiver, Sender
except ImportError:
    from network.UDP import Receiver, Sender

from .packets import ManualControlPacket, TransformFixedPacket, VehicleInfoPacket


class VehicleInfoReceiver:
    """
    Binds UDP and receives fixed-size vehicle status using ``UDP.Receiver`` + ``VehicleInfoPacket``.
    Poll ``get_data()`` (same pattern as ``Receiver.get_data``): returns the live ctypes struct
    that is updated in the background.
    """

    def __init__(self, ip: str, port: int):
        self._packet = VehicleInfoPacket()
        self._receiver = Receiver(ip, int(port), self._packet)

    @property
    def receiver(self) -> Receiver:
        return self._receiver

    @property
    def packet(self) -> VehicleInfoPacket:
        return self._packet

    def get_data(self) -> VehicleInfoPacket:
        return self._receiver.get_data()

    def get_dict(self) -> dict:
        """Snapshot of current packet as dict (reads ``get_data()``)."""
        return self.get_data().as_dict()


class ManualControlSender:
    """
    ``Manual Control.py`` — throttle / brake / steering as three doubles, ``UDP.Sender.send``.
    """

    def __init__(self, ip: str, port: int):
        self._sender = Sender(ip, int(port))
        self._cmd = ManualControlPacket()

    @property
    def sender(self) -> Sender:
        return self._sender

    @property
    def command(self) -> ManualControlPacket:
        return self._cmd

    def send(
        self,
        throttle: float,
        brake: float,
        steering: float,
        *,
        use_command_fields: bool = False,
    ) -> None:
        """
        If ``use_command_fields`` is True, send current ``self.command`` field values as-is.
        Otherwise set fields from arguments then send.
        """
        if not use_command_fields:
            self._cmd.throttle = float(throttle)
            self._cmd.brake = float(brake)
            self._cmd.steering = float(steering)
        self._sender.send(self._cmd)


class TransformFixedSender:
    """
    ``Transform Control_Fixed.py`` — seven floats; optional RPY remap like the Qt sample.
    """

    def __init__(self, ip: str, port: int):
        self._sender = Sender(ip, int(port))
        self._pkt = TransformFixedPacket()

    @property
    def sender(self) -> Sender:
        return self._sender

    @property
    def packet(self) -> TransformFixedPacket:
        return self._pkt

    def send_packet(self) -> None:
        """Send current ``self.packet`` contents."""
        self._sender.send(self._pkt)

    def send_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        steer_angle: float,
        *,
        remap_rpy_like_sample_gui: bool = True,
    ) -> None:
        """
        Pack and send. If ``remap_rpy_like_sample_gui`` is True, apply the same roll/pitch/yaw
        permutation as ``Transform Control_Fixed.py`` (UI Roll→wire yaw, Pitch→wire roll, etc.).
        If False, write roll/pitch/yaw to the wire fields directly.
        """
        if remap_rpy_like_sample_gui:
            self._pkt.set_from_gui_rpy(x, y, z, roll, pitch, yaw, steer_angle)
        else:
            self._pkt.x = float(x)
            self._pkt.y = float(y)
            self._pkt.z = float(z)
            self._pkt.roll = float(roll)
            self._pkt.pitch = float(pitch)
            self._pkt.yaw = float(yaw)
            self._pkt.steer_angle = float(steer_angle)
        self._sender.send(self._pkt)
