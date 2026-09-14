"""
MORAI Sim UDP helpers built on ``lib.network.UDP`` (ctypes ``Receiver`` / ``Sender``).

Layouts match ``lib/morai2`` samples:

- **Vehicle Info** — receive only, ``VehicleInfoReceiver`` + ``VehicleInfoPacket``
- **Manual Control** — send, ``ManualControlSender`` + ``ManualControlPacket``
- **Transform Control_Fixed** — send, ``TransformFixedSender`` + ``TransformFixedPacket``

**sys.path**

- Parent of ``lib/`` (``morai_udp_networkmodule``)::

    from lib.morai2_lib import VehicleInfoReceiver, ManualControlSender

- Or ``.../morai_udp_networkmodule/lib``::

    from morai2_lib import VehicleInfoReceiver, ManualControlSender

Receive: poll ``get_data()`` like ``UDP.Receiver`` (returns the ctypes packet struct).
"""

from .clients import (
    ManualControlSender,
    TransformFixedSender,
    VehicleInfoReceiver,
)
from .packets import (
    ManualControlPacket,
    TransformFixedPacket,
    VehicleInfoPacket,
)

__all__ = [
    "VehicleInfoPacket",
    "VehicleInfoReceiver",
    "ManualControlPacket",
    "ManualControlSender",
    "TransformFixedPacket",
    "TransformFixedSender",
]
