"""Live camera parameter tweaking from inside a display loop.

camera_aravis2 re-publishes the GenICam features listed in
configs/camera_dynamic_parameters.yaml as ROS parameters, so exposure can be
changed while the stream is running.  That matters outdoors: the correct
exposure depends on the sun, the lens aperture and where the camera is
pointed, and guessing it from a desk is hopeless.  Dial it while watching the
picture, then read the number off the screen and pass it to the next run.

The exact parameter name is not guessed.  It is discovered from the running
node, because it differs between camera_aravis2 versions (`ExposureTime` vs
`vis.ExposureTime`).  If nothing is found, every call is a no-op and the
caller keeps working with the launch-time value.
"""
from __future__ import annotations

import subprocess
import threading


class DynamicCameraParams:
    """Best-effort live setter for camera_aravis2 dynamic parameters."""

    def __init__(self, node_name: str = "/camera_driver_gv", timeout: float = 10.0):
        self.node_name = node_name
        self.available = {}
        self.last_error = None
        self._lock = threading.Lock()
        self._pending = 0
        try:
            listed = subprocess.run(
                ["ros2", "param", "list", node_name],
                capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.last_error = f"ros2 param list failed: {exc}"
            return
        if listed.returncode != 0:
            self.last_error = (listed.stderr or listed.stdout or "").strip()[:200]
            return
        for line in listed.stdout.splitlines():
            name = line.strip()
            if not name:
                continue
            for feature in ("ExposureTime", "Gain", "AcquisitionFrameRate"):
                if name == feature or name.endswith(f".{feature}"):
                    self.available.setdefault(feature, name)

    def supports(self, feature: str) -> bool:
        return feature in self.available

    @property
    def summary(self) -> str:
        if self.available:
            return "live: " + ", ".join(sorted(self.available))
        return f"no dynamic parameters ({self.last_error or 'none advertised'})"

    def get(self, feature: str, timeout: float = 5.0):
        """Read the parameter once, at startup.

        Verified on a real MV-CS023-10GC: a `set` takes effect on the sensor
        (the image visibly brightens) but `get` keeps returning the value the
        node was launched with.  So read once for the starting point and track
        subsequent changes locally -- do not poll this to display the current
        exposure, it will lie.
        """
        name = self.available.get(feature)
        if name is None:
            return None
        try:
            result = subprocess.run(
                ["ros2", "param", "get", self.node_name, name],
                capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        # "Double value is: 15000.0"
        token = result.stdout.strip().rsplit(" ", 1)[-1]
        try:
            return float(token)
        except ValueError:
            return None

    def set_async(self, feature: str, value: float, on_done=None):
        """Set in a worker thread so a 21 FPS render loop never blocks on it."""
        name = self.available.get(feature)
        if name is None:
            return False
        with self._lock:
            if self._pending >= 2:  # drop spam from a held-down key
                return False
            self._pending += 1

        def worker():
            ok = False
            try:
                result = subprocess.run(
                    ["ros2", "param", "set", self.node_name, name, f"{value:.1f}"],
                    capture_output=True, text=True, timeout=10.0)
                ok = result.returncode == 0 and "successful" in result.stdout.lower()
                if not ok:
                    self.last_error = (result.stderr or result.stdout or "").strip()[:200]
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.last_error = str(exc)[:200]
            finally:
                with self._lock:
                    self._pending -= 1
                if on_done is not None:
                    on_done(ok, value)

        threading.Thread(target=worker, name=f"set-{feature}", daemon=True).start()
        return True


def step_exposure(current: float, direction: int, low: float = 30.0, high: float = 40000.0):
    """Multiplicative steps: exposure is perceptually logarithmic."""
    factor = 1.25 if direction > 0 else 1 / 1.25
    return float(min(high, max(low, current * factor)))
