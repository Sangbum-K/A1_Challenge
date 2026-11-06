"""Hikrobot hardware-ROI launch for the 1280x720 A2 path.

Indoor monitor runs and outdoor road runs need different photometry, so every
value that has to change in the field is a launch argument.  The defaults
reproduce the verified 2026-09-14 indoor configuration byte for byte
(exposure_auto=Off, exposure_us=15000, gain 0, white balance off), so an
existing command keeps behaving exactly as before.

Outdoor daylight needs a completely different exposure regime; run_ui.sh passes
exposure_us from A1_CAMERA_EXPOSURE_US (default 3000).

The ROI is a crop of the 1920x1200 sensor.  It matches the senior input pixel
dimensions but not necessarily its horizontal FOV or calibration.  offset_y
sets where the crop sits vertically on the sensor and therefore decides, once
the camera is bolted to a vehicle, whether the road surface lands inside the
model band y=235:485.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'

# name, default, description
ARGUMENTS = [
    ("guid", "169.254.222.118", "Hikrobot serial number or IPv4 address"),
    ("frame_id", "hikrobot_camera", "ROS frame_id stamped on every image"),
    ("width", "1280", "Transported ROI width in pixels"),
    ("height", "720", "Transported ROI height in pixels"),
    ("offset_x", "320", "Sensor ROI horizontal offset in pixels"),
    ("offset_y", "240", "Sensor ROI vertical offset; aims the model band"),
    ("fps", "21.0", "AcquisitionFrameRate in Hz"),
    ("packet_delay", "2000", "GevSCPD inter-packet delay in camera ticks"),
    ("packet_size", "1500", "GevSCPSPacketSize; 1500 unless jumbo frames are enabled"),
    ("exposure_auto", "Off", "Off | Once | Continuous"),
    ("exposure_us", "15000.0", "Manual exposure in microseconds; used when exposure_auto=Off"),
    ("exposure_min_us", "0.0", "Auto-exposure lower limit in us; 0 leaves the camera default"),
    ("exposure_max_us", "0.0", "Auto-exposure upper limit in us; 0 leaves the camera default"),
    ("gain_auto", "Off", "Off | Once | Continuous"),
    ("gain_db", "0.0", "Manual analog gain in dB; used when gain_auto=Off"),
    ("gain_max_db", "0.0", "Auto-gain upper limit in dB; 0 leaves the camera default"),
    ("balance_white_auto", "Off", "Off | Once | Continuous"),
]


def _launch_setup(context, *args, **kwargs):
    def text(name):
        return LaunchConfiguration(name).perform(context).strip()

    def number(name, cast):
        return cast(float(text(name)))

    exposure_auto = text("exposure_auto")
    gain_auto = text("gain_auto")

    image_format = {
        "BEGIN": {"OffsetX": 0, "OffsetY": 0},
        "PixelFormat": "BayerRG8",
        "BinningHorizontal": 1,
        "BinningVertical": 1,
        "Width": number("width", int),
        "Height": number("height", int),
        "OffsetX": number("offset_x", int),
        "OffsetY": number("offset_y", int),
    }

    acquisition = {
        "ExposureMode": "Timed",
        "ExposureAuto": exposure_auto,
        "AcquisitionFrameRateEnable": True,
        "AcquisitionFrameRate": number("fps", float),
    }
    if exposure_auto == "Off":
        acquisition["ExposureTime"] = number("exposure_us", float)
    else:
        # Auto exposure must stay under the acquisition period or the camera
        # silently drops below the requested FPS.  These GenICam names are
        # Hikrobot-specific; confirm them with `./run.sh features` if the
        # driver logs an unknown-feature warning.
        lower = number("exposure_min_us", float)
        upper = number("exposure_max_us", float)
        if lower > 0.0:
            acquisition["AutoExposureTimeLowerLimit"] = lower
        if upper > 0.0:
            acquisition["AutoExposureTimeUpperLimit"] = upper

    analog = {
        "BEGIN": {"Gain": 0.0},
        "GainAuto": gain_auto,
        "BalanceWhiteAuto": text("balance_white_auto"),
    }
    if gain_auto == "Off":
        analog["BEGIN"] = {"Gain": number("gain_db", float)}
    else:
        upper_gain = number("gain_max_db", float)
        if upper_gain > 0.0:
            analog["AutoGainUpperLimit"] = upper_gain

    return [Node(
        package="camera_aravis2",
        executable="camera_driver_gv",
        name="camera_driver_gv",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "guid": text("guid"),
            "frame_id": text("frame_id"),
            "stream_names": ["vis"],
            "camera_info_urls": [str(
                CONFIG_DIR / "camera_info_1280x720_uncalibrated.yaml")],
            "dynamic_parameters_yaml_url": str(
                CONFIG_DIR / "camera_dynamic_parameters.yaml"),
            "verbose": False,
            "TransportLayerControl": {"GevSCPSPacketSize": number("packet_size", int), "GevSCPD": number("packet_delay", int)},
            "ImageFormatControl": image_format,
            "AcquisitionControl": acquisition,
            "AnalogControl": analog,
        }],
    )]


def generate_launch_description():
    declarations = [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in ARGUMENTS
    ]
    return LaunchDescription(declarations + [OpaqueFunction(function=_launch_setup)])
