"""Hikrobot hardware-ROI launch for the 1280x720 A2 throughput run.

The ROI is a centered crop of the 1920x1200 sensor. It matches the senior
input pixel dimensions but not necessarily its horizontal FOV or calibration.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    guid = LaunchConfiguration("guid")
    exposure_us = LaunchConfiguration("exposure_us")
    project_root = Path(__file__).resolve().parents[1]
    return LaunchDescription([
        DeclareLaunchArgument(
            "guid",
            default_value="192.168.1.51",
            description="Hikrobot serial number or IPv4 address",
        ),
        DeclareLaunchArgument(
            "exposure_us",
            default_value="15000.0",
            description="Manual camera exposure time in microseconds",
        ),
        Node(
            package="camera_aravis2",
            executable="camera_driver_gv",
            name="camera_driver_gv",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "guid": guid,
                "frame_id": "hikrobot_camera",
                "stream_names": ["vis"],
                "camera_info_urls": [str(
                    project_root / "configs" / "camera_info_1280x720_uncalibrated.yaml")],
                "dynamic_parameters_yaml_url": str(
                    project_root / "configs" / "camera_dynamic_parameters.yaml"),
                "verbose": False,
                "TransportLayerControl": {"GevSCPSPacketSize": 1500},
                "ImageFormatControl": {
                    "BEGIN": {"OffsetX": 0, "OffsetY": 0},
                    "PixelFormat": "BayerRG8",
                    "BinningHorizontal": 1,
                    "BinningVertical": 1,
                    "Width": 1280,
                    "Height": 720,
                    "OffsetX": 320,
                    "OffsetY": 240,
                },
                "AcquisitionControl": {
                    "ExposureMode": "Timed",
                    "ExposureAuto": "Off",
                    "ExposureTime": ParameterValue(exposure_us, value_type=float),
                    "AcquisitionFrameRateEnable": True,
                    "AcquisitionFrameRate": 21.0,
                },
                "AnalogControl": {
                    "BEGIN": {"Gain": 0.0},
                    "GainAuto": "Off",
                    "BalanceWhiteAuto": "Off",
                },
            }],
        ),
    ])
