"""Detector component package."""

from .camera import CameraManager
from .pipeline import FrameProcessor
from .trigger import Detection, TriggerController
from .ui import DetectorUI

__all__ = [
    "CameraManager",
    "FrameProcessor",
    "Detection",
    "TriggerController",
    "DetectorUI",
]
