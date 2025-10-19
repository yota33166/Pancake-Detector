"""UI helpers for the pancake detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np


@dataclass
class MaskThresholds:
    lower: np.ndarray
    upper: np.ndarray
    min_area: int


class DetectorUI:
    """OpenCVウィンドウとトラックバーの管理を行い、インタラクティブな調整を可能にする。
    
    Args:
        window_name: メインウィンドウの名前。
        frame_size: フレームのサイズ (幅, 高さ)。
        initial_lower: 初期のHSV下限値。
        initial_upper: 初期のHSV上限値。
        initial_min_area: 初期の最小面積閾値。
        initial_max_area: 初期の最大面積閾値。Noneの場合、フレーム全体の面積が使用される。
    """

    def __init__(
        self,
        window_name: str,
        frame_size: Tuple[int, int],
        initial_lower: np.ndarray,
        initial_upper: np.ndarray,
        initial_min_area: int,
        initial_max_area: Optional[int] = None,
    ) -> None:
        self.window_name = window_name
        self.control_window = "Controls"
        if initial_max_area is not None:
            self.max_area = max(1, initial_max_area)
        else:
            self.max_area = max(1, frame_size[0] * frame_size[1])

        self._lower = initial_lower.astype(np.uint8)
        self._upper = initial_upper.astype(np.uint8)
        self._min_area = int(initial_min_area)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.control_window, 420, 240)

        self._create_trackbars()

    def _create_trackbars(self) -> None:
        h_lower, s_lower, v_lower = map(int, self._lower)
        h_upper, s_upper, v_upper = map(int, self._upper)

        cv2.createTrackbar("H_lower", self.control_window, h_lower, 179, lambda _: None)
        cv2.createTrackbar("H_upper", self.control_window, h_upper, 179, lambda _: None)
        cv2.createTrackbar("S_lower", self.control_window, s_lower, 255, lambda _: None)
        cv2.createTrackbar("S_upper", self.control_window, s_upper, 255, lambda _: None)
        cv2.createTrackbar("V_lower", self.control_window, v_lower, 255, lambda _: None)
        cv2.createTrackbar("V_upper", self.control_window, v_upper, 255, lambda _: None)
        cv2.createTrackbar("MinArea", self.control_window, self._min_area, self.max_area, lambda _: None)

    def read_thresholds(self) -> MaskThresholds:
        h_lower = cv2.getTrackbarPos("H_lower", self.control_window)
        h_upper = cv2.getTrackbarPos("H_upper", self.control_window)
        s_lower = cv2.getTrackbarPos("S_lower", self.control_window)
        s_upper = cv2.getTrackbarPos("S_upper", self.control_window)
        v_lower = cv2.getTrackbarPos("V_lower", self.control_window)
        v_upper = cv2.getTrackbarPos("V_upper", self.control_window)
        min_area = cv2.getTrackbarPos("MinArea", self.control_window)

        h_lower, h_upper = sorted((h_lower, h_upper))
        s_lower, s_upper = sorted((s_lower, s_upper))
        v_lower, v_upper = sorted((v_lower, v_upper))

        self._lower = np.array([h_lower, s_lower, v_lower], dtype=np.uint8)
        self._upper = np.array([h_upper, s_upper, v_upper], dtype=np.uint8)
        self._min_area = max(0, int(min_area))

        return MaskThresholds(lower=self._lower, upper=self._upper, min_area=self._min_area)
    