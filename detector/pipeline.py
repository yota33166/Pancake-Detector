"""Image processing pipeline pieces for pancake detection."""

from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import cv2
import numpy as np


class FrameProcessor:
    """フレームから輪郭検出用のマスクを生成するための処理を行う。
    
    Args:
        history_size: フレーム履歴の最大枚数。1以上の値を指定。
    """

    def __init__(self, history_size: int) -> None:
        self.history: Deque[np.ndarray] = deque(maxlen=max(1, history_size))

    def detect_contours(
        self,
        frame: np.ndarray,
        lower_color: np.ndarray,
        upper_color: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, ...]]:
        """指定された色範囲に基づいて輪郭を検出する。
        
        Args:
            frame: 入力フレーム（BGR形式）。
            lower_color: HSV色空間での下限値。
            upper_color: HSV色空間での上限値。
        """
        processed = self._preprocess(frame)
        hsv_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        mask = self._create_mask(hsv_frame, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return mask, hsv_frame, tuple(contours)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """ガウシアンぼかしとフレーム平均化による前処理を行う。"""
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        self.history.append(blurred)
        if len(self.history) == 1:
            return blurred
        return np.mean(self.history, axis=0).astype(np.uint8)

    @staticmethod
    def _create_mask(hsv_frame: np.ndarray, lower_color: np.ndarray, upper_color: np.ndarray) -> np.ndarray:
        """指定された色範囲に基づいてマスクを生成する。モルフォロジー処理を適用。"""
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        return mask

    @staticmethod
    def calculate_average_hsv(mask: np.ndarray, hsv_frame: np.ndarray):
        """マスク領域内の平均HSV値を計算する。"""
        masked_hsv = hsv_frame[mask > 0]
        if masked_hsv.size == 0:
            return None
        return tuple(np.mean(masked_hsv, axis=0))
