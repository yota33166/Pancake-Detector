"""Camera management utilities for the pancake detector."""

from __future__ import annotations

import glob
from typing import Optional, Tuple

import cv2
import numpy as np


class CameraManager:
    """カメラの初期化と設定を管理するラッパークラス。
    
    Args:
        camera_index: 使用するカメラのインデックスまたはデバイスパス。
        frame_size: フレームのサイズ (幅, 高さ)。
        fps: フレームレート。
        calibration_file: キャリブレーションデータのファイルパス。
        autofocus: オートフォーカスを有効にするかどうか。
        focus_value: 手動フォーカス値（0〜255）。autofocusがFalseの場合に使用。
    """

    def __init__(
        self,
        camera_index: int,
        frame_size: Tuple[int, int],
        fps: int,
        calibration_file: str,
        autofocus: bool = False,
        focus_value: Optional[int] = 100,
    ) -> None:
        self.cap = self._init_camera(camera_index)
        if self.cap is None:
            raise RuntimeError("Failed to initialise camera")

        width, height = frame_size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.configure_focus(autofocus=autofocus, focus_value=focus_value)

        self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = self._load_calibration(calibration_file)

    def _init_camera(self, index: int) -> Optional[cv2.VideoCapture]:
        """カメラデバイスを初期化する。"""
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap

        print(f"カメラ{index}が見つかりませんでした。")
        for device in glob.glob("/dev/video*"):
            print(f"試行中のデバイス: {device}")
            fallback = cv2.VideoCapture(device)
            if fallback.isOpened():
                print(f"カメラが見つかりました: {device}")
                return fallback
        print("カメラが見つかりませんでした。")
        return None

    def configure_focus(self, autofocus: bool, focus_value: Optional[int]) -> None:
        """フォーカス機能の設定を行う。

        Args:
            autofocus (bool): オートフォーカスを有効にするかどうか。
            focus_value (Optional[int]): 手動フォーカス値（0〜255）。autofocusがFalseの場合に使用。
        """
        try:
            if autofocus:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            else:
                success = self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                if not success:
                    print("警告: CAP_PROP_AUTOFOCUSでオートフォーカスを無効化できませんでした。")
                else:
                    print("オートフォーカスを無効にしました。")

            if focus_value is not None:
                if not 0 <= focus_value <= 255:
                    print("警告: focus_value は 0〜255 の範囲で指定してください。")
                else:
                    success = self.cap.set(cv2.CAP_PROP_FOCUS, float(focus_value))
                    if not success:
                        print("警告: CAP_PROP_FOCUS で手動フォーカス値を設定できませんでした。")
                    else:
                        print(f"フォーカス値を {focus_value} に設定しました。")

            exposure_success = self.cap.set(cv2.CAP_PROP_EXPOSURE, 0.25)
            if not exposure_success:
                print("警告: CAP_PROP_EXPOSURE で露出を設定できませんでした。")
            else:
                print("露出を 0.25 に設定しました。")
        except Exception as exc:  # pragma: no cover - hardware specific
            print(f"フォーカス設定中にエラーが発生しました: {exc}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()

    def _load_calibration(self, file_name: str):
        try:
            if glob.glob(file_name):
                data = np.load(file_name)
                return data.get("cameraMatrix"), data.get("distCoeffs"), data.get("rvecs"), data.get("tvecs")
            print(f"キャリブレーションファイルが見つかりません: {file_name}")
            return None, None, None, None
        except Exception as exc:
            print(f"キャリブレーションデータの読み込みエラー: {exc}")
            return None, None, None, None

    def convert_to_real_coordinates(self, pixel_coords: Tuple[float, float]) -> Optional[np.ndarray]:
        if not all(
            attr is not None for attr in (self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs)
        ):
            return None

        undist = cv2.undistortPoints(
            np.array([[pixel_coords]], dtype=np.float32),
            self.camera_matrix,
            self.dist_coeffs,
        )
        return undist.reshape(-1)
    