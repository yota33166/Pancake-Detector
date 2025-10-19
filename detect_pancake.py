import multiprocessing
import sys
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from detector.camera import CameraManager
from detector.pipeline import FrameProcessor
from detector.ui import DetectorUI
from serial_handler import SerialHandler


class PancakeDetector:
    LOWER_COLOR = np.array([10, 50, 50], dtype=np.uint8)
    UPPER_COLOR = np.array([40, 255, 255], dtype=np.uint8)
    MIN_CONTOUR_AREA = 1000
    MAX_CONTOUR_AREA = 10000
    HISTORY_SIZE = 1

    def __init__(
        self,
        camera_index=0,
        calibration_file="calibration.npz",
        frame_width=480,
        frame_height=320,
        fps=30,
        serial_port: Optional[str] = None,
        baudrate=115200,
        timeout=1,
        command_queue: Optional[multiprocessing.Queue] = None,
        trigger_region: Optional[Tuple[int, int, int, int]] = None,
        trigger_cooldown_s: float = 1.0, # 生地注ぎのクールダウン時間
        max_pour_time_s: float = 5.0, # 最大注ぎ時間
        # release_delay_s: float = 0.5, # 最低注ぎ継続時間
        trigger_area_threshold: Optional[float] = None,
        ):
        frame_size = (frame_width, frame_height)
        self.window_name = "PancakeDetector"
        self.frame_width = frame_width
        self.left_boundary = int(frame_width * 0.4)
        self.right_boundary = int(frame_width * 0.6)

        try:
            self.camera = CameraManager(
                camera_index=camera_index,
                frame_size=frame_size,
                fps=fps,
                calibration_file=calibration_file,
            )
        except RuntimeError as exc:
            print(f"カメラの初期化に失敗しました: {exc}")
            sys.exit(1)

        self.frame_processor = FrameProcessor(self.HISTORY_SIZE)
        self.ui = DetectorUI(
            window_name=self.window_name,
            frame_size=frame_size,
            initial_lower=self.LOWER_COLOR,
            initial_upper=self.UPPER_COLOR,
            initial_min_area=self.MIN_CONTOUR_AREA,
            initial_max_area=self.MAX_CONTOUR_AREA,
        )

        self.command_queue = command_queue
        self.trigger_region = trigger_region
        self.trigger_cooldown_s = max(0.0, trigger_cooldown_s)
        # self.release_delay_s = max(0.0, release_delay_s)
        self.max_pour_time_s = max_pour_time_s
        self.target_area = (
            float(trigger_area_threshold)
            if trigger_area_threshold is not None
            else None
        )

        now = time.time()
        self.active_sides: Dict[str, bool] = {"left": False, "right": False}
        self.last_seen_ts: Dict[str, float] = {"left": now, "right": now}
        self.last_trigger_ts: Dict[str, float] = {"left": 0.0, "right": 0.0}
        self.last_center: Dict[str, Optional[Tuple[int, int]]] = {"left": None, "right": None}

        self.serial = None
        if serial_port:
            try:
                self.serial = SerialHandler(port=serial_port, baudrate=baudrate, timeout=timeout)
                self.serial.start_listening(lambda line: print(f"[Pico] {line}"))
            except Exception as exc:
                print(f"シリアル通信の初期化に失敗しました: {exc}")
                self.serial = None

    def _draw_detection(
        self,
        frame: np.ndarray,
        contour,
        center: Tuple[int, int],
        average_hsv,
        real_coords,
    ) -> None:
        center_x, center_y = center

        if average_hsv is not None:
            h, s, v = average_hsv
            cv2.putText(
                frame,
                f"HSV({h:.1f}, {s:.1f}, {v:.1f})",
                (center_x + 10, center_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
            )

        if real_coords is not None:
            cv2.putText(
                frame,
                f"World(Norm): ({real_coords[0]:.3f}, {real_coords[1]:.3f})",
                (center_x + 10, center_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 128, 255),
                2,
            )

        cv2.putText(
            frame,
            f"Center: ({center_x}, {center_y})",
            (center_x + 10, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
        )
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)


    def run(self) -> None:
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("フレームの読み込みに失敗しました。")
                break

            thresholds = self.ui.read_thresholds()
            mask, hsv_frame, contours = self.frame_processor.detect_contours(
                frame,
                thresholds.lower,
                thresholds.upper,
            )

            now = time.time()
            side_data = {
                "left": {"area": 0.0, "center": None},
                "right": {"area": 0.0, "center": None},
            }
            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= thresholds.min_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)

                mask_roi = mask[y:y + h, x:x + w]
                hsv_roi = hsv_frame[y:y + h, x:x + w]
                average_hsv = FrameProcessor.calculate_average_hsv(mask_roi, hsv_roi)
                real_coords = self.camera.convert_to_real_coordinates(center)

                self._draw_detection(frame, contour, center, average_hsv, real_coords)

                # シリアル通信でパンケーキの座標を送信
                if self.serial and real_coords is not None:
                    self.serial.send(f"({real_coords[0]:.3f}, {real_coords[1]:.3f})")

                # トリガー領域内にあるか確認
                if not self._is_within_trigger_region(center):
                    continue

                # サイドごとに最大面積のパンケーキを記録
                side = self._classify_side(center[0])
                if side == "left":
                    self._update_side_data(side_data, "left", area, center)
                elif side == "right":
                    self._update_side_data(side_data, "right", area, center)
                else:
                    self._update_side_data(side_data, "left", area, center)
                    self._update_side_data(side_data, "right", area, center)

            # 各サイドの評価とコマンド送信
            for side in ("left", "right"):
                area = side_data[side]["area"]
                center = side_data[side]["center"]
                self._evaluate_side(side, area, center, now)

            cv2.imshow(self.window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 終了処理
        self.camera.release()
        if self.command_queue:
            for side in ("left", "right"):
                if self.active_sides[side]:
                    self._send_command(f"stop_pour_{side}", reason="shutdown")
                    self.active_sides[side] = False
        if self.serial:
            self.serial.close()
        cv2.destroyAllWindows()

    def _classify_side(self, center_x: int) -> str:
        """入力されたx座標に基づいてパンケーキが左側にいるか右側にいるかを分類する。

        Args:
            center_x (int): パンケーキの中心x座標

        Returns:
            str: パンケーキの位置（"left"、"right"、"center"）
        """
        if center_x < self.left_boundary:
            return "left"
        if center_x > self.right_boundary:
            return "right"
        return "center"

    def _is_within_trigger_region(self, center: Tuple[int, int]) -> bool:
        """検出座標がトリガー領域内に存在するかを判定する。trigger_regionがNoneの場合は常にTrueを返す。"""
        if self.trigger_region is None:
            return True
        x_min, x_max, y_min, y_max = self.trigger_region
        x, y = center
        return x_min <= x <= x_max and y_min <= y <= y_max

    def _update_side_data(
        self,
        side_data: Dict[str, Dict[str, float]],
        side: str,
        area: float,
        center: Tuple[int, int],
    ) -> None:
        """サイドごとのパンケーキのデータを更新する。areaが現在記録されている面積より大きい場合に更新を行う。"""
        current_area = side_data[side]["area"]
        if area > current_area:
            side_data[side]["area"] = area
            side_data[side]["center"] = center

    def _evaluate_side(
        self,
        side: str,
        area: float,
        center: Optional[Tuple[int, int]],
        now: float,
    ) -> None:
        """サイドごとのパンケーキの状態を評価し、必要に応じてコマンドを送信する。"""

        # 開始条件の評価（面積と中心座標の確認，クールダウン時間の経過確認）
        if area >= self.MIN_CONTOUR_AREA and center is not None:
            self.last_seen_ts[side] = now
            self.last_center[side] = center
            cooldown_elapsed = now - self.last_trigger_ts[side] >= self.trigger_cooldown_s
            area_below_target = self.target_area is None or area < self.target_area
            if self.command_queue is not None and not self.active_sides[side] and cooldown_elapsed and area_below_target:
                self._send_command(
                    f"start_pour_{side}",
                    center=center,
                    area=area,
                )
                self.active_sides[side] = True
                self.last_trigger_ts[side] = now

        # activeでないサイドの場合は停止条件の評価をスキップ
        if not self.active_sides[side]:
            return

        # 停止条件の評価（面積が閾値を超えたか）
        stop_due_to_area = False
        if self.target_area is not None and area > 0:
            stop_due_to_area = area >= self.target_area

        if stop_due_to_area:
            stop_center = center or self.last_center[side]
            self._send_command(
                f"stop_pour_{side}",
                center=stop_center,
                area=area,
                reason="area_threshold",
            )
            self.active_sides[side] = False
            return

        # max_pour_time_s経過後に必ず停止する
        if now - self.last_seen_ts[side] >= self.max_pour_time_s:
            stop_center = self.last_center[side]
            self._send_command(
                f"stop_pour_{side}",
                center=stop_center,
                reason="timeout",
            )
            self.active_sides[side] = False

    def _send_command(self, command: str, **payload) -> None:
        if self.command_queue is None:
            return
        message = {"type": command, "timestamp": time.time(), **payload}
        try:
            self.command_queue.put_nowait(message)
            print(f"[Detector] Enqueued {command}: {payload}")
        except Exception as exc:
            print(f"キューへの送信に失敗しました: {exc}")

if __name__ == '__main__':
    detector = PancakeDetector(camera_index=0, serial_port='COM3')
    detector.run()
