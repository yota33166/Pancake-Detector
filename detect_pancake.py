import multiprocessing
import sys
import time
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk

from detector.camera import CameraManager
from detector.pipeline import FrameProcessor
from detector.ui import DetectorUI
from mediapipe_hands.hand_gesture_recog import GestureRecognizerRunner
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
        fps=20,
        serial_port: Optional[str] = None,
        baudrate=115200,
        timeout=1,
        command_queue: Optional[multiprocessing.Queue] = None,
        trigger_region: Optional[Tuple[int, int, int, int]] = None,
        trigger_cooldown_s: float = 1.0, # 生地注ぎのクールダウン時間
        max_pour_time_s: float = 2.0, # 最大注ぎ時間
        # release_delay_s: float = 0.5, # 最低注ぎ継続時間
        trigger_area_threshold: Optional[float] = 5000.0,
        ):
        frame_size = (frame_width, frame_height)
        self.window_name = "PancakeDetector"
        self.processing_interval_s = (1.0 / fps) if fps > 0 else 0.0
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

        try:
            self.gesture_recognizer_runner = GestureRecognizerRunner()
        except RuntimeError as exc:
            print(f"mediapipeジェスチャー認識器の初期化に失敗しました: {exc}")
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
        self.last_trigger_ts: Dict[str, float] = {"left": 0.0, "right": 0.0}
        self.last_center: Dict[str, Optional[Tuple[int, int]]] = {"left": None, "right": None}
        self.key = -1
        self.dual_session_active = False

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

        # if average_hsv is not None:
        #     h, s, v = average_hsv
        #     cv2.putText(
        #         frame,
        #         f"HSV({h:.1f}, {s:.1f}, {v:.1f})",
        #         (center_x + 10, center_y + 15),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1.0,
        #         (255, 0, 0),
        #         2,
        #     )

        # Build label and color
        if real_coords is not None:
            text = f"World(Norm): ({real_coords[0]:.3f}, {real_coords[1]:.3f})"
            text_color = (0, 255, 0)
        else:
            text = f"Pancake: ({center_x}, {center_y})"
            text_color = (141, 192, 177)
        # Text rendering params
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        pad = 4

        # Text origin (baseline at this point)
        org = (center_x + 10, center_y + 35)

        # Measure text and draw background rectangle
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x1 = org[0] - pad
        y1 = org[1] - text_h - baseline - pad
        x2 = org[0] + text_w + pad
        y2 = org[1] + pad
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED)

        # Draw text over the background
        cv2.putText(frame, text, org, font, font_scale, text_color, thickness)


        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)


    def run(self) -> None:
        side_data = {
            "left": {"area": 0.0, "center": None},
            "right": {"area": 0.0, "center": None},
        }
        root = tk.Tk()
        root.withdraw()  # ウィンドウを表示せずに処理だけ行う

        # 画面の幅と高さを取得
        max_screen_width = root.winfo_screenwidth()
        max_screen_height = root.winfo_screenheight()
        root.destroy()

        detected_hand_gesture = "None"
        recognizer = None
        try:
            # mediapipeジェスチャー認識器の初期化
            recognizer = self.gesture_recognizer_runner.init_recognizer()
            last_processed_ts = 0.0
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("フレームの読み込みに失敗しました。")
                    break

                now = time.time()
                if self.processing_interval_s > 0.0 and (now - last_processed_ts) < self.processing_interval_s:
                    self.key = cv2.waitKey(1) & 0xFF
                    if self.key == ord('q'):
                        break
                    continue
                last_processed_ts = now

                # フレーム前処理
                thresholds = self.ui.read_thresholds()
                mask, hsv_frame, contours = self.frame_processor.detect_contours(
                    frame,
                    thresholds.lower,
                    thresholds.upper,
                )
                # Mediapipeジェスチャー認識
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                recognizer.recognize_async(mp_image, int(time.time() * 1000))
                snapshot = self.gesture_recognizer_runner.latest_snapshot
                if snapshot:
                    detected_result = snapshot.top_category()
                    detected_hand_gesture = detected_result.name if detected_result else "None"
                    self.gesture_recognizer_runner.render_overlay(frame)
                gesture_targets = self._resolve_gesture_targets(detected_hand_gesture)

                now = time.time()
                # サーボがアクティブでないサイドのデータを初期化
                if not self.active_sides["left"]:
                    side_data["left"] = {"area": 0.0, "center": None}
                if not self.active_sides["right"]:
                    side_data["right"] = {"area": 0.0, "center": None}

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
                    if center is not None:
                        side = self._classify_side(center)
                    else:
                        side = None
                        
                    if side == "left":
                        self._update_side_data(side_data, "left", area, center)
                    elif side == "right":
                        self._update_side_data(side_data, "right", area, center)
                    else:
                        self._update_side_data(side_data, "left", area, center)
                        self._update_side_data(side_data, "right", area, center)

                # 両側同時開始が必要であれば先に評価する
                if gesture_targets == {"left", "right"}:
                    self._attempt_dual_start(side_data, now)

                # 各サイドの評価とコマンド送信
                dual_area_candidates: Dict[str, Dict[str, object]] = {}
                for side in ("left", "right"):
                    area = side_data[side]["area"]
                    center = side_data[side]["center"]
                    eval_result = self._evaluate_and_send_commands(side, area, center, gesture_targets, now)
                    if self.dual_session_active and eval_result.get("area_stop"):
                        dual_area_candidates[side] = {
                            "center": eval_result.get("stop_center"),
                            "area": eval_result.get("stop_area", 0.0),
                        }

                if self.dual_session_active and all(side in dual_area_candidates for side in ("left", "right")):
                    self._finalize_dual_area_stop(dual_area_candidates)

                frame_resized = cv2.resize(frame, (max_screen_width, max_screen_height), interpolation=cv2.INTER_NEAREST)
                cv2.imshow(self.window_name, frame_resized)

                self.key = cv2.waitKey(1) & 0xFF
                if self.key == ord('q'):
                    break

        # 終了処理
        finally:
            self.camera.release()
            if recognizer:
                recognizer.close()
            if self.command_queue:
                for side in ("left", "right"):
                    if self.active_sides[side]:
                        self._send_command(f"stop_pour_{side}", reason="shutdown")
                        self.active_sides[side] = False
            if self.serial:
                self.serial.close()
            cv2.destroyAllWindows()

    def _classify_side(self, center: Tuple[int, int]) -> str:
        """入力されたx座標に基づいてパンケーキが左側にいるか右側にいるかを分類する。

        Args:
            center (Tuple[int, int]): パンケーキの中心座標 (x, y)

        Returns:
            str: パンケーキの位置（"left"、"right"、"center"）
        """
        center_x, center_y = center
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
        side_data: Dict[str, Dict[str, object]],
        side: str,
        area: float,
        center: Tuple[int, int],
    ) -> None:
        """サイドごとのパンケーキのデータを更新する。areaが現在記録されている面積より大きい場合に更新を行う。"""
        current_area = side_data[side]["area"]
        if area > current_area:
            side_data[side]["area"] = area
            side_data[side]["center"] = center

    def _resolve_gesture_targets(self, gesture_name: str) -> Set[str]:
        """ジェスチャー名から対象サイド集合を返す。"""

        mapping = {
            "Pointing_Up": {"left"},
            "Victory": {"right"},
            "Thumb_Up": {"left", "right"},
        }
        return mapping.get(gesture_name, set())

    def _attempt_dual_start(
        self,
        side_data: Dict[str, Dict[str, object]],
        now: float,
    ) -> None:
        """"Thumb_Up"ジェスチャー時に両側同時開始を試みる。"""

        sides = ("left", "right")

        # どちらか一方でも稼働中なら個別ロジックに任せる
        if any(self.active_sides[side] for side in sides):
            return

        # クールダウン未経過であれば開始できない
        cooldown_ready = all(
            now - self.last_trigger_ts[side] >= self.trigger_cooldown_s for side in sides
        )
        if not cooldown_ready:
            return

        # 面積しきい値を超えている場合は同時開始を行わない
        if self.target_area is not None:
            for side in sides:
                area = side_data[side].get("area", 0.0) or 0.0
                if area > 0.0 and area >= self.target_area:
                    return

        # 中心座標を記録（Noneのままでもよい）
        centers_payload: Dict[str, Optional[Tuple[int, int]]] = {}
        areas_payload: Dict[str, float] = {}
        for side in sides:
            center = side_data[side].get("center")
            area = float(side_data[side].get("area", 0.0) or 0.0)
            self.last_center[side] = center
            centers_payload[side] = center
            areas_payload[side] = area

        self._send_command(
            "start_pour",
            centers=centers_payload,
            areas=areas_payload,
        )
        for side in sides:
            self.active_sides[side] = True
            self.last_trigger_ts[side] = now
        self.dual_session_active = True

    def _evaluate_and_send_commands(
        self,
        side: str,
        area: float,
        center: Optional[Tuple[int, int]],
        gesture_targets: Set[str],
        now: float,
    ) -> Dict[str, object]:
        """サイドごとのパンケーキの状態を評価し、必要に応じてコマンドを送信する。
        
        - start_pour_(side) コマンドの送信条件:
            1. クールダウン時間が経過している
            2. キー入力（a: left, l: right）による手動開始
            3. ジェスチャー認識結果による開始
        - stop_pour_(side) コマンドの送信条件:
            1. 検出面積がtarget_areaを超えた場合
            （max_pour_time_s経過後に必ず停止する）
        """
        result: Dict[str, object] = {
            "area_stop": False,
            "time_stop": False,
            "stop_center": None,
            "stop_area": float(area),
        }
        manual_mapping = {
            ord('a'): 'left',
            ord('l'): 'right',
        }
        # 停止条件の評価（面積が閾値を超えたか）
        stop_due_to_area = False
        if self.target_area is not None and area > 0:
            stop_due_to_area = area >= self.target_area

        # 開始条件の評価
        # 2. キー入力による手動開始
        is_target_key_pressed = self.key in manual_mapping and manual_mapping[self.key] == side
        # 3. mediapipe_handsのジェスチャー認識結果を用いた開始条件
        is_gesture_start = side in gesture_targets

        if is_target_key_pressed or is_gesture_start:
            # 1. クールダウン時間が経過している
            self.last_center[side] = center
            cooldown_elapsed = now - self.last_trigger_ts[side] >= self.trigger_cooldown_s

            if not self.active_sides[side] and cooldown_elapsed and not stop_due_to_area: # and area_below_target:
                self._send_command(
                    f"start_pour_{side}",
                    center=center,
                    area=area,
                )
                self.active_sides[side] = True
                self.last_trigger_ts[side] = now

        # activeでないサイドの場合は停止条件の評価をスキップ
        if not self.active_sides[side]:
            return result

        if stop_due_to_area:
            stop_center = center or self.last_center[side]
            result["area_stop"] = True
            result["stop_center"] = stop_center
            if not self.dual_session_active:
                self._send_command(
                    f"stop_pour_{side}",
                    center=stop_center,
                    area=area,
                    reason="area_threshold",
                )
                self.active_sides[side] = False
            return result

        # max_pour_time_s経過後に必ず停止する
        if now - self.last_trigger_ts[side] >= self.max_pour_time_s:
            stop_center = self.last_center[side]
            self._send_command(
                f"stop_pour_{side}",
                center=stop_center,
                reason="max_pour_time",
            )
            self.active_sides[side] = False
            result["time_stop"] = True
            result["stop_center"] = stop_center
            if self.dual_session_active:
                self.dual_session_active = False
            return result

        return result

    def _finalize_dual_area_stop(self, area_candidates: Dict[str, Dict[str, object]]) -> None:
        """両側同時開始セッションで両方が面積停止条件を満たした際に同時停止させる。"""

        centers_payload: Dict[str, Optional[Tuple[int, int]]] = {}
        areas_payload: Dict[str, float] = {}
        for side in ("left", "right"):
            candidate = area_candidates.get(side, {})
            stop_center = candidate.get("center") or self.last_center[side]
            stop_area = float(candidate.get("area", 0.0))
            self.last_center[side] = stop_center
            centers_payload[side] = stop_center
            areas_payload[side] = stop_area
            self.active_sides[side] = False

        self._send_command(
            "stop_pour",
            centers=centers_payload,
            areas=areas_payload,
            reason="area_threshold",
        )
        self.dual_session_active = False
        
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
    detector = PancakeDetector(camera_index=0, serial_port='COM3', command_queue=multiprocessing.Queue())
    detector.run()
