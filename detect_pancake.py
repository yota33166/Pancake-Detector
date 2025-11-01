import multiprocessing
import os
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

# import helpers from marker_calibration
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # app -> project root
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
try:
    from marker_calibration.calib_marker_to_board import (
        solve_pose_from_two_markers,
        rvec_tvec_to_homogeneous,
        project_uv_to_board_xy,
        ARUCO_MARKER_LENGTH,
        INTER_MARKER_OFFSET_MM,
    )
except Exception as _e:  # pragma: no cover - optional dependency at runtime
    solve_pose_from_two_markers = None  # type: ignore
    rvec_tvec_to_homogeneous = None  # type: ignore
    project_uv_to_board_xy = None  # type: ignore
    ARUCO_MARKER_LENGTH = 43
    INTER_MARKER_OFFSET_MM = ARUCO_MARKER_LENGTH + 10.0


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
        board_transform_path: Optional[str] = None,
        marker_hold_time_s: float = 0.75,
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
        # 検知領域の初期化: Noneの場合は「下半分の中央付近」(x:35%-65%, y:50%-100%) に制限
        if trigger_region is None:
            x_min = int(frame_width * 0.35)
            x_max = int(frame_width * 0.65)
            y_min = int(frame_height * 0.50)
            y_max = frame_height - 1
            self.trigger_region = (x_min, x_max, y_min, y_max)
        else:
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

        self.serial = None
        if serial_port:
            try:
                self.serial = SerialHandler(port=serial_port, baudrate=baudrate, timeout=timeout)
                self.serial.start_listening(lambda line: print(f"[Pico] {line}"))
            except Exception as exc:
                print(f"シリアル通信の初期化に失敗しました: {exc}")
                self.serial = None

        # ArUco + transform setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.target_marker_ids = {0, 1}
        self.T_marker_to_board: Optional[np.ndarray] = None
        # decide transform path
        tf_default = os.path.join(BASE_DIR, 'T_marker_to_board.npy')
        tf_path = board_transform_path or tf_default
        try:
            if os.path.exists(tf_path):
                self.T_marker_to_board = np.load(tf_path)
                print(f"Loaded T_marker_to_board: {tf_path}")
            else:
                print(f"警告: 変換行列が見つかりません（{tf_path}）。実座標への変換は無効です。")
        except Exception as exc:
            print(f"T_marker_to_board の読み込みに失敗: {exc}")
            self.T_marker_to_board = None
        self.T_cam_marker: Optional[np.ndarray] = None
        self.last_marker_ts = 0.0
        self.marker_hold_time_s = max(0.0, float(marker_hold_time_s))

    def _draw_detection(
        self,
        frame: np.ndarray,
        contour,
        center: Tuple[int, int],
        average_hsv,
        real_coords,
        real_area_mm2: Optional[float] = None,
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
            if real_area_mm2 is not None and real_area_mm2 > 0:
                text = f"Board(mm): ({real_coords[0]:.1f}, {real_coords[1]:.1f}), A={real_area_mm2:.0f} mm^2"
            else:
                text = f"Board(mm): ({real_coords[0]:.1f}, {real_coords[1]:.1f})"
            text_color = (0, 255, 255)
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

    def _estimate_marker_pose(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Estimate camera->marker pose using 2-marker PnP if available.
        Returns T_cam_marker (4x4) with origin at ID0 center, else None.
        """
        if self.camera.camera_matrix is None or self.camera.dist_coeffs is None:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)
        if ids is None or len(ids) == 0:
            return None
        # filter to target ids
        keep_idx = [i for i, mid in enumerate(ids.flatten()) if int(mid) in self.target_marker_ids]
        if not keep_idx:
            return None
        filt_corners = [corners[i] for i in keep_idx]
        filt_ids = ids[keep_idx]
        # try two-marker PnP
        if solve_pose_from_two_markers is not None and rvec_tvec_to_homogeneous is not None:
            ok2, rvec, tvec = solve_pose_from_two_markers(
                filt_corners, filt_ids, self.camera.camera_matrix, self.camera.dist_coeffs,
                marker_len_mm=ARUCO_MARKER_LENGTH, inter_offset_mm=INTER_MARKER_OFFSET_MM,
            )
            if ok2 and rvec is not None and tvec is not None:
                return rvec_tvec_to_homogeneous(rvec, tvec)
        # fallback to first visible marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            filt_corners, float(ARUCO_MARKER_LENGTH), self.camera.camera_matrix, self.camera.dist_coeffs
        )
        if rvecs is None or len(rvecs) == 0 or rvec_tvec_to_homogeneous is None:
            return None
        return rvec_tvec_to_homogeneous(rvecs[0], tvecs[0])

    @staticmethod
    def _polygon_area_mm2(xy: np.ndarray) -> float:
        if xy is None or len(xy) < 3:
            return 0.0
        x = xy[:, 0]
        y = xy[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


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

                # Estimate marker pose for this frame (if transform available)
                if self.T_marker_to_board is not None:
                    T = self._estimate_marker_pose(frame)
                    if T is not None:
                        self.T_cam_marker = T
                        self.last_marker_ts = now
                    else:
                        # hold last pose for short duration, then invalidate
                        if self.T_cam_marker is not None and (now - self.last_marker_ts) > self.marker_hold_time_s:
                            self.T_cam_marker = None

                # フレーム前処理
                thresholds = self.ui.read_thresholds()
                mask, hsv_frame, contours = self.frame_processor.detect_contours(
                    frame,
                    thresholds.lower,
                    thresholds.upper,
                )
                # パンケーキ検知は trigger_region で限定。ハンドジェスチャーはフルフレームで実施。
                # 視覚化のために検知領域を描画
                if self.trigger_region is not None:
                    x_min, x_max, y_min, y_max = self.trigger_region
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (60, 180, 255), 2)
                    cv2.putText(frame, "Pancake ROI", (x_min + 5, max(y_min - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 180, 255), 1, cv2.LINE_AA)
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
                    real_coords = None
                    real_area_mm2 = None
                    # Project to board coordinates if possible
                    if (
                        self.T_marker_to_board is not None and
                        self.T_cam_marker is not None and
                        project_uv_to_board_xy is not None and
                        self.camera.camera_matrix is not None and
                        self.camera.dist_coeffs is not None
                    ):
                        try:
                            xy = project_uv_to_board_xy(
                                np.array(center, dtype=np.float32),
                                self.camera.camera_matrix,
                                self.camera.dist_coeffs,
                                self.T_cam_marker,
                                self.T_marker_to_board,
                            )
                            real_coords = (float(xy[0, 0]), float(xy[0, 1]))
                            # area: project contour polygon and compute area
                            pts = contour.reshape(-1, 2).astype(np.float32)
                            xy_poly = project_uv_to_board_xy(
                                pts,
                                self.camera.camera_matrix,
                                self.camera.dist_coeffs,
                                self.T_cam_marker,
                                self.T_marker_to_board,
                            )
                            real_area_mm2 = self._polygon_area_mm2(xy_poly)
                        except Exception as exc:
                            # fallback to normalized coords if projection fails
                            real_coords = self.camera.convert_to_real_coordinates(center)
                            real_area_mm2 = None
                    else:
                        real_coords = self.camera.convert_to_real_coordinates(center)

                    self._draw_detection(frame, contour, center, average_hsv, real_coords, real_area_mm2)

                    # シリアル通信でパンケーキの座標を送信（ボード座標が得られた場合はそれを送る）
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

                    # 面積を更新
                    if real_area_mm2 is not None:
                        area = real_area_mm2
                    
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

                # 両側同時停止: 同一フレームで両サイドが停止条件に到達したら同時停止コマンドを送る
                combined_stop_handled = False
                if self.active_sides["left"] and self.active_sides["right"]:
                    # 面積停止条件の評価
                    left_area = float(side_data["left"].get("area", 0.0) or 0.0)
                    right_area = float(side_data["right"].get("area", 0.0) or 0.0)
                    left_stop_area = (
                        self.target_area is not None and left_area > 0.0 and left_area >= float(self.target_area)
                    )
                    right_stop_area = (
                        self.target_area is not None and right_area > 0.0 and right_area >= float(self.target_area)
                    )

                    # タイムアウト停止条件の評価
                    left_stop_timeout = (now - self.last_trigger_ts["left"]) >= self.max_pour_time_s
                    right_stop_timeout = (now - self.last_trigger_ts["right"]) >= self.max_pour_time_s

                    if (left_stop_area or left_stop_timeout) and (right_stop_area or right_stop_timeout):
                        centers_payload = {
                            "left": side_data["left"].get("center") or self.last_center["left"],
                            "right": side_data["right"].get("center") or self.last_center["right"],
                        }
                        reasons_payload = {
                            "left": "area_threshold" if left_stop_area else "max_pour_time",
                            "right": "area_threshold" if right_stop_area else "max_pour_time",
                        }
                        self._send_command(
                            "stop_pour",  # サーボ側では両停止として処理
                            centers=centers_payload,
                            reasons=reasons_payload,
                        )
                        self.active_sides["left"] = False
                        self.active_sides["right"] = False
                        combined_stop_handled = True

                # 各サイドの評価とコマンド送信
                for side in ("left", "right"):
                    area = side_data[side]["area"]
                    center = side_data[side]["center"]
                    self._evaluate_and_send_commands(side, area, center, gesture_targets, now)

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

    def _evaluate_and_send_commands(
        self,
        side: str,
        area: float,
        center: Optional[Tuple[int, int]],
        gesture_targets: Set[str],
        now: float,
    ) -> None:
        """サイドごとのパンケーキの状態を評価し、必要に応じてコマンドを送信する。
        
        - start_pour_(side) コマンドの送信条件:
            1. クールダウン時間が経過している
            2. キー入力（a: left, l: right）による手動開始
            3. ジェスチャー認識結果による開始
        - stop_pour_(side) コマンドの送信条件:
            1. 検出面積がtarget_areaを超えた場合
            （max_pour_time_s経過後に必ず停止する）
        """
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
            return

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
        if now - self.last_trigger_ts[side] >= self.max_pour_time_s:
            stop_center = self.last_center[side]
            self._send_command(
                f"stop_pour_{side}",
                center=stop_center,
                reason="max_pour_time",
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
    detector = PancakeDetector(camera_index=0, serial_port='COM3', command_queue=multiprocessing.Queue())
    detector.run()
