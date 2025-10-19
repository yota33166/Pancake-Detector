import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from detector.camera import CameraManager
from detector.pipeline import FrameProcessor
from detector.trigger import Detection, TriggerController
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
        serial_port=None,
        baudrate=115200,
        timeout=1,
        command_queue=None,
        trigger_region: Optional[Tuple[int, int, int, int]] = None,
        trigger_cooldown_s: float = 1.0, # 生地注ぎのクールダウン時間
        release_delay_s: float = 0.5, # 最低注ぎ継続時間
        trigger_area_threshold: Optional[float] = None,
        ):
        frame_size = (frame_width, frame_height)
        self.window_name = "PancakeDetector"

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

        area_threshold = (
            float(trigger_area_threshold)
            if trigger_area_threshold is not None
            else float(self.MIN_CONTOUR_AREA)
        )
        self.trigger = TriggerController(
            command_queue=command_queue,
            trigger_region=trigger_region,
            cooldown_s=trigger_cooldown_s,
            release_delay_s=release_delay_s,
            area_threshold=area_threshold,
        )

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

                if self.serial and real_coords is not None:
                    self.serial.send(f"({real_coords[0]:.3f}, {real_coords[1]:.3f})")

                detection = Detection(center=center, area=area)
                self.trigger.handle_detection(detection, now=now)
                self.trigger.handle_release(detection=detection, now=time.time())

            cv2.imshow(self.window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        self.trigger.shutdown()
        if self.serial:
            self.serial.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = PancakeDetector(camera_index=0, serial_port='COM3')
    detector.run()
