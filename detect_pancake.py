import cv2
import numpy as np
import glob
import time

from typing import Optional, Tuple

from serial_handler import SerialHandler

class PancakeDetector:
    LOWER_COLOR = np.array([10, 50, 50])  # 黄褐色の下限値 (HSV)
    UPPER_COLOR = np.array([40, 255, 255])  # 黄褐色の上限値 (HSV)
    MIN_CONTOUR_AREA = 1000  # 輪郭の最小面積
    HISTORY_SIZE = 1 # フレームの履歴サイズ(大きいほど遅延がすごい)

    def __init__(
        self,
        camera_index=0,
        calibration_file="calibration.npz",
        frame_width=1920,
        frame_height=1080,
        fps=30,
        serial_port=None,
        baudrate=115200,
        timeout=1,
        command_queue=None,
        trigger_region: Optional[Tuple[int, int, int, int]] = None,
        trigger_cooldown_s: float = 1.0, # 生地注ぎのクールダウン時間
        release_delay_s: float = 0.5, # 最低注ぎ継続時間
        ):

        self.cap = self.init_camera(camera_index)
        if self.cap is None:
            print("カメラの初期化に失敗しました。")
            exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # 平均化用のフレーム履歴
        self.frame_history = []

        # キャリブレーションデータの読み込み（存在する場合）
        self.cameraMatrix, self.distCoeffs, self.rvecs, self.tvecs = self.load_calibration(calibration_file)

        # UI準備
        self.window_name = 'PancakeDetector'
        self.max_area = frame_width * frame_height
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.init_ui()

        # シリアル通信の初期化（必要なら）
        if serial_port:
            try:
                self.serial = SerialHandler(port=serial_port, baudrate=baudrate, timeout=timeout)
                self.serial.start_listening(lambda line: print(f"[Pico] {line}"))
            except Exception as e:
                print(f"シリアル通信の初期化に失敗しました: {e}")
                self.serial = None

        self.command_queue = command_queue
        self.trigger_region = trigger_region
        self.trigger_cooldown_s = max(0.0, trigger_cooldown_s)
        self.release_delay_s = max(0.0, release_delay_s)
        self._last_trigger_ts = 0.0
        self._last_seen_ts = 0.0
        self._pour_active = False

    def init_camera(self, camera_index):
        """カメラを初期化"""
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # WindowsでのDirectShow使用
        if not cap.isOpened():
            print(f'カメラ{camera_index}が見つかりませんでした。')
            video_devices = glob.glob('/dev/video*')
            for device in video_devices:
                print(f"試行中のデバイス: {device}")
                cap = cv2.VideoCapture(device)
                if cap.isOpened():
                    print(f"カメラが見つかりました: {device}")
                    return cap
            print("カメラが見つかりませんでした。")
            return None

        self.configure_focus(cap)
        return cap
    
    def configure_focus(self, cap, autofocus=False, focus_value=100):
        """オートフォーカスを制御"""
        try:
            if autofocus is False:
                success = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                if not success:
                    print("警告: CAP_PROP_AUTOFOCUSでオートフォーカスを無効化できませんでした。")
                else:
                    print("オートフォーカスを無効にしました。")
            elif autofocus is True:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

            if focus_value is not None:
                if not 0 <= focus_value <= 255:
                    print("警告: focus_value は 0〜255 の範囲で指定してください。")
                else:
                    success = cap.set(cv2.CAP_PROP_FOCUS, float(focus_value))
                    if not success:
                        print("警告: CAP_PROP_FOCUS で手動フォーカス値を設定できませんでした。")
                    else:
                        print(f"フォーカス値を {focus_value} に設定しました。")
            
            # success = cap.set(cv2.CAP_PROP_EXPOSURE, 0.25)
            if not success:
                print("警告: CAP_PROP_EXPOSURE で露出を設定できませんでした。")
            else:
                print("露出を 0.25 に設定しました。")
                
        except Exception as e:
            print(f"フォーカス設定中にエラーが発生しました: {e}")

    def init_ui(self):
        """HSV色域と最小面積を調整するトラックバーを作成"""
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Controls', 420, 240)
        # 初期値
        hL, sL, vL = int(self.LOWER_COLOR[0]), int(self.LOWER_COLOR[1]), int(self.LOWER_COLOR[2])
        hU, sU, vU = int(self.UPPER_COLOR[0]), int(self.UPPER_COLOR[1]), int(self.UPPER_COLOR[2])
        min_area = int(self.MIN_CONTOUR_AREA)

        cv2.createTrackbar('H_lower',  'Controls', hL, 179, lambda v: None)
        cv2.createTrackbar('H_upper', 'Controls', hU, 179, lambda v: None)
        cv2.createTrackbar('S_lower',  'Controls', sL, 255, lambda v: None)
        cv2.createTrackbar('S_upper', 'Controls', sU, 255, lambda v: None)
        cv2.createTrackbar('V_lower',  'Controls', vL, 255, lambda v: None)
        cv2.createTrackbar('V_upper', 'Controls', vU, 255, lambda v: None)
        cv2.createTrackbar('MinArea','Controls', min_area, max(1, self.max_area), lambda v: None)

    def update_params_from_trackbar(self):
        """トラックバー値を読み取り、パラメータを更新"""
        hL = cv2.getTrackbarPos('H_lower',  'Controls')
        hU = cv2.getTrackbarPos('H_upper', 'Controls')
        sL = cv2.getTrackbarPos('S_lower',  'Controls')
        sU = cv2.getTrackbarPos('S_upper', 'Controls')
        vL = cv2.getTrackbarPos('V_lower',  'Controls')
        vU = cv2.getTrackbarPos('V_upper', 'Controls')
        min_area = cv2.getTrackbarPos('MinArea', 'Controls')

        hL, hU = sorted((hL, hU))
        sL, sU = sorted((sL, sU))
        vL, vU = sorted((vL, vU))

        self.LOWER_COLOR = np.array([hL, sL, vL], dtype=np.uint8)
        self.UPPER_COLOR = np.array([hU, sU, vU], dtype=np.uint8)
        self.MIN_CONTOUR_AREA = int(max(0, min_area))

    def load_calibration(self, file_name):
        """カメラのキャリブレーションデータを読み込む"""
        try:
            if glob.glob(file_name): 
                data = np.load(file_name)
                return data['cameraMatrix'], data['distCoeffs'], data['rvecs'], data['tvecs']
            else:
                print(f"キャリブレーションファイルが見つかりません: {file_name}")
                return None, None, None, None
        except Exception as e:
            print(f"キャリブレーションデータの読み込みエラー: {e}")
            return None, None, None, None

    @staticmethod
    def blur_frame(frame):
        """フレームをぼかす"""
        # ノイズ除去：ガウシアンフィルタを適用
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # ノイズ除去：メディアンフィルタを使用
        # frame = cv2.medianBlur(frame, 11)
        return frame

    def create_mask(self, hsv_frame):
        """指定した色範囲でマスクを作成"""
        # 指定した色範囲でマスクを作成
        mask = cv2.inRange(hsv_frame, self.LOWER_COLOR, self.UPPER_COLOR)

        # モルフォロジー処理：膨張と収縮を使ってノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # 閉処理：ノイズ除去と穴埋め
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)   # 開処理：小さな物体を除去
        return mask
    
    @staticmethod
    def calculate_average_hsv(mask, hsv_image):
        """指定されたマスクに基づいて、HSV画像から平均色を計算"""
        masked_hsv = hsv_image[mask > 0]
        if masked_hsv.size > 0:
            return tuple(np.mean(masked_hsv, axis=0))
        return None

    def detect_contour(self, frame):
        """フレームを処理して、輪郭を検知する"""
        # フレームをぼかす
        frame_blur = self.blur_frame(frame)

        # フレームの履歴を保持
        self.frame_history.append(frame_blur)
        # 履歴が設定したサイズを超えた場合、最古のフレームを削除
        if len(self.frame_history) > self.HISTORY_SIZE:
            self.frame_history.pop(0)

        # 履歴のフレームを平均化
        proc_frame = (
            np.mean(self.frame_history, axis=0).astype(np.uint8)
            if len(self.frame_history) > 1 else frame_blur
        )

        # HSV色空間に変換
        hsv_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
        #hsv = cv2.bilateralFilter(hsv, d=9, sigmaColor=75, sigmaSpace=75)

        # マスクを作成
        mask = self.create_mask(hsv_frame)

        # 輪郭を検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return mask, hsv_frame, contours


    def _is_within_trigger_region(self, center_x: int, center_y: int) -> bool:
        """指定された座標がトリガー領域内にあるか確認"""
        if self.trigger_region is None:
            return False
        x_min, x_max, y_min, y_max = self.trigger_region
        return x_min <= center_x <= x_max and y_min <= center_y <= y_max


    def _send_command(self, command: str, **payload) -> None:
        """コマンドをキューに送信"""
        # キューがない場合は何もしない
        if self.command_queue is None:
            return

        message = {"type": command, "timestamp": time.time(), **payload}
        try:
            self.command_queue.put_nowait(message)
            print(f"[Detector] Enqueued {command}: {payload}")
        except Exception as exc:
            print(f"キューへの送信に失敗しました: {exc}")


    def maybe_send_start(self, center_x: int, center_y: int, area: float) -> None:
        """検出があった場合に開始コマンドを送信"""
        now = time.time()
        self._last_seen_ts = now

        # キューがない、または注ぎ中であれば何もしない
        if self.command_queue is None or self._pour_active:
            return

        # 直近のトリガーからクールダウン時間が経過していなければ何もしない
        if now - self._last_trigger_ts < self.trigger_cooldown_s:
            return

        self._send_command("start_pour", center=(center_x, center_y), area=area)
        self._pour_active = True
        self._last_trigger_ts = now


    def maybe_send_stop(self, now: float) -> None:
        """検出が途絶えた場合に停止コマンドを送信"""
        # キューがない、または注ぎ中でない場合は何もしない
        if self.command_queue is None or not self._pour_active:
            return

        # リリース遅延時間内であれば何もしない
        if now - self._last_seen_ts < self.release_delay_s:
            return

        self._send_command("stop_pour")
        self._pour_active = False


    def convert_to_real_coordinates(self, pixel_coords):
        """
        画像座標を現実座標に変換（z=0平面を仮定）
        注意: 実際には外部パラメータと平面方程式（またはホモグラフィ）が必要
        """
        if self.cameraMatrix is None or self.distCoeffs is None or self.rvecs is None or self.tvecs is None:
            return None

        # 歪み補正後の正規化画像座標
        undist = cv2.undistortPoints(
            np.array([[pixel_coords]], dtype=np.float32),
            self.cameraMatrix,
            self.distCoeffs
        )

        # 簡易なz=0仮定での位置（用途に応じて要調整）
        # ここでは表示目的として正規化座標を返すだけに留める
        return undist.reshape(-1)

    def draw_contour(self, frame, contour, center_x, center_y, average_hsv=None):
        """輪郭を描画"""
        # 平均色相値を計算
        
        if average_hsv is not None:
            h, s, v = average_hsv
            cv2.putText(
                frame, 
                f'HSV({h:.1f}, {s:.1f}, {v:.1f})', 
                (center_x + 10, center_y + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0,
                (255, 0, 0),
                2
            )

        cv2.putText(
            frame,
            f'Center: ({center_x}, {center_y})', 
            (center_x + 10, center_y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0,
            (255, 0, 0),
            2
        )
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1) 
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    def run(self):
        if self.cap is None:
            print("カメラが初期化されていません。")
            return
        
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                   print("フレームの読み込みに失敗しました。")
                   break
            except Exception as e:
                print(e)
                break

            self.update_params_from_trackbar()
            mask, hsv_frame, contours = self.detect_contour(frame)
            # 検出した輪郭を描画
            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= self.MIN_CONTOUR_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                avg_hsv = self.calculate_average_hsv(mask[y:y + h, x:x + w], hsv_frame[y:y + h, x:x + w])
                # 現実座標に変換
                # 現実座標（必要なら表示用に別テキストで出す）
                real = self.convert_to_real_coordinates((center_x, center_y))
                if real is not None:
                    real_center_x, real_center_y = float(real[0]), float(real[1])
                    cv2.putText(
                        frame,
                        f"World(Norm): ({real_center_x:.3f}, {real_center_y:.3f})",
                        (center_x + 10, center_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 128, 255),
                        2
                    )

                self.draw_contour(frame, contour, center_x, center_y, avg_hsv)

                # シリアル通信で座標を送信
                if self.serial:
                    self.serial.send(f"({real_center_x:.3f}, {real_center_y:.3f})")

                if self._is_within_trigger_region(center_x, center_y):
                    # トリガー領域内であれば開始コマンドをキューに送信
                    self.maybe_send_start(center_x, center_y, area)

            # 検出が途絶えた場合に停止コマンドをキューに送信
            self.maybe_send_stop(time.time())

            # 結果を表示
            cv2.imshow(self.window_name, frame)
            #cv2.imshow('Mask', mask)
            #cv2.imshow('Filtered', result)
    
            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if self.cap.isOpened():
            self.cap.release()

        if self.command_queue and self._pour_active:
            self._send_command("stop_pour", reason="shutdown")
            self._pour_active = False

        if self.serial:
            self.serial.close()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = PancakeDetector(camera_index=0, serial_port='COM3')
    detector.run()
