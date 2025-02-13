import cv2
import numpy as np
import glob

class PancakeDetector:
    LOWER_COLOR = np.array([10, 50, 50])  # 黄褐色の下限値 (HSV)
    UPPER_COLOR = np.array([40, 255, 255])  # 黄褐色の上限値 (HSV)
    MIN_CONTOUR_AREA = 1000  # 輪郭の最小面積
    HISTORY_SIZE = 1 # フレームの履歴サイズ(大きいほど遅延がすごい)

    def __init__(self, camera_index=0, calibration_file="calibration.npz"):
        self.cap = self.initialize_camera(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame_history = []
        self.cameraMatrix, self.distCoeffs, self.rvecs, self.tvecs = self.load_calibration(calibration_file)

    def initialize_camera(self, camera_index):
        """カメラを初期化"""
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print('カメラが見つかりませんでした。')
            return None
        return cap
    
    def load_calibration(self, file_name):
        if glob.glob(file_name): 
            data = np.load(file_name)
            return data['cameraMatrix'], data['distCoeffs'], data['rvecs'], data['tvecs']
        return None, None, None, None

    def blur_frame(self, frame):
        """フレームをぼかす"""
        # ノイズ除去：ガウシアンフィルタを適用
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # ノイズ除去：メディアンフィルタを使用
        #frame = cv2.medianBlur(frame, 11)

        return frame

    def detect_contour(self, frame):
        """フレームを処理して、輪郭を検知する"""
        # フレームをぼかす
        frame = self.blur_frame(frame)

        # フレームの履歴を保持
        self.frame_history.append(frame)
        # 履歴が設定したサイズを超えた場合、最古のフレームを削除
        if len(self.frame_history) > self.HISTORY_SIZE:
            self.frame_history.pop(0)

        # 履歴のフレームを平均化
        if len(self.frame_history) > 1:
            frame = np.mean(self.frame_history, axis=0).astype(np.uint8)

        # HSV色空間に変換
        self.hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #hsv = cv2.bilateralFilter(hsv, d=9, sigmaColor=75, sigmaSpace=75)

        # マスクを作成
        self.mask = self.create_mask(self.hsv_frame)

        # マスクを適用して元画像をフィルタリング
        result = cv2.bitwise_and(frame, frame, mask=self.mask)

        # 輪郭を検出
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return result, contours

    def calculate_average_hsv(self, mask, hsv_image):
        """指定されたマスクに基づいて、HSV画像から平均色を計算"""
        masked_hsv = hsv_image[mask > 0]
        if masked_hsv.size > 0:
            return tuple(np.mean(masked_hsv, axis=0))
        return None

    def create_mask(self, hsv_frame):
        """指定した色範囲でマスクを作成"""
        # 指定した色範囲でマスクを作成
        mask = cv2.inRange(hsv_frame, self.LOWER_COLOR, self.UPPER_COLOR)

        # モルフォロジー処理：膨張と収縮を使ってノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # 閉処理：ノイズ除去と穴埋め
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)   # 開処理：小さな物体を除去
        return mask

    def convert_to_real_coordinates(self, pixel_coords):
        """画像座標を現実座標に変換する"""
        # ピクセル座標を歪み補正してカメラ座標に変換
        undistorted_points = cv2.undistortPoints(np.array([pixel_coords], dtype=np.float32), self.cameraMatrix, self.distCoeffs)

        # カメラ座標系での位置を計算 (平面上 z=0 と仮定)
        object_points = cv2.projectPoints(
            np.array([[undistorted_points[0][0][0], undistorted_points[0][0][1], 0]], dtype=np.float32),
            self.rvecs[0], self.tvecs[0], self.cameraMatrix, self.distCoeffs
        )
        return object_points[0].flatten()  # 3D座標を返す

    def draw_contour(self, frame, contour, center_x, center_y, average_hsv=None):
        """輪郭を描画"""
        # 平均色相値を計算
        
        if average_hsv is not None:
            average_h, average_s, average_v = average_hsv
            cv2.putText(frame, f'HSV({average_h:.1f}, {average_s:.1f}, {average_v:.1f})', 
                        (center_x + 10, center_y + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 0, 0), 2)
            
        cv2.putText(frame, f'Center: ({center_x}, {center_y})', 
                    (center_x + 10, center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (255, 0, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1) 
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    def run(self):
        while True:
            try:
                frame = self.cap.read()[1]
            except Exception as e:
                print(e)
                break
            result, contours = self.detect_contour(frame)
            # 検出した輪郭を描画
            for contour in contours:
                if cv2.contourArea(contour) > self.MIN_CONTOUR_AREA:  # 小さなノイズを除外
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # 現実座標に変換
                    if self.cameraMatrix is not None:
                        real_coordinates = self.convert_to_real_coordinates((center_x, center_y))
                        center_x, center_y = real_coordinates[0], real_coordinates[1]

                    average_hsv = self.calculate_average_hsv(self.mask[y:y+h, x:x+w], self.hsv_frame[y:y+h, x:x+w])
                    self.draw_contour(frame, contour, center_x, center_y, average_hsv)

            # 結果を表示

            cv2.namedWindow('PancakeDetector', cv2.WINDOW_NORMAL)
            cv2.imshow('PancakeDetector', frame)
            #cv2.imshow('Mask', mask)
            #cv2.imshow('Filtered', result)

            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if self.cap.isOpened():
            self.cap.release()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = PancakeDetector(1)
    detector.run()