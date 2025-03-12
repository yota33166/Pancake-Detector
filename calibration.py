import cv2
import numpy as np
import glob
import os

class CameraCalibration:
    def __init__(self, checkerboard_size=(9, 12), square_size_mm=20.0):
        """
        カメラキャリブレーションクラスの初期化
        
        Args:
            checkerboard_size (tuple): チェッカーボードの内部コーナー数 (幅, 高さ)
            square_size_mm (float): チェッカーボードの1マスのサイズ（mm単位）
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # 3D空間の点（世界座標系）を生成
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size_mm
        
        # キャリブレーション結果を保存する変数
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
    
    def calibrate_intrinsic(self, images_path, display=True, delay=500):
        """
        内部パラメータのキャリブレーション
        
        Args:
            images_path (str): キャリブレーション画像のパスパターン
            display (bool): キャリブレーション過程を表示するかどうか
            delay (int): 表示時の待機時間（ミリ秒）
            
        Returns:
            bool: キャリブレーションの成功/失敗
        """
        objpoints = []  # 3D空間の点
        imgpoints = []  # 画像上の点
        
        images = glob.glob(images_path)
        if not images:
            print(f"エラー: {images_path} にマッチする画像がありません")
            return False
        
        img_shape = None
        
        print(f"内部パラメータのキャリブレーション: {len(images)}枚の画像を処理します")
        
        for i, image_path in enumerate(images):
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告: 画像を読み込めませんでした: {image_path}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_shape = gray.shape[::-1]  # (width, height)
            
            # チェッカーボードのコーナーを検出
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                objpoints.append(self.objp)
                
                # サブピクセル精度でコーナー位置を調整
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                imgpoints.append(corners_refined)
                
                if display:
                    # コーナーを描画
                    img_display = img.copy()
                    cv2.drawChessboardCorners(img_display, self.checkerboard_size, corners_refined, ret)
                    cv2.putText(img_display, f"Image {i+1}/{len(images)}", (20, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('チェッカーボード検出', img_display)
                    cv2.waitKey(delay)
        
        if display:
            cv2.destroyAllWindows()
        
        if not objpoints:
            print("エラー: チェッカーボードを検出できませんでした")
            return False
        
        print("内部パラメータを計算中...")
        flags = 0
        # キャリブレーション実行（内部パラメータのみ）
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None, flags=flags)
        
        print(f"内部パラメータのキャリブレーション完了: {'成功' if ret else '失敗'}")
        if ret:
            self._print_intrinsic_params()
        
        return ret
    
    def calibrate_extrinsic(self, image_path, display=True):
        """
        外部パラメータのキャリブレーション
        
        Args:
            image_path (str): チェッカーボードを配置した鉄板の画像パス
            display (bool): キャリブレーション過程を表示するかどうか
            
        Returns:
            bool: キャリブレーションの成功/失敗
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("エラー: 外部パラメータを計算する前に内部パラメータをキャリブレーションしてください")
            return False
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"エラー: 画像を読み込めませんでした: {image_path}")
            return False
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # チェッカーボードのコーナーを検出
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if not ret:
            print(f"エラー: 画像からチェッカーボードを検出できませんでした: {image_path}")
            return False
        
        # サブピクセル精度でコーナー位置を調整
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        
        # 外部パラメータ（カメラ姿勢）を計算
        # PancakeDetector互換のため、rvecsとtvecsをリスト形式で保持
        ret, rvec, tvec = cv2.solvePnP(self.objp, corners_refined, self.camera_matrix, self.dist_coeffs)
        
        if ret:
            # rvecsとtvecsを通常の配列として保存（PancakeDetectorと互換性を保つため）
            self.rvecs = [rvec]
            self.tvecs = [tvec]
            
            if display:
                # 座標軸を描画
                img_axes = img.copy()
                self._draw_coordinate_axes(img_axes, corners_refined[0][0], rvec, tvec)
                cv2.imshow('外部パラメータ（カメラ姿勢）', img_axes)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            print("外部パラメータのキャリブレーション完了")
            self._print_extrinsic_params()
        
        return ret
    
    def save_calibration(self, file_path):
        """
        PancakeDetectorと互換性のあるキャリブレーション結果を保存
        
        Args:
            file_path (str): 保存先のファイルパス
        
        Returns:
            bool: 保存の成功/失敗
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("エラー: キャリブレーション結果がありません")
            return False
        
        # PancakeDetectorが期待する形式で保存
        np.savez(file_path,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                rvecs=self.rvecs,
                tvecs=self.tvecs)
        
        print(f"キャリブレーション結果を保存しました: {file_path}")
        return True
    
    def load_calibration(self, file_path):
        """
        キャリブレーション結果を読み込み
        
        Args:
            file_path (str): 読み込むファイルパス
        
        Returns:
            bool: 読み込みの成功/失敗
        """
        if not os.path.exists(file_path):
            print(f"エラー: ファイルが存在しません: {file_path}")
            return False
        
        try:
            data = np.load(file_path)
            self.camera_matrix = data['cameraMatrix']
            self.dist_coeffs = data['distCoeffs']
            
            if 'rvecs' in data and 'tvecs' in data:
                self.rvecs = data['rvecs']
                self.tvecs = data['tvecs']
            
            print(f"キャリブレーション結果を読み込みました: {file_path}")
            self._print_intrinsic_params()
            
            if self.rvecs is not None:
                self._print_extrinsic_params()
            
            return True
        except Exception as e:
            print(f"エラー: キャリブレーション結果の読み込みに失敗しました: {e}")
            return False
    
    def image_to_world_coordinates(self, image_points, z_plane=0.0):
        """
        画像座標をワールド座標に変換
        
        Args:
            image_points (array): 画像上の点の座標 [(x,y), ...] 
            z_plane (float): 変換先のZ平面（鉄板の高さ）
        
        Returns:
            array: ワールド座標の点
        """
        if self.camera_matrix is None or self.rvecs is None or not len(self.rvecs):
            print("エラー: 完全なキャリブレーション結果がありません")
            return None
            
        # 画像点を配列形式に変換
        image_points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        
        # 画像点を正規化（歪み補正）
        undistorted_points = cv2.undistortPoints(image_points, self.camera_matrix, self.dist_coeffs)
        
        # 回転ベクトルを回転行列に変換
        R, _ = cv2.Rodrigues(self.rvecs[0])  # 最初の回転ベクトルを使用
        R_inv = np.linalg.inv(R)
        
        # カメラ中心の座標
        C = -np.matmul(R_inv, self.tvecs[0])
        
        world_points = []
        for pt in undistorted_points:
            # 正規化座標から光線の方向ベクトルを計算
            ray = np.array([pt[0][0], pt[0][1], 1.0])
            ray = np.matmul(R_inv, ray)
            
            # Z平面との交点を計算
            t = (z_plane - C[2]) / ray[2]
            X = C[0] + t * ray[0]
            Y = C[1] + t * ray[1]
            Z = z_plane
            
            world_points.append([X[0], Y[0], Z])
        
        return np.array(world_points)
    
    def _print_intrinsic_params(self):
        """内部パラメータを表示"""
        print("--- 内部パラメータ ---")
        print("カメラ行列:")
        print(self.camera_matrix)
        print("歪み係数:")
        print(self.dist_coeffs)
    
    def _print_extrinsic_params(self):
        """外部パラメータを表示"""
        if self.rvecs is None or self.tvecs is None or not len(self.rvecs):
            return
            
        print("--- 外部パラメータ ---")
        print("回転ベクトル:")
        print(self.rvecs[0])
        R, _ = cv2.Rodrigues(self.rvecs[0])
        print("回転行列:")
        print(R)
        print("並進ベクトル:")
        print(self.tvecs[0])
    
    def _draw_coordinate_axes(self, img, origin, rvec, tvec, axis_length=50.0):
        """
        カメラ姿勢を視覚化するための座標軸を描画
        
        Args:
            img (array): 描画する画像
            origin (array): チェッカーボードの原点
            rvec (array): 回転ベクトル
            tvec (array): 並進ベクトル
            axis_length (float): 描画する軸の長さ
        """
        # 世界座標系の軸
        axis_points = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],  # X軸
            [0, axis_length, 0],  # Y軸
            [0, 0, axis_length]   # Z軸
        ])
        
        # 世界座標系から画像座標系へ変換
        imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        
        # 座標軸を描画
        img = cv2.line(img, tuple(imgpts[0].ravel().astype(int)), 
                       tuple(imgpts[1].ravel().astype(int)), (0, 0, 255), 3)  # X軸: 赤
        img = cv2.line(img, tuple(imgpts[0].ravel().astype(int)), 
                       tuple(imgpts[2].ravel().astype(int)), (0, 255, 0), 3)  # Y軸: 緑
        img = cv2.line(img, tuple(imgpts[0].ravel().astype(int)), 
                       tuple(imgpts[3].ravel().astype(int)), (255, 0, 0), 3)  # Z軸: 青
        
        return img


def main():
    """メイン関数"""
    # キャリブレーションインスタンスの作成（チェッカーボードのサイズを指定）
    calibration = CameraCalibration(checkerboard_size=(9, 12), square_size_mm=20.0)
    
    # コマンドライン引数解析用（実行モードの選択）
    import argparse
    
    parser = argparse.ArgumentParser(description='カメラキャリブレーションツール')
    parser.add_argument('--mode', choices=['intrinsic', 'extrinsic', 'both'], default='both',
                        help='キャリブレーションモード: intrinsic=内部パラメータのみ, extrinsic=外部パラメータのみ, both=両方')
    parser.add_argument('--intrinsic-images', default='calibration_images/*.jpg',
                        help='内部パラメータ用のキャリブレーション画像パス（ワイルドカード対応）')
    parser.add_argument('--extrinsic-image', default='extrinsic_image.jpg',
                        help='外部パラメータ用のキャリブレーション画像パス')
    parser.add_argument('--output', default='calibration.npz',
                        help='キャリブレーション結果の保存先ファイル')
    parser.add_argument('--load', default=None,
                        help='内部パラメータを読み込むファイルパス（外部パラメータのみの計算時）')
    parser.add_argument('--nodisplay', action='store_true',
                        help='処理中の画像表示を無効化')
    
    args = parser.parse_args()
    
    display = not args.nodisplay
    
    # 内部パラメータのロードまたはキャリブレーション
    if args.mode in ['intrinsic', 'both']:
        # 内部パラメータをキャリブレーション
        calibration.calibrate_intrinsic(args.intrinsic_images, display=display)
        
        # 内部パラメータのみの場合、ここで保存して終了
        if args.mode == 'intrinsic':
            calibration.save_calibration(args.output)
            print(f"内部パラメータをファイルに保存しました: {args.output}")
            return
    else:
        # 外部パラメータのみの場合、内部パラメータをロード
        if args.load:
            if not calibration.load_calibration(args.load):
                print("内部パラメータの読み込みに失敗しました。終了します。")
                return
        else:
            print("エラー: 外部パラメータのみのモードでは --load オプションが必要です")
            return
    
    # 外部パラメータのキャリブレーション
    if args.mode in ['extrinsic', 'both']:
        calibration.calibrate_extrinsic(args.extrinsic_image, display=display)
    
    # 結果を保存
    calibration.save_calibration(args.output)
    print(f"キャリブレーション結果をファイルに保存しました: {args.output}")


if __name__ == '__main__':
    main()