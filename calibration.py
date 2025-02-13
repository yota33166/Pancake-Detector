import cv2
import numpy as np
import glob

def main():
    # チェスボードパターンのサイズ
    CHECKERBOARD = (9, 12)

    # 物体点と画像点を格納するリスト
    objpoints = []
    imgpoints = []

    # 物体点の準備（3Dポイント）
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # キャリブレーション画像のパス
    images = glob.glob('calibration_images/*.jpg')

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # チェスボードコーナーを検出
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # コーナーを描画
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # キャリブレーション実行
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.savez('calibration.npz',
            cameraMatrix=cameraMatrix,
            distCoeffs=distCoeffs,
            rvecs=rvecs,
            tvecs=tvecs)


def load_calibration(file_path):
    data = np.load(file_path)
    return data['cameraMatrix'], data['distCoeffs'], data['rvecs'], data['tvecs']

if __name__ == '__main__':
    main()
    cameraMatrix, distCoeffs, rvecs, tvecs = load_calibration('calibration.npz')
    print(cameraMatrix, distCoeffs, rvecs, tvecs)