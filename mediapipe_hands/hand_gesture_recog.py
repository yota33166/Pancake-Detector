"""MediaPipe のジェスチャー検出を扱う補助モジュール。

主な目的は以下のとおり:

- 可読性向上: 日本語Docstringと型ヒントを整備。
- 再利用性向上: dataclassとクラス設計で結果を扱いやすく整理。
- メンテナンス性向上: loggingを導入し、動作状況を把握しやすくする。

将来的に thumb up ジェスチャーを外部モジュールへ通知する土台を用意する。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


# MediaPipe のユーティリティオブジェクトをモジュールレベルで共有する。
IMAGE_FILENAMES = ['thumbs_down.jpg', 'victory.jpg', 'thumbs_up.jpg', 'pointing_up.jpg']
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
GestureRecognizerResultType = Any
MODEL_PATH = Path(__file__).parent / "gesture_recognizer.task"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class GestureCategoryResult:
	"""ジェスチャー分類結果の代表候補を表現する。"""

	name: str
	score: float


@dataclass
class GestureDetectionSnapshot:
	"""単一タイムスタンプでのジェスチャー認識結果。"""

	timestamp_ms: int
	categories: list[GestureCategoryResult] = field(default_factory=list)
	landmarks: list[Sequence[landmark_pb2.NormalizedLandmark]] = field(default_factory=list)

	def top_category(self) -> Optional[GestureCategoryResult]:
		"""最もスコアの高いジェスチャー候補を返す。"""

		return self.categories[0] if self.categories else None

	def is_matching(self, gesture_name: str, threshold: float = 0.8) -> bool:
		"""指定ジェスチャーが所定スコア以上で検出されたか判定。"""

		top = self.top_category()
		if not top:
			return False
		return top.name == gesture_name and top.score >= threshold


ResultCallback = Callable[[GestureDetectionSnapshot], None]


class GestureRecognizerRunner:
	"""MediaPipe Gesture Recognizer を利用したリアルタイム推論ラッパー。"""

	def __init__(
		self,
		camera_index: int = 0,
		model_path: Path = MODEL_PATH,
		*,
		num_hands: int = 1,
		min_score: float = 0.8,
		result_callback: Optional[ResultCallback] = None,
		window_name: str = "MediaPipe Gesture",
	) -> None:
		"""推論に必要な依存性とコールバックを設定する。"""

		self.camera_index = camera_index
		self.model_path = model_path
		self.num_hands = num_hands
		self.min_score = min_score
		self.result_callback = result_callback
		self.window_name = window_name
		self._latest_snapshot: Optional[GestureDetectionSnapshot] = None

	@property
	def latest_snapshot(self) -> Optional[GestureDetectionSnapshot]:
		"""現在保持している最新の結果を返す。"""

		return self._latest_snapshot

	def is_gesture_detected(self, gesture_name: str, *, threshold: Optional[float] = None) -> bool:
		"""最新結果がジェスチャー条件を満たすかを判定するユーティリティ。"""

		snapshot = self._latest_snapshot
		if not snapshot:
			return False
		score_threshold = threshold if threshold is not None else self.min_score
		return snapshot.is_matching(gesture_name, score_threshold)

	def init_recognizer(self):
		"""Gesture Recognizer を初期化する。"""

		options = GestureRecognizerOptions(
			base_options=BaseOptions(model_asset_path=str(self.model_path)),
			num_hands=self.num_hands,
			running_mode=VisionRunningMode.LIVE_STREAM,
			result_callback=self._handle_result,
		)
		recognizer = GestureRecognizer.create_from_options(options)
		return recognizer

	def run(self) -> None:
		"""カメラからフレームを取得しながらリアルタイム推論を実行する。"""

		if not self.model_path.exists():
			raise FileNotFoundError(f"gesture model not found: {self.model_path}")

		cap = cv2.VideoCapture(self.camera_index)
		if not cap.isOpened():
			raise RuntimeError(f"failed to open camera index {self.camera_index}")

		logger.info(
			"starting gesture recognition (camera_index=%s, model=%s)",
			self.camera_index,
			self.model_path,
		)

		recognizer = self.init_recognizer()
		try:
			while True:
				ret, frame = cap.read()
				if not ret:
					logger.warning("failed to read frame from camera")
					break

				frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
				recognizer.recognize_async(mp_image, int(time.time() * 1000))

				self.render_overlay(frame)

				cv2.imshow(self.window_name, frame)
				key = cv2.waitKey(1) & 0xFF
				if key in (27, ord('q')):
					logger.info("exit requested by keypress: %s", key)
					break
		finally:
			recognizer.close()
			cap.release()
			cv2.destroyWindow(self.window_name)
			cv2.destroyAllWindows()
			logger.info("gesture recognition stopped")

	def _handle_result(self, result: GestureRecognizerResultType, output_image, timestamp_ms: int) -> None:
		"""MediaPipe から受け取った結果をスナップショットに変換する。"""

		categories = self._extract_categories(result)
		snapshot = GestureDetectionSnapshot(
			timestamp_ms=timestamp_ms,
			categories=categories,
			landmarks=result.hand_landmarks or [],
		)
		self._latest_snapshot = snapshot

		top = snapshot.top_category()
		if top and top.score >= self.min_score:
			logger.debug("detected gesture: %s (score=%.2f)", top.name, top.score)

		if self.result_callback is not None:
			self.result_callback(snapshot)

	@staticmethod
	def _extract_categories(result: GestureRecognizerResultType) -> list[GestureCategoryResult]:
		"""MediaPipeの分類結果からトップ候補を抽出する。"""

		categories: list[GestureCategoryResult] = []
		for gesture_candidates in result.gestures:
			if not gesture_candidates:
				continue
			best_candidate = max(gesture_candidates, key=lambda c: c.score)
			categories.append(
				GestureCategoryResult(
					name=best_candidate.category_name,
					score=best_candidate.score,
				)
			)
		return categories

	def render_overlay(self, frame) -> None:
		"""最新結果をフレームに描画する。"""

		snapshot = self._latest_snapshot
		if not snapshot:
			return

		for landmarks, category in zip_longest(snapshot.landmarks, snapshot.categories):
			self._draw_hand_annotations(frame, landmarks, category)

	def _draw_hand_annotations(
		self,
		frame,
		hand_landmarks: Sequence[landmark_pb2.NormalizedLandmark],
		category: Optional[GestureCategoryResult],
	) -> None:
		"""検出した手のランドマークとラベルを描く。"""

		if not hand_landmarks:
			return

		# ランドマーク描画（なぜか上手く描画されない）
		# proto = landmark_pb2.NormalizedLandmarkList()
		# proto.landmark.extend(
		# 	landmark_pb2.NormalizedLandmark(
		# 		x=lm.x,
		# 		y=lm.y,
		# 		z=lm.z,
		# 		visibility=getattr(lm, "visibility", 0.0),
		# 		presence=getattr(lm, "presence", 0.0),
		# 	)
		# 	for lm in hand_landmarks
		# )
		# mp_drawing.draw_landmarks(
		# 	frame,
		# 	proto,
		# 	mp_hands.HAND_CONNECTIONS,
		# 	mp_drawing_styles.get_default_hand_landmarks_style(),
		# 	mp_drawing_styles.get_default_hand_connections_style(),
		# )

		if not category:
			return

		height, width = frame.shape[:2]
		wrist = hand_landmarks[0]
		x_px = min(max(int(wrist.x * width), 0), width - 1)
		y_px = min(max(int(wrist.y * height) - 10, 0), height - 1)

		if category.name == "Pointing_Up":
			show_name = "Pour Left"
		elif category.name == "Victory":
			show_name = "Pour Right"
		elif category.name == "Open_Palm":
			show_name = "Pour Both"
		else:
			show_name = category.name

		label = f"{show_name} ({category.score:.2f})"
		size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
		rect_x2 = min(x_px + size[0] + 10, width - 1)
		rect_y1 = max(y_px - size[1] - baseline - 6, 0)
		rect_y2 = min(y_px + baseline + 6, height - 1)
		cv2.rectangle(
			frame,
			(x_px - 5, rect_y1),
			(rect_x2, rect_y2),
			(0, 0, 0),
			cv2.FILLED,
		)
		cv2.putText(
			frame,
			label,
			(x_px, y_px),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(254, 220, 156),
			2,
			# cv2.LINE_AA,
		)


def run_realtime_detection(camera_index: int = 0, model_path: Path = MODEL_PATH) -> None:
	"""簡易実行用のラッパー。従来APIとの互換性を保つ。"""

	runner = GestureRecognizerRunner(camera_index=camera_index, model_path=model_path)
	runner.run()


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	run_realtime_detection()
#@markdown We implemented some functions to visualize the gesture recognition results. <br/> Run the following cell to activate the functions.
