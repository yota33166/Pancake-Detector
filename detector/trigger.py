"""Queue based trigger controller for servo commands."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple
import multiprocessing


@dataclass
class Detection:
    center: Tuple[int, int]
    area: float


class TriggerController:
    """サーボコマンドをキューに送信するタイミングを決定する。

    Args:
        command_queue: コマンドを送信するキュー。
        trigger_region: トリガー領域 (x_min, x_max, y_min, y_max)。
        cooldown_s: トリガーのクールダウン時間（秒）。
        release_delay_s: 停止コマンドを送信するまでの遅延時間（秒）。
        area_threshold: 停止コマンドを送信するための面積閾値。
    """

    def __init__(
        self,
        command_queue: Optional[multiprocessing.Queue],
        trigger_region: Optional[Tuple[int, int, int, int]],
        cooldown_s: float,
        release_delay_s: float,
        area_threshold: float,
    ) -> None:
        self.queue = command_queue
        self.trigger_region = trigger_region
        self.cooldown_s = max(0.0, cooldown_s)
        self.release_delay_s = max(0.0, release_delay_s)
        self.area_threshold = max(0.0, area_threshold)

        self._last_trigger_ts = 0.0
        self._last_seen_ts = 0.0
        self._active = False

    def handle_detection(self, detection: Detection, now: Optional[float] = None) -> None:
        """条件を満たした場合にサーボ開始コマンドを送信する。
        - 条件1: トリガーがアクティブでないこと。
        - 条件2: クールダウン時間（cooldown_s）が経過していること。
        - 条件3: トリガー領域内に検出座標が存在すること。

        Args:
            detection (Detection): 検出情報。
            now (Optional[float], optional): 現在時刻。
        """
        if self.queue is None:
            return
        if now is None:
            now = time.time()

        self._last_seen_ts = now

        if self._active:
            return
        if now - self._last_trigger_ts < self.cooldown_s:
            return
        if not self._is_within_region(detection.center):
            return

        self._send("start_pour", center=detection.center, area=detection.area)
        self._active = True
        self._last_trigger_ts = now

    def handle_release(self, detection: Detection, now: Optional[float] = None) -> None:
        """条件を満たした場合にサーボ停止コマンドを送信する。
        - 条件1: トリガーがアクティブであること。
        - 条件2: 最低停止時間（release_delay_s）が経過していること。
        - 条件3: 検出面積が閾値を超えていること。

        Args:
            detection (Detection): 検出情報。
            now (Optional[float], optional): 現在時刻。
        """
        if self.queue is None or not self._active:
            return
        if now is None:
            now = time.time()
        if now - self._last_seen_ts < self.release_delay_s:
            return
        if not self._is_over_region(detection.area):
            return

        self._send("stop_pour")
        self._active = False

    def shutdown(self) -> None:
        """トリガーコントローラーをシャットダウンし、必要に応じて停止コマンドを送信する。"""
        if self.queue is None or not self._active:
            return
        self._send("stop_pour", reason="shutdown")
        self._active = False

    def _is_within_region(self, center: Tuple[int, int]) -> bool:
        """検出座標がトリガー領域内に存在するかを判定する。"""
        if self.trigger_region is None:
            return True
        x_min, x_max, y_min, y_max = self.trigger_region
        x, y = center
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    def _is_over_region(self, area: float) -> bool:
        """検出面積が閾値を超えているかを判定する。"""
        if self.area_threshold is None:
            return True

        return area >= self.area_threshold

    def _send(self, command: str, **payload) -> None:
        """コマンドをキューに送信する。"""
        if self.queue is None:
            return
        message = {"type": command, "timestamp": time.time(), **payload}
        try:
            self.queue.put_nowait(message)
            print(f"[Detector] Enqueued {command}: {payload}")
        except Exception as exc:
            print(f"キューへの送信に失敗しました: {exc}")
