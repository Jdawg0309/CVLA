"""Execution loop helpers for per-frame timing."""

import time


class FrameTimer:
    def __init__(self):
        self._last_time = time.time()

    def tick(self) -> float:
        """Return delta time since last tick."""
        current = time.time()
        delta = current - self._last_time
        self._last_time = current
        return delta
