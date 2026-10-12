import collections
import threading
from typing import Any


class ThreadSafeCircularBuffer:
    def __init__(self, maxlen: int = 10):
        self._maxlen = maxlen
        self._lock = threading.Lock()
        self._buffer = collections.deque(maxlen=maxlen)

    def add(self, item: Any):
        with self._lock:
            self._buffer.append(item)

    def get_last(self) -> Any:
        with self._lock:
            return self._buffer[-1]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
