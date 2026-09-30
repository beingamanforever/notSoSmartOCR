"""A least-recently-used cache bounded by the bytes it holds, shared across requests.

Keys are content hashes, so a hit is the stored result for byte-identical input. It
lives in memory only: nothing reaches disk and a restart clears it. Keys are not
versioned by code, so the cache must not outlive the process it was filled by; a
restart is also how new code is deployed.
"""

from __future__ import annotations

import pickle
import threading
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class BoundedCache:
    """Values are pickled and zlib-compressed.

    The least recently used entries are dropped past `max_bytes`, and entries older
    than `ttl_seconds` are never returned, which bounds how long document text is kept.
    `get_or_compute` runs one computation per key at a time: concurrent requests for
    the same input wait for its result instead of paying for it again.
    """

    def __init__(self, max_bytes: int, ttl_seconds: float | None = None) -> None:
        if max_bytes <= 0:
            raise ValueError("A cache needs a positive byte bound")
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self.bytes = 0
        self.hits = self.misses = self.evictions = 0
        self._items: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._items.get(key)
            if item is not None and self.ttl_seconds is not None and time.monotonic() - item[0] > self.ttl_seconds:
                self._drop(key)
                item = None
            if item is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
        return pickle.loads(zlib.decompress(item[1]))

    def put(self, key: str, value: Any) -> None:
        blob = zlib.compress(pickle.dumps(value, pickle.HIGHEST_PROTOCOL), 3)
        if len(blob) > self.max_bytes:
            return
        with self._lock:
            if key in self._items:
                self._drop(key)
            self._items[key] = (time.monotonic(), blob)
            self.bytes += len(blob)
            while self.bytes > self.max_bytes:
                self._drop(next(iter(self._items)))
                self.evictions += 1

    def get_or_compute(
        self, key: str, compute: Callable[[], T], keep: Callable[[T], bool] = lambda _: True
    ) -> T:
        """The stored value, else `compute()`, stored when `keep` accepts it.

        Errors and rejected values are never stored; a waiter then computes its own.
        """
        while True:
            value = self.get(key)
            if value is not None:
                return value
            with self._lock:
                waiter = self._inflight.get(key)
                leader = waiter is None
                if leader:
                    waiter = self._inflight[key] = threading.Event()
            if not leader:
                waiter.wait()
                continue
            try:
                value = compute()
                if keep(value):
                    self.put(key, value)
                return value
            finally:
                with self._lock:
                    self._inflight.pop(key, None)
                waiter.set()

    def stats(self) -> dict[str, int | float | None]:
        with self._lock:
            return {
                "entries": len(self._items),
                "bytes": self.bytes,
                "max_bytes": self.max_bytes,
                "ttl_seconds": self.ttl_seconds,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }

    def _drop(self, key: str) -> None:
        _, blob = self._items.pop(key)
        self.bytes -= len(blob)
