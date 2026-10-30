"""Per-IP rate limiting for new assessments.

A live demo on a public URL spends real money: every uncached assessment is a web search
plus two or three model calls, roughly $0.07. A cached one costs nothing, so only new
work is counted and a visitor can replay the demo sites as often as they like.

In-memory and per-container, which is honest about what it is: the deployment pins itself
to one container, so this is the whole picture there. It is a spending guard, not a
security control.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

NEW_ASSESSMENTS_PER_HOUR = 10
WINDOW_S = 3600


class RateLimit:
    def __init__(self, limit: int = NEW_ASSESSMENTS_PER_HOUR, window_s: int = WINDOW_S):
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _recent(self, key: str, now: float) -> deque[float]:
        hits = self._hits[key]
        while hits and now - hits[0] > self.window_s:
            hits.popleft()
        return hits

    def check(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Is this caller allowed another new assessment? Returns (allowed, seconds to wait)."""
        now = time.time() if now is None else now
        hits = self._recent(key, now)
        if len(hits) < self.limit:
            return True, 0
        return False, int(self.window_s - (now - hits[0])) + 1

    def record(self, key: str, now: float | None = None) -> None:
        """Count one new assessment. Cached replays never call this."""
        now = time.time() if now is None else now
        self._recent(key, now).append(now)

    def remaining(self, key: str, now: float | None = None) -> int:
        now = time.time() if now is None else now
        return max(0, self.limit - len(self._recent(key, now)))


def client_ip(request) -> str:
    """The caller's address, trusting the first hop of X-Forwarded-For.

    Behind Modal's proxy the socket address is the proxy, so the header is the only
    signal available. A determined caller can forge it; that is acceptable for a spending
    guard, and is the reason this is not presented as a security control.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return getattr(getattr(request, "client", None), "host", "") or "unknown"
