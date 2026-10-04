"""
Small in-memory request limiter used by the public API and the Streamlit UI.

Every question triggers a paid OpenAI call, so a public deployment needs a
cost backstop. Two limits are applied:

- a per-client limit (requests per minute), and
- a global daily cap shared by all clients (the real cost ceiling: the
  per-client key comes from a proxy header and can be spoofed).

State lives in process memory: it resets on restart and is not shared
between instances, which is acceptable for a single free-tier instance.
"""
import threading
import time
from collections import defaultdict, deque


class QueryLimiter:
    def __init__(self, per_minute: int, daily_cap: int, clock=time.time):
        self.per_minute = per_minute
        self.daily_cap = daily_cap
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: dict[str, deque] = defaultdict(deque)
        self._day = self._today()
        self._day_count = 0

    def _today(self) -> int:
        return int(self._clock() // 86400)

    def check(self, client: str) -> str | None:
        """Registers one request. Returns None if allowed, otherwise the
        reason it was refused ('rate' or 'daily')."""
        now = self._clock()
        with self._lock:
            if self._today() != self._day:
                self._day = self._today()
                self._day_count = 0
            if self._day_count >= self.daily_cap:
                return "daily"
            window = self._recent[client]
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= self.per_minute:
                return "rate"
            window.append(now)
            self._day_count += 1
            return None
