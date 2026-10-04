"""Unit tests for src/ratelimit.py (uses a fake clock, no sleeping)."""
from src.ratelimit import QueryLimiter


class FakeClock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def test_allows_requests_under_the_limit():
    limiter = QueryLimiter(per_minute=3, daily_cap=100, clock=FakeClock())
    assert [limiter.check("a") for _ in range(3)] == [None, None, None]


def test_per_minute_limit_is_per_client():
    limiter = QueryLimiter(per_minute=1, daily_cap=100, clock=FakeClock())
    assert limiter.check("a") is None
    assert limiter.check("a") == "rate"
    assert limiter.check("b") is None


def test_window_slides_after_a_minute():
    clock = FakeClock()
    limiter = QueryLimiter(per_minute=1, daily_cap=100, clock=clock)
    assert limiter.check("a") is None
    assert limiter.check("a") == "rate"
    clock.now += 61
    assert limiter.check("a") is None


def test_daily_cap_applies_across_clients_and_resets_next_day():
    clock = FakeClock(now=86400 * 10)
    limiter = QueryLimiter(per_minute=100, daily_cap=2, clock=clock)
    assert limiter.check("a") is None
    assert limiter.check("b") is None
    assert limiter.check("c") == "daily"
    clock.now += 86400
    assert limiter.check("c") is None
