from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_at: datetime


class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, list[datetime]] = defaultdict(list)

    def check(self, key: str) -> RateLimitResult:
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)

        self._requests[key] = [
            ts for ts in self._requests[key] if ts > minute_ago
        ]

        remaining = max(0, self.requests_per_minute - len(self._requests[key]))

        if len(self._requests[key]) >= self.requests_per_minute:
            reset_at = self._requests[key][0] + timedelta(minutes=1)
            return RateLimitResult(allowed=False, remaining=0, reset_at=reset_at)

        self._requests[key].append(now)
        reset_at = now + timedelta(minutes=1)
        return RateLimitResult(allowed=True, remaining=remaining - 1, reset_at=reset_at)

    def reset(self, key: str) -> None:
        if key in self._requests:
            del self._requests[key]
