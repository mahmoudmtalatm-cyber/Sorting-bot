"""
Key pool + rotation state. Python port of js/api-rotation.js, with the
differences that matter for a headless bot:

  Same as the web app
    * 2 consecutive failures of ONE kind (429, or plain 400) exclude a key;
      the two kinds are separate streaks, labelled differently in status.
    * Excluded keys come back automatically after a cooldown (lazy expiry).
    * 401/403/API_KEY_INVALID marks a key invalid for good.
    * pick_next() walks forward from the current key (1->2->3->1...) preferring
      healthy ones, and still returns the next key when ALL are excluded.
    * SMART_ROTATION=0 disables switching entirely.
    * State is in-memory and keyed by key id, never by position.

  New
    * Quotas are per Google Cloud PROJECT. Keys can be tagged with a group;
      a daily-quota 429 excludes the whole group, not just one key.
    * A 429 whose body says "per day" excludes until midnight Pacific (when
      Gemini's daily quota resets) instead of a pointless 60 s cooldown.
    * Per-key pacing (min spacing between requests) instead of one global
      gate, so N healthy keys give ~N x the throughput.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")


def seconds_until_pacific_midnight(now_ts: float | None = None) -> float:
    now = datetime.fromtimestamp(now_ts if now_ts is not None else time.time(), PACIFIC)
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
    return max(1.0, (nxt - now).total_seconds())


@dataclass(frozen=True)
class ApiKey:
    id: str
    key: str
    group: str
    label: str  # safe to log/show: never the key itself


@dataclass
class KeyState:
    consecutive429: int = 0
    consecutive400: int = 0
    excluded_until: float = 0.0
    reason: str | None = None        # rate_limited | daily_quota | model_error
    invalid: bool = False
    next_allowed_at: float = 0.0     # pacing
    model: str | None = None         # None = primary; set when this key needed the fallback


class KeyPool:
    def __init__(
        self,
        keys: list[tuple[str | None, str]],
        *,
        smart_rotation: bool = True,
        spacing_s: float = 6.5,
        cooldown_s: float = 60.0,
        threshold: int = 2,
        clock: Callable[[], float] = time.time,
    ):
        self.clock = clock
        self.smart_rotation = smart_rotation
        self.spacing_s = spacing_s
        self.cooldown_s = cooldown_s
        self.threshold = threshold
        self.keys: list[ApiKey] = []
        self._st: dict[str, KeyState] = {}
        for group, k in keys:
            kid = hashlib.sha256(k.encode()).hexdigest()[:8]
            self.keys.append(ApiKey(kid, k, group or kid, "..." + k[-4:]))
            self._st[kid] = KeyState()
        self._by_id = {k.id: k for k in self.keys}
        self.active_id: str | None = self.keys[0].id if self.keys else None

    # ── state helpers ──────────────────────────────────────
    def state(self, kid: str) -> KeyState:
        return self._st[kid]

    def is_excluded(self, kid: str) -> bool:
        st = self._st[kid]
        if st.invalid:
            return True
        if st.excluded_until:
            if self.clock() >= st.excluded_until:  # cooldown lapsed: lazy recovery
                st.excluded_until, st.reason = 0.0, None
                st.consecutive429 = st.consecutive400 = 0
                return False
            return True
        return False

    def all_excluded(self) -> bool:
        return bool(self.keys) and all(self.is_excluded(k.id) for k in self.keys)

    def usable_count(self) -> int:
        return sum(1 for k in self.keys if not self._st[k.id].invalid)

    # ── outcomes ───────────────────────────────────────────
    def record_success(self, kid: str) -> None:
        st = self._st[kid]
        st.consecutive429 = st.consecutive400 = 0
        st.excluded_until, st.reason, st.invalid = 0.0, None, False

    def record_failure(self, kid: str, status: int, *, scope: str = "minute",
                       retry_after: float | None = None) -> bool:
        """Returns True if this failure just tipped the key into 'excluded'."""
        st = self._st[kid]
        now = self.clock()
        if status == 429:
            if scope == "day":
                until = now + seconds_until_pacific_midnight(now)
                grp = self._by_id[kid].group
                for k in self.keys:               # quota is per project -> whole group
                    if k.group == grp:
                        s = self._st[k.id]
                        s.excluded_until, s.reason = until, "daily_quota"
                return True
            st.consecutive429 += 1
            if st.consecutive429 >= self.threshold and not st.excluded_until:
                st.excluded_until = now + max(self.cooldown_s, (retry_after or 0) + 1)
                st.reason = "rate_limited"
                return True
            return False
        if status == 400:
            st.consecutive400 += 1
            if st.consecutive400 >= self.threshold and not st.excluded_until:
                st.excluded_until = now + self.cooldown_s
                st.reason = "model_error"
                return True
            return False
        st.consecutive429 = st.consecutive400 = 0   # any other failure breaks both streaks
        return False

    def mark_invalid(self, kid: str) -> None:
        self._st[kid].invalid = True

    def forgive(self, kid: str) -> None:
        """Undo 400-streak punishment (the *request* was bad, not the key)."""
        st = self._st[kid]
        st.consecutive400 = 0
        if st.reason == "model_error":
            st.excluded_until, st.reason = 0.0, None

    # ── selection ──────────────────────────────────────────
    def pick_next(self, current_id: str | None) -> ApiKey | None:
        """Direct port of pickNextApiKey()."""
        if not self.smart_rotation or len(self.keys) < 2:
            return None
        idx = max(0, next((i for i, k in enumerate(self.keys) if k.id == current_id), 0))
        ordered = [k for k in self.keys[idx + 1:] + self.keys[: idx + 1] if k.id != current_id]
        healthy = next((k for k in ordered if not self.is_excluded(k.id)), None)
        return healthy or (ordered[0] if ordered else None)

    def seconds_until_available(self) -> float:
        now = self.clock()
        waits = [max(0.0, self._st[k.id].excluded_until - now)
                 for k in self.keys if not self._st[k.id].invalid]
        return min(waits) if waits else float("inf")

    def acquire(self) -> tuple[ApiKey | None, float]:
        """
        Pick a key and reserve its next request slot. Returns (key, seconds_to_wait),
        or (None, seconds_until_some_key_recovers) when everything is excluded.
        Prefers the active key; switches only if another has a strictly shorter wait.
        """
        now = self.clock()
        if not self.keys:
            return None, float("inf")
        if not self.smart_rotation:
            active = self._by_id[self.active_id]
            cands = [] if self._st[active.id].invalid else [active]
        else:
            cands = [k for k in self.keys if not self.is_excluded(k.id)]
        if not cands:
            return None, self.seconds_until_available()
        n = len(self.keys)
        a_idx = next(i for i, k in enumerate(self.keys) if k.id == self.active_id)

        def rank(k: ApiKey):
            wait = max(0.0, self._st[k.id].next_allowed_at - now)
            return (round(wait, 3), (self.keys.index(k) - a_idx) % n)

        best = min(cands, key=rank)
        st = self._st[best.id]
        wait = max(0.0, st.next_allowed_at - now)
        st.next_allowed_at = now + wait + self.spacing_s
        self.active_id = best.id
        return best, wait

    # ── UI ─────────────────────────────────────────────────
    def status_rows(self) -> list[dict]:
        now = self.clock()
        rows = []
        for k in self.keys:
            st = self._st[k.id]
            if st.invalid:
                state, left = "invalid", 0
            elif self.is_excluded(k.id):
                state, left = st.reason or "excluded", max(0, int(st.excluded_until - now))
            else:
                state, left = "ok", 0
            rows.append({"label": k.label, "group": k.group, "state": state, "retry_in_s": left,
                         "model": st.model or "primary", "active": k.id == self.active_id})
        return rows
