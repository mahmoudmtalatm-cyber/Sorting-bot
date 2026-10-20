from tgsorter.keypool import KeyPool, seconds_until_pacific_midnight


class Clock:
    def __init__(self): self.t = 1_000_000.0
    def __call__(self): return self.t


def make(keys=("k1", "k2", "k3"), **kw):
    c = Clock()
    return KeyPool([(None, k) for k in keys], clock=c, spacing_s=0, **kw), c


def test_two_strikes_exclude_then_cooldown_recovers():
    p, c = make()
    k1 = p.keys[0]
    assert p.record_failure(k1.id, 429) is False          # strike 1: still usable
    assert not p.is_excluded(k1.id)
    assert p.record_failure(k1.id, 429) is True           # strike 2: excluded
    assert p.is_excluded(k1.id) and p.state(k1.id).reason == "rate_limited"
    c.t += 61                                             # 60 s cooldown lapses lazily
    assert not p.is_excluded(k1.id)


def test_429_and_400_are_independent_streaks():
    p, _ = make()
    k = p.keys[0].id
    p.record_failure(k, 429)
    p.record_failure(k, 400)                              # different kind: must NOT combine into a trip
    assert not p.is_excluded(k)
    assert p.record_failure(k, 400) is True
    assert p.state(k).reason == "model_error"


def test_other_failure_breaks_streaks():
    p, _ = make()
    k = p.keys[0].id
    p.record_failure(k, 429)
    p.record_failure(k, 500)
    assert p.record_failure(k, 429) is False


def test_pick_next_cycles_and_skips_excluded():
    p, _ = make()
    k1, k2, k3 = p.keys
    assert p.pick_next(k1.id) == k2
    assert p.pick_next(k3.id) == k1                       # wraps
    p.mark_invalid(k2.id)
    assert p.pick_next(k1.id) == k3                       # skips excluded
    p.mark_invalid(k3.id)
    assert p.pick_next(k1.id) == k2                       # all others excluded: still returns next in line


def test_pick_next_none_with_one_key_or_rotation_off():
    p, _ = make(keys=("only",))
    assert p.pick_next(p.keys[0].id) is None
    p2, _ = make(smart_rotation=False)
    assert p2.pick_next(p2.keys[0].id) is None


def test_daily_quota_excludes_whole_project_group_until_pacific_midnight():
    c = Clock()
    p = KeyPool([("projA", "a1"), ("projA", "a2"), ("projB", "b1")], clock=c, spacing_s=0)
    a1, a2, b1 = p.keys
    assert p.record_failure(a1.id, 429, scope="day") is True
    assert p.is_excluded(a1.id) and p.is_excluded(a2.id)  # same project: shared quota
    assert not p.is_excluded(b1.id)
    c.t += 60 * 60                                        # 1 h later: still out (not the 60 s cooldown)
    assert p.is_excluded(a2.id)
    c.t += 25 * 3600
    assert not p.is_excluded(a1.id)
    assert 0 < seconds_until_pacific_midnight(c.t) <= 86400 + 5


def test_retry_after_extends_cooldown():
    p, c = make()
    k = p.keys[0].id
    p.record_failure(k, 429, retry_after=120)
    p.record_failure(k, 429, retry_after=120)
    c.t += 90
    assert p.is_excluded(k)                               # Google said 120 s, not our default 60
    c.t += 40
    assert not p.is_excluded(k)


def test_acquire_prefers_active_and_spreads_by_pacing():
    c = Clock()
    p = KeyPool([(None, "a"), (None, "b")], clock=c, spacing_s=6.5)
    k1, k2 = p.keys
    key, wait = p.acquire(); assert key == k1 and wait == 0
    key, wait = p.acquire(); assert key == k2 and wait == 0   # k1 is paced -> use k2 with no wait
    key, wait = p.acquire(); assert wait > 0                  # both paced now: shortest wait


def test_acquire_all_excluded_reports_recovery_time():
    p, c = make(keys=("a", "b"))
    for k in p.keys:
        p.record_failure(k.id, 429); p.record_failure(k.id, 429)
    key, wait = p.acquire()
    assert key is None and 0 < wait <= 60
    p.mark_invalid(p.keys[0].id); p.mark_invalid(p.keys[1].id)
    assert p.acquire()[1] == float("inf")


def test_forgive_clears_model_error_exclusion():
    p, _ = make()
    k = p.keys[0].id
    p.record_failure(k, 400); p.record_failure(k, 400)
    assert p.is_excluded(k)
    p.forgive(k)
    assert not p.is_excluded(k)


def test_success_resets_everything():
    p, _ = make()
    k = p.keys[0].id
    p.record_failure(k, 429); p.record_failure(k, 429)
    p.record_success(k)
    assert not p.is_excluded(k)
