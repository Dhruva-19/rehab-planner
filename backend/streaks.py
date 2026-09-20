"""
backend/streaks.py

Purpose (Day 26, step 4): the counting logic behind the dashboard's streak
tiles. Pure functions only -- no database, no web framework -- so every rule
below is tested with made-up dates in scripts/test_streaks.py.

Definitions
-----------
* An "active day" is a LOCAL calendar day on which the user saved at least one
  session containing a real exercise set (rest-only recordings do not count).
* Current streak = how many consecutive active days end today. If today has no
  session yet, a streak that ended YESTERDAY is still alive: it only breaks
  once a whole day passes with nothing.
* Longest streak = the longest run of consecutive active days ever.
* Sessions this week = number of sessions (not days) from Monday to today.

Why local days
--------------
The database stores UTC. A workout at 11:30 pm in Mumbai (IST, UTC+5:30) is
already "tomorrow" in UTC, so counting UTC days would put it on the wrong
date. The dashboard page sends the phone's UTC offset in a cookie
(minutes east of UTC; IST = 330) and these helpers apply it. A fixed offset is
exact for places without daylight saving (India); elsewhere it can be off by
an hour around clock changes, which only matters for sessions near midnight.
"""

from datetime import date, datetime, timedelta, timezone

MIN_OFFSET_MIN = -12 * 60      # UTC-12:00
MAX_OFFSET_MIN = 14 * 60       # UTC+14:00


def parse_offset(cookie_value: str | None) -> int:
    """Minutes east of UTC from the cookie. Missing or invalid -> 0 (UTC)."""
    try:
        minutes = int(cookie_value)
    except (TypeError, ValueError):
        return 0
    return minutes if MIN_OFFSET_MIN <= minutes <= MAX_OFFSET_MIN else 0


def to_local_date(iso_utc: str, offset_min: int) -> date:
    """'2026-09-20T19:00:00+00:00' with offset 330 -> date(2026, 9, 21)."""
    moment = datetime.fromisoformat(iso_utc)
    if moment.tzinfo is None:                       # stored values are UTC
        moment = moment.replace(tzinfo=timezone.utc)
    utc_moment = moment.astimezone(timezone.utc)
    return (utc_moment + timedelta(minutes=offset_min)).date()


def local_today(now_utc: datetime, offset_min: int) -> date:
    return (now_utc + timedelta(minutes=offset_min)).date()


def streak_stats(session_days: list[date], today: date) -> dict:
    """
    session_days: one local date per exercise session (a day with two sessions
    appears twice). Returns:
        current        consecutive active days ending today (or yesterday)
        longest        longest run of consecutive active days
        week_sessions  sessions from this week's Monday up to today
        active_today   whether today already has a session
    """
    # A date after "today" can only come from a clock/time-zone change; ignore it.
    session_days = [d for d in session_days if d <= today]
    active = set(session_days)
    one_day = timedelta(days=1)

    # --- current streak: anchor on today, else yesterday, then walk backwards
    if today in active:
        cursor = today
    elif (today - one_day) in active:
        cursor = today - one_day
    else:
        cursor = None
    current = 0
    while cursor is not None and cursor in active:
        current += 1
        cursor -= one_day

    # --- longest streak: scan the sorted active days for consecutive runs
    longest = run = 0
    previous = None
    for day in sorted(active):
        run = run + 1 if previous is not None and day - previous == one_day else 1
        longest = max(longest, run)
        previous = day

    # --- sessions this week (Monday = weekday 0)
    week_start = today - timedelta(days=today.weekday())
    week_sessions = sum(1 for d in session_days if d >= week_start)

    return {
        "current": current,
        "longest": longest,
        "week_sessions": week_sessions,
        "active_today": today in active,
    }
