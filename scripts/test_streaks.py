"""
scripts/test_streaks.py

Purpose: test the streak rules with made-up dates. No database, no server.

Run from the project root, inside your fastapi-env:
    python scripts/test_streaks.py
"""

import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from streaks import local_today, parse_offset, streak_stats, to_local_date

TODAY = date(2026, 9, 23)          # a Wednesday; this week's Monday is Sept 21


def days(*day_numbers):
    """days(20, 21) -> [date(2026, 9, 20), date(2026, 9, 21)]"""
    return [date(2026, 9, d) for d in day_numbers]


# --- current streak ------------------------------------------------------------
s = streak_stats([], TODAY)
assert s == {"current": 0, "longest": 0, "week_sessions": 0, "active_today": False}

s = streak_stats(days(23), TODAY)
assert s["current"] == 1 and s["active_today"] is True

s = streak_stats(days(21, 22, 23), TODAY)                    # three in a row, ending today
assert s["current"] == 3 and s["longest"] == 3

s = streak_stats(days(20, 21, 22), TODAY)                    # today not done YET: still alive
assert s["current"] == 3 and s["active_today"] is False

s = streak_stats(days(19, 20, 21), TODAY)                    # yesterday AND today missed: broken
assert s["current"] == 0 and s["longest"] == 3

s = streak_stats(days(15, 16, 17, 20, 22, 23), TODAY)        # gaps: current run is only 22-23
assert s["current"] == 2 and s["longest"] == 3
print("current/longest streak: OK")

# --- several sessions on one day: one active day, but each session counts for the week
s = streak_stats(days(22, 22, 22, 23), TODAY)
assert s["current"] == 2 and s["longest"] == 2 and s["week_sessions"] == 4
print("multiple sessions per day: OK")

# --- sessions this week (Monday to today) ----------------------------------------------
s = streak_stats(days(20, 21, 23), TODAY)                    # Sunday 20th is LAST week
assert s["week_sessions"] == 2
s = streak_stats(days(21), date(2026, 9, 27))                # Sunday: the week still counts Monday
assert s["week_sessions"] == 1
s = streak_stats(days(27), date(2026, 9, 28))                # new Monday: the week resets
assert s["week_sessions"] == 0 and s["current"] == 1         # ...but the streak continues
print("week boundaries: OK")

# --- dates in the future (clock or time-zone change) are ignored ------------------------------
s = streak_stats(days(23, 24, 25), TODAY)
assert s["current"] == 1 and s["longest"] == 1 and s["week_sessions"] == 1
print("future dates ignored: OK")

# --- time zones ---------------------------------------------------------------------------------
assert parse_offset("330") == 330 and parse_offset("-300") == -300
assert parse_offset(None) == 0 and parse_offset("abc") == 0 and parse_offset("") == 0
assert parse_offset("99999") == 0 and parse_offset("-9999") == 0         # nonsense -> UTC

# 19:00 UTC on the 20th is 00:30 on the 21st in India: a different calendar day.
assert to_local_date("2026-09-20T19:00:00+00:00", 0) == date(2026, 9, 20)
assert to_local_date("2026-09-20T19:00:00+00:00", 330) == date(2026, 9, 21)
assert to_local_date("2026-09-20T19:00:00+00:00", -300) == date(2026, 9, 20)
assert to_local_date("2026-09-20T19:00:00", 330) == date(2026, 9, 21)     # no zone stored -> UTC
assert to_local_date("2026-09-20T23:59:59.999999+00:00", 0) == date(2026, 9, 20)

now = datetime(2026, 9, 21, 20, 0, tzinfo=timezone.utc)
assert local_today(now, 0) == date(2026, 9, 21)
assert local_today(now, 330) == date(2026, 9, 22)                          # 01:30 IST next day
print("time zones: OK")

print("\nAll streak checks passed.")
