"""
backend/dashboard_routes.py

Purpose (Day 26, step 3): the landing page after login.

  GET /              your dashboard: summary cards, streak tiles, weekly goal,
                     most recent sessions
  GET  /goals        set or change your weekly session goal
  POST /goals        save it (then back to the dashboard)
  POST /goals/clear  remove it

Design
------
* Server-rendered. The page is built as HTML on the server from two database
  queries (db_store.get_summary / get_recent_sessions), so there is no extra
  JSON API to secure and nothing that can fail half-way in the browser.
* Tap-to-expand uses the native <details>/<summary> element: no JavaScript.
* Only the logged-in user's rows are ever queried (user id comes from the
  login cookie, never from the URL), so one user cannot see another's data.
* Every piece of stored text (session name, feedback) is HTML-escaped.
* Small scripts convert UTC timestamps to the viewer's local time and send the
  phone's UTC offset in a cookie, so streaks count LOCAL calendar days (see
  streaks.py for the rules).
* `Cache-Control: no-store` makes the browser re-fetch the page every time, so
  after a capture the dashboard is never a stale cached copy.

Look: same dark green-mono style as the login pages (shares their base CSS).
"""

import html
from datetime import datetime, timezone
from statistics import mean

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

import db_store
import streaks
from auth_routes import _CSS as BASE_CSS, require_user_page

router = APIRouter()

RECENT_LIMIT = 10
TZ_COOKIE = "tz_offset_min"        # minutes east of UTC, set by the page's script
NO_STORE = {"Cache-Control": "no-store"}


def _utc_now() -> datetime:
    """The current time. A function (not inline) so tests can freeze the clock."""
    return datetime.now(timezone.utc)

_DASH_CSS = """
:root{--orange:#ff8c00;--yellow:#ffe600}
.wrap{max-width:520px}
.topbar{display:flex;justify-content:space-between;align-items:center;
        font-size:14px;color:var(--muted)}
.topbar b{color:var(--cyan)}
.topbar form{margin:0}
.logoutBtn,.ghostBtn{width:auto;height:auto;min-height:40px;margin:0;padding:10px 16px;
           background:none;color:var(--red);border:1px solid var(--red);
           border-radius:20px;font-size:14px;font-weight:normal}
.cta{display:block;margin:18px 0;height:54px;line-height:54px;text-align:center;
     background:var(--green);color:#000;border-radius:27px;font-weight:bold;
     font-size:1.05rem;text-decoration:none}
.cta:active{opacity:.8}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:8px 0 10px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;
      padding:12px 6px;text-align:center}
.card .num{display:block;color:var(--green);font-size:1.7rem;font-weight:bold}
.card .lbl{display:block;color:var(--muted);font-size:.78rem;margin-top:2px}
h2{color:var(--cyan);font-size:1.1rem;margin:0 0 6px}
.empty{border:1px dashed var(--border);border-radius:10px;padding:18px;
       color:var(--muted);text-align:center}
details.session{background:var(--panel);border:1px solid var(--border);
                border-radius:10px;margin:10px 0}
details.session summary{list-style:none;cursor:pointer;padding:12px 14px;position:relative}
details.session summary::-webkit-details-marker{display:none}
details.session summary::after{content:"\\25BE";position:absolute;right:14px;top:12px;
                               color:var(--muted)}
details.session[open] summary::after{content:"\\25B4"}
.sname{color:#e6e6e6;font-weight:bold;word-break:break-word;padding-right:24px}
.stime{color:var(--muted);font-size:.85rem;margin-top:2px}
.chips{margin-top:8px}
.chip{display:inline-block;border:1px solid var(--border);border-radius:14px;
      padding:3px 10px;margin:0 6px 6px 0;font-size:.82rem;color:var(--green)}
.chip.dim{color:var(--muted)}
.setlist{border-top:1px solid var(--border);padding:4px 14px 12px}
.set{padding:10px 0;border-bottom:1px solid #222}
.set:last-child{border-bottom:0}
.sethead{display:flex;justify-content:space-between;align-items:center}
.ex{color:var(--cyan);font-weight:bold}
.q{font-weight:bold}
.q-good{color:var(--green)} .q-ok{color:var(--yellow)} .q-low{color:var(--orange)}
.meta{color:var(--muted);font-size:.85rem;margin-top:2px}
.fb{margin-top:4px;font-size:.9rem}
.more{color:var(--muted);text-align:center;font-size:.85rem;margin-top:10px}
.streakNote{color:var(--muted);font-size:.85rem;text-align:center;margin:2px 0 24px}
.goalCard{background:var(--panel);border:1px solid var(--border);border-radius:10px;
          padding:14px;margin:0 0 24px}
.goalHead{display:flex;justify-content:space-between;align-items:center;
          color:var(--cyan);font-weight:bold}
.goalHead a{color:var(--muted);font-weight:normal;font-size:.85rem}
.goalNums{margin:8px 0;color:#e6e6e6}
.goalNums b{color:var(--green);font-size:1.5rem}
.bar{height:12px;background:#222;border-radius:6px;overflow:hidden}
.fill{height:100%;background:var(--green)}
.goalNote{color:var(--muted);font-size:.85rem;margin-top:8px}
.cta.small{height:46px;line-height:46px;font-size:1rem;margin:12px 0 0}
.navLink{color:var(--cyan);text-decoration:none;display:inline-block;padding:8px 0}
.pills{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.pill{width:auto;height:44px;margin:0;padding:0 18px;background:none;color:var(--green);
      border:1px solid var(--border);border-radius:22px;font-size:1rem;font-weight:normal}
.pill:active{opacity:1;border-color:var(--green)}
.goalForm{margin-top:8px}
.danger{margin-top:28px;text-align:center}
"""

# Turns "2026-09-20T09:11:41+00:00" (UTC) into the viewer's local time.
_LOCAL_TIME_JS = """<script>
document.querySelectorAll('time[data-local]').forEach(function (t) {
  var d = new Date(t.getAttribute('datetime'));
  if (!isNaN(d)) {
    t.textContent = d.toLocaleString(undefined,
      {day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'});
  }
});
</script>"""

# Tells the server which calendar day it is for THIS phone. Runs before the
# first render decision: on the first visit (no cookie yet) it sets the cookie
# and reloads once; afterwards the cookie matches and nothing happens. If
# cookies are blocked the cookie cannot be read back, so it never reloads.
_TZ_JS = """<script>
(function () {
  var offset = -new Date().getTimezoneOffset();      // minutes east of UTC (IST = 330)
  var m = document.cookie.match(/(?:^|; )tz_offset_min=(-?\\d+)/);
  if (m && parseInt(m[1], 10) === offset) return;
  document.cookie = 'tz_offset_min=' + offset + '; path=/; max-age=31536000; SameSite=Lax';
  if (document.cookie.indexOf('tz_offset_min=' + offset) !== -1) location.reload();
})();
</script>"""


# ================================================================== helpers ==
def _esc(value) -> str:
    return html.escape(str(value))


def _pretty_label(label: str) -> str:
    """'dumbbell_shoulder_press' -> 'Dumbbell Shoulder Press'."""
    return _esc(label.replace("_", " ").title())


def _reps(value) -> str:
    return "-" if value is None else str(int(round(value)))


def _quality_class(score) -> str:
    if score is None:
        return ""
    if score >= 80:
        return "q-good"
    return "q-ok" if score >= 60 else "q-low"


def _time_tag(iso_utc: str) -> str:
    """<time> whose text is a UTC fallback; the script swaps in local time."""
    fallback = iso_utc[:16].replace("T", " ") + " UTC"
    return f'<time datetime="{_esc(iso_utc)}" data-local>{_esc(fallback)}</time>'


# =============================================================== HTML pieces ==
def _cards_html(summary: dict) -> str:
    avg = summary["avg_quality"]
    avg_text = "-" if avg is None else f"{avg:.0f}"
    return f"""
<div class="cards">
  <div class="card"><span class="num">{summary["sessions"]}</span><span class="lbl">Sessions</span></div>
  <div class="card"><span class="num">{summary["total_reps"]}</span><span class="lbl">Total reps</span></div>
  <div class="card"><span class="num">{avg_text}</span><span class="lbl">Avg quality</span></div>
</div>"""


def _streak_html(stats: dict) -> str:
    if stats["active_today"]:
        note = "You have trained today. Keep it going tomorrow!"
    elif stats["current"] > 0:
        note = f'Exercise today to keep your {stats["current"]}-day streak alive.'
    else:
        note = "Record a session today to start a streak."
    return f"""
<div class="cards">
  <div class="card"><span class="num">{stats["current"]}</span><span class="lbl">Day streak</span></div>
  <div class="card"><span class="num">{stats["longest"]}</span><span class="lbl">Best streak</span></div>
  <div class="card"><span class="num">{stats["week_sessions"]}</span><span class="lbl">This week</span></div>
</div>
<div class="streakNote">{_esc(note)}</div>"""


def _goal_html(goal: dict) -> str:
    """Weekly-goal card: progress bar, or a prompt to set a goal."""
    target = goal["target"]
    if target is None:
        return """
<div class="goalCard">
  <div class="goalHead"><span>Weekly goal</span></div>
  <div class="goalNote">Set a weekly session goal to track your progress.</div>
  <a class="cta small" href="/goals">Set a weekly goal</a>
</div>"""

    done = goal["done"]
    percent = min(100, int(round(done * 100 / target)))
    if done >= target:
        note = "Goal reached. Great work!"
    else:
        days_left = goal["days_left"]
        note = (f"{target - done} more to go &middot; "
                f"{days_left} day{'s' if days_left != 1 else ''} left this week")
    return f"""
<div class="goalCard">
  <div class="goalHead"><span>Weekly goal</span><a href="/goals">Edit</a></div>
  <div class="goalNums"><b>{done}</b> / {target} sessions</div>
  <div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100"
       aria-valuenow="{percent}"><div class="fill" style="width:{percent}%"></div></div>
  <div class="goalNote">{note}</div>
</div>"""


def _shell(title: str, body: str, scripts: str = "") -> str:
    """The page wrapper shared by the dashboard and the goals page."""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="theme-color" content="#0d0d0d">'
        f"<title>{_esc(title)}</title><style>{BASE_CSS}{_DASH_CSS}</style></head>"
        f'<body><div class="wrap">{body}</div>{scripts}</body></html>'
    )


_PILL_JS = """<script>
document.querySelectorAll('.pill').forEach(function (b) {
  b.addEventListener('click', function () {
    document.getElementById('target').value = b.dataset.value;
  });
});
</script>"""


def render_goals_page(current: int | None, error: str = "") -> str:
    """The page where the user sets their weekly session goal."""
    error_html = f'<div class="error">{_esc(error)}</div>' if error else ""
    value = "" if current is None else str(current)
    pills = "".join(f'<button type="button" class="pill" data-value="{n}">{n}</button>'
                    for n in (2, 3, 4, 5, 7))
    remove = ("""
<form method="post" action="/goals/clear" class="danger">
  <button type="submit" class="ghostBtn">Remove goal</button>
</form>""" if current is not None else "")
    body = f"""
<a class="navLink" href="/">&larr; Dashboard</a>
<h1>Weekly goal</h1>
<p class="sub">How many sessions do you want to complete each week (Monday to Sunday)?</p>
{error_html}
<form method="post" action="/goals" class="goalForm">
  <label for="target">Sessions per week</label>
  <input id="target" name="target" type="number" inputmode="numeric"
         min="1" max="{db_store.MAX_WEEKLY_GOAL}" value="{_esc(value)}" required>
  <div class="pills">{pills}</div>
  <button type="submit">Save goal</button>
</form>{remove}"""
    return _shell("Weekly goal - Rehab Planner", body, _PILL_JS)


def _set_html(st: dict) -> str:
    score = st["quality_score"]
    score_html = ('<span class="q">-</span>' if score is None else
                  f'<span class="q {_quality_class(score)}">{score:.0f}</span>')
    feedback = f'<div class="fb">{_esc(st["feedback"])}</div>' if st["feedback"] else ""
    return f"""
<div class="set">
  <div class="sethead"><span class="ex">{_pretty_label(st["label"])}</span>{score_html}</div>
  <div class="meta">Reps: {_reps(st["estimated_reps"])} &middot; Duration: {st["duration_s"]:.0f}s</div>
  {feedback}
</div>"""


def _session_html(session: dict) -> str:
    reps_by_exercise: dict[str, float] = {}
    scores = []
    for st in session["sets"]:
        reps_by_exercise[st["label"]] = (
            reps_by_exercise.get(st["label"], 0) + (st["estimated_reps"] or 0)
        )
        if st["quality_score"] is not None:
            scores.append(st["quality_score"])

    chips = "".join(
        f'<span class="chip">{_pretty_label(label)} &times;{int(round(reps))}</span>'
        for label, reps in reps_by_exercise.items()
    )
    if scores:
        avg = mean(scores)
        chips += f'<span class="chip {_quality_class(avg)}">Quality {avg:.0f}</span>'
    if not chips:
        chips = '<span class="chip dim">No exercise detected</span>'

    if session["sets"]:
        body = "".join(_set_html(st) for st in session["sets"])
    else:
        body = '<div class="meta">No exercise sets were detected in this recording.</div>'

    return f"""
<details class="session">
  <summary>
    <div class="sname">{_esc(session["source_name"])}</div>
    <div class="stime">{_time_tag(session["uploaded_at"])}</div>
    <div class="chips">{chips}</div>
  </summary>
  <div class="setlist">{body}</div>
</details>"""


def render_dashboard(username: str, summary: dict, streak: dict, goal: dict,
                     recent: list[dict]) -> str:
    """Build the complete dashboard page (pure function: easy to test)."""
    if recent:
        sessions_html = "".join(_session_html(s) for s in recent)
        if summary["sessions"] > len(recent):
            sessions_html += (f'<div class="more">Showing your latest {len(recent)} '
                              f'of {summary["sessions"]} sessions</div>')
    else:
        sessions_html = ('<div class="empty">No sessions yet.<br>'
                         'Tap "Record a session" to capture your first workout.</div>')

    body = f"""
<div class="topbar">
  <span>Logged in as <b>{_esc(username)}</b></span>
  <form method="post" action="/logout"><button type="submit" class="logoutBtn">Logout</button></form>
</div>
<h1>Your progress</h1>
<a class="cta" href="/capture">&#9654; Record a session</a>
{_cards_html(summary)}
{_streak_html(streak)}
{_goal_html(goal)}
<h2>Recent sessions</h2>
{sessions_html}"""
    return _shell("Dashboard - Rehab Planner", body, _TZ_JS + _LOCAL_TIME_JS)


# ================================================================== routes ==
@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request, user: dict = Depends(require_user_page)):
    summary = db_store.get_summary(user["id"])
    recent = db_store.get_recent_sessions(user["id"], limit=RECENT_LIMIT)

    # Streaks count the user's LOCAL calendar days (offset comes from the cookie).
    offset = streaks.parse_offset(request.cookies.get(TZ_COOKIE))
    today = streaks.local_today(_utc_now(), offset)
    session_days = [streaks.to_local_date(t, offset)
                    for t in db_store.get_exercise_session_times(user["id"])]
    streak = streaks.streak_stats(session_days, today)

    # Goal progress uses the same "sessions this week" number as the streak tiles.
    goal = {
        "target": db_store.get_weekly_goal(user["id"]),
        "done": streak["week_sessions"],
        "days_left": 7 - today.weekday(),          # including today; Monday = 7
    }

    return HTMLResponse(
        render_dashboard(user["username"], summary, streak, goal, recent),
        headers=NO_STORE,
    )


@router.get("/goals", response_class=HTMLResponse)
def goals_page(user: dict = Depends(require_user_page)):
    return HTMLResponse(
        render_goals_page(db_store.get_weekly_goal(user["id"])), headers=NO_STORE
    )


@router.post("/goals")
def goals_save(target: str = Form(""), user: dict = Depends(require_user_page)):
    try:
        db_store.set_weekly_goal(user["id"], int(target.strip()))
    except ValueError:               # not a number, or outside 1..MAX_WEEKLY_GOAL
        return HTMLResponse(
            render_goals_page(
                db_store.get_weekly_goal(user["id"]),
                error=f"Enter a whole number from 1 to {db_store.MAX_WEEKLY_GOAL}.",
            ),
            status_code=400, headers=NO_STORE,
        )
    return RedirectResponse("/", status_code=303)      # show the new progress bar


@router.post("/goals/clear")
def goals_clear(user: dict = Depends(require_user_page)):
    db_store.clear_weekly_goal(user["id"])
    return RedirectResponse("/", status_code=303)