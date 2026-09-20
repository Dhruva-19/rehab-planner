"""
backend/dashboard_routes.py

Purpose (Day 26, step 3): the landing page after login.

  GET /   your dashboard: summary cards + your most recent sessions

Design
------
* Server-rendered. The page is built as HTML on the server from two database
  queries (db_store.get_summary / get_recent_sessions), so there is no extra
  JSON API to secure and nothing that can fail half-way in the browser.
* Tap-to-expand uses the native <details>/<summary> element: no JavaScript.
* Only the logged-in user's rows are ever queried (user id comes from the
  login cookie, never from the URL), so one user cannot see another's data.
* Every piece of stored text (session name, feedback) is HTML-escaped.
* The one small script converts UTC timestamps to the viewer's local time.
* `Cache-Control: no-store` makes the browser re-fetch the page every time, so
  after a capture the dashboard is never a stale cached copy.

Look: same dark green-mono style as the login pages (shares their base CSS).
"""

import html
from statistics import mean

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

import db_store
from auth_routes import _CSS as BASE_CSS, require_user_page

router = APIRouter()

RECENT_LIMIT = 10

_DASH_CSS = """
:root{--orange:#ff8c00;--yellow:#ffe600}
.wrap{max-width:520px}
.topbar{display:flex;justify-content:space-between;align-items:center;
        font-size:14px;color:var(--muted)}
.topbar b{color:var(--cyan)}
.topbar form{margin:0}
.logoutBtn{width:auto;height:auto;min-height:40px;margin:0;padding:10px 16px;
           background:none;color:var(--red);border:1px solid var(--red);
           border-radius:20px;font-size:14px;font-weight:normal}
.cta{display:block;margin:18px 0;height:54px;line-height:54px;text-align:center;
     background:var(--green);color:#000;border-radius:27px;font-weight:bold;
     font-size:1.05rem;text-decoration:none}
.cta:active{opacity:.8}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:8px 0 24px}
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


def render_dashboard(username: str, summary: dict, recent: list[dict]) -> str:
    """Build the complete dashboard page (pure function: easy to test)."""
    if recent:
        sessions_html = "".join(_session_html(s) for s in recent)
        if summary["sessions"] > len(recent):
            sessions_html += (f'<div class="more">Showing your latest {len(recent)} '
                              f'of {summary["sessions"]} sessions</div>')
    else:
        sessions_html = ('<div class="empty">No sessions yet.<br>'
                         'Tap "Record a session" to capture your first workout.</div>')

    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="theme-color" content="#0d0d0d">'
        f"<title>Dashboard - Rehab Planner</title><style>{BASE_CSS}{_DASH_CSS}</style></head>"
        f"""<body><div class="wrap">
<div class="topbar">
  <span>Logged in as <b>{_esc(username)}</b></span>
  <form method="post" action="/logout"><button type="submit" class="logoutBtn">Logout</button></form>
</div>
<h1>Your progress</h1>
<a class="cta" href="/capture">&#9654; Record a session</a>
{_cards_html(summary)}
<h2>Recent sessions</h2>
{sessions_html}
</div>{_LOCAL_TIME_JS}</body></html>"""
    )


# ================================================================== routes ==
@router.get("/", response_class=HTMLResponse)
def dashboard(user: dict = Depends(require_user_page)):
    summary = db_store.get_summary(user["id"])
    recent = db_store.get_recent_sessions(user["id"], limit=RECENT_LIMIT)
    return HTMLResponse(
        render_dashboard(user["username"], summary, recent),
        headers={"Cache-Control": "no-store"},
    )
