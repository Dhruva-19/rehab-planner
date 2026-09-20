"""
backend/auth_routes.py

Purpose (Day 26, step 2): everything about logging in, as a FastAPI router.

  GET  /register   sign-up page            POST /register   create account + log in
  GET  /login      sign-in page            POST /login      check password + log in
  POST /logout     end this login          GET  /api/me     who am I? (JSON)

It also exports two "guards" other routes can depend on:
  require_user_page  -> for HTML pages: not logged in => redirect to /login
  require_user_api   -> for JSON routes (e.g. /ingest): not logged in => 401

How a login works
-----------------
1. The user submits the form; db_store.authenticate() checks the password hash.
2. db_store.create_login_token() makes a random token. Its HASH is stored in
   the database; the raw token is sent to the browser as an HttpOnly cookie
   (JavaScript on the page cannot read it, which limits XSS damage).
3. On every later request the cookie comes back, and get_user_by_token() maps
   it to a user.

Why plain `def` (not `async def`) handlers: password hashing is deliberately
slow CPU work. FastAPI runs plain `def` handlers in a thread pool, so one
person logging in does not freeze every other request.

Import note: like the rest of the backend, this expects main.py to have put
src/storage on sys.path, so `import db_store` works.
"""

import html
import os

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

import db_store
from auth_utils import MIN_PASSWORD_LENGTH

router = APIRouter()

COOKIE_NAME = "rehab_token"
COOKIE_MAX_AGE = db_store.TOKEN_LIFETIME_DAYS * 24 * 3600   # same as the DB expiry


def _cookie_secure() -> bool:
    """
    'Secure' cookies are only sent over HTTPS. Render serves HTTPS (and sets
    RENDER=true), so we switch it on there. Locally you use plain http://, where
    a Secure cookie would be silently dropped and login would appear broken.
    COOKIE_SECURE=true/false overrides the automatic choice.
    """
    flag = os.environ.get("COOKIE_SECURE", os.environ.get("RENDER", ""))
    return flag.strip().lower() in ("1", "true", "yes")


# =========================================================== guards (Depends) ==
def current_user_or_none(request: Request) -> dict | None:
    """Return {'id', 'username'} for the logged-in user, or None."""
    return db_store.get_user_by_token(request.cookies.get(COOKIE_NAME))


def require_user_page(user: dict | None = Depends(current_user_or_none)) -> dict:
    """Guard for HTML pages: send anonymous visitors to the login page."""
    if user is None:
        raise HTTPException(status_code=303, detail="Login required",
                            headers={"Location": "/login"})
    return user


def require_user_api(user: dict | None = Depends(current_user_or_none)) -> dict:
    """Guard for JSON routes: anonymous callers get 401 instead of a redirect."""
    if user is None:
        raise HTTPException(status_code=401, detail="Not logged in")
    return user


def _set_login_cookie(response, token: str) -> None:
    response.set_cookie(
        COOKIE_NAME, token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,          # not readable from page JavaScript
        samesite="lax",         # not sent on cross-site POSTs (basic CSRF defence)
        secure=_cookie_secure(),
        path="/",
    )


# ============================================================ HTML templates ==
# Look matches sensor_capture_test.html: near-black background, green title,
# cyan labels, monospace. Mobile-first: full-width fields, 48px+ touch targets,
# 16px input text (smaller text makes iPhones zoom in when a field is tapped).
_CSS = """
:root{--bg:#0d0d0d;--panel:#141414;--border:#333;--green:#00ff41;
      --cyan:#00e5ff;--text:#bdbdbd;--muted:#777;--red:#ff5252}
*{box-sizing:border-box}
body{margin:0;padding:28px 16px;background:var(--bg);color:var(--text);
     font-family:"Courier New",Consolas,monospace;font-size:1rem;line-height:1.4}
.wrap{max-width:420px;margin:0 auto}
h1{color:var(--green);font-size:1.7rem;margin:8px 0 6px}
.sub{color:var(--muted);margin:0 0 18px}
label{display:block;color:var(--cyan);margin:18px 0 6px}
input{width:100%;height:50px;padding:0 14px;background:var(--panel);
      border:1px solid var(--border);border-radius:8px;color:var(--green);
      font:inherit;font-size:16px}
input:focus{outline:2px solid var(--green);border-color:var(--green)}
.hint{color:var(--muted);font-size:.85rem;margin-top:6px}
button{width:100%;height:54px;margin-top:26px;background:var(--green);color:#000;
       border:0;border-radius:27px;font:inherit;font-size:1.05rem;font-weight:bold}
button:active{opacity:.8}
.error{margin:14px 0 0;padding:10px 12px;border:1px solid var(--red);
       border-radius:8px;background:#1a0f0f;color:var(--red)}
.switch{margin-top:26px;text-align:center}
a{color:var(--cyan)}
"""


def _page(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="theme-color" content="#0d0d0d">'
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        f'<body><div class="wrap">{body}</div></body></html>'
    )


def _error_box(error: str) -> str:
    return f'<div class="error">{html.escape(error)}</div>' if error else ""


def _login_html(error: str = "", username: str = "") -> str:
    # Everything user-supplied is passed through html.escape (XSS protection).
    return _page("Login - Rehab Planner", f"""
<h1>Rehab Planner</h1>
<p class="sub">Log in to see your sessions.</p>
{_error_box(error)}
<form method="post" action="/login">
  <label for="username">Username</label>
  <input id="username" name="username" value="{html.escape(username, quote=True)}"
         autocomplete="username" autocapitalize="none" autocorrect="off"
         spellcheck="false" required>
  <label for="password">Password</label>
  <input id="password" name="password" type="password"
         autocomplete="current-password" required>
  <button type="submit">Log in</button>
</form>
<p class="switch">No account? <a href="/register">Create one</a></p>
""")


def _register_html(error: str = "", username: str = "") -> str:
    return _page("Register - Rehab Planner", f"""
<h1>Create account</h1>
<p class="sub">Your sessions and progress are saved to your account.</p>
{_error_box(error)}
<form method="post" action="/register">
  <label for="username">Username</label>
  <input id="username" name="username" value="{html.escape(username, quote=True)}"
         autocomplete="username" autocapitalize="none" autocorrect="off"
         spellcheck="false" required>
  <div class="hint">3-30 characters: letters, numbers, underscore.</div>
  <label for="password">Password</label>
  <input id="password" name="password" type="password"
         autocomplete="new-password" minlength="{MIN_PASSWORD_LENGTH}" required>
  <div class="hint">At least {MIN_PASSWORD_LENGTH} characters.</div>
  <label for="confirm">Confirm password</label>
  <input id="confirm" name="confirm" type="password"
         autocomplete="new-password" required>
  <button type="submit">Create account</button>
</form>
<p class="switch">Already registered? <a href="/login">Log in</a></p>
""")


# ================================================================== routes ==
@router.get("/login", response_class=HTMLResponse)
def login_page(user: dict | None = Depends(current_user_or_none)):
    if user:                                   # already logged in
        return RedirectResponse("/", status_code=303)
    return HTMLResponse(_login_html())


@router.post("/login")
def login_submit(username: str = Form(""), password: str = Form("")):
    user_id = db_store.authenticate(username, password)
    if user_id is None:
        # One generic message for "no such user" AND "wrong password", so the
        # form cannot be used to discover which usernames exist.
        return HTMLResponse(
            _login_html("Wrong username or password.", username), status_code=401
        )
    response = RedirectResponse("/", status_code=303)
    _set_login_cookie(response, db_store.create_login_token(user_id))
    return response


@router.get("/register", response_class=HTMLResponse)
def register_page(user: dict | None = Depends(current_user_or_none)):
    if user:
        return RedirectResponse("/", status_code=303)
    return HTMLResponse(_register_html())


@router.post("/register")
def register_submit(username: str = Form(""), password: str = Form(""),
                    confirm: str = Form("")):
    if password != confirm:
        return HTMLResponse(
            _register_html("Passwords do not match.", username), status_code=400
        )
    try:
        user_id = db_store.create_user(username, password)
    except ValueError as e:                    # invalid input or username taken
        return HTMLResponse(_register_html(str(e), username), status_code=400)

    response = RedirectResponse("/", status_code=303)   # log in straight away
    _set_login_cookie(response, db_store.create_login_token(user_id))
    return response


@router.post("/logout")
def logout(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        db_store.delete_login_token(token)     # this device only
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


@router.get("/api/me")
def me(user: dict = Depends(require_user_api)):
    """Small JSON endpoint pages can call to show 'Logged in as ...'."""
    return {"id": user["id"], "username": user["username"]}
