"""
backend/report_routes.py

Purpose (Day 26, step 6): download one session's PDF report.

  GET /sessions/<session_id>/report.pdf

How it works
------------
1. The login guard identifies the user (anonymous visitors go to /login).
2. db_store.get_session(session_id, user_id) only returns the session if it
   belongs to THIS user. Someone else's session and a non-existent id both
   give the same 404, so ids of other users cannot be probed.
3. The saved sets are read back from the database and handed to the existing
   pdf_report.generate_session_pdf_report(), together with the summary from
   quality_scorer.session_summary() -- the same functions the live results use.
4. The PDF is built in memory and sent as a download; nothing is written to disk.

A plain `def` handler on purpose: building a PDF is CPU work, and FastAPI runs
plain `def` handlers in a thread pool so it cannot block other requests.

Import note: main.py puts src/feedback, src/storage and src/reporting on
sys.path, so these plain-name imports work (same style as the rest of the app).
"""

import re

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse, Response

import db_store
from auth_routes import require_user_page
from pdf_report import generate_session_pdf_report
from quality_scorer import session_summary

router = APIRouter()


@router.get("/sessions/{session_id}/report.pdf")
def session_report(session_id: str, user: dict = Depends(require_user_page)):
    session = db_store.get_session(session_id, user["id"])
    if session is None:
        return PlainTextResponse("Session not found.", status_code=404)

    sets_df = db_store.get_sets_for_session(session_id, user["id"])

    uploaded = session["uploaded_at"][:16].replace("T", " ")
    label = f'{session["source_name"]} - uploaded {uploaded} UTC'
    pdf_bytes = generate_session_pdf_report(sets_df, session_summary(sets_df), label)

    # Filename: letters, digits, "_" and "-" only, so it can never break the header.
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", session["source_name"]).strip("_") or "session"
    return Response(
        pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="rehab_report_{safe_name}.pdf"',
            "Cache-Control": "private, no-store",      # personal data: never cached
        },
    )
