"""
src/reporting/pdf_report.py

Purpose: Generate a downloadable PDF report for one session's scored
sets, visually mirroring the color-coded table shown in the Streamlit
dashboard (render_results() in app.py) -- for offline sharing with a
physiotherapist, or as a leave-behind artifact for a viva demo.

Design note: color bands (SCORE_GOOD/OK/BAD/NA) intentionally match
app.py's _score_color() thresholds (85 / 60 / 0) exactly, so the PDF
and the live dashboard never visually disagree.
"""

from io import BytesIO
from datetime import datetime, timezone

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)

SCORE_GOOD = colors.HexColor("#b7e4c7")
SCORE_OK = colors.HexColor("#ffe8a3")
SCORE_BAD = colors.HexColor("#ffb3b3")
SCORE_NA = colors.HexColor("#e0e0e0")

# (source column, PDF header label)
TABLE_COLUMNS = [
    ("set_index", "Set #"),
    ("label", "Exercise"),
    ("start_mmss", "Start"),
    ("end_mmss", "End"),
    ("duration_s", "Dur (s)"),
    ("num_windows", "Windows"),
    ("mean_confidence", "Confidence"),
    ("is_short", "Short?"),
    ("quality_score", "Quality"),
    ("feedback", "Feedback"),
]

COL_WIDTHS_CM = [1.3, 3.4, 1.6, 1.6, 1.8, 1.8, 2.0, 1.6, 1.8, 6.5]


def _score_band_color(score) -> colors.Color:
    if pd.isna(score):
        return SCORE_NA
    if score >= 85:
        return SCORE_GOOD
    if score >= 60:
        return SCORE_OK
    return SCORE_BAD


def _format_cell(col: str, val, small_style: ParagraphStyle):
    if col == "is_short":
        return "Yes" if val in (1, True) else "No"
    if col in ("mean_confidence", "duration_s", "quality_score"):
        return f"{val:.3f}" if (col == "mean_confidence" and pd.notna(val)) else \
               (f"{val:.1f}" if pd.notna(val) else "-")
    if col == "feedback":
        return Paragraph(str(val), small_style)
    return str(val)


def generate_session_pdf_report(scored_df: pd.DataFrame,
                                 summary: dict,
                                 session_label: str) -> bytes:
    """
    Build a PDF report for one session and return it as raw bytes,
    ready to hand directly to st.download_button (no disk write needed).

    Parameters
    ----------
    scored_df : the same DataFrame passed to app.py's render_results()
        (either the freshly-scored set, or one reloaded from the DB).
        Must contain the columns listed in TABLE_COLUMNS.
    summary : dict from quality_scorer.session_summary().
    session_label : session_id / display label shown in the report header.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()
    small_style = ParagraphStyle("SmallCell", parent=styles["Normal"], fontSize=7.5, leading=9)

    story = []

    # --- Header ---
    story.append(Paragraph("AI Personalized Rehabilitation Planner", styles["Title"]))
    story.append(Paragraph(f"Session Report: {session_label}", styles["Heading2"]))
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph(f"Generated: {generated_at}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # --- Summary block ---
    avg_score = summary.get("avg_quality_score")
    for line in [
        f"Average Quality Score: {avg_score:.1f}" if avg_score is not None else "Average Quality Score: N/A",
        f"Scored Sets: {summary.get('num_scored_sets', 0)}",
        f"Short Sets Flagged: {summary.get('num_short_sets', 0)}",
    ]:
        story.append(Paragraph(line, styles["Normal"]))
    story.append(Spacer(1, 16))

    # --- Per-set table ---
    table_data = [[label for _, label in TABLE_COLUMNS]]
    for _, row in scored_df.iterrows():
        table_data.append([_format_cell(col, row.get(col), small_style) for col, _ in TABLE_COLUMNS])

    table = Table(table_data, colWidths=[w * cm for w in COL_WIDTHS_CM], repeatRows=1)

    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b2b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
    ]

    quality_col_idx = [c for c, _ in TABLE_COLUMNS].index("quality_score")
    for i, (_, row) in enumerate(scored_df.iterrows(), start=1):
        band_color = _score_band_color(row.get("quality_score"))
        style_commands.append(("BACKGROUND", (quality_col_idx, i), (quality_col_idx, i), band_color))

    table.setStyle(TableStyle(style_commands))
    story.append(table)

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Good form (score &gt;= 85)  |  Some inconsistency (60-84)  |  "
        "Unstable, review form (&lt; 60)  |  Not scored (rest period)",
        styles["Normal"],
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
