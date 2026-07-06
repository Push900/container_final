"""
surveyor.report — professional inspection report generation.

Produces a legally-usable inspection report from an :class:`InspectionRecord`
dict, including: timestamp, GPS, inspector (AI Surveyor), container & seal info,
type/size, damage summary & severity, annotated images, confidence values,
repair recommendations, final pass/fail, a QR code linking to the inspection
data, a digital signature hash, and the complete audit trail.

Primary output is PDF via ReportLab; if ReportLab is unavailable the generator
degrades to a self-contained HTML report so reporting never hard-fails.
A canonical SHA-256 signature hash is computed over the record regardless of
output format, giving every report a verifiable fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("data/reports")

_RESULT_COLORS = {"PASS": "#16a34a", "FAIL": "#dc2626", "REVIEW": "#d97706"}
_SEV_COLORS = {"critical": "#dc2626", "severe": "#ea580c", "moderate": "#d97706",
               "minor": "#ca8a04", "cosmetic": "#65a30d", "none": "#16a34a"}


def signature_hash(record: dict) -> str:
    """Deterministic SHA-256 over the record (excluding volatile/derived fields)."""
    payload = {k: v for k, v in record.items()
               if k not in ("signature_hash", "report_path", "annotated_images")}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _make_qr(data: str, path: Path) -> Path | None:
    try:
        import qrcode
        img = qrcode.make(data)
        img.save(str(path))
        return path
    except Exception as exc:
        logger.info("QR generation unavailable (%s).", exc)
        return None


def generate_report(
    record: dict,
    annotated_images: list | None = None,
    audit_trail: list | None = None,
    out_dir: Path = REPORTS_DIR,
    qr_payload: str | None = None,
) -> dict:
    """Generate a report file for *record*.

    Returns ``{"path": str, "format": "pdf"|"html", "signature_hash": str}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rid = record.get("id", "INSPECTION")
    sig = record.get("signature_hash") or signature_hash(record)
    record["signature_hash"] = sig

    qr_path = _make_qr(qr_payload or f"inspection://{rid}", out_dir / f"{rid}_qr.png")

    try:
        path = _generate_pdf(record, annotated_images, audit_trail, out_dir, qr_path, sig)
        return {"path": str(path), "format": "pdf", "signature_hash": sig}
    except ImportError:
        logger.info("ReportLab not installed — generating HTML report instead.")
    except Exception as exc:
        logger.warning("PDF generation failed (%s) — falling back to HTML.", exc)
    path = _generate_html(record, annotated_images, audit_trail, out_dir, qr_path, sig)
    return {"path": str(path), "format": "html", "signature_hash": sig}


# ── PDF (ReportLab) ──────────────────────────────────────────────────────────

def _generate_pdf(record, annotated_images, audit_trail, out_dir, qr_path, sig) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    )
    from PIL import Image as PILImage

    path = out_dir / f"{record.get('id','inspection')}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            topMargin=15 * mm, bottomMargin=15 * mm,
                            leftMargin=15 * mm, rightMargin=15 * mm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=18, spaceAfter=4)
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9, textColor=colors.grey)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
    body = styles["Normal"]
    story = []

    result = (record.get("result") or "REVIEW").upper()
    story.append(Paragraph("AI Container Surveyor — Inspection Report", h1))
    story.append(Paragraph(
        f"Inspection ID: {record.get('id','—')} &nbsp;|&nbsp; "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", sub))
    story.append(Spacer(1, 6))

    # verdict banner
    rc = colors.HexColor(_RESULT_COLORS.get(result, "#6b7280"))
    verdict = Table([[Paragraph(f"<b>FINAL DECISION: {result}</b>",
                                ParagraphStyle("v", parent=body, fontSize=14,
                                               textColor=colors.white))]],
                    colWidths=[180 * mm])
    verdict.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), rc),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                 ("TOPPADDING", (0, 0), (-1, -1), 8),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story.append(verdict)
    story.append(Spacer(1, 8))

    # metadata table (+ QR on the right)
    meta_rows = [
        ["Inspector", record.get("inspector", "AI Surveyor")],
        ["Timestamp", record.get("created_at", "—")],
        ["GPS location", record.get("gps") or "Not available"],
        ["Persistent ID", str(record.get("track_id", "—"))],
        ["AI confidence", _pct(record.get("ai_confidence"))],
        ["Frames analysed", str(record.get("frames_analysed", "—"))],
    ]
    meta_tbl = Table([[Paragraph(f"<b>{k}</b>", body), Paragraph(str(v), body)] for k, v in meta_rows],
                     colWidths=[40 * mm, 80 * mm])
    meta_tbl.setStyle(_grid())
    if qr_path and Path(qr_path).exists():
        qr_img = RLImage(str(qr_path), width=32 * mm, height=32 * mm)
        layout = Table([[meta_tbl, qr_img]], colWidths=[125 * mm, 35 * mm])
        layout.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(layout)
    else:
        story.append(meta_tbl)

    # container info
    story.append(Paragraph("Container Information", h2))
    cont = record.get("container", {})
    st = record.get("size_type", {})
    cont_rows = [
        ["Container number", record.get("container_number", "—")],
        ["Check digit", "Valid" if cont.get("check_digit_valid") else "Invalid/Unknown"],
        ["Owner / operator", cont.get("owner_name") or cont.get("owner_code") or "—"],
        ["Equipment category", cont.get("equipment_category_desc") or "—"],
        ["Serial number", cont.get("serial_number") or "—"],
        ["ISO size/type code", record.get("size_type_code") or "—"],
        ["Container type", record.get("container_type") or "—"],
        ["Dimensions", _dims(st)],
        ["High-Cube", "Yes" if st.get("high_cube") else "No / Unknown"],
        ["Shipping line", record.get("shipping_line") or "—"],
    ]
    t = Table([[k, str(v)] for k, v in cont_rows], colWidths=[50 * mm, 110 * mm])
    t.setStyle(_grid()); story.append(t)

    # seal info
    story.append(Paragraph("Seal Inspection", h2))
    seal = record.get("seal", {})
    seal_rows = [
        ["Seal status", seal.get("status", "—")],
        ["Status confidence", _pct(seal.get("status_confidence"))],
        ["Seal type", seal.get("seal_type", "—")],
        ["Seal number", seal.get("seal_number") or "—"],
        ["Seal number confidence", _pct(seal.get("seal_number_confidence"))],
        ["Locking bar found", "Yes" if seal.get("locking_bar_found") else "No"],
        ["Method", seal.get("method", "—")],
    ]
    t = Table([[k, str(v)] for k, v in seal_rows], colWidths=[50 * mm, 110 * mm])
    t.setStyle(_grid()); story.append(t)

    # damage summary
    story.append(Paragraph("Damage Assessment", h2))
    dmg = record.get("damage", {})
    sev = dmg.get("worst_severity", "none")
    dmg_head = [
        ["Inspection score", f"{dmg.get('inspection_score', 100):.0f} / 100"],
        ["Worst severity", sev],
        ["Defect count", str(len(dmg.get("defects", [])))],
        ["Repair priority", dmg.get("repair_priority", "none")],
        ["Fit for service", "YES" if dmg.get("fit_for_service", True) else "NO"],
    ]
    t = Table([[k, str(v)] for k, v in dmg_head], colWidths=[50 * mm, 110 * mm])
    t.setStyle(_grid([("TEXTCOLOR", (1, 1), (1, 1),
                       colors.HexColor(_SEV_COLORS.get(sev, "#000000")))]))
    story.append(t)

    defects = dmg.get("defects", [])
    if defects:
        rows = [["Type", "Severity", "Component", "Confidence", "Area %"]]
        for d in defects[:25]:
            rows.append([d["defect_type"], d["severity"], d.get("component", "—"),
                         _pct(d.get("confidence")), f"{d.get('area_fraction',0)*100:.2f}"])
        dt = Table(rows, colWidths=[40*mm, 28*mm, 30*mm, 30*mm, 22*mm])
        dt.setStyle(_grid([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        story.append(Spacer(1, 4)); story.append(dt)

    # checklist
    checklist = record.get("checklist")
    if checklist:
        story.append(Paragraph("Surveyor Checklist", h2))
        for section, items in checklist.items():
            rows = [[Paragraph(f"<b>{section}</b>", body), ""]]
            for it in items:
                rows.append([it["item"], f"{it['status']} — {it['detail']}"])
            ct = Table(rows, colWidths=[55 * mm, 105 * mm])
            ct.setStyle(_grid([("FONTSIZE", (0, 0), (-1, -1), 8),
                               ("SPAN", (0, 0), (1, 0)),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff"))]))
            story.append(ct); story.append(Spacer(1, 2))

    # recommendations
    recs = record.get("recommendations") or []
    if recs:
        story.append(Paragraph("Repair Recommendations", h2))
        for r in recs:
            story.append(Paragraph(f"• {r}", body))

    # annotated images
    if annotated_images:
        story.append(Paragraph("Annotated Inspection Images", h2))
        for img in annotated_images[:6]:
            try:
                tmp = out_dir / f"{record.get('id','i')}_img{annotated_images.index(img)}.jpg"
                (img if isinstance(img, PILImage.Image) else PILImage.fromarray(img)).convert("RGB").save(tmp)
                story.append(RLImage(str(tmp), width=120 * mm, height=80 * mm))
                story.append(Spacer(1, 4))
            except Exception:
                continue

    # signature + audit
    story.append(Paragraph("Authenticity & Audit Trail", h2))
    story.append(Paragraph(f"<b>Digital signature (SHA-256):</b> {sig}", sub))
    if audit_trail:
        rows = [["Time", "Event", "Detail"]]
        for a in audit_trail[-12:]:
            rows.append([a.get("ts", ""), a.get("event", ""), (a.get("detail") or "")[:48]])
        at = Table(rows, colWidths=[55 * mm, 40 * mm, 65 * mm])
        at.setStyle(_grid([("FONTSIZE", (0, 0), (-1, -1), 7),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]))
        story.append(at)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This report was produced autonomously by the AI Container Surveyor. "
        "The signature hash above verifies the integrity of the stored inspection record.",
        sub))

    doc.build(story)
    return path


def _grid(extra=None):
    from reportlab.lib import colors
    from reportlab.platypus import TableStyle
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]
    if extra:
        style += extra
    return TableStyle(style)


# ── HTML fallback ────────────────────────────────────────────────────────────

def _generate_html(record, annotated_images, audit_trail, out_dir, qr_path, sig) -> Path:
    path = out_dir / f"{record.get('id','inspection')}.html"
    result = (record.get("result") or "REVIEW").upper()
    cont = record.get("container", {}); st = record.get("size_type", {})
    seal = record.get("seal", {}); dmg = record.get("damage", {})

    def table(d):
        return "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in d.items())

    defects_html = "".join(
        f"<tr><td>{d['defect_type']}</td><td>{d['severity']}</td>"
        f"<td>{d.get('component','')}</td><td>{d.get('confidence',0):.2f}</td></tr>"
        for d in dmg.get("defects", []))
    audit_html = "".join(
        f"<tr><td>{a.get('ts','')}</td><td>{a.get('event','')}</td><td>{a.get('detail','')}</td></tr>"
        for a in (audit_trail or []))

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Inspection {record.get('id','')}</title>
<style>body{{font-family:system-ui,Arial;margin:32px;color:#111}}
h1{{margin-bottom:0}} .sub{{color:#6b7280;font-size:13px}}
.banner{{background:{_RESULT_COLORS.get(result,'#6b7280')};color:#fff;padding:12px;border-radius:8px;font-size:20px;font-weight:700;margin:12px 0}}
table{{border-collapse:collapse;margin:8px 0;width:100%}} th,td{{border:1px solid #d1d5db;padding:6px;text-align:left;font-size:13px}}
th{{background:#f3f4f6;width:30%}} h2{{margin-top:22px}}
.sig{{font-family:monospace;font-size:11px;word-break:break-all;background:#f9fafb;padding:8px;border-radius:6px}}</style></head><body>
<h1>AI Container Surveyor — Inspection Report</h1>
<div class="sub">ID: {record.get('id','—')} | Generated: {datetime.now():%Y-%m-%d %H:%M:%S} | Inspector: {record.get('inspector','AI Surveyor')}</div>
<div class="banner">FINAL DECISION: {result}</div>
<h2>Container Information</h2><table>{table({
    'Container number': record.get('container_number','—'),
    'Owner / operator': cont.get('owner_name') or cont.get('owner_code') or '—',
    'Serial number': cont.get('serial_number') or '—',
    'Check digit valid': cont.get('check_digit_valid'),
    'ISO size/type': record.get('size_type_code') or '—',
    'Container type': record.get('container_type') or '—',
    'Dimensions': _dims(st),
    'High-Cube': 'Yes' if st.get('high_cube') else 'No/Unknown',
    'Shipping line': record.get('shipping_line') or '—',
})}</table>
<h2>Seal Inspection</h2><table>{table({
    'Seal status': seal.get('status','—'),
    'Status confidence': _pct(seal.get('status_confidence')),
    'Seal type': seal.get('seal_type','—'),
    'Seal number': seal.get('seal_number') or '—',
})}</table>
<h2>Damage Assessment</h2><table>{table({
    'Inspection score': f"{dmg.get('inspection_score',100):.0f}/100",
    'Worst severity': dmg.get('worst_severity','none'),
    'Repair priority': dmg.get('repair_priority','none'),
    'Fit for service': dmg.get('fit_for_service',True),
})}</table>
<table><tr><th>Type</th><th>Severity</th><th>Component</th><th>Conf</th></tr>{defects_html}</table>
<h2>Authenticity & Audit Trail</h2>
<div class="sig"><b>SHA-256:</b> {sig}</div>
<table><tr><th>Time</th><th>Event</th><th>Detail</th></tr>{audit_html}</table>
<p class="sub">Produced autonomously by the AI Container Surveyor.</p>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    return path


# ── small formatters ─────────────────────────────────────────────────────────

def _pct(v):
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _dims(st: dict) -> str:
    if not st:
        return "—"
    parts = []
    if st.get("length_label"):
        parts.append(f"L {st['length_label']}")
    if st.get("height_label"):
        parts.append(f"H {st['height_label']}")
    if st.get("width_mm"):
        parts.append(f"W ~{st['width_mm']}mm")
    return " · ".join(parts) or "—"
