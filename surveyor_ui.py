"""
surveyor_ui.py — Gradio glue for the AI Surveyor dashboard & inspection history.

Kept separate from app.py so the existing three pages stay untouched.  The heavy
AISurveyor (YOLO + EasyOCR + MiDaS-free) is lazy-loaded on first inspection so
app startup is not slowed down.
"""

from __future__ import annotations

import threading
import traceback

from PIL import Image

from surveyor.database import get_db

_SURVEYOR = None
_LOCK = threading.Lock()

_RESULT_COLOR = {"PASS": "#16a34a", "FAIL": "#dc2626", "REVIEW": "#d97706"}
_SEV_COLOR = {"critical": "#dc2626", "severe": "#ea580c", "moderate": "#d97706",
              "minor": "#ca8a04", "cosmetic": "#65a30d", "none": "#16a34a", None: "#6b7280"}
_STATUS_COLOR = {"PASS": "#16a34a", "FAIL": "#dc2626", "ATTENTION": "#d97706",
                 "NOT_VISIBLE": "#6b7280", "PENDING": "#9ca3af"}
_SEAL_COLOR = {"INTACT": "#16a34a", "BROKEN": "#dc2626", "MISSING": "#dc2626",
               "UNKNOWN": "#6b7280"}


def get_surveyor():
    """Lazy singleton so models load only when the surveyor tab is first used."""
    global _SURVEYOR
    if _SURVEYOR is None:
        with _LOCK:
            if _SURVEYOR is None:
                from surveyor import AISurveyor
                _SURVEYOR = AISurveyor(save_to_db=True)
    return _SURVEYOR


# ── card helpers ──────────────────────────────────────────────────────────────

def _card(label, value, sub="", color="#111827"):
    return f"""
    <div style="flex:1;min-width:150px;background:#fff;border:1px solid #e5e7eb;
                border-radius:10px;padding:12px 14px;margin:5px;box-shadow:0 1px 2px rgba(0,0,0,.04);">
        <div style="font-size:.7em;letter-spacing:.04em;color:#6b7280;font-weight:600;text-transform:uppercase;">{label}</div>
        <div style="font-size:1.35em;font-weight:700;color:{color};margin-top:2px;">{value}</div>
        <div style="font-size:.72em;color:#9ca3af;margin-top:1px;">{sub}</div>
    </div>"""


def dashboard_html(rec: dict) -> str:
    if not rec:
        return "<p style='color:#999;padding:30px;text-align:center;'>Run an inspection to populate the dashboard.</p>"
    st = rec.get("size_type", {}) or {}
    seal = rec.get("seal", {}) or {}
    result = (rec.get("result") or "REVIEW").upper()
    sev = rec.get("worst_severity") or "none"
    dims = " · ".join(p for p in [
        st.get("length_label"), st.get("height_label"),
        ("HC" if st.get("high_cube") else None)] if p) or "—"

    cards = "".join([
        _card("Persistent ID", rec.get("track_id") or rec.get("id", "—")[-6:], "tracking"),
        _card("Container No.", rec.get("container_number") or "—",
              "✔ ISO check digit" if (rec.get("container") or {}).get("check_digit_valid") else "check digit unverified",
              "#111827" if (rec.get("container") or {}).get("check_digit_valid") else "#d97706"),
        _card("OCR Confidence", f"{rec.get('ai_confidence', 0)*100:.0f}%", "fused multi-frame"),
        _card("Container Type", rec.get("container_type") or "—", rec.get("size_type_code") or ""),
        _card("Dimensions", dims, "L · H"),
        _card("Seal Status", seal.get("status", "—"),
              f"{seal.get('status_confidence',0)*100:.0f}% conf", _SEAL_COLOR.get(seal.get("status"), "#111827")),
        _card("Seal No.", seal.get("seal_number") or "—",
              f"{seal.get('seal_number_confidence',0)*100:.0f}% conf"),
        _card("Damage Count", rec.get("damage_count", 0), f"worst: {sev}", _SEV_COLOR.get(sev, "#111827")),
        _card("Severity", sev, rec.get("repair_priority", "—"), _SEV_COLOR.get(sev, "#111827")),
        _card("Inspection Score", f"{rec.get('inspection_score', 100):.0f}/100", "condition"),
        _card("Completion", f"{rec.get('completion_pct', 0):.0f}%", "of checklist"),
        _card("AI Confidence", f"{rec.get('ai_confidence', 0)*100:.0f}%", "overall"),
        _card("Frame Quality", f"{(rec.get('frame_quality') or 0)*100:.0f}%" if rec.get('frame_quality') else "—", "best frame"),
        _card("Frames", rec.get("frames_analysed", "—"), "analysed"),
        _card("Inspector", "AI Surveyor", rec.get("gps") or "no GPS"),
    ])

    banner = f"""
    <div style="background:{_RESULT_COLOR.get(result,'#6b7280')};color:#fff;border-radius:10px;
                padding:16px 20px;margin:6px 5px 10px;display:flex;justify-content:space-between;align-items:center;">
        <div style="font-size:1.5em;font-weight:800;">FINAL DECISION: {result}</div>
        <div style="font-size:.85em;opacity:.9;">Inspection {rec.get('id','')} · {rec.get('created_at','')[:19].replace('T',' ')}</div>
    </div>"""
    return banner + f"<div style='display:flex;flex-wrap:wrap;'>{cards}</div>"


def checklist_html(rec: dict) -> str:
    checklist = (rec or {}).get("checklist")
    if not checklist:
        return ""
    out = ["<div style='margin-top:8px;'>"]
    for section, items in checklist.items():
        out.append(f"<div style='font-weight:700;color:#1f2937;margin:12px 0 4px;'>{section}</div>")
        out.append("<table style='width:100%;border-collapse:collapse;font-size:.85em;'>")
        for it in items:
            c = _STATUS_COLOR.get(it["status"], "#6b7280")
            out.append(
                f"<tr><td style='padding:4px 6px;border-bottom:1px solid #f0f0f0;width:32%;'>{it['item']}</td>"
                f"<td style='padding:4px 6px;border-bottom:1px solid #f0f0f0;width:13%;'>"
                f"<span style='color:{c};font-weight:700;'>{it['status']}</span></td>"
                f"<td style='padding:4px 6px;border-bottom:1px solid #f0f0f0;color:#6b7280;'>{it['detail']}</td></tr>")
        out.append("</table>")
    out.append("</div>")
    return "".join(out)


def recommendations_html(rec: dict) -> str:
    recs = (rec or {}).get("recommendations") or []
    if not recs:
        return ""
    items = "".join(f"<li style='margin:3px 0;'>{r}</li>" for r in recs)
    return (f"<div style='background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:12px 16px;margin-top:8px;'>"
            f"<div style='font-weight:700;color:#92400e;margin-bottom:4px;'>Repair Recommendations</div>"
            f"<ul style='margin:0;padding-left:18px;color:#78350f;'>{items}</ul></div>")


# ── inspection handler ────────────────────────────────────────────────────────

def run_inspection(files, gps):
    """files: list of filepaths (gr.Files) or a single numpy image fallback."""
    try:
        imgs = _load_images(files)
        if not imgs:
            return ("<p style='color:#dc2626;padding:20px;'>No valid images uploaded.</p>",
                    None, "", "", None)
        surveyor = get_surveyor()
        meta = {"gps": gps.strip()} if gps and gps.strip() else {}
        if len(imgs) == 1:
            rec = surveyor.inspect_image(imgs[0], meta=meta)
        else:
            rec = surveyor.inspect_frames(imgs, meta=meta)

        best = imgs[0]  # preview the first supplied frame
        report = rec.get("report_path")
        # trimmed record for display
        display = {k: v for k, v in rec.items() if k not in ("checklist", "container", "size_type", "seal", "damage")}
        display["damage"] = rec.get("damage", {}).get("counts_by_severity")
        return (dashboard_html(rec),
                best,
                checklist_html(rec) + recommendations_html(rec),
                display,
                report)
    except Exception as e:
        err = f"<p style='color:#dc2626;padding:20px;'>Inspection error: {e}</p><pre style='font-size:.7em;color:#991b1b;'>{traceback.format_exc()[-600:]}</pre>"
        return err, None, "", {"error": str(e)}, None


def _load_images(files):
    imgs = []
    if files is None:
        return imgs
    if not isinstance(files, list):
        files = [files]
    for f in files:
        try:
            if isinstance(f, str):
                imgs.append(Image.open(f).convert("RGB"))
            elif hasattr(f, "name"):
                imgs.append(Image.open(f.name).convert("RGB"))
            else:
                import numpy as np
                imgs.append(Image.fromarray(np.asarray(f)).convert("RGB"))
        except Exception:
            continue
    return imgs


# ── history search ────────────────────────────────────────────────────────────

def search_history(container_number, seal_number, date_from, date_to, shipping_line, result, damage_category):
    try:
        db = get_db()
        rows = db.search(
            container_number=container_number or None,
            seal_number=seal_number or None,
            date_from=date_from or None,
            date_to=date_to or None,
            shipping_line=shipping_line or None,
            result=(result if result and result != "Any" else None),
            damage_category=damage_category or None,
            limit=200,
        )
        if not rows:
            return f"<p style='color:#6b7280;padding:16px;'>No inspections found. (Total in DB: {db.count()})</p>"
        head = ("<tr style='background:#1f2937;color:#fff;'>"
                + "".join(f"<th style='padding:6px 8px;text-align:left;font-size:.8em;'>{h}</th>"
                          for h in ["ID", "Date", "Container", "Type", "Seal", "Seal#", "Result", "Score", "Severity", "Line"])
                + "</tr>")
        body = []
        for r in rows:
            rc = _RESULT_COLOR.get((r.get("result") or "").upper(), "#6b7280")
            body.append(
                "<tr style='border-bottom:1px solid #eee;font-size:.8em;'>"
                f"<td style='padding:5px 8px;font-family:monospace;'>{r.get('id','')}</td>"
                f"<td style='padding:5px 8px;'>{(r.get('created_at') or '')[:19].replace('T',' ')}</td>"
                f"<td style='padding:5px 8px;font-weight:600;'>{r.get('container_number') or '—'}</td>"
                f"<td style='padding:5px 8px;'>{r.get('container_type') or '—'}</td>"
                f"<td style='padding:5px 8px;'>{r.get('seal_status') or '—'}</td>"
                f"<td style='padding:5px 8px;'>{r.get('seal_number') or '—'}</td>"
                f"<td style='padding:5px 8px;color:{rc};font-weight:700;'>{r.get('result') or '—'}</td>"
                f"<td style='padding:5px 8px;'>{r.get('inspection_score') if r.get('inspection_score') is not None else '—'}</td>"
                f"<td style='padding:5px 8px;'>{r.get('worst_severity') or '—'}</td>"
                f"<td style='padding:5px 8px;'>{r.get('shipping_line') or '—'}</td></tr>")
        return (f"<div style='font-size:.85em;color:#6b7280;margin-bottom:6px;'>{len(rows)} result(s) · "
                f"{db.count()} total inspections · audit chain "
                f"{'✔ valid' if db.verify_audit_chain() else '✗ TAMPERED'}</div>"
                f"<table style='width:100%;border-collapse:collapse;'>{head}{''.join(body)}</table>")
    except Exception as e:
        return f"<p style='color:#dc2626;padding:16px;'>Search error: {e}</p>"
