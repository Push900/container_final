"""
surveyor.surveyor — the AISurveyor orchestrator.

Runs the end-to-end inspection workflow that mimics an experienced human
surveyor, accumulating evidence across frames before finalising any decision:

    1.  Detect container               9.  Determine seal status
    2.  Assign persistent tracking ID  10. Read seal number
    3.  Read container number          11. Compare multiple OCR frames
    4.  Validate ISO check digit       12. Inspect all visible surfaces
    5.  Decode container information    13. Detect damage
    6.  Detect container doors          14. Classify severity
    7.  Detect locking bars            15. Estimate repair urgency
    8.  Detect seal                    16. Generate final score + report

Use it streaming (live camera) or batch::

    surveyor = AISurveyor()
    s = surveyor.start_session({"gps": "12.97,77.59"})
    for frame in frames:
        live = surveyor.process_frame(s, frame)   # dashboard snapshot
    record = surveyor.finalize(s)                  # full InspectionRecord

or in one call::

    record = surveyor.inspect_frames(frames, meta={"gps": ...})

Every component is fetched from :mod:`surveyor.registry`, so swapping in a newer
detector / OCR / seal / damage model needs no change here or in the UI.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from surveyor import iso6346
from surveyor.fusion import TemporalFuser
from surveyor.registry import get_component, register_default_components
from surveyor.checklist import build_checklist
from surveyor.database import get_db

logger = logging.getLogger(__name__)

_MARKING_PATTERNS = {
    "csc": re.compile(r"CSC|SAFETY\s*APPROVAL|ACEP", re.I),
    "max_gr": re.compile(r"MAX\s*GROSS|MAX\s*GR|MGW", re.I),
    "tare": re.compile(r"\bTARE\b", re.I),
    "payload": re.compile(r"PAYLOAD|NET\s*(WT|WEIGHT)", re.I),
    "hazard": re.compile(r"FLAMMABLE|CORROSIVE|TOXIC|EXPLOSIVE|OXIDIZER|CLASS\s*[1-9]", re.I),
}


@dataclass
class InspectionSession:
    id: str
    started_at: float
    meta: dict = field(default_factory=dict)
    frames: int = 0
    track_id: int | None = None
    # fusers / accumulators
    container_readings: list = field(default_factory=list)
    size_type_readings: list = field(default_factory=list)
    detection_fuser: TemporalFuser = field(default_factory=lambda: TemporalFuser(min_support=3, min_confidence=0.4))
    markings: set = field(default_factory=set)
    best_frame: Image.Image | None = None
    best_frame_quality: float = -1.0
    annotated: list = field(default_factory=list)
    last_snapshot: dict = field(default_factory=dict)


class AISurveyor:
    def __init__(self, save_to_db: bool = True):
        register_default_components()
        self.ocr = get_component("ocr_engine")
        self.detector = get_component("container_detector")
        self.seal = get_component("seal_inspector")
        self.damage = get_component("damage_assessor")
        self.save_to_db = save_to_db
        self._db = get_db() if save_to_db else None

    # ── session lifecycle ────────────────────────────────────────────────────
    def start_session(self, meta: dict | None = None) -> InspectionSession:
        sid = "INS-" + datetime.now().strftime("%Y%m%d%H%M%S")
        # fresh stateful components per session
        self.seal.reset()
        self.damage.reset()
        self.detector.reset()
        return InspectionSession(id=sid, started_at=time.time(), meta=meta or {})

    # ── per-frame processing (steps 1-15) ────────────────────────────────────
    def process_frame(self, s: InspectionSession, frame) -> dict:
        if frame is None:
            return s.last_snapshot
        pil = frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame))
        pil = pil.convert("RGB")
        s.frames += 1

        # 1-2. Detect + track
        dets = self.detector.detect(pil, track=True)
        container = max(dets, key=lambda d: d.get("confidence", 0)) if dets else None
        det_conf = container.get("confidence", 0.0) if container else 0.0
        if container:
            s.detection_fuser.observe("container", det_conf)
            tid = container.get("track_id")
            if tid is not None and s.track_id is None:
                s.track_id = tid
            crop = container.get("crop") or pil
        else:
            crop = pil

        # track best frame for the report (highest detection conf)
        if det_conf > s.best_frame_quality:
            s.best_frame_quality = det_conf
            s.best_frame = crop

        # 3-5. Read & decode container number + ISO size/type in a single OCR pass
        # (heavy denoise enabled only on the first frame for a thorough first read).
        ocr_all = self.ocr.read_all(pil, thorough=(s.frames == 1))
        reading = ocr_all["container"]
        if reading.text:
            s.container_readings.append(reading)
        for frag in ocr_all["fragments"]:
            self._scan_markings(frag["text"], s)
        st_reading = ocr_all["size_type"]
        if st_reading.text:
            s.size_type_readings.append(st_reading)

        # 6-11. Doors / locking bars / seal / seal number (door ROI = container crop)
        seal_now = self.seal.inspect_frame(self._door_roi(crop))

        # 12-15. Surfaces / damage
        self.damage.assess_frame(crop, component="body")

        snapshot = self._snapshot(s, det_conf, reading, seal_now)
        s.last_snapshot = snapshot
        return snapshot

    # ── finalisation (step 16) ───────────────────────────────────────────────
    def finalize(self, s: InspectionSession, *, save: bool = True, make_report: bool = True) -> dict:
        cont_fused = self.ocr.fuse_container_readings(s.container_readings)
        cont_decoded = iso6346.decode_container_number(cont_fused.text)

        st_fused = self._fuse_size_type(s.size_type_readings)
        st_decoded = iso6346.decode_size_type_code(st_fused) if st_fused else {}
        container_type = iso6346.container_type_label(st_decoded) if st_decoded.get("valid") else "Unknown"

        seal_res = self.seal.result().as_dict()
        damage_report = self.damage.report().as_dict()

        evidence = {
            "container_decoded": cont_decoded,
            "size_type_decoded": st_decoded,
            "seal": seal_res,
            "damage": damage_report,
            "detected_markings": s.markings,
        }
        checklist = build_checklist(evidence)

        ai_conf = self._overall_confidence(cont_fused, seal_res, s)
        result, recs = self._decide(cont_decoded, seal_res, damage_report, ai_conf)

        record = {
            "id": s.id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inspector": "AI Surveyor",
            "gps": s.meta.get("gps"),
            "track_id": s.track_id,
            "frames_analysed": s.frames,
            "ai_confidence": ai_conf,
            "container_number": cont_decoded.get("normalized") or cont_fused.text,
            "container": cont_decoded,
            "size_type_code": st_fused,
            "size_type": st_decoded,
            "container_type": container_type,
            "shipping_line": cont_decoded.get("owner_name"),
            "seal_number": seal_res.get("seal_number"),
            "seal_status": seal_res.get("status"),
            "seal": seal_res,
            "damage": damage_report,
            "inspection_score": damage_report.get("inspection_score"),
            "worst_severity": damage_report.get("worst_severity"),
            "damage_count": len(damage_report.get("defects", [])),
            "repair_priority": damage_report.get("repair_priority"),
            "checklist": checklist,
            "detected_markings": sorted(s.markings),
            "result": result,
            "recommendations": recs,
            "completion_pct": self._completion(cont_decoded, st_decoded, seal_res, damage_report),
        }

        from surveyor.report import signature_hash
        record["signature_hash"] = signature_hash(record)

        if save and self._db is not None:
            try:
                self._db.save(record)
            except Exception as exc:
                logger.warning("DB save failed: %s", exc)

        if make_report:
            try:
                from surveyor.report import generate_report
                audit = self._db.audit_trail(s.id) if self._db else None
                rep = generate_report(
                    record,
                    annotated_images=[s.best_frame] if s.best_frame else None,
                    audit_trail=audit,
                    qr_payload=f"inspection://{s.id}",
                )
                record["report_path"] = rep["path"]
                record["report_format"] = rep["format"]
                if self._db is not None:
                    self._db.update_report_path(s.id, rep["path"], rep["signature_hash"])
            except Exception as exc:
                logger.warning("Report generation failed: %s", exc)

        return record

    # ── convenience ──────────────────────────────────────────────────────────
    def inspect_frames(self, frames: list, meta: dict | None = None, **kw) -> dict:
        s = self.start_session(meta)
        for f in frames:
            self.process_frame(s, f)
        return self.finalize(s, **kw)

    def inspect_image(self, img, meta: dict | None = None, **kw) -> dict:
        # A single still gets a few internal passes so temporal logic still applies.
        s = self.start_session(meta)
        for _ in range(3):
            self.process_frame(s, img)
        return self.finalize(s, **kw)

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _door_roi(crop: Image.Image) -> Image.Image:
        """Heuristic door region: doors are at a container end → right third.
        Falls back to the whole crop when it is too narrow to split."""
        w, h = crop.size
        if w < 120:
            return crop
        return crop.crop((int(w * 0.55), 0, w, h))

    def _scan_markings(self, text: str, s: InspectionSession):
        for key, pat in _MARKING_PATTERNS.items():
            if pat.search(text or ""):
                s.markings.add(key)

    @staticmethod
    def _fuse_size_type(readings) -> str:
        if not readings:
            return ""
        from collections import Counter
        valid = [r.text for r in readings if r.valid]
        pool = valid or [r.text for r in readings]
        return Counter(pool).most_common(1)[0][0] if pool else ""

    @staticmethod
    def _overall_confidence(cont_fused, seal_res, s: InspectionSession) -> float:
        det = s.detection_fuser.result().confidence
        ocr = cont_fused.confidence if cont_fused.text else 0.0
        ocr_valid_bonus = 0.15 if cont_fused.valid else 0.0
        seal_c = seal_res.get("status_confidence", 0.0)
        conf = 0.4 * det + 0.4 * min(1.0, ocr + ocr_valid_bonus) + 0.2 * seal_c
        return float(round(min(1.0, conf), 4))

    @staticmethod
    def _completion(cont, st, seal, damage) -> float:
        checks = [
            bool(cont.get("normalized")),
            bool(cont.get("check_digit_valid")),
            bool(st.get("valid")),
            seal.get("status") in ("INTACT", "BROKEN", "MISSING"),
            bool(damage.get("defects") is not None),
        ]
        return round(100.0 * sum(checks) / len(checks), 1)

    @staticmethod
    def _decide(cont, seal, damage, ai_conf):
        recs: list[str] = []
        result = "PASS"

        if seal.get("status") == "BROKEN":
            result = "FAIL"; recs.append("Seal is broken/cut — escalate: possible tampering, verify cargo.")
        elif seal.get("status") == "MISSING":
            result = "FAIL"; recs.append("Seal missing where expected — hold container, raise exception.")
        elif seal.get("status") not in ("INTACT",):
            result = "REVIEW"; recs.append("Seal status unconfirmed — capture clearer door/seal frames.")

        if not damage.get("fit_for_service", True):
            result = "FAIL"
            recs.append(f"Container not fit for service (worst: {damage.get('worst_severity')}). "
                        "Remove from rotation pending repair.")
        elif damage.get("worst_severity") in ("severe", "moderate"):
            if result == "PASS":
                result = "REVIEW"
            recs.append(f"{damage.get('worst_severity').title()} damage detected — "
                        f"repair priority {damage.get('repair_priority')}.")

        if not cont.get("normalized"):
            result = "REVIEW"; recs.append("Container number not read — re-image the marking from the front/side.")
        elif not cont.get("check_digit_valid"):
            if result == "PASS":
                result = "REVIEW"
            recs.append("Container number check digit failed — verify reading manually.")

        if ai_conf < 0.45 and result == "PASS":
            result = "REVIEW"; recs.append("Overall AI confidence is low — additional frames recommended.")

        if not recs:
            recs.append("No issues found. Container, seal and structure verified within confidence thresholds.")
        return result, recs

    def _snapshot(self, s, det_conf, reading, seal_now) -> dict:
        cont_fused = self.ocr.fuse_container_readings(s.container_readings) if s.container_readings else None
        elapsed = time.time() - s.started_at
        fps = s.frames / elapsed if elapsed > 0 else 0.0
        damage_now = self.damage.report().as_dict()
        cont_text = (cont_fused.text if cont_fused else "") or ""
        return {
            "inspection_id": s.id,
            "track_id": s.track_id,
            "frame": s.frames,
            "fps": round(fps, 2),
            "elapsed_s": round(elapsed, 1),
            "detection_confidence": round(det_conf, 3),
            "container_number": cont_text,
            "container_number_valid": cont_fused.valid if cont_fused else False,
            "ocr_confidence": round(cont_fused.confidence, 3) if cont_fused else 0.0,
            "seal_status": seal_now.get("status"),
            "seal_confidence": round(seal_now.get("status_confidence", 0.0), 3),
            "seal_number": seal_now.get("seal_number"),
            "seal_number_confidence": round(seal_now.get("seal_number_confidence", 0.0), 3),
            "damage_count": len(damage_now.get("defects", [])),
            "worst_severity": damage_now.get("worst_severity"),
            "inspection_score": damage_now.get("inspection_score"),
            "frame_quality": round(s.best_frame_quality, 3),
            "ai_confidence": self._overall_confidence(
                cont_fused or _empty_reading(), seal_now, s),
        }


class _EmptyReading:
    text = ""
    confidence = 0.0
    valid = False


def _empty_reading():
    return _EmptyReading()
