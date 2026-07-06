"""
surveyor.damage_assessor — structural & cosmetic damage assessment.

Detects and grades the defects a human surveyor records: rust, corrosion, paint
damage, dents, holes, cracks, structural deformation, bent locking rods, etc.,
then classifies each into a five-level severity scale, computes an overall
inspection score, estimates repair priority, and flags containers unsuitable for
service.

Severity scale (industry-aligned)::
    cosmetic  → surface only, no action
    minor     → log, no immediate repair
    moderate  → schedule repair
    severe    → repair before next load
    critical  → out of service / not seaworthy

Honesty note
------------
Production-grade defect detection needs a model trained on labelled container
damage (e.g. an instance-segmentation model).  That model is not shipped here.
This module provides:

  * a pluggable detector slot (`models/damage_detector.pt`) consumed unchanged
    once a trained model exists, and
  * a transparent **classical-CV baseline** (colour-based rust/corrosion +
    edge/texture anomaly + dark-blob holes) producing *provisional* defects with
    low, clearly-labelled confidence.

Severity grading, scoring, repair-priority and pass/fail logic are deterministic
and apply identically to baseline output and to a future trained model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from surveyor.fusion import TemporalFuser

logger = logging.getLogger(__name__)

DAMAGE_WEIGHTS = Path("models/damage_detector.pt")

SEVERITY_RANK = {"cosmetic": 1, "minor": 2, "moderate": 3, "severe": 4, "critical": 5}
SEVERITY_PENALTY = {"cosmetic": 2, "minor": 5, "moderate": 12, "severe": 25, "critical": 45}

# defect type → (is_structural, base severity by small/large area)
_STRUCTURAL_TYPES = {"hole", "crack", "deformation", "bent_rod", "puncture", "tear"}


@dataclass
class Defect:
    defect_type: str
    severity: str
    confidence: float
    area_fraction: float
    bbox: list | None = None
    component: str = "body"      # which surface/component (door, roof, wall…)
    structural: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["confidence"] = round(d["confidence"], 4)
        d["area_fraction"] = round(d["area_fraction"], 5)
        return d


@dataclass
class DamageReport:
    defects: list = field(default_factory=list)
    counts_by_severity: dict = field(default_factory=dict)
    counts_by_type: dict = field(default_factory=dict)
    worst_severity: str = "none"
    inspection_score: float = 100.0       # 0-100, higher = better condition
    repair_priority: str = "none"         # none/low/medium/high/urgent
    fit_for_service: bool = True
    method: str = "baseline"
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "defects": [d.as_dict() for d in self.defects],
            "counts_by_severity": self.counts_by_severity,
            "counts_by_type": self.counts_by_type,
            "worst_severity": self.worst_severity,
            "inspection_score": round(self.inspection_score, 1),
            "repair_priority": self.repair_priority,
            "fit_for_service": self.fit_for_service,
            "method": self.method,
            "notes": self.notes,
        }


class DamageAssessor:
    """Pluggable damage assessor with deterministic grading."""

    def __init__(self, weights: Path = DAMAGE_WEIGHTS):
        self._model = None
        self.method = "baseline"
        self._accumulated: list[Defect] = []
        self._frames = 0
        self._load_model(weights)

    def _load_model(self, weights: Path):
        try:
            if weights.exists():
                from ultralytics import YOLO
                self._model = YOLO(str(weights))
                self.method = "model"
                logger.info("DamageAssessor loaded trained model %s", weights)
        except Exception as exc:
            logger.info("Damage model unavailable (%s); using classical baseline.", exc)

    # ── per-frame observation ───────────────────────────────────────────────
    def assess_frame(self, crop, component: str = "body") -> list[Defect]:
        self._frames += 1
        defects = (self._model_detect(crop, component) if self._model is not None
                   else self._baseline_detect(crop, component))
        self._accumulated.extend(defects)
        return defects

    # ── trained-model slot ──────────────────────────────────────────────────
    def _model_detect(self, crop, component) -> list[Defect]:
        rgb = np.array(crop.convert("RGB")) if isinstance(crop, Image.Image) else np.asarray(crop)
        H, W = rgb.shape[:2]
        try:
            res = self._model(rgb, verbose=False, conf=0.25)[0]
        except Exception as exc:
            logger.warning("damage model inference failed: %s", exc)
            return []
        names = res.names or {}
        out: list[Defect] = []
        if res.boxes is not None:
            for b in res.boxes:
                label = names.get(int(b.cls[0]), "defect").lower()
                conf = float(b.conf[0])
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                area = ((x2 - x1) * (y2 - y1)) / float(max(1, W * H))
                out.append(self._grade(label, conf, area, [x1, y1, x2, y2], component))
        return out

    # ── classical baseline ──────────────────────────────────────────────────
    # Deliberately CONSERVATIVE: only flags strong, colour/shape-distinct
    # evidence (heavy rust, clear through-holes) at low confidence.  It does NOT
    # attempt dents/cracks/deformation — those need a trained model, and naive
    # texture heuristics flag undamaged flat walls (false "severe" failures).
    # Baseline severity is capped at "moderate" so heuristics never auto-fail a
    # container; only a trained model (self.method == 'model') may assign
    # severe/critical.
    def _baseline_detect(self, crop, component) -> list[Defect]:
        bgr = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2BGR) \
            if isinstance(crop, Image.Image) else np.asarray(crop)
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        H, W = bgr.shape[:2]
        area_px = float(max(1, H * W))
        defects: list[Defect] = []

        # 1. Rust / corrosion — rusty hues that form connected patches (not just
        #    the container's own orange/brown paint scattered everywhere).
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        rust = cv2.inRange(hsv, (5, 80, 40), (22, 255, 200))
        rust = cv2.morphologyEx(rust, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(rust, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rust_patches = [c for c in cnts if cv2.contourArea(c) > 0.01 * area_px]
        rust_frac = sum(cv2.contourArea(c) for c in rust_patches) / area_px
        if rust_frac > 0.05 and len(rust_patches) >= 1:
            x, y, w, h = cv2.boundingRect(max(rust_patches, key=cv2.contourArea))
            defects.append(self._grade("rust", min(0.45, 0.15 + rust_frac),
                                       rust_frac, [x, y, x + w, y + h], component, cap="moderate"))

        # 2. Holes / punctures — very dark, compact, isolated blobs.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dark = cv2.inRange(gray, 0, 30)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if 0.0008 * area_px < a < 0.02 * area_px:
                x, y, w, h = cv2.boundingRect(c)
                fill = a / float(max(1, w * h))
                if 0.5 < w / max(1, h) < 2.0 and fill > 0.6:  # compact & solid → hole-like
                    defects.append(self._grade("hole", 0.25, a / area_px,
                                               [x, y, x + w, y + h], component, cap="moderate"))

        return defects

    # ── deterministic grading ───────────────────────────────────────────────
    def _grade(self, dtype, conf, area_frac, bbox, component, cap: str | None = None) -> Defect:
        dtype = dtype.lower().replace(" ", "_")
        structural = any(k in dtype for k in _STRUCTURAL_TYPES)
        if structural:
            if area_frac > 0.02 or "crack" in dtype or "hole" in dtype:
                sev = "critical" if area_frac > 0.05 else "severe"
            else:
                sev = "moderate"
        elif "rust" in dtype or "corros" in dtype:
            sev = ("severe" if area_frac > 0.25 else
                   "moderate" if area_frac > 0.08 else
                   "minor" if area_frac > 0.02 else "cosmetic")
        elif "dent" in dtype or "deform" in dtype:
            sev = ("severe" if area_frac > 0.2 else
                   "moderate" if area_frac > 0.06 else "minor")
        elif "scratch" in dtype or "paint" in dtype:
            sev = "cosmetic"
        else:
            sev = "minor"
        # Clamp severity to the cap (used by the conservative classical baseline).
        if cap and SEVERITY_RANK.get(sev, 0) > SEVERITY_RANK.get(cap, 5):
            sev = cap
        return Defect(dtype, sev, conf, area_frac, bbox, component, structural)

    # ── fused report across all observed frames/components ──────────────────
    def report(self) -> DamageReport:
        defects = self._dedup(self._accumulated)
        by_sev: dict[str, int] = {}
        by_type: dict[str, int] = {}
        score = 100.0
        for d in defects:
            by_sev[d.severity] = by_sev.get(d.severity, 0) + 1
            by_type[d.defect_type] = by_type.get(d.defect_type, 0) + 1
            score -= SEVERITY_PENALTY.get(d.severity, 5) * (0.5 + 0.5 * d.confidence)
        score = float(max(0.0, score))

        worst = "none"
        for sev in ("critical", "severe", "moderate", "minor", "cosmetic"):
            if by_sev.get(sev):
                worst = sev
                break

        priority = {"critical": "urgent", "severe": "high", "moderate": "medium",
                    "minor": "low", "cosmetic": "low", "none": "none"}[worst]
        fit = not (worst == "critical" or
                   (worst == "severe" and any(d.structural for d in defects)) or
                   score < 40)

        return DamageReport(
            defects=defects, counts_by_severity=by_sev, counts_by_type=by_type,
            worst_severity=worst, inspection_score=score, repair_priority=priority,
            fit_for_service=fit, method=self.method,
            notes=("Provisional classical-CV detection — plug a trained damage "
                   "model into models/damage_detector.pt for production accuracy."
                   if self.method == "baseline" else ""),
        )

    @staticmethod
    def _dedup(defects: list[Defect]) -> list[Defect]:
        """Collapse repeated detections of the same type/component, keep worst."""
        best: dict[tuple, Defect] = {}
        for d in defects:
            key = (d.defect_type, d.component)
            cur = best.get(key)
            if cur is None or SEVERITY_RANK[d.severity] > SEVERITY_RANK[cur.severity] \
                    or (d.severity == cur.severity and d.confidence > cur.confidence):
                best[key] = d
        return sorted(best.values(), key=lambda d: SEVERITY_RANK[d.severity], reverse=True)

    def reset(self):
        self._accumulated.clear()
        self._frames = 0
