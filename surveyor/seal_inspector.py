"""
surveyor.seal_inspector — door locking-bar & seal inspection.

Goal: locate the container door locking bars, find the seal attached to them,
and classify it as one of three states accumulated over multiple frames:

    INTACT  — seal present and its pin/cable is continuous
    BROKEN  — seal present but the pin/cable is physically severed
    MISSING — locking bar present, no seal where one is expected

It also reads the (tiny) seal number once a seal looks intact, by cropping the
seal region and pushing it through the super-resolution OCR path in
:mod:`surveyor.ocr_engine`.

Honesty note
------------
Reliable seal-state classification and seal-type recognition (bolt / cable /
plastic / high-security ISO) require a trained detector + classifier on labelled
seal imagery, which is not shipped here.  This module therefore provides:

  * a clean, pluggable interface (`SealInspector.inspect_frame`) and a model slot
    (`models/seal_detector.pt`) that a future model plugs into unchanged, and
  * a transparent **classical-CV baseline** (edge-continuity heuristic) that
    produces a *provisional* status with calibrated low confidence.

Because the baseline is heuristic, the inspector **never** finalises from one
frame: evidence is fused across frames with :class:`surveyor.fusion.TemporalFuser`
and the status is only finalised once enough agreeing frames are seen.
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

INTACT, BROKEN, MISSING, UNKNOWN = "INTACT", "BROKEN", "MISSING", "UNKNOWN"
SEAL_TYPES = ("bolt", "cable", "plastic", "high_security_iso", "unknown")
SEAL_WEIGHTS = Path("models/seal_detector.pt")


@dataclass
class SealResult:
    status: str = UNKNOWN
    status_confidence: float = 0.0
    finalized: bool = False
    seal_type: str = "unknown"
    seal_number: str = ""
    seal_number_confidence: float = 0.0
    frames_observed: int = 0
    locking_bar_found: bool = False
    method: str = "baseline"          # 'model' once a trained detector is plugged in
    notes: str = ""
    seal_bbox: list | None = None

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("status_confidence", "seal_number_confidence"):
            d[k] = round(d[k], 4)
        return d


class SealInspector:
    """Pluggable seal inspector with temporal evidence fusion."""

    def __init__(self, weights: Path = SEAL_WEIGHTS, ocr_engine=None):
        self._model = None
        self._ocr = ocr_engine
        self.method = "baseline"
        self._fuser = TemporalFuser(min_support=3, min_confidence=0.5)
        self._type_fuser = TemporalFuser(min_support=2, min_confidence=0.4)
        self._seal_readings = []
        self._frames = 0
        self._last_seal_bbox = None
        self._bar_found_any = False
        self._load_model(weights)

    def _load_model(self, weights: Path):
        try:
            if weights.exists():
                from ultralytics import YOLO
                self._model = YOLO(str(weights))
                self.method = "model"
                logger.info("SealInspector loaded trained model %s", weights)
        except Exception as exc:
            logger.info("Seal model unavailable (%s); using classical baseline.", exc)

    def _get_ocr(self):
        if self._ocr is None:
            from surveyor.ocr_engine import OCREngine
            self._ocr = OCREngine()
        return self._ocr

    # ── per-frame entry point ───────────────────────────────────────────────
    def inspect_frame(self, door_crop) -> dict:
        """Observe one frame of the door region; accumulate evidence.

        ``door_crop`` should be the door / locking-bar region of the container
        (the orchestrator supplies the right ROI).  Returns the current fused
        :class:`SealResult` as a dict.
        """
        self._frames += 1
        if self._model is not None:
            obs = self._model_observe(door_crop)
        else:
            obs = self._baseline_observe(door_crop)

        status, status_conf, seal_bbox, bar_found, seal_type, type_conf = obs
        self._bar_found_any = self._bar_found_any or bar_found
        if seal_bbox is not None:
            self._last_seal_bbox = seal_bbox

        if status != UNKNOWN:
            self._fuser.observe(status, status_conf)
        if seal_type and seal_type != "unknown":
            self._type_fuser.observe(seal_type, type_conf)

        # Attempt to read the seal number when a seal appears present.
        if status in (INTACT, BROKEN) and seal_bbox is not None:
            self._read_seal(door_crop, seal_bbox)

        return self.result().as_dict()

    # ── fused result ────────────────────────────────────────────────────────
    def result(self) -> SealResult:
        fused = self._fuser.result()
        type_fused = self._type_fuser.result()
        seal = self._best_seal_number()
        status = fused.value or (MISSING if (self._bar_found_any and self._frames >= 3
                                             and not self._seal_readings) else UNKNOWN)
        return SealResult(
            status=status,
            status_confidence=fused.confidence,
            finalized=fused.finalized,
            seal_type=type_fused.value or "unknown",
            seal_number=seal[0],
            seal_number_confidence=seal[1],
            frames_observed=self._frames,
            locking_bar_found=self._bar_found_any,
            method=self.method,
            seal_bbox=self._last_seal_bbox,
            notes=("Provisional classical-CV estimate — plug a trained seal model "
                   "into models/seal_detector.pt for production accuracy."
                   if self.method == "baseline" else ""),
        )

    # ── trained-model path (slot) ───────────────────────────────────────────
    def _model_observe(self, door_crop):
        """Use a trained seal detector/classifier when available.

        Expected class names include locking_bar / seal_intact / seal_broken /
        seal_missing and seal types.  This adapts whatever labels the model
        exposes; unknown labels are ignored.
        """
        rgb = np.array(door_crop.convert("RGB")) if isinstance(door_crop, Image.Image) else np.asarray(door_crop)
        try:
            res = self._model(rgb, verbose=False, conf=0.2)[0]
        except Exception as exc:
            logger.warning("seal model inference failed: %s", exc)
            return UNKNOWN, 0.0, None, False, "unknown", 0.0
        names = res.names or {}
        status, conf, bbox, bar, stype, tconf = UNKNOWN, 0.0, None, False, "unknown", 0.0
        if res.boxes is not None:
            for b in res.boxes:
                label = names.get(int(b.cls[0]), "").lower()
                c = float(b.conf[0])
                xyxy = [int(v) for v in b.xyxy[0].tolist()]
                if "bar" in label:
                    bar = True
                elif "intact" in label and c > conf:
                    status, conf, bbox = INTACT, c, xyxy
                elif ("broken" in label or "cut" in label) and c > conf:
                    status, conf, bbox = BROKEN, c, xyxy
                elif "missing" in label and c > conf:
                    status, conf = MISSING, c
                for t in SEAL_TYPES:
                    if t in label and c > tconf:
                        stype, tconf = t, c
        return status, conf, bbox, bar, stype, tconf

    # ── classical baseline (transparent heuristic) ──────────────────────────
    def _baseline_observe(self, door_crop):
        """Heuristic: find vertical locking bars; look for a seal blob on the
        right-hand bar; judge pin continuity via vertical-edge connectivity."""
        bgr = cv2.cvtColor(np.array(door_crop.convert("RGB")), cv2.COLOR_RGB2BGR) \
            if isinstance(door_crop, Image.Image) else np.asarray(door_crop)
        if bgr.ndim == 3 and bgr.shape[2] == 3 and isinstance(door_crop, Image.Image):
            pass
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        H, W = gray.shape[:2]
        if H < 40 or W < 40:
            return UNKNOWN, 0.0, None, False, "unknown", 0.0

        # vertical locking bars → strong vertical gradients in narrow columns
        sob = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        vstrength = np.abs(sob).sum(axis=0)
        vstrength = cv2.GaussianBlur(vstrength.reshape(1, -1), (1, 9), 0).ravel()
        thr = vstrength.mean() + 1.5 * vstrength.std()
        bar_cols = np.where(vstrength > thr)[0]
        bar_found = len(bar_cols) >= 2
        if not bar_found:
            return UNKNOWN, 0.0, None, bar_found, "unknown", 0.0

        # Seal usually sits mid-height on a locking bar handle. Search a band
        # around the strongest bar for a compact bright/dark blob.
        cx = int(bar_cols[np.argmax(vstrength[bar_cols])])
        band = gray[H // 4: 3 * H // 4, max(0, cx - W // 12): min(W, cx + W // 12)]
        if band.size == 0:
            return UNKNOWN, 0.0, None, bar_found, "unknown", 0.0
        _, blob = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 0.01 * band.size]
        if not contours:
            # bar but no seal blob → provisional MISSING (low confidence)
            return MISSING, 0.25, None, bar_found, "unknown", 0.0

        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        seal_bbox = [int(max(0, cx - W // 12) + x), int(H // 4 + y),
                     int(max(0, cx - W // 12) + x + w), int(H // 4 + y + h)]
        # Continuity check: a continuous pin/cable connects above & below blob.
        roi = gray[max(0, seal_bbox[1] - h): min(H, seal_bbox[3] + h),
                   max(0, seal_bbox[0]): min(W, seal_bbox[2])]
        edges = cv2.Canny(roi, 50, 150)
        col_density = edges.mean(axis=0) if edges.size else np.array([0])
        continuity = float((col_density > 0).mean())
        if continuity > 0.5:
            return INTACT, 0.3, seal_bbox, bar_found, "unknown", 0.0
        return BROKEN, 0.28, seal_bbox, bar_found, "unknown", 0.0

    # ── seal-number OCR ─────────────────────────────────────────────────────
    def _read_seal(self, door_crop, seal_bbox):
        try:
            if isinstance(door_crop, Image.Image):
                crop = door_crop.crop(tuple(seal_bbox))
            else:
                x1, y1, x2, y2 = seal_bbox
                crop = Image.fromarray(np.asarray(door_crop)[y1:y2, x1:x2])
            reading = self._get_ocr().read_seal_number(crop)
            if reading.text:
                self._seal_readings.append(reading)
        except Exception as exc:
            logger.warning("seal OCR failed: %s", exc)

    def _best_seal_number(self):
        if not self._seal_readings:
            return "", 0.0
        fused = self._get_ocr().fuse_seal_readings(self._seal_readings)
        return fused.text, fused.confidence

    def reset(self):
        self._fuser.reset(); self._type_fuser.reset()
        self._seal_readings.clear()
        self._frames = 0
        self._last_seal_bbox = None
        self._bar_found_any = False
