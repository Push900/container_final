"""
surveyor.ocr_engine — industrial OCR engine for container & seal numbers.

Wraps the existing EasyOCR reader (shared singleton from ``ocr_validator`` so the
model loads only once) and adds the heavy preprocessing and multi-frame fusion a
surveyor needs to read numbers under dirt, rust, faded paint, shadows, motion
blur, low light, perspective distortion and partial occlusion:

    perspective correction → denoise → CLAHE contrast → adaptive threshold
    → sharpen / edge enhance → super-resolution upscaling

Each preprocessing produces several *variants* of the input; OCR runs on all of
them and the best valid (check-digit-passing) reading wins.  When several frames
are available, :meth:`fuse_container_readings` merges them per-character.

The seal number is tiny relative to the frame, so :meth:`read_seal_number`
crops, super-resolves and deblurs the seal region before OCR.

This module degrades gracefully: with EasyOCR absent it returns empty results
instead of raising, so the rest of the surveyor keeps working.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from surveyor.iso6346 import is_valid_container_number, check_digit
from surveyor.fusion import fuse_text_by_char

logger = logging.getLogger(__name__)

_STRICT = re.compile(r"[A-Z]{3}[UJZR][0-9]{6}[0-9]")
_RELAXED = re.compile(r"[A-Z]{4}[0-9]{6,7}")
_SEAL_RE = re.compile(r"[A-Z]{0,3}[0-9]{5,9}")
# OCR digit-confusion fixes applied only to the numeric tail of a code.
_DIGIT_SUBS = str.maketrans("OIBSZGQ", "0185296")
_ALPHA_SUBS = str.maketrans("0185", "OIBS")


@dataclass
class OCRReading:
    text: str
    confidence: float
    valid: bool = False          # passes ISO 6346 check digit (container only)
    source: str = ""             # which preprocessing variant produced it
    raw_fragments: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "valid": self.valid,
            "source": self.source,
        }


# ── preprocessing primitives ────────────────────────────────────────────────

def _to_bgr(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    arr = np.asarray(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def perspective_correct(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Best-effort fronto-parallel rectification of the dominant text panel.

    Finds the largest 4-corner contour and warps it to a rectangle.  Returns
    None when no convincing quadrilateral is found (caller keeps the original).
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 0.15 * w * h:
        return None
    peri = cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, 0.02 * peri, True)
    if len(approx) != 4:
        return None
    pts = approx.reshape(4, 2).astype("float32")
    # order corners: tl, tr, br, bl
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect = np.array([
        pts[np.argmin(s)], pts[np.argmin(d)],
        pts[np.argmax(s)], pts[np.argmax(d)],
    ], dtype="float32")
    (tl, tr, br, bl) = rect
    wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
    W = int(max(wA, wB)); H = int(max(hA, hB))
    if W < 20 or H < 20:
        return None
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(bgr, M, (W, H))


def denoise(bgr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)


def clahe_enhance(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def unsharp(bgr: np.ndarray, amount: float = 1.5, radius: int = 3) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (0, 0), radius)
    return cv2.addWeighted(bgr, 1 + amount, blur, -amount, 0)


def deblur(bgr: np.ndarray) -> np.ndarray:
    """Light Wiener-style deblur via sharpening kernel (motion-blur mitigation)."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(bgr, -1, kernel)


def adaptive_binary(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
    )
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


_SR_MODEL = None
_SR_TRIED = False


def _try_load_sr():
    """Load an EDSR/FSRCNN dnn_superres model if weights are present on disk."""
    global _SR_MODEL, _SR_TRIED
    if _SR_TRIED:
        return _SR_MODEL
    _SR_TRIED = True
    try:
        from pathlib import Path
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        for name, scale, fname in [("fsrcnn", 4, "FSRCNN_x4.pb"),
                                   ("edsr", 4, "EDSR_x4.pb"),
                                   ("espcn", 4, "ESPCN_x4.pb")]:
            p = Path("models") / fname
            if p.exists():
                sr.readModel(str(p))
                sr.setModel(name, scale)
                _SR_MODEL = sr
                logger.info("Loaded super-resolution model %s", fname)
                break
    except Exception as exc:  # pragma: no cover - optional path
        logger.info("dnn_superres unavailable (%s); using interpolation upscaling.", exc)
    return _SR_MODEL


def super_resolve(bgr: np.ndarray, scale: int = 4, target_min: int = 320) -> np.ndarray:
    """Upscale a small crop.  Uses a learned SR model if available, else a
    high-quality Lanczos upscale followed by detail enhancement."""
    h, w = bgr.shape[:2]
    if min(h, w) >= target_min and scale <= 1:
        return bgr
    model = _try_load_sr()
    if model is not None and min(h, w) < target_min:
        try:
            return model.upsample(bgr)
        except Exception:
            pass
    factor = max(scale, target_min / max(1, min(h, w)))
    up = cv2.resize(bgr, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_LANCZOS4)
    return unsharp(up, amount=0.8, radius=2)


def build_variants(img, *, want_perspective=True, upscale=True, thorough=False) -> list[tuple[str, np.ndarray]]:
    """Produce labelled preprocessing variants for OCR to try.

    ``thorough=True`` adds the expensive non-local-means denoise pass — worth it
    for a final high-stakes read, too slow for every live frame (the default
    fast set already covers contrast, sharpening, thresholding and deblur).
    """
    bgr = _to_bgr(img)
    variants: list[tuple[str, np.ndarray]] = [("original", bgr)]

    base = bgr
    if upscale and min(bgr.shape[:2]) < 600:
        base = super_resolve(bgr, scale=2, target_min=600)
        variants.append(("upscaled", base))

    if want_perspective:
        warped = perspective_correct(bgr)
        if warped is not None:
            variants.append(("perspective", warped))

    try:
        clahe = clahe_enhance(base)
        variants.append(("clahe", clahe))
        variants.append(("clahe_sharp", unsharp(clahe)))
        variants.append(("binary", adaptive_binary(clahe)))
        variants.append(("deblur", deblur(base)))
        if thorough:
            variants.append(("denoise_clahe", clahe_enhance(denoise(base))))
    except Exception as exc:
        logger.warning("variant generation partial failure: %s", exc)
    return variants


# ── normalisation helpers ───────────────────────────────────────────────────

def _normalise_container(text: str) -> str:
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    # fix the numeric tail and the alpha head separately
    m = re.search(r"([A-Z0-9]{4})([A-Z0-9]+)", t)
    if not m:
        return t
    head, tail = t[:4], t[4:]
    head = head.translate(_ALPHA_SUBS)
    tail = tail.translate(_DIGIT_SUBS)
    return head + tail


# ── the engine ──────────────────────────────────────────────────────────────

class OCREngine:
    """Pluggable OCR engine.  Swap the backend by overriding ``_read_raw``."""

    def __init__(self) -> None:
        self._reader = None
        self._reader_tried = False

    def _get_reader(self):
        if self._reader is not None or self._reader_tried:
            return self._reader
        self._reader_tried = True
        try:
            # reuse the existing shared reader so EasyOCR loads only once
            from ocr_validator import _get_reader as legacy_reader
            self._reader = legacy_reader()
        except Exception as exc:
            logger.warning("OCR reader unavailable: %s", exc)
            self._reader = None
        return self._reader

    def _read_raw(self, bgr: np.ndarray) -> list[tuple[str, float]]:
        """Return raw ``(text, confidence)`` fragments from the OCR backend."""
        reader = self._get_reader()
        if reader is None:
            return []
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dets = reader.readtext(rgb, detail=1, paragraph=False)
            return [(t, float(c)) for (_, t, c) in dets]
        except Exception as exc:
            logger.warning("OCR readtext failed: %s", exc)
            return []

    # ── unified single-pass read (container number + size/type together) ────
    def read_all(self, img, *, min_conf: float = 0.05, thorough: bool = False) -> dict:
        """Run OCR over the preprocessing variants *once* and extract everything.

        Returns ``{"container": OCRReading, "size_type": OCRReading,
        "fragments": [ {text, confidence, source} ... ]}``.  Used by the
        orchestrator so each frame costs one variant-suite pass, not three.
        """
        from surveyor.iso6346 import decode_size_type_code
        best_c = OCRReading(text="", confidence=0.0)
        best_st = OCRReading(text="", confidence=0.0)
        fragments: list[dict] = []
        for source, variant in build_variants(img, thorough=thorough):
            raw = self._read_raw(variant)
            for t, c in raw:
                if c >= min_conf:
                    fragments.append({"text": t, "confidence": round(c, 3), "source": source})
            # container number
            cand = self._extract_container(raw)
            if cand is not None:
                number, conf = cand
                valid = is_valid_container_number(number)
                if (valid and not best_c.valid) or (valid == best_c.valid and conf > best_c.confidence):
                    best_c = OCRReading(number, conf, valid, source)
            # size/type code (4-char marking)
            for t, c in raw:
                norm = re.sub(r"[^A-Z0-9]", "", t.upper())
                m = re.search(r"[1-4LMNPABGHK][0-9CDEFLMNP][GVBSRHUPTAK][0-9]", norm)
                if m and c > best_st.confidence and decode_size_type_code(m.group(0))["valid"]:
                    best_st = OCRReading(m.group(0), c, True, source)
            if best_c.valid and best_c.confidence > 0.4:
                break  # confident & checksum-valid — stop early
        best_c.raw_fragments = fragments
        return {"container": best_c, "size_type": best_st, "fragments": fragments}

    # ── container number ───────────────────────────────────────────────────
    def read_container_number(self, img, *, min_conf: float = 0.05, thorough: bool = False) -> OCRReading:
        """Read & ISO-6346-validate a container number from a single image."""
        return self.read_all(img, min_conf=min_conf, thorough=thorough)["container"]

    def _extract_container(self, fragments: list[tuple[str, float]]):
        """Find the best ISO 6346 candidate within a fragment list."""
        best_num, best_conf, best_valid = "", 0.0, False
        # 1. per-fragment
        for t, c in fragments:
            norm = _normalise_container(t)
            for pat in (_STRICT, _RELAXED):
                m = pat.search(norm)
                if m:
                    num = m.group(0)
                    valid = is_valid_container_number(num)
                    if (valid and not best_valid) or (valid == best_valid and c > best_conf):
                        best_num, best_conf, best_valid = num, c, valid
                    break
        # 2. concatenated (numbers split across boxes)
        if not best_valid and fragments:
            concat = _normalise_container("".join(t for t, _ in fragments))
            maxc = max((c for _, c in fragments), default=0.0)
            for pat in (_STRICT, _RELAXED):
                m = pat.search(concat)
                if m:
                    num = m.group(0)
                    valid = is_valid_container_number(num)
                    if valid or maxc > best_conf:
                        best_num, best_conf, best_valid = num, maxc, valid
                    break
        if not best_num:
            return None
        return best_num, best_conf

    def fuse_container_readings(self, readings: list[OCRReading]) -> OCRReading:
        """Fuse per-frame container readings into the single most reliable one."""
        valid = [r for r in readings if r.valid and r.text]
        pool = valid or [r for r in readings if r.text]
        if not pool:
            return OCRReading("", 0.0)
        fused_text, fused_conf = fuse_text_by_char(
            [(r.text, r.confidence) for r in pool], expected_len=11
        )
        # If fusion broke the checksum but a single valid reading exists, prefer it.
        if not is_valid_container_number(fused_text) and valid:
            best = max(valid, key=lambda r: r.confidence)
            return OCRReading(best.text, best.confidence, True, "fused->best_valid")
        return OCRReading(fused_text, fused_conf, is_valid_container_number(fused_text), "fused")

    # ── ISO size/type code (the 4-char marking below the number) ────────────
    def read_size_type_code(self, img) -> OCRReading:
        from surveyor.iso6346 import decode_size_type_code
        best = OCRReading("", 0.0)
        for source, variant in build_variants(img, want_perspective=False):
            for t, c in self._read_raw(variant):
                norm = re.sub(r"[^A-Z0-9]", "", t.upper())
                m = re.search(r"[1-4LMNPABGHK][0-9CDEFLMNP][GVBSRHUPTAK][0-9]", norm)
                if m and c > best.confidence:
                    code = m.group(0)
                    if decode_size_type_code(code)["valid"]:
                        best = OCRReading(code, c, True, source)
        return best

    # ── seal number (tiny region → super-res → OCR) ─────────────────────────
    def read_seal_number(self, crop, *, min_conf: float = 0.05) -> OCRReading:
        """Aggressively upscale & deblur a small seal crop, then OCR it."""
        bgr = _to_bgr(crop)
        enhanced = super_resolve(bgr, scale=4, target_min=400)
        candidates = [
            ("seal_sr", enhanced),
            ("seal_sr_clahe", clahe_enhance(enhanced)),
            ("seal_sr_sharp", unsharp(clahe_enhance(enhanced))),
            ("seal_sr_deblur", deblur(enhanced)),
            ("seal_sr_binary", adaptive_binary(clahe_enhance(enhanced))),
        ]
        best = OCRReading("", 0.0)
        for source, var in candidates:
            for t, c in self._read_raw(var):
                if c < min_conf:
                    continue
                norm = re.sub(r"[^A-Z0-9]", "", t.upper())
                m = _SEAL_RE.search(norm)
                if m and (c > best.confidence or len(m.group(0)) > len(best.text)):
                    best = OCRReading(m.group(0), c, False, source)
        return best

    def fuse_seal_readings(self, readings: list[OCRReading]) -> OCRReading:
        pool = [r for r in readings if r.text]
        if not pool:
            return OCRReading("", 0.0)
        # seals vary in length; fuse on the modal length
        lengths = [len(r.text) for r in pool]
        modal = max(set(lengths), key=lengths.count)
        fused_text, fused_conf = fuse_text_by_char(
            [(r.text, r.confidence) for r in pool], expected_len=modal
        )
        return OCRReading(fused_text, fused_conf, False, "fused")
