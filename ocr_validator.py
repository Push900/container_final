"""
ocr_validator.py — Container number OCR validation gate.

Reads the container number from an image using EasyOCR and validates it
against the ISO 6346 format:
    4 letters  (3 owner-code letters + 1 equipment-category letter U/J/Z/R)
  + 6 digits   (serial number)
  + 1 digit    (check digit, sometimes omitted)
  Example: MSCU 123456 7  →  MSCU1234567

This module is imported lazily by ContainerPredictor.__init__.  If EasyOCR is
not installed the gate is automatically disabled (passthrough mode) so the rest
of the pipeline is unaffected.

Thresholds
----------
OCR_CHAR_CONF_THRESHOLD   Minimum per-detection EasyOCR confidence (default 0.10).
                          Detections below this are discarded before pattern matching.
                          Kept very low because real-world container markings (dirty,
                          angled, partial) often produce sub-0.3 EasyOCR scores.

OCR_NUMBER_CONF_THRESHOLD Minimum confidence to accept a container number as valid
                          and allow the image to proceed to classification (default 0.15).
                          Images that fail this threshold are routed to INVALID with
                          the message "No container number identified".
"""

import re
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Confidence thresholds ─────────────────────────────────────────────────────

# Kept intentionally low: real containers photographed at angles, with dirt or
# partial occlusion, routinely produce EasyOCR scores well below 0.30.
# The gate is meant to reject images with NO container text (e.g. a photo of a
# car), not to penalise images where the number is visible but OCR is uncertain.
OCR_CHAR_CONF_THRESHOLD = 0.10

OCR_NUMBER_CONF_THRESHOLD = 0.15

# ── ISO 6346 container-number patterns ───────────────────────────────────────

# Strict ISO 6346: 3 owner-code letters + 1 category letter (U/J/Z/R)
# + 6 serial digits + optional check digit.
_STRICT_PATTERN = re.compile(r'[A-Z]{3}[UJZR][0-9]{6}[0-9]?', re.IGNORECASE)

# Relaxed fallback: any 4 letters + 6–7 digits.
_RELAXED_PATTERN = re.compile(r'[A-Z]{4}[0-9]{6,7}', re.IGNORECASE)

# ── Marking-panel keyword detector ───────────────────────────────────────────
# These words appear on the container SIDE WALL (weight/capacity data plate),
# not on the container opening.  Detecting them means the image shows the
# exterior marking panel rather than the opening → route to INVALID.
_PANEL_KWS = re.compile(
    r'(MAX\s*GR|TARE|PAYLOAD|CU\.?\s*CAP|CUBIC\s*CAP|CSC\s*SAFETY|NET\s*WEIGHT|MGW)',
    re.IGNORECASE,
)

# ── EasyOCR lazy singleton ────────────────────────────────────────────────────

_reader = None  # loaded at most once per process


def _get_reader():
    """Return the shared EasyOCR Reader, initialising it on first call."""
    global _reader
    if _reader is not None:
        return _reader
    try:
        import easyocr  # noqa: PLC0415
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        logger.info("EasyOCR reader loaded (CPU mode, English).")
        return _reader
    except ImportError:
        logger.warning(
            "easyocr not installed — container number OCR disabled.  "
            "Install with:  pip install easyocr"
        )
        return None
    except Exception as exc:
        logger.warning("EasyOCR initialisation failed: %s", exc)
        return None


# ── Text normalisation ────────────────────────────────────────────────────────

# Common OCR substitutions on the digit side of the container number.
# Applied after splitting the string at the letter/digit boundary so we don't
# accidentally turn the owner-code letters into digits.
_DIGIT_SUBS = str.maketrans("OIBS", "0185")

def _normalise(text: str) -> str:
    """Strip whitespace and punctuation, uppercase, fix common OCR digit errors."""
    t = re.sub(r'[\s\-_.,/|\\]', '', text).upper()
    # Find first run of digits (or digit-lookalikes) and apply substitutions only
    # to the numeric portion so letter owner codes are not corrupted.
    m = re.search(r'[0-9OIBS]{4,}', t)
    if m:
        t = t[:m.start()] + m.group(0).translate(_DIGIT_SUBS) + t[m.end():]
    return t


# ── Public API ────────────────────────────────────────────────────────────────

def validate_container_number(pil_img: Image.Image) -> dict:
    """
    Attempt to identify a container number (ISO 6346) in a PIL image.

    Pipeline
    --------
    1. Run EasyOCR on the full image.
    2. Discard detections below OCR_CHAR_CONF_THRESHOLD.
    3. Normalise each fragment and test against ISO 6346 patterns (strict first,
       then relaxed).  Track the highest-confidence match.
    4. If no individual fragment matches, concatenate all surviving fragments
       into a single string and retry the pattern search.
    5. Accept the best match only when its confidence ≥ OCR_NUMBER_CONF_THRESHOLD.

    Parameters
    ----------
    pil_img : PIL.Image
        Full image (before any YOLO crop) so exterior markings are visible.

    Returns
    -------
    dict
        found            bool   True if a valid container number was found.
        number           str    Detected number (e.g. "MSCU1234567") or "" if none.
        confidence       float  Confidence score in [0, 1] for the best match.
        raw_texts        list   [{"text": str, "confidence": float}, ...]
        message          str    Human-readable status.
        is_marking_panel bool   True when marking-panel keywords (TARE, MAX GR …)
                                are detected — image shows the exterior side wall,
                                not the container opening.
    """
    reader = _get_reader()

    # Gate disabled — easyocr unavailable; let the image pass through.
    if reader is None:
        return {
            "found":            True,
            "number":           "",
            "confidence":       1.0,
            "raw_texts":        [],
            "is_marking_panel": False,
            "message":          "OCR unavailable — container number validation skipped.",
        }

    rgb = np.array(pil_img.convert("RGB"))

    try:
        detections = reader.readtext(rgb, detail=1)
    except Exception as exc:
        logger.warning("EasyOCR readtext failed: %s", exc)
        return {
            "found":            False,
            "number":           "",
            "confidence":       0.0,
            "raw_texts":        [],
            "is_marking_panel": False,
            "message":          f"OCR processing error — {exc}",
        }

    # ── Step 1: Collect confident fragments; check each individually ──────────
    raw_texts:   list[dict] = []
    best_number: str        = ""
    best_conf:   float      = 0.0
    panel_detected:  bool   = False

    for (_, text, conf) in detections:
        conf = float(conf)
        if conf < OCR_CHAR_CONF_THRESHOLD:
            continue
        raw_texts.append({"text": text, "confidence": round(conf, 3)})
        if _PANEL_KWS.search(text):
            panel_detected = True
        norm = _normalise(text)
        for pattern in (_STRICT_PATTERN, _RELAXED_PATTERN):
            m = pattern.search(norm)
            if m and conf > best_conf:
                best_number = m.group(0).upper()
                best_conf   = conf
                break  # take the strictest match for this fragment

    if best_number and best_conf >= OCR_NUMBER_CONF_THRESHOLD:
        if panel_detected:
            return {
                "found":            True,
                "number":           best_number,
                "confidence":       round(best_conf, 3),
                "raw_texts":        raw_texts,
                "is_marking_panel": True,
                "message":          (
                    f"Marking panel detected (number: {best_number}) — "
                    "not a container opening"
                ),
            }
        return {
            "found":            True,
            "number":           best_number,
            "confidence":       round(best_conf, 3),
            "raw_texts":        raw_texts,
            "is_marking_panel": False,
            "message":          (
                f"Container number identified: {best_number} "
                f"(conf={best_conf:.2f})"
            ),
        }

    # ── Step 2: Concatenated-fragment fallback ────────────────────────────────
    # Container numbers are often split across OCR boxes ("ATRU" + "816125" + "8").
    # Use MAX confidence so unrelated low-confidence boxes don't drag the score down.
    if raw_texts:
        concat   = _normalise(" ".join(d["text"] for d in raw_texts))
        max_conf = max(d["confidence"] for d in raw_texts)
        for pattern in (_STRICT_PATTERN, _RELAXED_PATTERN):
            m = pattern.search(concat)
            if m and max_conf >= OCR_NUMBER_CONF_THRESHOLD:
                number = m.group(0).upper()
                if panel_detected:
                    return {
                        "found":            True,
                        "number":           number,
                        "confidence":       round(max_conf, 3),
                        "raw_texts":        raw_texts,
                        "is_marking_panel": True,
                        "message":          (
                            f"Marking panel detected (number: {number}) — "
                            "not a container opening"
                        ),
                    }
                return {
                    "found":            True,
                    "number":           number,
                    "confidence":       round(max_conf, 3),
                    "raw_texts":        raw_texts,
                    "is_marking_panel": False,
                    "message":          (
                        f"Container number identified: {number} "
                        f"(conf={max_conf:.2f})"
                    ),
                }

    # ── No valid container number found ──────────────────────────────────────
    return {
        "found":            False,
        "number":           "",
        "confidence":       round(best_conf, 3),
        "raw_texts":        raw_texts,
        "is_marking_panel": False,
        "message":          "No container number identified",
    }
