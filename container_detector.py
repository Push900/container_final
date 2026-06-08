"""
container_detector.py — YOLO-based shipping-container presence detector.

Verifies a shipping container appears in each uploaded image and provides a
bounding-box crop for downstream classification.

Model priority
--------------
1. Custom weights at  models/container_detector.pt  (trained by train_yolo.py)
2. YOLOv8n pretrained on COCO — coarse fallback mode
3. No ultralytics installed  — passthrough mode (accepts every image)

Install ultralytics:
    pip install ultralytics>=8.0.0
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DETECTOR_MODEL_PATH = Path("models/container_detector.pt")
YOLO_FALLBACK_MODEL = "yolov8n.pt"   # auto-downloaded by ultralytics on first use
DETECTION_THRESHOLD = 0.80           # default confidence gate
CROP_PADDING        = 0.05           # fractional padding added around the bbox crop

# COCO classes that serve as rough container proxies when using the fallback model.
# We do NOT restrict to these — any large detection qualifies — but we boost score
# for these classes slightly.
_COCO_CONTAINER_PROXIES = {
    "truck", "bus", "train", "car", "boat", "airplane",
    "suitcase", "backpack",
}


# ── Ultralytics import helper ─────────────────────────────────────────────────

def _try_import_yolo():
    try:
        from ultralytics import YOLO  # noqa: PLC0415
        return YOLO
    except ImportError:
        return None


# ── Main class ────────────────────────────────────────────────────────────────

class ContainerDetector:
    """
    YOLO-based shipping-container presence checker and region extractor.

    Attributes
    ----------
    threshold : float
        Detection confidence gate.  Images below this are classified
        as NOT_CONTAINER and excluded from further processing.
    mode : str
        'custom'      — custom-trained single-class container detector
        'fallback'    — YOLOv8n COCO, largest high-confidence detection used
        'passthrough' — no YOLO available; every image accepted with conf=1.0
    """

    def __init__(
        self,
        model_path: Path = DETECTOR_MODEL_PATH,
        threshold: float = DETECTION_THRESHOLD,
    ):
        self.threshold = threshold
        self.model     = None
        self.mode      = "passthrough"

        YOLO = _try_import_yolo()
        if YOLO is None:
            logger.warning(
                "ultralytics not installed — container detection disabled "
                "(passthrough mode).  Install with:  pip install ultralytics"
            )
            return

        # 1. Try custom weights
        if model_path.exists():
            try:
                self.model = YOLO(str(model_path))
                self.mode  = "custom"
                logger.info("ContainerDetector: loaded custom weights from %s", model_path)
                return
            except Exception as exc:
                logger.warning("Failed to load custom weights (%s) — trying fallback.", exc)

        # 2. Fall back to YOLOv8n COCO
        try:
            self.model = YOLO(YOLO_FALLBACK_MODEL)
            self.mode  = "fallback"
            logger.info("ContainerDetector: using YOLOv8n COCO fallback mode")
        except Exception as exc:
            logger.warning("YOLOv8n fallback also failed (%s) — passthrough mode.", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, pil_img: Image.Image) -> dict:
        """
        Detect a shipping container in a single PIL image.

        Returns
        -------
        dict with keys:
            detected      bool         True if container found above threshold
            confidence    float        Detection confidence (0–1); 1.0 in passthrough
            bbox          list[int]    [x1, y1, x2, y2] pixel coords (full image if not detected)
            crop          PIL.Image    Cropped container region (or full image)
            mode          str          'custom' | 'fallback' | 'passthrough'
            label         str          Detected class label
        """
        if self.mode == "passthrough" or self.model is None:
            return _passthrough_result(pil_img)

        rgb = np.array(pil_img.convert("RGB"))
        try:
            results = self.model(rgb, verbose=False)[0]
        except Exception as exc:
            logger.warning("YOLO inference error: %s — falling back to passthrough.", exc)
            return _passthrough_result(pil_img)

        best = self._best_detection(results, pil_img.size)

        if best is None or best["confidence"] < self.threshold:
            conf = best["confidence"] if best else 0.0
            return {
                "detected":   False,
                "confidence": float(conf),
                "bbox":       [0, 0, pil_img.width, pil_img.height],
                "crop":       pil_img,
                "mode":       self.mode,
                "label":      best["label"] if best else "none",
            }

        crop = _crop_with_padding(pil_img, best["bbox"])
        return {
            "detected":   True,
            "confidence": float(best["confidence"]),
            "bbox":       best["bbox"],
            "crop":       crop,
            "mode":       self.mode,
            "label":      best["label"],
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _best_detection(self, results, img_size: tuple) -> Optional[dict]:
        """Pick the highest-confidence, large-enough detection from YOLO results."""
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return None

        W, H       = img_size
        img_area   = W * H
        names      = results.names or {}
        best_conf  = -1.0
        best_box   = None
        best_label = "object"

        for box in boxes:
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = names.get(cls, str(cls))
            xyxy  = [int(v) for v in box.xyxy[0].tolist()]
            area  = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])

            # In fallback mode skip very small detections (< 5 % of image)
            if self.mode == "fallback" and area < 0.05 * img_area:
                continue

            # In fallback mode boost score for container-proxy classes
            effective_conf = conf
            if self.mode == "fallback" and label.lower() in _COCO_CONTAINER_PROXIES:
                effective_conf = min(1.0, conf * 1.10)

            if effective_conf > best_conf:
                best_conf  = effective_conf
                best_box   = xyxy
                best_label = label

        if best_box is None:
            return None
        return {"confidence": best_conf, "bbox": best_box, "label": best_label}


# ── Module-level helpers ──────────────────────────────────────────────────────

def _passthrough_result(pil_img: Image.Image) -> dict:
    return {
        "detected":   True,
        "confidence": 1.0,
        "bbox":       [0, 0, pil_img.width, pil_img.height],
        "crop":       pil_img,
        "mode":       "passthrough",
        "label":      "container",
    }


def _crop_with_padding(
    pil_img: Image.Image,
    bbox: list,
    pad: float = CROP_PADDING,
) -> Image.Image:
    """Crop bbox from image with fractional padding on all sides."""
    W, H = pil_img.size
    x1, y1, x2, y2 = bbox
    pw = int((x2 - x1) * pad)
    ph = int((y2 - y1) * pad)
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(W, x2 + pw)
    y2 = min(H, y2 + ph)
    return pil_img.crop((x1, y1, x2, y2))
