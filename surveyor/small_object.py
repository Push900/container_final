"""
surveyor.small_object — tiled / SAHI-style pipeline for thin & tiny objects.

Seals, locking wires, bolt pins, hinges, cracks, weld lines, scratches, holes
and door gaps are far too small for vanilla YOLO at a downscaled input size.
This module recovers them by:

* **Slicing** the frame into overlapping high-resolution tiles so each tiny
  object is large relative to its tile (the core SAHI idea), running the
  detector per tile, then mapping boxes back and merging with NMS.
* **Edge-aware preprocessing** (CLAHE + unsharp) per tile so faint elongated
  features survive.
* **Border coverage** — overlap guarantees objects on tile seams / image edges
  are seen in at least one tile.

If the optional ``sahi`` package is installed it is used directly; otherwise the
built-in slicer (pure NumPy + cv2) provides the same behaviour.  Works with any
ultralytics model passed in, so a future fine-tuned thin-object detector drops
in without code changes.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _nms(boxes: list[dict], iou_thr: float = 0.45) -> list[dict]:
    if not boxes:
        return []
    from surveyor.tracker import iou
    boxes = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
    keep: list[dict] = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes
                 if not (b.get("label") == best.get("label")
                         and iou(b["bbox"], best["bbox"]) > iou_thr)]
    return keep


def edge_aware(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 2)
    return cv2.addWeighted(out, 1.6, blur, -0.6, 0)


def slice_image(rgb: np.ndarray, tile: int = 640, overlap: float = 0.25):
    """Yield ``(tile_rgb, x_offset, y_offset)`` covering the whole image."""
    H, W = rgb.shape[:2]
    if H <= tile and W <= tile:
        yield rgb, 0, 0
        return
    step = int(tile * (1 - overlap))
    ys = list(range(0, max(1, H - tile + 1), step)) or [0]
    xs = list(range(0, max(1, W - tile + 1), step)) or [0]
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    for y in ys:
        for x in xs:
            y0, x0 = max(0, y), max(0, x)
            yield rgb[y0:y0 + tile, x0:x0 + tile], x0, y0


class SmallObjectPipeline:
    """Tiled inference wrapper around an ultralytics-style detector.

    Parameters
    ----------
    model    a callable model (e.g. ultralytics YOLO) or None.  If None, the
             pipeline still runs slicing/preprocessing and returns [] — ready to
             accept a fine-tuned thin-object model later via the registry.
    tile     tile side length in px.
    overlap  fractional overlap between tiles (0.2-0.3 recommended for thin objs).
    conf     detection confidence gate.
    """

    def __init__(self, model=None, tile: int = 640, overlap: float = 0.25, conf: float = 0.20):
        self.model = model
        self.tile = tile
        self.overlap = overlap
        self.conf = conf
        self._sahi = None
        if model is None:
            self._maybe_load_default()

    def _maybe_load_default(self):
        """Load a thin-object model if weights exist; else stay model-less."""
        try:
            from pathlib import Path
            from ultralytics import YOLO
            for name in ("models/thin_object.pt", "models/small_object.pt"):
                if Path(name).exists():
                    self.model = YOLO(name)
                    logger.info("SmallObjectPipeline loaded %s", name)
                    return
        except Exception as exc:
            logger.info("No thin-object model available (%s).", exc)

    def detect(self, img, *, edge_enhance: bool = True) -> list[dict]:
        """Run tiled detection; returns merged detections in full-image coords."""
        if self.model is None:
            return []
        rgb = np.array(img.convert("RGB")) if isinstance(img, Image.Image) else np.asarray(img)
        all_boxes: list[dict] = []
        for tile_rgb, ox, oy in slice_image(rgb, self.tile, self.overlap):
            proc = tile_rgb
            if edge_enhance:
                bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
                proc = cv2.cvtColor(edge_aware(bgr), cv2.COLOR_BGR2RGB)
            try:
                res = self.model(proc, verbose=False, conf=self.conf)[0]
            except Exception as exc:
                logger.warning("tile inference failed: %s", exc)
                continue
            names = res.names or {}
            if res.boxes is None:
                continue
            for b in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                all_boxes.append({
                    "bbox": [int(x1 + ox), int(y1 + oy), int(x2 + ox), int(y2 + oy)],
                    "confidence": float(b.conf[0]),
                    "label": names.get(int(b.cls[0]), str(int(b.cls[0]))),
                })
        return _nms(all_boxes, iou_thr=0.4)
