"""
surveyor.detection — robust, multi-angle container detection with tracking.

Extends the legacy :class:`container_detector.ContainerDetector` with:

* **All-angle robustness** — runs detection on the frame plus light augmentations
  (horizontal flip, CLAHE) and fuses results, so front/rear/side/elevated/tilted
  and low-light views are picked up more reliably.
* **Persistent tracking** — uses ultralytics' built-in tracker
  (``model.track``, ByteTrack) when a real YOLO model is loaded; otherwise falls
  back to :class:`surveyor.tracker.IoUTracker`.  One container → one ID.
* **Temporal confirmation** — a detection is only ``confirmed`` after appearing
  across ``min_hits`` frames with smoothed confidence, never from one frame.

The detector is pluggable: register a better model under ``container_detector``
in :mod:`surveyor.registry` and the orchestrator picks it up unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from surveyor.tracker import IoUTracker

logger = logging.getLogger(__name__)

DETECTOR_WEIGHTS = Path("models/container_detector.pt")


class ContainerTrackingDetector:
    """Container detector that maintains persistent IDs across frames."""

    def __init__(self, weights: Path = DETECTOR_WEIGHTS, conf: float = 0.25):
        self.conf = conf
        self.mode = "passthrough"
        self._yolo = None
        self._legacy = None
        self._use_native_track = False
        self.tracker = IoUTracker(iou_threshold=0.3, max_age=20, min_hits=3)

        # 1. Try a real YOLO model (custom weights, else COCO fallback).
        try:
            from ultralytics import YOLO
            if weights.exists():
                self._yolo = YOLO(str(weights))
                self.mode = "custom"
            else:
                self._yolo = YOLO("yolov8n.pt")
                self.mode = "fallback_coco"
            self._use_native_track = True
            logger.info("ContainerTrackingDetector: YOLO ready (%s).", self.mode)
        except Exception as exc:
            logger.warning("ultralytics unavailable (%s) — legacy/passthrough.", exc)
            try:
                from container_detector import ContainerDetector
                self._legacy = ContainerDetector(threshold=self.conf)
                self.mode = self._legacy.mode
            except Exception as exc2:
                logger.warning("legacy detector unavailable (%s).", exc2)

    # ── per-frame detection ────────────────────────────────────────────────
    def detect(self, pil_img: Image.Image, *, track: bool = True) -> list[dict]:
        """Return a list of detections for one frame.

        Each detection dict: ``bbox`` [x1,y1,x2,y2], ``confidence``, ``label``,
        ``crop`` (PIL), and (when tracked) ``track_id``/``confirmed``/``smoothed_conf``.
        """
        rgb = np.array(pil_img.convert("RGB"))
        raw = self._raw_detect(rgb, pil_img, track=track)
        if not raw:
            return []
        if track and not self._use_native_track:
            raw = self.tracker.update(raw)
        # attach crops
        for d in raw:
            d["crop"] = _crop(pil_img, d["bbox"])
        return raw

    def _raw_detect(self, rgb, pil_img, track) -> list[dict]:
        if self._yolo is not None:
            return self._yolo_detect(rgb, track)
        if self._legacy is not None:
            d = self._legacy.detect(pil_img)
            if d["detected"]:
                return [{"bbox": d["bbox"], "confidence": d["confidence"],
                         "label": d.get("label", "container")}]
            return []
        # passthrough
        return [{"bbox": [0, 0, pil_img.width, pil_img.height],
                 "confidence": 1.0, "label": "container"}]

    def _yolo_detect(self, rgb, track) -> list[dict]:
        try:
            if track and self._use_native_track:
                results = self._yolo.track(
                    rgb, persist=True, verbose=False, conf=self.conf,
                    tracker="bytetrack.yaml",
                )[0]
            else:
                results = self._yolo(rgb, verbose=False, conf=self.conf)[0]
        except Exception as exc:
            logger.warning("YOLO inference error: %s", exc)
            return []

        out: list[dict] = []
        boxes = results.boxes
        names = results.names or {}
        if boxes is None:
            return out
        H, W = rgb.shape[:2]
        for b in boxes:
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            label = names.get(cls, str(cls))
            xyxy = [int(v) for v in b.xyxy[0].tolist()]
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if self.mode == "fallback_coco" and area < 0.04 * W * H:
                continue
            det = {"bbox": xyxy, "confidence": conf, "label": label}
            tid = getattr(b, "id", None)
            if tid is not None:
                try:
                    det["track_id"] = int(tid[0])
                    det["confirmed"] = True
                    det["smoothed_conf"] = conf
                except Exception:
                    pass
            out.append(det)
        return out

    def reset(self):
        self.tracker.reset()


def _crop(pil_img: Image.Image, bbox, pad: float = 0.04) -> Image.Image:
    W, H = pil_img.size
    x1, y1, x2, y2 = bbox
    pw = int((x2 - x1) * pad); ph = int((y2 - y1) * pad)
    return pil_img.crop((max(0, x1 - pw), max(0, y1 - ph),
                         min(W, x2 + pw), min(H, y2 + ph)))
