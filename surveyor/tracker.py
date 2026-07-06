"""
surveyor.tracker — lightweight IoU/centroid multi-object tracker.

Assigns a persistent integer ID to each container across video frames so a
single physical container receives one ID for the whole inspection.  Pure NumPy,
no extra dependencies — used as the fallback when ultralytics' built-in
ByteTrack is unavailable, and as the reference implementation for images.

A track also carries an exponential-moving-average confidence so a container is
*confirmed* only after appearing in several consecutive frames (temporal
confirmation), instead of trusting a single-frame spike.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: list
    confidence: float
    hits: int = 1
    age: int = 0                 # frames since creation
    time_since_update: int = 0
    smoothed_conf: float = 0.0
    confirmed: bool = False
    history: list = field(default_factory=list)


class IoUTracker:
    """SORT-style tracker (IoU association + age/hit gating).

    Parameters
    ----------
    iou_threshold   minimum IoU to associate a detection with an existing track.
    max_age         frames a track survives without a matching detection.
    min_hits        consecutive hits before a track becomes ``confirmed``.
    ema             smoothing factor for the per-track confidence.
    """

    def __init__(self, iou_threshold=0.3, max_age=15, min_hits=3, ema=0.6):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.ema = ema
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[dict]) -> list[dict]:
        """Advance one frame.

        ``detections`` is a list of dicts each with ``bbox`` ([x1,y1,x2,y2]) and
        ``confidence``.  Returns the same detections annotated with ``track_id``,
        ``confirmed`` and ``smoothed_conf`` for the matched track.
        """
        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1

        unmatched = list(range(len(detections)))
        # greedy IoU matching, highest IoU first
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di in unmatched:
                score = iou(t.bbox, detections[di]["bbox"])
                if score >= self.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        annotated = [None] * len(detections)
        for score, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            t = self.tracks[ti]
            det = detections[di]
            t.bbox = det["bbox"]
            t.confidence = det["confidence"]
            t.hits += 1
            t.time_since_update = 0
            t.smoothed_conf = self.ema * t.smoothed_conf + (1 - self.ema) * det["confidence"]
            if t.hits >= self.min_hits:
                t.confirmed = True
            t.history.append(det["bbox"])
            annotated[di] = self._annotate(det, t)

        # new tracks for unmatched detections
        for di in range(len(detections)):
            if di in used_d:
                continue
            det = detections[di]
            t = Track(
                track_id=self._next_id, bbox=det["bbox"], confidence=det["confidence"],
                smoothed_conf=det["confidence"], history=[det["bbox"]],
            )
            self._next_id += 1
            self.tracks.append(t)
            annotated[di] = self._annotate(det, t)

        # prune dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [a for a in annotated if a is not None]

    @staticmethod
    def _annotate(det: dict, t: Track) -> dict:
        out = dict(det)
        out["track_id"] = t.track_id
        out["confirmed"] = t.confirmed
        out["smoothed_conf"] = round(t.smoothed_conf, 4)
        out["hits"] = t.hits
        return out

    def reset(self):
        self.tracks.clear()
        self._next_id = 1
