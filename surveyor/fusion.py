"""
surveyor.fusion — evidence accumulation & temporal confidence fusion.

A human surveyor never decides from a single glance; they watch the container
move and build confidence over time.  This module provides that behaviour for
*any* decision in the pipeline (container number, seal status, container type,
individual damage flags…).

Two primitives:

* :class:`TemporalFuser` — weighted confidence voting over discrete labels.
  Feed it ``(label, confidence)`` observations across frames; it returns the
  winning label, a fused confidence, and how much evidence supports it.

* :func:`fuse_text_by_char` — character-level majority voting used to fuse OCR
  strings read from several frames into the single most reliable string.

These are deliberately model-agnostic so newer detectors/OCR engines plug in
without touching the fusion logic.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Hashable, Optional


@dataclass
class FusedValue:
    value: Optional[Hashable]
    confidence: float           # fused confidence in [0, 1]
    support: int                # number of observations backing the winner
    total_observations: int     # observations across all labels
    agreement: float            # winner_weight / total_weight in [0, 1]
    runner_up: Optional[Hashable] = None
    finalized: bool = False     # met the confirmation criteria

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "support": self.support,
            "total_observations": self.total_observations,
            "agreement": round(self.agreement, 4),
            "runner_up": self.runner_up,
            "finalized": self.finalized,
        }


class TemporalFuser:
    """Confidence-weighted voting over a stream of labelled observations.

    Parameters
    ----------
    min_support
        Minimum number of observations for the winner before ``finalized`` is set.
    min_confidence
        Minimum fused confidence before ``finalized`` is set.
    decay
        Per-update multiplicative decay applied to existing weights so recent
        frames matter more (1.0 = no decay / pure cumulative).
    """

    def __init__(
        self,
        min_support: int = 3,
        min_confidence: float = 0.55,
        decay: float = 1.0,
    ) -> None:
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.decay = decay
        self._weight: dict[Hashable, float] = defaultdict(float)
        self._count: dict[Hashable, int] = defaultdict(int)
        self._n = 0

    def observe(self, label: Hashable, confidence: float = 1.0) -> None:
        """Record one observation for *label* with the given confidence."""
        if label is None:
            return
        c = float(max(0.0, min(1.0, confidence)))
        if self.decay != 1.0:
            for k in list(self._weight):
                self._weight[k] *= self.decay
        self._weight[label] += c
        self._count[label] += 1
        self._n += 1

    def observe_many(self, observations) -> None:
        for item in observations:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                self.observe(item[0], item[1])
            else:
                self.observe(item, 1.0)

    @property
    def n_observations(self) -> int:
        return self._n

    def result(self) -> FusedValue:
        if not self._weight:
            return FusedValue(None, 0.0, 0, 0, 0.0, None, False)

        ranked = sorted(self._weight.items(), key=lambda kv: kv[1], reverse=True)
        winner, win_w = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else None
        total_w = sum(self._weight.values())

        agreement = win_w / total_w if total_w else 0.0
        support = self._count[winner]
        # Fused confidence blends agreement (how dominant the winner is) with
        # average per-observation confidence of the winner.
        avg_conf = win_w / support if support else 0.0
        confidence = 0.5 * agreement + 0.5 * avg_conf

        finalized = support >= self.min_support and confidence >= self.min_confidence
        return FusedValue(
            value=winner,
            confidence=float(confidence),
            support=support,
            total_observations=self._n,
            agreement=float(agreement),
            runner_up=runner_up,
            finalized=finalized,
        )

    def reset(self) -> None:
        self._weight.clear()
        self._count.clear()
        self._n = 0


# ── OCR string fusion ────────────────────────────────────────────────────────

def fuse_text_by_char(
    readings: list[tuple[str, float]],
    expected_len: int | None = None,
) -> tuple[str, float]:
    """Fuse multiple OCR strings into one via per-position confidence voting.

    Parameters
    ----------
    readings
        list of ``(text, confidence)`` — strings need not be equal length.
    expected_len
        If given, only readings of that exact length contribute to per-character
        voting (keeps a stray 9-char read from corrupting an 11-char number);
        if no reading matches, falls back to the highest-confidence reading.

    Returns
    -------
    (fused_string, fused_confidence)
    """
    cleaned = [(t.strip().upper(), float(c)) for t, c in readings if t and t.strip()]
    if not cleaned:
        return "", 0.0

    pool = cleaned
    if expected_len is not None:
        same = [(t, c) for t, c in cleaned if len(t) == expected_len]
        if same:
            pool = same
        else:
            # No reading of the right length — return the best raw reading.
            best = max(cleaned, key=lambda x: x[1])
            return best

    length = max(len(t) for t, _ in pool)
    fused_chars: list[str] = []
    per_pos_conf: list[float] = []
    for i in range(length):
        votes: dict[str, float] = defaultdict(float)
        for t, c in pool:
            if i < len(t):
                votes[t[i]] += c
        if not votes:
            continue
        ch, w = max(votes.items(), key=lambda kv: kv[1])
        total = sum(votes.values())
        fused_chars.append(ch)
        per_pos_conf.append(w / total if total else 0.0)

    fused = "".join(fused_chars)
    conf = sum(per_pos_conf) / len(per_pos_conf) if per_pos_conf else 0.0
    return fused, float(conf)
