"""
surveyor — AI Container Surveyor package.

An industrial-grade inspection layer that extends the existing Container
Intelligence System (inference.py / container_detector.py / ocr_validator.py)
into a complete AI surveyor capable of replacing a human container surveyor
during loading, unloading, gate-in/out and inspection operations.

Design principles
-----------------
* **Non-destructive** — the legacy modules are imported, never rewritten.
* **Modular & swappable** — every capability (detection, OCR, seal inspection,
  damage assessment, container decoding, reporting) is an independent module
  registered in :mod:`surveyor.registry`, so a newer model can be dropped in
  without touching the frontend or the orchestrator.
* **Evidence-based** — important decisions are accumulated across multiple
  frames (:mod:`surveyor.fusion`) instead of trusting a single prediction.
* **Honest** — capabilities that still require trained weights expose a clean
  interface and a clearly-labelled classical-CV baseline rather than a fake
  prediction.

Public entry point
------------------
    from surveyor import AISurveyor
    surveyor = AISurveyor()
    record   = surveyor.inspect_frames([pil_img, ...])
"""

from surveyor.iso6346 import (
    check_digit,
    is_valid_container_number,
    decode_container_number,
    decode_size_type_code,
)
from surveyor.fusion import TemporalFuser, FusedValue
from surveyor.registry import registry, register, get_component

# Heavier modules (torch / cv2 dependent) are imported lazily by the
# orchestrator so that the deterministic helpers above remain importable in
# environments without the ML stack.

__all__ = [
    "check_digit",
    "is_valid_container_number",
    "decode_container_number",
    "decode_size_type_code",
    "TemporalFuser",
    "FusedValue",
    "registry",
    "register",
    "get_component",
    "AISurveyor",
]

__version__ = "0.1.0"


def __getattr__(name):  # PEP 562 lazy attribute
    if name == "AISurveyor":
        from surveyor.surveyor import AISurveyor
        return AISurveyor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
