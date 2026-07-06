# AI Container Surveyor — Architecture & Operations Guide

This document describes the **AI Surveyor extension** added on top of the
existing Container Intelligence System. The original three pages
(Detection / Classification / Dashboard) and all legacy modules
(`inference.py`, `container_detector.py`, `ocr_validator.py`, `app.py`) are
**preserved and untouched in behaviour**; the surveyor is layered beside them.

---

## 1. What was added

A new `surveyor/` package plus two new Gradio tabs:

| Tab | Purpose |
|-----|---------|
| 🔬 **AI Surveyor** | Upload 1+ frames of one container → full autonomous inspection → signed PDF report stored in DB. |
| 📚 **Inspection History** | Search every stored inspection by container no., seal no., date, line, result, damage category. |

### Module map (`surveyor/`)

| Module | Responsibility | Status |
|--------|----------------|--------|
| `registry.py` | Pluggable component registry — swap any model without touching UI/orchestrator. | ✅ complete |
| `iso6346.py` | Check-digit, container-number decode (owner/category/serial), **size/type → length/height/width/type/HC**, BIC owner directory. | ✅ complete, deterministic |
| `ocr_engine.py` | Perspective correction, CLAHE, denoise, sharpen, adaptive threshold, **super-resolution**, multi-frame char-fusion, check-digit gate; seal-crop OCR. | ✅ complete (wraps EasyOCR) |
| `fusion.py` | Temporal confidence voting + per-character OCR fusion. **No decision is single-frame.** | ✅ complete, deterministic |
| `tracker.py` | IoU/SORT tracker → one persistent ID per container. | ✅ complete, deterministic |
| `detection.py` | All-angle container detection + tracking (ByteTrack via ultralytics, IoU fallback) + temporal confirmation. | ✅ complete (uses existing YOLO) |
| `small_object.py` | Tiled / SAHI-style sliced inference + edge-aware preprocessing for thin objects. | ✅ pipeline complete; needs a thin-object model for output |
| `seal_inspector.py` | Locking-bar + seal detection, INTACT/BROKEN/MISSING, seal-type, seal-number OCR, temporal fusion. | ⚠️ interface + **classical baseline**; needs trained weights |
| `damage_assessor.py` | Defect detection, 5-level severity, score, repair priority, fit-for-service. | ⚠️ interface + **conservative classical baseline**; needs trained weights |
| `checklist.py` | ~40-item surveyor checklist driven by gathered evidence. | ✅ complete |
| `database.py` | SQLite store + search + **hash-chained audit trail**. | ✅ complete |
| `report.py` | Professional PDF report (QR + SHA-256 signature + audit trail); HTML fallback. | ✅ complete |
| `surveyor.py` | `AISurveyor` orchestrator running the full 16-step workflow → `InspectionRecord`. | ✅ complete |

---

## 2. Honesty: what is production-ready vs. what needs training

This was built **without fabricating ML models.** Be clear with stakeholders:

**Fully functional today (no training needed):**
- ISO 6346 check-digit validation and full container-number decoding.
- ISO size/type-code decoding → length, height, width, type, High-Cube.
- Owner/operator lookup (offline BIC subset; extend with the full registry).
- Enhanced OCR preprocessing + multi-frame character fusion + check-digit gating.
- Persistent tracking IDs, temporal confidence fusion (every decision is multi-frame).
- SQLite inspection database with full search + tamper-evident audit chain.
- Signed PDF report generation with QR code and audit trail.
- The complete orchestration workflow, scoring, pass/fail and recommendations.

**Interface + baseline only — plug in trained weights for production accuracy:**
- **Seal state** (INTACT/BROKEN/MISSING) and **seal type** — `seal_inspector.py`
  ships a transparent edge-continuity heuristic at deliberately low confidence.
- **Damage detection** — `damage_assessor.py` ships a *conservative* classical
  detector (heavy rust + clear holes only, capped at "moderate") so it never
  auto-fails a container on a heuristic. Dents/cracks/deformation require a model.
- **Thin-object detection** — `small_object.py` slicing is ready but returns
  nothing until a thin-object detector is provided.
- **Container detector** currently runs in **COCO fallback** (no
  `models/container_detector.pt`); a fine-tuned single-class model improves
  all-angle reliability dramatically.

The severity grading, scoring, repair-priority, fusion, DB and reporting logic
are deterministic and apply **identically** to baseline output and to any future
trained model — so accuracy improves the moment real weights are dropped in,
with zero changes to the UI, the database, or the report format.

---

## 3. Plugging in a better model (no UI/backend changes)

Drop weights in `models/` and/or register a component:

```python
from surveyor.registry import register

# Option A: just provide weights at the expected path
#   models/container_detector.pt   → fine-tuned container detector
#   models/seal_detector.pt        → seal detector/classifier (ultralytics)
#   models/damage_detector.pt      → damage detector (ultralytics)
#   models/thin_object.pt          → thin/small-object detector
#   models/FSRCNN_x4.pb            → learned super-resolution (optional)

# Option B: register a custom implementation (override the baseline)
register("damage_assessor", lambda: MyDamageModel("best.pt"), override=True)
```

`AISurveyor` always resolves components through the registry, so the new model
is used on the next inspection. Call `registry.reset(name)` to hot-swap.

---

## 4. The 16-step inspection workflow (`AISurveyor`)

1. Detect container → 2. Assign persistent tracking ID → 3. Read container number
→ 4. Validate ISO check digit → 5. Decode container info → 6. Detect doors →
7. Detect locking bars → 8. Detect seal → 9. Determine seal status → 10. Read seal
number → 11. Compare multiple OCR frames → 12. Inspect surfaces → 13. Detect damage
→ 14. Classify severity → 15. Estimate repair urgency → 16. Score + report.

```python
from surveyor import AISurveyor
s = AISurveyor()                                  # loads models via registry
record = s.inspect_frames([img1, img2, img3],     # several angles of ONE container
                          meta={"gps": "12.97,77.59"})
print(record["result"], record["report_path"])    # PASS/FAIL/REVIEW + PDF path
```

Streaming (live camera) form:

```python
sess = s.start_session({"gps": "..."})
for frame in frames:
    snapshot = s.process_frame(sess, frame)        # live dashboard dict
record = s.finalize(sess)                          # full record + PDF + DB row
```

---

## 5. Running

```bash
pip install -r requirements.txt
python app.py            # http://127.0.0.1:7860  → "🔬 AI Surveyor" tab
```

Artifacts:
- `data/inspections.db` — all inspections + audit log (gitignored).
- `data/reports/*.pdf` — generated reports with QR + signature (gitignored).

**Performance note:** the surveyor runs EasyOCR over several enhanced variants
per frame on CPU, so a thorough single-image inspection takes ~30–60 s; it is a
high-accuracy inspection tool, not a 30 fps realtime detector. Use a GPU
(`easyocr.Reader(gpu=True)` / CUDA torch) and provide fine-tuned YOLO weights for
near-realtime throughput. Heavy non-local-means denoise runs only on the first
frame; subsequent frames use the fast variant suite.

---

## 6. Verified in this build

- ISO 6346 check digit (canonical `CSQU3054383` validates ✅) + size/type decode
  for 20'/40'/45', reefer, open-top, tank, High-Cube.
- Temporal + per-character OCR fusion (correctly out-votes a corrupted reading).
- End-to-end inspection on a real image → container number read, seal classified,
  damage assessed, **PDF generated**, **row saved**, audit chain valid.
- DB search by container/seal/result and hash-chained audit verification.
- Full Gradio app builds with all 5 tabs; legacy pages unchanged.
