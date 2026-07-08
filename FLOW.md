# Container Intelligence System — Flow

Run `python run.py` → Gradio app at **http://127.0.0.1:7860** with 5 tabs.

**Startup:** check for `models/container_classifier.pth` (train if missing) → load
classifier + OCR, background-load depth model → show tabs.

**Two paths:** Tabs **1→2→3** = quick full/empty triage. Tabs **4→5** = full
autonomous survey. They're independent.

---

### 1 · Detection
Upload image → **Analyse**. Shows: colours → detection badge (passthrough, always
"detected") → OCR number. Orientation only; saves nothing.

### 2 · Classification — FULL / EMPTY / INVALID
Upload container opening → **Classify**. Check order:
`OCR marking-panel gate → detection → quality (blur/brightness/contrast) →
classifier → depth map → decision`. Side data-plate or low confidence → INVALID.

### 3 · Dashboard
**Refresh** → combines latest Screen 1 + 2 into a final verdict (session-only).

### 4 · AI Surveyor — full inspection
Upload 1+ frames of **one** container (+ optional GPS) → **Run** (first run loads
models, ~30–60 s/img). 16-step order:
`detect → track ID → read number → check digit + decode → size/type + markings →
doors → locking bars → seal → seal status → seal number → fuse frames → surfaces →
damage → severity → repair priority → score/decide/sign`.
Output: PASS/FAIL/REVIEW, metric cards, checklist, and a **signed PDF saved to DB**.

### 5 · Inspection History
Filter by container/seal/line/result/date/damage → **Search**. Shows results table
+ verifies the hash-chained audit trail (✔ valid / ✗ tampered).

---

```
Launch → tabs up
   ├─ 1 Detection → 2 Classify → 3 Dashboard      (quick triage)
   └─ 4 AI Surveyor → 5 History                    (full survey → PDF + DB)
```

Source: [app.py](app.py), [inference.py](inference.py),
[surveyor/surveyor.py](surveyor/surveyor.py), [surveyor_ui.py](surveyor_ui.py).
