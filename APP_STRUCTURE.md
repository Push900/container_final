99/# Container Intelligence System — Three-Page Architecture

**Status**: ✅ Running at `http://127.0.0.1:7860`

---

## Overview

A Gradio web application with three specialized pages for freight/container analysis. All backend logic (`inference.py`, `container_detector.py`, `ocr_validator.py`) remains unchanged.

---

## Page 1: Image Understanding & Container Detection

**Purpose**: Initial analysis of any uploaded image.

**Features**:
- **OCR Text Extraction**: Attempts to extract and validate container numbers via EasyOCR
  - Shows: Found/not-found status, extracted number, confidence score, raw OCR snippets
  - Detects if image shows exterior marking panel (auto-routes to Invalid)

- **Dominant Colour Profile**: Extracts 5 dominant colours from the image
  - Displays as interactive colour swatches with hex codes
  - Useful for quality control and visual consistency checks

- **Container Detection**: YOLO-based object detection
  - Classifies image as "Container" or "Not a Container"
  - Shows detection confidence and mode (custom/fallback/passthrough)
  - Flags unclear or invalid inputs with confidence scores

**Workflow**:
1. Upload any image
2. System auto-analyses on change, or click "Analyse Image"
3. View OCR results, colour profile, and detection badge

---

## Page 2: Container State Classification

**Purpose**: Classify confirmed container images as Full / Empty / Invalid.

**Features**:
- **Full EfficientNet-B0 Pipeline**:
  - Image quality gate (blur, brightness, contrast checks)
  - EfficientNet-B0 classifier (transfer-learned on dataset)
  - MiDaS monocular depth estimation
  - Confidence fusion across modalities

- **Verdict Card**:
  - Prediction: FULL / EMPTY / INVALID
  - Confidence percentage
  - Class probability bars (Full / Empty / Invalid)

- **Depth Visualization**:
  - MiDaS depth map with INFERNO colourmap
  - Bright regions = close to camera, Dark = far from camera
  - Useful for understanding the model's visual reasoning

- **Quality & Depth Metrics JSON**:
  - Blur score, brightness, contrast measurements
  - Depth metrics: empty_score, full_score, reliability flag
  - Quality pass/fail status

- **Detailed Reasoning**:
  - Multi-line explanation of the prediction
  - Gates applied (quality, confidence threshold)
  - Conflict detection between classifiers and depth

**Workflow**:
1. Upload a container interior/opening image
2. Click "Classify Container"
3. Review verdict card, depth map, and metrics
4. Read detailed reasoning for explainability

---

## Page 3: Results & Decision Dashboard

**Purpose**: Aggregate outputs from Pages 1 & 2, track history, export results.

**Features**:
- **Live Summary**:
  - Side-by-side blocks for Page 1 (Detection) and Page 2 (Classification)
  - Shows: timestamp, confidence, detected/predicted values, OCR status
  - **Final Verdict**: Synthesized result (e.g., "FULL CONTAINER", "NOT A CONTAINER", "INVALID / UNCLEAR")

- **Session History Log**:
  - Chronological table of all analyses (newest first)
  - Columns: Time, Page, Result, Confidence, OCR Found, Details
  - Automatically populated as you run analyses on Pages 1 & 2

- **Refresh Dashboard**:
  - Updates summary and history from current session state
  - Click after running analyses on Pages 1 or 2

- **Export Results (JSON)**:
  - Downloads a structured JSON file with:
    - Timestamp of export
    - Last result from Page 1 (OCR, colours, detection)
    - Last result from Page 2 (classification, metrics, reasoning)
    - Full session history (all analyses run)
  - Suitable for reports, audits, or downstream processing

---

## Backend Architecture (Unchanged)

### Core Components

**inference.py** (ContainerPredictor):
- Multi-stage pipeline: quality check → classifier → depth analysis
- Classes: EMPTY, FULL, INVALID
- Ensemble voting for multi-image scenarios
- Depth metrics synthesis

**container_detector.py** (ContainerDetector):
- YOLO-based object detection
- Fallback to COCO pretrained if custom model unavailable
- Passthrough mode if YOLO not installed
- Returns: detection flag, confidence, bounding box, cropped image

**ocr_validator.py** (validate_container_number):
- EasyOCR with lazy loading
- ISO 6346 pattern matching (strict + relaxed)
- Marking panel keyword detection
- Per-character and full-number confidence thresholds

---

## Data Flow

### Page 1 Workflow
```
User uploads image
  ↓
OCR validation: extract container number
  ↓
Colour extraction: dominant 5 colours
  ↓
YOLO detection: is this a container?
  ↓
Display results + save to state
```

### Page 2 Workflow
```
User uploads image
  ↓
Quality checks (blur, brightness, contrast)
  ↓
EfficientNet-B0 classification
  ↓
MiDaS depth estimation
  ↓
Confidence fusion
  ↓
Display verdict + metrics + reasoning + save to state
```

### Page 3 Workflow
```
User clicks "Refresh Dashboard"
  ↓
Read Page 1 state + Page 2 state
  ↓
Generate summary HTML with final verdict
  ↓
Rebuild history table from all session entries
  ↓
Display aggregated view

User clicks "Export Results"
  ↓
Serialize Page 1, Page 2, and history to JSON
  ↓
Write to temp file
  ↓
Return download link
```

---

## Key Design Decisions

1. **Stateful UI**:
   - Gradio `gr.State()` objects track Page 1 result, Page 2 result, and history
   - Non-destructive: each analysis appends to history, doesn't overwrite
   - Users can run multiple analyses and compare

2. **No Destructive OCR**:
   - OCR_GATE_ENABLED = False (temporarily disabled due to easyocr init timeout)
   - Can be re-enabled once easyocr dependency is resolved
   - Fallback to passthrough mode in inference.py

3. **Container Detection Passthrough**:
   - YOLO detection requires `ultralytics` package (not installed)
   - Detector gracefully falls back to passthrough mode (accepts all images)
   - Page 1 interface still shows detection confidence, but always passes

4. **CSS & Gradio Compatibility**:
   - CSS is applied via the `gr.Blocks()` constructor to ensure styling is bundled with the UI component.

5. **Stateless Backend**:
   - `inference.py`, `container_detector.py`, `ocr_validator.py` are unchanged
   - Can be imported by FastAPI/Flask apps with zero modifications
   - Gradio is purely a UI wrapper

---

## Deployment Notes

### Prerequisites
- Python 3.13+
- PyTorch 2.0+ (CPU or GPU)
- Gradio 4.0+
- See `requirements.txt` for full list

### Known Limitations
- **EasyOCR**: Currently disabled (init timeout). Can be re-enabled with async loading.
- **YOLO Detection**: `ultralytics` package not installed. Detector runs in passthrough mode.
- **Port**: Defaults to 7860. Set `GRADIO_SERVER_PORT=7861` for alternative port.

### Example Launch
```bash
cd d:\internship\container_final
python app.py
# App available at http://127.0.0.1:7860
```

---

## Future Enhancements

1. **Async OCR Loading**: Load EasyOCR in background without blocking startup
2. **Batch Processing**: Upload multiple images for bulk analysis
3. **Image Comparison**: Side-by-side view of similar images from history
4. **Custom Thresholds**: UI controls for confidence/quality gates
5. **Analytics Dashboard**: Metrics on prediction accuracy, common failure modes
6. **Database Backend**: Persist results for long-term tracking
7. **API Integration**: FastAPI wrapper for integration with external systems

---

**Built**: 2026-06-10  
**Status**: Production-ready UI, demo-mode backend (OCR/YOLO optional)
