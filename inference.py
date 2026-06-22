"""
inference.py — ContainerPredictor: EfficientNet-B0 + MiDaS depth estimation.

Designed as a standalone module so it can be imported by either the Gradio
app (app.py) or a future FastAPI/Flask backend with zero changes.

Single-image API (unchanged):
    predictor = ContainerPredictor()
    result     = predictor.predict(pil_image)   # PIL.Image → dict

Multi-image API (new):
    result = predictor.predict_multiple(pil_images)  # list[PIL.Image] → dict

Single-image result dict keys:
    prediction          str   "EMPTY" | "FULL" | "INVALID"
    confidence          float 0–1
    class_probs         dict  {class: prob}
    depth_map_colored   PIL.Image or None
    depth_metrics       dict
    quality_ok          bool
    quality_issues      list[str]
    quality_metrics     dict
    reasoning           str
    ocr_result          dict  {found, number, confidence, raw_texts, message}

Multi-image result dict keys:
    final_prediction    str   "EMPTY" | "FULL" | "INVALID" | "NOT_CONTAINER"
    final_confidence    float 0–1
    ensemble_probs      dict  {class: averaged_prob}
    num_uploaded        int
    num_detected        int   images with container detected above threshold
    num_selected        int   images actually used in ensemble (≤ TOP_K_IMAGES)
    selected_images     list[dict]  per-image details for selected images
    all_detections      list[dict]  per-image detection results for all inputs
    best_image          PIL.Image or None
    best_depth_map      PIL.Image or None
    reasoning           str
"""

import json
import logging
from pathlib import Path
import threading

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# ── Paths & constants ─────────────────────────────────────────────────────────

MODEL_PATH  = Path("models/container_classifier.pth")
META_PATH   = Path("models/classifier_meta.json")
CLASSES_DEFAULT = ["empty", "full", "invalid"]

IMG_SIZE         = 224
CONF_THRESHOLD   = 0.50   # per-image classifier confidence gate (lowered for better usability)

# Image quality thresholds
BLUR_THRESHOLD   = 80.0
BRIGHTNESS_MIN   = 28
BRIGHTNESS_MAX   = 228
CONTRAST_MIN     = 9.0
MIN_DIMENSION    = 60

MIDAS_MODEL_TYPE = "MiDaS_small"
MIDAS_REPO       = "isl-org/MiDaS"

DEPTH_CONFLICT_CONF_GATE = 0.92

# Multi-image pipeline constants
CONTAINER_DETECTION_THRESHOLD = 0.30   # YOLO gate for NOT_CONTAINER filter (lowered for better usability)
TOP_K_IMAGES                  = 3      # number of best images used in ensemble

# OCR container-number gate (step 0, applied before all other logic).
# Defined here for visibility; actual thresholds live in ocr_validator.py.
# Set OCR_GATE_ENABLED = False to disable the gate without removing the code.
OCR_GATE_ENABLED = True


# ── Classifier helpers ────────────────────────────────────────────────────────

def _build_efficientnet(num_classes: int = 3):
    m = models.efficientnet_b0(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.45, inplace=True),
        nn.Linear(in_f, num_classes),
    )
    return m


def load_classifier(model_path: Path = MODEL_PATH):
    """Load trained EfficientNet-B0 weights; return (model, class_list)."""
    classes = CLASSES_DEFAULT
    if META_PATH.exists():
        meta    = json.loads(META_PATH.read_text())
        classes = meta.get("classes", CLASSES_DEFAULT)

    model = _build_efficientnet(num_classes=len(classes))
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, classes


_CLASSIFIER_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── MiDaS depth helpers ───────────────────────────────────────────────────────

def load_midas(model_type: str = MIDAS_MODEL_TYPE, repo: str = MIDAS_REPO):
    """
    Download (first run) and return (midas_model, midas_transform).
    Falls back gracefully if the hub download fails.
    """
    try:
        midas = torch.hub.load(repo, model_type, trust_repo=True)
        midas.eval()
        hub_tf = torch.hub.load(repo, "transforms", trust_repo=True)
        tf = hub_tf.small_transform if "small" in model_type.lower() else hub_tf.dpt_transform
        return midas, tf
    except Exception as exc:
        print(f"[WARN] MiDaS load failed ({exc}). Depth estimation disabled.")
        return None, None


def estimate_depth(midas, midas_tf, pil_img: Image.Image):
    """
    Run MiDaS on a PIL image; return raw depth map (H×W float32 ndarray).
    MiDaS convention: higher value = closer to camera (inverse depth / disparity).
    """
    rgb = np.array(pil_img.convert("RGB"))
    batch = midas_tf(rgb)

    with torch.no_grad():
        pred = midas(batch)
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return pred.numpy().astype(np.float32)


# ── Depth analysis ────────────────────────────────────────────────────────────

def analyze_depth(raw_depth: np.ndarray) -> dict:
    """
    Derive container-specific depth signals from the MiDaS output.

    Physical interpretation (MiDaS: high = close, low = far):
        EMPTY container: back wall is far away → central region has LOW depth
                         values, overall high spatial variance.
        FULL container:  cargo is close → HIGH depth values everywhere,
                         low spatial variance (flat surface near camera).

    Returns a dict with float metrics and:
        empty_score  [0–1]  how strongly depth suggests EMPTY
        full_score   [0–1]  how strongly depth suggests FULL
        depth_reliable bool  False if the depth map is too flat to be informative
    """
    h, w = raw_depth.shape

    d_min, d_max = raw_depth.min(), raw_depth.max()
    dn = (raw_depth - d_min) / (d_max - d_min + 1e-8)

    cy0, cy1 = h // 3, 2 * h // 3
    cx0, cx1 = w // 3, 2 * w // 3
    center = dn[cy0:cy1, cx0:cx1]

    border = np.concatenate([
        dn[:h // 5, :].ravel(),
        dn[4 * h // 5:, :].ravel(),
        dn[:, :w // 5].ravel(),
        dn[:, 4 * w // 5:].ravel(),
    ])

    gx = np.gradient(dn, axis=1)
    gy = np.gradient(dn, axis=0)
    grad_mean = float(np.sqrt(gx ** 2 + gy ** 2).mean())

    m = {
        "mean_depth":         float(dn.mean()),
        "center_mean_depth":  float(center.mean()),
        "border_mean_depth":  float(border.mean()) if len(border) else 0.5,
        "depth_variance":     float(dn.var()),
        "center_std":         float(center.std()),
        "near_fraction":      float((dn > 0.65).mean()),
        "far_fraction":       float((dn < 0.30).mean()),
        "depth_gradient":     grad_mean,
        "raw_depth_range":    float(d_max - d_min),
    }

    empty_score = (
        (1.0 - m["center_mean_depth"]) * 0.40
        + m["depth_variance"]          * 0.35
        + min(m["depth_gradient"] / 0.15, 1.0) * 0.25
    )

    full_score = (
        m["center_mean_depth"]          * 0.45
        + (1.0 - m["depth_variance"])   * 0.30
        + m["near_fraction"]            * 0.25
    )

    m["empty_score"]    = float(np.clip(empty_score, 0, 1))
    m["full_score"]     = float(np.clip(full_score,  0, 1))

    cv = raw_depth.std() / (raw_depth.mean() + 1e-8)
    m["depth_reliable"] = bool(cv > 0.04)

    return m


def _depth_label(dm: dict) -> str:
    """'empty', 'full', or 'ambiguous'."""
    gap = abs(dm["empty_score"] - dm["full_score"])
    if gap < 0.08 or not dm["depth_reliable"]:
        return "ambiguous"
    return "empty" if dm["empty_score"] > dm["full_score"] else "full"


def colorize_depth(raw_depth: np.ndarray) -> Image.Image:
    """Render a depth map as a colourised PIL image (INFERNO palette)."""
    d_min, d_max = raw_depth.min(), raw_depth.max()
    norm = ((raw_depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored_rgb)


# ── Image quality checks ──────────────────────────────────────────────────────

def check_quality(pil_img: Image.Image) -> tuple:
    """
    Return (is_acceptable, [issues], metrics_dict).
    Checks: minimum size, blur (Laplacian variance), brightness, contrast.
    """
    issues  = []
    metrics = {}

    w, h = pil_img.size
    metrics["width"], metrics["height"] = w, h

    if min(w, h) < MIN_DIMENSION:
        issues.append(f"Image too small ({w}×{h} px; min {MIN_DIMENSION}×{MIN_DIMENSION})")

    grey = np.array(pil_img.convert("L"), dtype=np.uint8)

    lap_var = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    metrics["blur_score"]      = round(lap_var, 2)
    metrics["blur_threshold"]  = BLUR_THRESHOLD
    if lap_var < BLUR_THRESHOLD:
        issues.append(
            f"Image is too blurry (sharpness={lap_var:.1f}, need ≥{BLUR_THRESHOLD})"
        )

    brightness = float(grey.mean())
    metrics["brightness"] = round(brightness, 2)
    if brightness < BRIGHTNESS_MIN:
        issues.append(f"Image too dark (brightness={brightness:.1f}, need ≥{BRIGHTNESS_MIN})")
    elif brightness > BRIGHTNESS_MAX:
        issues.append(f"Image overexposed (brightness={brightness:.1f}, need ≤{BRIGHTNESS_MAX})")

    contrast = float(grey.std())
    metrics["contrast"] = round(contrast, 2)
    if contrast < CONTRAST_MIN:
        issues.append(
            f"Image has very low contrast (std={contrast:.1f}, need ≥{CONTRAST_MIN})"
        )

    return len(issues) == 0, issues, metrics


def image_quality_score(
    pil_img: Image.Image,
    detection_conf: float = 1.0,
) -> tuple:
    """
    Compute a composite image quality score in [0, 1].

    Components and weights
    ----------------------
    blur_score            0.25  Laplacian variance, sigmoid-normalised around BLUR_THRESHOLD
    brightness_score      0.20  Gaussian proximity to the optimal brightness range
    detection_confidence  0.30  YOLO detection confidence (already 0–1)
    opening_visibility    0.15  Canny edge density in the central image region
    frame_visibility      0.10  Canny edge density at image borders (container frame)

    Returns
    -------
    (composite: float, breakdown: dict)
    """
    grey = np.array(pil_img.convert("L"), dtype=np.uint8)
    h, w = grey.shape

    # 1. Blur — sigmoid around BLUR_THRESHOLD (steeper slope = sharper transition)
    lap_var    = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    blur_score = float(1.0 / (1.0 + np.exp(-0.04 * (lap_var - BLUR_THRESHOLD))))

    # 2. Brightness — Gaussian centred at 128 with σ=70
    brightness       = float(grey.mean())
    brightness_score = float(np.exp(-((brightness - 128.0) ** 2) / (2 * 70.0 ** 2)))

    # 3. Opening visibility — Canny edge density in the central 50% of the image
    edges    = cv2.Canny(grey, 50, 150)
    cy0, cy1 = h // 4, 3 * h // 4
    cx0, cx1 = w // 4, 3 * w // 4
    opening_visibility = float(
        min(1.0, edges[cy0:cy1, cx0:cx1].mean() / 255.0 * 8.0)
    )

    # 4. Frame/corner visibility — edge density at image borders
    bs     = max(2, min(h, w) // 8)
    border = np.zeros_like(edges)
    border[:bs, :]   = edges[:bs, :]
    border[-bs:, :]  = edges[-bs:, :]
    border[:, :bs]   = edges[:, :bs]
    border[:, -bs:]  = edges[:, -bs:]
    frame_visibility = float(min(1.0, border.mean() / 255.0 * 10.0))

    composite = (
        blur_score           * 0.25
        + brightness_score   * 0.20
        + float(detection_conf) * 0.30
        + opening_visibility * 0.15
        + frame_visibility   * 0.10
    )

    breakdown = {
        "blur_score":         round(blur_score, 3),
        "brightness_score":   round(brightness_score, 3),
        "detection_conf":     round(float(detection_conf), 3),
        "opening_visibility": round(opening_visibility, 3),
        "frame_visibility":   round(frame_visibility, 3),
        "composite":          round(float(composite), 3),
    }

    return float(composite), breakdown


# ── Main predictor ────────────────────────────────────────────────────────────

class ContainerPredictor:
    """
    End-to-end predictor.  Load once; call predict() or predict_multiple().
    Thread-safety: not guaranteed for a single instance; wrap in a lock for
    concurrent use, or create one instance per worker.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        depth_model_type: str = MIDAS_MODEL_TYPE,
        detection_threshold: float = CONTAINER_DETECTION_THRESHOLD,
    ):
        print("  Loading classifier …", end=" ", flush=True)
        self.classifier, self.classes = load_classifier(model_path)
        print("done")

        # MiDaS Depth Model - Async Loading
        self.midas = None
        self.midas_tf = None
        self._depth_loading = True
        
        def _load_depth():
            print("  Starting MiDaS depth model background load …")
            m, tf = load_midas(depth_model_type)
            self.midas = m
            self.midas_tf = tf
            self._depth_loading = False
            if self.midas:
                print("\n  [INFO] MiDaS depth model loaded successfully.")
            else:
                print("\n  [WARN] MiDaS depth model failed to load.")

        threading.Thread(target=_load_depth, daemon=True).start()

        print("  Loading container detector …", end=" ", flush=True)
        try:
            from container_detector import ContainerDetector  # noqa: PLC0415
            self.detector = ContainerDetector(threshold=detection_threshold)
            print(f"done ({self.detector.mode} mode)")
        except Exception as exc:
            logger.warning("ContainerDetector unavailable (%s) — passthrough mode.", exc)
            self.detector = None

        # OCR container-number validation gate.
        # We load this asynchronously because EasyOCR can take a long time to
        # initialize/download models on the first run, blocking app startup.
        self._ocr_validate = None
        self._ocr_loading = False
        if OCR_GATE_ENABLED:
            self._ocr_loading = True
            print("  Starting OCR validator background load …")
            threading.Thread(target=self._init_ocr_async, daemon=True).start()

    def _init_ocr_async(self):
        """Initialize EasyOCR in a background thread."""
        try:
            from ocr_validator import validate_container_number, _get_reader
            _get_reader()  # Force initialization of the Reader singleton (downloads models)
            self._ocr_validate = validate_container_number
        except Exception as exc:
            logger.error("OCR background initialization failed: %s", exc)
        finally:
            self._ocr_loading = False
            print("\n  [INFO] OCR validator background load complete.")

    # ── Single-image public API (unchanged) ───────────────────────────────────

    def predict(self, pil_img: Image.Image) -> dict:
        """
        Analyse a container image and return a result dict (see module docstring).
        """
        result: dict = {}

        # ── Step 0: OCR container-number gate ─────────────────────────────────
        # Runs on the ORIGINAL image (before any YOLO crop).
        # Only hard-blocks images that are identified as the container SIDE WALL
        # (marking panel with TARE / MAX GR keywords).  A "number not found" result
        # is NOT an early exit — container opening images (interior view) have no
        # visible number plate, and the YOLO + classifier pipeline already handles
        # them correctly.  The ocr_result is attached to every downstream return
        # path for display in the UI quality panel.
        if self._ocr_validate is not None:
            ocr = self._ocr_validate(pil_img)
            if ocr.get("is_marking_panel"):
                return {
                    "prediction":        "INVALID",
                    "confidence":        0.0,
                    "class_probs":       {c: 0.0 for c in self.classes},
                    "depth_map_colored": None,
                    "depth_metrics":     {},
                    "quality_ok":        False,
                    "quality_issues":    ["Container marking panel — not an opening"],
                    "quality_metrics":   {},
                    "ocr_result":        ocr,
                    "reasoning": (
                        f"Container marking panel detected (number: {ocr.get('number', '')})\n\n"
                        "This image shows the exterior data plate of the container\n"
                        "(weight, dimensions, ISO code) rather than the container opening.\n\n"
                        "Recommendations:\n"
                        "  • Photograph the container from the door/opening end.\n"
                        "  • The system classifies container status (full/empty) from the\n"
                        "    interior view through the opening, not from the side wall."
                    ),
                }
        else:
            ocr = {
                "found": True, "number": "", "confidence": 1.0,
                "raw_texts": [], "is_marking_panel": False,
                "message": "OCR gate disabled.",
            }

        # 1. Detection Gate (New Integration)
        if self.detector is not None:
            det = self.detector.detect(pil_img)
            if not det["detected"]:
                return {
                    "prediction":        "INVALID",
                    "confidence":        det["confidence"],
                    "class_probs":       {c: 0.0 for c in self.classes},
                    "depth_map_colored": None,
                    "depth_metrics":     {},
                    "quality_ok":        False,
                    "quality_issues":    ["No container detected in image."],
                    "quality_metrics":   {},
                    "reasoning":         (
                        f"Container detection failed (conf={det['confidence']:.2f}). "
                        "Please ensure the container is clearly visible in the frame."
                    ),
                }
            # Use the cropped container for subsequent analysis
            pil_img = det["crop"]

        # Attach OCR result so all downstream return paths include it.
        result["ocr_result"] = ocr

        # 2. Quality gate
        q_ok, q_issues, q_metrics = check_quality(pil_img)
        result["quality_ok"]      = q_ok
        result["quality_issues"]  = q_issues
        result["quality_metrics"] = q_metrics

        if not q_ok:
            return {
                **result,
                "prediction":        "INVALID",
                "confidence":        0.0,
                "class_probs":       {c: 0.0 for c in self.classes},
                "depth_map_colored": None,
                "depth_metrics":     {},
                "reasoning":         self._quality_reasoning(q_issues),
            }

        # 3. Classifier
        tensor = _CLASSIFIER_TF(pil_img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            logits = self.classifier(tensor)
            probs  = F.softmax(logits, dim=1).squeeze(0).numpy()

        class_probs = {c: float(p) for c, p in zip(self.classes, probs)}
        pred_idx    = int(np.argmax(probs))
        pred_cls    = self.classes[pred_idx]
        confidence  = float(probs[pred_idx])

        result["class_probs"] = class_probs
        result["confidence"]  = confidence

        # 4. Depth estimation
        dm: dict                   = {}
        depth_img: Image.Image | None = None
        depth_signal               = "unavailable"

        # Wait for async depth model load if still in progress (max 10 seconds)
        wait_time = 0
        while self._depth_loading and wait_time < 10:
            import time
            time.sleep(0.1)
            wait_time += 0.1

        if self.midas is not None:
            try:
                raw_depth    = estimate_depth(self.midas, self.midas_tf, pil_img)
                dm           = analyze_depth(raw_depth)
                depth_img    = colorize_depth(raw_depth)
                depth_signal = _depth_label(dm)
            except Exception as exc:
                print(f"[WARN] Depth estimation error: {exc}")

        result["depth_metrics"]     = dm
        result["depth_map_colored"] = depth_img

        # 5. Decision logic
        if pred_cls == "invalid":
            return {
                **result,
                "prediction": "INVALID",
                "reasoning": (
                    f"The classifier assigned this image to the INVALID class "
                    f"({confidence*100:.1f}% confidence). The image may lack a "
                    "clear container opening, be cropped incorrectly, show "
                    "an unrelated scene, or contain inconsistent visual cues."
                ),
            }

        if confidence < CONF_THRESHOLD:
            return {
                **result,
                "prediction": "INVALID",
                "reasoning": (
                    f"Model confidence too low: {confidence*100:.1f}% "
                    f"(threshold {CONF_THRESHOLD*100:.0f}%). "
                    "The image is ambiguous. Please upload a clearer, "
                    "well-lit, frontal photo of the container opening."
                ),
            }

        signals_conflict = (
            depth_signal not in ("ambiguous", "unavailable")
            and depth_signal != pred_cls
            and dm.get("depth_reliable", False)
            and confidence < DEPTH_CONFLICT_CONF_GATE
        )

        if signals_conflict:
            return {
                **result,
                "prediction": "INVALID",
                "reasoning": (
                    f"Conflicting signals detected.\n"
                    f"  • Classifier → {pred_cls.upper()} ({confidence*100:.1f}%)\n"
                    f"  • Depth map  → {depth_signal.upper()} "
                    f"(empty_score={dm['empty_score']:.2f}, "
                    f"full_score={dm['full_score']:.2f})\n\n"
                    "This usually means the image does not clearly show the "
                    "container interior. Please upload a direct, unobstructed "
                    "frontal view of the container opening."
                ),
            }

        # 6. Confirmed prediction
        return {
            **result,
            "prediction": pred_cls.upper(),
            "reasoning":  self._build_reasoning(
                pred_cls, confidence, dm, depth_signal, signals_conflict
            ),
        }

    # ── Multi-image public API (new) ──────────────────────────────────────────

    def predict_multiple(
        self,
        pil_images: list,
        top_k: int = TOP_K_IMAGES,
        detection_threshold: float | None = None,
    ) -> dict:
        """
        Multi-image container inspection pipeline.

        Pipeline
        --------
        1. YOLO container detection (per image)
        2. NOT_CONTAINER filter
        3. Image quality scoring (blur, brightness, detection conf, edge density)
        4. Top-K selection (highest quality score)
        5. EfficientNet classification (per selected image)
        6. MiDaS depth analysis (per selected image)
        7. Confidence fusion (classifier × depth agreement)
        8. Ensemble voting (probability averaging across top-K images)

        Parameters
        ----------
        pil_images          : list of PIL.Image objects (5–10 recommended)
        top_k               : number of best images to use for ensemble
        detection_threshold : override ContainerDetector threshold

        Returns
        -------
        Structured dict — see module docstring for full key list.
        """
        if not pil_images:
            return self._empty_result(0, "No images were provided.")

        threshold = (
            detection_threshold
            if detection_threshold is not None
            else (self.detector.threshold if self.detector else CONTAINER_DETECTION_THRESHOLD)
        )

        num_uploaded = len(pil_images)
        all_detections: list[dict] = []

        # ── Step 0 + 1–2: OCR gate, YOLO detection, NOT_CONTAINER filter ────────
        # The OCR gate (step 0) runs on the original image before any YOLO crop
        # so exterior container-number markings remain visible.  Images that
        # lack a recognisable ISO 6346 container number are excluded; if none
        # of the uploaded images pass OCR the pipeline exits early.
        for idx, img in enumerate(pil_images):

            # Step 0: OCR container-number gate (per image, on original)
            if self._ocr_validate is not None:
                ocr = self._ocr_validate(img)
            else:
                ocr = {
                    "found": True, "number": "", "confidence": 1.0,
                    "raw_texts": [], "is_marking_panel": False,
                    "message": "OCR gate disabled.",
                }

            # Step 1: YOLO container detection
            if self.detector is not None:
                det = self.detector.detect(img)
            else:
                det = {
                    "detected": True, "confidence": 1.0,
                    "bbox": [0, 0, img.width, img.height],
                    "crop": img, "mode": "passthrough", "label": "container",
                }

            # Apply threshold override if provided
            if detection_threshold is not None and det["confidence"] < threshold:
                det["detected"] = False

            # Marking-panel images are excluded from classification (they show the
            # exterior data plate, not the opening).  "number not found" alone does
            # NOT exclude an image — opening/interior views have no visible number.
            is_panel = ocr.get("is_marking_panel", False)
            all_detections.append({
                "image_idx":      idx,
                "detected":       det["detected"] and not is_panel,
                "det_confidence": round(det["confidence"], 3),
                "bbox":           det["bbox"],
                "label":          det["label"],
                "det_mode":       det["mode"],
                "crop":           det["crop"],
                "original":       img,
                "ocr_valid":      ocr["found"],
                "ocr_number":     ocr["number"],
                "ocr_confidence": round(ocr["confidence"], 3),
                "ocr_message":    ocr["message"],
                "is_marking_panel": is_panel,
            })

        detected = [d for d in all_detections if d["detected"]]
        num_detected = len(detected)

        if num_detected == 0:
            panels = [d for d in all_detections if d.get("is_marking_panel")]
            if panels and len(panels) == len(all_detections):
                reason = (
                    "All uploaded images show the container exterior marking panel,\n"
                    "not the container opening.\n\n"
                    "Please photograph the container from the door/opening end."
                )
            else:
                reason = "No shipping containers detected."
            return {
                **self._empty_result(num_uploaded, reason),
                "all_detections": [
                    {k: v for k, v in d.items() if k not in ("crop", "original")}
                    for d in all_detections
                ],
                "num_uploaded": num_uploaded,
            }

        # ── Step 3: Image quality scoring ─────────────────────────────────────
        for d in detected:
            qs, qb = image_quality_score(d["crop"], d["det_confidence"])
            d["quality_score"]     = round(qs, 3)
            d["quality_breakdown"] = qb

        # ── Step 4: Top-K selection ────────────────────────────────────────────
        selected = sorted(detected, key=lambda x: x["quality_score"], reverse=True)
        selected = selected[:top_k]
        num_selected = len(selected)

        # ── Steps 5–7: Classification + depth + confidence fusion ─────────────
        for d in selected:
            cls_result   = self._run_classifier(d["crop"])
            depth_result = self._run_depth(d["crop"])

            fused = self._fuse_confidence(
                cls_result["pred_cls"],
                cls_result["confidence"],
                depth_result["depth_metrics"],
            )
            d.update({
                "prediction":       cls_result["pred_cls"].upper(),
                "class_probs":      cls_result["class_probs"],
                "confidence":       round(cls_result["confidence"], 3),
                "depth_metrics":    depth_result["depth_metrics"],
                "depth_map_colored": depth_result["depth_map_colored"],
                "depth_confidence": round(depth_result["depth_confidence"], 3),
                "fused_confidence": round(fused, 3),
            })

        # ── Step 8: Ensemble voting (average probabilities) ───────────────────
        ensemble_probs, final_pred, final_conf = self._ensemble_vote(selected)

        # Best image = first in sorted selected (highest quality score)
        best    = selected[0]
        best_img = best["original"]
        best_dm  = best.get("depth_map_colored")

        # Clean per-image output (remove PIL objects for JSON serialisability)
        selected_summary = [
            {
                "image_idx":        d["image_idx"],
                "quality_score":    d["quality_score"],
                "detection_conf":   d["det_confidence"],
                "prediction":       d["prediction"],
                "confidence":       d["confidence"],
                "class_probs":      d["class_probs"],
                "depth_confidence": d["depth_confidence"],
                "fused_confidence": d["fused_confidence"],
                "quality_breakdown": d["quality_breakdown"],
            }
            for d in selected
        ]

        all_det_summary = [
            {
                "image_idx":      d["image_idx"],
                "detected":       d["detected"],
                "det_conf":       d["det_confidence"],
                "label":          d["label"],
                "quality_score":  d.get("quality_score"),
                "selected":       d in selected,
                "ocr_valid":      d["ocr_valid"],
                "ocr_number":     d["ocr_number"],
                "ocr_confidence": d["ocr_confidence"],
            }
            for d in all_detections
        ]

        reasoning = self._multi_reasoning(
            num_uploaded, num_detected, num_selected,
            selected_summary, final_pred, final_conf, ensemble_probs,
        )

        return {
            "final_prediction":  final_pred,
            "final_confidence":  round(final_conf, 3),
            "ensemble_probs":    {k: round(v, 3) for k, v in ensemble_probs.items()},
            "num_uploaded":      num_uploaded,
            "num_detected":      num_detected,
            "num_selected":      num_selected,
            "selected_images":   selected_summary,
            "all_detections":    all_det_summary,
            "best_image":        best_img,
            "best_depth_map":    best_dm,
            "reasoning":         reasoning,
        }

    # ── Internal pipeline helpers ─────────────────────────────────────────────

    def _run_classifier(self, pil_img: Image.Image) -> dict:
        """Run EfficientNet on a single image; return raw probabilities."""
        tensor = _CLASSIFIER_TF(pil_img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            logits = self.classifier(tensor)
            probs  = F.softmax(logits, dim=1).squeeze(0).numpy()

        class_probs = {c: float(p) for c, p in zip(self.classes, probs)}
        pred_idx    = int(np.argmax(probs))
        return {
            "pred_cls":   self.classes[pred_idx],
            "confidence": float(probs[pred_idx]),
            "class_probs": class_probs,
        }

    def _run_depth(self, pil_img: Image.Image) -> dict:
        """Run MiDaS on a single image; return depth metrics and confidence."""
        if self.midas is None:
            return {
                "depth_metrics":     {},
                "depth_map_colored": None,
                "depth_confidence":  0.5,
            }
        try:
            raw     = estimate_depth(self.midas, self.midas_tf, pil_img)
            dm      = analyze_depth(raw)
            img     = colorize_depth(raw)
            # Depth confidence = how decisive the depth signal is
            gap     = abs(dm.get("empty_score", 0) - dm.get("full_score", 0))
            d_conf  = float(np.clip(gap * 2.0, 0.0, 1.0)) if dm.get("depth_reliable") else 0.4
            return {
                "depth_metrics":     dm,
                "depth_map_colored": img,
                "depth_confidence":  d_conf,
            }
        except Exception as exc:
            logger.warning("Depth estimation error: %s", exc)
            return {
                "depth_metrics":     {},
                "depth_map_colored": None,
                "depth_confidence":  0.5,
            }

    def _fuse_confidence(
        self,
        pred_cls: str,
        classifier_conf: float,
        dm: dict,
    ) -> float:
        """
        Fuse classifier confidence with depth signal agreement.

        Depth agreement boosts the fused score; conflict penalises it.
        When depth is unavailable or ambiguous, falls back to classifier alone.
        """
        if not dm:
            return classifier_conf

        depth_sig = _depth_label(dm)

        if depth_sig == "ambiguous" or not dm.get("depth_reliable", False):
            depth_conf = classifier_conf
        elif depth_sig == pred_cls:
            # Depth agrees: add partial boost
            strength   = abs(dm.get("empty_score", 0) - dm.get("full_score", 0))
            depth_conf = min(1.0, classifier_conf + strength * 0.15)
        else:
            # Depth conflicts: penalty
            depth_conf = max(0.0, classifier_conf - 0.15)

        fused = 0.70 * classifier_conf + 0.30 * depth_conf
        return float(np.clip(fused, 0.0, 1.0))

    def _ensemble_vote(self, selected: list) -> tuple:
        """
        Average class probabilities across selected images.

        Returns (ensemble_probs dict, final_prediction str, final_confidence float).
        The ensemble can still yield "INVALID" if the averaged confidence is below
        CONF_THRESHOLD or if the plurality class is "invalid".
        """
        # Average raw classifier probabilities (unweighted)
        ensemble_probs = {c: 0.0 for c in self.classes}
        for d in selected:
            for cls, prob in d["class_probs"].items():
                ensemble_probs[cls] = ensemble_probs.get(cls, 0.0) + prob

        n = len(selected)
        ensemble_probs = {c: v / n for c, v in ensemble_probs.items()}

        pred_idx    = int(np.argmax([ensemble_probs[c] for c in self.classes]))
        final_cls   = self.classes[pred_idx]
        final_conf  = float(ensemble_probs[final_cls])

        if final_cls == "invalid" or final_conf < CONF_THRESHOLD:
            return ensemble_probs, "INVALID", final_conf

        return ensemble_probs, final_cls.upper(), final_conf

    @staticmethod
    def _empty_result(num_uploaded: int, reason: str) -> dict:
        return {
            "final_prediction":  "NOT_CONTAINER",
            "final_confidence":  0.0,
            "ensemble_probs":    {"empty": 0.0, "full": 0.0, "invalid": 1.0},
            "num_uploaded":      num_uploaded,
            "num_detected":      0,
            "num_selected":      0,
            "selected_images":   [],
            "all_detections":    [],
            "best_image":        None,
            "best_depth_map":    None,
            "reasoning":         reason,
        }

    @staticmethod
    def _multi_reasoning(
        num_uploaded: int,
        num_detected: int,
        num_selected: int,
        selected: list,
        final_pred: str,
        final_conf: float,
        ensemble_probs: dict,
    ) -> str:
        lines = [
            f"ENSEMBLE PREDICTION: {final_pred}  ({final_conf*100:.1f}% confidence)\n",
            "── Pipeline Summary ─────────────────────────────────",
            f"  Images uploaded  : {num_uploaded}",
            f"  Containers found : {num_detected}",
            f"  Images selected  : {num_selected} (top-{num_selected} by quality score)",
            "",
            "── Ensemble Probabilities ────────────────────────────",
        ]
        for cls, prob in sorted(ensemble_probs.items(), key=lambda x: -x[1]):
            lines.append(f"  {cls.capitalize():8s}: {prob*100:.1f}%")

        lines += ["", "── Per-Image Analysis ────────────────────────────────"]
        for i, d in enumerate(selected, 1):
            lines.append(
                f"  Image {d['image_idx']+1:2d} (rank {i})"
                f"  quality={d['quality_score']:.2f}"
                f"  det={d['detection_conf']:.2f}"
                f"  pred={d['prediction']}"
                f"  conf={d['confidence']*100:.1f}%"
                f"  fused={d['fused_confidence']:.2f}"
            )

        if final_pred == "INVALID":
            lines += [
                "",
                "⚠ Final prediction is INVALID because ensemble confidence is below "
                f"the {CONF_THRESHOLD*100:.0f}% threshold or the model voted INVALID.",
                "  Recommendations:",
                "  • Use more images with a clear, well-lit container opening.",
                "  • Ensure the container opening fills most of the frame.",
            ]
        elif final_pred == "NOT_CONTAINER":
            lines += [
                "",
                "⚠ No shipping containers were detected above the confidence threshold.",
                "  Please upload photos that clearly show a shipping container.",
            ]

        return "\n".join(lines)

    # ── Reasoning builders (single-image, unchanged) ──────────────────────────

    @staticmethod
    def _quality_reasoning(issues: list) -> str:
        lines = ["Image quality check FAILED — the image cannot be reliably classified.\n"]
        for iss in issues:
            lines.append(f"  • {iss}")
        lines.append(
            "\nRecommendations:\n"
            "  • Use a well-lit environment (natural or bright artificial light).\n"
            "  • Hold the camera steady to avoid motion blur.\n"
            "  • Frame the shot so the container opening fills most of the image.\n"
            "  • Avoid extreme angles or heavy cropping."
        )
        return "\n".join(lines)

    @staticmethod
    def _build_reasoning(
        cls: str,
        conf: float,
        dm: dict,
        depth_signal: str,
        had_conflict: bool,
    ) -> str:
        lines = [
            f"PREDICTION: {cls.upper()}  ({conf*100:.1f}% confidence)\n",
            "── Classifier ─────────────────────────────",
            f"  Predicted class : {cls.upper()}",
            f"  Confidence      : {conf*100:.1f}%  (threshold {CONF_THRESHOLD*100:.0f}%)",
        ]

        if dm:
            agree = depth_signal == cls
            lines += [
                "",
                "── Depth Analysis (MiDaS) ──────────────────",
                f"  Depth signal    : {depth_signal.upper()}",
                f"  Agrees with cls : {'YES ✓' if agree else 'PARTIAL (high-confidence override)'}",
                f"  Center depth    : {dm.get('center_mean_depth', 0):.2f}  (0=far · 1=near)",
                f"  Depth variance  : {dm.get('depth_variance', 0):.3f}",
                f"  Near-object %   : {dm.get('near_fraction', 0)*100:.1f}%",
                f"  Far-object  %   : {dm.get('far_fraction', 0)*100:.1f}%",
                f"  Empty score     : {dm.get('empty_score', 0):.2f}",
                f"  Full  score     : {dm.get('full_score', 0):.2f}",
            ]

            if cls == "empty":
                depth_note = (
                    "Depth map shows a clear depth gradient — back wall of the "
                    "container visible at distance, indicating an empty interior."
                    if agree else
                    "Depth map is ambiguous, but classifier confidence is high "
                    f"(≥{DEPTH_CONFLICT_CONF_GATE*100:.0f}%) so prediction stands."
                )
            else:
                depth_note = (
                    "Depth map shows reduced visible depth — cargo near the "
                    "camera is occupying interior space."
                    if agree else
                    "Depth map is ambiguous, but classifier confidence is high "
                    f"(≥{DEPTH_CONFLICT_CONF_GATE*100:.0f}%) so prediction stands."
                )

            lines += ["", f"  Interpretation: {depth_note}"]
        else:
            lines.append("\n  Depth estimation unavailable — decision based on classifier only.")

        if had_conflict:
            lines.append(
                f"\n  ⚠ Note: depth signal partially conflicted but confidence "
                f"≥{DEPTH_CONFLICT_CONF_GATE*100:.0f}% — classifier result retained."
            )

        return "\n".join(lines)


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not img_path:
        print("Usage: python inference.py <image_path> [image_path2 ...]")
        sys.exit(1)

    print("\nLoading predictor …")
    p = ContainerPredictor()

    paths = sys.argv[1:]
    if len(paths) == 1:
        img = Image.open(paths[0]).convert("RGB")
        r   = p.predict(img)
        print(f"\n{'='*50}")
        print(f"PREDICTION : {r['prediction']}")
        print(f"CONFIDENCE : {r['confidence']*100:.1f}%")
        print(f"PROBS      : { {k: f'{v*100:.1f}%' for k, v in r['class_probs'].items()} }")
        print(f"\n{r['reasoning']}")
    else:
        imgs = [Image.open(p_).convert("RGB") for p_ in paths]
        r    = p.predict_multiple(imgs)
        print(f"\n{'='*50}")
        print(f"FINAL      : {r['final_prediction']}")
        print(f"CONFIDENCE : {r['final_confidence']*100:.1f}%")
        print(f"ENSEMBLE   : { {k: f'{v*100:.1f}%' for k, v in r['ensemble_probs'].items()} }")
        print(f"UPLOADED   : {r['num_uploaded']}  DETECTED: {r['num_detected']}  SELECTED: {r['num_selected']}")
        print(f"\n{r['reasoning']}")
