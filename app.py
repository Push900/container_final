"""
app.py — Gradio web interface for the Container Status Classifier.

Run:
    python app.py

Prerequisites:
    1. Train the model first:  python train.py
    2. (MiDaS weights download automatically on first run — ~82 MB)

Backend integration note:
    ContainerPredictor in inference.py is framework-agnostic.
    To plug into FastAPI/Flask, import it and call predictor.predict(pil_image).
    The Gradio layer here is purely a thin UI wrapper.
"""

import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from inference import ContainerPredictor, MODEL_PATH

# ── Predictor (loaded once at startup) ───────────────────────────────────────

if not MODEL_PATH.exists():
    sys.exit(
        f"\n[ERROR] Trained model not found at '{MODEL_PATH}'.\n"
        "  Please run:  python train.py\n"
        "  then restart this app."
    )

print("\nInitialising ContainerPredictor …")
predictor = ContainerPredictor()
print("Ready.\n")

# ── UI helpers ────────────────────────────────────────────────────────────────

_COLORS = {
    "EMPTY":   ("#16a34a", "#f0fdf4"),   # green
    "FULL":    ("#ea580c", "#fff7ed"),   # orange
    "INVALID": ("#dc2626", "#fef2f2"),   # red
}

_ICONS = {"EMPTY": "📦", "FULL": "🧱", "INVALID": "⚠️"}


def _verdict_html(prediction: str, confidence: float, class_probs: dict) -> str:
    border_c, bg_c = _COLORS.get(prediction, ("#6b7280", "#f9fafb"))
    icon = _ICONS.get(prediction, "?")

    prob_bars = ""
    for cls, p in class_probs.items():
        bar_color = _COLORS.get(cls.upper(), ("#6b7280", ""))[0]
        prob_bars += f"""
        <div style="display:flex; align-items:center; margin:5px 0; font-size:0.9em;">
          <span style="width:62px; color:#374151; font-weight:500;">{cls.capitalize()}</span>
          <div style="flex:1; background:#e5e7eb; border-radius:6px; height:14px; margin:0 10px;">
            <div style="width:{p*100:.1f}%; background:{bar_color}; height:14px;
                        border-radius:6px; transition:width 0.3s;"></div>
          </div>
          <span style="width:46px; text-align:right; color:#6b7280;">{p*100:.1f}%</span>
        </div>"""

    return f"""
    <div style="border:2px solid {border_c}; border-radius:14px; background:{bg_c};
                padding:18px 22px; margin-bottom:8px;">
      <div style="font-size:2.4em; font-weight:700; color:{border_c};
                  display:flex; align-items:center; gap:10px;">
        <span>{icon}</span>
        <span>{prediction}</span>
      </div>
      <div style="color:#6b7280; font-size:1.05em; margin-top:4px;">
        Classifier confidence: <strong>{confidence*100:.1f}%</strong>
      </div>
      <div style="margin-top:14px;">
        <div style="font-size:0.85em; color:#9ca3af; margin-bottom:6px;">CLASS PROBABILITIES</div>
        {prob_bars}
      </div>
    </div>
    """


def _no_model_depth_html() -> str:
    return """
    <div style="color:#9ca3af; text-align:center; padding:40px; font-size:0.95em;">
        Depth map unavailable<br>(MiDaS not loaded)
    </div>"""


# ── Core prediction function called by Gradio ─────────────────────────────────

def classify(image: np.ndarray):
    """
    Gradio callback.
    Input:  numpy HxWx3 RGB array (from gr.Image type="numpy")
    Returns: depth_img, verdict_html, reasoning_text, quality_json
    """
    if image is None:
        return None, "<p style='color:#9ca3af;'>Upload an image to get started.</p>", "", {}

    pil_img = Image.fromarray(image).convert("RGB")
    result  = predictor.predict(pil_img)

    # Depth map
    depth_out = (
        np.array(result["depth_map_colored"])
        if result["depth_map_colored"] is not None
        else None
    )

    verdict_html = _verdict_html(
        result["prediction"],
        result["confidence"],
        result["class_probs"],
    )

    quality_dict = {
        "quality_ok":    result["quality_ok"],
        "issues":        result["quality_issues"],
        "blur_score":    result["quality_metrics"].get("blur_score"),
        "brightness":    result["quality_metrics"].get("brightness"),
        "contrast":      result["quality_metrics"].get("contrast"),
        "image_size":    f"{result['quality_metrics'].get('width')}×{result['quality_metrics'].get('height')}",
    }

    dm = result.get("depth_metrics", {})
    if dm:
        quality_dict["depth"] = {
            "empty_score":  round(dm.get("empty_score", 0), 3),
            "full_score":   round(dm.get("full_score", 0), 3),
            "depth_signal": "empty" if dm.get("empty_score", 0) > dm.get("full_score", 0) else "full",
            "reliable":     dm.get("depth_reliable", False),
        }

    return depth_out, verdict_html, result["reasoning"], quality_dict


# ── Gradio layout ─────────────────────────────────────────────────────────────

APP_CSS = """
#app-title { text-align: center; }
#upload-col { min-width: 320px; }
#depth-col  { min-width: 320px; }
.verdict-box { margin-top: 0 !important; }
footer { display: none !important; }
"""

_DESCRIPTION = """
Upload a photo of a **shipping container opening**.
The system combines an **EfficientNet-B0** classifier (transfer-learned on your dataset)
with **MiDaS monocular depth estimation** to determine container status.

**Rules:** confidence must exceed 80 %; image must be sharp, well-lit, and show a clear container opening.
"""

# Gather example images (up to 3 per class)
def _example_paths():
    examples = []
    for cls in ("empty", "full", "invalid"):
        folder = Path("asset") / cls
        imgs   = sorted(folder.glob("*.*"))[:3] if folder.exists() else []
        examples.extend([[str(p)] for p in imgs])
    return examples[:9]   # cap at 9 total examples


with gr.Blocks(title="Container Status Classifier") as demo:

    gr.Markdown("# 📦 Container Status Classifier", elem_id="app-title")
    gr.Markdown(_DESCRIPTION)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_id="upload-col"):
            upload = gr.Image(
                label="Upload Container Image",
                type="numpy",
                height=340,
            )
            analyse_btn = gr.Button("Analyse", variant="primary", size="lg")

        with gr.Column(scale=1, elem_id="depth-col"):
            depth_out = gr.Image(
                label="MiDaS Depth Map  (bright = near · dark = far)",
                type="numpy",
                height=340,
                interactive=False,
            )

    verdict_html = gr.HTML(
        value="<p style='color:#9ca3af; text-align:center; padding:12px;'>"
              "Upload an image and click <b>Analyse</b>.</p>",
        elem_classes=["verdict-box"],
    )

    with gr.Row():
        with gr.Column(scale=3):
            reasoning_box = gr.Textbox(
                label="Analysis Reasoning",
                lines=14,
                interactive=False,
            )
        with gr.Column(scale=2):
            quality_json = gr.JSON(label="Quality & Depth Metrics")

    # Examples gallery
    examples = _example_paths()
    if examples:
        gr.Examples(
            examples=examples,
            inputs=upload,
            label="Example Images (click to load)",
            examples_per_page=9,
        )

    gr.Markdown(
        "_Backend integration: import `ContainerPredictor` from `inference.py` "
        "and call `predictor.predict(pil_image)` — the Gradio layer is purely cosmetic._",
        elem_id="footer-note",
    )

    # Wiring
    analyse_btn.click(
        fn=classify,
        inputs=upload,
        outputs=[depth_out, verdict_html, reasoning_box, quality_json],
    )
    upload.change(          # auto-run on image change
        fn=classify,
        inputs=upload,
        outputs=[depth_out, verdict_html, reasoning_box, quality_json],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        css=APP_CSS,
    )
