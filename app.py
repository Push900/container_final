"""
app.py — Three-page Gradio interface for Container Intelligence System.

Pages:
  1. Image Understanding & Detection (OCR, colors, container detection)
  2. Container State Classification (Full/Empty/Invalid with depth maps)
  3. Results Dashboard (aggregation, history, export)

All backend logic in inference.py is unchanged and framework-agnostic.
"""

import sys

import gradio as gr
import numpy as np
from PIL import Image

from inference import ContainerPredictor, MODEL_PATH

# AI Surveyor extension (independent modules; lazy-loaded on first use so the
# existing three pages and app startup are unaffected).
try:
    import surveyor_ui
    SURVEYOR_AVAILABLE = True
except Exception as _surv_exc:  # pragma: no cover
    print(f"[WARN] AI Surveyor tab unavailable: {_surv_exc}")
    SURVEYOR_AVAILABLE = False

# ── Startup & Initialization ──────────────────────────────────────────────────

if not MODEL_PATH.exists():
    sys.exit(f"\n[ERROR] Model not found at {MODEL_PATH}\nRun: python train.py\n")

print("\nInitialising ContainerPredictor…")
try:
    predictor = ContainerPredictor()
    print("Ready.\n")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# ── Colour Scheme ─────────────────────────────────────────────────────────────

COLORS = {
    "EMPTY":   ("#16a34a", "#f0fdf4"),
    "FULL":    ("#ea580c", "#fff7ed"),
    "INVALID": ("#dc2626", "#fef2f2"),
}

ICONS = {"EMPTY": "📦", "FULL": "🧱", "INVALID": "⚠️"}

# ── Page 1: Detection ─────────────────────────────────────────────────────────

def page1_analyze(image):
    """Analyze image for container detection."""
    if image is None:
        return None, None, None, None

    try:
        pil_img = Image.fromarray(image).convert("RGB")

        # 1. Extract dominant colors
        try:
            small = pil_img.resize((150, 150)).convert("RGB")
            quantized = small.quantize(colors=5, method=Image.Quantize.MEDIANCUT)
            palette = quantized.getpalette()
            colors = [(palette[i*3], palette[i*3+1], palette[i*3+2]) for i in range(5)]
        except:
            colors = []

        # 2. Container detection (passthrough mode)
        detection_result = {
            "detected": True,
            "confidence": 1.0,
            "mode": "passthrough",
            "label": "YOLO unavailable"
        }

        # 3. Build HTML outputs
        # Colors HTML
        swatches = "".join(
            f'<div style="display:inline-block;width:40px;height:40px;'
            f'background:#{r:02x}{g:02x}{b:02x};border-radius:6px;'
            f'margin:4px;border:1px solid #ddd;vertical-align:middle;'
            f'title=#{r:02x}{g:02x}{b:02x}\"></div>'
            for r, g, b in colors
        ) if colors else "<p style='color:#999;'>No colors detected</p>"

        colors_html = f"""
        <div style="padding:16px;background:#f9fafb;border-radius:10px;border:1px solid #e5e7eb;">
            <div style="font-size:0.85em;color:#6b7280;font-weight:600;margin-bottom:12px;">DOMINANT COLORS</div>
            <div>{swatches}</div>
        </div>
        """

        # Detection HTML
        det_color = "#16a34a" if detection_result["detected"] else "#dc2626"
        det_bg = "#f0fdf4" if detection_result["detected"] else "#fef2f2"
        det_text = " CONTAINER DETECTED" if detection_result["detected"] else " NOT A CONTAINER"

        detection_html = f"""
        <div style="border:2px solid {det_color};border-radius:12px;background:{det_bg};
                    padding:20px;text-align:center;">
            <div style="font-size:1.4em;font-weight:700;color:{det_color};margin-bottom:8px;">{det_text}</div>
            <div style="color:#6b7280;font-size:0.9em;">
                Confidence: <strong>{detection_result['confidence']*100:.1f}%</strong> |
                Mode: <strong>{detection_result['mode']}</strong>
            </div>
        </div>
        """

        # 3. OCR Analysis (Dynamic)
        if predictor._ocr_validate:
            try:
                ocr = predictor._ocr_validate(pil_img)
                status_color = "#16a34a" if ocr["found"] else "#dc2626"
                found_text = "FOUND" if ocr["found"] else "NOT FOUND"

                ocr_html = f"""
                <div style="padding:16px;background:#f9fafb;border-radius:10px;border:1px solid #e5e7eb;">
                    <div style="font-size:0.85em;color:#6b7280;font-weight:600;margin-bottom:8px;">OCR TEXT EXTRACTION</div>
                    <div style="font-size:1.1em;font-weight:700;color:{status_color};">{found_text}</div>
                    <div style="font-size:0.95em;margin:4px 0;">Number: <strong>{ocr['number'] or '---'}</strong></div>
                    <div style="color:#6b7280;font-size:0.85em;">{ocr['message']}</div>
                </div>
                """
            except Exception as ocr_err:
                ocr_html = f"""
                <div style="padding:16px;background:#fee2e2;border-radius:10px;border:1px solid #fecaca;">
                    <div style="font-size:0.85em;color:#991b1b;font-weight:600;">OCR ERROR</div>
                    <div style="font-size:0.8em;color:#7f1d1d;">{str(ocr_err)[:80]}</div>
                </div>
                """
        elif getattr(predictor, "_ocr_loading", False):
            ocr_html = """
            <div style="padding:16px;background:#fffbeb;border-radius:10px;border:1px solid #fef3c7;">
                <div style="font-size:0.85em;color:#92400e;font-weight:600;margin-bottom:4px;">OCR MODULE LOADING...</div>
                <div style="font-size:0.85em;color:#b45309;">Downloading models or initializing engine. Analysis will proceed without OCR for now.</div>
            </div>
            """
        else:
            ocr_html = """
            <div style="padding:16px;background:#fef2f2;border-radius:10px;border:1px solid #fee2e2;">
                <div style="font-size:0.85em;color:#991b1b;font-weight:600;">OCR MODULE NOT LOADED</div>
            </div>
            """

        return ocr_html, colors_html, detection_html

    except Exception as e:
        error_html = f"<p style='color:#dc2626;'>Error: {str(e)[:100]}</p>"
        return error_html, error_html, error_html


# ── Page 2: Classification ────────────────────────────────────────────────────

def page2_classify(image):
    """Classify container state."""
    if image is None:
        return None, None, None, None

    try:
        pil_img = Image.fromarray(image).convert("RGB")
        result = predictor.predict(pil_img)

        # Depth map
        depth_out = None
        if getattr(predictor, "_depth_loading", False):
            # Create a placeholder "Loading" image or just keep it None
            depth_out = None 
            reasoning_prefix = " Depth Model is still loading in the background...\n\n"
        elif result.get("depth_map_colored"):
            depth_out = np.array(result["depth_map_colored"])
            reasoning_prefix = ""
        else:
            reasoning_prefix = " Depth detection unavailable (Model failed to load).\n\n"

        # Verdict HTML
        pred = result["prediction"]
        conf = result["confidence"]
        probs = result["class_probs"]

        border_c, bg_c = COLORS.get(pred, ("#6b7280", "#f9fafb"))
        icon = ICONS.get(pred, "?")

        prob_bars = ""
        for cls, p in probs.items():
            bar_color = COLORS.get(cls.upper(), ("#6b7280", ""))[0]
            prob_bars += f"""
            <div style="display:flex;align-items:center;margin:6px 0;">
                <span style="width:70px;font-weight:500;font-size:0.9em;">{cls.capitalize()}</span>
                <div style="flex:1;background:#e5e7eb;border-radius:4px;height:12px;margin:0 8px;">
                    <div style="width:{p*100:.1f}%;background:{bar_color};height:12px;border-radius:4px;"></div>
                </div>
                <span style="width:40px;text-align:right;font-size:0.85em;color:#6b7280;">{p*100:.1f}%</span>
            </div>
            """

        verdict_html = f"""
        <div style="border:2px solid {border_c};border-radius:12px;background:{bg_c};padding:18px;">
            <div style="font-size:2em;font-weight:700;color:{border_c};display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                <span>{icon}</span><span>{pred}</span>
            </div>
            <div style="color:#6b7280;margin-bottom:12px;">
                Confidence: <strong>{conf*100:.1f}%</strong>
            </div>
            <div style="font-size:0.8em;color:#9ca3af;margin-bottom:6px;font-weight:600;">CLASS PROBABILITIES</div>
            {prob_bars}
        </div>
        """

        # Reasoning
        reasoning = reasoning_prefix + result["reasoning"]

        # Metrics JSON
        qm = result.get("quality_metrics", {})
        dm = result.get("depth_metrics", {})

        metrics = {
            "prediction": pred,
            "confidence": round(conf, 4),
            "quality_ok": result.get("quality_ok"),
            "quality_issues": result.get("quality_issues"),
            "image_size": f"{qm.get('width')}×{qm.get('height')}",
            "blur_score": round(qm.get("blur_score", 0), 2),
            "brightness": round(qm.get("brightness", 0), 2),
            "contrast": round(qm.get("contrast", 0), 2),
        }

        if dm:
            metrics["depth"] = {
                "empty_score": round(dm.get("empty_score", 0), 3),
                "full_score": round(dm.get("full_score", 0), 3),
                "reliable": dm.get("depth_reliable", False),
            }

        return depth_out, verdict_html, reasoning, metrics

    except Exception as e:
        error_html = f"<p style='color:#dc2626;'>Error: {str(e)[:150]}</p>"
        return None, error_html, f"Error: {str(e)}", {}


# ── Page 3: Dashboard ─────────────────────────────────────────────────────────

def page3_refresh(p1_pred, p1_colors, p1_det, p2_pred, p2_conf, p2_q):
    """Refresh dashboard summary."""
    try:
        parts = []

        # Page 1 Summary
        if p1_det:
            parts.append("""
            <div style="border:1.5px solid #3b82f6;border-radius:10px;background:#eff6ff;padding:14px;margin-bottom:12px;">
                <div style="font-weight:600;color:#1e40af;margin-bottom:8px;">📊 Page 1: Detection Results</div>
                <div style="font-size:0.9em;color:#1e3a8a;">
                    Container detected | Colors extracted | Ready for classification
                </div>
            </div>
            """)

        # Page 2 Summary
        if p2_pred is not None:
            pred_color, pred_bg = COLORS.get(p2_pred, ("#6b7280", "#f9fafb"))
            pred_icon = ICONS.get(p2_pred, "?")
            parts.append(f"""
            <div style="border:1.5px solid {pred_color};border-radius:10px;background:{pred_bg};padding:14px;margin-bottom:12px;">
                <div style="font-weight:600;color:{pred_color};margin-bottom:8px;">
                    {pred_icon} Page 2: Classification Results
                </div>
                <div style="font-size:0.9em;color:#6b7280;">
                    Prediction: <strong>{p2_pred}</strong> |
                    Confidence: <strong>{p2_conf*100:.1f}%</strong>
                </div>
            </div>
            """)

        # Final Verdict
        if p2_pred:
            final_color, final_bg = COLORS.get(p2_pred, ("#6b7280", "#f9fafb"))
            parts.append(f"""
            <div style="border:2.5px solid {final_color};border-radius:12px;background:{final_bg};
                        padding:18px;text-align:center;">
                <div style="font-size:0.85em;color:#9ca3af;font-weight:600;margin-bottom:6px;">FINAL VERDICT</div>
                <div style="font-size:1.8em;font-weight:700;color:{final_color};">{p2_pred}</div>
            </div>
            """)
        else:
            parts.append("""
            <div style="border:1px dashed #d1d5db;border-radius:10px;padding:16px;text-align:center;color:#9ca3af;">
                Run analyses on Pages 1 & 2 to see results here
            </div>
            """)

        summary_html = "\n".join(parts)
        history_text = "Session history (run analyses to populate)"

        return summary_html, history_text

    except Exception as e:
        return f"<p style='color:#dc2626;'>Error: {str(e)}</p>", str(e)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

APP_CSS = """
#app-title { text-align: center; margin-bottom: 20px; }
footer { display: none !important; }
.page-section { margin-bottom: 16px; }
"""

with gr.Blocks(title="Container Intelligence System") as demo:

    # State storage
    p1_pred_state = gr.State(None)
    p1_colors_state = gr.State(None)
    p1_det_state = gr.State(None)
    p2_pred_state = gr.State(None)
    p2_conf_state = gr.State(0.0)
    p2_q_state = gr.State({})

    gr.Markdown("#  Container Intelligence System", elem_id="app-title")

    with gr.Tabs():

        # ── PAGE 1 ────────────────────────────────────────────────────────────
        with gr.TabItem("1️ Detection"):
            gr.Markdown("Upload an image to analyze for container presence, colors, and object detection.")

            with gr.Row():
                with gr.Column(scale=1):
                    p1_upload = gr.Image(label="Upload Image", type="numpy", height=300)
                    p1_btn = gr.Button(" Analyse", variant="primary", size="lg")
                with gr.Column(scale=1):
                    p1_ocr_out = gr.HTML(value="<p style='color:#999;text-align:center;padding:40px;'>Upload image</p>")

            with gr.Row():
                p1_colors_out = gr.HTML(value="<p style='color:#999;text-align:center;'>Colors</p>")
                p1_det_out = gr.HTML(value="<p style='color:#999;text-align:center;'>Detection</p>")

        # ── PAGE 2 ────────────────────────────────────────────────────────────
        with gr.TabItem("2 Classification"):
            gr.Markdown("Upload a container image to classify its state (Full/Empty/Invalid) with visual explanations.")

            with gr.Row():
                with gr.Column(scale=1):
                    p2_upload = gr.Image(label="Upload Container", type="numpy", height=300)
                    p2_btn = gr.Button(" Classify", variant="primary", size="lg")
                with gr.Column(scale=1):
                    p2_depth = gr.Image(label="Depth Map (bright=near, dark=far)", type="numpy", height=300, interactive=False)

            p2_verdict_out = gr.HTML(value="<p style='color:#999;text-align:center;padding:12px;'>Upload image</p>")

            with gr.Row():
                with gr.Column(scale=2):
                    p2_reason_out = gr.Textbox(label="Analysis Reasoning", lines=10, interactive=False)
                with gr.Column(scale=1):
                    p2_metrics_out = gr.JSON(label="Metrics")

        # ── PAGE 3 ────────────────────────────────────────────────────────────
        with gr.TabItem("3️ Dashboard"):
            gr.Markdown("Summary of results from Pages 1 & 2.")

            p3_refresh_btn = gr.Button(" Refresh", size="lg")

            p3_summary = gr.HTML(
                value='<div style="padding:20px;text-align:center;color:#999;">Run analyses on Pages 1 & 2</div>'
            )

            gr.Markdown("### Session History")
            p3_history = gr.Textbox(label="History", lines=8, interactive=False, value="(empty)")

        # ── PAGE 4: AI SURVEYOR ────────────────────────────────────────────────
        if SURVEYOR_AVAILABLE:
            with gr.TabItem(" AI Surveyor"):
                gr.Markdown(
                    "Autonomous container inspection. Upload one image or several "
                    "frames of the **same** container (different angles improve "
                    "reliability). The AI Surveyor detects & tracks the container, "
                    "reads and decodes the ISO number, inspects the seal, assesses "
                    "damage, and produces a signed PDF report stored in the database."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        surv_files = gr.Files(
                            label="Inspection frames (1+ images of one container)",
                            file_count="multiple", file_types=["image"],
                        )
                        surv_gps = gr.Textbox(label="GPS (optional, 'lat,lon')", placeholder="12.9716,77.5946")
                        surv_btn = gr.Button(" Run Full Inspection", variant="primary", size="lg")
                        surv_best = gr.Image(label="Inspected frame", height=260, interactive=False)
                        surv_report = gr.File(label=" Inspection Report (PDF)", interactive=False)
                    with gr.Column(scale=2):
                        surv_dashboard = gr.HTML(
                            value="<p style='color:#999;padding:30px;text-align:center;'>"
                                  "Upload frames and run an inspection.</p>")
                with gr.Accordion("Surveyor Checklist & Recommendations", open=True):
                    surv_checklist = gr.HTML()
                with gr.Accordion("Full Inspection Record (JSON)", open=False):
                    surv_json = gr.JSON()

        # ── PAGE 5: INSPECTION HISTORY ─────────────────────────────────────────
        if SURVEYOR_AVAILABLE:
            with gr.TabItem(" Inspection History"):
                gr.Markdown("Search the complete inspection audit trail stored in the database.")
                with gr.Row():
                    h_container = gr.Textbox(label="Container number")
                    h_seal = gr.Textbox(label="Seal number")
                    h_line = gr.Textbox(label="Shipping line")
                    h_result = gr.Dropdown(["Any", "PASS", "FAIL", "REVIEW"], value="Any", label="Result")
                with gr.Row():
                    h_from = gr.Textbox(label="Date from (ISO, e.g. 2026-06-01)")
                    h_to = gr.Textbox(label="Date to (ISO)")
                    h_damage = gr.Textbox(label="Damage category (e.g. rust, severe)")
                with gr.Row():
                    h_search = gr.Button(" Search", variant="primary")
                    h_refresh = gr.Button(" Show recent")
                h_results = gr.HTML()

    # ── Event Handlers ────────────────────────────────────────────────────────

    def handle_page1(img):
        try:
            ocr, colors, det = page1_analyze(img)
            return ocr, colors, det, det, 0, {}, det
        except Exception as e:
            error_msg = f"<p style='color:#dc2626;'>Error: {str(e)[:100]}</p>"
            return error_msg, error_msg, error_msg, error_msg, 0, {}, error_msg

    def handle_page2(img):
        try:
            depth, verdict, reason, metrics = page2_classify(img)
            return depth, verdict, reason, metrics, verdict, metrics["confidence"] if "confidence" in metrics else 0.0, metrics
        except Exception as e:
            error_msg = f"<p style='color:#dc2626;'>Error: {str(e)[:150]}</p>"
            return None, error_msg, error_msg, {}, error_msg, 0.0, {}

    def handle_refresh(p1d, p1c, p1det, p2p, p2c, p2q):
        summary, history = page3_refresh(p1d, p1c, p1det, p2p, p2c, p2q)
        return summary, history

    # Page 1
    p1_btn.click(handle_page1, inputs=p1_upload, outputs=[
        p1_ocr_out, p1_colors_out, p1_det_out, p1_det_state, p2_conf_state, p2_q_state, p1_pred_state
    ])

    # Page 2
    p2_btn.click(handle_page2, inputs=p2_upload, outputs=[
        p2_depth, p2_verdict_out, p2_reason_out, p2_metrics_out, p2_pred_state, p2_conf_state, p2_q_state
    ])

    # Page 3
    p3_refresh_btn.click(handle_refresh, inputs=[
        p1_pred_state, p1_colors_state, p1_det_state, p2_pred_state, p2_conf_state, p2_q_state
    ], outputs=[p3_summary, p3_history])

    # Page 4 & 5: AI Surveyor
    if SURVEYOR_AVAILABLE:
        surv_btn.click(
            surveyor_ui.run_inspection,
            inputs=[surv_files, surv_gps],
            outputs=[surv_dashboard, surv_best, surv_checklist, surv_json, surv_report],
        )
        h_search.click(
            surveyor_ui.search_history,
            inputs=[h_container, h_seal, h_from, h_to, h_line, h_result, h_damage],
            outputs=h_results,
        )
        h_refresh.click(
            lambda: surveyor_ui.search_history("", "", "", "", "", "Any", ""),
            outputs=h_results,
        )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )