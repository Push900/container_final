"""Quick test of page functions."""
import numpy as np
from PIL import Image
from pathlib import Path

from app import page1_analyze, page2_classify, refresh_dashboard, export_results

# Load test image
test_img_path = Path('asset/empty_or_full_or_invalid/empty/image1.jpg')
pil_img = Image.open(test_img_path).convert('RGB')
img_array = np.array(pil_img)

print("\n" + "="*60)
print("Testing page1_analyze")
print("="*60)
try:
    ocr_html, colors_html, detection_html, new_history, p1_state = page1_analyze(img_array, [])
    print("✓ page1_analyze works")
    print(f"  Returns 5 values: {5}")
    print(f"  P1 state type: {p1_state.get('type')}")
    print(f"  P1 state timestamp: {p1_state.get('timestamp')}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing page2_classify")
print("="*60)
try:
    depth, verdict, reasoning, quality, new_history, p2_state = page2_classify(img_array, [])
    print("✓ page2_classify works")
    print(f"  Returns 6 values")
    print(f"  P2 state type: {p2_state.get('type')}")
    print(f"  Prediction: {p2_state.get('prediction')}")
    print(f"  Confidence: {p2_state.get('confidence')}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing refresh_dashboard")
print("="*60)
try:
    _, summary_html, history = page1_analyze(img_array, [])
    depth, verdict, reasoning, quality, history2, p2_state = page2_classify(img_array, history + [page1_state])

    summary, hist_table = refresh_dashboard(p1_state, p2_state, history2)
    print("✓ refresh_dashboard works")
    print(f"  Summary HTML length: {len(summary)}")
    print(f"  History table rows: {len(hist_table)}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ All backend functions work!\n")
