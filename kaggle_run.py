"""
kaggle_run.py — Kaggle-compatible launcher
Handles Kaggle's port restrictions and environment differences
"""

import os
import sys
from pathlib import Path

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Check if model exists, if not download or train
MODEL_PATH = Path("models/container_classifier.pth")
if not MODEL_PATH.exists():
    print("⚠️  Model not found. Please upload models/container_classifier.pth to Kaggle dataset.")
    sys.exit(1)

print("🚀 Launching Container Intelligence System on Kaggle...")
print("📊 Open http://127.0.0.1:7860 in your browser\n")

from app import demo, APP_CSS
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=True,  # ← CHANGED: Enable Kaggle share link
    show_error=True,
)
