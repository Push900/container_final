"""
run.py — One-shot launcher.

Usage:
    python run.py           # train if needed, then launch UI
    python run.py --train   # force re-train even if model exists
    python run.py --app     # launch UI only (skip training check)

Open http://127.0.0.1:7860 in your browser after startup.
"""

import argparse
import sys
from pathlib import Path

MODEL_PATH = Path("models/container_classifier.pth")


def main():
    parser = argparse.ArgumentParser(description="Container Status Classifier launcher")
    parser.add_argument("--train", action="store_true", help="Force re-training")
    parser.add_argument("--app",   action="store_true", help="Skip training, launch app directly")
    args = parser.parse_args()

    # ── Training step ─────────────────────────────────────────────────────────
    needs_train = (not MODEL_PATH.exists()) or args.train

    if not args.app and needs_train:
        if args.train:
            print("Force re-training requested.\n")
        else:
            print(f"No trained model found at '{MODEL_PATH}'. Starting training …\n")

        from train import train
        train()
        print()

    elif args.app and not MODEL_PATH.exists():
        sys.exit(
            f"\n[ERROR] Model not found at '{MODEL_PATH}'.\n"
            "  Run without --app to train first:  python run.py\n"
        )

    elif MODEL_PATH.exists() and not args.train:
        print(f"Existing model found at '{MODEL_PATH}'. Skipping training.")
        print("  (Use  python run.py --train  to force re-training.)\n")

    # ── App step ──────────────────────────────────────────────────────────────
    print("Launching Gradio interface …")
    print("  Open http://127.0.0.1:7860 in your browser.\n")

    from app import demo, APP_CSS
    demo.launch(
         server_name="127.0.0.1",
        server_port=7860,
        # Gradio will find an available port starting from 7860
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
