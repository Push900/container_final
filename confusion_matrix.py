"""Generate confusion matrix for container classification model."""

import json
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

from inference import ContainerPredictor

# ── Configuration ─────────────────────────────────────────────────────────────

TEST_DATA_DIR = Path("asset/empty_or_full_or_invalid")
CLASSES = ["empty", "full", "invalid"]

# ── Load test data ────────────────────────────────────────────────────────────

def load_test_dataset(data_dir: Path) -> tuple:
    """Load all test images and their true labels from directory structure."""
    true_labels = []
    image_paths = []

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist")
            continue

        # Find all image files
        image_files = list(class_dir.glob("*.*"))
        image_files = [f for f in image_files if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]

        print(f"Found {len(image_files)} images in {class_name}/")

        for img_path in image_files:
            image_paths.append(img_path)
            true_labels.append(class_name)

    return image_paths, true_labels


def predict_images(predictor: ContainerPredictor, image_paths: list) -> list:
    """Predict labels for all images."""
    predictions = []

    for idx, img_path in enumerate(image_paths):
        try:
            pil_img = Image.open(img_path).convert("RGB")
            result = predictor.predict(pil_img)
            pred = result["prediction"].lower()
            conf = result["confidence"]

            # Normalize prediction to lowercase class names
            if pred == "invalid":
                pred_label = "invalid"
            elif pred == "empty":
                pred_label = "empty"
            elif pred == "full":
                pred_label = "full"
            else:
                # Fallback for unexpected predictions
                pred_label = "invalid"

            predictions.append(pred_label)

            if (idx + 1) % 5 == 0:
                print(f"  Processed {idx + 1}/{len(image_paths)} images")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            predictions.append("invalid")

    return predictions


def main():
    print("="*70)
    print("CONTAINER CLASSIFICATION — CONFUSION MATRIX EVALUATION")
    print("="*70)

    # Load test data
    print(f"\n[1] Loading test dataset from {TEST_DATA_DIR}")
    image_paths, true_labels = load_test_dataset(TEST_DATA_DIR)
    print(f"Total images: {len(image_paths)}")
    print(f"Label distribution: {dict(zip(*np.unique(true_labels, return_counts=True)))}")

    # Initialize predictor
    print(f"\n[2] Initializing ContainerPredictor...")
    predictor = ContainerPredictor()
    print("Ready.\n")

    # Run predictions
    print(f"[3] Running predictions on {len(image_paths)} images...")
    predictions = predict_images(predictor, image_paths)
    print(f"Completed predictions.")

    # Compute metrics
    print(f"\n[4] Computing metrics...")
    cm = confusion_matrix(true_labels, predictions, labels=CLASSES)
    accuracy = accuracy_score(true_labels, predictions)

    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(true_labels, predictions, labels=CLASSES)}")

    # Save metrics to JSON
    true_unique, true_counts = np.unique(true_labels, return_counts=True)
    pred_unique, pred_counts = np.unique(predictions, return_counts=True)

    metrics = {
        "overall_accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "class_distribution": {
            "true": {str(k): int(v) for k, v in zip(true_unique, true_counts)},
            "predicted": {str(k): int(v) for k, v in zip(pred_unique, pred_counts)},
        },
        "num_samples": int(len(image_paths)),
    }

    metrics_path = Path("confusion_matrix_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved to {metrics_path}")

    # Generate visualization
    print(f"\n[5] Generating confusion matrix visualization...")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar_kws={"label": "Count"},
        square=True,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.title(f"Container Classification Confusion Matrix\n(Accuracy: {accuracy*100:.2f}%)",
              fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    # Save figure
    figure_path = Path("confusion_matrix.png")
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {figure_path}")
    plt.close()

    print("\n" + "="*70)
    print("✓ Confusion matrix evaluation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
