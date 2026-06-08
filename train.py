"""
train.py — EfficientNet-B0 transfer-learning trainer for container classification.

Usage:
    python train.py

Reads images from  asset/{empty,full,invalid}/
Saves model to     models/container_classifier.pth
Saves metadata to  models/classifier_meta.json
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR   = Path("asset")
MODEL_DIR  = Path("models")
MODEL_PATH = MODEL_DIR / "container_classifier.pth"
META_PATH  = MODEL_DIR / "classifier_meta.json"

CLASSES    = ["empty", "full", "invalid"]
IMG_SIZE   = 224
BATCH_SIZE = 8        # small — 41 total images
EPOCHS     = 60
LR_HEAD    = 1e-3     # phase-1: head only
LR_FULL    = 5e-6     # phase-2: whole network
HEAD_EPOCHS = 20      # epochs for phase-1
VAL_SPLIT  = 0.20
PATIENCE   = 12       # early-stopping patience (phase-2)
SEED       = 42


# ── Transforms ───────────────────────────────────────────────────────────────

# Heavy augmentation for tiny dataset
TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE + 40, IMG_SIZE + 40)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.15),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
    transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), scale=(0.80, 1.20)),
    transforms.RandomPerspective(distortion_scale=0.35, p=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
])

VAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class ContainerDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int], transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def _collect_images():
    """Return parallel lists (paths, labels) from asset/{class} folders."""
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths, labels = [], []
    for lbl, cls in enumerate(CLASSES):
        folder = DATA_DIR / cls
        if not folder.exists():
            print(f"  [WARN] Missing folder: {folder}")
            continue
        found = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
        paths.extend([str(p) for p in found])
        labels.extend([lbl] * len(found))
        print(f"  {cls:10s}: {len(found)} images")
    return paths, labels


def make_loaders():
    print("Scanning dataset …")
    paths, labels = _collect_images()
    print(f"  Total: {len(paths)} images")

    train_p, val_p, train_l, val_l = train_test_split(
        paths, labels, test_size=VAL_SPLIT, stratify=labels, random_state=SEED
    )
    print(f"  Train: {len(train_p)}  |  Val: {len(val_p)}")

    # Weighted sampler to correct class imbalance during training
    counts = np.bincount(train_l, minlength=len(CLASSES)).astype(float)
    w      = 1.0 / (counts + 1e-9)
    sample_w = [w[l] for l in train_l]
    sampler  = WeightedRandomSampler(sample_w, num_samples=len(train_l), replacement=True)

    train_ds = ContainerDataset(train_p, train_l, TRAIN_TF)
    val_ds   = ContainerDataset(val_p,   val_l,   VAL_TF)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=False)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

    # Inverse-frequency class weights for the loss
    total      = len(train_l)
    loss_w     = torch.tensor(
        [total / (len(CLASSES) * (c + 1e-9)) for c in counts], dtype=torch.float32
    )
    return train_ld, val_ld, loss_w, {"train_paths": train_p, "val_paths": val_p}


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 3):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f  = model.classifier[1].in_features   # 1280 for EfficientNet-B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.45, inplace=True),
        nn.Linear(in_f, num_classes),
    )
    return model


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss = correct = total = 0

    ctx = torch.enable_grad if training else torch.no_grad
    with ctx():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            if training:
                optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == lbls).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


# ── Main training loop ────────────────────────────────────────────────────────

def train():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_ld, val_ld, loss_w, split_info = make_loaders()

    model     = build_model(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_w.to(device), label_smoothing=0.1)

    best_val_acc = 0.0
    history      = []

    # Phase 1: head only
    print(f"\n{'-'*60}")
    print(f"Phase 1 - head only  ({HEAD_EPOCHS} epochs, lr={LR_HEAD})")
    print(f"{'-'*60}")

    for param in model.features.parameters():
        param.requires_grad = False

    opt1  = optim.AdamW(model.classifier.parameters(), lr=LR_HEAD, weight_decay=1e-4)
    sch1  = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=HEAD_EPOCHS, eta_min=1e-6)

    pbar = tqdm(range(1, HEAD_EPOCHS + 1), desc="Phase-1", ncols=80)
    for ep in pbar:
        tl, ta = run_epoch(model, train_ld, criterion, opt1, device, training=True)
        vl, va = run_epoch(model, val_ld,   criterion, opt1, device, training=False)
        sch1.step()

        flag = ""
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), MODEL_PATH)
            flag = " *saved*"
        history.append(dict(epoch=ep, phase=1, t_loss=tl, t_acc=ta, v_loss=vl, v_acc=va))
        pbar.set_postfix(t_acc=f"{ta:.3f}", v_acc=f"{va:.3f}", best=f"{best_val_acc:.3f}", flag=flag)

    # Phase 2: full fine-tune
    phase2_epochs = EPOCHS - HEAD_EPOCHS
    print(f"\n{'-'*60}")
    print(f"Phase 2 - full fine-tune ({phase2_epochs} epochs, backbone lr={LR_FULL})")
    print(f"{'-'*60}")

    for param in model.parameters():
        param.requires_grad = True

    opt2 = optim.AdamW([
        {"params": model.features.parameters(),    "lr": LR_FULL},
        {"params": model.classifier.parameters(),  "lr": LR_FULL * 20},
    ], weight_decay=1e-4)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=phase2_epochs, eta_min=1e-8)

    patience_ctr = 0
    pbar2 = tqdm(range(HEAD_EPOCHS + 1, EPOCHS + 1), desc="Phase-2", ncols=80)
    for ep in pbar2:
        tl, ta = run_epoch(model, train_ld, criterion, opt2, device, training=True)
        vl, va = run_epoch(model, val_ld,   criterion, opt2, device, training=False)
        sch2.step()

        flag = ""
        if va > best_val_acc:
            best_val_acc = va
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_PATH)
            flag = " ★"
        else:
            patience_ctr += 1

        history.append(dict(epoch=ep, phase=2, t_loss=tl, t_acc=ta, v_loss=vl, v_acc=va))
        pbar2.set_postfix(t_acc=f"{ta:.3f}", v_acc=f"{va:.3f}",
                          best=f"{best_val_acc:.3f}", pat=patience_ctr, flag=flag)

        if patience_ctr >= PATIENCE:
            print(f"\nEarly stop at epoch {ep}  (patience={PATIENCE})")
            break

    # ── Save metadata ─────────────────────────────────────────────
    meta = {
        "classes":          CLASSES,
        "img_size":         IMG_SIZE,
        "model_arch":       "efficientnet_b0",
        "best_val_acc":     best_val_acc,
        "num_train":        len(split_info["train_paths"]),
        "num_val":          len(split_info["val_paths"]),
        "training_history": history,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"\n✓ Training complete.  Best val acc: {best_val_acc:.4f}")
    print(f"  Model  → {MODEL_PATH}")
    print(f"  Meta   → {META_PATH}")


if __name__ == "__main__":
    train()
