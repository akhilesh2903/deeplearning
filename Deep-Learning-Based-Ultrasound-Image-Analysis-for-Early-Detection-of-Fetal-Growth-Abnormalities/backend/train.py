"""
Training Script - Fetal Ultrasound U-Net Classifier
===================================================
Robust transfer learning training for 3-class fetal growth restriction
and abnormality classification using U-Net with ResNet18 encoder.

Trains across all dataset sources:
  - Datasets/ (benign, malignant, normal)
  - OverlayedImages/ (benign, malignant, normal)
  - train/ (benign/fgr, malignant/abnormal, normal)
  - validation/ (benign/fgr, malignant/abnormal, normal)
  - test/ (benign/fgr, malignant/abnormal, normal)

Classes (fixed index order):
    0 - normal    -> Low Risk
    1 - fgr       -> Moderate Risk
    2 - abnormal  -> High Risk

Usage:
    python train.py --dataset_path "Datasets" --epochs 25 --batch_size 16 --lr 0.0003
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model.unet_model import (
    CLASS_LABELS,
    CLASS_ORDER,
    CLASS_TO_IDX,
    RISK_LABELS,
    UNetClassifier,
)

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
CHECKPOINT_DIR = SCRIPT_DIR / "model" / "checkpoint"
PLOTS_DIR = CHECKPOINT_DIR / "training_plots"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


# -------------------------------------------------------------------------
# DATASET
# -------------------------------------------------------------------------
def is_valid_training_image(path: Path) -> bool:
    name = path.name.lower()
    if "mask" in name or "annotation" in name or "_denoised" in name:
        return False
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


class ListDataset(Dataset):
    """Dataset initialized directly from a list of (image_path, label_idx) tuples."""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# -------------------------------------------------------------------------
# TRANSFORMS
# -------------------------------------------------------------------------
def get_transforms(image_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, eval_tf


# -------------------------------------------------------------------------
# DATA LOADING - ALL FOLDERS INCLUDED
# -------------------------------------------------------------------------
def load_datasets(dataset_path, train_tf, eval_tf, batch_size=16):
    root = Path(dataset_path).resolve()
    
    # Target subdirectories to pool from
    target_parents = ["Datasets", "OverlayedImages", "train", "validation", "val", "test"]
    folder_aliases = {
        0: ["normal"],
        1: ["fgr", "benign"],
        2: ["abnormal", "malignant"]
    }

    all_samples = []
    seen = set()

    for parent_name in target_parents:
        p_dir = root / parent_name
        if not p_dir.exists():
            continue
        
        for label_idx, aliases in folder_aliases.items():
            for alias in aliases:
                sub_dir = p_dir / alias
                if sub_dir.exists() and sub_dir.is_dir():
                    for img_file in sorted(sub_dir.glob("*.*")):
                        if is_valid_training_image(img_file):
                            canonical = str(img_file.resolve()).lower()
                            if canonical not in seen:
                                seen.add(canonical)
                                all_samples.append((str(img_file), label_idx))

    # Fallback if no subdirs matched
    if not all_samples:
        for label_idx, aliases in folder_aliases.items():
            for alias in aliases:
                sub_dir = root / alias
                if sub_dir.exists() and sub_dir.is_dir():
                    for img_file in sorted(sub_dir.glob("*.*")):
                        if is_valid_training_image(img_file):
                            canonical = str(img_file.resolve()).lower()
                            if canonical not in seen:
                                seen.add(canonical)
                                all_samples.append((str(img_file), label_idx))

    if not all_samples:
        raise FileNotFoundError(f"No valid ultrasound images found in dataset folder: {root}")

    print(f"\n  Found {len(all_samples)} total images across all dataset folders.")

    # Stratified 80% Train, 10% Val, 10% Test
    X = [s[0] for s in all_samples]
    y = [s[1] for s in all_samples]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    train_samples = list(zip(X_train, y_train))
    val_samples   = list(zip(X_val, y_val))
    test_samples  = list(zip(X_test, y_test))

    train_ds = ListDataset(train_samples, transform=train_tf)
    val_ds   = ListDataset(val_samples,   transform=eval_tf)
    test_ds  = ListDataset(test_samples,  transform=eval_tf)

    class_counts = [0] * len(CLASS_ORDER)
    for _, label in train_samples:
        class_counts[label] += 1

    total = max(sum(class_counts), 1)
    class_weights = torch.tensor(
        [total / (len(CLASS_ORDER) * max(c, 1)) for c in class_counts],
        dtype=torch.float32
    )

    print("\n  Class distribution in Training Set:")
    for idx, name in enumerate(CLASS_ORDER):
        print(f"    {idx}: {name:<10} -> {RISK_LABELS[idx]:<15} ({class_counts[idx]} images)")
    print(f"\n  Balanced Class Weights: {[round(w.item(), 3) for w in class_weights]}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, class_weights, {
        "train": len(train_ds),
        "val":   len(val_ds),
        "test":  len(test_ds),
        "total": len(all_samples),
    }


# -------------------------------------------------------------------------
# TRAINING & EVALUATION
# -------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device, epoch=0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    num_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 50 == 0 or batch_idx == num_batches - 1:
            print(f"    Epoch {epoch:02d} | Batch {batch_idx+1:>3}/{num_batches} | Loss: {loss.item():.4f}", flush=True)

    return total_loss / max(total, 1), correct / max(total, 1) * 100


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = {}
    if all_labels:
        metrics = {
            "accuracy":    accuracy_score(all_labels, all_preds) * 100,
            "precision":   precision_score(all_labels, all_preds, average="weighted", zero_division=0) * 100,
            "recall":      recall_score(all_labels, all_preds, average="weighted", zero_division=0) * 100,
            "f1":          f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100,
            "predictions": all_preds,
            "labels":      all_labels,
        }
    return total_loss / max(total, 1), correct / max(total, 1) * 100, metrics


# -------------------------------------------------------------------------
# CHECKPOINT & PLOTS
# -------------------------------------------------------------------------
def save_checkpoint(model, path, epoch, val_acc, class_names, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch":        epoch,
        "model_name":   "unet_resnet18",
        "class_names":  class_names,
        "class_to_idx": CLASS_TO_IDX,
        "class_labels": CLASS_LABELS,
        "risk_labels":  RISK_LABELS,
        "num_classes":  len(class_names),
        "val_accuracy": val_acc,
        "state_dict":   model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"         >> Checkpoint saved: {path}", flush=True)


def plot_training_curves(history, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, [h["train_acc"] for h in history], label="Train", marker="o", markersize=3)
    axes[0].plot(epochs, [h["val_acc"] for h in history], label="Val", marker="s", markersize=3)
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [h["train_loss"] for h in history], label="Train", marker="o", markersize=3)
    axes[1].plot(epochs, [h["val_loss"] for h in history], label="Val", marker="s", markersize=3)
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "training_curves.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    names = [RISK_LABELS[i] for i in range(len(CLASS_ORDER))]

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
    fig.colorbar(cax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), va="center", ha="center", fontsize=12)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20)
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=150)
    plt.close()


# -------------------------------------------------------------------------
# MAIN TRAIN FUNCTION
# -------------------------------------------------------------------------
def train(dataset_path="Datasets", epochs=25, batch_size=16, lr=3e-4, device_name="auto", patience=10, image_size=224):
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"\n  Device: {device}", flush=True)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    best_ckpt    = MODELS_DIR / "fgr_unet_model.pth"
    compat_ckpt  = CHECKPOINT_DIR / "best_model.pth"
    history_file = CHECKPOINT_DIR / "training_history.json"

    train_tf, eval_tf = get_transforms(image_size)
    train_loader, val_loader, test_loader, class_weights, split_counts = load_datasets(
        dataset_path, train_tf, eval_tf, batch_size
    )
    print(f"\n  Split sizes: Train={split_counts['train']} | Val={split_counts['val']} | Test={split_counts['test']} | Total={split_counts['total']}", flush=True)

    model = UNetClassifier(num_classes=len(CLASS_ORDER), pretrained=True).to(device)

    # Balanced CrossEntropy
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Fine-tuning with AdamW
    head_params = [p for m in [model.bottleneck, model.classifier, model.decoder0, model.decoder1, model.decoder2, model.decoder3, model.attention_conv] for p in m.parameters()]
    head_ids = {id(p) for p in head_params}
    encoder_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": lr * 0.3},
        {"params": head_params,    "lr": lr},
    ], weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = []
    best_val_acc = 0.0
    epochs_no_improve = 0

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Starting U-Net Training on Full Dataset ({split_counts['total']} images)", flush=True)
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}", flush=True)
    print(f"{'=' * 60}", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch)

        if val_loader:
            vl_loss, vl_acc, _ = evaluate(model, val_loader, criterion, device)
        else:
            vl_loss, vl_acc = 0.0, 0.0

        scheduler.step(vl_acc)
        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[1]["lr"]

        print(
            f"  [Epoch {epoch:02d}/{epochs:02d}]  Train Acc: {tr_acc:.2f}% | Val Acc: {vl_acc:.2f}% | "
            f"Loss: {tr_loss:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s",
            flush=True,
        )

        history.append({
            "epoch":      epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc":  round(tr_acc, 2),
            "val_loss":   round(vl_loss, 4),
            "val_acc":    round(vl_acc, 2),
            "lr":         round(current_lr, 8),
        })

        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        if val_loader and vl_acc > best_val_acc:
            best_val_acc = vl_acc
            epochs_no_improve = 0
            save_checkpoint(model, best_ckpt, epoch, vl_acc, CLASS_ORDER)
            save_checkpoint(model, compat_ckpt, epoch, vl_acc, CLASS_ORDER)
            print(f"         >> New Best Validation Accuracy: {vl_acc:.2f}%!", flush=True)
        else:
            epochs_no_improve += 1

        if val_loader and epochs_no_improve >= patience:
            print(f"\n  Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs)", flush=True)
            break

    plot_training_curves(history, PLOTS_DIR)

    # Evaluate best checkpoint
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"\n  Loaded best checkpoint from epoch {ckpt.get('epoch', '?')} (Val: {ckpt.get('val_accuracy', 0):.2f}%)", flush=True)

    eval_criterion = nn.CrossEntropyLoss()
    _, final_train_acc, _            = evaluate(model, train_loader, eval_criterion, device)
    _, final_val_acc,   val_metrics  = evaluate(model, val_loader,  eval_criterion, device) if val_loader  else (0, 0, {})
    _, test_acc,        test_metrics = evaluate(model, test_loader, eval_criterion, device) if test_loader else (0, 0, {})

    if test_metrics.get("labels"):
        plot_confusion_matrix(test_metrics["labels"], test_metrics["predictions"], PLOTS_DIR)
        print("\n  Test Classification Report:", flush=True)
        print(classification_report(
            test_metrics["labels"], test_metrics["predictions"],
            target_names=[RISK_LABELS[i] for i in range(len(CLASS_ORDER))],
            zero_division=0,
        ), flush=True)

    precision = test_metrics.get("precision", val_metrics.get("precision", 0))
    recall    = test_metrics.get("recall",    val_metrics.get("recall",    0))
    f1        = test_metrics.get("f1",        val_metrics.get("f1",        0))

    print(f"\n{'=' * 40}", flush=True)
    print("  FINAL MODEL PERFORMANCE", flush=True)
    print(f"{'=' * 40}", flush=True)
    print(f"  Training Accuracy   : {final_train_acc:.2f}%", flush=True)
    print(f"  Validation Accuracy : {final_val_acc:.2f}%", flush=True)
    print(f"  Test Accuracy       : {test_acc:.2f}%", flush=True)
    print(f"  Precision           : {precision:.2f}%", flush=True)
    print(f"  Recall              : {recall:.2f}%", flush=True)
    print(f"  F1 Score            : {f1:.2f}%", flush=True)
    print(f"  Best Val Accuracy   : {best_val_acc:.2f}%", flush=True)
    print(f"{'=' * 40}", flush=True)

    results = {
        "train_accuracy":    round(final_train_acc, 2),
        "val_accuracy":      round(final_val_acc, 2),
        "test_accuracy":     round(test_acc, 2),
        "precision":         round(precision, 2),
        "recall":            round(recall, 2),
        "f1":                round(f1, 2),
        "best_val_accuracy": round(best_val_acc, 2),
        "epochs_trained":    len(history),
        "split_counts":      split_counts,
    }
    with open(CHECKPOINT_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net fetal ultrasound classifier across all sources")
    parser.add_argument("--dataset_path", type=str, default="Datasets", help="Path to dataset root folder")
    parser.add_argument("--epochs",       type=int,   default=25)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--device",       type=str,   default="auto")
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--image_size",   type=int,   default=224)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
        patience=args.patience,
        image_size=args.image_size,
    )
