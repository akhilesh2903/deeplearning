"""
Training Script — Fetal Ultrasound U-Net Classifier
====================================================
Trains a U-Net encoder-decoder with classification head.

Classes (fixed index order):
    0 — normal    → Low Risk
    1 — fgr       → Moderate Risk
    2 — abnormal  → High Risk

Usage:
    python train.py --dataset_path "Datasets" --epochs 30
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def is_valid_training_image(path: Path) -> bool:
    name = path.name.lower()
    if "annotation" in name:
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


class FetalUltrasoundDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        for class_name in CLASS_ORDER:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            label = CLASS_TO_IDX[class_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and is_valid_training_image(img_path):
                    self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(image_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
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


def load_datasets(dataset_path, train_tf, eval_tf, batch_size=16):
    root = Path(dataset_path).resolve()
    train_dir = root / "train"
    val_dir = root / "validation" if (root / "validation").exists() else root / "val"
    test_dir = root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")

    train_ds = FetalUltrasoundDataset(train_dir, transform=train_tf)
    val_ds = FetalUltrasoundDataset(val_dir, transform=eval_tf) if val_dir.exists() else None
    test_ds = FetalUltrasoundDataset(test_dir, transform=eval_tf) if test_dir.exists() else None

    class_counts = [0] * len(CLASS_ORDER)
    for _, label in train_ds.samples:
        class_counts[label] += 1

    total = sum(class_counts)
    class_weights = [total / (len(CLASS_ORDER) * c) if c > 0 else 1.0 for c in class_counts]

    print("\n  Class mapping:")
    for idx, name in enumerate(CLASS_ORDER):
        print(f"    {idx}: {name:<10} -> {RISK_LABELS[idx]:<15} ({class_counts[idx]} train images)")

    sample_weights = [1.0 / class_counts[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0) if test_ds else None

    return train_loader, val_loader, test_loader, class_weights, {
        "train": len(train_ds),
        "val": len(val_ds) if val_ds else 0,
        "test": len(test_ds) if test_ds else 0,
    }


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
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 20 == 0 or batch_idx == num_batches - 1:
            print(f"    Epoch {epoch} batch {batch_idx+1}/{num_batches} loss={loss.item():.4f}", flush=True)

    return total_loss / total, correct / total * 100


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
            "accuracy": accuracy_score(all_labels, all_preds) * 100,
            "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0) * 100,
            "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0) * 100,
            "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100,
            "predictions": all_preds,
            "labels": all_labels,
        }

    return total_loss / max(total, 1), correct / max(total, 1) * 100, metrics


def save_checkpoint(model, path, epoch, val_acc, class_names, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_name": "unet_resnet18",
        "class_names": class_names,
        "class_to_idx": CLASS_TO_IDX,
        "class_labels": CLASS_LABELS,
        "risk_labels": RISK_LABELS,
        "num_classes": len(class_names),
        "val_accuracy": val_acc,
        "state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"         >> Checkpoint written: {path}", flush=True)


def plot_training_curves(history, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    for fname, key, title, ylabel in [
        ("training_accuracy.png", "train_acc", "Training Accuracy", "Accuracy (%)"),
        ("training_loss.png", "train_loss", "Training Loss", "Loss"),
        ("validation_accuracy.png", "val_acc", "Validation Accuracy", "Accuracy (%)"),
        ("validation_loss.png", "val_loss", "Validation Loss", "Loss"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [h[key] for h in history], marker="o", markersize=3)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / fname, dpi=150)
        plt.close()


def plot_confusion_matrix(y_true, y_pred, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = [RISK_LABELS[i] for i in range(len(CLASS_ORDER))]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_ORDER))))

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix — U-Net FGR Model")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=30, ha="right")
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def set_encoder_trainable(model, trainable: bool):
    for module in [model.encoder0, model.encoder1, model.encoder2, model.encoder3, model.encoder4]:
        for param in module.parameters():
            param.requires_grad = trainable


def train(dataset_path, epochs=30, batch_size=16, lr=1e-4, device_name="auto", patience=8, image_size=224):
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

    best_ckpt = MODELS_DIR / "fgr_unet_model.pth"
    compat_ckpt = CHECKPOINT_DIR / "best_model.pth"
    history_file = CHECKPOINT_DIR / "training_history.json"

    train_tf, eval_tf = get_transforms(image_size)
    train_loader, val_loader, test_loader, class_weights, split_counts = load_datasets(
        dataset_path, train_tf, eval_tf, batch_size
    )
    print(f"\n  Split sizes: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}", flush=True)

    model = UNetClassifier(num_classes=len(CLASS_ORDER), pretrained=True).to(device)

    # Remove weight tensor since WeightedRandomSampler already balances the batches
    criterion = nn.CrossEntropyLoss()

    head_params = [p for m in [model.bottleneck, model.classifier] for p in m.parameters()]
    decoder_params = [p for n, p in model.named_parameters() if "decoder" in n or "attention" in n]
    reserved = {id(p) for p in head_params + decoder_params}
    encoder_params = [p for p in model.parameters() if id(p) not in reserved]

    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr},
        {"params": decoder_params, "lr": lr * 0.5},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = []
    best_val_acc = 0.0
    epochs_no_improve = 0

    print(f"\n{'=' * 60}", flush=True)
    print(f"  U-Net FGR Training | Epochs: {epochs} | Batch: {batch_size}", flush=True)
    print(f"{'=' * 60}", flush=True)

    for epoch in range(1, epochs + 1):
        set_encoder_trainable(model, True)

        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch)

        if val_loader:
            vl_loss, vl_acc, _ = evaluate(model, val_loader, criterion, device)
        else:
            vl_loss, vl_acc = 0.0, 0.0

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"  {epoch:>4}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.2f}%  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.2f}%  ({elapsed:.0f}s)",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 2),
            "val_loss": round(vl_loss, 4),
            "val_acc": round(vl_acc, 2),
        })

        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        if val_loader and vl_acc > best_val_acc:
            best_val_acc = vl_acc
            epochs_no_improve = 0
            save_checkpoint(model, best_ckpt, epoch, vl_acc, CLASS_ORDER)
            save_checkpoint(model, compat_ckpt, epoch, vl_acc, CLASS_ORDER)
            print(f"         >> Saved best model: {vl_acc:.2f}% val accuracy", flush=True)
        else:
            epochs_no_improve += 1

        if val_loader and epochs_no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}", flush=True)
            break

    plot_training_curves(history, PLOTS_DIR)

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])

    _, final_train_acc, _ = evaluate(model, train_loader, criterion, device)
    _, final_val_acc, val_metrics = evaluate(model, val_loader, criterion, device) if val_loader else (0, 0, {})
    _, test_acc, test_metrics = evaluate(model, test_loader, criterion, device) if test_loader else (0, 0, {})

    if test_metrics.get("labels"):
        plot_confusion_matrix(test_metrics["labels"], test_metrics["predictions"], PLOTS_DIR)
        print("\n  Test classification report:", flush=True)
        print(classification_report(
            test_metrics["labels"], test_metrics["predictions"],
            target_names=[RISK_LABELS[i] for i in range(len(CLASS_ORDER))],
            zero_division=0,
        ), flush=True)

    precision = test_metrics.get("precision", val_metrics.get("precision", 0))
    recall = test_metrics.get("recall", val_metrics.get("recall", 0))
    f1 = test_metrics.get("f1", val_metrics.get("f1", 0))

    print(f"\n{'=' * 40}", flush=True)
    print("  U-NET FGR MODEL PERFORMANCE", flush=True)
    print(f"{'=' * 40}", flush=True)
    print(f"  Training Accuracy   : {final_train_acc:.2f}%", flush=True)
    print(f"  Validation Accuracy : {final_val_acc:.2f}%", flush=True)
    print(f"  Test Accuracy       : {test_acc:.2f}%", flush=True)
    print(f"  Precision           : {precision:.2f}%", flush=True)
    print(f"  Recall              : {recall:.2f}%", flush=True)
    print(f"  F1 Score            : {f1:.2f}%", flush=True)
    print(f"{'=' * 40}", flush=True)
    print(f"  Best model saved to : {best_ckpt}", flush=True)

    results = {
        "train_accuracy": round(final_train_acc, 2),
        "val_accuracy": round(final_val_acc, 2),
        "test_accuracy": round(test_acc, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "best_val_accuracy": round(best_val_acc, 2),
        "epochs_trained": len(history),
        "split_counts": split_counts,
    }
    with open(CHECKPOINT_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net fetal ultrasound classifier")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
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
