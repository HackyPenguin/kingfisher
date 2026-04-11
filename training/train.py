"""
train.py — Project Kestrel
Two-phase EfficientNetV2-S training pipeline for UK bird species classification.

Phase 1: Head-only warm-up (frozen backbone, LinearLR warmup)
Phase 2: Partial fine-tune of top blocks (CosineAnnealingLR)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = 300
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLITS_DIR = DATA_DIR / "splits"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
SPECIES_PATH = DATA_DIR / "uk_bird_taxa_full.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BirdDataset(torch.utils.data.Dataset):
    """Dataset for UK bird species images backed by a CSV manifest."""

    def __init__(
        self,
        csv_path: str,
        label_to_idx: dict,
        transform=None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = row["label"]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_index = self.label_to_idx[label]
        return image, label_index


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transforms(train: bool) -> transforms.Compose:
    """Return appropriate torchvision transforms for train or validation."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    desc: str = "Train",
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Val",
) -> tuple[float, float, float]:
    """Run one validation epoch. Returns (avg_loss, top1_acc, top3_acc)."""
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        # Top-1
        preds_top1 = outputs.argmax(dim=1)
        correct_top1 += (preds_top1 == labels).sum().item()

        # Top-3
        k = min(3, outputs.size(1))
        _, preds_top3 = outputs.topk(k, dim=1)
        labels_expanded = labels.unsqueeze(1).expand_as(preds_top3)
        correct_top3 += preds_top3.eq(labels_expanded).any(dim=1).sum().item()

        total += images.size(0)

    avg_loss = total_loss / total
    top1_acc = correct_top1 / total
    top3_acc = correct_top3 / total
    return avg_loss, top1_acc, top3_acc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def _unfreeze_by_name(model: nn.Module, name_fragments: list[str]) -> None:
    for name, param in model.named_parameters():
        if any(fragment in name for fragment in name_fragments):
            param.requires_grad = True


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    val_acc: float,
    phase: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "phase": phase,
        },
        path,
    )


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        "Resumed from %s (epoch %s, phase %s, val_acc=%.4f)",
        checkpoint_path,
        checkpoint.get("epoch", "unknown"),
        checkpoint.get("phase", "unknown"),
        checkpoint.get("val_acc", float("nan")),
    )


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EfficientNetV2-S on UK bird species."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=15)
    parser.add_argument("--phase1-lr", type=float, default=1e-3)
    parser.add_argument("--phase2-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from.",
    )
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = _select_device()
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"

    # Build label mapping from the training split
    train_df_labels = pd.read_csv(train_csv)["label"]
    unique_labels = sorted(train_df_labels.unique().tolist())
    label_to_idx: dict[str, int] = {
        label: idx for idx, label in enumerate(unique_labels)
    }
    idx_to_label: dict[str, str] = {
        str(idx): label for label, idx in label_to_idx.items()
    }
    num_classes = len(label_to_idx)

    logger.info("Number of species classes: %d", num_classes)

    # Persist label mapping
    label_mapping_path = CHECKPOINT_DIR / "label_mapping.json"
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {"label_to_idx": label_to_idx, "idx_to_label": idx_to_label},
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Label mapping saved to %s", label_mapping_path)

    train_dataset = BirdDataset(
        csv_path=str(train_csv),
        label_to_idx=label_to_idx,
        transform=get_transforms(train=True),
    )
    val_dataset = BirdDataset(
        csv_path=str(val_csv),
        label_to_idx=label_to_idx,
        transform=get_transforms(train=False),
    )

    # num_workers=0: MPS does not support multiprocess data loading well
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = timm.create_model(
        "tf_efficientnetv2_s",
        pretrained=True,
        num_classes=num_classes,
    )

    if args.resume:
        _load_checkpoint(model, args.resume)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ==================================================================
    # Phase 1 — Head only (frozen backbone)
    # ==================================================================
    logger.info("=" * 60)
    logger.info("Phase 1: Head-only training (%d epochs)", args.phase1_epochs)
    logger.info("=" * 60)

    _freeze_all(model)
    _unfreeze_module(model.classifier)

    phase1_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase1_lr,
        weight_decay=args.weight_decay,
    )

    # LinearLR warmup over the first epoch's batches
    warmup_steps = len(train_loader)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        phase1_optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    for epoch in range(args.phase1_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        desc = f"Phase1 E{epoch + 1}/{args.phase1_epochs} [Train]"

        for images, labels in tqdm(train_loader, desc=desc, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            phase1_optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            phase1_optimizer.step()

            # Only step warmup scheduler during the first epoch
            if epoch == 0:
                warmup_scheduler.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        val_loss, val_acc, val_top3 = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            desc=f"Phase1 E{epoch + 1}/{args.phase1_epochs} [Val]",
        )

        logger.info(
            "Phase 1 | Epoch %d/%d | "
            "train_loss=%.4f  train_acc=%.4f | "
            "val_loss=%.4f  val_acc=%.4f  val_top3=%.4f",
            epoch + 1,
            args.phase1_epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_top3,
        )

    # ==================================================================
    # Phase 2 — Partial fine-tune (top blocks + head)
    # ==================================================================
    logger.info("=" * 60)
    logger.info("Phase 2: Fine-tuning top blocks (%d epochs)", args.phase2_epochs)
    logger.info("=" * 60)

    _freeze_all(model)
    _unfreeze_module(model.classifier)
    _unfreeze_by_name(model, ["blocks.4", "blocks.5", "conv_head", "bn2"])

    phase2_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase2_lr,
        weight_decay=args.weight_decay,
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        phase2_optimizer,
        T_max=args.phase2_epochs,
        eta_min=1e-6,
    )

    best_val_acc = 0.0

    for epoch in range(args.phase2_epochs):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            phase2_optimizer,
            device,
            desc=f"Phase2 E{epoch + 1}/{args.phase2_epochs} [Train]",
        )

        val_loss, val_acc, val_top3 = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            desc=f"Phase2 E{epoch + 1}/{args.phase2_epochs} [Val]",
        )

        current_lr = _current_lr(phase2_optimizer)
        cosine_scheduler.step()

        logger.info(
            "Phase 2 | Epoch %d/%d | "
            "train_loss=%.4f  train_acc=%.4f | "
            "val_loss=%.4f  val_acc=%.4f  val_top3=%.4f | "
            "lr=%.2e",
            epoch + 1,
            args.phase2_epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_top3,
            current_lr,
        )

        # Save best checkpoint by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(
                CHECKPOINT_DIR / "best_phase2.pt",
                model,
                epoch,
                val_acc,
                phase=2,
            )
            logger.info(
                "  -> New best val_acc=%.4f, checkpoint saved.",
                best_val_acc,
            )

        # Periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            periodic_path = CHECKPOINT_DIR / f"phase2_epoch_{epoch + 1:03d}.pt"
            _save_checkpoint(periodic_path, model, epoch, val_acc, phase=2)
            logger.info("  -> Periodic checkpoint saved: %s", periodic_path)

    logger.info("Training complete. Best phase-2 val_acc=%.4f", best_val_acc)


if __name__ == "__main__":
    main()
