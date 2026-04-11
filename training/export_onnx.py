"""
Export a trained EfficientNetV2-S PyTorch checkpoint to ONNX format.

Reads training/checkpoints/label_mapping.json to determine the number of
output classes, reconstructs the model architecture via timm, loads the
saved weights, and exports to ONNX opset 17 with a fixed batch size of 1
and input shape [1, 3, 300, 300].

After export, parity between the PyTorch model and the ONNX runtime session
is validated over 10 random inputs.  The script prints the ONNX model size
and a PASS / WARNING line based on the parity check result.

Usage:
    python training/export_onnx.py
    python training/export_onnx.py --checkpoint path/to/ckpt.pt --output path/to/model.onnx
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths  (relative to this file so the script is location-independent)
# ---------------------------------------------------------------------------

_TRAINING_DIR = Path(__file__).resolve().parent

DEFAULT_CHECKPOINT = _TRAINING_DIR / "checkpoints" / "best_phase2.pt"
DEFAULT_OUTPUT = _TRAINING_DIR.parent / "analyzer" / "models" / "model.onnx"
LABEL_MAPPING_PATH = _TRAINING_DIR / "checkpoints" / "label_mapping.json"

# Parity validation settings
_PARITY_RUNS = 10
_PARITY_THRESHOLD = 1e-4
_INPUT_SHAPE = (1, 3, 300, 300)
_MODEL_ARCH = "tf_efficientnetv2_s"
_OPSET_VERSION = 17


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_num_classes(label_mapping_path: Path) -> int:
    """Return the number of output classes from label_mapping.json.

    The file is expected to have the shape::

        {"label_to_idx": {...}, "idx_to_label": {...}}

    We derive num_classes from the length of ``label_to_idx``.
    """
    if not label_mapping_path.exists():
        raise FileNotFoundError(
            f"Label mapping not found: {label_mapping_path}"
        )
    with label_mapping_path.open(encoding="utf-8") as fh:
        mapping: dict = json.load(fh)

    label_to_idx: dict = mapping["label_to_idx"]
    num_classes = len(label_to_idx)
    log.info("Detected %d classes from label mapping.", num_classes)
    return num_classes


def build_model(num_classes: int) -> torch.nn.Module:
    """Instantiate EfficientNetV2-S with the given output dimension."""
    model = timm.create_model(
        _MODEL_ARCH,
        pretrained=False,
        num_classes=num_classes,
    )
    return model


def load_checkpoint(
    model: torch.nn.Module, checkpoint_path: Path
) -> torch.nn.Module:
    """Load model weights from a checkpoint file.

    The checkpoint is expected to contain::

        {"epoch": int, "model_state_dict": {...}, "val_acc": float, "phase": int}
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint: dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    epoch = checkpoint.get("epoch", "unknown")
    val_acc = checkpoint.get("val_acc", float("nan"))
    phase = checkpoint.get("phase", "unknown")
    log.info(
        "Loaded checkpoint — epoch: %s, val_acc: %.4f, phase: %s",
        epoch,
        val_acc,
        phase,
    )
    return model


def export_onnx(model: torch.nn.Module, output_path: Path) -> None:
    """Export *model* to ONNX at *output_path* using opset 17."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*_INPUT_SHAPE)
    log.info("Exporting to ONNX: %s", output_path)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["data"],
        output_names=["model_output"],
        opset_version=_OPSET_VERSION,
        dynamic_axes=None,
    )
    log.info("ONNX export complete.")


def validate_parity(
    model: torch.nn.Module,
    onnx_path: Path,
    runs: int = _PARITY_RUNS,
    threshold: float = _PARITY_THRESHOLD,
) -> float:
    """Compare PyTorch and ONNX outputs on random inputs.

    Returns the maximum absolute difference observed across all *runs*.
    """
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    max_diff = 0.0

    for _ in range(runs):
        x = torch.randn(*_INPUT_SHAPE)

        with torch.no_grad():
            pt_output: np.ndarray = model(x).numpy()

        ort_output: np.ndarray = session.run(None, {input_name: x.numpy()})[0]

        diff = float(np.max(np.abs(pt_output - ort_output)))
        if diff > max_diff:
            max_diff = diff

    return max_diff


def print_model_size_mb(onnx_path: Path) -> None:
    """Print the ONNX file size in megabytes."""
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained EfficientNetV2-S checkpoint to ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=(
            "Path to the .pt checkpoint file "
            f"(default: {DEFAULT_CHECKPOINT})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Destination path for the exported .onnx file "
            f"(default: {DEFAULT_OUTPUT})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Determine number of classes from label mapping
    num_classes = load_num_classes(LABEL_MAPPING_PATH)

    # 2. Build model architecture
    model = build_model(num_classes)

    # 3. Load trained weights
    model = load_checkpoint(model, args.checkpoint)

    # 4. Switch to inference mode (disables dropout, batch-norm uses running stats)
    model.train(mode=False)

    # 5. Export to ONNX
    export_onnx(model, args.output)

    # 6. Parity validation
    log.info("Running parity validation over %d inputs ...", _PARITY_RUNS)
    max_diff = validate_parity(model, args.output)

    # 7. Report
    print_model_size_mb(args.output)

    if max_diff < _PARITY_THRESHOLD:
        print(
            f"PASS  — max absolute difference: {max_diff:.2e} "
            f"(threshold {_PARITY_THRESHOLD:.0e})"
        )
    else:
        print(
            f"WARNING — max absolute difference: {max_diff:.2e} exceeds "
            f"threshold {_PARITY_THRESHOLD:.0e}. "
            "Review numerical precision of the ONNX export."
        )


if __name__ == "__main__":
    main()
