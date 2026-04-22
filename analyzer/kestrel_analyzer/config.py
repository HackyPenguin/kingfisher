from pathlib import Path

ANALYZER_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ANALYZER_DIR / "models"


def _read_app_version() -> str:
    for candidate in (
        ANALYZER_DIR / "VERSION.txt",
        ANALYZER_DIR.parent / "VERSION.txt",
    ):
        try:
            if candidate.is_file():
                for raw_line in candidate.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.lower().startswith("version:"):
                        return line.split(":", 1)[1].strip() or "unknown"
                    return line
        except Exception:
            continue
    return "unknown"


VERSION = _read_app_version()

SPECIESCLASSIFIER_PATH = MODELS_DIR / "model.onnx"
SPECIESCLASSIFIER_LABELS = MODELS_DIR / "labels.txt"
QUALITYCLASSIFIER_PATH = MODELS_DIR / "quality.keras"
QUALITY_NORMALIZATION_DATA_PATH = MODELS_DIR / "quality_normalization_data.csv"
MASK_RCNN_WEIGHTS_PATH = MODELS_DIR / "mask_rcnn_resnet50_fpn_v2.pth"

WILDLIFE_CATEGORIES = [
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "bird"
]

RAW_EXTENSIONS = [".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".pef", ".sr2"]
JPEG_EXTENSIONS = [".jpg", ".jpeg", ".png", '.tiff', '.tif']

DATABASE_NAME = "kingfisher_database.csv"
METADATA_FILENAME = "kingfisher_metadata.json"
SCENEDATA_FILENAME = "kingfisher_scenedata.json"
KESTREL_DIR_NAME = ".kingfisher"
LOG_FILENAME_PREFIX = "kingfisher_error"
LOG_FILE_EXTENSION = "json"
