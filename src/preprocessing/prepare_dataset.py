"""
prepare_dataset.py — Batch Face Detection & Preprocessing
Project: P007 — Privacy-Preserving Face Recognition using Federated Learning
Step 1 | Author: Abderrahim

Iterates data/raw/<person_name>/*.jpg|png
Calls detect_face() on every image
Saves successful detections as PyTorch tensors to data/cropped/<person_name>/<stem>.pt
Logs all skipped images with the reason

Usage (from project root):
    python src/preprocessing/prepare_dataset.py

Output directory is created automatically if it does not exist.
A full log is written to logs/preprocessing.log
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch
from facenet_pytorch import MTCNN

# ── Add project root to sys.path ───────────────────────────────────────────────
# This ensures `from src.preprocessing.detect import detect_face` works
# regardless of which directory you run this script from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.detect import detect_face  # noqa: E402

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CROPPED_DATA_DIR = PROJECT_ROOT / "data" / "cropped"
LOG_DIR = PROJECT_ROOT / "logs"

SUPPORTED_EXTENSIONS: set = {".jpg", ".jpeg", ".png"}

# MTCNN parameters — must match project spec exactly
MTCNN_IMAGE_SIZE: int = 160     # InceptionResNetV1 input size
MTCNN_MARGIN: int = 20          # Padding around bounding box (pixels)
MTCNN_MIN_FACE: int = 40        # Ignore detections smaller than this
MTCNN_KEEP_ALL: bool = False    # Return only the largest/best face
MTCNN_POST_PROCESS: bool = True # Auto-normalize output to [-1, 1]

# Blur detection
CHECK_BLUR: bool = True
BLUR_THRESHOLD: float = 100.0   # Laplacian variance threshold


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "preprocessing.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)


def create_mtcnn(device: torch.device) -> MTCNN:
    """
    Initialize MTCNN with the parameters specified in the project spec.

    This is called ONCE in main and reused for all images.
    Creating MTCNN loads neural network weights from disk — doing it
    per-image would be extremely slow (seconds per image instead of ms).
    """
    return MTCNN(
        image_size=MTCNN_IMAGE_SIZE,
        margin=MTCNN_MARGIN,
        min_face_size=MTCNN_MIN_FACE,
        keep_all=MTCNN_KEEP_ALL,
        post_process=MTCNN_POST_PROCESS,
        device=device,
    )


def process_person(
    person_dir: Path,
    output_dir: Path,
    mtcnn: MTCNN,
    logger: logging.Logger,
) -> Dict:
    """
    Run face detection on all images for one person.

    Args:
        person_dir: data/raw/<person_name>/
        output_dir: data/cropped/<person_name>/  (created if absent)
        mtcnn:      Pre-initialized MTCNN instance
        logger:     Logger instance

    Returns:
        stats dict: {total, saved, skipped, skipped_files: List[str]}
    """
    person_name = person_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict = {
        "total": 0,
        "saved": 0,
        "skipped": 0,
        "skipped_files": [],
    }

    # Collect all image files, sorted for deterministic processing order
    image_files: List[Path] = sorted(
        f for f in person_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        logger.warning(f"[{person_name}] No supported images found in {person_dir}")
        return stats

    for image_path in image_files:
        stats["total"] += 1
        logger.info(f"  [{person_name}] Processing: {image_path.name}")

        # Core detection call
        face_tensor = detect_face(
            image_path=str(image_path),
            mtcnn=mtcnn,
            check_blur=CHECK_BLUR,
            blur_threshold=BLUR_THRESHOLD,
        )

        if face_tensor is None:
            stats["skipped"] += 1
            stats["skipped_files"].append(image_path.name)
            # Warning already logged inside detect_face(); add a summary line here
            logger.info(f"  [{person_name}] → SKIPPED: {image_path.name}")
            continue

        # Save tensor as .pt (preserves exact float32 values; Step 2 loads this directly)
        output_path = output_dir / (image_path.stem + ".pt")
        torch.save(face_tensor, output_path)

        stats["saved"] += 1
        logger.info(
            f"  [{person_name}] → SAVED:   {output_path.name} | "
            f"shape={list(face_tensor.shape)} | "
            f"min={face_tensor.min().item():.4f} | "
            f"max={face_tensor.max().item():.4f}"
        )

    return stats


def run_preprocessing() -> None:
    logger = setup_logging()

    logger.info("=" * 65)
    logger.info("Step 1 — Face Detection & Preprocessing")
    logger.info(f"Project root : {PROJECT_ROOT}")
    logger.info(f"Input        : {RAW_DATA_DIR}")
    logger.info(f"Output       : {CROPPED_DATA_DIR}")
    logger.info("=" * 65)

    # ── Validate input directory ───────────────────────────────────────────
    if not RAW_DATA_DIR.exists():
        logger.error(f"Raw data directory does not exist: {RAW_DATA_DIR}")
        logger.error(
            "Create it and add one subdirectory per person, each containing "
            "3 JPEG or PNG photos.\n"
            "Expected layout:\n"
            "  data/raw/\n"
            "      abderrahim/photo1.jpg  photo2.jpg  photo3.jpg\n"
            "      kosai/    photo1.jpg  ...\n"
        )
        sys.exit(1)

    # ── Select device ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Initialize MTCNN once ──────────────────────────────────────────────
    logger.info("Initializing MTCNN (downloading weights on first run)...")
    mtcnn = create_mtcnn(device)
    logger.info("MTCNN ready.")
    logger.info("-" * 65)

    # ── Discover person folders ────────────────────────────────────────────
    person_dirs = sorted(d for d in RAW_DATA_DIR.iterdir() if d.is_dir())

    if not person_dirs:
        logger.error(f"No subdirectories found in {RAW_DATA_DIR}")
        logger.error("Each person must have their own folder: data/raw/<name>/")
        sys.exit(1)

    logger.info(f"Found {len(person_dirs)} person(s): {[d.name for d in person_dirs]}")
    logger.info("")

    # ── Process each person ────────────────────────────────────────────────
    global_stats: Dict = {"total": 0, "saved": 0, "skipped": 0}
    per_person_report: Dict = {}

    for person_dir in person_dirs:
        person_name = person_dir.name
        output_dir = CROPPED_DATA_DIR / person_name

        logger.info(f"── Person: {person_name} ──")
        person_stats = process_person(person_dir, output_dir, mtcnn, logger)

        for key in ("total", "saved", "skipped"):
            global_stats[key] += person_stats[key]

        per_person_report[person_name] = person_stats

        logger.info(
            f"  [{person_name}] Done: {person_stats['saved']}/{person_stats['total']} saved"
            + (f", {person_stats['skipped']} skipped: {person_stats['skipped_files']}"
               if person_stats["skipped"] else "")
        )
        logger.info("")

    # ── Write JSON report ──────────────────────────────────────────────────
    report = {
        "global": global_stats,
        "per_person": per_person_report,
        "config": {
            "image_size": MTCNN_IMAGE_SIZE,
            "margin": MTCNN_MARGIN,
            "min_face_size": MTCNN_MIN_FACE,
            "post_process": MTCNN_POST_PROCESS,
            "check_blur": CHECK_BLUR,
            "blur_threshold": BLUR_THRESHOLD,
        },
    }
    report_path = LOG_DIR / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ── Final summary ──────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"  Total images processed : {global_stats['total']}")
    logger.info(f"  Successfully saved     : {global_stats['saved']}")
    logger.info(f"  Skipped                : {global_stats['skipped']}")
    logger.info(f"  Output directory       : {CROPPED_DATA_DIR}")
    logger.info(f"  Full log               : {LOG_DIR / 'preprocessing.log'}")
    logger.info(f"  JSON report            : {report_path}")
    logger.info("=" * 65)

    if global_stats["saved"] == 0:
        logger.error(
            "No tensors were saved! Check:\n"
            "  1. Photos are in data/raw/<name>/ as .jpg or .png\n"
            "  2. Each photo contains a clearly visible front-facing face\n"
            "  3. Photos are not too blurry (lower BLUR_THRESHOLD if needed)\n"
            "  4. Faces are large enough in the frame (min_face_size=40px)"
        )
        sys.exit(1)

    # Warn if any person has zero saved tensors
    for name, stats in per_person_report.items():
        if stats["saved"] == 0:
            logger.warning(
                f"Person '{name}' has 0 saved tensors! "
                "This person will be missing from the training set. "
                "Have them retake their photos."
            )


if __name__ == "__main__":
    run_preprocessing()