"""
detect.py — Face Detection & Preprocessing
Project: P007 — Privacy-Preserving Face Recognition using Federated Learning
Step 1 | Author: Abderrahim

This module provides detect_face(), the core preprocessing function.
Create one MTCNN instance in prepare_dataset.py and pass it here.
Never create MTCNN inside this function (expensive weight reload each call).
"""

import logging
from typing import Optional

import torch
from facenet_pytorch import MTCNN
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


def is_image_blurry(image_path: str, threshold: float = 100.0) -> bool:
    """
    Detect blur using the Laplacian variance method.

    A sharp image has strong, high-frequency edges → high variance in the Laplacian.
    A blurry image has soft edges → low variance.

    Args:
        image_path: Path to the image file.
        threshold:  Variance below this → image is considered blurry.
                    100.0 is a reliable starting value for face photos.
                    Lower to 50 if too many images are rejected.
                    Raise to 150 if blurry images are slipping through.

    Returns:
        True  → blurry (should skip)
        False → acceptably sharp (proceed with detection)
    """
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python not installed; skipping blur check.")
        return False

    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        # cv2 couldn't load it — treat as problematic
        return True

    variance = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    is_blurry = variance < threshold
    if is_blurry:
        logger.debug(f"Blur variance={variance:.1f} < threshold={threshold} → blurry: {image_path}")
    return is_blurry


def detect_face(
    image_path: str,
    mtcnn: MTCNN,
    check_blur: bool = True,
    blur_threshold: float = 100.0,
) -> Optional[torch.Tensor]:
    """
    Detect, align, crop, and normalize a single face from an image file.

    Full pipeline executed inside this function:
        1. Load image as PIL RGB (handles JPEG, PNG, grayscale, RGBA)
        2. Optional blur check via Laplacian variance
        3. MTCNN detection:
              P-Net → R-Net → O-Net → eye landmark alignment → crop → normalize
        4. Output validation (shape + value range sanity check)

    The MTCNN instance must be configured with:
        image_size=160, margin=20, min_face_size=40,
        keep_all=False, post_process=True

    With post_process=True, MTCNN automatically applies:
        normalized_pixel = (raw_pixel - 127.5) / 128.0
    → output range is [-1.0, 1.0]. Do NOT re-normalize after calling this function.

    Args:
        image_path:     Absolute or relative path to input image (JPEG/PNG).
        mtcnn:          Pre-initialized MTCNN instance. Create once, pass everywhere.
        check_blur:     Skip the image if Laplacian variance is below blur_threshold.
        blur_threshold: Variance threshold for blur detection (default: 100.0).

    Returns:
        torch.Tensor of shape [3, 160, 160], dtype float32, values ∈ [-1.0, 1.0]
        — OR —
        None  if any of the following occur:
              • File cannot be opened or is corrupted
              • Image is too blurry (when check_blur=True)
              • MTCNN fails to detect a face
              • Output tensor has unexpected shape or extreme values
    """
    # ── 1. Load image ──────────────────────────────────────────────────────
    try:
        img = Image.open(image_path).convert("RGB")
        # .convert("RGB") is non-negotiable:
        #   PNG can be RGBA (4ch) → drops alpha
        #   Grayscale JPEG is "L" (1ch) → triplicates to 3ch
        #   Both cases would crash MTCNN without this conversion
    except FileNotFoundError:
        logger.warning(f"SKIP [not found]   {image_path}")
        return None
    except UnidentifiedImageError:
        logger.warning(f"SKIP [unreadable]  {image_path} — corrupted or unsupported format")
        return None
    except OSError as e:
        logger.warning(f"SKIP [OS error]    {image_path}: {e}")
        return None

    # ── 2. Blur check ──────────────────────────────────────────────────────
    if check_blur and is_image_blurry(image_path, threshold=blur_threshold):
        logger.warning(f"SKIP [too blurry]  {image_path}")
        return None

    # ── 3. MTCNN detection ─────────────────────────────────────────────────
    # With keep_all=False:
    #   • Multiple faces → returns the highest-confidence one only
    #   • No face → returns None
    # With post_process=True:
    #   • Output tensor is already normalized to [-1, 1]
    try:
        face_tensor = mtcnn(img)
    except Exception as e:
        # Rare: can happen on extremely small or malformed images
        logger.warning(f"SKIP [MTCNN error] {image_path}: {e}")
        return None

    # ── 4. Handle no detection ─────────────────────────────────────────────
    if face_tensor is None:
        logger.warning(f"SKIP [no face]     {image_path}")
        return None

    # ── 5. Shape validation ────────────────────────────────────────────────
    expected_shape = torch.Size([3, 160, 160])
    if face_tensor.shape != expected_shape:
        logger.warning(
            f"SKIP [bad shape {face_tensor.shape}] {image_path} — "
            f"expected {expected_shape}"
        )
        return None

    # ── 6. Value range sanity check ────────────────────────────────────────
    # Legitimate normalized tensors should be in [-1.0, 1.0]
    # A 10% margin accounts for floating-point edge cases
    if face_tensor.min() < -1.1 or face_tensor.max() > 1.1:
        logger.warning(
            f"SKIP [bad range min={face_tensor.min():.3f} max={face_tensor.max():.3f}] "
            f"{image_path}"
        )
        return None

    # ── 7. Return validated tensor ─────────────────────────────────────────
    # Shape:  [3, 160, 160]  (C, H, W — PyTorch convention)
    # dtype:  torch.float32
    # range:  ≈ [-1.0, 1.0]
    return face_tensor