"""
test_step1.py — Quick validation scripts for Step 1
Run these after implementing detect.py and prepare_dataset.py.

Usage (from project root):
    python test_step1.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_single_image(image_path: str):
    """
    Run detect_face on one image and print a full diagnostic report.
    Change image_path to any photo in your data/raw/ folder.
    """
    import torch
    from facenet_pytorch import MTCNN
    from src.preprocessing.detect import detect_face

    print(f"\n{'='*55}")
    print("TEST 1: Single image detection")
    print(f"{'='*55}")
    print(f"Image: {image_path}")

    device = torch.device("cpu")
    mtcnn = MTCNN(
        image_size=160, margin=20, min_face_size=40,
        keep_all=False, post_process=True, device=device
    )

    tensor = detect_face(image_path, mtcnn, check_blur=True)

    if tensor is None:
        print("RESULT: FAILED — detect_face returned None")
        print("Action: Check the image path, and ensure the photo has a clear face.")
        return False

    # Run all checks
    checks = {
        "Shape == [3, 160, 160]": tensor.shape == torch.Size([3, 160, 160]),
        "dtype == float32":       tensor.dtype == torch.float32,
        "min >= -1.1":            tensor.min().item() >= -1.1,
        "max <= 1.1":             tensor.max().item() <= 1.1,
    }

    print("\nDiagnostics:")
    print(f"  shape : {tensor.shape}")
    print(f"  dtype : {tensor.dtype}")
    print(f"  min   : {tensor.min().item():.4f}")
    print(f"  max   : {tensor.max().item():.4f}")
    print(f"  mean  : {tensor.mean().item():.4f}")

    print("\nChecks:")
    all_passed = True
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}  {check_name}")
        if not result:
            all_passed = False

    if all_passed:
        print("\nRESULT: ALL CHECKS PASSED ✓")
    else:
        print("\nRESULT: SOME CHECKS FAILED ✗")

    return all_passed


def visualize_saved_tensor(tensor_path: str, save_path: str = "debug_crop.png"):
    """
    Load a saved .pt tensor and convert it back to a viewable PNG image.
    Open debug_crop.png and verify:
      - Face is centered
      - Eyes are roughly horizontal (aligned)
      - No black borders
      - No face cut off at edges
    """
    import torch
    import numpy as np
    from PIL import Image

    print(f"\n{'='*55}")
    print("TEST 2: Visual verification of saved tensor")
    print(f"{'='*55}")
    print(f"Tensor: {tensor_path}")

    tensor = torch.load(tensor_path, weights_only=True)

    # Undo MTCNN's normalization: x_orig = (x * 128.0) + 127.5
    img_array = (tensor.permute(1, 2, 0).numpy() * 128.0 + 127.5)
    img_array = img_array.clip(0, 255).astype(np.uint8)

    img = Image.fromarray(img_array)
    img.save(save_path)
    print(f"Saved to: {save_path}")
    print("Open this file and verify:")
    print("  • Face is centered in the 160×160 frame")
    print("  • Eyes are roughly on the same horizontal line (aligned)")
    print("  • No black borders or padding artifacts")
    print("  • Face is not cut off at any edge")
    print("  • Correct person's face")

    try:
        img.show()
    except Exception:
        print("  (Could not auto-open image viewer — open debug_crop.png manually)")


def validate_all_tensors():
    """
    Load every .pt file in data/cropped/ and run shape/dtype/range checks.
    Reports any files that fail.
    """
    import torch

    cropped_dir = PROJECT_ROOT / "data" / "cropped"

    print(f"\n{'='*55}")
    print("TEST 3: Validate all saved tensors")
    print(f"{'='*55}")

    pt_files = sorted(cropped_dir.rglob("*.pt"))

    if not pt_files:
        print(f"No .pt files found in {cropped_dir}")
        print("Run prepare_dataset.py first.")
        return

    print(f"Found {len(pt_files)} tensor file(s)\n")

    failures = []
    for pt_path in pt_files:
        try:
            tensor = torch.load(pt_path, weights_only=True)
        except Exception as e:
            failures.append((str(pt_path), f"Load error: {e}"))
            print(f"  ✗ LOAD ERROR: {pt_path.name} — {e}")
            continue

        errors = []
        if tensor.shape != torch.Size([3, 160, 160]):
            errors.append(f"shape={tensor.shape} (expected [3,160,160])")
        if tensor.dtype != torch.float32:
            errors.append(f"dtype={tensor.dtype} (expected float32)")
        if tensor.min().item() < -1.1:
            errors.append(f"min={tensor.min().item():.3f} (expected >= -1.1)")
        if tensor.max().item() > 1.1:
            errors.append(f"max={tensor.max().item():.3f} (expected <= 1.1)")

        rel_path = pt_path.relative_to(PROJECT_ROOT)
        if errors:
            failures.append((str(rel_path), "; ".join(errors)))
            print(f"  ✗ FAIL: {rel_path} — {'; '.join(errors)}")
        else:
            print(f"  ✓ OK:   {rel_path} | "
                  f"shape={list(tensor.shape)} | "
                  f"min={tensor.min().item():.3f} | "
                  f"max={tensor.max().item():.3f}")

    print(f"\nSummary: {len(pt_files) - len(failures)}/{len(pt_files)} passed")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for path, reason in failures:
            print(f"  {path}: {reason}")
    else:
        print("All tensors are valid ✓")


if __name__ == "__main__":
    # ── Test 1: Change this path to a real image in your data/raw/ ──
    SAMPLE_IMAGE = str(PROJECT_ROOT / "data" / "raw" / "abderrahim" / "photo1.jpg")
    test_single_image(SAMPLE_IMAGE)

    # ── Test 2: Change this path to a saved tensor after running prepare_dataset.py ──
    SAMPLE_TENSOR = str(PROJECT_ROOT / "data" / "cropped" / "abderrahim" / "photo1.pt")
    tensor_exists = Path(SAMPLE_TENSOR).exists()
    if tensor_exists:
        visualize_saved_tensor(SAMPLE_TENSOR, save_path="debug_crop.png")
    else:
        print(f"\nSkipping Test 2: {SAMPLE_TENSOR} not found yet.")
        print("Run prepare_dataset.py first, then re-run this script.")

    # ── Test 3: Validate everything in data/cropped/ ──
    validate_all_tensors()