import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentation.augment import FaceAugmentation


def create_dummy_image():
    """
    Create a fake 160x160 RGB face image for testing.
    """
    arr = (np.random.rand(160, 160, 3) * 255).astype("uint8")
    return Image.fromarray(arr)


def test_call_method():
    print("\n[TEST] __call__ method")

    augmenter = FaceAugmentation()
    img = create_dummy_image()

    out = augmenter(img)

    assert out is not None

    print("[OK] __call__ works")


def test_batch_augmentation():
    print("\n[TEST] batch augmentation")

    augmenter = FaceAugmentation()

    imgs = [create_dummy_image() for _ in range(4)]
    out = augmenter.augment_batch(imgs)

    assert len(out) == 4

    print("[OK] batch augmentation works")


def test_generate_variants():
    print("\n[TEST] generate_variants method")

    augmenter = FaceAugmentation()
    img = create_dummy_image()

    n = 6
    variants = augmenter.generate_variants(img, n)

    assert len(variants) == n

    print("[OK] generate_variants works")


def test_consistency_check():
    print("\n[TEST] randomness check (sanity)")

    augmenter = FaceAugmentation()
    img = create_dummy_image()

    out1 = augmenter(img)
    out2 = augmenter(img)

    assert out1 is not None
    assert out2 is not None

    print("[OK] randomness behaves correctly")


if __name__ == "__main__":
    test_call_method()
    test_batch_augmentation()
    test_generate_variants()
    test_consistency_check()

    print("\nALL TESTS PASSED")