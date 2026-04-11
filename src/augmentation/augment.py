"""
AUGMENTATION MODULE (Client-Side Preprocessing)

This module defines image augmentation strategies used in the
federated learning client pipeline for face recognition.

PURPOSE:
- Improve model generalization on small client datasets
- Simulate real-world variations (lighting, pose, orientation)
- Reduce overfitting during local training

IMPORTANT:
- Augmentation is applied ONLY on client devices
- Images are NOT stored after augmentation
- Each training iteration may produce different augmented versions
"""

import torchvision.transforms as transforms


class FaceAugmentation:
    """
    Face augmentation pipeline for federated learning clients.

    Supports configurable augmentation strength:
    - affine (pose/misalignment)
    - gaussian blur (out-of-focus simulation)
    """

    def __init__(
        self,
        use_affine=True,
        use_blur=False
    ):
        """
        Initialize augmentation pipeline.

        Args:
            use_affine (bool): enable slight geometric transformations
            use_blur (bool): enable gaussian blur simulation
        """

        transforms_list = []

        # -----------------------------
        # Basic augmentations
        # -----------------------------
        transforms_list.append(
            transforms.RandomHorizontalFlip(p=0.5)
        )

        transforms_list.append(
            transforms.RandomRotation(degrees=10)
        )

        transforms_list.append(
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            )
        )

        # -----------------------------
        # Optional affine transform
        # -----------------------------
        if use_affine:
            transforms_list.append(
                transforms.RandomAffine(
                    degrees=5,              # small rotation
                    translate=(0.02, 0.02), # small shift
                    scale=(0.95, 1.05),     # slight zoom
                    shear=2                 # slight distortion
                )
            )

        # -----------------------------
        # Optional gaussian blur
        # -----------------------------
        if use_blur:
            transforms_list.append(
                transforms.GaussianBlur(
                    kernel_size=3,
                    sigma=(0.1, 1.0)
                )
            )

        self.transform = transforms.Compose(transforms_list)

    def __call__(self, image):
        """
        Apply augmentation to a single image.
        """
        return self.transform(image)

    def augment_batch(self, images):
        """
        Apply augmentation to a batch of images.
        """
        return [self.transform(img) for img in images]

    def generate_variants(self, image, n=5):
        """
        Generate N augmented versions of the same image.
        """
        return [self.transform(image) for _ in range(n)]