"""
AUGMENTATION MODULE (Client-Side Training Pipeline)

This module defines image augmentation strategies used in the
federated learning client pipeline for face recognition.

PURPOSE:
- Improve generalization on small client datasets
- Simulate real-world variations (lighting, pose, slight misalignment)
- Reduce overfitting during local training

IMPORTANT:
- Augmentation is applied ONLY during client training
- No augmented images are stored
- Each forward pass may generate different variants
"""

import torchvision.transforms as transforms


class FaceAugmentation:
    """
    Face augmentation pipeline for federated learning clients.
    Designed for tensor-based face images (160x160 from preprocessing).

    Supports:
    - Horizontal flip
    - Color jitter (lighting variation)
    - Optional affine transforms (pose variation)
    - Optional gaussian blur (out-of-focus simulation)
    """

    def __init__(
        self,
        use_affine: bool = True,
        use_blur: bool = False
    ):
        transforms_list = []

        # -----------------------------
        # Geometric augmentation
        # -----------------------------
        transforms_list.append(
            transforms.RandomHorizontalFlip(p=0.5)
        )

        # -----------------------------
        # Photometric augmentation
        # -----------------------------
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
        # (small pose variations only)
        # -----------------------------
        if use_affine:
            transforms_list.append(
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.02, 0.02),
                    scale=(0.95, 1.05),
                    shear=2
                )
            )

        # -----------------------------
        # Optional blur simulation
        # -----------------------------
        if use_blur:
            transforms_list.append(
                transforms.GaussianBlur(
                    kernel_size=5,
                    sigma=(0.1, 2.0)
                )
            )

        self.transform = transforms.Compose(transforms_list)

    def __call__(self, image):
        """
        Apply augmentation to a single image tensor.

        Args:
            image (Tensor or PIL.Image): input face image

        Returns:
            Augmented image
        """
        return self.transform(image)

    def augment_batch(self, images):
        """
        Apply augmentation to a batch of images.

        Args:
            images (list or iterable of tensors)

        Returns:
            list of augmented images
        """
        return [self.transform(img) for img in images]

    def generate_variants(self, image, n: int = 5):
        """
        Generate multiple stochastic augmented versions
        of the same image.

        Useful for:
        - contrastive learning
        - robustness testing
        - data balancing

        Args:
            image: input image tensor
            n: number of variants

        Returns:
            list of augmented images
        """
        return [self.transform(image) for _ in range(n)]