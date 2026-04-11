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

    This class applies random transformations to face images
    before feeding them into the model during training.
    """

    def __init__(self):
        """
        Initialize augmentation pipeline.
        """

        self.transform = transforms.Compose([
            # Randomly flip face horizontally (helps with left/right variation)
            transforms.RandomHorizontalFlip(p=0.5),

            # Small rotation to simulate head tilt
            transforms.RandomRotation(degrees=10),

            # Adjust brightness, contrast, and saturation
            # simulates different lighting conditions
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            )
        ])

    def __call__(self, image):
        """
        Apply augmentation to a single image.

        Args:
            image (Tensor or PIL Image):
                Input face image.

        Returns:
            Tensor or PIL Image:
                Augmented image.
        """
        return self.transform(image)

    def augment_batch(self, images):
        """
        Apply augmentation to a batch of images.

        Args:
            images (list or tensor batch):
                Collection of face images.

        Returns:
            list:
                Augmented images.
        """
        return [self.transform(img) for img in images]
    
def generate_variants(self, image, n=5):
    """
    Generate N augmented versions of the same image.

    Useful for dataset expansion or testing scenarios.

    Args:
        image:
            Input face image (Tensor or PIL Image)

        n (int):
            Number of augmented versions to generate

    Returns:
        list:
            List of augmented images
    """
    return [self.transform(image) for _ in range(n)]