import albumentations as A
import numpy as np
from pytorch_image import Image


def test_augment_image_horizontal_flip():
    # Load the original image
    original_image = Image.open("tests/dragon.jpg")
    
    # Apply horizontal flip augmentation
    augmented_image = original_image.augment(A.HorizontalFlip(p=1.0))
    
    # Convert images to numpy arrays for comparison
    original_array = original_image.numpy()
    augmented_array = augmented_image.numpy()
    
    # Check if the augmented image is horizontally flipped
    assert np.array_equal(original_array, np.fliplr(augmented_array)), "The image was not correctly flipped horizontally"
    
    # Check if the augmented image has the same shape as the original
    assert original_array.shape == augmented_array.shape, "The augmented image shape does not match the original"