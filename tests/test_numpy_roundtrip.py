import numpy as np

from pytorch_image import Image


def test_numpy_roundtrip():
    # Load the original image
    original_image = Image.open("tests/dragon.jpg")

    # Convert to numpy
    numpy_array = original_image.numpy()

    # Convert back to Image
    reconstructed_image = Image.from_numpy(numpy_array)

    # Convert both images to numpy for comparison
    original_array = original_image.numpy()
    reconstructed_array = reconstructed_image.numpy()

    # Check if the arrays are exactly equal (no tolerance needed now)
    np.testing.assert_array_equal(original_array, reconstructed_array)

    # Check if the shapes are the same
    assert (
        original_array.shape == reconstructed_array.shape
    ), "The shape of the reconstructed image does not match the original"

    # Check if the data type is preserved
    assert (
        original_array.dtype == reconstructed_array.dtype
    ), "The data type of the reconstructed image does not match the original"

    # Check if the range of values is preserved (assuming 8-bit images)
    assert (
        np.min(reconstructed_array) >= 0 and np.max(reconstructed_array) <= 255
    ), "The value range of the reconstructed image is not preserved"

    # Check if the mean absolute difference is small
    mean_abs_diff = np.mean(
        np.abs(original_array.astype(float) - reconstructed_array.astype(float))
    )
    assert (
        mean_abs_diff < 1.0
    ), f"Mean absolute difference ({mean_abs_diff}) is too large"
