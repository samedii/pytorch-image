import numpy as np
from pytorch_image import Image


def test_numpy_conversion():
    image = Image.open("tests/dragon.jpg")
    numpy_array = image.numpy()
    
    assert isinstance(numpy_array, np.ndarray)
    assert numpy_array.ndim == 3
    assert numpy_array.dtype == np.uint8
    assert numpy_array.shape[2] == 3  # Assuming RGB image