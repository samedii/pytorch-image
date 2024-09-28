from pytorch_image import Image


def test_open_image():
    image = Image.open("tests/dragon.jpg")
    assert isinstance(image, Image)
    assert image.path is not None
    assert image.data.shape[1] == 3  # Assuming RGB image