import PIL.Image
from pytorch_image import Image


def test_pil_conversion():
    image = Image.open("tests/dragon.jpg")
    pil_image = image.pil_image()
    
    assert isinstance(pil_image, PIL.Image.Image)
    assert pil_image.mode == "RGB"  # Assuming RGB image