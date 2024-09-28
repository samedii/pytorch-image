import numpy as np
import PIL.Image
import pytest
import torch

from pytorch_image import Image


@pytest.fixture
def sample_image_data():
    return torch.rand(1, 3, 64, 64)


@pytest.fixture
def sample_numpy_image():
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


def test_image_creation(sample_image_data):
    image = Image(data=sample_image_data)
    assert isinstance(image, Image)
    assert torch.allclose(image.data, sample_image_data)


def test_open_image(tmp_path):
    file_path = tmp_path / "test_image.png"
    PIL.Image.new("RGB", (64, 64)).save(file_path)

    image = Image.open(file_path)
    assert isinstance(image, Image)
    assert image.path == file_path
    assert image.data.shape == (1, 3, 64, 64)


def test_from_pil_image():
    pil_image = PIL.Image.new("RGB", (64, 64))
    image = Image.from_pil_image(pil_image)
    assert isinstance(image, Image)
    assert image.data.shape == (1, 3, 64, 64)


def test_from_numpy(sample_numpy_image):
    image = Image.from_numpy(sample_numpy_image)
    assert isinstance(image, Image)
    assert image.data.shape == (1, 3, 64, 64)
    assert torch.allclose(
        image.data,
        torch.from_numpy(sample_numpy_image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255),
    )


def test_cat():
    images = [Image(data=torch.rand(1, 3, 64, 64)) for _ in range(3)]
    cat_image = Image.cat(images)
    assert isinstance(cat_image, Image)
    assert cat_image.data.shape == (3, 3, 64, 64)


def test_shape_and_len(sample_image_data):
    image = Image(data=sample_image_data)
    assert image.shape == (1, 3, 64, 64)
    assert len(image) == 1


def test_iteration(sample_image_data):
    image = Image(data=sample_image_data)
    for sub_image in image:
        assert isinstance(sub_image, Image)
        assert sub_image.data.shape == (1, 3, 64, 64)


def test_getitem(sample_image_data):
    image = Image(data=sample_image_data)
    sub_image = image[0]
    assert isinstance(sub_image, Image)
    assert sub_image.data.shape == (1, 3, 64, 64)


def test_resize():
    image = Image(data=torch.rand(1, 3, 64, 64))
    resized_image = image.resize((32, 32))
    assert isinstance(resized_image, Image)
    assert resized_image.data.shape == (1, 3, 32, 32)


def test_map():
    image = Image(data=torch.rand(1, 3, 64, 64))
    mapped_image = image.map(lambda x: x * 2)
    assert isinstance(mapped_image, Image)
    assert torch.allclose(mapped_image.data, image.data * 2)


def test_replace():
    image = Image(data=torch.rand(1, 3, 64, 64))
    new_data = torch.rand(1, 3, 32, 32)
    replaced_image = image.replace(data=new_data)
    assert isinstance(replaced_image, Image)
    assert torch.allclose(replaced_image.data, new_data)


def test_torch():
    image = Image(data=torch.rand(1, 3, 64, 64))
    tensor = image.torch()
    assert isinstance(tensor, torch.Tensor)
    assert torch.allclose(tensor, image.data)


def test_representation():
    image = Image(data=torch.rand(1, 3, 64, 64) * 2 - 1)
    rep = image.representation()
    assert isinstance(rep, torch.Tensor)
    assert rep.min() >= 0 and rep.max() <= 1


def test_save_and_pil_image(tmp_path):
    image = Image(data=torch.rand(1, 3, 64, 64))
    file_path = tmp_path / "test_save.png"
    image.save(file_path)
    assert file_path.exists()

    pil_image = image.pil_image()
    assert isinstance(pil_image, PIL.Image.Image)
    assert pil_image.size == (64, 64)
