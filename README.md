# pytorch-image

Minimalistic library designed to be an interface for the model input with a lightweight `Image` class.

```python
from my_model import MyModel, Image

image = Image.from_numpy(numpy_image)
predictions = model(image)
```

It also offers a simple interface for essential image operations such as loading, resizing, augmenting, and saving.

## Installation

```bash
pip install pytorch-image
```

```bash
poetry add pytorch-image
```

## Usage

```python
from pytorch_image import Image
import albumentations


image = Image.open("path/to/image.png")

image.augment(albumentations.HorizontalFlip())

image.torch()
```

## Available Methods

The `Image` class provides the following methods:

### Class Methods

- `Image.open(path)`: Open an image file and create an Image instance.
- `Image.from_pil_image(pil_image)`: Create an Image instance from a PIL Image.
- `Image.from_numpy(numpy_image)`: Create an Image instance from a NumPy array. Expects HWC format, uint8 (0-255).
- `Image.from_torch(torch_tensor)`: Create an Image instance from a PyTorch tensor. Expects NCHW format, float (0-1).
- `Image.cat(images, dim=0)`: Concatenate multiple Image instances.

### Instance Methods

- `resize(shape, mode="bicubic", align_corners=None, antialias=None)`: Resize the image.
- `augment(augmenter)`: Apply augmentations to the image.
- `map(func)`: Apply a function to the image data.
- `replace(**kwargs)`: Create a new Image instance with replaced attributes.
- `torch()`: Get the image data as a PyTorch tensor.
- `numpy()`: Get the image data as a NumPy array.
- `representation()`: Get a normalized representation of the image data.
- `save(path)`: Save the image to a file.
- `pil_image()`: Convert the image to a PIL Image.

### Properties and Special Methods

- `shape`: Get the shape of the image data.
- `__len__()`: Get the number of images in the batch.
- `__iter__()`: Iterate over images in the batch.
- `__getitem__(index)`: Get a specific image from the batch.
- `_repr_png_()`: Get a PNG representation for Jupyter notebook display.

For detailed information on each method, please refer to the docstrings in the source code.
