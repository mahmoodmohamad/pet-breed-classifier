import numpy as np
import pytest
from PIL import Image

from data.corruptions import CORRUPTIONS, apply_corruption


@pytest.fixture
def img() -> Image.Image:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(160, 200, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def distance(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    return float(np.abs(x - y).mean())


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_shape_and_mode_preserved(img, name):
    for severity in (1, 2, 3):
        out = apply_corruption(img, name, severity)
        assert out.size == img.size
        assert out.mode == "RGB"


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_severity_increases_distortion(img, name):
    d = [distance(img, apply_corruption(img, name, s)) for s in (1, 2, 3)]
    assert 0 < d[0] < d[1] < d[2]