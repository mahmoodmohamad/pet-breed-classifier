import io

import pytest
import torch
from PIL import Image

from petclf.preprocessing import preprocess


@pytest.mark.parametrize(
    ("mode", "fmt"),
    [("L", "JPEG"), ("RGB", "JPEG"), ("CMYK", "JPEG"), ("RGBA", "PNG")],
)
def test_any_color_mode_gives_rgb_tensor(tmp_path, mode, fmt):
    path = tmp_path / f"img_{mode}.{fmt.lower()}"
    Image.new(mode, (300, 200)).save(path, format=fmt)
    out = preprocess(path)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_path_and_bytes_give_identical_tensor():
    path = "data/oxford-iiit-pet/images/Abyssinian_100.jpg"
    with open(path, "rb") as f:
        raw = f.read()
    assert torch.equal(preprocess(path), preprocess(io.BytesIO(raw)))
