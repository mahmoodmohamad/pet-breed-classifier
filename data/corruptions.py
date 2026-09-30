import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

RESAMPLE = Image.Resampling.BILINEAR


def gaussian_blur(img: Image.Image, severity: int) -> Image.Image:
    radius = {1: 1, 2: 2, 3: 4}[severity]
    return img.filter(ImageFilter.GaussianBlur(radius))


def brightness_up(img: Image.Image, severity: int) -> Image.Image:
    factor = {1: 1.3, 2: 1.6, 3: 2.0}[severity]
    return ImageEnhance.Brightness(img).enhance(factor)


def brightness_down(img: Image.Image, severity: int) -> Image.Image:
    factor = {1: 0.7, 2: 0.5, 3: 0.3}[severity]
    return ImageEnhance.Brightness(img).enhance(factor)


def jpeg_compression(img: Image.Image, severity: int) -> Image.Image:
    quality = {1: 50, 2: 30, 3: 10}[severity]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def downscale_upscale(img: Image.Image, severity: int) -> Image.Image:
    size = {1: 128, 2: 96, 3: 64}[severity]
    small = img.resize((size, size), RESAMPLE)
    return small.resize(img.size, RESAMPLE)


def motion_blur(img: Image.Image, severity: int) -> Image.Image:
    k = {1: 5, 2: 11, 3: 21}[severity]
    arr = np.asarray(img, dtype=np.float32)
    padded = np.pad(arr, ((0, 0), (k // 2, k // 2), (0, 0)), mode="edge")
    width = arr.shape[1]
    out = sum(padded[:, i : i + width, :] for i in range(k)) / k
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


CORRUPTIONS = {
    "gaussian_blur": gaussian_blur,
    "brightness_up": brightness_up,
    "brightness_down": brightness_down,
    "jpeg_30": jpeg_compression,
    "downscale_96": downscale_upscale,
    "motion_blur": motion_blur,
}


def apply_corruption(img: Image.Image, name: str, severity: int) -> Image.Image:
    return CORRUPTIONS[name](img.convert("RGB"), severity)