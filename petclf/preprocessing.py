import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def load_image(source) -> Image.Image:
    """Open a path or file-like object and force RGB (grayscale/CMYK/alpha)."""
    with Image.open(source) as img:
        return img.convert("RGB")


def preprocess(source) -> torch.Tensor:
    """Shared eval preprocessing: image source -> (3, 224, 224) float tensor."""
    return eval_transform(load_image(source))
