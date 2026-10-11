import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from petclf.dataset import PetDataset
from petclf.models import build_backbone

name = sys.argv[1]
model = build_backbone(name)
out_dir = Path("data/features")
out_dir.mkdir(parents=True, exist_ok=True)

for split in ("train", "val"):
    start = time.time()
    loader = DataLoader(PetDataset(split), batch_size=64, shuffle=False)
    xs, ys = [], []
    with torch.inference_mode():
        for images, labels in loader:
            xs.append(model(images))
            ys.append(labels)
    x, y = torch.cat(xs), torch.cat(ys)
    torch.save({"x": x, "y": y}, out_dir / f"{name}_{split}.pt")
    print(name, split, tuple(x.shape), f"{time.time() - start:.0f}s")
