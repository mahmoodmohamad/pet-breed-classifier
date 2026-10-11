from pathlib import Path

import pytest
import torch

from petclf.dataset import PetDataset

NEEDED = [
    Path("models/resnet50_lr0.001.pt"),
    Path("models/resnet50_lr0.001_threshold.json"),
    Path("data/features/resnet50_val.pt"),
]


@pytest.mark.skipif(
    not all(p.exists() for p in NEEDED), reason="model artifacts not available"
)
def test_api_logits_match_training_logits():
    from petclf.inference import Predictor

    predictor = Predictor()
    cached = torch.load("data/features/resnet50_val.pt")["x"]
    rows = PetDataset("val").rows
    for i in (0, 100, 500):
        api = predictor.logits(rows[i]["path"])
        with torch.inference_mode():
            train = predictor.head(cached[i : i + 1])[0]
        assert (api - train).abs().max().item() < 1e-4
