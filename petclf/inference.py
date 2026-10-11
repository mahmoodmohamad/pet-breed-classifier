import json
from pathlib import Path

import torch
from torch import nn

from petclf.models import FEATURE_DIMS, build_backbone
from petclf.preprocessing import preprocess


class Predictor:
    def __init__(self, name="resnet50", lr="0.001", models_dir="models"):
        d = Path(models_dir)
        self.backbone = build_backbone(name)
        self.head = nn.Linear(FEATURE_DIMS[name], 37)
        self.head.load_state_dict(torch.load(d / f"{name}_lr{lr}.pt"))
        self.head.eval()
        cfg = json.loads((d / f"{name}_lr{lr}_threshold.json").read_text())
        self.temperature = cfg["temperature"]
        self.threshold = cfg["threshold"]
        labels = json.loads(Path("data/label_map.json").read_text())
        self.labels = {int(k): v for k, v in labels.items()}

    def logits(self, source) -> torch.Tensor:
        with torch.inference_mode():
            x = preprocess(source).unsqueeze(0)
            return self.head(self.backbone(x))[0]

    def predict(self, source) -> dict:
        probs = (self.logits(source) / self.temperature).softmax(0)
        p, idx = probs.topk(3)
        top3 = [
            {"breed": self.labels[i.item()], "probability": round(v.item(), 4)}
            for v, i in zip(p, idx)
        ]
        decision = "uncertain" if p[0].item() < self.threshold else "accept"
        return {"decision": decision, "top3": top3}
