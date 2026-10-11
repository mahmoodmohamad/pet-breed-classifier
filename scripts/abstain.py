import json
import sys

import mlflow
import torch
from torch import nn

from petclf.models import FEATURE_DIMS

name, lr, target = sys.argv[1], sys.argv[2], float(sys.argv[3])
va = torch.load(f"data/features/{name}_val.pt")
head = nn.Linear(FEATURE_DIMS[name], 37)
head.load_state_dict(torch.load(f"models/{name}_lr{lr}.pt"))
T = torch.load(f"models/{name}_lr{lr}_temp.pt")["temperature"]
with torch.no_grad():
    probs = (head(va["x"]) / T).softmax(1)
conf, pred = probs.max(1)
correct = (pred == va["y"]).float()

chosen = None
for t in [x / 100 for x in range(30, 100, 5)]:
    keep = conf >= t
    coverage = keep.float().mean().item()
    sel_acc = correct[keep].mean().item() if keep.any() else 0.0
    print(f"threshold={t:.2f} coverage={coverage:.3f} selective_acc={sel_acc:.3f}")
    if chosen is None and sel_acc >= target:
        chosen = (t, coverage, sel_acc)

if chosen is None:
    raise SystemExit(f"no threshold reaches selective accuracy {target}")
t, coverage, sel_acc = chosen
print(f"CHOSEN threshold={t:.2f} coverage={coverage:.3f} selective_acc={sel_acc:.3f}")
with open(f"models/{name}_lr{lr}_threshold.json", "w") as f:
    json.dump({"threshold": t, "temperature": T}, f)

mlflow.set_experiment("pet-breed-head")
with mlflow.start_run(run_name=f"{name}-lr{lr}-abstention"):
    mlflow.log_params({"backbone": name, "target_selective_acc": target})
    mlflow.log_metrics({"threshold": t, "coverage": coverage, "selective_acc": sel_acc})
