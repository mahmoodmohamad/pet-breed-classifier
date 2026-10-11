import sys

import mlflow
import torch
from torch import nn

from petclf.models import FEATURE_DIMS

name, lr = sys.argv[1], sys.argv[2]
va = torch.load(f"data/features/{name}_val.pt")
head = nn.Linear(FEATURE_DIMS[name], 37)
head.load_state_dict(torch.load(f"models/{name}_lr{lr}.pt"))
with torch.no_grad():
    logits = head(va["x"])
y = va["y"]


def ece(probs, y, bins=10):
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean() * (conf[m].mean() - correct[m].mean()).abs()
    return float(total)


log_t = torch.zeros(1, requires_grad=True)
opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)


def closure():
    opt.zero_grad()
    loss = nn.functional.cross_entropy(logits / log_t.exp(), y)
    loss.backward()
    return loss


opt.step(closure)
T = log_t.exp().item()
before = ece(logits.softmax(1), y)
after = ece((logits / T).softmax(1), y)
print(f"T={T:.3f} ECE before={before:.4f} after={after:.4f}")

mlflow.set_experiment("pet-breed-head")
with mlflow.start_run(run_name=f"{name}-lr{lr}-calibration"):
    mlflow.log_params({"backbone": name, "lr": lr})
    mlflow.log_metrics({"temperature": T, "ece_before": before, "ece_after": after})
torch.save({"temperature": T}, f"models/{name}_lr{lr}_temp.pt")
