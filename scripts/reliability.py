import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import torch
from torch import nn

from petclf.models import FEATURE_DIMS

name, lr = sys.argv[1], sys.argv[2]
va = torch.load(f"data/features/{name}_val.pt")
head = nn.Linear(FEATURE_DIMS[name], 37)
head.load_state_dict(torch.load(f"models/{name}_lr{lr}.pt"))
T = torch.load(f"models/{name}_lr{lr}_temp.pt")["temperature"]
with torch.no_grad():
    logits = head(va["x"])
y = va["y"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (title, probs) in zip(
    axes,
    [
        ("before (T=1)", logits.softmax(1)),
        (f"after (T={T:.3f})", (logits / T).softmax(1)),
    ],
):
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, 11)
    xs, accs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean().item())
            accs.append(correct[m].mean().item())
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.bar(xs, accs, width=0.08, alpha=0.7, label="accuracy per bin")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend()
fig.tight_layout()
out = f"docs/reliability_{name}.png"
fig.savefig(out, dpi=120)

mlflow.set_experiment("pet-breed-head")
with mlflow.start_run(run_name=f"{name}-lr{lr}-reliability"):
    mlflow.log_artifact(out)
print("saved", out)
