import sys
from pathlib import Path

import mlflow
import torch
from torch import nn

from petclf.models import FEATURE_DIMS

name, lr, epochs = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(42)

tr = torch.load(f"data/features/{name}_train.pt")
va = torch.load(f"data/features/{name}_val.pt")

head = nn.Linear(FEATURE_DIMS[name], 37)
opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
loss_fn = nn.CrossEntropyLoss()

mlflow.set_experiment("pet-breed-head")
with mlflow.start_run(run_name=f"{name}-lr{lr}"):
    mlflow.log_params({"backbone": name, "lr": lr, "epochs": epochs, "seed": 42})
    for epoch in range(epochs):
        perm = torch.randperm(len(tr["y"]))
        for i in range(0, len(perm), 128):
            idx = perm[i : i + 128]
            opt.zero_grad()
            loss = loss_fn(head(tr["x"][idx]), tr["y"][idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        val_logits = head(va["x"])
        val_acc = (val_logits.argmax(1) == va["y"]).float().mean().item()
        val_loss = loss_fn(val_logits, va["y"]).item()
    mlflow.log_metrics({"val_acc": val_acc, "val_loss": val_loss})
    Path("models").mkdir(exist_ok=True)
    out = Path("models") / f"{name}_lr{lr}.pt"
    torch.save(head.state_dict(), out)
    mlflow.log_artifact(str(out))
    print(name, lr, f"val_acc={val_acc:.4f} val_loss={val_loss:.4f}")
