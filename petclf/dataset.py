import csv
from pathlib import Path

from torch.utils.data import Dataset

from petclf.preprocessing import eval_transform, load_image


class PetDataset(Dataset):
    def __init__(self, split, manifest="data/manifest.csv", transform=eval_transform):
        with Path(manifest).open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.rows = [
            r for r in rows if r["split"] == split and r["corruption"] == "none"
        ]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = load_image(row["path"])
        return self.transform(image), int(row["class_index"])
