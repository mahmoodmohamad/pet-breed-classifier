import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image

ANN_FILE = Path("data/oxford-iiit-pet/annotations/trainval.txt")
IMG_DIR = Path("data/oxford-iiit-pet/images")
LABEL_MAP = json.loads(Path("data/label_map.json").read_text())
OUT_FILE = Path("data/manifest.csv")
SPECIES = {1: "cat", 2: "dog"}

info = {}
for line in ANN_FILE.read_text().splitlines():
    image_id, class_id, species, _ = line.split()
    info[image_id] = (int(class_id) - 1, int(species))

rows = []
for split in ("train", "val"):
    ids = Path(f"data/splits/{split}.txt").read_text().split()
    for image_id in ids:
        class_index, species = info[image_id]
        path = IMG_DIR / f"{image_id}.jpg"
        with Image.open(path) as img:
            width, height = img.size
        rows.append(
            {
                "image_id": image_id,
                "path": path.as_posix(),
                "breed": LABEL_MAP[str(class_index)],
                "species": SPECIES[species],
                "class_index": class_index,
                "split": split,
                "corruption": "none",
                "severity": 0,
                "width": width,
                "height": height,
            }
        )

rows.sort(key=lambda r: r["image_id"])
with OUT_FILE.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(len(rows), Counter(r["split"] for r in rows))