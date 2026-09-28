from collections import defaultdict
from pathlib import Path
import random

TRAINVAL_FILE = Path("data/oxford-iiit-pet/annotations/trainval.txt")
SEED = 42
random.seed(SEED)

lines = TRAINVAL_FILE.read_text().splitlines()
records = []

for line in lines:
    image_id, class_id, species, breed_id = line.split()
    records.append(
        {
            "image_id": image_id,
            "class_id": int(class_id),
            "species": int(species),
            "breed_id": int(breed_id),
        }
    )

records_by_class = defaultdict(list)
for record in records:
    records_by_class[record["class_id"]].append(record)

train_records_all = []
val_records_all = []

for class_id, class_records in records_by_class.items():
    random.shuffle(class_records)

    split_index = int(len(class_records) * 0.8)

    train_records = class_records[:split_index]
    val_records = class_records[split_index:]

    train_records_all.extend(train_records)
    val_records_all.extend(val_records)

    print(
        f"Class {class_id}: "
        f"total={len(class_records)}, "
        f"train={len(train_records)}, "
        f"val={len(val_records)}"
    )

print("train:", len(train_records_all))
print("val:", len(val_records_all))
print("total:", len(train_records_all) + len(val_records_all))

out_dir = Path("data/splits")
out_dir.mkdir(parents=True, exist_ok=True)

train_ids = sorted(r["image_id"] for r in train_records_all)
val_ids = sorted(r["image_id"] for r in val_records_all)

(out_dir / "train.txt").write_text("\n".join(train_ids) + "\n")
(out_dir / "val.txt").write_text("\n".join(val_ids) + "\n")