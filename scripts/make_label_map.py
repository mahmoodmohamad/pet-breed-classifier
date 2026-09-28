import json
from pathlib import Path

TRAINVAL_FILE = Path("data/oxford-iiit-pet/annotations/trainval.txt")
OUT_FILE = Path("data/label_map.json")

names = {}
for line in TRAINVAL_FILE.read_text().splitlines():
    image_id, class_id, _, _ = line.split()
    name = image_id.rsplit("_", 1)[0]
    class_id = int(class_id)
    assert names.setdefault(class_id, name) == name, (class_id, name)

label_map = {str(cid - 1): names[cid] for cid in sorted(names)}
assert list(label_map) == [str(i) for i in range(37)]

OUT_FILE.write_text(json.dumps(label_map, indent=2, ensure_ascii=False) + "\n")
print(len(label_map), label_map["0"], label_map["36"])