from collections import Counter
from pathlib import Path

ann = Path("data/oxford-iiit-pet/annotations/trainval.txt")
class_of = {}
for line in ann.read_text().splitlines():
    image_id, class_id, _, _ = line.split()
    class_of[image_id] = int(class_id)

train = Path("data/splits/train.txt").read_text().split()
val = Path("data/splits/val.txt").read_text().split()

print("overlap:", len(set(train) & set(val)))
print("unique total:", len(set(train) | set(val)))
print("all ids known:", set(train + val) <= set(class_of))

train_counts = Counter(class_of[i] for i in train)
print("classes in train:", len(train_counts))
print("min train per class:", min(train_counts.values()))