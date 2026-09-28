import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

MANIFEST = Path("data/manifest.csv")
LABEL_MAP = Path("data/label_map.json")


def load_rows() -> list[dict]:
    with MANIFEST.open() as f:
        return list(csv.DictReader(f))


def test_every_path_exists_and_opens():
    for row in load_rows():
        with Image.open(row["path"]) as img:
            img.verify()


def test_no_image_id_in_multiple_splits():
    splits = defaultdict(set)
    for row in load_rows():
        splits[row["image_id"]].add(row["split"])
    assert all(len(s) == 1 for s in splits.values())


def test_label_map_has_37_sorted_classes():
    label_map = json.loads(LABEL_MAP.read_text())
    assert list(label_map) == [str(i) for i in range(37)]


def test_manifest_breeds_match_label_map():
    label_map = json.loads(LABEL_MAP.read_text())
    for row in load_rows():
        assert label_map[row["class_index"]] == row["breed"]


def test_at_least_50_train_images_per_class():
    counts = Counter(r["class_index"] for r in load_rows() if r["split"] == "train")
    assert len(counts) == 37
    assert min(counts.values()) >= 50


def test_no_image_smaller_than_32():
    for row in load_rows():
        assert int(row["width"]) >= 32
        assert int(row["height"]) >= 32