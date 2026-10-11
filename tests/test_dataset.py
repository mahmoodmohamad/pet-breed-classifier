import json
from pathlib import Path

from petclf.dataset import PetDataset

LABELS = json.loads(Path("data/label_map.json").read_text())


def test_split_sizes():
    assert len(PetDataset("train")) == 2941
    assert len(PetDataset("val")) == 739


def test_labels_in_range_and_match_label_map():
    ds = PetDataset("val")
    for row in ds.rows:
        idx = int(row["class_index"])
        assert 0 <= idx <= 36
        assert LABELS[str(idx)] == row["breed"]


def test_item_shape():
    x, y = PetDataset("val")[0]
    assert x.shape == (3, 224, 224)
    assert isinstance(y, int)
