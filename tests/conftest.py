from pathlib import Path

import pytest

DATA = Path("data/oxford-iiit-pet/images")
MODULES = {"test_dataset.py", "test_manifest.py"}
TESTS = {"test_path_and_bytes_give_identical_tensor"}


def pytest_collection_modifyitems(config, items):
    if DATA.exists():
        return
    skip = pytest.mark.skip(reason="dataset not available (DVC, no remote)")
    for item in items:
        if item.path.name in MODULES or item.name in TESTS:
            item.add_marker(skip)
