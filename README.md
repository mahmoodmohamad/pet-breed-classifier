# Pet Breed Classifier (MLOps Final Project, Track C)

Oxford-IIIT Pet (37 breeds). Frozen ImageNet backbone + linear head, temperature
scaling, and an abstention threshold (`decision="uncertain"`), served via FastAPI.

## What is implemented

- Fixed stratified 80/20 split of official `trainval` (seed 42): 2941 train / 739 val.
  Split files, `data/label_map.json` (37 sorted classes) and manifest builder are committed.
- Manifest (`data/manifest.csv`) built by a DVC stage; images tracked with DVC.
- Corruption suite `data/corruptions.py` (6 corruptions x 3 severities) with tests.
- Shared eval preprocessing (`petclf/preprocessing.py`): `convert("RGB")`, resize 256,
  center-crop 224, ImageNet mean/std. Used by training features, evaluation and the API.
- Training: backbone frozen, features cached once, linear head trained on features
  (CPU only machine). 5 MLflow runs across 3 backbones.
- Calibration (temperature scaling on val) and abstention threshold, logged to MLflow.
- FastAPI `/predict` (top-3 calibrated probabilities, `accept` / `uncertain`) and `/health`.
- Test that API logits match the training-path logits within 1e-4.
- GitHub Actions workflow (ruff + pytest) defined in `.github/workflows/ci.yml`;
  not yet executed on GitHub.

## Results (validation set, 739 images)

| Backbone | lr | val_acc | val_loss |
|---|---|---|---|
| resnet18 | 0.001 | 0.9175 | 0.2873 |
| resnet18 | 0.01 | 0.9066 | 0.3348 |
| mobilenet_v3_small | 0.001 | 0.8579 | 0.4441 |
| mobilenet_v3_small | 0.01 | 0.8593 | 0.4521 |
| resnet50 | 0.001 | **0.9323** | 0.2150 |

Final model: resnet50, lr 0.001.

- Temperature scaling: T = 1.052, ECE 0.0168 -> 0.0183 (did not improve ECE; the
  model was already well calibrated, and T minimizes NLL rather than ECE).
- Abstention: threshold 0.85 gives coverage 0.861 and selective accuracy 0.973.
  The threshold and temperature were fit on the same validation set, so these numbers
  are optimistic. No evaluation on the test split has been done.

## Limitations

- Backbone is frozen and there is no data augmentation, so accuracy is below full fine-tuning.
- DVC has no remote configured; CI cannot pull images, so data-dependent tests
  are skipped there (`tests/conftest.py`).
- `models/` and `data/features/` are not versioned (git-ignored).

## Not implemented

BentoML, TensorRT FP16, ONNX/OpenVINO export, Locust load tests, the 5 optimization
techniques, monitoring/drift, Airflow retraining, reliability diagram plots, corrupted
copies of the test split, final test-split evaluation.

## Run

```bash
uv sync
uv run python -m scripts.extract_features resnet50
uv run python -m scripts.train_head resnet50 0.001 30
uv run python -m scripts.calibrate resnet50 0.001
uv run python -m scripts.abstain resnet50 0.001 0.97
uv run uvicorn petclf.api:app
uv run pytest -q
```
