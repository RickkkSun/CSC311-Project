# CSC311-Project

This repository is split into two parts:

- Root directory: minimal submission version for hidden-test inference.
- `local_experiment/`: local reproducible experiment version with training, local evaluation, and plots.

For the actual project submission, only the root submission files are needed. `local_experiment/` is for local reference and should not be required by the grader.

## Submission Files

- `pred.py`: prediction entry point.
- `mlp_preprocess.py`: preprocessing and feature-construction code used by `pred.py`.
- `mlp_metadata.json`: combined preprocessing metadata for the primary and fallback MLPs.
- `mlp_weights.npz`: combined exported weights for the primary and fallback MLPs.

## Hidden-Test Usage

```bash
python3 pred.py "<test_csv_path>"
```

## Local Reproduction

Use [local_experiment/README.md](/Users/rickkk0417/CSC311-Project/local_experiment/README.md) for the full train/validation/test reproduction workflow, learning curves, confusion matrices, and local test evaluation.
