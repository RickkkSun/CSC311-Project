# Local Experiment

Self-contained local reproduction directory for the CSC311 MLP model.

## Main Files

- `shared_preprocess_v2.py`: shared first-pass preprocessing and strict grouped split logic.
- `mlp_preprocess.py`: MLP-specific second-pass preprocessing and feature construction.
- `train_mlp.py`: strict grouped train/validation/test training and hyperparameter search.
- `pred.py`: local inference script using exported artifacts.
- `analyze_overfit.py`: learning-curve and overfitting diagnostics.
- `plot_final_split_comparison.py`: final train/validation/test comparison figure.
- `plot_confusion_matrices.py`: outer-validation and local-test confusion matrices.
- `artifacts/`: exported model weights, metadata, and training summary.
- `analysis/`: generated plots and CSV summaries.

## Reproduce

Train and export artifacts:

```bash
python3 train_mlp.py --search
```

Generate learning-curve and overfitting plots:

```bash
python3 analyze_overfit.py
python3 plot_final_split_comparison.py
python3 plot_confusion_matrices.py
```

Run local inference:

```bash
python3 pred.py "/Users/rickkk0417/Downloads/training_data_202601 (1).csv"
```
