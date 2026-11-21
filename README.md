# DSCI 552 — Homework 8

Chengfeng Tang · GitHub: tcf2021

[Homework notebook](notebook/Tang_Chengfeng_HW8.ipynb) contains the answers, executed result tables, ROC curves, confusion matrices, and learning curves.

| Path | Contents |
|---|---|
| `data/` | Original WDBC and banknote data supplied with the homework |
| `src/experiments.py` | Complete training, cross-validation, self-training, and acquisition procedures |
| `notebook/` | Question-by-question notebook with saved output |
| `results/breast_metrics.csv` | Training and test metrics for all four methods over 30 repetitions |
| `results/breast_cv.csv` | All penalty candidates for supervised and initial self-training fits |
| `results/self_training_trace.csv` | Each individual pseudo-label addition |
| `results/bank_learning_errors.csv` | All 9,000 learning-curve errors (50 runs × 90 sizes × 2 methods) |
| `results/bank_cv.csv` | Five-fold scores for every penalty candidate at every pool size |
| `results/bank_acquisition_order.csv` | Complete acquisition orders for both strategies in every run |

## Run

Tested with Python 3.12. From the repository root:

```sh
python -m pip install -r requirements.txt
python -m notebook notebook/Tang_Chengfeng_HW8.ipynb
```

The notebook uses relative paths and displays the saved full experiments by default. Set `RECOMPUTE=True`, restart the kernel, and run all cells to repeat the 30 WDBC and 50 banknote experiments. The helper uses four workers; the complete computation can take tens of minutes. Required result files are also generated automatically when absent.

## Evaluation

WDBC labels are benign = 0 and malignant = 1. All four methods share the same stratified test rows within each repetition. Self-training adds one pseudo-labeled observation at a time. K-means uses 30 random starts and the 30 nearest training observations to name each centroid.

Spectral clustering uses RBF gamma = 1 and `fit_predict` on each batch. Test clusters are matched to training clusters through feature-only mean RBF affinities. Test labels are never used to name or match clusters; this is a batch/transductive test procedure, explained in the notebook.

For banknotes, the 900/472 train/test split is fixed. Each repetition shares its initial ten rows between active and passive learning. Both strategies retune C with five-fold CV at every pool size and acquire without replacement.

Sources: [WDBC](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) and [Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication), UCI Machine Learning Repository.
