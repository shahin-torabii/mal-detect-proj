# Malware Detection with Supervised Contrastive Learning on API Call Sequences

## Overview

This repository contains the official implementation of the paper:

> **"Windows Malware Detection Based on API Calls Using Contrastive Learning"** (not yet published)

The work explores **Supervised Contrastive Learning (SupCon)** for classifying malware families from dynamic API-call sequences. A two-stage pipeline is used:

1. **Pretraining (Stage 1)** — an LSTM encoder is trained with a supervised contrastive loss using two augmented views (original + Gaussian noise) of each sample, pulling same-class embeddings closer together in the representation space.
2. **Classification (Stage 2)** — the encoder is frozen and a classifier head is trained on its features. Three classifier options are provided: LogisticRegression, SVC (both with GridSearchCV), and a PyTorch LinearClassifier.

### Paper status

The paper is **not yet published**. The full study evaluates **4 datasets** and **5 downstream classifiers** on top of the SupCon-pretrained encoder. This repository contains a **subset** of that work:

- **2 datasets out of 4** are included (the other two, and the additional preprocessing variants explored for them, are omitted from this public release — see [Datasets](#datasets) below).
- **3 classifiers out of 5** are included: Logistic Regression, SVM (SVC), and a PyTorch Linear classifier. **KNN** and **Random Forest** are evaluated in the paper but their code is **not included** here.
- **ISAO (Iterative Sequential Augmentation Oversampling)**, the paper's proposed oversampling technique, is only relevant to — and only applied to — the *Malware Analysis Datasets: API Call Sequences* dataset, since that is the only one of the four datasets with severe class imbalance (~40:1 malware-to-benign ratio). The other datasets are reasonably balanced and are used as-is, with no oversampling step. This repo includes the **imbalanced (pre-ISAO)** pipeline for that dataset; results **after** ISAO balancing are reported below from the paper for reference, but the ISAO implementation itself is not included in this public code.

---

## Project Structure

```
malware-detection/
│
├── README.md                  # this file
├── requirements.txt           # Python dependencies
├── LICENSE
├── config.yaml                # all paths & hyperparameters (no hard-coded paths)
├── main.py                    # single CLI entry point
│
├── data/
│   ├── raw/                   # place dataset files here
│   └── processed/             # cached TF-IDF embeddings
│
├── src/
│   ├── models/
│   │   ├── encoder.py              # LSTMEncoder
│   │   ├── projection_head.py      # MLP projection head (used only in SupCon)
│   │   ├── supcon.py               # SupConModel + SupConLoss
│   │   └── classifier.py           # LinearClassifier + sklearn grid-search builders
│   │
│   ├── data/
│   │   └── dataset.py              # ApiDataSet, data loaders, dataset dispatch
│   │
│   ├── preprocessing/
│   │   └── tfidf.py                # TF-IDF fitting / caching (for mal-api-2019)
│   │
│   ├── training/
│   │   ├── train_supcon.py         # Stage 1: contrastive pretraining
│   │   └── train_classifier.py     # Stage 2: classifier training
│   │
│   ├── evaluation/
│   │   └── evaluate.py             # test-set metrics + confusion matrix
│   │
│   └── utils/
│       ├── config.py               # YAML config loader, device helper
│       └── visualization.py        # t-SNE, loss curves, confusion matrix plots
│
├── notebooks/
│   ├── malware-analyze dataset/    # original exploratory notebooks
│   │   ├── classifier_v2.1.ipynb           # SVC + GridSearchCV
│   │   ├── classifier_v2.4.ipynb           # LogisticRegression + GridSearchCV
│   │   └── malware_supcons_v2_imbalanced.ipynb  # SupCon pretraining
│   │
│   └── mal-api-2019 dataset/
│       ├── classifier_v2.ipynb             # PyTorch LinearClassifier
│       ├── classifier_v2.1.ipynb           # SVC + GridSearchCV
│       └── malware_supcons_v2.ipynb        # SupCon pretraining
│
├── checkpoints/                # saved model weights (gitignored)
├── results/
│   ├── figures/                # saved plots (gitignored)
│   └── tables/                 # saved metrics CSV (gitignored)
│
└── config.yaml                 # single source of truth for paths & hyperparameters
```

---

## Key Features

| Feature | Description |
|---|---|
| **Two datasets included** (of 4 in the paper) | Malware Analysis Datasets (Kaggle) — CSV of API call counts; mal-api-2019 — raw text → TF-IDF |
| **SupCon pretraining** | LSTM encoder trained with supervised contrastive loss (Khosla et al., 2020) |
| **3 classifier options included** (of 5 in the paper) | LogisticRegression, SVC (both sklearn + GridSearchCV), PyTorch LinearClassifier — KNN and Random Forest are not included |
| **Config-driven** | `config.yaml` controls dataset, classifier type, paths, and all hyperparameters |
| **CLI pipeline** | `main.py` with `pretrain`, `train-classifier`, `evaluate` stages |
| **No hard-coded paths** | Dataset paths and all settings live in `config.yaml` |
| **Visualization** | t-SNE embedding plots, training curves, confusion matrices |
| **Reproducible** | Configurable random seed, all results saved to `results/` |

---

## Datasets

### 1. Malware Analysis Datasets — API Call Sequences (Kaggle)

- **Config name:** `malware-analyze`
- **Format:** CSV with 100 API-call-count columns (`t_0` … `t_99`) plus `malware` label (0 = Benign, 1 = Malware)
- **Size:** ~43,800 samples
- **Class distribution:** Highly imbalanced (≈97.5% malware, 2.5% benign)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/ang3loliveira/malware-analysis-datasets-api-call-sequences)

### 2. mal-api-2019

- **Config name:** `mal-api-2019`
- **Format:** Raw text file of comma-separated API call sequences + separate `labels.csv` with malware family names
- **Classes (8):** Adware, Backdoor, Downloader, Dropper, Spyware, Trojan, Virus, Worms
- **Size:** ~7,100 samples
- **Preprocessing:** TF-IDF vectorization with n-gram range (3,5) → sparse matrix (~440K features)

### Datasets not included in this repository

The paper evaluates **two additional datasets** beyond the two above — one used for malware-family classification and one used for malware-vs-benign detection. Their identities and raw data are not included in this repository, but for reference, the best results achieved by the paper's SupCon framework on them are:

| | Best Accuracy | Best Macro-F1 | Best AUC | Best classifier |
|---|---|---|---|---|
| Additional dataset A (family classification) | 96.31% | 93.33% | 99.72% | Logistic Regression *(included in this repo)* |
| Additional dataset B (binary detection) | 100.0% | 100.0% | 100.0% | Random Forest *(not included in this repo)* |

Both used (3,5)-gram TF-IDF embeddings on top of the SupCon encoder.

---

## Classifiers

The paper evaluates **5 classifiers** on top of the frozen SupCon encoder embeddings: a single-layer Linear classifier, SVM, KNN, Random Forest, and Logistic Regression. This repository implements **3 of the 5**, available for **any** included dataset via `classifier_type` in `config.yaml`:

| Classifier | Config Value | Implementation | Source Notebook |
|---|---|---|---|
| LogisticRegression + GridSearchCV | `logreg` | sklearn (`solver`, `penalty` grid) | `classifier_v2.4` (malware-analyze) |
| SVC + GridSearchCV | `svc` | sklearn (`C`, `kernel` grid, OvR, balanced weights) | `classifier_v2.1` (both datasets) |
| LinearClassifier (PyTorch) | `pytorch` | Single `nn.Linear` layer trained with Adam | `classifier_v2` (mal-api-2019) |

**KNN** and **Random Forest** are evaluated in the paper (and sometimes outperform the classifiers above — see [Results](#results)) but their code is **not included** in this public repository.

**ISAO (Iterative Sequential Augmentation Oversampling)** — the paper's proposed oversampling technique for the imbalanced *Malware Analysis Datasets: API Call Sequences* dataset — is likewise **not included** here.

---

## Results

Results reported directly below each dataset heading are from **this repository's** notebook executions (the classifiers listed under [Classifiers](#classifiers)). Each subsection also lists the **best result reported in the paper** for that dataset, which may come from a classifier not included here.

### mal-api-2019 (8 malware families)

#### SVC + GridSearchCV (this repo)

| Metric | CV (mean) | Test |
|---|---|---|
| Accuracy | 0.7273 | 0.7454 |
| F1-Macro | 0.7359 | 0.7457 |
| F1-Weighted | — | 0.7491 |
| Precision-Macro | 0.7396 | 0.7475 |
| Recall-Macro | 0.7342 | 0.7481 |
| AUC (OvR) | 0.9395 | 0.9534 |

**Best params:** `C=1.5, kernel='rbf'`

#### PyTorch LinearClassifier (this repo, 100 epochs)

| Metric | Validation (final epoch) |
|---|---|
| Accuracy | 0.9641 |
| F1-Macro | 0.9641 |
| AUC (OvR) | 0.9970 |
| Precision-Macro | 0.9658 |
| Recall-Macro | 0.9635 |

#### Best result reported in the paper

Using **Logistic Regression** (included in this repo, `classifier_type: "logreg"`) with **(3,5)-gram TF-IDF** embeddings:

| Metric | Value |
|---|---|
| Accuracy | 99.02% |
| F1-Macro | 99.03% |
| F1-Weighted | 99.01% |
| Precision-Macro | 99.05% |
| Recall-Macro | 99.03% |
| AUC | 99.92% |

### Malware Analysis Datasets: API Call Sequences (binary, imbalanced)

#### LogisticRegression + GridSearchCV (this repo)

| Metric | CV (mean) | Test |
|---|---|---|
| Accuracy | 0.9754 | 0.9739 |
| F1-Macro | 0.5006 | 0.4934 |
| F1-Weighted | — | 0.9614 |
| Precision-Macro | 0.6058 | 0.4871 |
| Recall-Macro | 0.5033 | 0.4998 |
| AUC (OvR) | 0.5023 | 0.5024 |

**Best params:** `penalty='l2', solver='saga'`

> The high accuracy but low macro-F1 reflects the extreme class imbalance in this dataset (≈2.5% benign samples).

#### Best result reported in the paper — before balancing (imbalanced, same setting as this repo)

Using **Random Forest** (not included in this repo) — only marginally ahead of the Logistic Regression numbers above:

| Metric | Value |
|---|---|
| Accuracy | 97.42% |
| F1-Macro | 49.35% |
| F1-Weighted | 96.15% |
| AUC | 50.00% |

#### Best result reported in the paper — after ISAO balancing

ISAO (not included in this repo) is applied only to this dataset to correct its ~40:1 class imbalance. The best classifier afterward is **Logistic Regression** (included in this repo, though the ISAO oversampling step itself is not):

| Metric | Value |
|---|---|
| Accuracy | 98.80% |
| F1-Macro | 86.79% |
| F1-Weighted | 98.74% |
| Precision-Macro | 91.28% |
| Recall-Macro | 83.23% |
| AUC | 98.54% |

Balancing with ISAO lifts macro-F1 from ~49% to ~87% on this dataset, at the cost of a small change in raw accuracy — a direct illustration of why accuracy alone is misleading under severe class imbalance.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place your dataset files under `data/raw/` and update paths in `config.yaml`.

---

## Usage

### Configure

Edit `config.yaml`:

```yaml
dataset: "malware-analyze"        # or "mal-api-2019"
classifier_type: "svc"            # "logreg" | "svc" | "pytorch"
```

### Run Pipeline

```bash
# Stage 0 (optional): precompute TF-IDF features (only needed for mal-api-2019)
python main.py preprocess

# Stage 1: pretrain the encoder with SupCon loss
python main.py pretrain

# Stage 2: train classifier on frozen encoder features
python main.py train-classifier

# Stage 3: evaluate on held-out test set
python main.py evaluate --class-names Adware Backdoor Downloader Dropper Spyware Trojan Virus Worms
```

### CLI Overrides

```bash
python main.py train-classifier --classifier-type svc --method sklearn
python main.py evaluate --class-names Benign Malware
python main.py train-classifier --config experiments/other_config.yaml
```

---

## Workflow Summary

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌────────────┐
│ Load dataset │────▶│ SupCon Pretrain  │────▶│ Train Classifier│────▶│  Evaluate   │
│ (CSV/TF-IDF) │     │ (LSTM + SupLoss) │     │ (sklearn/PyTorch)│     │ (test set)  │
└─────────────┘     └──────────────────┘     └─────────────────┘     └────────────┘
       │                      │                       │                      │
       │                      ▼                       │                      │
       │            checkpoints/encoder.pth           │               results/tables/
       │                                              │               results/figures/
       │                                    checkpoints/classifier.pth
       │                                    (or .joblib for sklearn)
```

---

## Notes

- The `notebooks/` folder contains the original exploratory notebooks used during development. All reusable logic has been migrated into `src/`.
- `checkpoints/`, `results/figures/`, and `results/tables/` are git-ignored except for `.gitkeep` files.
- This repository is a partial release: **2 of the 4 datasets**, **3 of the 5 classifiers**, and no ISAO oversampling code. The full paper — covering all 4 datasets, all 5 classifiers, and the ISAO method — will be published in a future release.
