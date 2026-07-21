"""PyTorch Dataset(s) and loading helpers.

Two dataset modes:
  - ApiDataSet(contrastive=True):  returns two augmented "views" of the
    same sample (original + Gaussian noise), used for SupCon pretraining.
  - ApiDataSet(contrastive=False): returns a single view, used for
    training/evaluating the downstream classifier.

Two dataset variants:
  - "malware-analyze" (CSV: API call counts, binary label).
  - "mal-api-2019"   (TF-IDF sparse matrix from raw text, 8 class names).
"""
from pathlib import Path

import pandas as pd
import torch
from scipy import sparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ..preprocessing.tfidf import load_embedding


# ──────────────────────────── Dataset ────────────────────────────

class ApiDataSet(Dataset):
    def __init__(self, data, labels, contrastive: bool = False, noise_std: float = 0.05):
        super().__init__()
        self.data = data
        self.labels = labels
        self.contrastive = contrastive
        self.noise_std = noise_std
        self.is_sparse = sparse.issparse(data)

    def __len__(self):
        return self.data.shape[0] if self.is_sparse else len(self.data)

    def _row_as_tensor(self, idx):
        row = self.data[idx].toarray() if self.is_sparse else self.data[idx]
        return torch.tensor(row, dtype=torch.float32).flatten()

    def __getitem__(self, idx):
        x = self._row_as_tensor(idx)
        y = self.labels[idx]

        if not self.contrastive:
            return x.unsqueeze(0), y

        x_noisy = x + torch.normal(0.0, self.noise_std, size=x.shape)
        view1 = x.unsqueeze(0)
        view2 = x_noisy.unsqueeze(0)
        views = torch.stack([view1, view2], dim=0)
        return views, y


# ─────────────────────── Data loaders ───────────────────────

def load_csv_dataset(csv_path: str, label_column: str, drop_columns: list):
    """Malware-analyze CSV: numeric API-call-count columns + integer label."""
    df = pd.read_csv(csv_path)
    labels = torch.tensor(df[label_column].values, dtype=torch.long)
    features = df.drop(columns=drop_columns).values.tolist()
    return features, labels


def encode_labels_from_csv(labels_path: str):
    """mal-api-2019: read labels.csv (class name per row), encode → tensor."""
    df = pd.read_csv(labels_path, header=None)
    class_names = df[0].unique()
    name_to_idx = {name: i for i, name in enumerate(sorted(class_names))}
    labels = torch.tensor(df[0].map(name_to_idx).values, dtype=torch.long)
    return labels, name_to_idx


def load_tfidf_dataset(embedding_path: str, labels_path: str):
    """mal-api-2019: TF-IDF sparse matrix + labels.csv with class names."""
    features = load_embedding(embedding_path)
    labels, _ = encode_labels_from_csv(labels_path)
    return features, labels


def load_dataset(dataset_name: str, cfg: dict):
    """Dispatch to the correct loader based on the dataset name.

    Returns (features, labels, num_classes).
    """
    data_cfg = cfg["data"][dataset_name]

    if dataset_name == "malware-analyze":
        features, labels = load_csv_dataset(
            data_cfg["raw_csv_path"],
            data_cfg["label_column"],
            data_cfg["drop_columns"],
        )
    elif dataset_name == "mal-api-2019":
        emb_path = Path(data_cfg["tfidf_embedding_path"])
        if not emb_path.exists():
            print("TF-IDF embedding not found – running preprocessing …")
            from ..preprocessing.tfidf import fit_tfidf, load_raw_sequences, save_embedding
            sequences = load_raw_sequences(data_cfg["raw_text_path"])
            embedding, _ = fit_tfidf(sequences, **cfg.get("tfidf", {}))
            save_embedding(embedding, str(emb_path))
        features, labels = load_tfidf_dataset(
            str(emb_path), data_cfg["labels_path"],
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return features, labels, data_cfg["num_classes"]


# ────────────────────── Splitting & DataLoaders ─────────────────────

def train_val_test_split(features, labels, test_size=0.2, val_size=0.5, random_state=42):
    """Split into train / val / test.  Default: 80/10/10."""
    x_train, x_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, shuffle=True
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=val_size, random_state=random_state, shuffle=True
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def make_dataloader(features, labels, batch_size=32, shuffle=True,
                     contrastive=False, noise_std=0.05):
    dataset = ApiDataSet(features, labels, contrastive=contrastive, noise_std=noise_std)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
