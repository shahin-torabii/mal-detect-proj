"""Stage 2: freeze the pretrained encoder and train a classification head.

Two paths:
  - "pytorch": train a LinearClassifier (nn.Module) with SGD/Adam.
  - "sklearn": extract features from the frozen encoder, then run
    GridSearchCV with either LogisticRegression (malware-analyze) or
    SVC (mal-api-2019), matching the original notebook behaviour.

Run:
    python -m src.training.train_classifier --config config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm.auto import tqdm

from ..data.dataset import load_dataset, make_dataloader, train_val_test_split
from ..models.classifier import (LinearClassifier, build_logreg_grid_search,
                                  build_svc_grid_search)
from ..models.supcon import SupConModel
from ..utils.config import get_device, load_config
from ..utils.visualization import plot_training_curves


# ──────────────── PyTorch LinearClassifier path ────────────────

def train_step(encoder, classifier, loss_fn, optimizer, dataloader, device):
    encoder.eval()
    classifier.train()
    train_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            features = encoder.get_features(x)

        out = classifier(features)
        loss = loss_fn(out, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = torch.softmax(out, dim=1)
        all_preds.append(probs.argmax(dim=1).cpu())
        all_labels.append(y.cpu())
        all_probs.append(probs.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    return (
        train_loss / len(dataloader),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average="macro"),
        roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro"),
        precision_score(all_labels, all_preds, average="macro"),
        recall_score(all_labels, all_preds, average="macro"),
    )


def eval_step(encoder, classifier, loss_fn, dataloader, device):
    encoder.eval()
    classifier.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            features = encoder.get_features(x)
            out = classifier(features)
            loss = loss_fn(out, y)
            total_loss += loss.item()

            probs = torch.softmax(out, dim=1)
            all_preds.append(probs.argmax(dim=1).cpu())
            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    return (
        total_loss / len(dataloader),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average="macro"),
        roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro"),
        precision_score(all_labels, all_preds, average="macro"),
        recall_score(all_labels, all_preds, average="macro"),
    )


def train_pytorch(encoder, classifier, loss_fn, optimizer, train_loader, val_loader, epochs, device):
    history = {k: [] for k in [
        "train_loss", "train_acc", "train_f1", "train_auc", "train_prec", "train_rec",
        "val_loss", "val_acc", "val_f1", "val_auc", "val_prec", "val_rec",
    ]}

    for epoch in tqdm(range(epochs)):
        tr = train_step(encoder, classifier, loss_fn, optimizer, train_loader, device)
        va = eval_step(encoder, classifier, loss_fn, val_loader, device)

        for key, value in zip(
            ["train_loss", "train_acc", "train_f1", "train_auc", "train_prec", "train_rec"], tr
        ):
            history[key].append(value)
        for key, value in zip(
            ["val_loss", "val_acc", "val_f1", "val_auc", "val_prec", "val_rec"], va
        ):
            history[key].append(value)

        print(
            f"Epoch: {epoch + 1} | train_loss: {tr[0]:.4f} | val_loss: {va[0]:.4f} | "
            f"val_acc: {va[1]:.4f} | val_f1: {va[2]:.4f} | val_auc: {va[3]:.4f}"
        )

    return history


# ──────────────── sklearn GridSearchCV path ────────────────

def extract_features(encoder, dataloader, device):
    """Run frozen encoder over a DataLoader → numpy arrays."""
    encoder.eval()
    X_list, Y_list = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            features = encoder.get_features(x)
            X_list.append(features.cpu().numpy())
            Y_list.append(y.cpu().numpy())

    return np.vstack(X_list), np.concatenate(Y_list)


def train_sklearn(encoder, train_loader, val_loader, device, classifier_type):
    """Extract features, then run GridSearchCV with the requested sklearn classifier."""
    X_train, y_train = extract_features(encoder, train_loader, device)
    X_val, y_val = extract_features(encoder, val_loader, device)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    if classifier_type == "logreg":
        grid = build_logreg_grid_search()
    elif classifier_type == "svc":
        grid = build_svc_grid_search()
    else:
        raise ValueError(f"Unknown sklearn classifier type: {classifier_type}")

    grid.fit(X_train, y_train)

    cv_results = grid.cv_results_
    best_idx = grid.best_index_

    print("======== TRAIN (CV) RESULTS ========")
    print(f"Accuracy (CV):          {cv_results['mean_test_accuracy'][best_idx]:.4f}")
    print(f"F1-Macro (CV):          {cv_results['mean_test_f1_macro'][best_idx]:.4f}")
    print(f"Precision-Macro (CV):   {cv_results['mean_test_precision_macro'][best_idx]:.4f}")
    print(f"Recall-Macro (CV):      {cv_results['mean_test_recall_macro'][best_idx]:.4f}")
    print(f"AUC (OvR) (CV):         {cv_results['mean_test_auc_ovr'][best_idx]:.4f}")
    print("Best Params:", grid.best_params_)
    print("====================================\n")

    y_pred = grid.predict(X_val)
    y_prob = grid.predict_proba(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1_macro": f1_score(y_val, y_pred, average="macro"),
        "f1_weighted": f1_score(y_val, y_pred, average="weighted"),
        "precision_macro": precision_score(y_val, y_pred, average="macro"),
        "recall_macro": recall_score(y_val, y_pred, average="macro"),
        "auc_ovr": roc_auc_score(y_val, y_prob, multi_class="ovr", average="macro"),
    }

    print("=========== VALIDATION RESULTS ===========")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}")
    print("==========================================\n")

    return grid, scaler, metrics


# ───────────────────── main ─────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a classifier on frozen SupCon features.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--method", default="auto",
                        choices=["auto", "pytorch", "sklearn"],
                        help="'auto' picks sklearn for known datasets, pytorch otherwise")
    parser.add_argument("--classifier-type", default=None,
                        choices=["logreg", "svc", "pytorch"],
                        help="Override the classifier type from config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    torch.manual_seed(cfg.get("seed", 42))

    dataset_name = cfg.get("dataset", "malware-analyze")
    features, labels, num_classes = load_dataset(dataset_name, cfg)

    # classifier_type: config → CLI override → default
    classifier_type = args.classifier_type or cfg.get("classifier_type", "logreg")
    print(f"Dataset: {dataset_name}  |  Classifier: {classifier_type}")

    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
        features, labels,
        test_size=cfg["data"]["test_size"], val_size=cfg["data"]["val_size"],
        random_state=cfg["data"]["random_state"],
    )

    input_size = features.shape[1] if hasattr(features, 'shape') else len(features[0])
    encoder = SupConModel(
        input_size=input_size,
        hidden_dim=cfg["model"]["hidden_dim"],
        projection_dim=cfg["model"]["projection_dim"],
    ).to(device)
    encoder.load_state_dict(torch.load(cfg["paths"]["encoder_checkpoint"], map_location=device))
    for p in encoder.parameters():
        p.requires_grad = False

    clf_cfg = cfg["training"]["classifier"]

    # Determine which method to use
    method = args.method
    if method == "auto":
        method = "sklearn" if classifier_type in ("logreg", "svc") else "pytorch"

    if method == "sklearn":
        train_loader = make_dataloader(x_train, y_train, batch_size=clf_cfg["batch_size"])
        val_loader = make_dataloader(x_val, y_val, batch_size=clf_cfg["batch_size"])

        grid, scaler, metrics = train_sklearn(encoder, train_loader, val_loader, device, classifier_type)

        # Save the sklearn model
        import joblib
        ckpt_path = Path(cfg["paths"]["classifier_checkpoint"])
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"grid": grid, "scaler": scaler}, ckpt_path.with_suffix(".joblib"))
        print(f"Saved sklearn classifier to {ckpt_path.with_suffix('.joblib')}")

    else:
        train_loader = make_dataloader(x_train, y_train, batch_size=clf_cfg["batch_size"])
        val_loader = make_dataloader(x_val, y_val, batch_size=clf_cfg["batch_size"])

        classifier = LinearClassifier(
            input_dim=cfg["model"]["hidden_dim"], num_classes=num_classes
        ).to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=clf_cfg["learning_rate"])

        history = train_pytorch(
            encoder, classifier, loss_fn, optimizer, train_loader, val_loader,
            clf_cfg["epochs"], device,
        )

        ckpt_path = Path(cfg["paths"]["classifier_checkpoint"])
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), ckpt_path)
        print(f"Saved classifier checkpoint to {ckpt_path}")

        plot_training_curves(
            clf_cfg["epochs"],
            history["train_loss"], history["val_loss"],
            history["train_acc"], history["val_acc"],
            history["train_f1"], history["val_f1"],
            history["train_auc"], history["val_auc"],
            history["train_prec"], history["val_prec"],
            history["train_rec"], history["val_rec"],
            cfg["paths"]["figures_dir"],
        )


if __name__ == "__main__":
    main()
