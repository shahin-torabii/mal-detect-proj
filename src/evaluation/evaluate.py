"""Final evaluation of the pretrained encoder + classifier on the
held-out test set. Supports both PyTorch and sklearn classifiers,
and both dataset variants.

Run:
    python -m src.evaluation.evaluate --config config.yaml
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)

from ..data.dataset import load_dataset, make_dataloader, train_val_test_split
from ..models.classifier import LinearClassifier
from ..models.supcon import SupConModel
from ..utils.config import get_device, load_config
from ..utils.visualization import plot_confusion_matrix


def final_evaluate(encoder, classifier, dataloader, device):
    encoder.eval()
    classifier.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            features = encoder.get_features(x)
            out = classifier(features)
            probs = torch.softmax(out, dim=1)

            all_preds.append(probs.argmax(dim=1).cpu())
            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        "precision_macro": precision_score(all_labels, all_preds, average="macro"),
        "recall_macro": recall_score(all_labels, all_preds, average="macro"),
        "auc_macro": roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro"),
    }
    cm = confusion_matrix(all_labels, all_preds)
    return metrics, cm


def final_evaluate_sklearn(encoder, grid, scaler, dataloader, device):
    """Evaluate using sklearn GridSearchCV model (no backprop)."""
    from ..training.train_classifier import extract_features

    X_test, y_test = extract_features(encoder, dataloader, device)
    X_test = scaler.transform(X_test)

    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "auc_macro": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def save_metrics_csv(metrics: dict, tables_dir: str, filename: str = "test_metrics.csv"):
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    path = Path(tables_dir) / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
    print(f"Saved metrics to {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate encoder + classifier on the test set.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--class-names", nargs="*", default=None,
        help="Optional list of class names for the confusion matrix labels.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))

    dataset_name = cfg.get("dataset", "malware-analyze")
    features, labels, num_classes = load_dataset(dataset_name, cfg)

    _, _, x_test, _, _, y_test = train_val_test_split(
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

    test_loader = make_dataloader(
        x_test, y_test, batch_size=cfg["training"]["classifier"]["batch_size"], shuffle=False
    )

    # Try sklearn path first; fall back to PyTorch
    sklearn_path = Path(cfg["paths"]["classifier_checkpoint"]).with_suffix(".joblib")
    if sklearn_path.exists():
        import joblib

        state = joblib.load(sklearn_path)
        grid, scaler = state["grid"], state["scaler"]
        metrics, cm = final_evaluate_sklearn(encoder, grid, scaler, test_loader, device)
        print("Evaluated with sklearn classifier.")
    else:
        classifier = LinearClassifier(
            input_dim=cfg["model"]["hidden_dim"], num_classes=num_classes
        ).to(device)
        classifier.load_state_dict(
            torch.load(cfg["paths"]["classifier_checkpoint"], map_location=device)
        )
        metrics, cm = final_evaluate(encoder, classifier, test_loader, device)
        print("Evaluated with PyTorch classifier.")

    print("\nTest set results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("\nConfusion matrix:")
    print(cm)

    save_metrics_csv(metrics, cfg["paths"]["tables_dir"])

    class_names = args.class_names or [str(i) for i in range(num_classes)]
    plot_confusion_matrix(cm, class_names, cfg["paths"]["figures_dir"])


if __name__ == "__main__":
    main()
