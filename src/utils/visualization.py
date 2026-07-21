"""Plotting helpers: embedding t-SNE, training curves, confusion matrix.

All functions save the figure to `results/figures/` (as well as
showing it, when run interactively) instead of only calling plt.show(),
so results survive headless / notebook-less runs.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def _save(fig, figures_dir: str, filename: str):
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(figures_dir) / filename, bbox_inches="tight", dpi=150)


def visualize_embed(features, labels, title: str, figures_dir: str = "results/figures"):
    """3D t-SNE plot of encoder embeddings, colored by class."""
    tsne = TSNE(n_components=3, init="random")
    reduced = tsne.fit_transform(features)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=labels, cmap="viridis")

    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Class ID")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    ax.set_title(title)

    _save(fig, figures_dir, f"{title.replace(' ', '_')}.png")
    plt.show()


def plot_supcon_loss(losses, figures_dir: str = "results/figures"):
    fig = plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.title("SupCon Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    _save(fig, figures_dir, "supcon_training_loss.png")
    plt.show()


def plot_training_curves(epochs, train_loss, val_loss, train_acc, val_acc,
                          train_f1, val_f1, train_auc, val_auc,
                          train_prec, val_prec, train_rec, val_rec,
                          figures_dir: str = "results/figures"):
    fig, axes = plt.subplots(3, 2, figsize=(18, 10))
    x = range(1, epochs + 1)

    def _pair(ax, train_vals, val_vals, ylabel, title):
        ax.plot(x, train_vals, color="#1f77b4", label=f"train {ylabel.lower()}", linewidth=2)
        ax.plot(x, val_vals, color="#ff7f0e", label=f"validation {ylabel.lower()}", linewidth=2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Epoch")
        ax.legend()

    _pair(axes[0][0], train_loss, val_loss, "Loss", "Train/Validation Loss")
    _pair(axes[0][1], train_acc, val_acc, "Accuracy", "Validation Accuracy")
    _pair(axes[1][0], train_f1, val_f1, "F1 score", "Validation F1 score")
    _pair(axes[1][1], train_auc, val_auc, "AUC", "Validation AUC")
    _pair(axes[2][0], train_prec, val_prec, "Precision", "Precision")
    _pair(axes[2][1], train_rec, val_rec, "Recall", "Recall")

    plt.tight_layout()
    _save(fig, figures_dir, "training_curves.png")
    plt.show()


def plot_confusion_matrix(cm, class_names, figures_dir: str = "results/figures"):
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    _save(fig, figures_dir, "confusion_matrix.png")
    plt.show()
