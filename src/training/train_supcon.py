"""Stage 1: pretrain the LSTM encoder with a Supervised Contrastive
(SupCon) loss, using two augmented views (original + Gaussian noise)
of each sample.

Supports both datasets (malware-analyze CSV / mal-api-2019 TF-IDF).

Run:
    python -m src.training.train_supcon --config config.yaml
"""
import argparse
from pathlib import Path
from timeit import default_timer as timer

import torch
from tqdm.auto import tqdm

from ..data.dataset import load_dataset, make_dataloader
from ..models.supcon import SupConLoss, SupConModel
from ..utils.config import get_device, load_config
from ..utils.visualization import plot_supcon_loss, visualize_embed


def train_step(model, loss_fn, dataloader, optimizer, device):
    model.train()
    train_loss = 0.0
    all_embeds, all_labels = [], []

    for views, label in dataloader:
        views, label = views.to(device), label.to(device)

        view1 = views[:, 0, :, :]
        view2 = views[:, 1, :, :]

        out1 = model(view1)
        out2 = model(view2)
        features = torch.stack([out1, out2], dim=1)

        loss = loss_fn(features, label)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_embeds.extend([out1.detach(), out2.detach()])
        all_labels.extend([label, label])

    all_embeds = torch.cat(all_embeds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    train_loss /= len(dataloader)
    return train_loss, all_embeds, all_labels


def train(model, loss_fn, optimizer, dataloader, epochs, device, figures_dir):
    losses = []
    embeds, labels = None, None

    for epoch in tqdm(range(epochs)):
        train_loss, embeds, labels = train_step(model, loss_fn, dataloader, optimizer, device)
        losses.append(train_loss)
        print(f"Epoch: {epoch + 1} | train_loss: {train_loss:.4f}")

    visualize_embed(embeds, labels, "Embeddings after SupCon pretraining", figures_dir)
    return losses


def main():
    parser = argparse.ArgumentParser(description="Pretrain the encoder with SupCon loss.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    torch.manual_seed(cfg.get("seed", 42))

    dataset_name = cfg.get("dataset", "malware-analyze")
    features, labels, num_classes = load_dataset(dataset_name, cfg)

    sc_cfg = cfg["training"]["supcon"]
    dataloader = make_dataloader(
        features, labels, batch_size=sc_cfg["batch_size"], shuffle=True,
        contrastive=True, noise_std=sc_cfg["noise_std"],
    )

    input_size = features.shape[1] if hasattr(features, 'shape') else len(features[0])
    model = SupConModel(
        input_size=input_size,
        hidden_dim=cfg["model"]["hidden_dim"],
        projection_dim=cfg["model"]["projection_dim"],
    ).to(device)

    loss_fn = SupConLoss(temperature=sc_cfg["temperature"])
    optimizer = torch.optim.Adam(model.parameters(), lr=sc_cfg["learning_rate"])

    start = timer()
    losses = train(
        model, loss_fn, optimizer, dataloader, sc_cfg["epochs"], device,
        cfg["paths"]["figures_dir"],
    )
    print(f"Pretraining time: {timer() - start:.1f}s")

    ckpt_path = Path(cfg["paths"]["encoder_checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved encoder checkpoint to {ckpt_path}")

    plot_supcon_loss(losses, cfg["paths"]["figures_dir"])


if __name__ == "__main__":
    main()
