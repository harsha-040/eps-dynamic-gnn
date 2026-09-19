"""Unified training entry point for all four models.

Usage:
    python -m src.train --config configs/dynamic_gnn.yaml
    python -m src.train --config configs/static_gnn.yaml
    python -m src.train --config configs/cnn_baseline.yaml
    python -m src.train --config configs/lstm_baseline.yaml
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .data.dataset import EPSWindowDataset
from .models.cnn_baseline import CNNBaseline
from .models.dynamic_gnn import DynamicGNN
from .models.lstm_baseline import LSTMBaseline
from .models.static_gnn import StaticGNN
from .simulation import constants as C

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(name, params, n_nodes, n_features, n_classes):
    if name == "dynamic_gnn":
        return DynamicGNN(n_nodes, n_features, n_classes, **params)
    if name == "static_gnn":
        return StaticGNN(n_nodes, n_features, n_classes, C.STATIC_EDGE_INDEX, **params)
    if name == "cnn_baseline":
        return CNNBaseline(n_nodes, n_features, n_classes, **params)
    if name == "lstm_baseline":
        return LSTMBaseline(n_nodes, n_features, n_classes, **params)
    raise ValueError(f"unknown model {name}")


def apply_augmentation(x: torch.Tensor, aug_cfg: dict) -> torch.Tensor:
    """Corrupt a batch of NORMALIZED input the same way `src.robustness`
    corrupts raw telemetry: since normalized channels have ~unit std on the
    train split, additive noise ~ N(0, level^2) directly on the normalized
    tensor is equivalent to adding `level * raw_std` noise pre-normalization,
    and zeroing an entry is equivalent to replacing it with the channel mean
    (which is 0 post-normalization). Severity is randomized per batch so
    training sees a spread of corruption levels, including clean batches.
    """
    if not aug_cfg or not aug_cfg.get("enabled", False):
        return x
    if torch.rand(1).item() > aug_cfg.get("prob", 0.7):
        return x  # leave this batch clean
    noise_level = torch.rand(1).item() * aug_cfg.get("noise_max", 0.5)
    dropout_p = torch.rand(1).item() * aug_cfg.get("dropout_max", 0.3)
    if noise_level > 0:
        x = x + torch.randn_like(x) * noise_level
    if dropout_p > 0:
        mask = torch.rand_like(x) < dropout_p
        x = x.masked_fill(mask, 0.0)
    return x


def run_epoch(model, loader, criterion, device, optimizer=None, aug_cfg=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if training:
            optimizer.zero_grad()
            x = apply_augmentation(x, aug_cfg)
        logits = model(x)
        loss = criterion(logits, y)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def train_one(config_path: str, epochs_override: int = None, quiet: bool = False, suffix: str = ""):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_name = cfg["model"]
    ckpt_name = model_name + suffix
    tcfg = cfg["train"]
    aug_cfg = tcfg.get("augment")
    seed = tcfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = EPSWindowDataset("train")
    val_ds = EPSWindowDataset("val")
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    device = get_device()
    model = build_model(model_name, cfg["model_params"], train_ds.n_nodes,
                         train_ds.n_features, train_ds.n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    epochs = epochs_override or tcfg["epochs"]
    patience = tcfg.get("patience", 10)
    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0
    history = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, aug_cfg=aug_cfg)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                             val_loss=val_loss, val_acc=val_acc))
        if not quiet:
            print(f"[{ckpt_name}] epoch {epoch:3d}/{epochs} "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if not quiet:
                    print(f"[{ckpt_name}] early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.join(RESULTS_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)
    ckpt_path = os.path.join(RESULTS_DIR, "checkpoints", f"{ckpt_name}.pt")
    torch.save(dict(state_dict=model.state_dict(), config=cfg, n_params=n_params), ckpt_path)

    hist_path = os.path.join(RESULTS_DIR, "metrics", f"{ckpt_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(dict(model=model_name, n_params=n_params, elapsed_sec=elapsed,
                        best_val_loss=best_val_loss, history=history), f, indent=2)

    print(f"[{ckpt_name}] done in {elapsed:.1f}s, {n_params} params, "
          f"best_val_loss={best_val_loss:.4f} -> saved {ckpt_path}")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--suffix", type=str, default="",
                         help="appended to checkpoint/history filenames, e.g. '_aug'")
    args = parser.parse_args()
    train_one(args.config, epochs_override=args.epochs, suffix=args.suffix)


if __name__ == "__main__":
    main()
