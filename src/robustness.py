"""Noise / missing-data robustness stress test.

The reference paper (Wu et al., IEEE EIECT 2025) claims their GNN "maintains
stable classification under noisy and partially missing data conditions."
This script tests that claim for all four of *our* trained models: take the
already-trained checkpoints (no retraining) and corrupt the held-out test
set at increasing severity along two independent axes:

  1. Additive sensor noise  -- Gaussian noise added to raw telemetry, scaled
     as a multiple of each (node, feature)'s training-set standard deviation.
  2. Missing telemetry      -- each (timestep, node, feature) reading is
     independently dropped with probability p and replaced by the
     training-set mean for that channel (a common "hold at expected value"
     fallback for missing sensor data).

Both corruptions are applied to the RAW (unnormalized) test data, then the
usual train-set normalization is applied before feeding the models, exactly
mirroring how the data pipeline works at training time.

Usage:
    python -m src.robustness
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .data.dataset import EPSWindowDataset, load_metadata
from .evaluate import CKPT_DIR, DISPLAY_NAMES, MODEL_ORDER
from .train import build_model, get_device

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]     # multiples of per-channel train std
DROPOUT_PROBS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]          # fraction of readings replaced with the channel mean
N_TRIALS = 3                                              # repeats per severity (corruption is random)


def load_raw_test():
    ds = EPSWindowDataset("test", normalize=False)
    meta = load_metadata()
    mean = np.array(meta["normalization"]["mean"], dtype=np.float32)  # (N,F)
    std = np.array(meta["normalization"]["std"], dtype=np.float32)
    return ds, mean, std


def normalize(X, mean, std):
    return (X - mean[None, None]) / std[None, None]


def add_noise(X, std, level, rng):
    if level == 0.0:
        return X
    noise = rng.normal(0.0, 1.0, size=X.shape).astype(np.float32) * (level * std[None, None])
    return X + noise


def add_dropout(X, mean, prob, rng):
    if prob == 0.0:
        return X
    mask = rng.random(size=X.shape) < prob
    X_out = X.copy()
    filler = np.broadcast_to(mean[None, None], X.shape)
    X_out[mask] = filler[mask]
    return X_out


@torch.no_grad()
def accuracy_on(model, device, X_norm, y, batch_size=32):
    model.eval()
    n = len(y)
    correct = 0
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(X_norm[i:i + batch_size]).to(device)
        yb = y[i:i + batch_size]
        logits = model(xb)
        preds = logits.argmax(dim=-1).cpu().numpy()
        correct += (preds == yb).sum()
    return correct / n


def load_models(test_ds):
    device = get_device()
    models = {}
    for name in MODEL_ORDER:
        ckpt_path = os.path.join(CKPT_DIR, f"{name}.pt")
        if not os.path.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(ckpt["config"]["model"], ckpt["config"]["model_params"],
                             test_ds.n_nodes, test_ds.n_features, test_ds.n_classes).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models[name] = model
    return models, device


def run_stress_axis(models, device, X_raw, y, mean, std, corrupt_fn, levels, seed_base=0):
    results = {name: [] for name in models}
    for level in levels:
        for name, model in models.items():
            accs = []
            for trial in range(N_TRIALS if level > 0 else 1):
                rng = np.random.default_rng(seed_base + trial * 1000 + int(level * 997))
                X_corrupt = corrupt_fn(X_raw, level, rng)
                X_norm = normalize(X_corrupt, mean, std).astype(np.float32)
                accs.append(accuracy_on(model, device, X_norm, y))
            results[name].append(dict(level=level, acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs))))
    return results


def main():
    test_ds, mean, std = load_raw_test()
    X_raw = test_ds.X.astype(np.float32)   # raw, unnormalized
    y = test_ds.y
    models, device = load_models(test_ds)
    print(f"loaded models: {list(models.keys())}")

    print("\n=== Sensor noise stress test ===")
    noise_results = run_stress_axis(
        models, device, X_raw, y, mean, std,
        corrupt_fn=lambda X, lvl, rng: add_noise(X, std, lvl, rng),
        levels=NOISE_LEVELS, seed_base=1,
    )
    for name, series in noise_results.items():
        print(f"  {DISPLAY_NAMES[name]:24s} " + " ".join(f"{p['acc_mean']*100:5.1f}%" for p in series))

    print("\n=== Missing-data (dropout) stress test ===")
    dropout_results = run_stress_axis(
        models, device, X_raw, y, mean, std,
        corrupt_fn=lambda X, lvl, rng: add_dropout(X, mean, lvl, rng),
        levels=DROPOUT_PROBS, seed_base=2,
    )
    for name, series in dropout_results.items():
        print(f"  {DISPLAY_NAMES[name]:24s} " + " ".join(f"{p['acc_mean']*100:5.1f}%" for p in series))

    os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "metrics", "robustness.json"), "w") as f:
        json.dump(dict(noise_levels=NOISE_LEVELS, dropout_probs=DROPOUT_PROBS,
                        n_trials=N_TRIALS, noise=noise_results, dropout=dropout_results), f, indent=2)

    plot_curves(noise_results, NOISE_LEVELS, "Additive sensor noise (× per-channel train std)",
                "noise_robustness.png")
    plot_curves(dropout_results, DROPOUT_PROBS, "Fraction of readings missing (replaced with channel mean)",
                "dropout_robustness.png")


def plot_curves(results, levels, xlabel, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"cnn_baseline": "#5b8cff", "lstm_baseline": "#ffb454",
              "static_gnn": "#ff6b6b", "dynamic_gnn": "#34d1a3"}
    for name, series in results.items():
        means = [p["acc_mean"] * 100 for p in series]
        stds = [p["acc_std"] * 100 for p in series]
        ax.errorbar(levels, means, yerr=stds, marker="o", label=DISPLAY_NAMES[name],
                    color=colors.get(name), capsize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Robustness under corrupted telemetry (models trained on clean data, not retrained)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "figures", filename)
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
