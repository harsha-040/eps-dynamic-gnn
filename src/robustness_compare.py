"""Before/after comparison: does noise/dropout-augmented training actually
fix the robustness collapse found in `src.robustness`?

Loads BOTH the original clean-trained checkpoints (`<model>.pt`) and the
newly augmented-trained checkpoints (`<model>_aug.pt`) and runs the same
noise/dropout stress axes on all 8, so each model's before/after curve can
be compared directly, plus the overall ranking under the augmented-trained
condition.

Usage:
    python -m src.robustness_compare
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data.dataset import load_metadata
from .robustness import (CKPT_DIR, DROPOUT_PROBS, NOISE_LEVELS, RESULTS_DIR,
                          add_dropout, add_noise, load_raw_test,
                          run_stress_axis)
from .train import build_model, get_device

MODEL_KEYS = ["cnn_baseline", "lstm_baseline", "static_gnn", "dynamic_gnn"]
DISPLAY_NAMES = {
    "cnn_baseline": "CNN", "lstm_baseline": "LSTM",
    "static_gnn": "Static-GNN", "dynamic_gnn": "Dynamic-GNN",
}
COLORS = {"cnn_baseline": "#5b8cff", "lstm_baseline": "#ffb454",
          "static_gnn": "#ff6b6b", "dynamic_gnn": "#34d1a3"}


def load_all_variants(test_ds):
    device = get_device()
    models = {}
    for key in MODEL_KEYS:
        for variant, fname in [("clean", f"{key}.pt"), ("aug", f"{key}_aug.pt")]:
            path = os.path.join(CKPT_DIR, fname)
            if not os.path.exists(path):
                print(f"  missing {path}, skipping")
                continue
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = build_model(ckpt["config"]["model"], ckpt["config"]["model_params"],
                                 test_ds.n_nodes, test_ds.n_features, test_ds.n_classes).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            models[f"{key}__{variant}"] = model
    return models, device


def main():
    test_ds, mean, std = load_raw_test()
    X_raw = test_ds.X.astype(np.float32)
    y = test_ds.y
    models, device = load_all_variants(test_ds)
    print(f"loaded {len(models)} model variants: {list(models.keys())}")

    print("\n=== clean-data accuracy: before vs after augmentation ===")
    clean_acc = {}
    for name, model in models.items():
        from .robustness import normalize, accuracy_on
        acc = accuracy_on(model, device, normalize(X_raw, mean, std).astype(np.float32), y)
        clean_acc[name] = acc
        print(f"  {name:28s} {acc*100:5.1f}%")

    print("\n=== noise stress (clean vs aug, all models) ===")
    noise_results = run_stress_axis(
        models, device, X_raw, y, mean, std,
        corrupt_fn=lambda X, lvl, rng: add_noise(X, std, lvl, rng),
        levels=NOISE_LEVELS, seed_base=11,
    )
    for name, series in noise_results.items():
        print(f"  {name:28s} " + " ".join(f"{p['acc_mean']*100:5.1f}%" for p in series))

    print("\n=== dropout stress (clean vs aug, all models) ===")
    dropout_results = run_stress_axis(
        models, device, X_raw, y, mean, std,
        corrupt_fn=lambda X, lvl, rng: add_dropout(X, mean, lvl, rng),
        levels=DROPOUT_PROBS, seed_base=12,
    )
    for name, series in dropout_results.items():
        print(f"  {name:28s} " + " ".join(f"{p['acc_mean']*100:5.1f}%" for p in series))

    os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "metrics", "robustness_augmented.json"), "w") as f:
        json.dump(dict(clean_test_accuracy=clean_acc, noise_levels=NOISE_LEVELS,
                        dropout_probs=DROPOUT_PROBS, noise=noise_results, dropout=dropout_results), f, indent=2)

    plot_before_after(noise_results, NOISE_LEVELS, "Additive sensor noise (x per-channel train std)",
                       "noise_robustness_before_after.png")
    plot_before_after(dropout_results, DROPOUT_PROBS, "Fraction of readings missing",
                       "dropout_robustness_before_after.png")


def plot_before_after(results, levels, xlabel, filename):
    fig, axes = plt.subplots(1, len(MODEL_KEYS), figsize=(5 * len(MODEL_KEYS), 4.5), sharey=True)
    for ax, key in zip(axes, MODEL_KEYS):
        for variant, style in [("clean", "--"), ("aug", "-")]:
            series = results.get(f"{key}__{variant}")
            if series is None:
                continue
            means = [p["acc_mean"] * 100 for p in series]
            ax.plot(levels, means, style, marker="o", color=COLORS[key],
                    alpha=0.55 if variant == "clean" else 1.0,
                    label=f"{DISPLAY_NAMES[key]} ({'clean-trained' if variant=='clean' else 'aug-trained'})")
        ax.set_title(DISPLAY_NAMES[key])
        ax.set_xlabel(xlabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Test accuracy (%)")
    fig.suptitle("Effect of noise/dropout-augmented training on robustness (dashed=before, solid=after)")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "figures", filename)
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
