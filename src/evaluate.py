"""Evaluate all trained models on the held-out test split: Accuracy, macro
Recall, macro F1, macro-OVR AUC, and confusion matrices.

Usage:
    python -m src.evaluate
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from .data.dataset import EPSWindowDataset, load_metadata
from .train import build_model, get_device

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
MODEL_ORDER = ["cnn_baseline", "lstm_baseline", "static_gnn", "dynamic_gnn"]
DISPLAY_NAMES = {
    "cnn_baseline": "CNN",
    "lstm_baseline": "LSTM",
    "static_gnn": "Static-GNN (ACS-style)",
    "dynamic_gnn": "Dynamic-GNN (proposed)",
}


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    probs = F.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(axis=-1)
    return preds, probs, y.numpy()


def evaluate_model(model_name: str, test_ds, class_names):
    ckpt_path = os.path.join(CKPT_DIR, f"{model_name}.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = get_device()
    model = build_model(cfg["model"], cfg["model_params"], test_ds.n_nodes,
                         test_ds.n_features, test_ds.n_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])

    loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    preds, probs, y_true = get_predictions(model, loader, device)

    acc = accuracy_score(y_true, preds)
    recall = recall_score(y_true, preds, average="macro", zero_division=0)
    f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, preds, labels=list(range(len(class_names))))

    return dict(
        model=model_name, n_params=ckpt.get("n_params"),
        accuracy=acc, recall_macro=recall, f1_macro=f1, auc_macro_ovr=auc,
        confusion_matrix=cm.tolist(),
    )


def plot_confusion_matrices(results, class_names, out_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.array(r["confusion_matrix"])
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"{DISPLAY_NAMES.get(r['model'], r['model'])}\nAcc={r['accuracy']:.3f}")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted")
        if ax is axes[0]:
            ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


def main():
    meta = load_metadata()
    class_names = meta["fault_classes"]
    test_ds = EPSWindowDataset("test")

    results = []
    for model_name in MODEL_ORDER:
        ckpt_path = os.path.join(CKPT_DIR, f"{model_name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"skip {model_name}: no checkpoint at {ckpt_path}")
            continue
        r = evaluate_model(model_name, test_ds, class_names)
        results.append(r)
        print(f"{model_name:22s} acc={r['accuracy']:.3f} recall={r['recall_macro']:.3f} "
              f"f1={r['f1_macro']:.3f} auc={r['auc_macro_ovr']:.3f} params={r['n_params']}")

    os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "metrics", "performance.json"), "w") as f:
        json.dump(dict(class_names=class_names, results=results), f, indent=2)

    # CSV table
    csv_path = os.path.join(RESULTS_DIR, "metrics", "performance_table.csv")
    with open(csv_path, "w") as f:
        f.write("model,n_params,accuracy,recall_macro,f1_macro,auc_macro_ovr\n")
        for r in results:
            f.write(f"{DISPLAY_NAMES.get(r['model'], r['model'])},{r['n_params']},"
                    f"{r['accuracy']:.4f},{r['recall_macro']:.4f},{r['f1_macro']:.4f},{r['auc_macro_ovr']:.4f}\n")
    print(f"saved {csv_path}")

    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    plot_confusion_matrices(results, class_names,
                             os.path.join(RESULTS_DIR, "figures", "confusion_matrices.png"))


if __name__ == "__main__":
    main()
