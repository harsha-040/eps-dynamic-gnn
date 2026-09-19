"""Interpretability analysis for the dynamic GNN:

1. Learned GAT attention weights per node, per fault class — do they line
   up with physical intuition (e.g. BAT node dominant for over-discharge,
   PCU node dominant for regulator faults)?
2. The learned dynamic adjacency A(t) across an eclipse -> sunlit
   transition, to show the graph structure actually changes with the
   satellite's orbital state (the key "dynamic" plot).

Usage:
    python -m src.interpretability
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import EPSWindowDataset, load_metadata
from .train import build_model, get_device

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def load_dynamic_model(test_ds):
    ckpt_path = os.path.join(CKPT_DIR, "dynamic_gnn.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = get_device()
    model = build_model(ckpt["config"]["model"], ckpt["config"]["model_params"],
                         test_ds.n_nodes, test_ds.n_features, test_ds.n_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, device


@torch.no_grad()
def per_class_attention(model, ds, device, class_names, max_per_class=40):
    """Average attention mass each node receives as an information SOURCE
    (i.e. how much other nodes' updated representations rely on it),
    averaged over time and over all test samples of each class."""
    n_nodes = ds.n_nodes
    importance = np.zeros((len(class_names), n_nodes))
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    by_class_logits_seen = {c: 0 for c in range(len(class_names))}
    for x, y in loader:
        x = x.to(device)
        _, extras = model(x, return_interpret=True)
        attn = extras["attention"].cpu().numpy()   # (B,T,N,N): [b,t,i,j] = weight j->i
        node_source_weight = attn.mean(axis=(1, 2))  # mean over t, dst-i -> (B, N) per-sample source importance
        for b in range(x.shape[0]):
            c = int(y[b])
            if by_class_logits_seen[c] >= max_per_class:
                continue
            importance[c] += node_source_weight[b]
            by_class_logits_seen[c] += 1

    counts = np.array([max(v, 1) for v in by_class_logits_seen.values()])
    importance = importance / counts[:, None]
    return importance  # (n_classes, n_nodes)


def plot_attention_by_class(importance, class_names, node_names, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    n_classes, n_nodes = importance.shape
    x = np.arange(n_classes)
    width = 0.8 / n_nodes
    colors = plt.cm.tab10(np.linspace(0, 1, n_nodes))
    for j in range(n_nodes):
        ax.bar(x + j * width, importance[:, j], width=width, label=node_names[j], color=colors[j])
    ax.set_xticks(x + width * (n_nodes - 1) / 2)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylabel("Mean GAT attention received (as info. source)")
    ax.set_title("Learned attention per node, per fault class (Dynamic GNN)")
    ax.legend(title="Node", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")

    # also report the argmax node per class as a plain-text finding
    findings = {}
    for c, cname in enumerate(class_names):
        top_node = node_names[int(np.argmax(importance[c]))]
        findings[cname] = dict(top_node=top_node,
                                scores={node_names[j]: float(importance[c, j]) for j in range(len(node_names))})
    return findings


@torch.no_grad()
def adjacency_across_eclipse_transition(model, ds, device, node_names, out_path, sample_idx=None):
    """Pick a nominal-class sample whose window contains a clean
    eclipse->sunlit transition and visualize A(t) before/at/after it."""
    # find a nominal sample (label 0) with a transition roughly mid-window
    nominal_idx = np.where(ds.y == 0)[0]
    chosen = None
    for idx in nominal_idx:
        ecl = ds.eclipse_mask[idx]
        transitions = np.where(np.diff(ecl.astype(int)) != 0)[0]
        if len(transitions) >= 1:
            chosen = idx
            transition_step = transitions[0]
            break
    if chosen is None:
        chosen = nominal_idx[0]
        transition_step = len(ds.eclipse_mask[chosen]) // 2

    x, y = ds[chosen]
    x = x.unsqueeze(0).to(device)   # (1,T,N,F)
    _, extras = model(x, return_interpret=True)
    a_soft = extras["a_soft"].cpu().numpy()[0]   # (T,N,N)

    t_before = max(transition_step - 3, 0)
    t_after = min(transition_step + 3, a_soft.shape[0] - 1)
    snapshots = {
        "deep eclipse": max(transition_step - 20, 0),
        "just before transition": t_before,
        "just after transition": t_after,
        "deep sunlit": min(transition_step + 20, a_soft.shape[0] - 1),
    }

    fig, axes = plt.subplots(1, len(snapshots), figsize=(4.2 * len(snapshots), 4.2))
    for ax, (label, step) in zip(axes, snapshots.items()):
        im = ax.imshow(a_soft[step], vmin=0, vmax=a_soft.max(), cmap="viridis")
        ax.set_title(f"{label}\n(t={step*ds.meta['dt_store_sec']/60:.1f} min)")
        ax.set_xticks(range(len(node_names)))
        ax.set_yticks(range(len(node_names)))
        ax.set_xticklabels(node_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(node_names, fontsize=8)
        ax.set_xlabel("source node j")
        if ax is axes[0]:
            ax.set_ylabel("destination node i")
    fig.suptitle("Learned dynamic adjacency A(t) across an eclipse->sunlit transition (Dynamic GNN)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")

    # companion time-series plot: how much attention SA receives as a
    # source (column sum) over the whole window, vs. eclipse state
    fig2, ax2 = plt.subplots(figsize=(9, 3.5))
    t_minutes = np.arange(a_soft.shape[0]) * ds.meta["dt_store_sec"] / 60.0
    sa_idx = node_names.index("SA")
    sa_out_weight = a_soft[:, :, sa_idx].mean(axis=1)  # mean over destinations
    ax2.plot(t_minutes, sa_out_weight, color="tab:orange", label="mean weight of edges (SA -> others)")
    ecl = ds.eclipse_mask[chosen]
    ax2.fill_between(t_minutes, 0, sa_out_weight.max() * 1.1, where=ecl, color="gray", alpha=0.25, label="eclipse")
    ax2.set_xlabel("time (min)")
    ax2.set_ylabel("learned edge weight")
    ax2.set_title("Learned SA->* edge weight over one orbit (Dynamic GNN)")
    ax2.legend()
    fig2.tight_layout()
    out_path2 = out_path.replace(".png", "_sa_edge_timeseries.png")
    fig2.savefig(out_path2, dpi=140)
    print(f"saved {out_path2}")

    return dict(sample_idx=int(chosen), transition_step=int(transition_step))


def main():
    meta = load_metadata()
    class_names = meta["fault_classes"]
    node_names = meta["node_names"]
    test_ds = EPSWindowDataset("test")

    model, device = load_dynamic_model(test_ds)

    os.makedirs(FIG_DIR, exist_ok=True)
    importance = per_class_attention(model, test_ds, device, class_names)
    findings = plot_attention_by_class(importance, class_names, node_names,
                                        os.path.join(FIG_DIR, "attention_by_class.png"))

    transition_info = adjacency_across_eclipse_transition(
        model, test_ds, device, node_names,
        os.path.join(FIG_DIR, "dynamic_adjacency_eclipse_transition.png"))

    os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)
    out = dict(attention_top_node_per_class=findings, eclipse_transition_sample=transition_info)
    with open(os.path.join(RESULTS_DIR, "metrics", "interpretability_findings.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Top attended node per fault class ===")
    for c, info in findings.items():
        print(f"  {c:28s} -> {info['top_node']}")


if __name__ == "__main__":
    main()
