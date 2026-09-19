"""Generate the labeled EPS fault-diagnosis dataset.

Simulates N_PER_CLASS episodes for each fault class (including a nominal /
no-fault class), splits stratified 70:20:10 into train/val/test, and saves
everything to `data/processed/{split}.npz` plus a `data/processed/metadata.json`
describing the schema, splits, fault definitions, and train-set normalization
statistics.

Run with:
    python -m src.data.generate_dataset
"""
import json
import os
import numpy as np

from ..simulation import constants as C
from ..simulation.eps_simulator import simulate_episode
from ..simulation.faults import FAULT_CLASSES, FAULT_LABEL_INDEX

N_PER_CLASS = 300           # -> 1500 total samples (>= 1200 required)
SPLIT_RATIOS = (0.70, 0.20, 0.10)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RNG_SEED = C.RNG_SEED_DEFAULT


def generate_all(n_per_class: int = N_PER_CLASS, seed: int = RNG_SEED):
    master_rng = np.random.default_rng(seed)

    all_X, all_y, all_fault_mask, all_eclipse_mask, all_meta = [], [], [], [], []

    for fault_type in FAULT_CLASSES:
        label = FAULT_LABEL_INDEX[fault_type]
        for i in range(n_per_class):
            sample_rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
            X, info = simulate_episode(fault_type, rng=sample_rng)
            all_X.append(X)
            all_y.append(label)
            all_fault_mask.append(info["fault_mask"])
            all_eclipse_mask.append(info["eclipse_mask"])
            ev = info["fault_event"]
            all_meta.append(dict(
                fault_type=fault_type,
                onset_sec=float(ev.onset_sec),
                duration_sec=float(ev.duration_sec),
                severity=float(ev.severity),
                mode=ev.mode,
                phase_offset=float(info["phase_offset"]),
            ))
        print(f"simulated {n_per_class} episodes for class '{fault_type}'")

    X = np.stack(all_X).astype(np.float32)                 # (N, T, nodes, feat)
    y = np.array(all_y, dtype=np.int64)                     # (N,)
    fault_mask = np.stack(all_fault_mask).astype(bool)       # (N, T)
    eclipse_mask = np.stack(all_eclipse_mask).astype(bool)   # (N, T)

    # ---- stratified 70/20/10 split ----
    rng = np.random.default_rng(seed + 1)
    train_idx, val_idx, test_idx = [], [], []
    for label in range(len(FAULT_CLASSES)):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * SPLIT_RATIOS[0]))
        n_val = int(round(n * SPLIT_RATIOS[1]))
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    train_idx = np.array(sorted(train_idx))
    val_idx = np.array(sorted(val_idx))
    test_idx = np.array(sorted(test_idx))

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- normalization stats from TRAIN split only ----
    X_train = X[train_idx]
    feat_mean = X_train.mean(axis=(0, 1))   # (nodes, feat)
    feat_std = X_train.std(axis=(0, 1)) + 1e-6

    splits = dict(train=train_idx, val=val_idx, test=test_idx)
    counts = {}
    for split_name, idx in splits.items():
        np.savez_compressed(
            os.path.join(OUT_DIR, f"{split_name}.npz"),
            X=X[idx], y=y[idx],
            fault_mask=fault_mask[idx], eclipse_mask=eclipse_mask[idx],
            sample_index=idx,
        )
        vals, cnts = np.unique(y[idx], return_counts=True)
        counts[split_name] = {FAULT_CLASSES[v]: int(c) for v, c in zip(vals, cnts)}
        print(f"saved {split_name}.npz: {len(idx)} samples -> {counts[split_name]}")

    metadata = dict(
        description="Simulated satellite EPS fault-diagnosis dataset "
                     "(SA/BAT/PCU/BUS/LOAD graph, one full-orbit window per sample).",
        node_names=C.NODE_NAMES,
        static_edges=C.STATIC_EDGES,
        static_edge_index=C.STATIC_EDGE_INDEX.tolist(),
        feature_names=C.FEATURE_NAMES,
        feature_notes={
            "soc": "State of charge in [0,1]; only physically meaningful for "
                   "the BAT node, zero-filled for SA/PCU/BUS/LOAD.",
            "voltage": "Volts (V)", "current": "Amps (A)", "power": "Watts (W)",
            "temperature": "Degrees C",
        },
        fault_classes=FAULT_CLASSES,
        fault_label_index=FAULT_LABEL_INDEX,
        n_steps=C.N_STEPS,
        dt_store_sec=C.DT_STORE_SEC,
        t_orbit_sec=C.T_ORBIT_SEC,
        eclipse_fraction=C.ECLIPSE_FRACTION,
        n_per_class=n_per_class,
        n_total=int(len(y)),
        split_ratios=dict(train=SPLIT_RATIOS[0], val=SPLIT_RATIOS[1], test=SPLIT_RATIOS[2]),
        split_counts=counts,
        normalization=dict(
            method="z-score per (node, feature), computed on TRAIN split only",
            mean=feat_mean.tolist(),
            std=feat_std.tolist(),
        ),
        sample_fault_params=all_meta,  # per-sample onset/duration/severity, indexed like X
        rng_seed=seed,
    )
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nsaved metadata.json to {OUT_DIR}")
    print(f"total samples: {len(y)} | classes: {FAULT_CLASSES}")


if __name__ == "__main__":
    generate_all()
