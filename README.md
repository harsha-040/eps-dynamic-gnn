# EPS Dynamic-GNN Fault Diagnosis

A dynamic Graph Neural Network (GNN) for fault diagnosis in a simulated
satellite **Electrical Power System (EPS)**. This project mirrors the
structure of Wu et al., *"Graph Neural Network-Assisted Fault Diagnosis for
Satellite Attitude Control Systems"* (IEEE EIECT 2025), but targets the EPS
subsystem instead of the ACS, and — the core novelty — replaces their fixed
adjacency matrix with a **dynamically learned graph structure** `A(t)` that
is recomputed from node features at every timestep.

## Project structure

```
eps-dynamic-gnn/
├── README.md
├── RESULTS.md                 # results summary (after running the pipeline)
├── requirements.txt
├── data/processed/            # generated dataset (train/val/test .npz + metadata.json)
├── src/
│   ├── simulation/            # EPS physics model + fault injection
│   │   ├── constants.py       # topology, physical constants, orbit params
│   │   ├── faults.py          # fault class definitions & injection logic
│   │   ├── eps_simulator.py   # the physics simulator itself
│   │   └── sanity_check.py    # quick physics sanity-check plot
│   ├── data/
│   │   ├── generate_dataset.py
│   │   └── dataset.py         # PyTorch Dataset
│   ├── models/
│   │   ├── layers.py          # shared dense GCN/GAT/dynamic-graph-learner layers
│   │   ├── dynamic_gnn.py     # proposed model
│   │   ├── static_gnn.py      # ACS-style fixed-graph baseline
│   │   ├── cnn_baseline.py
│   │   └── lstm_baseline.py
│   ├── train.py                # unified training entry point (all 4 models)
│   ├── evaluate.py             # test-set metrics + confusion matrices
│   ├── interpretability.py     # attention & dynamic-adjacency visualization
│   ├── robustness.py           # noise / missing-data stress test (clean-trained models)
│   └── robustness_compare.py   # before/after: clean-trained vs noise-augmented-trained
├── app/                         # live web demo (FastAPI backend + vanilla-JS frontend)
├── configs/                    # per-model hyperparameters (yaml)
│   └── *_aug.yaml              # same models, trained with noise/dropout augmentation
└── results/
    ├── checkpoints/            # trained model weights
    ├── metrics/                # JSON/CSV metrics, training history
    └── figures/                # plots (sanity check, confusion matrices, attention, adjacency)
```

## The EPS model

Five nodes, directed power/data-flow edges (used as the *fixed* topology for
the static-graph baseline, and as the graph the dynamic model may learn to
deviate from):

```
SA --> PCU --> BUS --> LOAD
        ^ |      |
        | v      |
       BAT <-----+ (BUS -> PCU voltage-regulation feedback)
```

| Node | Meaning | Features (voltage, current, power, temperature, soc) |
|---|---|---|
| SA   | Solar array            | `soc` is 0 (n/a) |
| BAT  | Battery                | `soc` is the real state of charge, [0,1] |
| PCU  | Power conditioning/regulation unit | `soc` is 0 (n/a) |
| BUS  | Power distribution bus | `soc` is 0 (n/a) |
| LOAD | Aggregate spacecraft load | `soc` is 0 (n/a) |

Each dataset sample is one **full LEO orbit window** (96 min, decimated to
240 stored timesteps, ~24 s/step), guaranteeing every sample contains a full
eclipse -> sunlit -> eclipse cycle. Governing equations (see
[`src/simulation/eps_simulator.py`](src/simulation/eps_simulator.py) and
[`src/simulation/constants.py`](src/simulation/constants.py) for exact
parameters):

- **Solar array**: `P_sa = P_max * cos(incidence_angle) * (1 - degradation_rate * t)`, zeroed in eclipse.
- **Battery SOC**: `dSOC/dt = (P_charge_eff - P_discharge - extra_drain) / (V_bat * C_bat) - self_discharge`.
- **Bus voltage regulation**: `V_bus = V_nominal - k * (P_load - P_source)`, clipped to regulator limits.
- **Power balance**: the source/battery split is solved so that `P_sa + P_bat_out = P_bus_load + P_losses` holds (with excess solar power beyond load+charge needs dumped by a shunt regulator, as in a real EPS) — verified in `sanity_check.py`, and deliberately *violated* by the bus-distribution fault to make that fault's signature detectable.

### Fault classes (balanced, 300 samples each)

1. `nominal` — no fault (harder benchmark than the ACS paper, which has no no-fault class).
2. `sa_degradation` — step/gradual drop in solar array `P_max` and/or incidence response.
3. `battery_over_discharge` — anomalous extra SOC drain and/or degraded charge acceptance.
4. `pcu_regulator_fault` — bus voltage regulation error (over/under-voltage) independent of source/load balance.
5. `bus_distribution_fault` — abnormal/intermittent BUS->LOAD current not explained by commanded load.

Run `python -m src.simulation.sanity_check` to regenerate
`results/figures/sanity_check.png`, a quick visual check that each fault
looks physically sensible (no NaNs/Infs, eclipse gating correct, fault
onset visible).

## The models

All four models share the same input `(batch, T=240, N=5, F=5)` and are
trained under identical conditions (same splits, similar parameter budget
~18k-32k params, Adam, early stopping on val loss).

- **CNN baseline** — 1D convolutions over the flattened `(N*F)`-channel time series, no graph structure.
- **LSTM baseline** — LSTM over the same flattened time series, no graph structure.
- **Static-GNN** — GCN -> GAT -> GRU -> classifier, using the **fixed** SA/BAT/PCU/BUS/LOAD topology at every timestep (reproduces the ACS paper's fixed-adjacency approach, adapted to EPS).
- **Dynamic-GNN (proposed)** — same GCN -> GAT -> GRU -> classifier, but the adjacency `A(t)` fed to the GCN (and its top-k sparsification fed to the GAT) is **learned at every timestep** from node features via a small MLP + attention scorer ([`DynamicGraphLearner`](src/models/layers.py)), rather than assumed fixed.

The GCN and GAT layers are implemented from scratch as dense
`(B, N, N)`-adjacency layers (see `src/models/layers.py`) rather than using
`torch_geometric`'s sparse `edge_index` API, because the dynamic model needs
a different, learned, weighted adjacency *per sample and per timestep*, and
because interpretability requires reading off the raw attention weights
directly.

## Running the pipeline

```bash
pip install -r requirements.txt

# 1. sanity-check the physics simulator
python -m src.simulation.sanity_check

# 2. generate the dataset (1500 samples, 70/20/10 stratified split)
python -m src.data.generate_dataset

# 3. train all four models
python -m src.train --config configs/cnn_baseline.yaml
python -m src.train --config configs/lstm_baseline.yaml
python -m src.train --config configs/static_gnn.yaml
python -m src.train --config configs/dynamic_gnn.yaml

# 4. evaluate on the held-out test set (metrics table + confusion matrices)
python -m src.evaluate

# 5. interpretability: attention-by-class + dynamic adjacency across eclipse transition
python -m src.interpretability

# 6. robustness: noise / missing-data stress test on the clean-trained checkpoints
python -m src.robustness

# 7. fix it: retrain with noise/dropout augmentation, then compare before vs after
python -m src.train --config configs/cnn_baseline_aug.yaml --suffix _aug
python -m src.train --config configs/lstm_baseline_aug.yaml --suffix _aug
python -m src.train --config configs/static_gnn_aug.yaml --suffix _aug
python -m src.train --config configs/dynamic_gnn_aug.yaml --suffix _aug
python -m src.robustness_compare

# 8. live demo (calls the real simulator + trained checkpoints on demand)
uvicorn app.backend.main:app --reload --port 8000   # then open http://localhost:8000
```

See [`RESULTS.md`](RESULTS.md) for the results summary after running the
pipeline.

## Running the live demo with Docker

No local Python setup needed — this builds a self-contained image (CPU-only
PyTorch, no GPU required) with the pretrained checkpoints already bundled in:

```bash
docker build -t eps-dynamic-gnn .
docker run --rm -p 8000:8000 eps-dynamic-gnn
# open http://localhost:8000
```

The image only ships what the live demo needs to run inference (code,
configs, trained checkpoints, and the dataset's normalization stats) — it
does not retrain anything, so it starts in seconds. To regenerate the
dataset or retrain a model, use the local Python workflow above instead.

## Live deployment

- **Live demo:** _TODO — add the deployed URL here once hosted (e.g. Hugging Face Spaces, Render, or Railway, all of which build directly from this repo's `Dockerfile`)._
- **Demo video:** _TODO — add a link to the recorded walkthrough here._

## Limitations

This is a **simulated-data** study with a simplified, first-order physics
model — not real EPS telemetry. See `RESULTS.md` for a fuller discussion.
