# Results

Pipeline: `generate_dataset.py` -> `train.py` (x4) -> `evaluate.py` -> `interpretability.py`.
All numbers below are from a single run with seed 42; see `results/metrics/` for the raw JSON/CSV and `results/figures/` for all plots.

## 1. Dataset composition

1500 simulated one-orbit (96 min, 240-step) windows over a 5-node EPS graph
(SA, BAT, PCU, BUS, LOAD), **exactly balanced** across 5 classes — 300
samples/class, split stratified 70:20:10:

| split | nominal | sa_degradation | battery_over_discharge | pcu_regulator_fault | bus_distribution_fault | total |
|---|---|---|---|---|---|---|
| train | 210 | 210 | 210 | 210 | 210 | 1050 |
| val   | 60  | 60  | 60  | 60  | 60  | 300  |
| test  | 30  | 30  | 30  | 30  | 30  | 150  |

Unlike the ACS baseline paper (3 fault classes, no nominal class), this
benchmark adds a `nominal` (no-fault) class, which is deliberately the
hardest class to separate from `battery_over_discharge` (both are
"everything looks normal except a subtle SOC trend") — see the confusion
matrices below.

Physics sanity checks (`results/figures/sanity_check.png`): no NaN/Inf in
any generated sample; eclipse gating on SA power is exact; the power-balance
residual `P_sa + P_bat_out - (P_load + P_losses)` averages ~40-45 W (~5% of
typical load) under nominal conditions from a small one-step-lag
approximation in the battery/shunt split, versus ~105 W mean (max ~880 W)
under the bus-distribution fault — i.e. the residual itself is a strong,
physically-grounded signature of that fault class, though it is **not**
exposed to any model as an input feature (only the 5 raw per-node channels are).

## 2. Performance (test set, n=150)

| Model | Params | Accuracy | Recall (macro) | F1 (macro) | AUC (macro, OVR) |
|---|---|---|---|---|---|
| CNN | 31,973 | 0.760 | 0.760 | 0.747 | 0.943 |
| LSTM | 17,861 | 0.827 | 0.827 | 0.832 | 0.951 |
| Static-GNN (ACS-style, fixed adjacency) | 29,109 | 0.813 | 0.813 | 0.813 | 0.960 |
| **Dynamic-GNN (proposed)** | 30,357 | **0.973** | **0.973** | **0.974** | **0.996** |

All four models were trained with the same splits, Adam, early stopping on
validation loss (patience 10, max 60 epochs), and a matched parameter budget
(~18k-32k trainable parameters). The dynamic-GNN did not trigger early
stopping within 60 epochs (val loss kept improving to 0.105); the
static-GNN plateaued and stopped at epoch 50 (best val loss 0.322) — see
`results/metrics/*_history.json` for full curves.

**Confusion matrices** (`results/figures/confusion_matrices.png`): every
model gets `pcu_regulator_fault` and `bus_distribution_fault` perfectly
right (100%) — both produce sharp, unambiguous signatures (bus voltage
railing near its regulator limit; erratic/spiky bus current). All three
non-dynamic models, however, substantially confuse `nominal` with
`battery_over_discharge` (CNN: 25/30 nominal samples misread as battery
fault; static-GNN: 9/30 and 17/30 respectively) — both classes differ only
in a slow SOC-trend anomaly, which a *fixed* graph structure and
non-graph baselines apparently struggle to isolate from normal
charge/discharge cycling. The dynamic-GNN reduces this confusion to 2/30 in
each direction, which is the single biggest source of its accuracy gain
over the static-graph baseline (which uses the identical GCN->GAT->GRU head
and a near-identical parameter count).

## 3. Interpretability findings

**(a) Attention shifts characteristically with fault class**
(`results/figures/attention_by_class.png`). For `nominal`, `sa_degradation`,
and `battery_over_discharge`, the model spreads attention fairly evenly
across SA/BAT/PCU (all ~0.20-0.26) with BUS/LOAD clearly lower — these three
classes only differ in the *source-side* power balance, and the model
treats them similarly at the node-attention level. For
**`pcu_regulator_fault`**, the pattern changes distinctly: PCU rises to the
top node (0.254) while SA's share drops to its lowest value across all
classes (0.174, vs. ~0.24 elsewhere) — consistent with a regulator fault
that decouples bus behavior from the solar array's actual output. For
**`bus_distribution_fault`**, BUS and LOAD both rise relative to their
levels in the other classes, reflecting attention shifting toward the
BUS-LOAD link where that fault physically lives. These shifts are
directionally sensible but not dramatic full reversals — a fair reading is
that the model relies on a mostly stable "core" attention pattern (SA/BAT/PCU
as primary information sources) and perturbs it in class-specific,
physically interpretable ways, rather than switching to a totally different
subgraph per class.

**(b) The learned adjacency A(t) visibly reconfigures across the
eclipse->sunlit transition** (`results/figures/dynamic_adjacency_eclipse_transition.png`,
`..._sa_edge_timeseries.png`) — this is the key plot supporting the
"dynamic" claim. During eclipse, BAT is the dominant *source* column in
A(t) (every other node draws heavily on BAT, as expected — it is the only
active power source). Within a few storage steps (~2.4 min) of the
eclipse->sunlit transition, the dominant source column flips to SA. The
SA-outgoing edge-weight time series shows this concretely: it sits around
0.20-0.25 through eclipse, jumps sharply (~+0.07) at the sunlit transition,
and mirrors the jump again at the next eclipse entry — the learned graph is
tracking real orbital/physical state, not just memorizing a fixed structure.

**(c) Overall**: the interpretability results support that the dynamic-GNN's
accuracy gain is not just a black-box improvement — its attention and
learned structure track physically meaningful state (source-of-power
switching, PCU-centric attention under regulator faults) in ways the fixed
topology cannot represent, since a static adjacency cannot change which
node ends up "mattering more" as the satellite crosses the terminator.

## 4. Robustness under corrupted telemetry

The reference ACS baseline paper (Wu et al.) claims their GNN "maintains
stable classification under noisy and partially missing data conditions."
We stress-tested that claim for all four of our models directly: take the
already-trained checkpoints (no retraining, no fine-tuning) and corrupt the
held-out test set along two independent axes before evaluating —
(`src/robustness.py`, `results/figures/noise_robustness.png` /
`dropout_robustness.png`, raw numbers in `results/metrics/robustness.json`):

- **Additive sensor noise**: Gaussian noise added to raw telemetry, scaled as
  a multiple of each channel's training-set standard deviation (0 to 1.0x).
- **Missing telemetry**: each individual (timestep, node, feature) reading
  independently dropped with probability *p* (0 to 50%) and replaced with
  the training-set mean for that channel — a standard "hold at expected
  value" fallback for missing sensor data.

**Result: the ranking flips at high corruption.** At clean-to-moderate
corruption, Dynamic-GNN's lead holds or grows (+14.7 pts over LSTM at clean
data, still +11.1 pts at 0.3x noise / +10.2 pts at 20% dropout). But past a
threshold, it degrades sharply and crosses below LSTM:

| Noise level (x std) | Dynamic-GNN | LSTM | Dynamic-GNN advantage |
|---|---|---|---|
| 0.0 (clean) | 97.3% | 82.7% | +14.7 |
| 0.3 | 94.0% | 82.9% | +11.1 |
| 0.5 | 71.1% | 82.2% | **-11.1** |
| 1.0 | 37.8% | 76.4% | **-38.7** |

| Missing-data fraction | Dynamic-GNN | LSTM | Dynamic-GNN advantage |
|---|---|---|---|
| 0% (clean) | 97.3% | 82.7% | +14.7 |
| 20% | 90.7% | 80.4% | +10.2 |
| 30% | 58.9% | 76.4% | **-17.6** |
| 50% | 39.8% | 63.8% | **-24.0** |

Both graph-based models (static *and* dynamic) degrade faster than the
non-graph baselines once corruption gets heavy — LSTM is the most robust
model overall at the highest corruption levels tested, despite being 15
points behind on clean data. CNN sits in between.

**Why this likely happens**: message-passing means a corrupted reading at
one node can contaminate its neighbors' representations too, not just its
own — the opposite of the redundancy CNN/LSTM get from treating channels
more independently. The dynamic model has a second, compounding failure
mode: the graph structure `A(t)` is itself *computed from* the same noisy
node features, so heavy corruption doesn't just feed bad data into a fixed
pipeline, it also corrupts *which* pipeline (which edges) gets used —
errors in structure estimation and errors in classification reinforce each
other. This is a real, currently-unaddressed weakness of the proposed
architecture, not just a baseline limitation: **none of the four models
were trained with any noise or dropout augmentation**, so all of them are
being asked to generalize to a corruption regime they never saw in
training. The likely fix is exactly that — adding corruption augmentation
during training.

## 5. Fixing it: noise/dropout-augmented training

We retrained all four models with random corruption injected during
training (`src/train.py`'s `apply_augmentation`, configs in
`configs/*_aug.yaml`): each training batch has a 70% chance of being
corrupted with noise level ~Uniform(0, 0.5x std) and/or dropout probability
~Uniform(0, 0.3), matching the same corruption process used in the stress
test. Validation/test data is never touched — only training input.
(`src/robustness_compare.py`, `results/figures/*_before_after.png`, raw
numbers in `results/metrics/robustness_augmented.json`.)

**It worked for the model that needed it most.** Dynamic-GNN's collapse
point moved dramatically: at 50% missing data it went from 39.3% -> 74.2%
(+34.9 pts); at 0.75x noise, 42.7% -> 82.7% (+40.0 pts). Its degradation
curve is now flat through most of the tested range instead of falling off a
cliff past 0.3x/20%.

**It cost clean-data accuracy, and helped inconsistently elsewhere** — an
honest result, not a clean sweep:

| Model | Clean acc: before | Clean acc: after | Change |
|---|---|---|---|
| CNN | 76.0% | 78.7% | +2.7 |
| LSTM | 82.7% | 78.7% | -4.0 |
| Static-GNN | 81.3% | 78.7% | -2.6 |
| Dynamic-GNN | 97.3% | 91.3% | -6.0 |

Every model gives up some clean-data accuracy for robustness — expected,
since training now spends part of its budget on corrupted examples instead
of only clean ones. More surprising: **augmentation didn't help every model
on every axis.** On the *noise* axis specifically, LSTM's augmented version
is worse than its clean version at every single corruption level, including
0 (dropping from 77.6% to 44.9% at the most extreme 1.0x noise — worse than
it was before augmentation). Static-GNN shows the same pattern at the
highest noise level (33.3% clean-trained vs 26.4% aug-trained at 1.0x). On
the *dropout* axis, augmentation helps every model, including LSTM. We
don't have a confirmed explanation for the noise-axis regression in
LSTM/Static-GNN — plausibly the specific noise range trained on
(0-0.5x) didn't generalize to the 0.75-1.0x range tested, or the added
training-time noise acted as a harmful regularizer for those two
architectures specifically. This would need controlled follow-up
(e.g. sweeping `noise_max`) to actually explain rather than speculate about.

**Net effect on the ranking**: augmented-trained Dynamic-GNN is no longer
the worst model under heavy corruption (as clean-trained Dynamic-GNN was) —
at 50% dropout it's now tied for best (74.2%, vs aug-LSTM's 74.7%); at 1.0x
noise it's second (64.9%, behind clean-trained LSTM's 77.6%, which remains
the single most noise-robust configuration found in either experiment).
Augmentation training is a real, effective fix for the specific failure
mode we found — closing most, not all, of the gap — but it is not a free
lunch, and "just add data augmentation" needed to be tested rather than
assumed.

## 6. Limitations

- **Simulated data, not real telemetry.** The EPS physics model
  (`src/simulation/`) is a simplified, first-order approximation (linear SOC
  dynamics, droop-based bus regulation, lumped thermal time constants, a
  single representative LEO orbit geometry). Real spacecraft telemetry has
  richer noise structure, sensor quantization/dropout, multi-orbit seasonal
  effects (beta angle, eclipse-season variation), and component-specific
  aging curves that this model does not capture.
- **Fault realism.** Fault injection uses parametric step/gradual
  perturbations to a handful of physical quantities per fault class; real
  faults (e.g. cell-level battery degradation, connector intermittents) can
  have more complex, multi-timescale signatures.
- **Single simulated topology.** All samples share one 5-node graph
  topology; the dynamic model's benefit here specifically comes from
  reweighting/re-routing importance *within* a fixed node set as physical
  state changes (day/night), not from discovering an unknown topology or
  handling a variable node set.
- **Augmentation's noise-axis regression in LSTM/Static-GNN is unexplained.**
  Section 5 found augmented training makes those two models *more* noise-
  fragile at the highest tested level, not less — we noted a plausible
  cause (training only sampled noise up to 0.5x std, tested to 1.0x) but
  didn't run the controlled sweep needed to confirm it.
- **Class balance is artificial.** Real fleet telemetry is overwhelmingly
  nominal; the current 1:1:1:1:1 balance was chosen (as in the ACS baseline
  paper) to make the classification benchmark itself tractable and
  comparable, not to reflect deployment-time class priors.
- **Single seed.** Results reflect one training run per model (seed 42);
  no variance/confidence intervals across seeds are reported.

Reproduce with:
```
python -m src.data.generate_dataset
for c in configs/cnn_baseline.yaml configs/lstm_baseline.yaml configs/static_gnn.yaml configs/dynamic_gnn.yaml; do python -m src.train --config $c; done
python -m src.evaluate && python -m src.interpretability && python -m src.robustness
for c in configs/*_aug.yaml; do python -m src.train --config $c --suffix _aug; done
python -m src.robustness_compare
```
