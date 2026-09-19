#!/usr/bin/env python3
import os
from build_helpers import bullet_p, numbered_p, plain_p, replace_title, replace_body

W = os.path.dirname(os.path.abspath(__file__))
SLIDES = os.path.join(W, "unpacked", "ppt", "slides")

def read(n):
    with open(os.path.join(SLIDES, f"slide{n}.xml"), encoding="utf-8") as f:
        return f.read()

def write(n, xml):
    with open(os.path.join(SLIDES, f"slide{n}.xml"), "w", encoding="utf-8") as f:
        f.write(xml)

def module_slide(slide_num, module_title, input_text, output_text, algo_lines):
    xml = read(slide_num)
    xml = replace_title(xml, module_title)
    paras = []
    paras.append(numbered_p("Input: " + input_text))
    paras.append(numbered_p("Expected Output: " + output_text))
    paras.append(numbered_p("Algorithm:"))
    for line in algo_lines:
        paras.append(numbered_p(line, lvl=1))
    xml = replace_body(xml, "".join(paras))
    write(slide_num, xml)
    print(f"slide{slide_num} done")

# ------------------------------------------------------------ MODULE 1 -----
module_slide(11, "Methodology – Module 1: EPS Simulation & Dataset Generation",
    "physical parameters (P_max, orbit period 5760s, fault type & severity, seed).",
    "labeled (240 × 5 nodes × 5 features) telemetry tensor + one fault-class label.",
    [
        "Solar power: P_sa = P_max · cos(θ) · (1 − k_d·t), zeroed during eclipse.",
        "Battery SOC: dSOC/dt = (P_charge,eff − P_discharge − P_extra) / E_bat − k_sd·SOC.",
        "Bus voltage: V_bus = V_nom − k·(P_load − P_source), clipped to regulator limits.",
        "Power balance (enforced every step): P_sa + P_bat,out = P_bus,load + P_losses.",
    ])

# ------------------------------------------------------------ MODULE 2 -----
module_slide(15, "Methodology – Module 2: Dynamic Graph Construction",
    "node features x_t ∈ R^(5×5) at each timestep t.",
    "dynamic adjacency A(t) ∈ R^(5×5) + sparsified top-3 mask M(t).",
    [
        "Encode node features via a 2-layer MLP: h_t = MLP(x_t).",
        "Scaled dot-product attention scoring: A_soft(t) = softmax(h_t · h_tᵀ / √D).",
        "Sparsify: keep each destination node's top-3 highest-weighted sources + a "
        "self-loop → binary mask M(t) used by the GAT layer.",
    ])

# ------------------------------------------------------------ MODULE 3 -----
module_slide(16, "Methodology – Module 3: GCN + GAT Propagation",
    "learned adjacency A(t), mask M(t), and node features x_t.",
    "per-node embeddings (32-dim × 5 nodes), mean-pooled into one graph embedding.",
    [
        "GCN (structural propagation): H_gcn = W_gcn · (D(t)⁻¹(A_soft(t) + I) · x_t).",
        "GAT (attention-weighted importance), restricted to M(t): "
        "α_ij = softmax_j(LeakyReLU(a_dstᵀh_i + a_srcᵀh_j)), 2 attention heads.",
        "Node embeddings mean-pooled over the 5 nodes → one graph embedding per timestep.",
    ])

# ------------------------------------------------------------ MODULE 4 -----
module_slide(17, "Methodology – Module 4: Temporal Fusion & Classification",
    "sequence of 240 per-timestep graph embeddings (one full orbit).",
    "probability distribution over the 5 fault classes.",
    [
        "GRU (80 hidden units) processes the 240-step sequence.",
        "Final hidden state → Dropout → Fully-Connected layer.",
        "Softmax classifier: ŷ = softmax(W_c · h_T + b_c).",
    ])

# ------------------------------------------------------------ MODULE 5 -----
module_slide(18, "Methodology – Module 5: Training, Augmentation & Evaluation",
    "labeled training batches (optionally noise/dropout-augmented).",
    "trained model weights + test-set performance metrics.",
    [
        "Loss: categorical cross-entropy L = − Σ yᵢ log(ŷᵢ); Adam optimizer, early "
        "stopping (patience 10), matched ~18k–32k parameter budget across all models.",
        "Robustness augmentation: 70% of batches get Gaussian noise (0–0.5× per-channel "
        "std) and/or dropout (0–30%, replaced with channel mean).",
        "Evaluated on 150 held-out test episodes across 6 metrics: Accuracy, Precision, "
        "Recall, F1, AUC-ROC, Log Loss.",
    ])

print("Part 3 (modules) complete.")
