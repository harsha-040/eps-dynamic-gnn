#!/usr/bin/env python3
import os
from build_helpers import bullet_p, numbered_p, plain_p, rich_bold_p, replace_title, replace_body

W = os.path.dirname(os.path.abspath(__file__))
SLIDES = os.path.join(W, "unpacked", "ppt", "slides")

def read(n):
    with open(os.path.join(SLIDES, f"slide{n}.xml"), encoding="utf-8") as f:
        return f.read()

def write(n, xml):
    with open(os.path.join(SLIDES, f"slide{n}.xml"), "w", encoding="utf-8") as f:
        f.write(xml)

# ---------------------------------------------------------------- SLIDE 1 --
xml = read(1)
xml = xml.replace("USN – Name 1", "1CR23CS165 – Sanjana P Sutrave")
xml = xml.replace("USN – Name 2", "1CR23CS135 – R Harsha")
xml = xml.replace("Name of Guide, Designation ", "Dr. D. Sumathi, Associate Professor")
write(1, xml)
print("slide1 done")

# ------------------------------------------------------------- SLIDE 3 -----
# Problem Statement
paras = "".join([
    bullet_p("Satellites depend on their Electrical Power System (EPS) — solar array, "
             "battery, regulator, bus, load — and undetected faults here can directly "
             "compromise the mission."),
    bullet_p("Existing GNN-based fault diagnosis methods (including our base paper) model the "
             "EPS as a graph with a FIXED adjacency matrix, assumed constant for the entire "
             "diagnosis process."),
    bullet_p("Gap: a fixed graph cannot represent how component importance shifts with orbital "
             "state — e.g. the EPS depends almost entirely on the battery during eclipse "
             "and on the solar array in sunlight — a shift no static graph can express. "
             "Existing methods also target binary anomaly detection, not multi-class fault "
             "diagnosis."),
    bullet_p("Proposed solution: a Dynamic Graph Neural Network (Dynamic-GNN) that learns a "
             "time-varying adjacency matrix A(t) directly from telemetry at every timestep, "
             "enabling accurate, interpretable, multi-class EPS fault diagnosis."),
])
xml = read(3)
xml = replace_body(xml, paras)
write(3, xml)
print("slide3 done")

# ------------------------------------------------------------- SLIDE 4 -----
# Objectives
paras = "".join([
    numbered_p("Construct a physically grounded simulation of a satellite EPS that generates "
               "labeled multivariate telemetry under nominal and four fault conditions."),
    numbered_p("Design a GNN-based fault diagnosis model whose graph structure is learned from "
               "telemetry at every timestep, instead of being fixed in advance."),
    numbered_p("Quantitatively establish, under identical training conditions and a matched "
               "parameter budget, whether a dynamic graph improves diagnostic accuracy over a "
               "fixed-topology GNN and non-graph (CNN, LSTM) baselines."),
    numbered_p("Evaluate the interpretability of the learned graph/attention and the model's "
               "robustness under noisy or missing telemetry."),
])
xml = read(4)
xml = replace_body(xml, paras)
write(4, xml)
print("slide4 done")

# ------------------------------------------------------------- SLIDE 5 -----
# Novelty
paras = "".join([
    bullet_p("All prior GNN-based fault/anomaly diagnosis work — including the base paper "
             "— constructs its graph ONCE and holds it fixed for the entire process."),
    bullet_p("Proposed Dynamic-GNN instead computes a fresh adjacency matrix A(t) at every "
             "timestep, from a small learned attention scorer over the current node "
             "embeddings, before propagating through GCN + GAT layers and a GRU."),
    bullet_p("This lets the model adaptively reweight which components matter most as orbital "
             "conditions change (e.g. battery-dominant in eclipse → solar-array-dominant "
             "in sunlight) — a shift a fixed graph has no mechanism to express."),
    bullet_p("To the best of our knowledge, the first model to apply timestep-wise dynamic "
             "graph construction specifically to satellite EPS fault diagnosis, directly "
             "addressing a limitation the base paper leaves as future work."),
])
xml = read(5)
xml = replace_body(xml, paras)
write(5, xml)
print("slide5 done")

# ------------------------------------------------------------- SLIDE 6 -----
# Base Paper
paras = "".join([
    bullet_p("Reference Paper — Journal Paper (Expert Systems With Applications, Elsevier):"),
    plain_p(""),
    plain_p("Y. Di, F. Wang, Z. Zhao, Z. Zhai, and X. Chen, “An interpretable graph neural "
            "network for real-world satellite power system anomaly detection based on graph "
            "filtering,” Expert Systems with Applications, vol. 254, p. 124348, 2024."),
    plain_p(""),
    bullet_p("Base paper uses an interpretable GNN with an adaptive graph filter for satellite "
             "power-system anomaly detection — but explicitly states the graph is STATIC "
             "(“inter-coupling relationship among the sensors... remains unchanged over "
             "time”) and targets binary anomaly detection, not multi-class fault diagnosis."),
])
xml = read(6)
xml = replace_body(xml, paras)
write(6, xml)
print("slide6 done")

# ------------------------------------------------------------- SLIDE 7 -----
# Progress After Review 1
paras = "".join([
    bullet_p("EPS physics simulator built and validated — governing equations for solar "
             "power, battery state-of-charge, bus voltage regulation, and system-wide power "
             "balance; sanity-checked for zero NaN/Inf and physically consistent behavior."),
    bullet_p("Dataset finalized: 1,500 labeled orbit episodes, 300 each across 5 balanced fault "
             "classes, split stratified 70:20:10."),
    bullet_p("All four models implemented and trained under identical conditions: CNN, LSTM, "
             "Static-Graph GNN (fixed-topology baseline), and the proposed Dynamic-GNN."),
    bullet_p("Full evaluation completed — Dynamic-GNN reaches 97.3% test accuracy vs. "
             "76.0–82.7% for the three baselines, with matched parameter budgets."),
    bullet_p("Interpretability analysis added: per-class attention weights, and visualization of "
             "the learned adjacency A(t) across an eclipse→sunlit orbital transition."),
    bullet_p("Robustness stress-tested under noisy/missing telemetry; found and partially fixed a "
             "real weakness via noise/dropout-augmented training."),
    bullet_p("Live interactive web demo built (FastAPI backend + browser frontend) that runs the "
             "simulator and all four trained models on demand."),
    bullet_p("Base paper identified and a full IEEE-format conference paper drafted (6 pages, 20 "
             "references)."),
])
xml = read(7)
xml = replace_body(xml, paras)
write(7, xml)
print("slide7 done")

print("Part 1 complete.")
