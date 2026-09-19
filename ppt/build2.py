#!/usr/bin/env python3
import os
from build_helpers import bullet_p, plain_p, replace_title, replace_body
from pic_helpers import add_picture_to_slide, insert_before_close_sptree

W = os.path.dirname(os.path.abspath(__file__))
SLIDES = os.path.join(W, "unpacked", "ppt", "slides")
FIGS = os.path.join(W, "..", "paper", "latex", "figs")
IMGS = os.path.join(W, "imgs")

def read(n):
    with open(os.path.join(SLIDES, f"slide{n}.xml"), encoding="utf-8") as f:
        return f.read()

def write(n, xml):
    with open(os.path.join(SLIDES, f"slide{n}.xml"), "w", encoding="utf-8") as f:
        f.write(xml)

EMU_PER_IN = 914400

# ------------------------------------------------------------- SLIDE 8 -----
# System Architecture
xml = read(8)
xml = replace_title(xml, "System Architecture")
paras = "".join([
    bullet_p("Dynamic-GNN pipeline (left): node telemetry → learned dynamic graph A(t) "
             "→ GCN → GAT → GRU (temporal fusion) → FC + Softmax classifier."),
    bullet_p("Fixed EPS power-flow topology (right): used only by the Static-GNN baseline for "
             "comparison — the proposed model learns its own graph instead."),
])
xml = replace_body(xml, paras)
pic1 = add_picture_to_slide(8, os.path.join(FIGS, "fig1_architecture.png"),
                             x=int(0.6*EMU_PER_IN), y=int(4.0*EMU_PER_IN), cx=int(5.9*EMU_PER_IN))
pic2 = add_picture_to_slide(8, os.path.join(FIGS, "fig2_topology.png"),
                             x=int(6.9*EMU_PER_IN), y=int(4.0*EMU_PER_IN), cx=int(5.6*EMU_PER_IN))
xml = insert_before_close_sptree(xml, pic1 + pic2)
write(8, xml)
print("slide8 done")

# ------------------------------------------------------------- SLIDE 9 -----
# Dataset Description
xml = read(9)
xml = replace_title(xml, "Dataset Description")
paras = "".join([
    bullet_p("Source: custom-built physics simulator (not third-party) implementing governing "
             "equations for solar power, battery SOC, bus voltage regulation, and system-wide "
             "power balance for a 5-node satellite EPS."),
    bullet_p("Size: 1,500 labeled samples — 300 per class, exactly balanced across 5 fault "
             "classes; split stratified 70:20:10 (train/val/test)."),
    bullet_p("Each sample: one full 96-minute orbit, 240 timesteps (24s apart), 5 nodes "
             "× 5 features (voltage, current, power, temperature, state-of-charge) "
             "— guaranteeing a full eclipse→sunlit cycle per sample."),
    bullet_p("Classes: nominal, sa_degradation, battery_over_discharge, pcu_regulator_fault, "
             "bus_distribution_fault — nominal is deliberately included as the hardest class."),
])
xml = replace_body(xml, paras)
pic = add_picture_to_slide(9, os.path.join(IMGS, "dataset_sample.png"),
                            x=int(0.6*EMU_PER_IN), y=int(5.6*EMU_PER_IN), cx=int(12.0*EMU_PER_IN))
xml = insert_before_close_sptree(xml, pic)
write(9, xml)
print("slide9 done")

# ------------------------------------------------------------ SLIDE 10 -----
# Proposed Methodology Modules
xml = read(10)
xml = replace_title(xml, "Proposed Methodology – Modules")
paras = "".join([
    bullet_p("Module 1 – EPS Physics Simulation & Dataset Generation"),
    bullet_p("Module 2 – Dynamic Graph Construction (learned A(t))"),
    bullet_p("Module 3 – GCN + GAT Structural & Attention Propagation"),
    bullet_p("Module 4 – Temporal Fusion (GRU) & Classification"),
    bullet_p("Module 5 – Training, Robustness Augmentation & Evaluation"),
])
xml = replace_body(xml, paras)
pic = add_picture_to_slide(10, os.path.join(FIGS, "fig1_architecture.png"),
                            x=int(3.7*EMU_PER_IN), y=int(4.55*EMU_PER_IN), cx=int(5.9*EMU_PER_IN))
xml = insert_before_close_sptree(xml, pic)
write(10, xml)
print("slide10 done")

print("Part 2 complete.")
