#!/usr/bin/env python3
import os
from build_helpers import bullet_p, plain_p, replace_title, replace_body
from pic_helpers import add_picture_to_slide, insert_before_close_sptree

W = os.path.dirname(os.path.abspath(__file__))
SLIDES = os.path.join(W, "unpacked", "ppt", "slides")
FIGS = os.path.join(W, "..", "paper", "latex", "figs")
IMGS = os.path.join(W, "imgs")
EMU_PER_IN = 914400

def read(n):
    with open(os.path.join(SLIDES, f"slide{n}.xml"), encoding="utf-8") as f:
        return f.read()

def write(n, xml):
    with open(os.path.join(SLIDES, f"slide{n}.xml"), "w", encoding="utf-8") as f:
        f.write(xml)

# ------------------------------------------------------------ SLIDE 12 -----
# Intermediate Output Screenshots
xml = read(12)
xml = replace_title(xml, "Intermediate Output Screenshots")
paras = "".join([
    bullet_p("Data flow: Simulated Telemetry → Learned Dynamic Graph A(t) → Fault "
             "Classification (shown below for a real run of the trained model)."),
])
xml = replace_body(xml, paras)
pic_telemetry = add_picture_to_slide(12, os.path.join(IMGS, "dataset_sample.png"),
                                      x=int(0.6*EMU_PER_IN), y=int(3.05*EMU_PER_IN), cx=int(12.1*EMU_PER_IN))
pic_adj = add_picture_to_slide(12, os.path.join(FIGS, "fig4_adjacency.png"),
                                x=int(0.9*EMU_PER_IN), y=int(4.55*EMU_PER_IN), cx=int(4.5*EMU_PER_IN))
pic_cm = add_picture_to_slide(12, os.path.join(FIGS, "fig3_confusion.png"),
                               x=int(6.0*EMU_PER_IN), y=int(4.55*EMU_PER_IN), cx=int(4.5*EMU_PER_IN))
xml = insert_before_close_sptree(xml, pic_telemetry + pic_adj + pic_cm)
write(12, xml)
print("slide12 done")

# ------------------------------------------------------------ SLIDE 13 -----
# Conclusion & Future Enhancements
paras = "".join([
    bullet_p("Presented a Dynamic Graph Neural Network for satellite EPS fault diagnosis, "
             "learning the adjacency matrix from telemetry at every timestep instead of "
             "holding it fixed — directly addressing a limitation left as future work by "
             "the base paper."),
    bullet_p("Achieved 97.3% accuracy, 97.4% F1-score, and 0.996 AUC-ROC, outperforming a "
             "same-architecture static-graph baseline (81.3%) and non-graph CNN (76.0%) / "
             "LSTM (82.7%) baselines under identical training conditions."),
    bullet_p("Interpretability analysis confirmed the learned graph tracks real physical "
             "state — the dominant power source in A(t) shifts from battery to solar array "
             "within minutes of the eclipse-to-sunlit transition."),
    bullet_p("Found and partially fixed a real robustness weakness: the dynamic graph's "
             "advantage collapses under heavy sensor corruption because its structure "
             "depends on the same corrupted inputs it classifies; noise/dropout-augmented "
             "training substantially closed this gap."),
    bullet_p("Future Enhancements: multi-seed statistical validation, ablation studies "
             "isolating individual design choices, fault localization in time, and "
             "validation against real spacecraft telemetry."),
])
xml = read(13)
xml = replace_body(xml, paras)
write(13, xml)
print("slide13 done")

# ------------------------------------------------------------ SLIDE 14 -----
# References
paras = "".join([
    plain_p("[1] Y. Di, F. Wang, Z. Zhao, Z. Zhai, and X. Chen, “An interpretable graph "
            "neural network for real-world satellite power system anomaly detection based "
            "on graph filtering,” Expert Systems with Applications, vol. 254, p. 124348, 2024."),
    plain_p("[2] Z. Wu, K. Zou, Y. Liu, and L. Zhu, “Graph Neural Network-Assisted Fault "
            "Diagnosis for Satellite Attitude Control Systems,” in Proc. 2025 5th Int. "
            "Conf. Electronic Information Engineering and Computer Technology (EIECT), 2025."),
    plain_p("[3] B. L. H. Nguyen et al., “Spatial-Temporal Recurrent Graph Neural Networks "
            "for Fault Diagnostics in Power Distribution Systems,” IEEE Access, vol. 11, 2023."),
    plain_p("[4] V. Pozdnyakov, I. Makarov, and A. Kovalenko, “Graph Neural Networks With "
            "Trainable Adjacency Matrices for Fault Diagnosis on Multivariate Sensor Data,” "
            "IEEE Access, vol. 12, pp. 152860–152872, 2024."),
    plain_p("[5] M. Jin et al., “A Survey on Graph Neural Networks for Time Series: "
            "Forecasting, Classification, Imputation, and Anomaly Detection,” IEEE Trans. "
            "Pattern Anal. Mach. Intell., vol. 46, no. 12, pp. 10466–10485, 2024."),
    plain_p("Full 20-reference IEEE-format list is included in the accompanying project report."),
])
xml = read(14)
xml = replace_body(xml, paras)
write(14, xml)
print("slide14 done")

print("Part 4 complete.")
