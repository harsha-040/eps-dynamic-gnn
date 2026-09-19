"""Quick standalone sanity check: simulate one episode per fault class and
print summary statistics + save a plot. Run with:
    python -m src.simulation.sanity_check
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import constants as C
from .eps_simulator import simulate_episode
from .faults import FAULT_CLASSES


def main():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(FAULT_CLASSES), 4, figsize=(16, 3 * len(FAULT_CLASSES)))

    for row, fault_type in enumerate(FAULT_CLASSES):
        X, info = simulate_episode(fault_type, rng=np.random.default_rng(row + 1))
        t = info["t"] / 60.0  # minutes
        print(f"\n=== {fault_type} ===")
        print(f"  eclipse frac: {info['eclipse_mask'].mean():.2f}")
        print(f"  fault active steps: {info['fault_mask'].sum()} / {len(t)}")
        print(f"  SOC range: [{X[:, C.NODE_INDEX['BAT'], 4].min():.3f}, {X[:, C.NODE_INDEX['BAT'], 4].max():.3f}]")
        print(f"  BUS voltage range: [{X[:, C.NODE_INDEX['BUS'], 0].min():.2f}, {X[:, C.NODE_INDEX['BUS'], 0].max():.2f}]")
        print(f"  power residual |mean|: {np.abs(info['power_balance_residual']).mean():.2f} W "
              f"(max {np.abs(info['power_balance_residual']).max():.2f} W)")
        assert not np.isnan(X).any(), "NaNs in simulated data!"
        assert not np.isinf(X).any(), "Infs in simulated data!"

        ax = axes[row]
        ax[0].plot(t, X[:, C.NODE_INDEX["SA"], 2], label="P_sa")
        ax[0].fill_between(t, 0, 1, where=info["eclipse_mask"], transform=ax[0].get_xaxis_transform(),
                            color="gray", alpha=0.2)
        ax[0].set_title(f"{fault_type}: SA power")
        ax[1].plot(t, X[:, C.NODE_INDEX["BAT"], 4], color="green", label="SOC")
        ax[1].set_title("BAT SOC")
        ax[2].plot(t, X[:, C.NODE_INDEX["BUS"], 0], color="orange", label="V_bus")
        ax[2].set_title("BUS voltage")
        ax[3].plot(t, X[:, C.NODE_INDEX["LOAD"], 1], color="red", label="I_load")
        ax[3].set_title("LOAD current")
        for a in ax:
            a.axvspan(info["fault_event"].onset_sec / 60.0,
                       (info["fault_event"].onset_sec + info["fault_event"].duration_sec) / 60.0,
                       color="red", alpha=0.1)

    fig.tight_layout()
    out_path = "results/figures/sanity_check.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved sanity plot to {out_path}")


if __name__ == "__main__":
    main()
