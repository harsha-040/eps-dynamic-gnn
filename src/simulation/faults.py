"""Fault mode definitions and injection logic for the EPS simulator.

Each fault is defined by a class name, an onset time (fraction of the
window), a duration (fraction of the window), and a severity, and is applied
by mutating simulator-internal parameters at each integration step via the
`apply(t, phase, ctx)` hook. `ctx` is a plain dict of the current-step
scratch variables the simulator exposes (see EPSSimulator._step).
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

FAULT_CLASSES = [
    "nominal",
    "sa_degradation",
    "battery_over_discharge",
    "pcu_regulator_fault",
    "bus_distribution_fault",
]
FAULT_LABEL_INDEX = {name: i for i, name in enumerate(FAULT_CLASSES)}


@dataclass
class FaultEvent:
    fault_type: str
    onset_sec: float
    duration_sec: float
    severity: float                # 0-1, arbitrary per-fault scaling
    mode: str = "step"             # "step" or "gradual" (used by SA fault)
    rng: Optional[np.random.Generator] = field(default=None, repr=False)

    def active(self, t: float) -> bool:
        if self.fault_type == "nominal":
            return False
        return self.onset_sec <= t <= self.onset_sec + self.duration_sec

    def progress(self, t: float) -> float:
        """0 before onset, ramps 0->1 over the fault duration."""
        if t < self.onset_sec:
            return 0.0
        if t >= self.onset_sec + self.duration_sec:
            return 1.0
        return (t - self.onset_sec) / max(self.duration_sec, 1e-6)


def sample_fault_event(fault_type: str, window_sec: float, rng: np.random.Generator) -> FaultEvent:
    """Randomize onset/duration/severity for a given fault class, keeping
    the fault window comfortably inside the sample so pre- and post-onset
    dynamics are both observable."""
    if fault_type == "nominal":
        return FaultEvent("nominal", onset_sec=window_sec, duration_sec=0.0, severity=0.0)

    onset_frac = rng.uniform(0.25, 0.60)
    duration_frac = rng.uniform(0.25, 0.65)
    duration_frac = min(duration_frac, 0.98 - onset_frac)
    onset_sec = onset_frac * window_sec
    duration_sec = duration_frac * window_sec
    severity = rng.uniform(0.5, 1.0)
    mode = "gradual" if rng.random() < 0.5 else "step"
    return FaultEvent(fault_type, onset_sec, duration_sec, severity, mode, rng)


# ---------------------------------------------------------------------------
# Per-fault effect application. Each function mutates and returns the
# relevant physical quantity given the nominal value and fault progress.
# ---------------------------------------------------------------------------

def apply_sa_degradation(event: FaultEvent, t: float, p_max_nominal: float, cos_incidence: float):
    """Gradual or step drop in P_max and/or abnormal incidence response."""
    if not event.active(t):
        return p_max_nominal, cos_incidence
    prog = event.progress(t) if event.mode == "gradual" else 1.0
    drop = 0.15 + 0.55 * event.severity  # 15%-70% capability loss
    p_max = p_max_nominal * (1.0 - drop * prog)
    incidence_penalty = 0.10 + 0.35 * event.severity
    cos_incidence = cos_incidence * (1.0 - incidence_penalty * prog)
    return p_max, max(cos_incidence, 0.0)


def apply_battery_fault(event: FaultEvent, t: float, extra_drain_w: float, charge_eff: float):
    """Anomalous SOC drift: extra unexplained drain and/or degraded charge
    acceptance during sunlit charging."""
    if not event.active(t):
        return extra_drain_w, charge_eff
    prog = event.progress(t) if event.mode == "gradual" else 1.0
    extra_drain_w = extra_drain_w + prog * (60.0 + 220.0 * event.severity)
    charge_eff = charge_eff * (1.0 - 0.6 * event.severity * prog)
    return extra_drain_w, max(charge_eff, 0.05)


def apply_pcu_fault(event: FaultEvent, t: float, v_bus: float, rng: np.random.Generator):
    """Regulation error exceeding tolerance: over- or under-voltage at BUS
    independent of the source/load balance."""
    if not event.active(t):
        return v_bus
    prog = event.progress(t) if event.mode == "gradual" else 1.0
    sign = 1.0 if (hash((id(event),)) % 2 == 0) else -1.0
    # deterministic-ish sign per event via severity fractional part
    sign = 1.0 if (event.severity * 1000) % 2 < 1 else -1.0
    offset = sign * prog * (2.0 + 6.0 * event.severity)
    jitter = rng.normal(0.0, 0.15 + 0.3 * event.severity)
    return v_bus + offset + jitter


def apply_bus_distribution_fault(event: FaultEvent, t: float, i_bus: float, rng: np.random.Generator):
    """Abnormal current draw / intermittent connectivity not explained by
    commanded load: random spikes and brief dropouts on the bus current."""
    if not event.active(t):
        return i_bus, False
    prog = event.progress(t) if event.mode == "gradual" else 1.0
    dropout = rng.random() < (0.04 + 0.10 * event.severity) * prog
    if dropout:
        return i_bus * rng.uniform(0.0, 0.2), True
    spike = rng.normal(0.0, (0.15 + 0.5 * event.severity) * prog) * abs(i_bus)
    return i_bus + spike, False
