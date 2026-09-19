"""Physics-based simulator for a simplified satellite EPS.

Produces one labeled sample per call to `simulate_episode`: a
(N_STEPS, N_NODES, N_FEATURES) multivariate time series covering one full
orbit (eclipse -> sunlit -> eclipse), optionally with an injected fault.

Integration is performed directly at the stored decimation step
(`DT_STORE_SEC`, ~24 s) rather than substepping at 1 Hz: every state
variable in this model (SOC, thermal states) has a time constant much
larger than 24 s, so Euler integration at the storage cadence is accurate
enough for this simplified model while keeping dataset generation fast.
"""
from typing import Optional
import numpy as np

from . import constants as C
from .faults import (
    FaultEvent, sample_fault_event,
    apply_sa_degradation, apply_battery_fault, apply_pcu_fault,
    apply_bus_distribution_fault,
)


class OUNoise:
    """Simple vectorized Ornstein-Uhlenbeck noise for smooth-ish channels."""

    def __init__(self, n_channels: int, tau: float, sigma: np.ndarray, rng: np.random.Generator):
        self.n = n_channels
        self.tau = tau
        self.sigma = sigma
        self.state = np.zeros(n_channels)
        self.rng = rng

    def step(self, dt: float) -> np.ndarray:
        decay = np.exp(-dt / self.tau)
        diffusion = self.sigma * np.sqrt(1 - decay ** 2)
        self.state = self.state * decay + diffusion * self.rng.normal(size=self.n)
        return self.state


def _thermal_step(t_prev, t_target, tau, dt, noise=0.0):
    return t_prev + (t_target - t_prev) * (dt / tau) + noise


def simulate_episode(
    fault_type: str,
    rng: Optional[np.random.Generator] = None,
    phase_offset: Optional[float] = None,
):
    """Simulate one full-orbit window.

    Returns
    -------
    X : np.ndarray, shape (N_STEPS, N_NODES, N_FEATURES)
    info : dict with eclipse_mask, fault_mask, fault_event, power_balance_residual, t
    """
    if rng is None:
        rng = np.random.default_rng()
    if phase_offset is None:
        phase_offset = rng.uniform(0.0, C.T_ORBIT_SEC)

    dt = C.DT_STORE_SEC
    n_steps = C.N_STEPS
    event = sample_fault_event(fault_type, C.T_ORBIT_SEC, rng)

    noise = OUNoise(
        n_channels=8, tau=45.0,
        sigma=np.array([8.0, 6.0, 0.02, 0.05, 0.02, 0.3, 0.3, 0.3]),
        rng=rng,
    )

    # persistent state
    soc = rng.uniform(*C.BAT_SOC_INIT_RANGE)
    t_sa = C.SA_TEMP_SUNLIT_EQ
    t_bat = C.BAT_TEMP_AMBIENT
    t_pcu = C.PCU_TEMP_AMBIENT
    t_bus = C.BUS_TEMP_AMBIENT
    t_load = C.LOAD_TEMP_AMBIENT
    v_bat_prev = C.BAT_V_MIN + (C.BAT_V_MAX - C.BAT_V_MIN) * soc

    X = np.zeros((n_steps, C.N_NODES, C.N_FEATURES), dtype=np.float32)
    eclipse_mask = np.zeros(n_steps, dtype=bool)
    fault_mask = np.zeros(n_steps, dtype=bool)
    residual = np.zeros(n_steps, dtype=np.float32)
    t_arr = np.zeros(n_steps, dtype=np.float32)
    p_losses_prev = 0.0  # one-step-lagged loss estimate so the source/battery
    # split can supply P_losses too, honoring P_sa + P_bat_out = P_load + P_losses

    for k in range(n_steps):
        t = k * dt
        t_global = t  # local window clock, used for fault onset/duration
        phase = (t + phase_offset) % C.T_ORBIT_SEC
        eclipse = phase < C.ECLIPSE_DURATION_SEC
        eclipse_mask[k] = eclipse
        fault_mask[k] = event.active(t_global)
        t_arr[k] = t

        n_sa_p, n_load_p, n_incid, n_pcu_j, n_bus_i, n_sa_t, n_bat_t, n_load_t = noise.step(dt)

        # ---- Solar array ----
        cos_incidence = np.clip(0.95 + n_incid, 0.0, 1.0)
        p_max = C.SA_P_MAX_NOMINAL
        if fault_type == "sa_degradation":
            p_max, cos_incidence = apply_sa_degradation(event, t_global, p_max, cos_incidence)
        degrade = 1.0 - C.SA_DEGRADATION_RATE_PER_SEC * (t + phase_offset)
        if eclipse:
            p_sa = 0.0
        else:
            p_sa = max(p_max * cos_incidence * degrade + n_sa_p, 0.0)
        v_sa = (C.SA_V_NOMINAL + 0.5 * n_sa_p / 8.0) if p_sa > 1.0 else max(1.5 + n_sa_p * 0.05, 0.0)
        i_sa = p_sa / v_sa if v_sa > 1e-3 else 0.0
        sa_temp_target = C.SA_TEMP_ECLIPSE_EQ if eclipse else C.SA_TEMP_SUNLIT_EQ
        t_sa = _thermal_step(t_sa, sa_temp_target, C.SA_THERMAL_TAU_SEC, dt, n_sa_t)

        # ---- Load command ----
        payload = 0.0 if eclipse else C.LOAD_P_PAYLOAD_SUNLIT
        p_load_cmd = max(C.LOAD_P_BASE + payload + n_load_p, 50.0)

        # ---- Power split between source and battery ----
        # source must cover both the commanded load and the (lagged-estimate)
        # system losses, per P_sa + P_bat_out = P_bus_load + P_losses
        net = p_sa - (p_load_cmd + p_losses_prev)
        if net >= 0:
            p_charge_raw = min(net, C.BAT_MAX_CHARGE_W)
            p_discharge_raw = 0.0
            # excess solar power beyond load+charge needs is dumped by a
            # shunt regulator (standard EPS practice), not left unbalanced
            p_shunt = net - p_charge_raw
        else:
            p_charge_raw = 0.0
            p_discharge_raw = min(-net, C.BAT_MAX_DISCHARGE_W)
            p_shunt = 0.0

        extra_drain_w, charge_eff = 0.0, C.BAT_CHARGE_EFFICIENCY
        if fault_type == "battery_over_discharge":
            extra_drain_w, charge_eff = apply_battery_fault(event, t_global, extra_drain_w, charge_eff)

        p_charge_eff = p_charge_raw * charge_eff
        battery_energy_capacity_J = C.BAT_CAPACITY_AH * C.BAT_V_MAX * 3600.0
        dsoc = (p_charge_eff - p_discharge_raw - extra_drain_w) / battery_energy_capacity_J
        dsoc -= C.BAT_SELF_DISCHARGE_PER_SEC * soc
        soc = float(np.clip(soc + dsoc * dt, 0.0, 1.0))

        i_bat = (p_charge_raw - p_discharge_raw) / max(v_bat_prev, 1e-3)
        v_bat = C.BAT_V_MIN + (C.BAT_V_MAX - C.BAT_V_MIN) * soc + i_bat * C.BAT_INTERNAL_R
        v_bat = float(np.clip(v_bat, C.BAT_V_MIN - 2.0, C.BAT_V_MAX + 2.0))
        p_bat = i_bat * v_bat
        v_bat_prev = v_bat
        bat_temp_target = C.BAT_TEMP_AMBIENT + 12.0 * (abs(i_bat) / max(C.BAT_MAX_DISCHARGE_W / C.BAT_V_MIN, 1.0))
        t_bat = _thermal_step(t_bat, bat_temp_target, C.BAT_THERMAL_TAU_SEC, dt, n_bat_t)

        # ---- PCU / bus voltage regulation ----
        p_source = p_sa + p_discharge_raw
        mismatch = p_load_cmd - p_source
        v_bus = C.PCU_V_NOMINAL - C.PCU_REG_GAIN_K * mismatch
        if fault_type == "pcu_regulator_fault":
            v_bus = apply_pcu_fault(event, t_global, v_bus, rng)
        else:
            v_bus = v_bus + n_pcu_j
        v_bus_clipped = float(np.clip(v_bus, *C.PCU_V_LIMITS))

        p_pcu_through = max(p_sa, 0.0) + p_discharge_raw
        p_losses_pcu = (1.0 - C.PCU_EFFICIENCY) * p_pcu_through
        i_pcu = p_pcu_through / max(v_bus_clipped, 1e-3)
        t_pcu = _thermal_step(t_pcu, C.PCU_TEMP_AMBIENT + 20.0 * (p_losses_pcu / 150.0), C.PCU_THERMAL_TAU_SEC, dt)

        # ---- Bus / distribution ----
        i_bus_nominal = p_load_cmd / max(v_bus_clipped, 1e-3)
        i_bus = i_bus_nominal + n_bus_i
        dropout = False
        if fault_type == "bus_distribution_fault":
            i_bus, dropout = apply_bus_distribution_fault(event, t_global, i_bus, rng)
        v_bus_node = v_bus_clipped - i_bus * C.BUS_LINE_RESISTANCE
        p_bus = i_bus * v_bus_node
        p_loss_bus = (i_bus ** 2) * C.BUS_LINE_RESISTANCE
        t_bus = _thermal_step(t_bus, C.BUS_TEMP_AMBIENT + 15.0 * (p_loss_bus / 40.0), C.BUS_THERMAL_TAU_SEC, dt)

        # ---- Load ----
        v_load = v_bus_node - 0.05 * abs(i_bus)
        i_load = i_bus
        p_load_actual = v_load * i_load if not dropout else 0.0
        t_load = _thermal_step(t_load, C.LOAD_TEMP_AMBIENT + 10.0 * (p_load_actual / 850.0), C.LOAD_THERMAL_TAU_SEC, dt, n_load_t)

        residual[k] = (p_sa + p_discharge_raw) - (p_load_actual + p_losses_pcu + p_loss_bus + p_charge_raw + p_shunt)
        p_losses_prev = p_losses_pcu + p_loss_bus

        X[k, C.NODE_INDEX["SA"]] = [v_sa, i_sa, p_sa, t_sa, 0.0]
        X[k, C.NODE_INDEX["BAT"]] = [v_bat, i_bat, p_bat, t_bat, soc]
        X[k, C.NODE_INDEX["PCU"]] = [v_bus_clipped, i_pcu, p_pcu_through, t_pcu, 0.0]
        X[k, C.NODE_INDEX["BUS"]] = [v_bus_node, i_bus, p_bus, t_bus, 0.0]
        X[k, C.NODE_INDEX["LOAD"]] = [v_load, i_load, p_load_actual, t_load, 0.0]

    info = dict(
        eclipse_mask=eclipse_mask,
        fault_mask=fault_mask,
        fault_event=event,
        power_balance_residual=residual,
        t=t_arr,
        phase_offset=phase_offset,
    )
    return X, info
