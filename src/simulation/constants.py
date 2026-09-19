"""Physical constants, node/edge topology, and nominal operating parameters
for the simplified satellite Electrical Power System (EPS) model.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Graph topology
# ---------------------------------------------------------------------------
NODE_NAMES = ["SA", "BAT", "PCU", "BUS", "LOAD"]
NODE_INDEX = {name: i for i, name in enumerate(NODE_NAMES)}
N_NODES = len(NODE_NAMES)

# Directed edges (power / data flow), used as the fixed topology for the
# static-graph baseline and as the "prior" structure the dynamic model may
# learn to deviate from.
STATIC_EDGES = [
    ("SA", "PCU"),   # solar array feeds PCU
    ("PCU", "BAT"),  # charge path
    ("BAT", "PCU"),  # discharge path
    ("PCU", "BUS"),  # regulated power to bus
    ("BUS", "LOAD"), # distribution to load
    ("BUS", "PCU"),  # voltage regulation feedback
]
STATIC_EDGE_INDEX = np.array(
    [[NODE_INDEX[a] for a, b in STATIC_EDGES],
     [NODE_INDEX[b] for a, b in STATIC_EDGES]],
    dtype=np.int64,
)

# ---------------------------------------------------------------------------
# Per-node feature schema (fixed order)
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["voltage", "current", "power", "temperature", "soc"]
N_FEATURES = len(FEATURE_NAMES)
# `soc` is only physically meaningful for BAT; it is zero-filled for the
# other nodes and documented as such in metadata.json.

# ---------------------------------------------------------------------------
# Orbit geometry
# ---------------------------------------------------------------------------
T_ORBIT_SEC = 5760.0          # ~96 minute LEO orbit
ECLIPSE_FRACTION = 0.35       # fraction of orbit spent in Earth's shadow
ECLIPSE_DURATION_SEC = T_ORBIT_SEC * ECLIPSE_FRACTION
SUNLIT_DURATION_SEC = T_ORBIT_SEC - ECLIPSE_DURATION_SEC

# Dataset sampling: one stored window == one full orbit (guarantees every
# sample contains a full eclipse->sunlit->eclipse cycle), decimated from a
# 1 Hz physics integration down to N_STEPS points for tractability.
DT_PHYSICS_SEC = 1.0
N_STEPS = 240
DECIMATION = int(T_ORBIT_SEC // N_STEPS)          # ~24 s between stored samples
DT_STORE_SEC = DECIMATION * DT_PHYSICS_SEC

# ---------------------------------------------------------------------------
# Solar array
# ---------------------------------------------------------------------------
SA_P_MAX_NOMINAL = 1800.0     # W, beginning-of-life max power
SA_DEGRADATION_RATE_PER_SEC = 1.0e-9  # slow natural aging
SA_V_NOMINAL = 100.0          # V, MPPT-regulated bus-side voltage
SA_TEMP_SUNLIT_EQ = 55.0      # deg C equilibrium temp while illuminated
SA_TEMP_ECLIPSE_EQ = -45.0    # deg C equilibrium temp in eclipse
SA_THERMAL_TAU_SEC = 400.0    # thermal time constant

# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------
BAT_CAPACITY_AH = 100.0        # Ah
BAT_V_MIN = 24.0               # V at SOC=0
BAT_V_MAX = 33.6               # V at SOC=1 (8s Li-ion-like pack)
BAT_CHARGE_EFFICIENCY = 0.95
BAT_SELF_DISCHARGE_PER_SEC = 1.0e-7
BAT_MAX_CHARGE_W = 500.0
BAT_MAX_DISCHARGE_W = 1200.0
BAT_INTERNAL_R = 0.06          # ohm, for I*R droop
BAT_SOC_INIT_RANGE = (0.45, 0.75)
BAT_THERMAL_TAU_SEC = 900.0
BAT_TEMP_AMBIENT = 15.0

# ---------------------------------------------------------------------------
# PCU (power conditioning / regulation unit)
# ---------------------------------------------------------------------------
PCU_V_NOMINAL = 28.0           # V, regulated bus setpoint
PCU_REG_GAIN_K = 0.0012        # V per W mismatch (droop)
PCU_V_LIMITS = (24.0, 32.0)    # regulator hard clip
PCU_EFFICIENCY = 0.93
PCU_THERMAL_TAU_SEC = 300.0
PCU_TEMP_AMBIENT = 20.0

# ---------------------------------------------------------------------------
# Bus / distribution
# ---------------------------------------------------------------------------
BUS_LINE_RESISTANCE = 0.015    # ohm, resistive distribution loss
BUS_THERMAL_TAU_SEC = 250.0
BUS_TEMP_AMBIENT = 20.0

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
LOAD_P_BASE = 500.0            # W, housekeeping baseline
LOAD_P_PAYLOAD_SUNLIT = 350.0  # W, extra payload duty during sunlit ops
LOAD_NOISE_STD = 15.0
LOAD_THERMAL_TAU_SEC = 250.0
LOAD_TEMP_AMBIENT = 20.0

RNG_SEED_DEFAULT = 42
