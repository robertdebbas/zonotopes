"""
Runs QEBVerif's DRA-Sym (IPVerifier, mode=Sym) on all 150 benchmark experiments.

Run from anywhere, in a Python environment with TensorFlow 2, gurobipy and numpy
(see QEBVerif/README.md; a Gurobi license is required):
  python benchmarks/collect_drasym_official.py

Outputs: results/data/drasym_results.csv
"""

import os, sys, csv, time

# Must run from the QEBVerif directory so relative imports work
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QEBVERIF_DIR = os.path.join(REPO_ROOT, "QEBVerif")
os.chdir(QEBVERIF_DIR)
sys.path.insert(0, QEBVERIF_DIR)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["METAL_LOG_LEVEL"] = "0"

import numpy as np
import gurobipy as gp

# Suppress Gurobi console output
gp.setParam("OutputFlag", 0)

from utils_AcasXu.quantization_util import *
from utils_AcasXu.quantized_layers import QuantizedModel
from gurobi_encoding.gurobi_encoding_IP_AcasXu_FP import *

# ── Experiment parameters (must match collect_data.jl) ──────────────────────
NETWORKS = [(1,1),(1,5),(1,9),(2,2),(2,5),(2,9),(3,3),(3,5),(4,1),(4,5)]
Q_VALUES  = [6, 8, 10]
RADII     = [3, 6, 13, 19, 26]
X0        = [-0.2, -0.25, -0.5, -0.3, -0.44]
PROPERTY  = 4
N_OUT     = 5

H5_DIR  = "benchmark/acasxu"
OUT_CSV = os.path.join(REPO_ROOT, "results", "data", "drasym_results.csv")

# ── ACAS Xu input normalisation (same constants as collect_data.jl) ──────────
IN_MIN  = np.array([0.0,      -3.141593, -3.141593, 100.0,   0.0])
IN_MAX  = np.array([60760.0,   3.141593,  3.141593, 1200.0, 1200.0])
IN_MEAN = np.array([1.9791091e4, 0.0,     0.0,       650.0,  600.0])
IN_VAR  = np.array([60261.0,  6.28318530718, 6.28318530718, 1100.0, 1200.0])
IN_LB   = (IN_MIN - IN_MEAN) / IN_VAR
IN_UB   = (IN_MAX - IN_MEAN) / IN_VAR

# Inputs are signed 8-bit fixed-point values with 8 fractional bits
INPUT_BITS       = 8
INT_MIN, INT_MAX = int_get_min_max_integer(INPUT_BITS, INPUT_BITS)

LAYER_SIZES = [50] * 6 + [N_OUT]


def weights_file(i, j):
    """Path of the Keras weights of ACAS Xu network (i, j), relative to QEBVerif/."""
    return f"{H5_DIR}/ACASXU_run2a_{i}_{j}_batch_2000.h5"


def load_model(Q, weight_path):
    """QEBVerif model of an ACAS Xu network quantized to Q bits."""
    model = QuantizedModel(LAYER_SIZES, input_bits=INPUT_BITS,
                           quantization_bits=Q, last_layer_signed=True)
    model.build((None, len(X0)))
    model.load_weights(weight_path)
    return model


def input_bounds(eps):
    """Integer bounds of the input box X0 ± eps, clipped to the valid input domain."""
    def to_fixed_point(x):
        return np.clip([real_round(v * 2 ** INPUT_BITS) for v in x], INT_MIN, INT_MAX)

    center = np.array(X0)
    lower  = np.clip(center - eps, IN_LB, IN_UB)
    upper  = np.clip(center + eps, IN_LB, IN_UB)
    return to_fixed_point(lower), to_fixed_point(upper)


def run_drasym(model, lb_int, ub_int):
    """
    Run DRA-Sym propagation and return (odiff, elapsed).
    odiff = mean diff-interval width over all N_OUT output neurons (real domain).
    """
    ilp = QNNEncoding_gurobi(model, 0, "Sym")
    ilp.assert_input_box(lb_int, ub_int, "Sym")

    t0 = time.time()
    ilp.propogation("Sym")
    elapsed = time.time() - t0

    out = ilp.dense_layers[-1]
    widths = [out.diff_concrete_upper[k] - out.diff_concrete_lower[k] for k in range(N_OUT)]
    return float(np.mean(widths)), elapsed


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    total  = len(NETWORKS) * len(Q_VALUES) * len(RADII)
    n_done = 0

    with open(OUT_CSV, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["net_i", "net_j", "Q", "r", "eps", "odiff_drasym", "time_drasym"])

        for (ni, nj) in NETWORKS:
            wp = weights_file(ni, nj)
            print(f"Network ({ni},{nj})  weights: {wp}")

            for Q in Q_VALUES:
                model = load_model(Q, wp)

                for r in RADII:
                    eps = r / 256.0
                    lb_int, ub_int = input_bounds(eps)

                    try:
                        odiff, elapsed = run_drasym(model, lb_int, ub_int)
                    except Exception as e:
                        print(f"  FAILED ({ni},{nj}) Q={Q} r={r}: {e}")
                        odiff, elapsed = float("nan"), float("nan")

                    n_done += 1
                    print(f"  [{n_done:3d}/{total}] Q={Q} r={r:2d}: "
                          f"O_Diff={odiff:.4f}  ({elapsed:.3f}s)")
                    writer.writerow([ni, nj, Q, r, eps, odiff, elapsed])
                    csvf.flush()

    print(f"\nDone. Results saved to {OUT_CSV}")
