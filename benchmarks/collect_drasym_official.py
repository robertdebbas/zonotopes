"""
Runs QEBVerif's DRA-Sym (IPVerifier, mode=Sym) on all 150 benchmark experiments.

Run from anywhere with the qebverif-env activated:
  source ~/qebverif-env/bin/activate
  python benchmarks/collect_drasym_official.py

Outputs: results/data/drasym_results.csv
"""

import os, sys, csv, time

# Must run from the QEBVerif directory so relative imports work
QEBVERIF_DIR = "/Users/robertdebbas/Desktop/Intern:Projects/zonotopes/QEBVerif"
os.chdir(QEBVERIF_DIR)
sys.path.insert(0, QEBVERIF_DIR)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["METAL_LOG_LEVEL"] = "0"

import numpy as np
import tensorflow as tf
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
OUT_CSV = "/Users/robertdebbas/Desktop/Intern:Projects/zonotopes/results/data/drasym_results.csv"

# ── Input normalisation bounds (from QEBVerif) ───────────────────────────────
in_min  = [0.0,     -3.141593, -3.141593, 100.0,  0.0]
in_max  = [60760.0,  3.141593,  3.141593, 1200.0, 1200.0]
in_mean = [1.9791091e+04, 0.0,  0.0,      650.0,  600.0]
in_var  = [60261.0,  6.28318530718, 6.28318530718, 1100.0, 1200.0]

IN_UB = [(in_max[k] - in_mean[k]) / in_var[k] for k in range(5)]
IN_LB = [(in_min[k] - in_mean[k]) / in_var[k] for k in range(5)]

input_min_v,   input_max_v   = int_get_min_max(8, 8)
input_min_int, input_max_int = int_get_min_max_integer(8, 8)


def ij_to_acasnet(i, j):
    """Map (i,j) network index to QEBVerif's --AcasNet integer."""
    if j == 9:
        return i * 9          # j_index=0 branch: file = (i_index, 9)
    else:
        return (i - 1) * 9 + j  # file = (i_index+1, j_index)


def h5_path(acasnet):
    i_index = acasnet // 9
    j_index = acasnet - i_index * 9
    if j_index == 0:
        return f"{H5_DIR}/ACASXU_run2a_{i_index}_{9}_batch_2000.h5"
    else:
        return f"{H5_DIR}/ACASXU_run2a_{i_index + 1}_{j_index}_batch_2000.h5"


def build_model(Q, weight_path):
    """Build, compile and load a QuantizedModel for the given Q."""
    m = QuantizedModel(
        [50, 50, 50, 50, 50, 50, 5],
        input_bits=8,
        quantization_bits=Q,
        last_layer_signed=True,
    )
    m.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    m.build((None, 5))
    m.load_weights(weight_path)
    return m


def input_bounds(eps):
    """Convert real-domain eps to QEBVerif integer input bounds."""
    x_ub = [np.clip(X0[k] + eps, IN_LB[k], IN_UB[k]) for k in range(5)]
    x_lb = [np.clip(X0[k] - eps, IN_LB[k], IN_UB[k]) for k in range(5)]
    ub_int = np.array([np.clip(real_round(x * 256), input_min_int, input_max_int) for x in x_ub])
    lb_int = np.array([np.clip(real_round(x * 256), input_min_int, input_max_int) for x in x_lb])
    return lb_int, ub_int


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
            acasnet = ij_to_acasnet(ni, nj)
            wp      = h5_path(acasnet)
            print(f"Network ({ni},{nj})  [AcasNet={acasnet}]  weights: {wp}")

            for Q in Q_VALUES:
                model = build_model(Q, wp)

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
