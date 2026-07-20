"""
Generate benchmark plots from collected Julia data and DRA-Sym re-implementation.
Run collect_data.jl then collect_drasym.py first.

All O_Diff values are in REAL domain (no scaling needed — all three methods output
real-domain bounds over the same ε-box input regions).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.size']    = 11

df  = pd.read_csv("results/data/benchmark_results.csv")
dra = pd.read_csv("results/data/drasym_results.csv")

RADII    = [3, 6, 13, 19, 26]
Q_VALUES = [6, 8, 10]

COLORS = {
    "Zonotope":             "steelblue",
    "Constrained Zonotope": "seagreen",
    "DRA-Sym":              "salmon",
}
MARKERS = {
    "Zonotope":             "o",
    "Constrained Zonotope": "s",
    "DRA-Sym":              "^",
}

def avg(r, Q, col, src=df):
    return src[(src["r"] == r) & (src["Q"] == Q)][col].mean()

def avg_dra(r, Q):
    return avg(r, Q, "odiff_drasym", dra)

def log_bar_labels(ax, bars):
    """Place value labels above bars on a log-scale y axis."""
    for b in bars:
        h = b.get_height()
        if np.isnan(h) or h <= 0:
            continue
        ax.text(b.get_x() + b.get_width() / 2, h * 1.18,
                f"{h:.2f}" if h < 10 else f"{h:.1f}",
                ha="center", va="bottom", fontsize=7.5)


# ── Figure 1: O_Diff vs Q  (r = 6 and r = 19)  — grouped bars, log y ────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
for ax, r in zip(axes, [6, 19]):
    eps = r / 256.0
    x, w = np.arange(len(Q_VALUES)), 0.25
    vz = [avg(r, Q, "odiff_zono") for Q in Q_VALUES]
    vc = [avg(r, Q, "odiff_con")  for Q in Q_VALUES]
    vd = [avg_dra(r, Q)           for Q in Q_VALUES]

    bz = ax.bar(x - w, vz, w, label="Zonotope",             color=COLORS["Zonotope"],             alpha=0.85)
    bc = ax.bar(x,     vc, w, label="Constrained Zonotope", color=COLORS["Constrained Zonotope"], alpha=0.85)
    bd = ax.bar(x + w, vd, w, label="DRA-Sym",              color=COLORS["DRA-Sym"],              alpha=0.85)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q={q}" for q in Q_VALUES])
    ax.set_xlabel("Quantization Bits (Q)")
    ax.set_ylabel("Average $O_{\\mathrm{Diff}}$ (log scale)")
    ax.set_title(f"$r = {r}$  ($\\varepsilon = {eps:.3f}$)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", which="both", alpha=0.3, linestyle="--")

plt.suptitle("Precision vs. Quantization Bits", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/odiff_vs_Q_r6_r19.pdf", bbox_inches="tight")
plt.close()
print("Saved odiff_vs_Q_r6_r19.pdf")


# ── Figure 2: O_Diff vs radius  (Q = 6 and Q = 10) — line plot, log y ───────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
for ax, Q in zip(axes, [6, 10]):
    vz = [avg(r, Q, "odiff_zono") for r in RADII]
    vc = [avg(r, Q, "odiff_con")  for r in RADII]
    vd = [avg_dra(r, Q)           for r in RADII]

    for label, vals in [("Zonotope", vz), ("Constrained Zonotope", vc), ("DRA-Sym", vd)]:
        ax.plot(RADII, vals,
                color=COLORS[label], marker=MARKERS[label],
                linewidth=2, markersize=7, label=label)

    # shade the region where Constrained beats DRA-Sym
    shade_r = [r for r, c, d in zip(RADII, vc, vd) if c < d]
    if shade_r:
        ax.axvspan(min(shade_r) - 1.5, max(shade_r) + 1.5,
                   alpha=0.10, color="seagreen", label="Con wins region")

    ax.set_yscale("log")
    ax.set_xticks(RADII)
    ax.set_xticklabels(RADII)
    ax.set_xlabel("Attack Radius $r$")
    ax.set_ylabel("Average $O_{\\mathrm{Diff}}$ (log scale)")
    ax.set_title(f"$Q = {Q}$ bits")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")

plt.suptitle("Precision vs. Attack Radius", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/odiff_vs_radius_q6_q10.pdf", bbox_inches="tight")
plt.close()
print("Saved odiff_vs_radius_q6_q10.pdf")


# ── Figure 3: Precision vs Speed  (log-log scatter, all real domain) ─────────
fig, ax = plt.subplots(figsize=(9, 6))

rng = np.random.default_rng(42)
jitter_z = np.exp(rng.normal(0, 0.08, len(df)))
jitter_c = np.exp(rng.normal(0, 0.08, len(df)))
jitter_d = np.exp(rng.normal(0, 0.08, len(dra)))

# Clip near-zero O_Diff values so log scale stays valid
CLIP_MIN = 1e-4
ax.scatter(df["time_zono"] * jitter_z, df["odiff_zono"].clip(lower=CLIP_MIN),
           s=50, alpha=0.50, color=COLORS["Zonotope"],             marker="o", label="Zonotope")
ax.scatter(df["time_con"]  * jitter_c, df["odiff_con"].clip(lower=CLIP_MIN),
           s=50, alpha=0.50, color=COLORS["Constrained Zonotope"], marker="s", label="Constrained Zonotope")
ax.scatter(dra["time_drasym"] * jitter_d, dra["odiff_drasym"].clip(lower=CLIP_MIN),
           s=50, alpha=0.50, color=COLORS["DRA-Sym"],              marker="^", label="DRA-Sym")

# Annotate means (consistent with summary table)
annot_cfg = [
    ("odiff_zono",   "time_zono",   df,  COLORS["Zonotope"],             "o",  (0.4, 2.5)),
    ("odiff_con",    "time_con",    df,  COLORS["Constrained Zonotope"], "s",  (0.4, 0.4)),
    ("odiff_drasym", "time_drasym", dra, COLORS["DRA-Sym"],              "^",  (4.0, 0.12)),
]
# (x_mult, y_mult, ha) — multiplicative offsets on log scale, close to each marker
annot_offsets = [
    (2.0,  1.5,  "left"),    # Zonotope: close right and slightly above
    (0.55, 2.5,  "center"),  # Constrained: above and slightly left
    (1.6,  0.42, "left"),    # DRA-Sym: below and to the right
]
for (od_col, t_col, src, color, marker, _), (xm, ym, ha) in zip(annot_cfg, annot_offsets):
    mo, mt = src[od_col].mean(), src[t_col].mean()
    ax.scatter([mt], [mo], s=250, edgecolors="black", linewidths=2,
               color=color, marker=marker, zorder=7)
    ax.text(mt * xm, mo * ym, f"{mo:.1f}\n{mt:.3f}s",
            fontsize=8.5, fontweight="bold", color=color, va="center", ha=ha)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Verification Time (s)  [log scale]")
ax.set_ylabel("$O_{\\mathrm{Diff}}$  [log scale]")
ax.set_title("Precision–Speed Trade-off  (all 150 experiments)")
ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0, fontsize=9)
ax.grid(True, which="both", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("results/figures/precision_vs_speed_all_radii_clean.pdf", bbox_inches="tight")
plt.close()
print("Saved precision_vs_speed_all_radii_clean.pdf")


# ── Summary ──────────────────────────────────────────────────────────────────
print("\n=== Summary (all 150 experiments, real domain) ===")
print(f"Zonotope:          avg O_Diff = {df['odiff_zono'].mean():.4f},  avg time = {df['time_zono'].mean():.4f} s")
print(f"Constrained Zono:  avg O_Diff = {df['odiff_con'].mean():.4f},  avg time = {df['time_con'].mean():.4f} s")
print(f"DRA-Sym:           avg O_Diff = {dra['odiff_drasym'].mean():.4f},  avg time = {dra['time_drasym'].mean():.4f} s")

zono_vs_dra = df['odiff_zono'].mean() / dra['odiff_drasym'].mean()
con_vs_dra  = df['odiff_con'].mean()  / dra['odiff_drasym'].mean()
print(f"\nZonotope is {zono_vs_dra:.1f}x less precise than DRA-Sym (avg across all 150)")
print(f"Constrained is {con_vs_dra:.1f}x less precise than DRA-Sym (avg across all 150)")
print(f"Zonotope is {dra['time_drasym'].mean()/df['time_zono'].mean():.0f}x faster than DRA-Sym")
print(f"Constrained is {dra['time_drasym'].mean()/df['time_con'].mean():.0f}x faster than DRA-Sym")

# Per-regime win rate
merged = df.merge(dra, on=['net_i', 'net_j', 'Q', 'r'])
small_r = merged[merged['r'] <= 6]
large_r = merged[merged['r'] >= 13]
con_wins_small = (small_r['odiff_con'] < small_r['odiff_drasym']).sum()
con_wins_large = (large_r['odiff_con'] < large_r['odiff_drasym']).sum()
print(f"\nConstrained beats DRA-Sym: {con_wins_small}/{len(small_r)} experiments at r≤6  (small perturbation)")
print(f"Constrained beats DRA-Sym: {con_wins_large}/{len(large_r)} experiments at r≥13 (large perturbation)")
