"""
Generate benchmark plots from the Julia results and the DRA-Sym (QEBVerif) results.
Run collect_data.jl and collect_drasym_official.py first, then run this script from
the repository root (requires pandas and matplotlib).

All O_Diff values are in REAL domain (no scaling needed — all methods output
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
    for b in bars:
        h = b.get_height()
        if np.isnan(h) or h <= 0:
            continue
        ax.text(b.get_x() + b.get_width() / 2, h * 1.18,
                f"{h:.2f}" if h < 10 else f"{h:.1f}",
                ha="center", va="bottom", fontsize=7.5)


# ── Figure 1: O_Diff vs Q  (r = 6 and r = 19) — grouped bars, log y ──────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
for ax, r in zip(axes, [6, 19]):
    eps = r / 256.0
    x, w = np.arange(len(Q_VALUES)), 0.25
    vz = [avg(r, Q, "odiff_zono")      for Q in Q_VALUES]
    vl = [avg(r, Q, "odiff_con_light") for Q in Q_VALUES]
    vd = [avg_dra(r, Q)                for Q in Q_VALUES]

    bz = ax.bar(x - w, vz, w, label="Zonotope",
                color=COLORS["Zonotope"], alpha=0.85)
    bl = ax.bar(x,     vl, w, label="Constrained Zonotope",
                color=COLORS["Constrained Zonotope"], alpha=0.85)
    bd = ax.bar(x + w, vd, w, label="DRA-Sym",
                color=COLORS["DRA-Sym"], alpha=0.85)

    log_bar_labels(ax, bz)
    log_bar_labels(ax, bl)
    log_bar_labels(ax, bd)

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


# ── Figure 2: O_Diff vs radius  (Q = 6 and Q = 10) — line plot, log y ────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
for ax, Q in zip(axes, [6, 10]):
    vz = [avg(r, Q, "odiff_zono")      for r in RADII]
    vl = [avg(r, Q, "odiff_con_light") for r in RADII]
    vd = [avg_dra(r, Q)                for r in RADII]

    series = [
        ("Zonotope",             vz),
        ("Constrained Zonotope", vl),
        ("DRA-Sym",              vd),
    ]
    for label, vals in series:
        ax.plot(RADII, vals,
                color=COLORS[label], marker=MARKERS[label],
                linewidth=2, markersize=7, label=label)

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


# ── Figure 3: Precision vs Speed  (log-log scatter, all real domain) ──────────
fig, ax = plt.subplots(figsize=(9, 6))

rng = np.random.default_rng(42)
jitter_z = np.exp(rng.normal(0, 0.08, len(df)))
jitter_l = np.exp(rng.normal(0, 0.08, len(df)))
jitter_d = np.exp(rng.normal(0, 0.08, len(dra)))

CLIP_MIN = 1e-4
ax.scatter(df["time_zono"] * jitter_z,
           df["odiff_zono"].clip(lower=CLIP_MIN),
           s=50, alpha=0.45, color=COLORS["Zonotope"], marker="o", label="Zonotope")
ax.scatter(df["time_con_light"] * jitter_l,
           df["odiff_con_light"].clip(lower=CLIP_MIN),
           s=50, alpha=0.45, color=COLORS["Constrained Zonotope"],
           marker="s", label="Constrained Zonotope")
ax.scatter(dra["time_drasym"] * jitter_d,
           dra["odiff_drasym"].clip(lower=CLIP_MIN),
           s=50, alpha=0.45, color=COLORS["DRA-Sym"], marker="^", label="DRA-Sym")

# Annotate means
annot_cfg = [
    ("odiff_zono",      "time_zono",      df,  COLORS["Zonotope"],             "o"),
    ("odiff_con_light", "time_con_light", df,  COLORS["Constrained Zonotope"], "s"),
    ("odiff_drasym",    "time_drasym",    dra, COLORS["DRA-Sym"],              "^"),
]
annot_offsets = [
    (2.0,  1.5,  "left"),
    (0.55, 2.5,  "center"),
    (1.6,  0.42, "left"),
]
for (od_col, t_col, src, color, marker), (xm, ym, ha) in zip(annot_cfg, annot_offsets):
    mo, mt = src[od_col].mean(), src[t_col].mean()
    ax.scatter([mt], [mo], s=250, edgecolors="black", linewidths=2,
               color=color, marker=marker, zorder=7)
    ax.text(mt * xm, mo * ym, f"{mo:.1f}\n{mt:.3f}s",
            fontsize=8.5, fontweight="bold", color=color, va="center", ha=ha)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Verification Time (s)  [log scale]")
ax.set_ylabel("$O_{\\mathrm{Diff}}$  [log scale]")
ax.set_title("Precision vs. Speed  (all 150 experiments)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, which="both", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("results/figures/precision_vs_speed_all_radii_clean.pdf", bbox_inches="tight")
plt.close()
print("Saved precision_vs_speed_all_radii_clean.pdf")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Summary (all 150 experiments, real domain) ===")
print(f"Zonotope:              avg O_Diff = {df['odiff_zono'].mean():.4f},  avg time = {df['time_zono'].mean():.4f} s")
print(f"Constrained Zonotope:  avg O_Diff = {df['odiff_con_light'].mean():.4f},  avg time = {df['time_con_light'].mean():.4f} s")
print(f"DRA-Sym:               avg O_Diff = {dra['odiff_drasym'].mean():.4f},  avg time = {dra['time_drasym'].mean():.4f} s")

print("\n=== Small radii (r <= 6) ===")
merged = df.merge(dra, on=['net_i', 'net_j', 'Q', 'r'])
small  = merged[merged['r'] <= 6]
large  = merged[merged['r'] >= 13]
print(f"Constrained Zonotope  avg O_Diff: {small['odiff_con_light'].mean():.4f}  vs DRA-Sym: {small['odiff_drasym'].mean():.4f}")
print(f"Constrained Zonotope  avg time:   {small['time_con_light'].mean():.4f}s  vs DRA-Sym: {small['time_drasym'].mean():.4f}s")

print("\n=== Large radii (r >= 13) ===")
print(f"Constrained Zonotope  avg O_Diff: {large['odiff_con_light'].mean():.4f}  vs DRA-Sym: {large['odiff_drasym'].mean():.4f}")
