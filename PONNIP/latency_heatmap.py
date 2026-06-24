"""
Generates Figure 1 of the PIP paper: per-inference latency of a 3-layer
photonic inference pipeline (784 -> 64 -> 32 -> 10), in PARALLEL (broadcast)
illumination mode, as a function of the source modulation frequency f_s and
the detector readout frequency f_d.

Model (deployed / pipeline mode, weight planes pre-loaded and static):
    T_layer = 1/f_s + N_out / f_d
    T_total = sum_l T_layer_l = L/f_s + (sum_l N_out_l) / f_d

For this network: L = 3, sum_l N_out_l = 64 + 32 + 10 = 106
"""
from matplotlib.figure import Figure
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np

L: int = 3
SUM_N_OUT: int = 64 + 32 + 10

f_s: np.ndarray = np.logspace(3, 8, 400)
f_d: np.ndarray = np.logspace(3, 8, 400)
FS: np.ndarray
FD: np.ndarray
FS, FD = np.meshgrid(f_s, f_d)

T_total_us: np.ndarray = (L / FS + SUM_N_OUT / FD) * 1e6

fig: Figure
ax: Axes
fig, ax = plt.subplots(figsize=(7.2, 5.6))

levels: np.ndarray = np.logspace(
    np.log10(T_total_us.min()),
    np.log10(T_total_us.max()),
    60
)

cf = ax.contourf(
    FS,
    FD,
    T_total_us,
    levels=levels,
    norm=plt.matplotlib.colors.LogNorm(),
    cmap="viridis"
)

cbar = fig.colorbar(cf, ax=ax)
cbar.set_label(r"Per-inference latency $T_{\mathrm{total}}$ ($\mu$s)")
cbar.ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Source modulation frequency $f_s$ (Hz)")
ax.set_ylabel(r"Detector readout frequency $f_d$ (Hz)")
ax.set_title(
    "Projected per-inference latency, 784-64-32-10 network\n"
    "(parallel illumination, weight planes pre-loaded)"
)

f_s_real: float = 2e4
f_d_real: float = 1e6

T_real_us: float = (L / f_s_real + SUM_N_OUT / f_d_real) * 1e6

ax.scatter(
    [f_s_real],
    [f_d_real],
    color="white",
    edgecolor="black",
    s=70,
    zorder=5,
    marker="*"
)

ax.annotate(
    f"DMD-class source + fast photodiode\n"
    f"$f_s$=20 kHz, $f_d$=1 MHz\n"
    f"$T_{{total}}$ ≈ {T_real_us:.0f} µs/inference",
    xy=(f_s_real, f_d_real),
    xytext=(3e4, 4e6),
    fontsize=8.5,
    color="white",
    arrowprops=dict(arrowstyle="->", color="white", lw=1)
)

plt.tight_layout()
plt.show()
