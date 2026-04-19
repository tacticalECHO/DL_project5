import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── palette ───────────────────────────────────────────────────────────────────
colors  = {"PConv": "#2196F3", "GAN": "#F44336", "CVAE": "#4CAF50"}
methods = ["PConv", "GAN", "CVAE"]
regimes = ["small", "medium", "large"]

# ── data ──────────────────────────────────────────────────────────────────────
celebahq_fid = {
    "small":  {"PConv": {"MSE": 0.000166, "FID": 0.560},
               "GAN":   {"MSE": 0.000197, "FID": 0.811},
               "CVAE":  {"MSE": 0.000229, "FID": 1.253}},
    "medium": {"PConv": {"MSE": 0.001633, "FID": 2.684},
               "GAN":   {"MSE": 0.001789, "FID": 3.195},
               "CVAE":  {"MSE": 0.002054, "FID": 7.939}},
    "large":  {"PConv": {"MSE": 0.005389, "FID": 19.879},
               "GAN":   {"MSE": 0.005758, "FID": 9.477},
               "CVAE":  {"MSE": 0.006484, "FID": 25.782}},
}

ffhq_fid = {
    "small":  {"PConv": {"MSE": 0.000197, "FID": 0.704},
               "GAN":   {"MSE": 0.000241, "FID": 1.052},
               "CVAE":  {"MSE": 0.000264, "FID": 1.457}},
    "medium": {"PConv": {"MSE": 0.001968, "FID": 3.501},
               "GAN":   {"MSE": 0.002169, "FID": 4.266},
               "CVAE":  {"MSE": 0.002412, "FID": 8.537}},
    "large":  {"PConv": {"MSE": 0.006488, "FID": 21.827},
               "GAN":   {"MSE": 0.006927, "FID": 13.554},
               "CVAE":  {"MSE": 0.007542, "FID": 27.823}},
}

celebahq_lpips = {
    "small":  {"PConv": {"MSE": 0.000166, "LPIPS": 0.0055},
               "GAN":   {"MSE": 0.000197, "LPIPS": 0.0069},
               "CVAE":  {"MSE": 0.000229, "LPIPS": 0.0098}},
    "medium": {"PConv": {"MSE": 0.001633, "LPIPS": 0.0338},
               "GAN":   {"MSE": 0.001789, "LPIPS": 0.0394},
               "CVAE":  {"MSE": 0.002054, "LPIPS": 0.0635}},
    "large":  {"PConv": {"MSE": 0.005389, "LPIPS": 0.0907},
               "GAN":   {"MSE": 0.005758, "LPIPS": 0.1033},
               "CVAE":  {"MSE": 0.006484, "LPIPS": 0.1647}},
}

ffhq_lpips = {
    "small":  {"PConv": {"MSE": 0.000197, "LPIPS": 0.0063},
               "GAN":   {"MSE": 0.000241, "LPIPS": 0.0082},
               "CVAE":  {"MSE": 0.000264, "LPIPS": 0.0102}},
    "medium": {"PConv": {"MSE": 0.001968, "LPIPS": 0.0396},
               "GAN":   {"MSE": 0.002169, "LPIPS": 0.0465},
               "CVAE":  {"MSE": 0.002412, "LPIPS": 0.0664}},
    "large":  {"PConv": {"MSE": 0.006488, "LPIPS": 0.1053},
               "GAN":   {"MSE": 0.006927, "LPIPS": 0.1209},
               "CVAE":  {"MSE": 0.007542, "LPIPS": 0.1722}},
}


# ── helper ────────────────────────────────────────────────────────────────────
def draw_panel(ax, data_dict, regime, y_key, y_label, show_ylabel):
    pts = [(m, data_dict[regime][m]["MSE"], data_dict[regime][m][y_key])
           for m in methods]
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]

    x_span = max(xs) - min(xs) or xs[0] * 0.05
    y_span = max(ys) - min(ys) or ys[0] * 0.05

    x_left  = min(xs) - x_span * 0.6
    x_right = max(xs) + x_span * 4.0
    y_bot   = min(ys) - y_span * 0.9
    y_top   = max(ys) + y_span * 0.9
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bot,  y_top)

    x_off = (x_right - x_left) * 0.04   # label gap = 4 % of axis width

    # stagger labels vertically by y-rank to avoid overlap
    sorted_by_y = sorted(pts, key=lambda p: p[2])
    dy_map = {
        sorted_by_y[0][0]: -y_span * 0.20,   # lowest  → nudge down
        sorted_by_y[1][0]:  0,                # middle  → stay
        sorted_by_y[2][0]: +y_span * 0.20,   # highest → nudge up
    }

    for method, mse, y in pts:
        ax.scatter(mse, y, color=colors[method], s=240, zorder=3,
                   edgecolors="white", linewidths=2.2)
        if y_key == "LPIPS":
            val = f"{y:.4f}"
        elif y < 10:
            val = f"{y:.3f}"
        else:
            val = f"{y:.1f}"
        ax.text(mse + x_off, y + dy_map[method], f"{method}   {val}",
                fontsize=11, color=colors[method], fontweight="semibold",
                ha="left", va="center")

    ax.set_xlabel("Distortion (MSE ↓)", fontsize=10, labelpad=5)
    if show_ylabel:
        ax.set_ylabel(y_label, fontsize=10, labelpad=5)
    ax.set_title(f"{regime.capitalize()} masks", fontsize=12, fontweight="bold", pad=9)
    ax.tick_params(labelsize=9)

    if y_key == "LPIPS":
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    else:
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.1f}" if v >= 10 else f"{v:.2f}"))


# ── figures ───────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")

import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

panels = [
    (celebahq_fid,   "FID",   "Perception  (FID ↓)",
     "Perception–Distortion Trade-off  —  CelebA-HQ", FIG_DIR / "fig_celebahq_fid.png"),
    (ffhq_fid,       "FID",   "Perception  (FID ↓)",
     "Perception–Distortion Trade-off  —  FFHQ",      FIG_DIR / "fig_ffhq_fid.png"),
    (celebahq_lpips, "LPIPS", "Perceptual Quality  (LPIPS ↓)",
     "MSE vs. LPIPS  —  CelebA-HQ",                   FIG_DIR / "fig_celebahq_lpips.png"),
    (ffhq_lpips,     "LPIPS", "Perceptual Quality  (LPIPS ↓)",
     "MSE vs. LPIPS  —  FFHQ",                        FIG_DIR / "fig_ffhq_lpips.png"),
]

for data_dict, y_key, y_label, title, fname in panels:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    for i, regime in enumerate(regimes):
        draw_panel(axes[i], data_dict, regime, y_key, y_label,
                   show_ylabel=(i == 0))

    # shared legend below the figure
    handles = [mpatches.Patch(color=colors[m], label=m) for m in methods]
    fig.legend(handles=handles, title="Method", ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.06),
               fontsize=10, title_fontsize=10,
               edgecolor="#cccccc", framealpha=0.92)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
