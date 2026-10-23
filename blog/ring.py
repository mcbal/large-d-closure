"""The ring picture of a high-dimensional vMF, for the post's one-spin section.

(a) Sketch: at large D the samples of a vMF spin sit on a thin ring at a fixed
    latitude of the sphere. The mean sits at the ring's center, so mean, noise
    and spin form a right triangle, |m|^2 + |ds|^2 = R^2.
(b) The exact density of the alignment s.h/(R|h|) at fixed beta |h| / R. As D
    grows it concentrates at |m|/R, away from the mode at alignment 1.

    python ring.py      # writes ring.{png,svg}
"""

from pathlib import Path

import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch

import core as C

HERE = Path(__file__).resolve().parent

INK, MUTED, GRID = "#29313d", "#707783", "#d9d7d1"
ORANGE, GREEN, BLUE = "#d89170", "#78a58f", "#4f7fa8"


def arrow(ax, a, b, color, lw=1.6, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, **kw))


def sketch(ax, t):
    sin = np.sqrt(1 - t**2)
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color=MUTED, lw=1)
    ax.add_patch(Ellipse((0, 0), 2, 0.5, fill=False, color=GRID, lw=0.8, ls="--"))  # equator
    ax.add_patch(Ellipse((t, 0), 0.5 * sin, 2 * sin, fill=False, color=BLUE, lw=2.5, alpha=0.85))  # the ring
    arrow(ax, (-1.25, 0), (1.3, 0), MUTED, lw=0.8)
    ax.text(1.32, -0.02, "$\\hat{\\mathbf{h}}$", color=MUTED, fontsize=11, va="center")
    s = (t, sin)  # a sample at the top of the ring
    arrow(ax, (0, 0), (t, 0), INK)
    arrow(ax, (t, 0), s, ORANGE)
    arrow(ax, (0, 0), s, BLUE, lw=1.2, ls=(0, (4, 2)))
    ax.plot(*s, "o", color=BLUE, ms=5, zorder=5)
    ax.plot([t - 0.07, t - 0.07, t], [0, 0.07, 0.07], color=INK, lw=0.8)  # right angle
    ax.text(0.5 * t, -0.13, "$\\mathbf{m}$", color=INK, fontsize=11, ha="center")
    ax.text(t + 0.05, 0.45 * sin, "$\\delta\\mathbf{s}$", color=ORANGE, fontsize=11)
    ax.text(0.5 * t - 0.17, 0.5 * sin + 0.02, "$\\mathbf{s}$", color=BLUE, fontsize=11)
    ax.text(t + 0.2 * sin, -0.85 * sin, "typical samples", color=BLUE, fontsize=8.5)
    ax.text(0, -1.27, "$\\|\\mathbf{m}\\|^2 + \\|\\delta\\mathbf{s}\\|^2 = R^2$", color=INK, fontsize=10, ha="center")
    ax.set(aspect="equal", xlim=(-1.3, 1.5), ylim=(-1.4, 1.15))
    ax.axis("off")
    ax.set_title("(a) a ring, not a cap", fontsize=10, color=INK)


def alignment(ax, field, dims=(8, 64, 512)):
    t_star = field / (1 + np.sqrt(1 + field**2))
    t = np.linspace(-1, 1, 4001)[1:-1]
    for D, color in zip(dims, (ORANGE, GREEN, BLUE)):
        kappa = field * (D / 2 - 1)  # beta R |h| with |h| = field R / beta
        logp = kappa * t + 0.5 * (D - 3) * np.log1p(-t * t)
        ax.plot(t, np.exp(logp - logp.max()), color=color, lw=2, label=f"$D={D}$")
    ax.axvline(t_star, color=INK, ls=":", lw=1)
    ax.text(t_star - 0.03, 1.04, "$\\|\\mathbf{m}\\|/R$", color=INK, fontsize=9, ha="right")
    ax.plot(1, 0, "x", color=MUTED, ms=6, clip_on=False)
    ax.text(0.98, 0.07, "mode", color=MUTED, fontsize=8, ha="right")
    ax.set(xlim=(-1, 1), ylim=(0, 1.12), yticks=[], xticks=[-1, -0.5, 0, 0.5, 1],
           xlabel="alignment with the field, $\\mathbf{s}\\cdot\\hat{\\mathbf{h}}/R$", ylabel="density (scaled)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"(b) the ring sharpens as $D$ grows", fontsize=10, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.rcParams.update(C.PLOT_STYLE)
    field = 2.0
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
    sketch(axes[0], field / (1 + np.sqrt(1 + field**2)))
    alignment(axes[1], field)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"ring.{ext}", dpi=200)
