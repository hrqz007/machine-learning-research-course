from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent


def box(ax, x, y, text, color="#d9eafa", w=2.3, h=0.8):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#34658a",
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)


def arrow(ax, a, b, text="", color="#34658a"):
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", color=color, lw=2))
    ax.text(
        (a[0] + b[0]) / 2,
        (a[1] + b[1]) / 2 + 0.12,
        text,
        ha="center",
        fontsize=9,
        color=color,
    )


for n in [97]:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set(xlim=(0, 11), ylim=(0, 4))
    ax.axis("off")
    if n == 93:
        for y, label in [(2.6, "Frozen"), (0.6, "Fine-tune")]:
            ax.text(0.1, y + 0.4, label, fontsize=12, va="center")
            box(ax, 2, y, "Image", w=1.4)
            box(ax, 4.3, y, "Encoder", "#e6e6e6" if y > 2 else "#d9eafa")
            box(ax, 7.4, y, "New head", "#def1df")
            arrow(ax, (3.5, y + 0.4), (4.2, y + 0.4))
            arrow(ax, (6.7, y + 0.4), (7.3, y + 0.4))
            arrow(ax, (9.6, y - 0.2), (7.4, y - 0.2), "head gradient", "#34834f")
        arrow(ax, (7.2, 0.3), (4.4, 0.3), "encoder gradient", "#34834f")
        ax.text(5.4, 2.2, "No parameter update", ha="center", color="#a45e19")
    elif n == 94:
        box(ax, 0.3, 1.5, "Training image", w=2)
        box(ax, 3.5, 2.6, "View A + noise")
        box(ax, 3.5, 0.4, "View B + noise")
        arrow(ax, (2.4, 2), (3.4, 3))
        arrow(ax, (2.4, 1.7), (3.4, 0.8))
        box(ax, 7, 1.5, "Shared encoder\n+ projector")
        arrow(ax, (5.9, 3), (7, 2.2))
        arrow(ax, (5.9, 0.8), (7, 1.7))
        ax.text(
            7.9,
            0.65,
            "Positive: same source\nNegatives: other instances",
            ha="center",
            fontsize=11,
            color="#a45e19",
        )
    elif n == 95:
        for x, t in [
            (0.2, "Predeclare\nquestion"),
            (3, "Train all\n15 runs"),
            (5.8, "Pair by seed\nCNN - MLP"),
            (8.6, "Report\nlimits"),
        ]:
            box(ax, x, 1.7, t, w=2)
        for x in [2.3, 5.1, 7.9]:
            arrow(ax, (x, 2.1), (x + 0.6, 2.1))
        ax.text(
            5.4,
            0.65,
            "Same data + updates; different parameters and compute\nKeep all seeds, errors, curves and timings",
            ha="center",
            fontsize=12,
            color="#a45e19",
        )
    elif n == 96:
        box(ax, 0.4, 1.7, "IDs\n[cat, cat, bus]", w=2)
        box(ax, 4, 1.7, "Shared table E", w=2.5)
        box(ax, 8, 1.7, "3 vectors", w=2)
        arrow(ax, (2.5, 2.1), (3.9, 2.1), "lookup")
        arrow(ax, (6.6, 2.1), (7.9, 2.1))
        arrow(ax, (8, 1.3), (5, 1.3), "backward: sum repeated rows", "#34834f")
        ax.text(
            5.5,
            0.45,
            "cat row gets 2 contributions; bus row gets 1",
            ha="center",
            fontsize=12,
        )
    else:
        for x, t in [(1, "h1"), (4, "h2"), (7, "h3")]:
            box(ax, x, 1.7, t, w=1.5)
            ax.text(x + 0.75, 3.25, "x" + t[-1], ha="center")
            arrow(ax, (x + 0.75, 3), (x + 0.75, 2.6))
        arrow(ax, (2.6, 2.1), (3.9, 2.1), "same W")
        arrow(ax, (5.6, 2.1), (6.9, 2.1), "same W")
        arrow(ax, (8.6, 2.1), (10, 2.1), "loss")
        ax.axvline(6.2, ymin=0.2, ymax=0.7, color="#c97c20", ls="--")
        ax.text(
            6.2,
            0.5,
            "Optional detach boundary\nvalues pass; gradients stop",
            ha="center",
            fontsize=11,
            color="#a45e19",
        )
        arrow(
            ax,
            (8.5, 1.3),
            (1.7, 1.3),
            "Full BPTT: accumulate shared-weight terms",
            "#34834f",
        )
    fig.tight_layout()
    fig.savefig(ROOT / "figures/03_mechanism.png", dpi=180)
    plt.close(fig)
