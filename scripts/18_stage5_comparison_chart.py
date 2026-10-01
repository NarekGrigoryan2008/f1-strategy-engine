"""
Stage 5 chart - extends the Stage 2 physics-vs-models comparison with the Monte
Carlo simulator's real validation result, on the exact same 286 held-out
attempts. Shows BOTH the original (pre-revisit) and current (post-revisit)
Monte Carlo numbers: the honest point is not just "physics alone is worse than
a model fit on outcomes" but "a diagnosed, targeted fix (replacing a hard
physics cutoff with a calibrated probability) closed most - not all - of a
13.3-point gap to the simpler model, without beating it."
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.paths import CHARTS_DIR

# Hand-entered from scripts/17_stage5_validate.py's printed output (2026-09-24
# runs). Logistic regression's number reflects the Stage 2 revisit's updated
# 10-feature classifier (0.741 on this split - kept deployed anyway for its
# cross-validated stability). Monte Carlo "pre-revisit" is the original
# hard-cutoff simulator; "deployed" is Step 1 (calibrated outcome rule) - the
# configuration actually used by src/tool.py. Step 2 (driver-specific tire
# management) was tried and found to HURT accuracy (72.4% -> 70.3%) and is
# intentionally not shown as a separate bar here, since it isn't the deployed
# configuration.
RESULTS = [
    ("Physics threshold\n(Part A, no outcome data)", 0.294, "#C0362C"),
    ("Monte Carlo, pre-revisit\n(hard physics cutoff)", 0.608, "#D9A441"),
    ("Always predict\n\"success\" (baseline)", 0.713, "#8C8C8C"),
    ("Monte Carlo, deployed\n(Step 1: calibrated)", 0.724, "#B8860B"),
    ("Logistic regression\n(Part B - deployed)", 0.741, "#1B7F3E"),
    ("Gradient-boosted\n(LightGBM, Part B)", 0.748, "#4C7EA8"),
]


def main():
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    labels = [r[0] for r in RESULTS]
    values = [r[1] for r in RESULTS]
    colors = [r[2] for r in RESULTS]

    bars = ax.bar(labels, values, color=colors, width=0.62, zorder=2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.1%}",
                ha="center", va="bottom", fontsize=10.5)

    ax.set_ylim(0, 0.92)
    ax.set_ylabel("Accuracy on the same held-out 2025-2026 attempts")
    ax.set_title("A diagnosed fix closes most of the gap, not all of it:\nreplacing a hard physics cutoff with a calibrated probability, 60.8% -> 72.4%")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, labelsize=8.5)
    ax.set_yticks([])
    ax.grid(False)
    ax.axhline(0, color="#333333", linewidth=0.8, zorder=1)
    fig.tight_layout()

    out_path = CHARTS_DIR / "stage5_model_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
