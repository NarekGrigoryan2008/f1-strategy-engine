"""
A second Stage 2 chart, complementing the flagship scatter: test-set accuracy of
the physics-only threshold vs. a majority-class baseline vs. the two fitted
models. This is the single clearest visual of the project's central empirical
finding - physics alone badly misjudges real outcomes; a simple model fit on
actual results does much better - so it's built as its own clean, uncluttered
chart rather than buried in a log table.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.paths import CHARTS_DIR

# Hand-entered from scripts/05_stage2_fit_model.py's printed output (2026-09-24
# run, after the Stage 2 revisit added team_pit_speed_relative and the
# gap_x_tire_age_b interaction term). On this single split LightGBM edges out
# logistic regression, but logistic regression stays the deployed model
# because it's the more STABLE one across the 3-fold expanding-window CV
# established in the same revisit.
RESULTS = [
    ("Physics threshold\n(Part A, no outcome data)", 0.294, "#C0362C"),
    ("Always predict\n\"success\" (baseline)", 0.713, "#8C8C8C"),
    ("Logistic regression\n(Part B - deployed)", 0.741, "#1B7F3E"),
    ("Gradient-boosted\n(LightGBM)", 0.748, "#4C7EA8"),
]


def main():
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    labels = [r[0] for r in RESULTS]
    values = [r[1] for r in RESULTS]
    colors = [r[2] for r in RESULTS]

    bars = ax.bar(labels, values, color=colors, width=0.6, zorder=2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.1%}",
                ha="center", va="bottom", fontsize=11)

    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Accuracy on held-out 2025-2026 attempts")
    ax.set_title("Predicting real undercut/overcut outcomes:\nphysics alone vs. models fit on actual results")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, labelsize=9)
    ax.set_yticks([])
    ax.grid(False)
    ax.axhline(0, color="#333333", linewidth=0.8, zorder=1)
    fig.tight_layout()

    out_path = CHARTS_DIR / "stage2_model_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
