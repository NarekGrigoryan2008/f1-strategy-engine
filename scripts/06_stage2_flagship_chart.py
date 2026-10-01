"""
The flagship chart the project spec calls out explicitly: gap-at-pitting vs.
tire-age delta between the two cars, colored by undercut/overcut success/fail.
Built from the full clean (non-SC-affected) attempt set.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.paths import CHARTS_DIR, DERIVED_DIR

# Status colors (success/fail), not categorical identity - green/red is the
# conventional, immediately-readable choice for a binary outcome like this, kept
# distinguishable in lightness too (not relying on hue alone) and given a shape
# difference as a secondary encoding for colorblind readers.
SUCCESS_COLOR = "#1B7F3E"
FAIL_COLOR = "#C0362C"


def main():
    df = pd.read_csv(DERIVED_DIR / "undercut_overcut_attempts.csv")
    clean = df[~df["sc_vsc_affected"]].dropna(subset=["tire_age_a", "tire_age_b"]).copy()
    clean["tire_age_delta"] = clean["tire_age_b"] - clean["tire_age_a"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for success, color, marker, label in [
        (1, SUCCESS_COLOR, "o", "Success (pitting car ahead at measurement lap)"),
        (0, FAIL_COLOR, "x", "Fail"),
    ]:
        g = clean[clean["success"] == success]
        ax.scatter(g["gap_seconds"], g["tire_age_delta"], s=22, alpha=0.55,
                   color=color, marker=marker, linewidths=1.0, label=f"{label} (n={len(g)})")

    ax.axvline(0, color="#999999", linewidth=0.8, zorder=0)
    ax.axhline(0, color="#999999", linewidth=0.8, zorder=0)
    ax.set_xlabel("Gap at pitting (s) - positive = pitting car (A) behind rival")
    ax.set_ylabel("Tire age delta (rival's age - A's age, laps) - positive = rival on older tires")
    ax.set_title("Undercut / overcut attempts: gap vs. tire-age advantage, colored by outcome\n"
                 f"{len(clean)} real attempts, 2018-2026, 20 circuits (Safety-Car-affected attempts excluded)")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#eeeeee", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_path = CHARTS_DIR / "flagship_gap_vs_tireage_by_outcome.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
