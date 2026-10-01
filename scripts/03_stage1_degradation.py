"""
Stage 1 - fit the tire degradation model and the pit-loss model, and save
diagnostic scatter plots per circuit.

Can be re-run at any point as more races get pulled (it just loads whatever is
currently under data/raw/) - early runs during the backfill are explicitly a
smoke test of the pipeline logic, not the final numbers; re-run once the full
158-race pull finishes for the real result that the rest of the project builds on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_loading import list_pulled_races, load_all_laps
from src.paths import CHARTS_DIR, MODELS_DIR
from src.stage1_degradation import (
    clean_laps_for_degradation_fit, fit_degradation_models, compute_pit_loss,
    compute_circuit_reference_laptimes,
)

# Real F1 tire-compound colors - recognizable at a glance to anyone in the sport,
# which for this project's audience beats a generic categorical palette. Kept
# colorblind-considerations in mind: SOFT/MEDIUM/HARD (red/gold/gray) are the
# compounds that actually have enough clean-lap data to plot in most races, and
# they're distinguished by lightness as well as hue; line style is a secondary
# encoding on top of color for INTERMEDIATE/WET so identity never rests on color
# alone.
COMPOUND_STYLE = {
    "SOFT":         {"color": "#DA291C", "linestyle": "-"},
    "MEDIUM":       {"color": "#B8860B", "linestyle": "-"},
    "HARD":         {"color": "#4D4D4D", "linestyle": "-"},
    "INTERMEDIATE": {"color": "#43B02A", "linestyle": "--"},
    "WET":          {"color": "#0067B1", "linestyle": ":"},
}


def make_diagnostic_plot(circuit: str, clean_laps_circuit, model_df_circuit):
    fig, ax = plt.subplots(figsize=(7, 5))
    for compound, g in clean_laps_circuit.groupby("Compound"):
        style = COMPOUND_STYLE.get(compound, {"color": "#888888", "linestyle": "-"})
        ax.scatter(g["TyreLife"], g["LapTimeDelta"], s=10, alpha=0.35,
                   color=style["color"], label=None)
        row = model_df_circuit[model_df_circuit["compound"] == compound]
        if not row.empty:
            r = row.iloc[0]
            x_line = np.array([g["TyreLife"].min(), g["TyreLife"].max()])
            y_line = r["slope"] * x_line + r["intercept"]
            ax.plot(x_line, y_line, color=style["color"], linestyle=style["linestyle"],
                    linewidth=2,
                    label=f"{compound} (n={int(r['n_laps'])}, R2={r['r_squared']:.2f})")
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
    ax.set_xlabel("Tire age (laps)")
    ax.set_ylabel("Lap time delta from race median (s)")
    ax.set_title(f"Tire degradation - {circuit}")
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_path = CHARTS_DIR / f"degradation_{circuit}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    races = list_pulled_races()
    print(f"Loading laps from {len(races)} pulled races...")
    laps = load_all_laps()
    print(f"Loaded {len(laps)} total lap rows across {laps['circuit'].nunique()} circuits, "
          f"{laps[['circuit','year']].drop_duplicates().shape[0]} races.\n")

    print("Cleaning laps for degradation fit...")
    clean = clean_laps_for_degradation_fit(laps)
    print(f"\n{len(clean)} clean laps remain for fitting.\n")

    print("Fitting degradation models per (circuit, compound)...")
    model_df = fit_degradation_models(clean)
    model_path = MODELS_DIR / "degradation_model.csv"
    model_df.to_csv(model_path, index=False)
    print(f"\nSaved {len(model_df)} fitted (circuit, compound) models -> {model_path}\n")
    print(model_df.to_string())

    ref_df = compute_circuit_reference_laptimes(clean)
    ref_path = MODELS_DIR / "circuit_reference_laptimes.csv"
    ref_df.to_csv(ref_path, index=False)
    print(f"\nSaved circuit reference lap times (fallback for expected_laptime()) -> {ref_path}")
    print(ref_df.to_string())

    print("\nComputing pit-loss per circuit...")
    pit_loss_df = compute_pit_loss(laps)
    pit_loss_path = MODELS_DIR / "pit_loss_model.csv"
    pit_loss_df.to_csv(pit_loss_path, index=False)
    print(f"Saved pit-loss for {len(pit_loss_df)} circuits -> {pit_loss_path}\n")
    print(pit_loss_df.to_string())

    print("\nGenerating diagnostic plots...")
    for circuit, g in clean.groupby("circuit"):
        m = model_df[model_df["circuit"] == circuit]
        if m.empty:
            continue
        out_path = make_diagnostic_plot(circuit, g, m)
        print(f"  {circuit} -> {out_path}")

    # Flag bad fits explicitly, honestly, per the spec's "report R2 honestly" instruction
    bad_fits = model_df[model_df["r_squared"] < 0.3]
    if len(bad_fits):
        print(f"\n{len(bad_fits)} (circuit, compound) fits have R2 < 0.3 (weak linear fit):")
        print(bad_fits.to_string())


if __name__ == "__main__":
    main()
