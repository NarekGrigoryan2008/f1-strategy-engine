"""
Stage 4 - test empirically whether higher track temperature increases tire
degradation, rather than assuming it (per the project spec's explicit
instruction not to assume this).

Method: merge each clean lap (same cleaning as Stage 1) with the nearest weather
reading from that race via session time, then fit ONE pooled regression across
the whole dry-compound dataset (not per circuit/compound - Stage 1's per-group
samples are too thin to reliably fit a 4th parameter each) with circuit and
compound as additive fixed effects, plus a tire_age x track_temp INTERACTION
term - the interaction is the direct test of "does heat change the degradation
RATE," as opposed to the main effect of track_temp, which would only say hotter
tracks have different baseline pace (already partly absorbed by the circuit fixed
effect anyway).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import statsmodels.formula.api as smf

from src.data_loading import load_all_laps, load_all_weather
from src.stage1_degradation import clean_laps_for_degradation_fit


def attach_track_temp(clean_laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.copy()
    weather["TimeSeconds"] = pd.to_timedelta(weather["Time"], errors="coerce").dt.total_seconds()
    clean_laps = clean_laps.copy()
    clean_laps["TimeSeconds"] = pd.to_timedelta(clean_laps["Time"], errors="coerce").dt.total_seconds()

    merged_frames = []
    for (circuit, year, round_number), laps_g in clean_laps.groupby(["circuit", "year", "round"]):
        w_g = weather[(weather["circuit"] == circuit) & (weather["year"] == year)
                       & (weather["round"] == round_number)]
        if w_g.empty or laps_g["TimeSeconds"].isna().all():
            continue
        laps_g = laps_g.sort_values("TimeSeconds")
        w_g = w_g.sort_values("TimeSeconds")
        merged = pd.merge_asof(laps_g, w_g[["TimeSeconds", "TrackTemp", "AirTemp"]],
                                on="TimeSeconds", direction="nearest")
        merged_frames.append(merged)
    return pd.concat(merged_frames, ignore_index=True)


def main():
    print("Loading laps and weather...")
    laps = load_all_laps()
    weather = load_all_weather()

    print("Cleaning laps (Stage 1 rules)...")
    clean = clean_laps_for_degradation_fit(laps)

    dry_compounds = {"SOFT", "MEDIUM", "HARD", "SUPERSOFT", "ULTRASOFT", "HYPERSOFT"}
    clean_dry = clean[clean["Compound"].isin(dry_compounds)].copy()
    print(f"\n{len(clean_dry)} clean dry-compound laps (WET/INTERMEDIATE excluded - "
          f"already flagged in Stage 1 as confounded by track-drying, not a fair test of temperature).")

    print("Attaching nearest track temperature reading per lap...")
    merged = attach_track_temp(clean_dry, weather)
    merged = merged.dropna(subset=["TrackTemp", "TyreLife", "LapTimeDelta"])
    print(f"{len(merged)} laps with a matched temperature reading.\n")

    # Center track temp so the tire_age main-effect coefficient stays interpretable
    # at a "typical" temperature rather than at TrackTemp=0 (never observed).
    merged["track_temp_c"] = merged["TrackTemp"] - merged["TrackTemp"].mean()

    print("Fitting pooled OLS: LapTimeDelta ~ tire_age * track_temp_c + circuit + compound ...")
    model = smf.ols(
        "LapTimeDelta ~ TyreLife * track_temp_c + C(circuit) + C(Compound)",
        data=merged,
    ).fit()

    interaction_term = "TyreLife:track_temp_c"
    coef = model.params[interaction_term]
    pval = model.pvalues[interaction_term]
    se = model.bse[interaction_term]
    print(f"\n=== Result: does higher track temperature increase degradation RATE? ===")
    print(f"tire_age x track_temp interaction coefficient: {coef:+.5f} sec/lap/degC "
          f"(SE {se:.5f}, p={pval:.4f})")
    if pval < 0.05:
        direction = "INCREASES" if coef > 0 else "DECREASES"
        print(f"Statistically significant at p<0.05: higher track temperature {direction} "
              f"the tire-age degradation slope.")
    else:
        print("NOT statistically significant at p<0.05 - the data does not support a clear "
              "temperature effect on degradation RATE in this pooled test.")

    print(f"\nModel R2: {model.rsquared:.4f}  (n={int(model.nobs)})")
    print("\nFull coefficient table (fixed effects omitted from display for brevity; saved in full below):")
    print(model.params[["Intercept", "TyreLife", "track_temp_c", interaction_term]].to_string())

    out_path = Path(__file__).resolve().parent.parent / "data" / "derived" / "stage4_temperature_model_summary.txt"
    with open(out_path, "w") as f:
        f.write(str(model.summary()))
    print(f"\nFull regression summary saved -> {out_path}")


if __name__ == "__main__":
    main()
