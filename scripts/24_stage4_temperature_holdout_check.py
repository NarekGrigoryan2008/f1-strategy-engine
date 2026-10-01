"""
Stage 4 revisit - an honest replication check, not a modeling upgrade: does the
tire_age x track_temp interaction found on the full 2018-2026 pooled sample
(+0.00111 sec/lap/degC, p<0.0001, n=132,883 - scripts/10_stage4_temperature_test.py)
replicate on the 2025-2026 races alone, which the original fit never saw in
isolation (they were part of the pooled sample, but a small fraction of it -
this checks whether the effect is visible in that slice on its own, adapting
Part 1's held-out validation logic to Stage 4 instead of adding any new
modeling complexity)?

Exact same model formula as the original test, restricted to year 2025-2026
laps only - not a different specification, so any difference in the result is
about the data slice, not a changed method.
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


def fit_and_report(merged: pd.DataFrame, label: str):
    model = smf.ols(
        "LapTimeDelta ~ TyreLife * track_temp_c + C(circuit) + C(Compound)",
        data=merged,
    ).fit()
    interaction_term = "TyreLife:track_temp_c"
    coef = model.params[interaction_term]
    pval = model.pvalues[interaction_term]
    se = model.bse[interaction_term]
    print(f"\n=== {label} ===")
    print(f"n={int(model.nobs)}, model R2={model.rsquared:.4f}")
    print(f"tire_age x track_temp interaction: {coef:+.5f} sec/lap/degC (SE {se:.5f}, p={pval:.4f})")
    if pval < 0.05:
        direction = "INCREASES" if coef > 0 else "DECREASES"
        print(f"Significant at p<0.05: higher track temperature {direction} the degradation slope.")
    else:
        print("NOT significant at p<0.05.")
    return coef, se, pval, int(model.nobs)


def main():
    print("Loading laps and weather...")
    laps = load_all_laps()
    weather = load_all_weather()

    print("Cleaning laps (Stage 1 rules)...")
    clean = clean_laps_for_degradation_fit(laps)

    dry_compounds = {"SOFT", "MEDIUM", "HARD", "SUPERSOFT", "ULTRASOFT", "HYPERSOFT"}
    clean_dry = clean[clean["Compound"].isin(dry_compounds)].copy()

    print("Attaching nearest track temperature reading per lap...")
    merged = attach_track_temp(clean_dry, weather)
    merged = merged.dropna(subset=["TrackTemp", "TyreLife", "LapTimeDelta"])

    # --- Original: full 2018-2026 pooled sample (reproduced for a side-by-side, not just quoted from memory) ---
    merged_full = merged.copy()
    merged_full["track_temp_c"] = merged_full["TrackTemp"] - merged_full["TrackTemp"].mean()
    full_coef, full_se, full_p, full_n = fit_and_report(
        merged_full, "ORIGINAL: full 2018-2026 pooled sample (reproduced)"
    )

    # --- Holdout check: 2025-2026 only, the years Stage 2's classifier holds out too ---
    merged_holdout = merged[merged["year"].isin([2025, 2026])].copy()
    if merged_holdout.empty:
        print("\nNo 2025-2026 dry-compound laps with a temperature reading - cannot run the holdout check.")
        return
    merged_holdout["track_temp_c"] = merged_holdout["TrackTemp"] - merged_holdout["TrackTemp"].mean()
    holdout_coef, holdout_se, holdout_p, holdout_n = fit_and_report(
        merged_holdout, "HOLDOUT CHECK: 2025-2026 laps only"
    )

    print("\n=== Side-by-side ===")
    print(f"{'Sample':<30}{'n':>10}{'coefficient':>16}{'SE':>10}{'p-value':>10}")
    print(f"{'Full 2018-2026':<30}{full_n:>10}{full_coef:>16.5f}{full_se:>10.5f}{full_p:>10.4f}")
    print(f"{'2025-2026 holdout only':<30}{holdout_n:>10}{holdout_coef:>16.5f}{holdout_se:>10.5f}{holdout_p:>10.4f}")

    same_sign = (full_coef > 0) == (holdout_coef > 0)
    print(f"\nSame direction as the original finding: {same_sign}")
    if holdout_p < 0.05 and same_sign:
        print("REPLICATES: the effect is directionally consistent and still statistically "
              "significant on data the original fit never saw in isolation.")
    elif same_sign:
        print("PARTIALLY REPLICATES: same direction, but not statistically significant on this "
              "smaller holdout slice alone - consistent with a real but small effect that needs "
              "a larger sample to detect cleanly, but not strong independent confirmation either.")
    else:
        print("DOES NOT REPLICATE: the holdout slice shows the opposite direction - the original "
              "full-sample finding may have been partly fitting noise in the pooled sample.")


if __name__ == "__main__":
    main()
