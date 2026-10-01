"""
Stage 1 - tire degradation model (lap_time = slope * tire_age + intercept, fit per
circuit per compound) and pit-loss model (seconds lost to a pit stop, per circuit).

Deliberately a simple straight-line fit, not a curve or a neural net, per the
project spec - the point is a model simple enough to explain and sanity-check by
eye, with R2 reported honestly so a bad fit is visible as a bad fit rather than
hidden behind a flexible model that can always find some fit.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import OUTLIER_STD_DEVS

MIN_LAPS_FOR_FIT = 5  # need at least a few points for a 2-parameter fit + R^2 to mean anything


def clean_laps_for_degradation_fit(laps: pd.DataFrame) -> pd.DataFrame:
    """Apply the Stage 1 cleaning rules from the project spec, in order, and return
    the surviving laps (one row per driver-lap) ready to fit degradation lines on.
    Adds a `LapTimeDelta` column: this lap's time minus that RACE's own median
    clean-lap time (see the module docstring's note on why raw seconds don't work).

    Rules (all from the spec, applied in this order):
      1. green-flag only: TrackStatus must be exactly "1" for the whole lap
      2. drop pit in-laps and out-laps
      3. drop the opening lap of the race (LapNumber == 1)
      4. keep only laps FastF1 itself flags as timing-consistent (IsAccurate)
      5. drop statistical outliers per (circuit, compound) group (> OUTLIER_STD_DEVS
         standard deviations from that group's mean LAP TIME DELTA, not raw seconds)
    """
    df = laps.copy()
    n0 = len(df)

    df = df[df["TrackStatus"] == "1"]
    n1 = len(df)

    df = df[df["PitInTime"].isna() & df["PitOutTime"].isna()]
    n2 = len(df)

    df = df[df["LapNumber"] != 1]
    n3 = len(df)

    df = df[df["IsAccurate"] == True]  # noqa: E712 (explicit True comparison reads clearer here)
    n4 = len(df)

    df = df[df["LapTimeSeconds"].notna() & df["Compound"].notna() & df["TyreLife"].notna()]
    n5 = len(df)

    # Normalize each lap to its own race's pace level BEFORE fitting anything.
    # Confirmed with the user on 2026-09-22: pooling raw lap times across
    # 2018-2026 conflates tire degradation with cross-year confounds (the 2022
    # regulation change, general car development, fuel load) that are much larger
    # than the tire-wear signal itself - on the partial dataset this made the
    # pooled-raw-seconds fit's R^2 come out near zero even where a real
    # degradation effect should be visible. The per-race median is computed from
    # this SAME cleaned population (green-flag, no pit in/out, no lap 1,
    # IsAccurate), across all compounds/drivers in that race, as "this race's
    # typical clean pace."
    race_reference = df.groupby(["circuit", "year"])["LapTimeSeconds"].transform("median")
    df["LapTimeDelta"] = df["LapTimeSeconds"] - race_reference

    # outlier removal per (circuit, compound) group, on the normalized delta
    stats = df.groupby(["circuit", "Compound"])["LapTimeDelta"].transform(
        lambda s: (s - s.mean()).abs() / s.std(ddof=0)
    )
    df = df[stats.fillna(0) <= OUTLIER_STD_DEVS]
    n6 = len(df)

    print(f"  clean_laps_for_degradation_fit: {n0} -> {n1} (green flag) -> {n2} (no pit in/out) "
          f"-> {n3} (no lap 1) -> {n4} (IsAccurate) -> {n5} (no NaN) -> {n6} (no outliers)")

    return df


def compute_circuit_reference_laptimes(clean_laps: pd.DataFrame) -> pd.DataFrame:
    """A per-circuit TYPICAL reference lap time (median of each race's own median
    clean lap, across all pulled years) - a fallback for callers of
    expected_laptime() who don't have a live reference (e.g. this race's own
    practice/qualifying pace) to hand. Not a spec-named deliverable table, just a
    convenience default; using it instead of a live reference will wash out
    cross-year pace differences again, so it should be treated as a rough default,
    not a precise prediction."""
    per_race = clean_laps.groupby(["circuit", "year"])["LapTimeSeconds"].median().reset_index()
    return (
        per_race.groupby("circuit")["LapTimeSeconds"]
        .agg(typical_reference_laptime="median", n_races="count")
        .reset_index()
    )


@dataclass
class DegradationModel:
    """One fitted (circuit, compound) tire degradation line, fit on lap-time DELTA
    from that race's own median clean lap (see clean_laps_for_degradation_fit)."""
    circuit: str
    compound: str
    n_laps: int
    slope: float       # seconds/lap of tire age
    intercept: float    # seconds, delta at tire_age = 0
    r_squared: float


def fit_degradation_models(clean_laps: pd.DataFrame) -> pd.DataFrame:
    """Fit lap_time_delta = slope * tire_age + intercept per (circuit, compound)
    group, where lap_time_delta is this lap's time minus that race's own median
    clean lap time (added by clean_laps_for_degradation_fit). Groups with fewer
    than MIN_LAPS_FOR_FIT laps are skipped (reported, not silently dropped) since a
    2-parameter fit to a handful of points isn't meaningful."""
    rows = []
    skipped = []
    for (circuit, compound), g in clean_laps.groupby(["circuit", "Compound"]):
        if len(g) < MIN_LAPS_FOR_FIT:
            skipped.append((circuit, compound, len(g)))
            continue
        x = g["TyreLife"].to_numpy(dtype=float)
        y = g["LapTimeDelta"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rows.append({
            "circuit": circuit, "compound": compound, "n_laps": len(g),
            "slope": slope, "intercept": intercept, "r_squared": r_squared,
        })
    if skipped:
        print(f"  Skipped {len(skipped)} (circuit, compound) groups with < {MIN_LAPS_FOR_FIT} "
              f"clean laps (not enough data for a meaningful fit): {skipped}")
    return pd.DataFrame(rows).sort_values(["circuit", "compound"]).reset_index(drop=True)


def expected_laptime(circuit: str, compound: str, tire_age: float, reference_laptime: float,
                      model_df: pd.DataFrame) -> float:
    """The Stage 1 deliverable: expected lap time (seconds) for a given circuit,
    compound, and tire age (laps), from the fitted degradation model.

    `reference_laptime` is the caller-supplied "current pace level" for this
    specific race (e.g. from practice/qualifying, or an early clean lap) - the
    model predicts a DELTA from that baseline due to tire age, not an absolute lap
    time from a historical average, because pace level varies by year/car/track
    evolution far more than tire wear does (see clean_laps_for_degradation_fit).
    If no live reference is available, compute_circuit_reference_laptimes() gives a
    rough historical fallback."""
    row = model_df[(model_df["circuit"] == circuit) & (model_df["compound"] == compound)]
    if row.empty:
        raise KeyError(f"No fitted degradation model for circuit={circuit!r}, compound={compound!r}")
    r = row.iloc[0]
    return float(reference_laptime + r["slope"] * tire_age + r["intercept"])


def compute_pit_loss(all_laps: pd.DataFrame) -> pd.DataFrame:
    """Pit-loss per circuit, from real in-lap/out-lap time deltas against a
    reference (median clean green-flag lap time at that circuit), using the MEDIAN
    across all real stops (robust to the handful of stops that happen under a
    Safety Car, where the loss is artificially smaller since the whole field is
    running slowly anyway).

    Method per stop: pit_loss = (in_lap_time + out_lap_time) - 2 * reference_lap_time
    where reference_lap_time is that circuit's median green-flag racing lap time
    (computed the same way as the degradation fit's cleaned laps, i.e. excluding
    pit laps themselves, so the reference isn't contaminated by the very thing
    being measured against it).

    SC/VSC-affected stops are explicitly EXCLUDED, not just left for the median to
    average out. Checking this empirically (on the Imola data available while the
    full backfill was still running) showed why: a stop made during a Safety Car
    doesn't come out artificially SMALLER by this formula, as might be assumed -
    it comes out LARGER, because the in-lap/out-lap times themselves are slowed by
    the SC/VSC delta while they're still being compared against a green-flag-pace
    reference. On the Imola sample, non-green-flagged stops had a median apparent
    loss of ~61s vs ~33s for green-flag stops - nearly double - and made up ~19%
    of stops, enough to meaningfully drag the overall median up rather than average
    out. The spec's underlying intent (don't let a handful of SC stops distort the
    pit-loss estimate) is honored by filtering them out directly using TrackStatus,
    which is more reliable here than trusting the median's robustness alone.
    """
    reference = (
        all_laps[(all_laps["TrackStatus"] == "1")
                  & all_laps["PitInTime"].isna() & all_laps["PitOutTime"].isna()
                  & (all_laps["IsAccurate"] == True)  # noqa: E712
                  & all_laps["LapTimeSeconds"].notna()]
        .groupby("circuit")["LapTimeSeconds"].median()
    )

    rows = []
    for circuit, g in all_laps.groupby("circuit"):
        ref = reference.get(circuit)
        if ref is None or np.isnan(ref):
            continue
        in_laps = g[g["PitInTime"].notna()][["Driver", "year", "Stint", "LapNumber", "LapTimeSeconds", "TrackStatus"]]
        out_laps = g[g["PitOutTime"].notna()][["Driver", "year", "Stint", "LapNumber", "LapTimeSeconds", "TrackStatus"]]
        # An out-lap is the first lap of the NEXT stint; pair each in-lap with the
        # out-lap immediately following it for the same driver in the same race.
        losses = []
        n_excluded_sc = 0
        for _, in_row in in_laps.iterrows():
            candidate = out_laps[
                (out_laps["Driver"] == in_row["Driver"])
                & (out_laps["year"] == in_row["year"])
                & (out_laps["LapNumber"] == in_row["LapNumber"] + 1)
            ]
            if candidate.empty:
                continue
            out_row = candidate.iloc[0]
            out_time = out_row["LapTimeSeconds"]
            in_time = in_row["LapTimeSeconds"]
            if pd.isna(in_time) or pd.isna(out_time):
                continue
            if in_row["TrackStatus"] != "1" or out_row["TrackStatus"] != "1":
                n_excluded_sc += 1
                continue
            losses.append((in_time + out_time) - 2 * ref)
        if len(losses) >= 3:  # need a handful of real green-flag stops to trust a median
            rows.append({
                "circuit": circuit,
                "pit_loss_seconds": float(np.median(losses)),
                "n_stops": len(losses),
                "n_excluded_sc_vsc_stops": n_excluded_sc,
                "reference_laptime_seconds": float(ref),
            })
    return pd.DataFrame(rows).sort_values("circuit").reset_index(drop=True)


def pit_loss(circuit: str, pit_loss_df: pd.DataFrame) -> float:
    """The Stage 1 deliverable: expected time (seconds) lost to a pit stop at a
    given circuit, from the computed pit-loss table."""
    row = pit_loss_df[pit_loss_df["circuit"] == circuit]
    if row.empty:
        raise KeyError(f"No pit-loss estimate for circuit={circuit!r}")
    return float(row.iloc[0]["pit_loss_seconds"])
