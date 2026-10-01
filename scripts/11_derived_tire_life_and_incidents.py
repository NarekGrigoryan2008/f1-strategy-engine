"""
Builds three required derived tables that weren't tied to one of the four main
build stages, but were specified up front as required outputs:

1. Tire life limits: per (circuit, compound), the median/p90/max real stint
   length drivers actually ran before pitting.
2. Driver tire management: per (driver, circuit, compound), how that driver's
   fitted degradation slope compares to the circuit-wide baseline from Stage 1.
3. Driver incidents log: DNF reasons (high confidence), track-limit lap
   deletions (high confidence), and race-control messages tied to a specific
   car via RacingNumber (medium confidence - free text).

Each is genuinely shaped around what varies (one row per stint-length-bucket-
group / one row per driver-circuit-compound / one row per incident) rather than
one row per circuit with everything crammed into columns, per the spec.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data_loading import load_all_laps, load_all_race_control, load_all_results
from src.paths import DERIVED_DIR
from src.stage1_degradation import clean_laps_for_degradation_fit

MIN_LAPS_FOR_DRIVER_FIT = 15  # a driver needs at least this many clean laps on a
                              # (circuit, compound) to trust a per-driver slope


def build_tire_life_limits(laps: pd.DataFrame) -> pd.DataFrame:
    stints = laps.dropna(subset=["Stint"]).copy()
    # One row per (circuit, year, round, driver, stint): its length in laps, and
    # whether it ended in a real pit stop (PitInTime on its last lap) vs was still
    # running when the race ended (censored - excluded, since an unfinished stint
    # doesn't tell us how long the driver was WILLING to run before pitting).
    grouped = stints.groupby(["circuit", "year", "round", "Driver", "Stint"])
    rows = []
    for (circuit, year, round_number, driver, stint), g in grouped:
        g = g.sort_values("LapNumber")
        compound = g["Compound"].mode().iloc[0] if not g["Compound"].mode().empty else g["Compound"].iloc[0]
        ended_in_pit = g["PitInTime"].notna().any()
        rows.append({
            "circuit": circuit, "year": year, "round": round_number, "driver": driver,
            "stint": stint, "compound": compound, "stint_length_laps": len(g),
            "ended_in_pit_stop": ended_in_pit,
        })
    stint_df = pd.DataFrame(rows)
    real_stints = stint_df[stint_df["ended_in_pit_stop"]]  # exclude the race-ending, censored stint

    summary = (
        real_stints.groupby(["circuit", "compound"])["stint_length_laps"]
        .agg(median_stint_laps="median",
             p90_stint_laps=lambda s: s.quantile(0.9),
             max_stint_laps="max",
             n_stints="count")
        .reset_index()
    )
    return summary.sort_values(["circuit", "compound"])


def build_driver_tire_management(clean_laps: pd.DataFrame, baseline_model: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline_lookup = {(r["circuit"], r["compound"]): r["slope"] for _, r in baseline_model.iterrows()}
    for (driver, circuit, compound), g in clean_laps.groupby(["Driver", "circuit", "Compound"]):
        baseline_slope = baseline_lookup.get((circuit, compound))
        if baseline_slope is None:
            continue
        if len(g) < MIN_LAPS_FOR_DRIVER_FIT:
            rows.append({"driver": driver, "circuit": circuit, "compound": compound,
                         "n_laps": len(g), "driver_slope": np.nan,
                         "baseline_slope": baseline_slope, "delta_vs_baseline": np.nan,
                         "reliable": False})
            continue
        x = g["TyreLife"].to_numpy(dtype=float)
        y = g["LapTimeDelta"].to_numpy(dtype=float)
        slope, _ = np.polyfit(x, y, 1)
        rows.append({"driver": driver, "circuit": circuit, "compound": compound,
                     "n_laps": len(g), "driver_slope": slope,
                     "baseline_slope": baseline_slope, "delta_vs_baseline": slope - baseline_slope,
                     "reliable": True})
    return pd.DataFrame(rows).sort_values(["circuit", "compound", "driver"])


def build_incidents_log(laps: pd.DataFrame, results: pd.DataFrame, race_control: pd.DataFrame) -> pd.DataFrame:
    incidents = []

    # 1. DNF / retirement reasons (high confidence)
    non_classified_ok = {"Finished"}
    dnf = results[~results["Status"].isin(non_classified_ok)
                  & ~results["Status"].str.contains(r"^\+\d+ Lap", regex=True, na=False)
                  & (results["Status"] != "Lapped")]
    for _, r in dnf.iterrows():
        incidents.append({
            "circuit": r["circuit"], "year": r["year"], "round": r["round"],
            "driver": r["Abbreviation"], "incident_type": "dnf_status",
            "detail": r["Status"], "lap": None, "confidence": "high",
        })

    # 2. Track-limit lap deletions (high confidence)
    deleted = laps[laps["Deleted"] == True]  # noqa: E712
    for _, r in deleted.iterrows():
        incidents.append({
            "circuit": r["circuit"], "year": r["year"], "round": r["round"],
            "driver": r["Driver"], "incident_type": "lap_deleted",
            "detail": r.get("DeletedReason", None), "lap": r["LapNumber"], "confidence": "high",
        })

    # 3. Race control messages tied to a specific car via RacingNumber (medium
    # confidence - free text, and a car number needs mapping back to a driver
    # code per race since numbers aren't globally unique across drivers/years).
    driver_number_lookup = laps.drop_duplicates(["circuit", "year", "round", "DriverNumber"]).set_index(
        ["circuit", "year", "round", "DriverNumber"])["Driver"].to_dict()
    rc_with_car = race_control.dropna(subset=["RacingNumber"])
    for _, r in rc_with_car.iterrows():
        key = (r["circuit"], r["year"], r["round"], r["RacingNumber"])
        driver = driver_number_lookup.get(key)
        incidents.append({
            "circuit": r["circuit"], "year": r["year"], "round": r["round"],
            "driver": driver, "incident_type": "race_control_message",
            "detail": r["Message"], "lap": r.get("Lap", None), "confidence": "medium",
        })

    return pd.DataFrame(incidents)


def main():
    print("Loading data...")
    laps = load_all_laps()
    results = load_all_results()
    race_control = load_all_race_control()

    print("\n=== Tire life limits ===")
    tire_life = build_tire_life_limits(laps)
    tire_life.to_csv(DERIVED_DIR / "tire_life_limits.csv", index=False)
    print(f"{len(tire_life)} (circuit, compound) rows -> data/derived/tire_life_limits.csv")
    print(tire_life.head(10).to_string())

    print("\n=== Driver tire management ===")
    clean = clean_laps_for_degradation_fit(laps)
    baseline_model = pd.read_csv(DERIVED_DIR.parent / "models" / "degradation_model.csv")
    driver_mgmt = build_driver_tire_management(clean, baseline_model)
    driver_mgmt.to_csv(DERIVED_DIR / "driver_tire_management.csv", index=False)
    n_reliable = driver_mgmt["reliable"].sum()
    print(f"{len(driver_mgmt)} (driver, circuit, compound) rows, {n_reliable} reliable "
          f"(>= {MIN_LAPS_FOR_DRIVER_FIT} laps) -> data/derived/driver_tire_management.csv")
    best = driver_mgmt[driver_mgmt["reliable"]].nsmallest(5, "delta_vs_baseline")
    print("Best tire management (most negative delta = degrades slower than baseline):")
    print(best.to_string())

    print("\n=== Driver incidents log ===")
    incidents = build_incidents_log(laps, results, race_control)
    incidents.to_csv(DERIVED_DIR / "driver_incidents_log.csv", index=False)
    print(f"{len(incidents)} incidents -> data/derived/driver_incidents_log.csv")
    print(incidents["incident_type"].value_counts().to_string())
    print(incidents["confidence"].value_counts().to_string())


if __name__ == "__main__":
    main()
