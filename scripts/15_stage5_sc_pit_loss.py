"""
Stage 5 prerequisite - measure the REAL relative cost of pitting during an
SC/VSC period, from real data, rather than assuming a "pit stops are free under
SC" folk-wisdom discount.

Stage 1's pit_loss_model.csv deliberately EXCLUDED SC/VSC-affected stops,
because measuring them against a GREEN-FLAG reference lap overstates their cost
(the in/out laps are slowed by the SC delta, which has nothing to do with the
actual stop). But for the Monte Carlo simulator, what matters isn't the absolute
lap-time cost - it's the cost RELATIVE TO A RIVAL WHO STAYS OUT during the same
SC period (since that rival is also going slowly). So this measures each SC-
affected stop's cost against the median lap time of OTHER cars on the SAME lap,
in the SAME race, who were NOT pitting that lap - a same-conditions reference,
not a green-flag one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data_loading import load_all_laps
from src.paths import MODELS_DIR


def main():
    laps = load_all_laps()

    rows = []
    for (circuit, year, round_number), race in laps.groupby(["circuit", "year", "round"]):
        in_laps = race[race["PitInTime"].notna()]
        for _, in_row in in_laps.iterrows():
            lap = in_row["LapNumber"]
            out_row = race[(race["Driver"] == in_row["Driver"]) & (race["LapNumber"] == lap + 1)]
            if out_row.empty:
                continue
            out_row = out_row.iloc[0]
            in_time, out_time = in_row["LapTimeSeconds"], out_row["LapTimeSeconds"]
            if pd.isna(in_time) or pd.isna(out_time):
                continue
            sc_affected = (in_row["TrackStatus"] != "1") or (out_row["TrackStatus"] != "1")
            if not sc_affected:
                continue  # this script only cares about the SC-affected population

            # Reference: median lap time of OTHER cars at the SAME lap number, in
            # the SAME race, who did NOT pit that lap - i.e. cars experiencing the
            # same real (SC-slowed) conditions without stopping.
            others = race[(race["LapNumber"] == lap) & (race["Driver"] != in_row["Driver"])
                          & race["PitInTime"].isna() & race["PitOutTime"].isna()
                          & race["LapTimeSeconds"].notna()]
            if len(others) < 3:
                continue
            reference = others["LapTimeSeconds"].median()
            relative_loss = (in_time + out_time) - 2 * reference
            rows.append({"circuit": circuit, "year": year, "round": round_number,
                        "driver": in_row["Driver"], "lap": lap, "relative_loss": relative_loss})

    df = pd.DataFrame(rows)
    print(f"{len(df)} SC/VSC-affected stops with a valid same-conditions reference.\n")
    print(f"Global median relative pit-loss under SC/VSC: {df['relative_loss'].median():.2f}s "
          f"(mean {df['relative_loss'].mean():.2f}s, std {df['relative_loss'].std():.2f}s)")

    print("\nPer-circuit counts (checking whether per-circuit estimates would be reliable):")
    per_circuit = df.groupby("circuit")["relative_loss"].agg(["count", "median"])
    print(per_circuit.sort_values("count", ascending=False).to_string())
    print(f"\nCircuits with < 5 SC-affected stops: {(per_circuit['count'] < 5).sum()}/{len(per_circuit)}")

    # Compare directly against Stage 1's real GREEN-FLAG pit loss to quantify the
    # discount as a ratio, not just an absolute number.
    green_pit_loss = pd.read_csv(MODELS_DIR / "pit_loss_model.csv")
    global_green_median = green_pit_loss["pit_loss_seconds"].median()
    global_sc_median = df["relative_loss"].median()
    print(f"\nGlobal green-flag pit loss (median across circuits): {global_green_median:.2f}s")
    print(f"Global SC/VSC relative pit loss (this measurement): {global_sc_median:.2f}s")
    print(f"Ratio (SC cost / green cost): {global_sc_median/global_green_median:.2f}")

    out = pd.DataFrame([{
        "sc_relative_pit_loss_seconds": float(global_sc_median),
        "green_flag_pit_loss_seconds_median": float(global_green_median),
        "discount_ratio": float(global_sc_median / global_green_median),
        "n_stops": len(df),
    }])
    out_path = MODELS_DIR / "sc_pit_loss_discount.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
