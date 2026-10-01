"""
Stage 2 revisit, Part 1 (new features) - builds each new candidate feature onto
the existing attempts table, one at a time, so each can be tested in isolation
before deciding whether it earns its place in the deployed model.

Feature 1: team pit-crew speed proxy. NOT the literal stationary "jacked up"
time - that's not cleanly derivable from what FastF1 exposes at the lap level
(checked: PitInTime/PitOutTime are pit-lane ENTRY/EXIT line crossings, and
race control messages don't report stop durations either - grepped for "PIT" in
the full race-control text and found flag/incident/penalty messages, nothing
resembling a stop-time report). What IS derivable: total pit-lane transit time
(entry to exit), which mixes a circuit-specific fixed component (driving
distance at the pit-lane speed limit) with the actual stop-duration component
we want. Proxy: for each real green-flag stop, transit time MINUS that
circuit's own median transit time (removing the fixed circuit component),
aggregated to a median per (team, year). Documented explicitly as a proxy for
crew speed dominated by, but not a pure isolation of, stop duration - not
pretended to be the literal thing asked for.

Feature 2: out-lap traffic. The existing a_in_traffic only captures traffic
BEFORE the stop; this captures whether the pitting car re-emerges into traffic
on its out-lap (lap_a + 1), which is mechanically closer to what actually
blocks a successful undercut (having to follow a backmarker out of the pits).

Feature 3 (no new data needed): gap_seconds * tire_age_b interaction term.

Feature 4: race progress as a fraction (lap_a / total_race_laps).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data_loading import load_all_laps
from src.paths import DERIVED_DIR


def build_team_pit_speed(laps: pd.DataFrame) -> pd.DataFrame:
    """Returns (team, year) -> relative_pit_speed_seconds (negative = faster
    than typical for that circuit; positive = slower)."""
    rows = []
    for (circuit, year, round_number), race in laps.groupby(["circuit", "year", "round"]):
        in_laps = race[race["PitInTime"].notna()]
        transit_times = []
        for _, in_row in in_laps.iterrows():
            lap = in_row["LapNumber"]
            out_row = race[(race["Driver"] == in_row["Driver"]) & (race["LapNumber"] == lap + 1)]
            if out_row.empty:
                continue
            out_row = out_row.iloc[0]
            if in_row["TrackStatus"] != "1" or out_row["TrackStatus"] != "1":
                continue  # green-flag stops only, consistent with Stage 1's pit_loss population
            in_t, out_t = in_row["PitInTimeSeconds"], out_row["PitOutTimeSeconds"]
            if pd.isna(in_t) or pd.isna(out_t):
                continue
            transit_times.append({
                "team": in_row["Team"], "driver": in_row["Driver"],
                "transit_seconds": out_t - in_t,
            })
        if not transit_times:
            continue
        t_df = pd.DataFrame(transit_times)
        circuit_median = t_df["transit_seconds"].median()
        t_df["relative_seconds"] = t_df["transit_seconds"] - circuit_median
        t_df["circuit"], t_df["year"] = circuit, year
        rows.append(t_df)

    all_stops = pd.concat(rows, ignore_index=True)
    team_speed = (
        all_stops.groupby(["team", "year"])["relative_seconds"]
        .agg(team_pit_speed_relative="median", n_stops="count")
        .reset_index()
    )
    return team_speed, all_stops


def build_out_lap_traffic(laps: pd.DataFrame, attempts: pd.DataFrame) -> pd.Series:
    """For each attempt, is there a THIRD car within 2.0s of driver_a at the
    OUT-lap (lap_a + 1)? Wider threshold than the pre-stop 1.0s a_in_traffic,
    since merging back into the field after a stop is a less precise moment
    than steady-state racing."""
    OUT_TRAFFIC_THRESHOLD_S = 2.0
    laps = laps.copy()
    laps["TimeSeconds"] = pd.to_timedelta(laps["Time"], errors="coerce").dt.total_seconds()

    results = []
    for (circuit, year, round_number), race in attempts.groupby(["circuit", "year", "round"]):
        race_laps = laps[(laps["circuit"] == circuit) & (laps["year"] == year) & (laps["round"] == round_number)]
        by_driver_lap = {(r.Driver, int(r.LapNumber)): r.TimeSeconds for r in race_laps.itertuples()}
        drivers_at_lap = race_laps.groupby("LapNumber")["Driver"].apply(list).to_dict()

        for idx, row in race.iterrows():
            out_lap = int(row["lap_a"]) + 1
            a_time = by_driver_lap.get((row["driver_a"], out_lap))
            if a_time is None or pd.isna(a_time):
                results.append((idx, np.nan))
                continue
            in_traffic = any(
                0 < abs(a_time - by_driver_lap.get((d, out_lap), np.nan)) <= OUT_TRAFFIC_THRESHOLD_S
                for d in drivers_at_lap.get(out_lap, [])
                if d not in (row["driver_a"], row["driver_b"])
                and pd.notna(by_driver_lap.get((d, out_lap), np.nan))
            )
            results.append((idx, int(in_traffic)))

    result_series = pd.Series(dict(results))
    return result_series.reindex(attempts.index)


def main():
    print("Loading laps and attempts...")
    laps = load_all_laps()
    attempts = pd.read_csv(DERIVED_DIR / "undercut_overcut_attempts.csv")

    print("\n=== Feature 1: team pit-crew speed proxy ===")
    team_speed, all_stops = build_team_pit_speed(laps)
    team_speed.to_csv(DERIVED_DIR / "team_pit_speed.csv", index=False)
    print(f"{len(team_speed)} (team, year) rows -> data/derived/team_pit_speed.csv")
    print(f"n_stops used: {len(all_stops)}")
    print("Fastest 5 (team, year) by median relative pit speed:")
    print(team_speed.sort_values("team_pit_speed_relative").head(5).to_string())
    print("Slowest 5:")
    print(team_speed.sort_values("team_pit_speed_relative").tail(5).to_string())

    # Need each attempt's team_a - look up driver_a's team from laps at that (circuit,year,round,lap_a)
    print("\nJoining team_a onto attempts...")
    team_lookup = laps.drop_duplicates(["circuit", "year", "round", "Driver"]).set_index(
        ["circuit", "year", "round", "Driver"])["Team"]
    attempts["team_a"] = attempts.apply(
        lambda r: team_lookup.get((r["circuit"], r["year"], r["round"], r["driver_a"]), None), axis=1
    )
    attempts = attempts.merge(team_speed, left_on=["team_a", "year"], right_on=["team", "year"], how="left")
    print(f"team_pit_speed_relative coverage: {attempts['team_pit_speed_relative'].notna().mean()*100:.1f}%")

    print("\n=== Feature 2: out-lap traffic ===")
    attempts["b_out_traffic"] = build_out_lap_traffic(laps, attempts)
    print(f"Coverage: {attempts['b_out_traffic'].notna().mean()*100:.1f}%")
    print(f"Rate: {attempts['b_out_traffic'].mean():.3f}")

    print("\n=== Feature 4: race progress fraction ===")
    race_lengths = laps.groupby(["circuit", "year", "round"])["LapNumber"].max().rename("total_race_laps")
    attempts = attempts.merge(race_lengths, on=["circuit", "year", "round"], how="left")
    attempts["race_progress"] = attempts["lap_a"] / attempts["total_race_laps"]
    print(attempts["race_progress"].describe())

    out_path = DERIVED_DIR / "undercut_overcut_attempts_enriched.csv"
    attempts.to_csv(out_path, index=False)
    print(f"\nSaved enriched attempts -> {out_path}")


if __name__ == "__main__":
    main()
