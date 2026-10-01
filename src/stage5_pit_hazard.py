"""
Stage 5, prerequisite - fit a real, data-driven distribution for "given a driver
is at tire age X on compound Y at lap Z, with gap G to the nearest rival, what's
the probability they pit on the very next lap?"

This directly replaces the old tool's assumption that a rival never pits during
the simulation's lookahead window (the known cause of the tool always
recommending the longest tested lookahead). Framed as a per-lap
HAZARD (P(pits on lap L+1 | still out at lap L)) rather than a fixed "P(pits
within K laps)" window, so it can be sampled lap-by-lap inside the Monte Carlo
simulator for any horizon, and so it uses every lap of the dataset as a data
point rather than just a fixed 1/2/3-lap slice of it.
"""
import numpy as np
import pandas as pd


def build_pit_hazard_dataset(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per (circuit, year, round, driver, lap) where the driver is out on
    track and it's meaningful to ask "do they pit on the NEXT lap" - i.e. every
    lap except each driver's last lap of the race (no next lap to check).

    Features: tire_age (TyreLife), compound, lap_number, gap_ahead_seconds,
    gap_behind_seconds (both signed-positive distances, NaN if no car within
    GAP_SEARCH_LIMIT_S - "no close rival" rather than a fabricated large number).
    Label: pits_next_lap (1 if PitInTime is set on lap+1, else 0 - including
    laps where the driver simply finishes the race without pitting again, which
    is a legitimate negative example for "did NOT pit on this specific next lap").
    """
    GAP_SEARCH_LIMIT_S = 10.0  # beyond this, a rival isn't really "close" for pit-timing purposes

    laps = laps.sort_values(["circuit", "year", "round", "Driver", "LapNumber"]).reset_index(drop=True)
    laps["TimeSeconds"] = pd.to_timedelta(laps["Time"], errors="coerce").dt.total_seconds()

    rows = []
    for (circuit, year, round_number), race in laps.groupby(["circuit", "year", "round"]):
        by_driver_lap = {(r.Driver, int(r.LapNumber)): r for r in race.itertuples()}
        pit_in_laps = set(
            (r.Driver, int(r.LapNumber)) for r in race.itertuples() if pd.notna(r.PitInTime)
        )
        drivers_at_lap = race.groupby("LapNumber")["Driver"].apply(list).to_dict()
        max_lap_per_driver = race.groupby("Driver")["LapNumber"].max().to_dict()

        for r in race.itertuples():
            lap = int(r.LapNumber)
            driver = r.Driver
            if lap >= max_lap_per_driver[driver]:
                continue  # no next lap to check for this driver
            if pd.isna(r.TimeSeconds) or pd.isna(r.TyreLife) or pd.isna(r.Compound):
                continue
            if (driver, lap) in pit_in_laps:
                continue  # this lap IS a pit lap - not a "still out, deciding" state

            gap_ahead, gap_behind = np.nan, np.nan
            for other in drivers_at_lap.get(lap, []):
                if other == driver:
                    continue
                r_other = by_driver_lap.get((other, lap))
                if r_other is None or pd.isna(r_other.TimeSeconds):
                    continue
                gap = r_other.TimeSeconds - r.TimeSeconds  # positive: other is AHEAD (finished lap sooner)
                if gap > 0 and gap <= GAP_SEARCH_LIMIT_S:
                    gap_ahead = min(gap_ahead, gap) if not np.isnan(gap_ahead) else gap
                elif gap < 0 and abs(gap) <= GAP_SEARCH_LIMIT_S:
                    gap_behind = min(gap_behind, -gap) if not np.isnan(gap_behind) else -gap

            rows.append({
                "circuit": circuit, "year": year, "round": round_number, "driver": driver,
                "lap": lap, "tire_age": r.TyreLife, "compound": r.Compound,
                "gap_ahead_seconds": gap_ahead, "gap_behind_seconds": gap_behind,
                "pits_next_lap": int((driver, lap + 1) in pit_in_laps),
            })

    return pd.DataFrame(rows)
