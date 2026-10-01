"""
Stage 2, Part B - detect real historical undercut/overcut ATTEMPTS from lap data,
extract features, and label each one success/fail per the rule decided on
2026-09-22 (see src/config.py).

An attempt is: driver A pits on lap L_A while running within
UNDERCUT_GAP_THRESHOLD_S of another driver B (the nearest car ahead, or the
nearest car behind - tracked as two separate candidate attempts, since undercutting
the car ahead and defending against the car behind are different strategic
situations), where B pits on some LATER lap L_B in the same race. Both directions
share one success rule: A is "ahead" once its Position beats B's Position at the
measurement lap.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    UNDERCUT_GAP_THRESHOLD_S, UNDERCUT_SUCCESS_WINDOW_LAPS,
    UNDERCUT_EXCLUDE_IF_WINDOW_UNAVAILABLE,
)


def _race_key(row) -> tuple:
    return (row["circuit"], row["year"], row["round"])


def detect_attempts_in_race(laps: pd.DataFrame, pit_loss_seconds: float) -> pd.DataFrame:
    """laps: one race's laps (already tagged with circuit/year/round), with a
    'TimeSeconds' column (cumulative session time, float seconds) already added.
    Returns one row per detected attempt (both directions, unlabeled candidates
    that failed the measurement-window check already dropped)."""
    laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    max_lap = int(laps["LapNumber"].max())

    # Fast lookups: (driver, lap) -> row, and driver -> sorted pit-in laps
    by_driver_lap = {(r.Driver, int(r.LapNumber)): r for r in laps.itertuples()}
    pit_laps_by_driver = {}
    for r in laps.itertuples():
        if pd.notna(r.PitInTime):
            pit_laps_by_driver.setdefault(r.Driver, []).append(int(r.LapNumber))
    for d in pit_laps_by_driver:
        pit_laps_by_driver[d].sort()

    drivers_at_lap = laps.groupby("LapNumber")["Driver"].apply(list).to_dict()

    attempts = []
    for driver_a, pit_laps in pit_laps_by_driver.items():
        for lap_a in pit_laps:
            gap_lap = lap_a - 1
            row_a_gap = by_driver_lap.get((driver_a, gap_lap))
            row_a_pit = by_driver_lap.get((driver_a, lap_a))
            row_a_out = by_driver_lap.get((driver_a, lap_a + 1))
            if row_a_gap is None or row_a_pit is None or row_a_out is None:
                continue
            if pd.isna(row_a_gap.TimeSeconds):
                continue
            # a genuine tire-strategy stop, not e.g. a penalty-only pit visit
            if row_a_pit.Compound == row_a_out.Compound:
                continue

            candidates = []  # (gap_seconds, driver_b) ; gap_seconds > 0 means A behind B
            for driver_b in drivers_at_lap.get(gap_lap, []):
                if driver_b == driver_a:
                    continue
                row_b = by_driver_lap.get((driver_b, gap_lap))
                if row_b is None or pd.isna(row_b.TimeSeconds):
                    continue
                if gap_lap in pit_laps_by_driver.get(driver_b, []):
                    continue  # B also pitting this same lap - not a clean A-first-vs-B comparison
                gap_seconds = row_a_gap.TimeSeconds - row_b.TimeSeconds
                if abs(gap_seconds) <= UNDERCUT_GAP_THRESHOLD_S:
                    candidates.append((gap_seconds, driver_b, row_b))

            # gap_seconds = TimeA - TimeB: positive means A took longer to reach this
            # lap than B did, i.e. A is BEHIND B (B is ahead); negative means A is ahead.
            ahead = [c for c in candidates if c[0] > 0]   # B ahead of A (A trailing, undercut candidate)
            behind = [c for c in candidates if c[0] < 0]  # B behind A (A leading, defending)
            nearest_ahead = min(ahead, key=lambda c: abs(c[0])) if ahead else None
            nearest_behind = min(behind, key=lambda c: abs(c[0])) if behind else None

            for candidate in (nearest_ahead, nearest_behind):
                if candidate is None:
                    continue
                gap_seconds, driver_b, row_b_gap = candidate

                # B's first stop strictly after A's stop, in this same race
                b_future_pits = [l for l in pit_laps_by_driver.get(driver_b, []) if l > lap_a]
                if not b_future_pits:
                    continue  # can't measure "once both have stopped" - no later B stop exists
                lap_b = b_future_pits[0]

                measurement_lap = lap_b + UNDERCUT_SUCCESS_WINDOW_LAPS
                if measurement_lap > max_lap:
                    if UNDERCUT_EXCLUDE_IF_WINDOW_UNAVAILABLE:
                        continue
                row_a_meas = by_driver_lap.get((driver_a, measurement_lap))
                row_b_meas = by_driver_lap.get((driver_b, measurement_lap))
                if row_a_meas is None or row_b_meas is None:
                    continue
                if pd.isna(row_a_meas.Position) or pd.isna(row_b_meas.Position):
                    continue

                # SC/VSC exposure across the whole window this attempt is judged over
                window_laps = range(lap_a, measurement_lap + 1)
                sc_affected = False
                for d, lp in ((driver_a, l) for l in window_laps):
                    r = by_driver_lap.get((d, lp))
                    if r is not None and r.TrackStatus != "1":
                        sc_affected = True
                        break
                if not sc_affected:
                    for lp in window_laps:
                        r = by_driver_lap.get((driver_b, lp))
                        if r is not None and r.TrackStatus != "1":
                            sc_affected = True
                            break

                # clean air proxy: is there a THIRD car within 1.0s directly ahead of A
                # at the gap-measurement lap (i.e. is A itself in traffic independent of B)?
                a_in_traffic = any(
                    0 < (row_a_gap.TimeSeconds - by_driver_lap[(d, gap_lap)].TimeSeconds) <= 1.0
                    for d in drivers_at_lap.get(gap_lap, [])
                    if d != driver_a and (d, gap_lap) in by_driver_lap
                    and pd.notna(by_driver_lap[(d, gap_lap)].TimeSeconds)
                )

                attempts.append({
                    "circuit": row_a_pit.circuit, "year": row_a_pit.year, "round": row_a_pit.round,
                    "driver_a": driver_a, "driver_b": driver_b,
                    "lap_a": lap_a, "lap_b": lap_b, "measurement_lap": measurement_lap,
                    "direction": "undercut_ahead" if gap_seconds > 0 else "defend_behind",
                    "gap_seconds": gap_seconds,
                    "tire_age_a": row_a_pit.TyreLife, "compound_a_before": row_a_pit.Compound,
                    "compound_a_after": row_a_out.Compound,
                    "tire_age_b": row_b_gap.TyreLife, "compound_b": row_b_gap.Compound,
                    "pit_loss_seconds": pit_loss_seconds,
                    "a_in_traffic": a_in_traffic,
                    "sc_vsc_affected": sc_affected,
                    "position_a_at_measurement": row_a_meas.Position,
                    "position_b_at_measurement": row_b_meas.Position,
                    "success": int(row_a_meas.Position < row_b_meas.Position),
                })

    return pd.DataFrame(attempts)
