"""
One-time backfill: re-extract laps.csv for every already-pulled race to add the
`Time` column (cumulative session time when each lap was completed).

Why this is needed after the fact: Stage 2 Part B (undercut/overcut attempt
detection) needs to compute the real time gap between two specific cars at a given
lap, which requires each driver's cumulative session time - not something
derivable from LapTime alone (summing LapTime per driver would drift under
red-flag-adjusted sessions). The original LAP_COLUMNS list in
scripts/02_pull_race_data.py didn't include it because Stage 1 didn't need it.

This was EXPECTED to be fast and free of rate-limit cost, since FastF1 already
cached the underlying API responses during the original bulk pull - but a
`fastf1.Cache.clear_cache()` call made while debugging the 2018 Monza data gap
wiped that cache, so this backfill turned out to need genuine fresh API calls
after all. It therefore uses the same rate-limit cooldown-and-retry pattern as
scripts/02_pull_race_data.py.

CORRECTION (found the hard way while building the incidents log):
this script's first version called session.load(..., messages=False), which
silently leaves Deleted/DeletedReason as NaN for every lap - FastF1 only
populates those two columns when messages=True, since it cross-references race
control messages internally to determine which laps were deleted for track
limits. The original Stage 0 pull used messages=True, so this bug only affected
the backfilled data, and only those two columns (Stage 1 and Stage 2 don't use
Deleted/DeletedReason). Fixed by setting messages=True here too, and this file
needs a second re-run to repair the damage its own first version caused.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

from src.data_loading import list_pulled_races
from src.paths import CACHE_DIR, RAW_DIR

fastf1.Cache.enable_cache(str(CACHE_DIR))

RATE_LIMIT_COOLDOWN_S = 65 * 60

# Kept identical to LAP_COLUMNS in scripts/02_pull_race_data.py (not imported from
# there since a filename starting with a digit isn't a valid Python module name).
LAP_COLUMNS = [
    "Time", "Driver", "DriverNumber", "Team", "LapNumber", "LapTime", "Stint",
    "Compound", "TyreLife", "FreshTyre", "PitInTime", "PitOutTime",
    "TrackStatus", "Position", "IsAccurate", "Deleted", "DeletedReason",
]


def main():
    races = list_pulled_races()
    print(f"Backfilling 'Time' column for {len(races)} races...\n")
    n_ok, n_fail, n_waits = 0, 0, 0
    i = 0
    while i < len(races):
        circuit, year, round_number = races[i]
        label = f"[{i + 1}/{len(races)}] {circuit} {year} r{round_number}"
        try:
            session = fastf1.get_session(year, round_number, "R")
            session.load(laps=True, telemetry=False, weather=False, messages=True)
            laps = session.laps
            cols = [c for c in LAP_COLUMNS if c in laps.columns]
            out_path = RAW_DIR / circuit / f"{year}_r{round_number}" / "laps.csv"
            laps[cols].to_csv(out_path, index=False)
            has_time = "Time" in laps.columns
            print(f"{label}: OK ({len(laps)} laps, Time column present: {has_time})")
            n_ok += 1
            i += 1
        except RateLimitExceededError as e:
            n_waits += 1
            print(f"{label}: rate limit hit ({e}). Sleeping {RATE_LIMIT_COOLDOWN_S/60:.0f} min "
                  f"before retrying (wait #{n_waits})...")
            time.sleep(RATE_LIMIT_COOLDOWN_S)
        except Exception as e:
            print(f"{label}: FAILED - {type(e).__name__}: {e}")
            n_fail += 1
            i += 1
    print(f"\nDone: {n_ok} ok, {n_fail} failed, {n_waits} rate-limit cooldown(s) hit.")


if __name__ == "__main__":
    main()
