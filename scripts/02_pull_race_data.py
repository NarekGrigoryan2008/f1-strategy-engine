"""
Stage 0 (continued) - pull raw lap/weather/race-control/results data for every
(circuit, season) combination among the top 20 circuits identified by
01_get_calendars.py.

This is the slow step: it downloads real per-lap timing data from FastF1's API
for up to 20 circuits x 9 seasons = up to 180 races. FastF1 caches every API
response to disk (data/cache/), so:
  - the first run is slow (real network transfer)
  - a re-run after an interruption is fast for anything already cached, and this
    script also skips any (circuit, year) whose output CSVs already exist, so it
    is safe to stop and restart.

For each race we pull, with telemetry OFF (we don't need car telemetry / speed
traces for strategy analysis - only lap summaries, tires, weather, messages,
results - and skipping telemetry is a large speed win):
  - laps: Driver, DriverNumber, LapNumber, LapTime, Stint, Compound, TyreLife,
    FreshTyre, PitInTime, PitOutTime, TrackStatus, Position, IsAccurate,
    Deleted, DeletedReason (plus Team, which is trivial to keep and useful
    later since drivers change teams between seasons).
  - weather: per-timestamp track/air temp, humidity, rainfall, wind.
  - race_control: full text log of flags/SC/VSC/penalties, with RacingNumber
    where FastF1 attaches one (more reliable than parsing message text for
    "which driver does this concern").
  - results: finishing position, team, Status (finished / DNF reason).

Output layout: data/raw/<circuit_slug>/<year>/{laps,weather,race_control,results}.csv
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

from src.paths import CACHE_DIR, DERIVED_DIR, RAW_DIR
from src.util import slugify

# FastF1 hard-codes a client-side limit of 500 API calls/hour (rolling window) to
# protect the free, volunteer-run APIs it depends on - see the comment block at the
# top of fastf1/req.py, which explicitly asks users not to patch around this. Each
# race pull costs ~8-10 calls (laps, weather, race control, session/driver/status/
# lap-count/track-status metadata), so a large backfill like this one (158 races)
# will hit that limit partway through a single run - it did, at 36/158 races, ~10
# minutes in. Rather than treat that as a failure, we catch it specifically and
# sleep out a fresh window, then retry the SAME race (the loop below does not
# advance past a rate-limited race). This makes one script invocation able to
# finish the whole backfill unattended, at the cost of real wall-clock time -
# roughly 50-60 races fit in a 500-call budget, so ~122 remaining races after the
# first burst implies a couple more hour-long waits before this finishes.
RATE_LIMIT_COOLDOWN_S = 65 * 60  # 65 min: a bit over the 60 min rolling window

fastf1.Cache.enable_cache(str(CACHE_DIR))

LAP_COLUMNS = [
    "Time", "Driver", "DriverNumber", "Team", "LapNumber", "LapTime", "Stint",
    "Compound", "TyreLife", "FreshTyre", "PitInTime", "PitOutTime",
    "TrackStatus", "Position", "IsAccurate", "Deleted", "DeletedReason",
]
# `Time` (cumulative session time when the lap was completed) was added after the
# initial bulk pull, once Stage 2 Part B's attempt-detection turned out to need it
# to compute the real time gap between two cars at a given lap - see
# scripts/02c_backfill_laptime_column.py, which re-extracts it from FastF1's cache
# for every already-pulled race without spending any more of the rate-limit budget.

WEATHER_COLUMNS = [
    "Time", "AirTemp", "TrackTemp", "Humidity", "Pressure", "Rainfall",
    "WindDirection", "WindSpeed",
]

RC_COLUMNS = [
    "Time", "Lap", "Category", "Status", "Flag", "Scope", "Sector",
    "RacingNumber", "Message",
]

RESULTS_COLUMNS = [
    "DriverNumber", "Abbreviation", "FullName", "TeamName", "GridPosition",
    "Position", "ClassifiedPosition", "Status", "Points",
]


def load_race_calendar():
    top20 = pd.read_csv(DERIVED_DIR / "top20_circuits.csv")
    all_cal = pd.read_csv(DERIVED_DIR / "all_calendars_2018_2026.csv")
    locations = set(top20["location"])
    cal = all_cal[all_cal["location"].isin(locations)].copy()
    cal = cal.sort_values(["location", "year"])
    return cal


def already_pulled(out_dir: Path) -> bool:
    needed = ["laps.csv", "weather.csv", "race_control.csv", "results.csv"]
    return all((out_dir / f).exists() for f in needed)


def pull_one_race(year: int, round_number: int, location: str, event_name: str):
    slug = slugify(location)
    # Keyed on (circuit, year, round), not just (circuit, year): some years have
    # TWO races at the same venue (2020's COVID-adjusted calendar doubled up
    # Spielberg, Silverstone, and Sakhir; 2021 doubled up Spielberg again). An
    # earlier version of this function keyed only on (circuit, year) and silently
    # skipped the second race of each pair as "already pulled".
    out_dir = RAW_DIR / slug / f"{year}_r{int(round_number)}"

    if already_pulled(out_dir):
        return "skipped (already pulled)"

    out_dir.mkdir(parents=True, exist_ok=True)

    session = fastf1.get_session(year, int(round_number), "R")
    session.load(laps=True, telemetry=False, weather=True, messages=True)

    # --- laps ---
    laps = session.laps
    cols = [c for c in LAP_COLUMNS if c in laps.columns]
    laps[cols].to_csv(out_dir / "laps.csv", index=False)
    n_laps = len(laps)

    # --- weather ---
    weather = session.weather_data
    if weather is not None and len(weather):
        cols = [c for c in WEATHER_COLUMNS if c in weather.columns]
        weather[cols].to_csv(out_dir / "weather.csv", index=False)
        n_weather = len(weather)
    else:
        pd.DataFrame(columns=WEATHER_COLUMNS).to_csv(out_dir / "weather.csv", index=False)
        n_weather = 0

    # --- race control messages ---
    try:
        rc = session.race_control_messages
    except Exception:
        rc = None
    if rc is not None and len(rc):
        cols = [c for c in RC_COLUMNS if c in rc.columns]
        rc[cols].to_csv(out_dir / "race_control.csv", index=False)
        n_rc = len(rc)
    else:
        pd.DataFrame(columns=RC_COLUMNS).to_csv(out_dir / "race_control.csv", index=False)
        n_rc = 0

    # --- results ---
    results = session.results
    if results is not None and len(results):
        cols = [c for c in RESULTS_COLUMNS if c in results.columns]
        results[cols].to_csv(out_dir / "results.csv", index=False)
        n_results = len(results)
    else:
        pd.DataFrame(columns=RESULTS_COLUMNS).to_csv(out_dir / "results.csv", index=False)
        n_results = 0

    return f"OK  laps={n_laps} weather={n_weather} race_control={n_rc} results={n_results}"


def main():
    cal = load_race_calendar()
    print(f"{len(cal)} (circuit, year) races to pull\n")

    log_rows = []
    t0 = time.time()
    rows = list(cal.iterrows())
    total = len(rows)
    i = 0
    n_rate_limit_waits = 0
    while i < total:
        _, row = rows[i]
        year = int(row["year"])
        rnd = row["round"]
        location = row["location"]
        event_name = row["event_name"]
        label = f"[{i + 1}/{total}] {year} {location} ({event_name})"
        try:
            status = pull_one_race(year, rnd, location, event_name)
            print(f"{label}: {status}")
            log_rows.append({"year": year, "location": location, "event_name": event_name,
                              "status": status, "error": ""})
            i += 1
        except RateLimitExceededError as e:
            n_rate_limit_waits += 1
            print(f"{label}: rate limit hit ({e}). Sleeping "
                  f"{RATE_LIMIT_COOLDOWN_S / 60:.0f} min before retrying this race "
                  f"(wait #{n_rate_limit_waits})...")
            time.sleep(RATE_LIMIT_COOLDOWN_S)
            # do NOT advance i - retry the same race once the window has cooled down
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"{label}: FAILED - {err}")
            log_rows.append({"year": year, "location": location, "event_name": event_name,
                              "status": "FAILED", "error": err})
            traceback.print_exc(limit=1)
            i += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min ({n_rate_limit_waits} rate-limit cooldown(s) hit)")

    log_df = pd.DataFrame(log_rows)
    log_path = DERIVED_DIR / "data_pull_log.csv"
    log_df.to_csv(log_path, index=False)

    n_ok = (log_df["status"].str.startswith("OK") | log_df["status"].str.startswith("skipped")).sum()
    n_fail = (log_df["status"] == "FAILED").sum()
    print(f"Summary: {n_ok} succeeded/skipped, {n_fail} failed. Full log -> {log_path}")
    if n_fail:
        print("\nFailed races:")
        for _, r in log_df[log_df["status"] == "FAILED"].iterrows():
            print(f"  {r['year']} {r['location']}: {r['error']}")


if __name__ == "__main__":
    main()
