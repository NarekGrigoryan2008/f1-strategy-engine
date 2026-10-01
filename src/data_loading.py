"""Shared helpers for loading the raw per-race CSVs back into combined DataFrames."""
import re
from pathlib import Path

import pandas as pd

from src.paths import RAW_DIR

# Race directories are named "{year}_r{round}", e.g. "2020_r2" - the round number
# is needed because some years have TWO races at the same venue (2020's
# COVID-adjusted calendar doubled up Spielberg/Silverstone/Sakhir; 2021 doubled up
# Spielberg again), so (circuit, year) alone isn't a unique key.
_RACE_DIR_RE = re.compile(r"^(\d{4})_r(\d+)$")


def _parse_race_dir(name: str):
    m = _RACE_DIR_RE.match(name)
    if not m:
        raise ValueError(f"Unexpected race directory name (expected 'YYYY_rN'): {name!r}")
    return int(m.group(1)), int(m.group(2))


def _parse_laptime_seconds(series: pd.Series) -> pd.Series:
    """FastF1 LapTime is saved as a pandas Timedelta string ('0 days 00:01:24.115000').
    Convert to float seconds; unparseable / missing values become NaN."""
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def list_pulled_races():
    """Every (circuit_slug, year, round) that actually has a laps.csv."""
    races = []
    for laps_path in sorted(RAW_DIR.glob("*/*/laps.csv")):
        year, round_number = _parse_race_dir(laps_path.parent.name)
        circuit = laps_path.parent.parent.name
        races.append((circuit, year, round_number))
    return races


def load_all_laps() -> pd.DataFrame:
    """Concatenate laps.csv across every pulled race, tagged with circuit + year
    (+ round, to distinguish same-venue doubleheaders), with LapTime/PitInTime/
    PitOutTime converted to float seconds."""
    frames = []
    for circuit, year, round_number in list_pulled_races():
        path = RAW_DIR / circuit / f"{year}_r{round_number}" / "laps.csv"
        df = pd.read_csv(path)
        df["circuit"] = circuit
        df["year"] = year
        df["round"] = round_number
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    laps = pd.concat(frames, ignore_index=True)
    laps["LapTimeSeconds"] = _parse_laptime_seconds(laps["LapTime"])
    laps["PitInTimeSeconds"] = _parse_laptime_seconds(laps["PitInTime"])
    laps["PitOutTimeSeconds"] = _parse_laptime_seconds(laps["PitOutTime"])
    # TrackStatus is a FastF1 STRING of digit codes (e.g. "12" means both status 1
    # and status 2 were active during the lap - not the integer twelve). On disk
    # some races have occasional missing TrackStatus values, which makes pandas
    # read that whole column as float64 for those files; after concatenation
    # across races the combined column ends up float64 too, so a plain
    # .astype(str) turns clean-flag laps into "1.0" instead of "1" and an exact
    # "1" comparison silently matches nothing. Strip a trailing ".0" (left over
    # from that float round-trip) before comparing.
    laps["TrackStatus"] = laps["TrackStatus"].astype(str).str.replace(r"\.0$", "", regex=True)
    return laps


def _load_all(filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted(RAW_DIR.glob(f"*/*/{filename}")):
        year, round_number = _parse_race_dir(path.parent.name)
        circuit = path.parent.parent.name
        df = pd.read_csv(path)
        df["circuit"] = circuit
        df["year"] = year
        df["round"] = round_number
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all_weather() -> pd.DataFrame:
    return _load_all("weather.csv")


def load_all_results() -> pd.DataFrame:
    return _load_all("results.csv")


def load_all_race_control() -> pd.DataFrame:
    return _load_all("race_control.csv")
