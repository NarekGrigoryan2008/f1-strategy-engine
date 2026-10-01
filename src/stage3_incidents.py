"""
Stage 3 - auto-detect Safety Car / VSC / red-flag periods from TrackStatus, then
fit a simple per-circuit, per-lap hazard curve.

FastF1's TrackStatus codes (standard, documented convention - not something this
project chose): 1 = track clear, 2 = yellow flag, 4 = Safety Car deployed,
5 = red flag, 6 = VSC deployed, 7 = VSC ending. A lap's TrackStatus is a STRING
that can contain more than one code if the status changed mid-lap (e.g. "46" means
both VSC and SC occurred during that lap), so detection checks for a code's
PRESENCE anywhere in the string, not an exact match.
"""
import pandas as pd

SC_CODE, VSC_CODES, RED_CODE = "4", {"6", "7"}, "5"


def _status_flags(track_status: str) -> dict:
    chars = set(str(track_status))
    return {
        "is_sc": SC_CODE in chars,
        "is_vsc": bool(chars & VSC_CODES),
        "is_red": RED_CODE in chars,
    }


def detect_periods(laps: pd.DataFrame) -> pd.DataFrame:
    """laps: laps for ONE race (already tagged circuit/year/round). Returns one
    row per detected contiguous SC/VSC/red period: circuit, year, round, type,
    start_lap, end_lap, duration_laps."""
    # Build a race-level per-lap status: a lap counts as under a given condition if
    # ANY driver's row for that lap shows it (drivers' lap boundaries can be
    # slightly offset from the exact flag timing, so union is more robust than
    # picking one driver arbitrarily).
    # NOTE: pandas' groupby(...).apply(lambda s: {...}) silently auto-expands a
    # returned dict into a MultiIndex Series instead of one dict per group -
    # building the dict-of-dicts explicitly (not via .apply) avoids that pitfall.
    lap_status = {}
    for lap_number, group in laps.groupby("LapNumber"):
        lap_status[lap_number] = {
            k: any(_status_flags(v)[k] for v in group["TrackStatus"])
            for k in ("is_sc", "is_vsc", "is_red")
        }
    per_lap_df = pd.DataFrame.from_dict(lap_status, orient="index").sort_index()

    periods = []
    for col, label in (("is_sc", "SC"), ("is_vsc", "VSC"), ("is_red", "RED")):
        active = per_lap_df[col]
        in_period = False
        start = None
        for lap_number, flag in active.items():
            if flag and not in_period:
                in_period = True
                start = lap_number
            elif not flag and in_period:
                in_period = False
                periods.append((label, start, prev_lap))
            prev_lap = lap_number
        if in_period:
            periods.append((label, start, prev_lap))

    if not periods:
        return pd.DataFrame(columns=["circuit", "year", "round", "type", "start_lap", "end_lap", "duration_laps"])

    first = laps.iloc[0]
    return pd.DataFrame([
        {"circuit": first["circuit"], "year": first["year"], "round": first["round"],
         "type": t, "start_lap": int(s), "end_lap": int(e), "duration_laps": int(e - s + 1)}
        for t, s, e in periods
    ])


def fit_hazard_curve(periods: pd.DataFrame, laps: pd.DataFrame, smooth_window: int = 5) -> pd.DataFrame:
    """Per-circuit, per-lap-number probability that an SC/VSC/red period STARTS at
    that lap, given the race reached that lap. Simple empirical hazard rate
    (incidents starting at lap L / races that included lap L), smoothed with a
    centered rolling mean within each circuit to reduce lap-to-lap noise - most
    circuits here only have 5-11 races, so a raw single-lap rate is mostly 0 or a
    noisy 1/n spike; the smoothing is a deliberate, documented tradeoff for
    usability, not a claim of finer-grained precision than the sample supports.
    """
    race_lap_counts = laps.groupby(["circuit", "year", "round"])["LapNumber"].max()
    races_by_circuit = race_lap_counts.groupby("circuit")

    rows = []
    starts = periods.groupby(["circuit", "type", "start_lap"]).size().rename("n_incidents").reset_index()

    for circuit, race_group in races_by_circuit:
        max_lap_in_circuit = int(race_group.max())
        n_races = len(race_group)
        for lap in range(1, max_lap_in_circuit + 1):
            races_reaching_lap = int((race_group >= lap).sum())
            if races_reaching_lap == 0:
                continue
            for incident_type in ("SC", "VSC", "RED"):
                n_incidents = starts[
                    (starts["circuit"] == circuit) & (starts["type"] == incident_type)
                    & (starts["start_lap"] == lap)
                ]["n_incidents"].sum()
                rows.append({
                    "circuit": circuit, "lap": lap, "type": incident_type,
                    "races_reaching_lap": races_reaching_lap, "n_incidents_starting": int(n_incidents),
                    "raw_hazard": n_incidents / races_reaching_lap,
                })

    hazard_df = pd.DataFrame(rows)
    hazard_df["smoothed_hazard"] = (
        hazard_df.groupby(["circuit", "type"])["raw_hazard"]
        .transform(lambda s: s.rolling(smooth_window, center=True, min_periods=1).mean())
    )
    return hazard_df


def sc_vsc_probability(circuit: str, lap: int, incident_type: str, hazard_df: pd.DataFrame) -> float:
    """The Stage 3 deliverable: probability of an SC/VSC/RED period starting at
    this lap, for this circuit."""
    row = hazard_df[(hazard_df["circuit"] == circuit) & (hazard_df["lap"] == lap)
                     & (hazard_df["type"] == incident_type)]
    if row.empty:
        raise KeyError(f"No hazard estimate for circuit={circuit!r}, lap={lap}, type={incident_type!r}")
    return float(row.iloc[0]["smoothed_hazard"])
