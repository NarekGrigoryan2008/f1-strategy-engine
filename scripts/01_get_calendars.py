"""
Stage 0 - figure out which 20 circuits to build this project around.

We don't hand-pick circuits from memory. Instead we pull every season's actual
calendar from FastF1 (2018-2026, the range FastF1's modern timing data source
covers) and count how many times each circuit appears by its `Location` field.
`Location` is the venue (e.g. "Spa-Francorchamps"), which is *mostly* stable
across years even when `EventName` changes for sponsorship reasons (e.g.
"Belgian Grand Prix" vs "Rolex Belgian Grand Prix") — but checking it empirically
turned up a handful of cases where it isn't (see LOCATION_ALIASES below). Counting
by Location, after fixing those, is what gives an apples-to-apples count of "how
many times has a race actually been held at this track."

Output: data/derived/circuit_calendar_counts.csv (every circuit seen, with its
count and which years it appeared) and data/derived/top20_circuits.csv (the
20 circuits this whole project is scoped to).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fastf1
import pandas as pd

from src.paths import CACHE_DIR, DERIVED_DIR

fastf1.Cache.enable_cache(str(CACHE_DIR))

YEARS = range(2018, 2027)  # 2018-2026 inclusive

# The project spec's assumption that FastF1's `Location` field is stable across
# years for a given venue turned out to be only mostly true. Checking empirically
# (grouping by event_name and looking for event_names that map to more than one
# Location string) turned up six cases. Three are the *same physical circuit*
# spelled two different ways across seasons and need to be merged before counting,
# or an established circuit gets undercounted and could be wrongly bumped out of
# the top 20 by that mislabeling alone:
#   - "Monte Carlo" / "Monaco"       -> same street circuit, Monaco GP
#   - "Miami" / "Miami Gardens"      -> same Hard Rock Stadium circuit, Miami GP
#   - "Yas Island" / "Yas Marina"    -> same Abu Dhabi circuit, Abu Dhabi GP
# One is a clear upstream metadata error, not a real alias: the 2026 Bahrain GP
# calendar entry lists Location="Kuala Lumpur" (Kuala Lumpur is in Malaysia, which
# hasn't hosted an F1 round since 2017; Bahrain's circuit is in Sakhir and every
# other season 2018-2025 confirms this). Treated as a one-off fix, not a rename of
# "Kuala Lumpur" in general, in case Malaysia ever returns to the calendar.
# One case is a REAL, non-alias change and is deliberately NOT merged: the Spanish
# GP moves from Barcelona (Circuit de Barcelona-Catalunya) to Madrid (a new street
# circuit) starting 2026. Same event_name, genuinely different circuits, so
# "Barcelona" and "Madrid" must stay separate rows or the top-20 selection would be
# silently wrong about which venue actually has the longer, more-raced history.
LOCATION_ALIASES = {
    "Monte Carlo": "Monaco",
    "Miami Gardens": "Miami",
    "Yas Marina": "Yas Island",
}


def canonicalize_location(row):
    loc = row["location"]
    if row["event_name"] == "Bahrain Grand Prix" and loc == "Kuala Lumpur":
        return "Sakhir"  # 2026 calendar metadata error, see note above
    return LOCATION_ALIASES.get(loc, loc)


def main():
    all_rows = []
    for year in YEARS:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"  [{year}] FAILED to fetch schedule: {e}")
            continue
        n = len(schedule)
        print(f"  [{year}] {n} events on calendar")
        for _, row in schedule.iterrows():
            all_rows.append({
                "year": year,
                "round": row.get("RoundNumber"),
                "event_name": row.get("EventName"),
                "location": row.get("Location"),
                "country": row.get("Country"),
                "official_event_name": row.get("OfficialEventName"),
                "event_date": row.get("EventDate"),
                "event_format": row.get("EventFormat"),
            })

    df = pd.DataFrame(all_rows)

    # Exclude testing-only / non-race entries defensively (include_testing=False
    # should already handle this, but be explicit).
    df = df[df["location"].notna()].copy()

    df["location_raw"] = df["location"]
    df["location"] = df.apply(canonicalize_location, axis=1)
    n_changed = (df["location"] != df["location_raw"]).sum()
    print(f"\nCanonicalized {n_changed} event-rows' location field (see LOCATION_ALIASES / "
          f"Bahrain fix in this script's header comment).")

    # Saved AFTER canonicalization: scripts/02_pull_race_data.py filters this file's
    # `location` column against top20_circuits.csv's `location` column, so the two
    # need to speak the same (canonical) circuit names. `location_raw` is kept
    # alongside so the original FastF1 value is never actually lost.
    raw_path = DERIVED_DIR / "all_calendars_2018_2026.csv"
    df.to_csv(raw_path, index=False)
    print(f"Saved full calendar pull ({len(df)} event-rows) -> {raw_path}")

    counts = (
        df.groupby("location")
        .agg(
            races=("year", "count"),
            years=("year", lambda s: sorted(s.unique().tolist())),
            countries=("country", lambda s: sorted(s.unique().tolist())),
            event_names=("event_name", lambda s: sorted(s.unique().tolist())),
        )
        .reset_index()
        .sort_values(["races", "location"], ascending=[False, True])
        .reset_index(drop=True)
    )
    counts_path = DERIVED_DIR / "circuit_calendar_counts.csv"
    counts.to_csv(counts_path, index=False)
    print(f"Saved per-circuit counts ({len(counts)} distinct circuits) -> {counts_path}")

    top20 = counts.head(20).copy()
    top20_path = DERIVED_DIR / "top20_circuits.csv"
    top20.to_csv(top20_path, index=False)
    print(f"Saved top 20 circuits -> {top20_path}\n")

    print("=== Top 20 circuits by race count, 2018-2026 ===")
    for _, r in top20.iterrows():
        print(f"  {r['races']}x  {r['location']:30s} years={r['years']}")

    print("\n=== Circuits just outside the top 20 (for reference) ===")
    for _, r in counts.iloc[20:26].iterrows():
        print(f"  {r['races']}x  {r['location']:30s} years={r['years']}")


if __name__ == "__main__":
    main()
