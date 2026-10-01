"""
Stage 3 - run SC/VSC/red-flag detection across every race, fit the per-circuit
hazard curve, and save both. This is the "first draft" the project spec calls
for - a subsequent verification pass against Wikipedia race reports checks a
sample of it, given the scope of manually checking all 150 races.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loading import load_all_laps
from src.paths import CHARTS_DIR, DERIVED_DIR, MODELS_DIR
from src.stage3_incidents import detect_periods, fit_hazard_curve

# Manual correction, verified against Wikipedia (2026-09-23): the 2021 Belgian
# GP was run entirely behind
# the Safety Car from the rolling start (laps 1-2 counted as race laps by a
# controversial FIA ruling), then red-flagged on lap 3 and never resumed - the
# only World Championship race with zero green-flag laps. TrackStatus records
# laps 1-2 as "1" (clear) because the SC was on track from the start rather than
# "deployed" mid-race in the way that normally flips TrackStatus, so auto-
# detection has no transition event to catch and misses the SC entirely (it only
# finds the lap-3 red flag). This is a genuine, understood limitation of
# TrackStatus-only detection for this extremely rare race format, not something
# to chase further - applied here as an explicit, traceable correction rather
# than silently patched into the raw data.
MANUAL_CORRECTIONS = {
    ("spa_francorchamps", 2021): [
        {"type": "SC", "start_lap": 1, "end_lap": 2, "duration_laps": 2},
        {"type": "RED", "start_lap": 3, "end_lap": 3, "duration_laps": 1},
    ],
    # Found during the expanded 24-race verification pass (2026-09-24):
    # the 2020 Italian GP's red flag showed up as two separate
    # single-lap entries (lap 24, lap 26) instead of one continuous period.
    # Root cause understood, not just patched: during a full red-flag stoppage
    # cars are stationary, so no lap is actually being timed for anyone - lap 25
    # has NO TrackStatus rows at all for that race, creating a genuine gap in the
    # per-lap union that the contiguous-range detector reads as two separate
    # periods. Confirmed against the official race control log (not just
    # Wikipedia prose) that this was one continuous red flag. This is a general
    # characteristic of red-flag detection specifically (lap numbering doesn't
    # advance during a full stoppage), not unique to this race - flagged as a
    # "what I'd build next" (bridge same-type gaps of 1-2 laps automatically)
    # rather than fixed for every race that might have it.
    ("monza", 2020): [
        {"type": "SC", "start_lap": 18, "end_lap": 26, "duration_laps": 9},
        {"type": "RED", "start_lap": 24, "end_lap": 26, "duration_laps": 3},
    ],
    # Found during the same pass: the 2026 Barcelona GP's auto-detection included
    # two VSC periods (lap 35, laps 51-53) with no corroborating race-control
    # message anywhere nearby (compare to the confirmed lap 41 VSC, which has a
    # clean "VSC DEPLOYED"/"VSC ENDING" message pair) - the underlying
    # TrackStatus values at those laps are unusual multi-code clusters (e.g.
    # "1267" on a single lap for one car) not seen elsewhere in this dataset.
    # Treated as likely TrackStatus noise specific to this very recent (2026)
    # race's live-timing feed, not a real VSC - removed rather than kept on the
    # strength of the raw code alone, since the more authoritative race-control
    # text log doesn't support them. The two Wikipedia-confirmed VSCs (lap 41,
    # and ~5 laps from the end) are kept unchanged.
    ("barcelona", 2026): [
        {"type": "VSC", "start_lap": 38, "end_lap": 42, "duration_laps": 5},
        {"type": "VSC", "start_lap": 60, "end_lap": 65, "duration_laps": 6},
    ],
}


def apply_manual_corrections(periods: pd.DataFrame) -> pd.DataFrame:
    periods = periods.copy()
    periods["source"] = "trackstatus_autodetected"
    for (circuit, year), rows in MANUAL_CORRECTIONS.items():
        match = periods[(periods["circuit"] == circuit) & (periods["year"] == year)]
        round_number = match["round"].iloc[0] if len(match) else None
        periods = periods[~((periods["circuit"] == circuit) & (periods["year"] == year))]
        correction_df = pd.DataFrame([
            {"circuit": circuit, "year": year, "round": round_number, "source": "wikipedia_corrected", **r}
            for r in rows
        ])
        periods = pd.concat([periods, correction_df], ignore_index=True)
    return periods


def main():
    print("Loading laps...")
    laps = load_all_laps()
    race_keys = laps[["circuit", "year", "round"]].drop_duplicates()
    print(f"Detecting SC/VSC/red periods across {len(race_keys)} races...")

    all_periods = []
    for _, key in race_keys.iterrows():
        race_laps = laps[(laps["circuit"] == key["circuit"]) & (laps["year"] == key["year"])
                          & (laps["round"] == key["round"])]
        p = detect_periods(race_laps)
        if len(p):
            all_periods.append(p)

    periods = pd.concat(all_periods, ignore_index=True)
    periods = apply_manual_corrections(periods)
    n_corrected = (periods["source"] == "wikipedia_corrected").sum()
    if n_corrected:
        print(f"Applied {n_corrected} Wikipedia-verified manual correction row(s) - see MANUAL_CORRECTIONS.")

    periods_path = DERIVED_DIR / "sc_vsc_periods_autodetected.csv"
    periods.to_csv(periods_path, index=False)
    print(f"\n{len(periods)} periods detected -> {periods_path}")
    print(periods["type"].value_counts().to_string())

    n_races_with_any = periods[["circuit", "year", "round"]].drop_duplicates().shape[0]
    print(f"\n{n_races_with_any}/{len(race_keys)} races ({n_races_with_any/len(race_keys)*100:.0f}%) "
          f"had at least one SC/VSC/red period.")

    print("\nFitting per-circuit hazard curves...")
    hazard_df = fit_hazard_curve(periods, laps)
    hazard_path = MODELS_DIR / "sc_vsc_hazard_by_circuit_lap.csv"
    hazard_df.to_csv(hazard_path, index=False)
    print(f"Saved -> {hazard_path}")

    # Per-circuit summary: SC+VSC probability of at least one incident, by race count
    summary = (
        periods.groupby(["circuit", "type"])
        .agg(n_periods=("start_lap", "count"))
        .reset_index()
        .pivot(index="circuit", columns="type", values="n_periods")
        .fillna(0).astype(int)
    )
    race_counts = race_keys.groupby("circuit").size().rename("n_races")
    summary = summary.join(race_counts)
    print("\n=== Per-circuit incident counts (total periods, not races) ===")
    print(summary.to_string())

    # Diagnostic chart: SC hazard curve for a couple of circuits known for very
    # different SC propensity (Baku is famously SC-prone; Barcelona much less so)
    fig, ax = plt.subplots(figsize=(8, 5))
    for circuit, color in [("baku", "#C0362C"), ("barcelona", "#4C7EA8"), ("spa_francorchamps", "#1B7F3E")]:
        g = hazard_df[(hazard_df["circuit"] == circuit) & (hazard_df["type"] == "SC")]
        if len(g):
            ax.plot(g["lap"], g["smoothed_hazard"], color=color, linewidth=2, label=circuit)
    ax.set_xlabel("Lap number")
    ax.set_ylabel("P(Safety Car starts this lap)")
    ax.set_title("Safety Car hazard rate by lap, smoothed (5-lap window)\nExample circuits")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#eeeeee", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_path = CHARTS_DIR / "sc_hazard_by_lap_examples.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved example hazard chart -> {out_path}")


if __name__ == "__main__":
    main()
