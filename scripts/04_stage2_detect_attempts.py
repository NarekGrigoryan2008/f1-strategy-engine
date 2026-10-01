"""
Stage 2, Part B (detection step) - run src/stage2_attempts.py's detector across
every pulled race, using each circuit's Stage 1 pit-loss constant, and save the
full labeled attempts table.

Deliberately does NOT fit a model here - the choice between logistic
regression and a gradient-boosted model gets made after seeing the real
sample size this script produces, not before.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data_loading import load_all_laps
from src.paths import DERIVED_DIR, MODELS_DIR
from src.stage2_attempts import detect_attempts_in_race


def main():
    print("Loading all laps...")
    laps = load_all_laps()
    laps["TimeSeconds"] = pd.to_timedelta(laps["Time"], errors="coerce").dt.total_seconds()
    n_missing_time = laps["TimeSeconds"].isna().sum()
    print(f"Loaded {len(laps)} lap rows. {n_missing_time} rows have no parseable Time "
          f"({n_missing_time/len(laps)*100:.1f}%).")

    pit_loss_df = pd.read_csv(MODELS_DIR / "pit_loss_model.csv")
    pit_loss_lookup = dict(zip(pit_loss_df["circuit"], pit_loss_df["pit_loss_seconds"]))

    race_keys = laps[["circuit", "year", "round"]].drop_duplicates()
    print(f"\nDetecting attempts across {len(race_keys)} races...")

    all_attempts = []
    for i, (_, key) in enumerate(race_keys.iterrows(), 1):
        circuit, year, round_number = key["circuit"], key["year"], key["round"]
        pit_loss = pit_loss_lookup.get(circuit)
        if pit_loss is None:
            continue  # no pit-loss estimate for this circuit (too few clean stops in Stage 1)
        race_laps = laps[(laps["circuit"] == circuit) & (laps["year"] == year)
                          & (laps["round"] == round_number)]
        attempts = detect_attempts_in_race(race_laps, pit_loss_seconds=pit_loss)
        if len(attempts):
            all_attempts.append(attempts)

    result = pd.concat(all_attempts, ignore_index=True) if all_attempts else pd.DataFrame()
    out_path = DERIVED_DIR / "undercut_overcut_attempts.csv"
    result.to_csv(out_path, index=False)

    print(f"\n{len(result)} total attempts detected -> {out_path}\n")
    if len(result) == 0:
        return

    print("=== Overall ===")
    print(f"Success rate: {result['success'].mean():.3f} ({result['success'].sum()}/{len(result)})")
    print(f"SC/VSC-affected: {result['sc_vsc_affected'].sum()} ({result['sc_vsc_affected'].mean()*100:.1f}%)")
    print(f"In traffic (a_in_traffic): {result['a_in_traffic'].sum()} ({result['a_in_traffic'].mean()*100:.1f}%)")

    print("\n=== By direction ===")
    print(result.groupby("direction")["success"].agg(["count", "mean"]).to_string())

    print("\n=== By circuit ===")
    print(result.groupby("circuit")["success"].agg(["count", "mean"]).sort_values("count", ascending=False).to_string())

    print("\n=== Clean (non-SC-affected) subset, the primary modeling set ===")
    clean = result[~result["sc_vsc_affected"]]
    print(f"n={len(clean)}, success rate={clean['success'].mean():.3f}")


if __name__ == "__main__":
    main()
