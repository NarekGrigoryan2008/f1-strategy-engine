"""Stage 5 prerequisite - build the pit-hazard dataset and inspect it honestly
before deciding how much to condition the fitted model on."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data_loading import load_all_laps
from src.paths import DERIVED_DIR
from src.stage5_pit_hazard import build_pit_hazard_dataset


def main():
    print("Loading laps...")
    laps = load_all_laps()
    print(f"{len(laps)} raw lap rows across {laps[['circuit','year','round']].drop_duplicates().shape[0]} races.")

    t0 = time.time()
    print("\nBuilding pit-hazard dataset (one row per driver-lap, with gap to nearest rivals)...")
    hazard_data = build_pit_hazard_dataset(laps)
    print(f"Done in {time.time()-t0:.1f}s. {len(hazard_data)} driver-lap rows.")

    out_path = DERIVED_DIR / "pit_hazard_dataset.csv"
    hazard_data.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    print(f"\nOverall pit rate (P(pits next lap), unconditional): {hazard_data['pits_next_lap'].mean():.4f} "
          f"({hazard_data['pits_next_lap'].sum()}/{len(hazard_data)})")

    print(f"\ngap_ahead_seconds coverage: {hazard_data['gap_ahead_seconds'].notna().mean()*100:.1f}% "
          f"of rows have a car within 10s ahead")
    print(f"gap_behind_seconds coverage: {hazard_data['gap_behind_seconds'].notna().mean()*100:.1f}% "
          f"of rows have a car within 10s behind")

    print("\nPit rate by tire_age bucket:")
    hazard_data["tire_age_bucket"] = pd.cut(hazard_data["tire_age"], bins=[0, 5, 10, 15, 20, 25, 30, 40, 100])
    print(hazard_data.groupby("tire_age_bucket", observed=True)["pits_next_lap"].agg(["count", "mean"]).to_string())

    print("\nPit rate by compound:")
    print(hazard_data.groupby("compound")["pits_next_lap"].agg(["count", "mean"]).sort_values("count", ascending=False).to_string())

    print("\nPit rate by whether a rival is close ahead vs not:")
    hazard_data["has_close_ahead"] = hazard_data["gap_ahead_seconds"].notna()
    hazard_data["has_close_behind"] = hazard_data["gap_behind_seconds"].notna()
    print(hazard_data.groupby("has_close_ahead")["pits_next_lap"].agg(["count", "mean"]).to_string())
    print(hazard_data.groupby("has_close_behind")["pits_next_lap"].agg(["count", "mean"]).to_string())

    # Density check: how many rows exist in the "thin" corners a full joint
    # conditional model would need - old tires AND a close rival, by compound
    thin_check = hazard_data[(hazard_data["tire_age"] >= 20) & (hazard_data["gap_behind_seconds"].notna())]
    print(f"\nRows with tire_age>=20 AND a close rival behind: {len(thin_check)} "
          f"(pit rate {thin_check['pits_next_lap'].mean():.3f})")
    print("By circuit x compound, count of such rows (checking for thin cells):")
    print(thin_check.groupby(["circuit", "compound"]).size().sort_values(ascending=False).head(10).to_string())
    print(f"...and the thinnest cells with >0 rows:")
    counts = thin_check.groupby(["circuit", "compound"]).size()
    print(counts[counts > 0].sort_values().head(10).to_string())


if __name__ == "__main__":
    main()
