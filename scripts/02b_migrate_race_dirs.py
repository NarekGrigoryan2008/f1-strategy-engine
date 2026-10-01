"""
One-time migration: rename data/raw/<circuit>/<year>/ to
data/raw/<circuit>/<year>_r<round>/, and delete the 4 directories where two real
races previously collapsed into one because (circuit, year) alone isn't a
unique key for a doubleheader season (two races at the same circuit in the
same year).

Safe to re-run: skips anything already in the new "<year>_r<round>" form.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.paths import DERIVED_DIR, RAW_DIR
from src.util import slugify

# (circuit slug, year) pairs where two races shared a venue in the old scheme.
# Both races' data need a fresh pull under the new round-aware directory naming,
# so these old, ambiguous directories are deleted rather than guessed at.
COLLISION_PAIRS = {
    ("sakhir", 2020),
    ("silverstone", 2020),
    ("spielberg", 2020),
    ("spielberg", 2021),
}


def main():
    cal = pd.read_csv(DERIVED_DIR / "all_calendars_2018_2026.csv")
    cal["slug"] = cal["location"].apply(slugify)

    old_dirs = [d for d in RAW_DIR.glob("*/*") if d.is_dir() and d.name.isdigit()]
    print(f"Found {len(old_dirs)} old-style '<circuit>/<year>/' directories.\n")

    n_deleted, n_renamed, n_skipped = 0, 0, 0
    for d in old_dirs:
        circuit = d.parent.name
        year = int(d.name)

        if (circuit, year) in COLLISION_PAIRS:
            print(f"  DELETE (doubleheader collision, needs fresh pull): {d}")
            shutil.rmtree(d)
            n_deleted += 1
            continue

        match = cal[(cal["slug"] == circuit) & (cal["year"] == year)]
        if len(match) != 1:
            print(f"  SKIP (expected exactly 1 calendar match for {circuit} {year}, "
                  f"found {len(match)}): {d}")
            n_skipped += 1
            continue

        round_number = int(match.iloc[0]["round"])
        new_dir = d.parent / f"{year}_r{round_number}"
        if new_dir.exists():
            print(f"  SKIP (target already exists): {d} -> {new_dir}")
            n_skipped += 1
            continue
        d.rename(new_dir)
        n_renamed += 1

    print(f"\nDone: {n_renamed} renamed, {n_deleted} deleted (need re-pull), {n_skipped} skipped.")


if __name__ == "__main__":
    main()
