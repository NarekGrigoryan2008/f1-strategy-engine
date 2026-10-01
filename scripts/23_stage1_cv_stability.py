"""
Stage 1 revisit - an honest robustness check, not a modeling upgrade: does each
(circuit, compound) degradation fit hold up on YEARS IT NEVER SAW, or is the
fitted slope unstable/drifting over time? Applies the exact same expanding-
window cross-validation methodology already used for Stage 2's classifier
(scripts/19_stage2_vif_and_cv.py) to Stage 1's degradation fits instead - same
fold structure, same "train on earlier years, test on later ones" logic - not a
curve, a neural net, or a per-driver correction, which the project spec
explicitly said to avoid for Stage 1.

For each fold and each (circuit, compound) group with enough clean laps in BOTH
the train and test portions: fits slope/intercept on TRAIN laps only, then
checks (a) out-of-sample R2 - how well the train-fitted line predicts TEST laps
it never saw, against (b) the ORIGINAL full-sample R2 already reported in
degradation_model.csv, and (c) slope stability - how close the train-period
slope is to a slope fit directly on the test period's own data (a direct,
literal check of "is this a stable physical constant or is it drifting").
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data_loading import load_all_laps
from src.paths import DERIVED_DIR, MODELS_DIR
from src.stage1_degradation import clean_laps_for_degradation_fit, MIN_LAPS_FOR_FIT

FOLDS = [
    (range(2018, 2023), [2023]),
    (range(2018, 2024), [2024]),
    (range(2018, 2025), [2025, 2026]),
]


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    print("Loading and cleaning laps (same Stage 1 rules, run once - race-level "
          "normalization doesn't depend on the CV split)...")
    laps = load_all_laps()
    clean = clean_laps_for_degradation_fit(laps)
    original_model = pd.read_csv(MODELS_DIR / "degradation_model.csv").set_index(["circuit", "compound"])

    all_rows = []
    for train_years, test_years in FOLDS:
        train_laps = clean[clean["year"].isin(train_years)]
        test_laps = clean[clean["year"].isin(test_years)]
        fold_label = f"train<={max(train_years)}, test={test_years}"
        print(f"\n=== Fold: {fold_label} ===")

        n_tested, oos_r2s, in_sample_r2s, slope_pairs = 0, [], [], []
        for (circuit, compound), test_g in test_laps.groupby(["circuit", "Compound"]):
            train_g = train_laps[(train_laps["circuit"] == circuit) & (train_laps["Compound"] == compound)]
            if len(train_g) < MIN_LAPS_FOR_FIT or len(test_g) < MIN_LAPS_FOR_FIT:
                continue

            train_slope, train_intercept = fit_line(train_g["TyreLife"].to_numpy(float),
                                                       train_g["LapTimeDelta"].to_numpy(float))
            test_x = test_g["TyreLife"].to_numpy(float)
            test_y = test_g["LapTimeDelta"].to_numpy(float)
            pred = train_slope * test_x + train_intercept
            oos_r2 = r_squared(test_y, pred)

            test_slope, _ = fit_line(test_x, test_y)  # a LOCAL fit on the test period alone, for slope comparison

            key = (circuit, compound)
            in_sample_r2 = original_model.loc[key, "r_squared"] if key in original_model.index else np.nan

            n_tested += 1
            oos_r2s.append(oos_r2)
            in_sample_r2s.append(in_sample_r2)
            slope_pairs.append((train_slope, test_slope))
            all_rows.append({
                "fold": fold_label, "circuit": circuit, "compound": compound,
                "n_train": len(train_g), "n_test": len(test_g),
                "train_slope": train_slope, "test_period_local_slope": test_slope,
                "out_of_sample_r2": oos_r2, "original_full_sample_r2": in_sample_r2,
            })

        oos_r2s = np.array(oos_r2s)
        in_sample_r2s = np.array(in_sample_r2s)
        slope_pairs = np.array(slope_pairs)
        print(f"{n_tested} (circuit, compound) groups had enough data (>= {MIN_LAPS_FOR_FIT} laps) "
              f"in both train and test periods.")
        print(f"Median out-of-sample R2: {np.median(oos_r2s):.4f} "
              f"(mean {oos_r2s.mean():.4f}, {(oos_r2s < 0).sum()}/{n_tested} groups negative - "
              f"worse than predicting the test period's own mean)")
        print(f"Median ORIGINAL full-sample R2 for the same groups: {np.nanmedian(in_sample_r2s):.4f} "
              f"(mean {np.nanmean(in_sample_r2s):.4f})")
        if n_tested >= 3:
            slope_corr = np.corrcoef(slope_pairs[:, 0], slope_pairs[:, 1])[0, 1]
            print(f"Correlation between train-period slope and test-period-local slope "
                  f"(same group, different years): {slope_corr:.3f}")

    out = pd.DataFrame(all_rows)
    out_path = DERIVED_DIR / "stage1_cv_stability.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved full per-group results -> {out_path}")

    print("\n=== Pooled across all 3 folds ===")
    print(f"Median out-of-sample R2: {out['out_of_sample_r2'].median():.4f}")
    print(f"Median original full-sample R2 (same groups): {out['original_full_sample_r2'].median():.4f}")
    overall_slope_corr = out["train_slope"].corr(out["test_period_local_slope"])
    print(f"Pooled train-slope vs. test-period-local-slope correlation: {overall_slope_corr:.3f}")


if __name__ == "__main__":
    main()
