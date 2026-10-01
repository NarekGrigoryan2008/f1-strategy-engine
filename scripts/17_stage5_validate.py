"""
Stage 5 - retrospective validation, same standard as Part A vs Part B.

For each of the 286 real, held-out (2025-2026) undercut/overcut attempts already
used to evaluate Stage 2's physics threshold and logistic regression, run the
Monte Carlo simulator AS IF evaluating that decision live - using only the state
that would have been available at the moment of the real decision (own tire age/
compound, the gap, the rival's tire age/compound), for the REAL strategy that was
actually chosen (lookahead=0, the real post-stop compound) - and compare its
predicted P(ahead) against the real outcome. This is deliberately NOT peeking at
the rival's real future pit lap - that's exactly what the Stage 5 hazard model
exists to forecast instead.

UPDATED for the Stage 5 accuracy-gap revisit: now loads and uses Step 1's
calibration model (scripts/22_stage5_fit_calibration.py) by default, so this
script's output reflects the CURRENT simulator, not the pre-revisit one. Pass
--no-calibration to reproduce the original (pre-revisit) hard-cutoff numbers.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from src.stage5_montecarlo import simulate_candidate, DEFAULT_N_SIMULATIONS

HORIZON_LAPS = 25  # covers the 95th percentile of real lap_b - lap_a gaps in the test set


def main():
    use_calibration = "--no-calibration" not in sys.argv
    use_driver_tire_mgmt = "--no-driver-tire-mgmt" not in sys.argv
    test = pd.read_csv("data/derived/stage2_test_predictions.csv")
    degradation = pd.read_csv("data/models/degradation_model.csv")
    pit_loss_df = pd.read_csv("data/models/pit_loss_model.csv")
    ref_df = pd.read_csv("data/models/circuit_reference_laptimes.csv")
    ref_lookup = dict(zip(ref_df["circuit"], ref_df["typical_reference_laptime"]))
    hazard_model = joblib.load("data/models/pit_hazard_model.joblib")
    sc_hazard_df = pd.read_csv("data/models/sc_vsc_hazard_by_circuit_lap.csv")
    sc_periods_df = pd.read_csv("data/derived/sc_vsc_periods_autodetected.csv")
    calibration_model = None
    if use_calibration:
        cal_path = Path("data/models/montecarlo_calibration_model.joblib")
        if cal_path.exists():
            calibration_model = joblib.load(cal_path)
            print("Using Step 1 calibration model for per-draw outcome sampling.")
        else:
            print("No calibration model found - falling back to the hard physics cutoff.")
    driver_tire_management = None
    if use_driver_tire_mgmt:
        driver_tire_management = pd.read_csv("data/derived/driver_tire_management.csv")
        print("Using Step 2 driver-specific tire management where reliable.")
    print()

    print(f"Running Monte Carlo (N={DEFAULT_N_SIMULATIONS}) on {len(test)} real held-out attempts...")
    t0 = time.time()

    mc_p_ahead = []
    errors = 0
    for i, row in test.iterrows():
        try:
            ref = ref_lookup[row["circuit"]]
            result = simulate_candidate(
                circuit=row["circuit"], current_lap=int(row["lap_a"]),
                own_tire_age=row["tire_age_a"], own_compound=row["compound_a_before"],
                lookahead_laps=0, candidate_compound=row["compound_a_after"],
                rival_gap_seconds=row["gap_seconds"], rival_tire_age=row["tire_age_b"],
                rival_compound=row["compound_b"],
                reference_laptime=ref, horizon_laps=HORIZON_LAPS, n_simulations=DEFAULT_N_SIMULATIONS,
                degradation_model=degradation, pit_loss_df=pit_loss_df, hazard_model=hazard_model,
                sc_hazard_df=sc_hazard_df, sc_periods_df=sc_periods_df,
                calibration_model=calibration_model,
                driver_a=row["driver_a"], driver_b=row["driver_b"],
                driver_tire_management=driver_tire_management,
                rival_in_traffic=bool(row["a_in_traffic"]), seed=i,
            )
            mc_p_ahead.append(result.p_ahead)
        except (KeyError, ValueError) as e:
            mc_p_ahead.append(np.nan)
            errors += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test)}...")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/len(test)*1000:.0f}ms/attempt). {errors} errors "
          f"(missing degradation model for that circuit/compound combo).")

    test["mc_p_ahead"] = mc_p_ahead
    test["mc_pred"] = (test["mc_p_ahead"] > 0.5).astype("Int64")
    test.loc[test["mc_p_ahead"].isna(), "mc_pred"] = pd.NA

    valid = test.dropna(subset=["mc_p_ahead"])
    print(f"\n=== Monte Carlo simulator, {len(valid)}/{len(test)} evaluable attempts ===")
    print(f"Accuracy: {accuracy_score(valid['success'], valid['mc_pred']):.3f}")
    print(f"P(ahead) distribution: min={valid['mc_p_ahead'].min():.3f}, "
          f"max={valid['mc_p_ahead'].max():.3f}, mean={valid['mc_p_ahead'].mean():.3f}, "
          f"n_unique_values={valid['mc_p_ahead'].nunique()}")
    try:
        auc = roc_auc_score(valid["success"], valid["mc_p_ahead"])
        print(f"AUC: {auc:.3f}")
        ll = log_loss(valid["success"], valid["mc_p_ahead"].clip(1e-4, 1 - 1e-4))
        print(f"Log-loss: {ll:.3f}")
    except ValueError as e:
        print(f"AUC/log-loss not computable: {e}")

    print("\n=== Comparison to already-established results on the SAME test set ===")
    print(f"Physics threshold (Part A):     accuracy {accuracy_score(valid['success'], valid['physics_pred']):.3f}")
    print(f"Logistic regression (Part B):   accuracy {accuracy_score(valid['success'], valid['logreg_pred']):.3f}")
    print(f"LightGBM (Part B):               accuracy {accuracy_score(valid['success'], valid['gbm_pred']):.3f}")
    print(f"Monte Carlo (Stage 5):           accuracy {accuracy_score(valid['success'], valid['mc_pred']):.3f}")
    majority = int(valid['success'].mean() > 0.5)
    print(f"Majority-class baseline:         accuracy {accuracy_score(valid['success'], [majority]*len(valid)):.3f}")

    print("\n=== Where does Monte Carlo agree/disagree with logistic regression? ===")
    agree = (valid["mc_pred"] == valid["logreg_pred"]).mean()
    print(f"Agreement rate: {agree*100:.1f}%")
    both_right = ((valid["mc_pred"] == valid["success"]) & (valid["logreg_pred"] == valid["success"])).sum()
    mc_only_right = ((valid["mc_pred"] == valid["success"]) & (valid["logreg_pred"] != valid["success"])).sum()
    logreg_only_right = ((valid["mc_pred"] != valid["success"]) & (valid["logreg_pred"] == valid["success"])).sum()
    both_wrong = ((valid["mc_pred"] != valid["success"]) & (valid["logreg_pred"] != valid["success"])).sum()
    print(f"Both correct: {both_right}, MC only correct: {mc_only_right}, "
          f"logreg only correct: {logreg_only_right}, both wrong: {both_wrong}")

    out_path = Path("data/derived/stage5_validation_results.csv")
    test.to_csv(out_path, index=False)
    print(f"\nSaved full results -> {out_path}")


if __name__ == "__main__":
    main()
