"""
Stage 5 revisit, Step 1 - replace the Monte Carlo simulator's hard physics-only
cutoff (ahead = final_gap < 0) with an empirically-calibrated probability.

Computes, for every clean (non-SC-affected) attempt, what the PHYSICS-ONLY final
gap at the measurement lap would have been (reusing the exact same
_cumulative_pace_gain() function the live simulator uses, run deterministically
for n=1 with the attempt's REAL lap_a/lap_b/compounds - not a new formula), then
fits P(success | physics_final_gap, tire_age_a, tire_age_b, is_undercut_ahead) on
the TRAIN split only (years < 2025, same split as every other Stage 2 model) -
this is what the Monte Carlo simulator will now sample a Bernoulli outcome from
per draw, instead of asking a hard cutoff on Stage 1's honestly-weak degradation
slopes to carry the whole decision alone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import UNDERCUT_MODEL_TEST_YEARS
from src.paths import DERIVED_DIR, MODELS_DIR
from src.stage5_montecarlo import _cumulative_pace_gain, DEFAULT_RIVAL_COMPOUND_AFTER


def compute_physics_final_gap(row, ref_lookup, degradation_df, driver_tire_management=None) -> float | None:
    """The deterministic, single-draw version of what the Monte Carlo simulator
    computes stochastically: physics-only final gap at the REAL measurement lap,
    using the attempt's REAL lap_a/lap_b/compounds for the ego car and rival's
    PRE-stop state, but the SAME assumed rival post-stop compound
    (DEFAULT_RIVAL_COMPOUND_AFTER) the live simulator uses - trained and predicted
    under the identical assumption, so there's no train/predict mismatch.

    UPDATED for Step 2 of the Stage 5 accuracy-gap revisit: passes driver_a/
    driver_b through to _cumulative_pace_gain so the calibration model's
    physics_final_gap feature is computed the SAME way (driver-specific slopes
    where reliable) it will be at simulation time - keeping train and predict
    consistent, same reasoning as the DEFAULT_RIVAL_COMPOUND_AFTER assumption."""
    circuit = row["circuit"]
    if circuit not in ref_lookup:
        return None
    reference_laptime = ref_lookup[circuit]
    measurement_offset = np.array([row["measurement_lap"] - row["lap_a"]])
    rival_pit_offset = np.array([row["lap_b"] - row["lap_a"]])
    sc_windows = np.array([[10_000, 10_000]])  # "never" sentinel - clean attempts only, no SC to model
    try:
        pace_gain = _cumulative_pace_gain(
            circuit=circuit, own_tire_age=row["tire_age_a"], own_compound=row["compound_a_before"],
            own_stop_offset=0, candidate_compound=row["compound_a_after"],
            rival_tire_age=row["tire_age_b"], rival_compound=row["compound_b"],
            rival_compound_after=DEFAULT_RIVAL_COMPOUND_AFTER,
            rival_pit_offsets=rival_pit_offset, pit_loss_s=row["pit_loss_seconds"],
            sc_windows=sc_windows, measurement_offset=measurement_offset,
            reference_laptime=reference_laptime, degradation_model=degradation_df,
            driver_a=row["driver_a"], driver_b=row["driver_b"],
            driver_tire_management=driver_tire_management,
        )
    except KeyError:
        return None
    return float(row["gap_seconds"] - pace_gain[0])


def main():
    attempts = pd.read_csv(DERIVED_DIR / "undercut_overcut_attempts_enriched.csv")
    clean = attempts[~attempts["sc_vsc_affected"]].copy()
    clean["is_undercut_ahead"] = (clean["direction"] == "undercut_ahead").astype(int)

    ref_df = pd.read_csv(MODELS_DIR / "circuit_reference_laptimes.csv")
    ref_lookup = dict(zip(ref_df["circuit"], ref_df["typical_reference_laptime"]))
    degradation_df = pd.read_csv(MODELS_DIR / "degradation_model.csv")
    driver_tire_management = pd.read_csv(DERIVED_DIR / "driver_tire_management.csv")

    print(f"Computing physics_final_gap for {len(clean)} clean attempts "
          f"(using driver-specific tire management where reliable)...")
    clean["physics_final_gap"] = clean.apply(
        lambda r: compute_physics_final_gap(r, ref_lookup, degradation_df, driver_tire_management), axis=1
    )
    n_missing = clean["physics_final_gap"].isna().sum()
    print(f"{n_missing}/{len(clean)} attempts couldn't get a physics_final_gap "
          f"(missing degradation model for that circuit/compound) - dropped.")
    clean = clean.dropna(subset=["physics_final_gap", "tire_age_a", "tire_age_b"])

    is_test = clean["year"].isin(UNDERCUT_MODEL_TEST_YEARS)
    train, test = clean[~is_test], clean[is_test]
    print(f"\nTrain: {len(train)}, Test: {len(test)}")
    print(f"physics_final_gap stats (train): mean={train['physics_final_gap'].mean():.2f}, "
          f"std={train['physics_final_gap'].std():.2f}, "
          f"correlation with success: {train['physics_final_gap'].corr(train['success']):.3f}")

    FEATURES = ["physics_final_gap", "tire_age_a", "tire_age_b", "is_undercut_ahead"]
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(train[FEATURES], train["success"])

    coefs = pd.Series(pipe.named_steps["clf"].coef_[0], index=FEATURES)
    print("\n=== Calibration model coefficients (standardized) ===")
    print(coefs.sort_values(ascending=False).to_string())

    pred = pipe.predict(test[FEATURES])
    proba = pipe.predict_proba(test[FEATURES])[:, 1]
    print(f"\n=== Calibration model's OWN accuracy on the 286-attempt test set ===")
    print(f"(for reference only - this model isn't used standalone, it's plugged into the "
          f"Monte Carlo simulator's per-draw outcome sampling)")
    print(f"Accuracy: {accuracy_score(test['success'], pred):.3f}")
    print(f"AUC: {roc_auc_score(test['success'], proba):.3f}")
    print(f"Log-loss: {log_loss(test['success'], proba):.3f}")

    # Sanity check: does the calibration curve make sense? Bin physics_final_gap
    # and check real success rate per bin, vs. what a raw cutoff (gap<0) would say
    print("\n=== Sanity check: real success rate vs. physics_final_gap, binned (train) ===")
    train_copy = train.copy()
    train_copy["gap_bin"] = pd.cut(train_copy["physics_final_gap"],
                                     bins=[-100, -10, -3, 0, 3, 10, 100])
    print(train_copy.groupby("gap_bin", observed=True)["success"].agg(["count", "mean"]).to_string())
    print(f"\nRaw cutoff (physics_final_gap < 0) train accuracy: "
          f"{accuracy_score(train['success'], (train['physics_final_gap'] < 0).astype(int)):.3f}")

    joblib.dump(pipe, MODELS_DIR / "montecarlo_calibration_model.joblib")
    print(f"\nSaved -> {MODELS_DIR / 'montecarlo_calibration_model.joblib'}")


if __name__ == "__main__":
    main()
