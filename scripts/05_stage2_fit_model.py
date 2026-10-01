"""
Stage 2, Part B (model-fitting step) - fit BOTH a logistic regression and a
shallow gradient-boosted model (LightGBM) on the clean (non-SC-affected) labeled
attempts, evaluate both on a genuinely held-out, chronological test set, and
compare both against the Part A physics-only threshold on the SAME test set.

Decided after seeing the real n=1368 clean-attempt sample size (2026-09-23):
fit both, report both honestly, let the held-out numbers - not a preference -
decide which one the tool actually uses.

UPDATED 2026-09-24 (Stage 2 revisit): feature set now includes
team_pit_speed_relative and gap_x_tire_age_b, added after testing four
candidate features individually - kept because both are
independently significant with sensible coefficient directions AND their
combination measurably reduces the model's year-to-year cross-validation
variance (0.025 -> 0.014 std across 3 expanding-window folds), not just because
they nudged a single split's accuracy. Reads from the enriched attempts table
(scripts/20_stage2_new_features.py) instead of the original.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.config import COMPOUND_SOFTNESS_ORDER, UNDERCUT_MODEL_TEST_YEARS
from src.paths import DERIVED_DIR, MODELS_DIR
from src.stage1_degradation import expected_laptime
from src.stage2_physics import undercut_physics_threshold

FEATURE_COLUMNS = [
    "gap_seconds", "tire_age_a", "tire_age_b", "pit_loss_seconds", "lap_a",
    "a_in_traffic", "is_undercut_ahead", "compound_advantage",
    "team_pit_speed_relative", "gap_x_tire_age_b",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_undercut_ahead"] = (df["direction"] == "undercut_ahead").astype(int)
    df["a_in_traffic"] = df["a_in_traffic"].astype(int)
    df["compound_advantage"] = (
        df["compound_b"].map(COMPOUND_SOFTNESS_ORDER)
        - df["compound_a_after"].map(COMPOUND_SOFTNESS_ORDER)
    )
    df["gap_x_tire_age_b"] = df["gap_seconds"] * df["tire_age_b"]
    return df


def main():
    df = pd.read_csv(DERIVED_DIR / "undercut_overcut_attempts_enriched.csv")
    clean = df[~df["sc_vsc_affected"]].copy()
    clean = build_features(clean)
    before = len(clean)
    clean = clean.dropna(subset=FEATURE_COLUMNS + ["success"])
    print(f"Clean attempts: {before} -> {len(clean)} after dropping rows with missing features "
          f"(unrecognized compound labels, missing tire age, or no team pit-speed estimate).\n")

    is_test = clean["year"].isin(UNDERCUT_MODEL_TEST_YEARS)
    train, test = clean[~is_test], clean[is_test]
    print(f"Train: {len(train)} attempts (years < {min(UNDERCUT_MODEL_TEST_YEARS)}), "
          f"success rate {train['success'].mean():.3f}")
    print(f"Test:  {len(test)} attempts (years {UNDERCUT_MODEL_TEST_YEARS}), "
          f"success rate {test['success'].mean():.3f}\n")

    X_train, y_train = train[FEATURE_COLUMNS], train["success"]
    X_test, y_test = test[FEATURE_COLUMNS], test["success"]

    # --- Logistic regression ---
    logreg = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    logreg.fit(X_train, y_train)
    logreg_pred = logreg.predict(X_test)
    logreg_proba = logreg.predict_proba(X_test)[:, 1]

    print("=== Logistic regression ===")
    print(f"Test accuracy: {accuracy_score(y_test, logreg_pred):.3f}")
    print(f"Test AUC: {roc_auc_score(y_test, logreg_proba):.3f}")
    print(f"Test log-loss: {log_loss(y_test, logreg_proba):.3f}")
    coefs = pd.Series(logreg.named_steps["clf"].coef_[0], index=FEATURE_COLUMNS)
    print("Standardized coefficients (positive = pushes toward success):")
    print(coefs.sort_values(ascending=False).to_string())

    # --- Shallow gradient-boosted model ---
    gbm = lgb.LGBMClassifier(
        n_estimators=100, max_depth=3, num_leaves=7, learning_rate=0.05,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=-1,
    )
    gbm.fit(X_train, y_train)
    gbm_pred = gbm.predict(X_test)
    gbm_proba = gbm.predict_proba(X_test)[:, 1]

    print("\n=== Shallow gradient-boosted (LightGBM) ===")
    print(f"Test accuracy: {accuracy_score(y_test, gbm_pred):.3f}")
    print(f"Test AUC: {roc_auc_score(y_test, gbm_proba):.3f}")
    print(f"Test log-loss: {log_loss(y_test, gbm_proba):.3f}")
    importances = pd.Series(gbm.feature_importances_, index=FEATURE_COLUMNS)
    print("Feature importances:")
    print(importances.sort_values(ascending=False).to_string())

    # --- Baseline: always predict the majority class ---
    majority = int(y_train.mean() > 0.5)
    baseline_acc = accuracy_score(y_test, [majority] * len(y_test))
    print(f"\n=== Baselines ===\nMajority-class baseline test accuracy: {baseline_acc:.3f} "
          f"(always predicts {'success' if majority else 'failure'})")

    # --- Part A physics threshold, evaluated on the SAME test set ---
    degradation_df = pd.read_csv(MODELS_DIR / "degradation_model.csv")
    pit_loss_df = pd.read_csv(MODELS_DIR / "pit_loss_model.csv")
    ref_df = pd.read_csv(MODELS_DIR / "circuit_reference_laptimes.csv")
    ref_lookup = dict(zip(ref_df["circuit"], ref_df["typical_reference_laptime"]))

    physics_preds = []
    physics_valid_mask = []
    for _, row in test.iterrows():
        try:
            ref = ref_lookup[row["circuit"]]
            laps_to_check = int(row["lap_b"] - row["lap_a"])
            result = undercut_physics_threshold(
                circuit=row["circuit"], gap_seconds=row["gap_seconds"],
                tire_age_a_now=row["tire_age_a"], compound_a_current=row["compound_a_before"],
                compound_a_new=row["compound_a_after"], tire_age_b_now=row["tire_age_b"],
                compound_b=row["compound_b"], laps_to_check=laps_to_check,
                reference_laptime=ref, degradation_model_df=degradation_df, pit_loss_df=pit_loss_df,
            )
            physics_preds.append(int(result.gains_position))
            physics_valid_mask.append(True)
        except KeyError:
            physics_preds.append(np.nan)
            physics_valid_mask.append(False)

    test_physics = test.copy()
    test_physics["physics_pred"] = physics_preds
    valid = test_physics.dropna(subset=["physics_pred"])
    print(f"\n=== Part A physics threshold, same test set ===")
    print(f"Evaluable on {len(valid)}/{len(test_physics)} test attempts "
          f"(missing degradation model for the rest's circuit/compound combo)")
    print(f"Physics-threshold test accuracy: {accuracy_score(valid['success'], valid['physics_pred']):.3f}")

    # Agreement/disagreement between physics and the fitted models, on the overlap
    comparison = test_physics.loc[valid.index].copy()
    comparison["logreg_pred"] = pd.Series(logreg_pred, index=test.index).loc[valid.index]
    comparison["gbm_pred"] = pd.Series(gbm_pred, index=test.index).loc[valid.index]
    agree_logreg = (comparison["physics_pred"] == comparison["logreg_pred"]).mean()
    agree_gbm = (comparison["physics_pred"] == comparison["gbm_pred"]).mean()
    print(f"Physics agrees with logistic regression on {agree_logreg*100:.1f}% of these attempts")
    print(f"Physics agrees with LightGBM on {agree_gbm*100:.1f}% of these attempts")

    out_path = DERIVED_DIR / "stage2_test_predictions.csv"
    comparison.to_csv(out_path, index=False)
    print(f"\nSaved full test-set comparison -> {out_path}")

    # Save both fitted (train-only) models AND a version of the chosen primary
    # model (logistic regression, chosen for its cross-validation stability)
    # refit on ALL clean data
    # (train+test combined). The train-only model is what produced the honest
    # test-set numbers reported above; the full-data refit is what the callable
    # tool actually uses, since there's no reason to withhold 2025-2026 data from
    # the deployed model once its accuracy has already been honestly measured.
    joblib.dump(logreg, MODELS_DIR / "undercut_model_logreg_trainonly.joblib")
    joblib.dump(gbm, MODELS_DIR / "undercut_model_lgbm_trainonly.joblib")

    X_all, y_all = clean[FEATURE_COLUMNS], clean["success"]
    logreg_full = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    logreg_full.fit(X_all, y_all)
    joblib.dump(logreg_full, MODELS_DIR / "undercut_model_logreg_deployed.joblib")
    print(f"\nSaved fitted models -> {MODELS_DIR}")
    print("  undercut_model_logreg_trainonly.joblib / undercut_model_lgbm_trainonly.joblib: "
          "produced the test-set numbers above")
    print("  undercut_model_logreg_deployed.joblib: logistic regression refit on ALL clean "
          "attempts (train+test) - this is what the final tool loads")


if __name__ == "__main__":
    main()
