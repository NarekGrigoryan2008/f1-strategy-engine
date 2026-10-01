"""
Stage 2 revisit, Part 1 (feature testing) - tests each new feature INDIVIDUALLY
added to the baseline feature set first (isolated effect), then builds up a
final combined set from whichever features earn their place (real significance,
sensible coefficient direction, real accuracy gain) - not thrown in as a batch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import COMPOUND_SOFTNESS_ORDER, UNDERCUT_MODEL_TEST_YEARS
from src.paths import DERIVED_DIR

BASELINE_FEATURES = [
    "gap_seconds", "tire_age_a", "tire_age_b", "pit_loss_seconds", "lap_a",
    "a_in_traffic", "is_undercut_ahead", "compound_advantage",
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


def fit_and_eval(train, test, feature_set, label):
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(train[feature_set], train["success"])
    pred = pipe.predict(test[feature_set])
    proba = pipe.predict_proba(test[feature_set])[:, 1]
    acc = accuracy_score(test["success"], pred)
    auc = roc_auc_score(test["success"], proba)
    ll = log_loss(test["success"], proba)
    coefs = pd.Series(pipe.named_steps["clf"].coef_[0], index=feature_set)
    print(f"\n--- {label} ---")
    print(f"n_features={len(feature_set)}, test accuracy={acc:.3f}, AUC={auc:.3f}, log-loss={ll:.3f}")
    print(coefs.sort_values(ascending=False).to_string())
    return acc, auc, coefs


def significance_check(df, new_col, label):
    """Univariate-ish significance of a candidate feature, controlling for the
    baseline features, using statsmodels for a real p-value."""
    formula = f"success ~ {new_col} + " + " + ".join(BASELINE_FEATURES)
    model = smf.logit(formula, data=df).fit(disp=0)
    coef, se, p = model.params[new_col], model.bse[new_col], model.pvalues[new_col]
    print(f"\n{label}: coef={coef:+.5f}, SE={se:.5f}, p={p:.4f} "
          f"{'(significant at p<0.05)' if p < 0.05 else '(NOT significant at p<0.05)'}")
    return p


def main():
    attempts = pd.read_csv(DERIVED_DIR / "undercut_overcut_attempts_enriched.csv")
    clean = attempts[~attempts["sc_vsc_affected"]].copy()
    clean = build_features(clean)

    all_needed = BASELINE_FEATURES + ["team_pit_speed_relative", "b_out_traffic",
                                        "gap_x_tire_age_b", "race_progress", "success", "year"]
    before = len(clean)
    clean = clean.dropna(subset=all_needed)
    print(f"n={before} -> {len(clean)} after dropping rows missing any candidate feature\n")

    is_test = clean["year"].isin(UNDERCUT_MODEL_TEST_YEARS)
    train, test = clean[~is_test], clean[is_test]
    print(f"Train: {len(train)}, Test: {len(test)}\n")

    print("=" * 70)
    print("BASELINE")
    print("=" * 70)
    base_acc, base_auc, base_coefs = fit_and_eval(train, test, BASELINE_FEATURES, "Baseline (8 features)")

    print("\n" + "=" * 70)
    print("FEATURE 1: team_pit_speed_relative (individually added)")
    print("=" * 70)
    significance_check(clean, "team_pit_speed_relative", "team_pit_speed_relative")
    fit_and_eval(train, test, BASELINE_FEATURES + ["team_pit_speed_relative"], "Baseline + team_pit_speed_relative")

    print("\n" + "=" * 70)
    print("FEATURE 2: b_out_traffic (individually added)")
    print("=" * 70)
    significance_check(clean, "b_out_traffic", "b_out_traffic")
    fit_and_eval(train, test, BASELINE_FEATURES + ["b_out_traffic"], "Baseline + b_out_traffic")

    print("\n" + "=" * 70)
    print("FEATURE 3: gap_seconds * tire_age_b interaction (individually added)")
    print("=" * 70)
    significance_check(clean, "gap_x_tire_age_b", "gap_x_tire_age_b")
    fit_and_eval(train, test, BASELINE_FEATURES + ["gap_x_tire_age_b"], "Baseline + interaction")

    print("\n" + "=" * 70)
    print("FEATURE 4a: race_progress ADDED alongside lap_a")
    print("=" * 70)
    significance_check(clean, "race_progress", "race_progress")
    fit_and_eval(train, test, BASELINE_FEATURES + ["race_progress"], "Baseline + race_progress")

    print("\n" + "=" * 70)
    print("FEATURE 4b: race_progress REPLACING lap_a")
    print("=" * 70)
    replaced = [f for f in BASELINE_FEATURES if f != "lap_a"] + ["race_progress"]
    fit_and_eval(train, test, replaced, "Baseline with lap_a -> race_progress")


if __name__ == "__main__":
    main()
