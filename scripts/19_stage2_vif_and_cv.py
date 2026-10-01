"""
Stage 2 revisit, Part 1 (foundational checks, before any new features) -
1. Variance inflation factors for the current feature set, to check whether the
   counterintuitive tire_age_a coefficient (-0.78) is a multicollinearity
   artifact or a real, independent effect.
2. Expanding-window time-series cross-validation, to see whether the single
   74.8% chronological-split number is a stable estimate or one split's luck.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import COMPOUND_SOFTNESS_ORDER
from src.paths import DERIVED_DIR

FEATURE_COLUMNS = [
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
    return df


def main():
    df = pd.read_csv(DERIVED_DIR / "undercut_overcut_attempts.csv")
    clean = df[~df["sc_vsc_affected"]].copy()
    clean = build_features(clean)
    clean = clean.dropna(subset=FEATURE_COLUMNS + ["success"])
    print(f"n={len(clean)} clean attempts\n")

    # === 1. VIF ===
    print("=== Variance inflation factors (current feature set) ===")
    X = clean[FEATURE_COLUMNS].astype(float)
    X_std = (X - X.mean()) / X.std()  # standardize first, VIF is scale-sensitive in interpretation otherwise
    vifs = pd.Series(
        [variance_inflation_factor(X_std.values, i) for i in range(X_std.shape[1])],
        index=FEATURE_COLUMNS,
    )
    print(vifs.sort_values(ascending=False).to_string())
    print("\n(Rule of thumb: VIF > 5 suggests meaningful multicollinearity, > 10 is a real problem.)")

    print("\n=== Correlation of tire_age_a with other features ===")
    corrs = clean[FEATURE_COLUMNS].astype(float).corr()["tire_age_a"].sort_values(ascending=False)
    print(corrs.to_string())

    # Direct check: is tire_age_a correlated with lap_a (older own-tire attempts
    # skew later in the race)?
    print(f"\ntire_age_a vs lap_a correlation: {clean['tire_age_a'].corr(clean['lap_a']):.3f}")

    # Refit WITHOUT lap_a to see if tire_age_a's coefficient/sign changes -
    # a direct test of whether lap_a is absorbing/masking tire_age_a's real effect
    print("\n=== Refit without lap_a - does tire_age_a's coefficient change? ===")
    from src.config import UNDERCUT_MODEL_TEST_YEARS
    is_test = clean["year"].isin(UNDERCUT_MODEL_TEST_YEARS)
    train, test = clean[~is_test], clean[is_test]

    for feature_set, label in [(FEATURE_COLUMNS, "WITH lap_a"),
                                 ([f for f in FEATURE_COLUMNS if f != "lap_a"], "WITHOUT lap_a")]:
        pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
        pipe.fit(train[feature_set], train["success"])
        coefs = pd.Series(pipe.named_steps["clf"].coef_[0], index=feature_set)
        acc = accuracy_score(test["success"], pipe.predict(test[feature_set]))
        print(f"\n{label}: test accuracy={acc:.3f}")
        print(f"  tire_age_a coef: {coefs.get('tire_age_a', float('nan')):+.4f}")
        print(coefs.sort_values(ascending=False).to_string())

    # === 2. Expanding-window time-series CV ===
    print("\n\n=== Expanding-window cross-validation ===")
    folds = [
        (range(2018, 2023), [2023]),
        (range(2018, 2024), [2024]),
        (range(2018, 2025), [2025, 2026]),
    ]
    accs, aucs = [], []
    for train_years, test_years in folds:
        tr = clean[clean["year"].isin(train_years)]
        te = clean[clean["year"].isin(test_years)]
        if len(tr) == 0 or len(te) == 0:
            print(f"  train={list(train_years)[0]}-{list(train_years)[-1]}, test={test_years}: SKIPPED (empty split)")
            continue
        pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
        pipe.fit(tr[FEATURE_COLUMNS], tr["success"])
        pred = pipe.predict(te[FEATURE_COLUMNS])
        proba = pipe.predict_proba(te[FEATURE_COLUMNS])[:, 1]
        acc = accuracy_score(te["success"], pred)
        try:
            auc = roc_auc_score(te["success"], proba)
        except ValueError:
            auc = float("nan")
        accs.append(acc)
        aucs.append(auc)
        print(f"  train={list(train_years)[0]}-{list(train_years)[-1]} (n={len(tr)}), "
              f"test={test_years} (n={len(te)}): accuracy={acc:.3f}, AUC={auc:.3f}")

    print(f"\nMean accuracy across folds: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"Mean AUC across folds: {np.nanmean(aucs):.3f} +/- {np.nanstd(aucs):.3f}")
    print(f"(For reference, the single chronological split reported elsewhere: 0.748 accuracy, 0.763 AUC)")


if __name__ == "__main__":
    main()
