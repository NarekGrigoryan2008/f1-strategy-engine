"""Fit the pit-hazard logistic regression and check, honestly, which features the
157K-row dataset actually supports before finalizing the model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.paths import DERIVED_DIR, MODELS_DIR


def main():
    df = pd.read_csv(DERIVED_DIR / "pit_hazard_dataset.csv")
    df = df[df["compound"] != "UNKNOWN"].copy()  # 40 rows, data artifact, not a real compound
    df["has_close_ahead"] = df["gap_ahead_seconds"].notna().astype(int)
    df["has_close_behind"] = df["gap_behind_seconds"].notna().astype(int)
    print(f"{len(df)} rows after dropping UNKNOWN-compound artifact rows.")

    print("\nFitting: pits_next_lap ~ tire_age + tire_age^2 + compound + circuit + lap "
          "+ has_close_ahead + has_close_behind")
    model = smf.logit(
        "pits_next_lap ~ tire_age + I(tire_age**2) + C(compound) + C(circuit) + lap "
        "+ has_close_ahead + has_close_behind",
        data=df,
    ).fit(disp=0)

    print(f"\nPseudo R2: {model.prsquared:.4f}  (n={int(model.nobs)})")
    key_terms = ["Intercept", "tire_age", "I(tire_age ** 2)", "lap", "has_close_ahead", "has_close_behind"]
    print("\nKey coefficients (circuit/compound fixed effects omitted from display):")
    for t in key_terms:
        if t in model.params.index:
            print(f"  {t:20s} coef={model.params[t]:+.5f}  p={model.pvalues[t]:.4f}")

    out_path = DERIVED_DIR / "stage5_pit_hazard_model_summary.txt"
    with open(out_path, "w") as f:
        f.write(str(model.summary()))
    print(f"\nFull summary saved -> {out_path}")

    # Sanity check: predicted hazard curve vs tire_age, holding other features at
    # a representative value (MEDIUM, budapest, lap 30, no close rival), vs the
    # RAW empirical rate from the same script's earlier inspection - do they agree?
    check_ages = [3, 8, 13, 18, 23, 28, 33]
    checks = pd.DataFrame({
        "tire_age": check_ages, "compound": "MEDIUM", "circuit": "budapest",
        "lap": 30, "has_close_ahead": 0, "has_close_behind": 0,
    })
    checks["predicted_hazard"] = model.predict(checks)
    print("\nPredicted per-lap pit hazard vs tire age (MEDIUM, Budapest, lap 30, no close rival):")
    print(checks.to_string())

    # Save the model
    joblib.dump(model, MODELS_DIR / "pit_hazard_model.joblib")
    print(f"\nSaved fitted hazard model -> {MODELS_DIR / 'pit_hazard_model.joblib'}")


if __name__ == "__main__":
    main()
