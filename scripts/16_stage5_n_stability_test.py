"""
Stage 5 - the N-simulations question, tested rather than picked as a round
number. For several candidate N values: measure real wall-clock runtime, and
measure result STABILITY by re-running the same scenario with 20 different
random seeds per N and reporting the spread (std) of the resulting p_ahead
estimate across those repeats - the real tradeoff to check before choosing,
rather than picking a round number and hoping.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from src.stage5_montecarlo import simulate_candidate


def main():
    degradation = pd.read_csv("data/models/degradation_model.csv")
    pit_loss_df = pd.read_csv("data/models/pit_loss_model.csv")
    ref_df = pd.read_csv("data/models/circuit_reference_laptimes.csv")
    hazard_model = joblib.load("data/models/pit_hazard_model.joblib")
    sc_hazard_df = pd.read_csv("data/models/sc_vsc_hazard_by_circuit_lap.csv")
    sc_periods_df = pd.read_csv("data/derived/sc_vsc_periods_autodetected.csv")

    # NOTE: an initial pass at these scenarios (both "coming from behind" cases)
    # showed p_ahead=0.0 deterministically at every N - not a bug, a real
    # consequence of pit_loss (~20-25s) dwarfing any realistic gap/degradation
    # swing, which makes "overcome a deficit" scenarios
    # useless for testing STABILITY (a constant 0 has no sampling noise to
    # measure). Switched to "defending" scenarios instead, where the outcome
    # genuinely hinges on the stochastic pit-timing/SC draws (both cars pay a
    # similar pit-loss cost if both eventually pit, so the small pre-existing
    # gap and the exact timing dominate) - these actually exercise the
    # randomness the stability test is supposed to characterize.
    scenarios = [
        dict(circuit="baku", current_lap=20, own_tire_age=10, own_compound="MEDIUM",
             lookahead_laps=2, candidate_compound="SOFT",
             rival_gap_seconds=-1.0, rival_tire_age=10, rival_compound="MEDIUM"),
        dict(circuit="barcelona", current_lap=25, own_tire_age=18, own_compound="MEDIUM",
             lookahead_laps=3, candidate_compound="SOFT",
             rival_gap_seconds=-0.5, rival_tire_age=18, rival_compound="MEDIUM"),
    ]

    n_values = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    n_repeats = 20

    for scenario in scenarios:
        circuit = scenario["circuit"]
        ref = ref_df[ref_df["circuit"] == circuit]["typical_reference_laptime"].iloc[0]
        print(f"\n{'='*70}\nScenario: {circuit}, gap={scenario['rival_gap_seconds']}s\n{'='*70}")
        print(f"{'N':>8} {'runtime_ms':>12} {'mean_p_ahead':>14} {'std_p_ahead':>13} {'95%_range':>20}")

        for n in n_values:
            t0 = time.time()
            p_aheads = []
            for seed in range(n_repeats):
                result = simulate_candidate(
                    **scenario, reference_laptime=ref, horizon_laps=12, n_simulations=n,
                    degradation_model=degradation, pit_loss_df=pit_loss_df, hazard_model=hazard_model,
                    sc_hazard_df=sc_hazard_df, sc_periods_df=sc_periods_df, seed=seed,
                )
                p_aheads.append(result.p_ahead)
            elapsed_per_call = (time.time() - t0) / n_repeats * 1000
            p_aheads = np.array(p_aheads)
            lo, hi = np.percentile(p_aheads, [2.5, 97.5])
            print(f"{n:>8} {elapsed_per_call:>10.1f}ms {p_aheads.mean():>14.4f} {p_aheads.std():>13.4f} "
                  f"[{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
