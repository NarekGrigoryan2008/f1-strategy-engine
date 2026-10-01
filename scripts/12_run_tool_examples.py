"""Runs the final tool against a few realistic example race states and prints
the real output - the project's callable-tool deliverable, demonstrated."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tool import recommend_strategy


def show(title, rec):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(f"RECOMMENDATION: {rec.action.upper()} -> {rec.compound}")
    if rec.confidence is not None:
        print(f"Confidence: {rec.confidence*100:.0f}% chance of gaining/holding track position")
    print("\nReasoning:")
    for r in rec.reasoning:
        print(f"  - {r}")
    if rec.warnings:
        print("\nWarnings:")
        for w in rec.warnings:
            print(f"  ! {w}")
    print("\nAll options considered:")
    print(rec.all_options.to_string(index=False))
    if rec.monte_carlo_options is not None:
        print("\nMonte Carlo cross-check options (supplementary - see reasoning above):")
        print(rec.monte_carlo_options.to_string(index=False))


def main():
    # Scenario 1: classic undercut opportunity - chasing a rival 1.5s ahead,
    # both on aging MEDIUM tires, at Barcelona lap 25.
    show(
        "Scenario 1: Undercut attempt - 1.5s behind a rival, both on 18-lap MEDIUM, Barcelona lap 25",
        recommend_strategy(
            circuit="barcelona", current_lap=25, own_tire_age=18, own_compound="MEDIUM",
            candidate_compounds=["SOFT", "MEDIUM", "HARD"],
            rival_gap_seconds=1.5, rival_tire_age=18, rival_compound="MEDIUM",
        ),
    )

    # Scenario 2: defending a lead - 1.0s ahead of a rival on fresher tires, Monza lap 30
    show(
        "Scenario 2: Defending - 1.0s ahead, rival on FRESHER tires (10 vs your 22), Monza lap 30",
        recommend_strategy(
            circuit="monza", current_lap=30, own_tire_age=22, own_compound="HARD",
            candidate_compounds=["SOFT", "MEDIUM", "HARD"],
            rival_gap_seconds=-1.0, rival_tire_age=10, rival_compound="MEDIUM",
        ),
    )

    # Scenario 3: no rival specified - pure pace/tire-life strategy call, Suzuka lap 10
    show(
        "Scenario 3: No rival specified - pure strategy call, Suzuka lap 10, on 10-lap SOFT",
        recommend_strategy(
            circuit="suzuka", current_lap=10, own_tire_age=10, own_compound="SOFT",
            candidate_compounds=["SOFT", "MEDIUM", "HARD"],
        ),
    )

    # Scenario 4: rain in the forecast, Spa lap 15
    show(
        "Scenario 4: Rain probability flag - 40% rain chance, Spa-Francorchamps lap 15",
        recommend_strategy(
            circuit="spa_francorchamps", current_lap=15, own_tire_age=15, own_compound="MEDIUM",
            candidate_compounds=["SOFT", "MEDIUM", "HARD"],
            rival_gap_seconds=2.5, rival_tire_age=15, rival_compound="MEDIUM",
            rain_probability=0.40,
        ),
    )

    # Scenario 5: same as Scenario 1, but with Stage 5's Monte Carlo cross-check
    # enabled - shows the two models genuinely disagreeing on this exact race
    # state (logistic regression: SOFT at lookahead 3, 65%; Monte Carlo: HARD at
    # lookahead 0, 54%): Monte Carlo's physics-based accumulation over a longer
    # horizon favors HARD's flatter degradation curve differently than a
    # pattern-matched classifier does. The primary
    # recommendation is UNCHANGED by this flag - Monte Carlo is supplementary,
    # per its honest 60.8%-vs-74.8% validation result.
    show(
        "Scenario 5: Same as Scenario 1, with the Stage 5 Monte Carlo cross-check enabled",
        recommend_strategy(
            circuit="barcelona", current_lap=25, own_tire_age=18, own_compound="MEDIUM",
            candidate_compounds=["SOFT", "MEDIUM", "HARD"],
            rival_gap_seconds=1.5, rival_tire_age=18, rival_compound="MEDIUM",
            include_monte_carlo=True,
        ),
    )


if __name__ == "__main__":
    main()
