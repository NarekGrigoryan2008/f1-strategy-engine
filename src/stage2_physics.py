"""
Stage 2, Part A - the physics-based undercut/overcut threshold.

This is deliberately the SIMPLE baseline, built from Stage 1's fitted constants
alone (no historical outcome labels), so Part B's empirically-fit model has
something principled to be compared against ("where do the model's predictions
agree/disagree with the physics threshold, and why").

The standard undercut argument, implemented directly: car A is currently `gap`
seconds behind car B (gap < 0 means A is already ahead). If A pits now, A
immediately loses `pit_loss` seconds (the one-time cost of the stop) but then
starts gaining ground each lap it spends on fresher tires than B, at a rate given
by the two cars' fitted degradation curves. A has gained track position once the
accumulated per-lap gain exceeds `gap + pit_loss`.

This intentionally does NOT model B's own future stop - it answers "does pitting
now put me ahead of a rival who keeps circulating on their current tires for the
laps in question," which is exactly the undercut/overcut question at the moment
the decision is made. B's eventual stop is a separate, later decision with its own
threshold calculation (from B's perspective, at whatever lap B actually considers
it) - collapsing both into one formula would require assuming a specific lap for
B's stop that the decision-maker, in reality, doesn't know in advance either.
For retrospective validation against real historical attempts (Stage 2 Part B),
the natural, apples-to-apples check is to evaluate this threshold at
K = (rival's actual pit lap - this car's actual pit lap), i.e. "did the physics
threshold correctly predict the outcome by the time the rival actually pitted?"
"""
from dataclasses import dataclass

import pandas as pd

from src.stage1_degradation import expected_laptime, pit_loss as get_pit_loss


@dataclass
class UndercutThresholdResult:
    circuit: str
    gap_seconds: float          # positive = the pitting car (A) is behind the rival (B)
    pit_loss_seconds: float
    laps_checked: int
    cumulative_gain_seconds: float  # total ground A gains on B over laps_checked laps
    breakeven_lap: int | None       # first lap number at which A's cumulative gain clears gap+pit_loss; None if never within laps_checked
    gains_position: bool             # True if cumulative_gain_seconds > gap + pit_loss at laps_checked


def undercut_physics_threshold(
    circuit: str,
    gap_seconds: float,
    tire_age_a_now: float,
    compound_a_current: str,
    compound_a_new: str,
    tire_age_b_now: float,
    compound_b: str,
    laps_to_check: int,
    reference_laptime: float,
    degradation_model_df: pd.DataFrame,
    pit_loss_df: pd.DataFrame,
) -> UndercutThresholdResult:
    """
    circuit: circuit slug
    gap_seconds: current gap from A to B (positive = A is behind B)
    tire_age_a_now / compound_a_current: A's tires AT THE MOMENT OF THE PIT DECISION
        (not actually used in the calculation itself - A is about to change tires -
        kept as an argument for callers/logging since it's part of the race state)
    compound_a_new: the compound A fits on this stop
    tire_age_b_now / compound_b: B's tires at the same moment, assumed to keep
        aging on the SAME compound for the laps being checked (B is not pitting
        in this window)
    laps_to_check: how many laps forward to accumulate A's fresh-tire advantage
        over (for retrospective validation, pass the real gap in laps between the
        two cars' actual pit stops)
    reference_laptime: this race's current pace level (see Stage 1 - the
        degradation model predicts a DELTA from this, not an absolute time)
    """
    pl = get_pit_loss(circuit, pit_loss_df)

    cumulative_gain = 0.0
    breakeven_lap = None
    threshold = gap_seconds + pl
    for k in range(1, laps_to_check + 1):
        a_time = expected_laptime(circuit, compound_a_new, tire_age=k - 1,
                                    reference_laptime=reference_laptime, model_df=degradation_model_df)
        b_time = expected_laptime(circuit, compound_b, tire_age=tire_age_b_now + (k - 1),
                                    reference_laptime=reference_laptime, model_df=degradation_model_df)
        cumulative_gain += (b_time - a_time)  # positive: A faster than B that lap, A gains ground
        if breakeven_lap is None and cumulative_gain > threshold:
            breakeven_lap = k

    return UndercutThresholdResult(
        circuit=circuit,
        gap_seconds=gap_seconds,
        pit_loss_seconds=pl,
        laps_checked=laps_to_check,
        cumulative_gain_seconds=cumulative_gain,
        breakeven_lap=breakeven_lap,
        gains_position=cumulative_gain > threshold,
    )
