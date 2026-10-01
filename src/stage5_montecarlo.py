"""
Stage 5 - the Monte Carlo strategy simulator.

For a candidate strategy (box at a specific lap, on a specific compound), runs N
simulated continuations of the race and returns a DISTRIBUTION of outcomes, not a
point estimate. What's sampled per draw, and from what real fitted source:

  - WHETHER/WHEN the rival pits during the simulated window: sampled from the
    per-lap hazard model fit in src/stage5_pit_hazard.py on 157,538 real
    driver-laps - NOT assumed fixed, which was the old tool's documented flaw.
  - WHETHER/WHEN a Safety Car or VSC period starts: sampled from Stage 3's real
    per-circuit, per-lap hazard curve, with duration sampled from the real
    historical distribution of SC/VSC period lengths (not a fixed guess).
  - Weather: NOT simulated here. Where a live rain probability is supplied, the
    existing rain-flag logic (unchanged from the Stage 1-4 tool) still applies;
    otherwise conditions are held constant for the simulated window, per the
    project's decision not to build new weather-transition modeling for Stage 5.

What ISN'T stochastic: the candidate strategy itself (that's what's being
evaluated) and the ego car's own compound choice. SC/VSC-affected laps freeze
the relative tire-degradation-driven pace delta between the two cars for that
lap (both are assumed to run at the same controlled pace) but do NOT discount
pit-loss - measured directly in scripts/15_stage5_sc_pit_loss.py and found not
to apply to close-quarters attempts.

Sign convention, kept consistent with the rest of the project (Stage 2 Part A,
src/tool.py): gap_seconds = ego's cumulative time minus rival's - POSITIVE means
the ego car is BEHIND the rival.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.stage1_degradation import pit_loss as get_pit_loss

DEFAULT_RIVAL_COMPOUND_AFTER = "MEDIUM"  # documented simplification - the real choice isn't knowable in advance
MEASUREMENT_OFFSET_LAPS = 3  # matches Stage 2's success rule
DEFAULT_N_SIMULATIONS = 20000  # chosen from real runtime/stability numbers (scripts/16_stage5_n_stability_test.py)


def _pit_hazard_sequence(hazard_model, circuit: str, compound: str, tire_age_start: float,
                          lap_start: int, has_close_ahead: bool, has_close_behind: bool,
                          horizon: int) -> np.ndarray:
    """The rival's per-lap hazard for each of the next `horizon` laps, ASSUMING
    the rival hasn't pitted yet at each point (their tire just keeps aging along
    the single "hasn't pitted" trajectory - this project doesn't model the rival
    changing compound mid-simulation before their eventual stop). Returns an
    array of length `horizon`."""
    laps = pd.DataFrame({
        "tire_age": [tire_age_start + k for k in range(1, horizon + 1)],
        "compound": compound, "circuit": circuit,
        "lap": [lap_start + k for k in range(1, horizon + 1)],
        "has_close_ahead": int(has_close_ahead), "has_close_behind": int(has_close_behind),
    })
    return hazard_model.predict(laps).to_numpy()


def _discrete_offset_distribution(hazard_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turns a per-lap hazard sequence into a discrete distribution over "event
    happens at offset k" (k=1..len(hazard)) plus "doesn't happen within the
    horizon" (offset = len+1, a sentinel). Standard discrete-time survival
    construction: P(event at k) = hazard[k] * prod_{j<k}(1 - hazard[j])."""
    survival = np.cumprod(1 - hazard_sequence)
    survival_before = np.concatenate([[1.0], survival[:-1]])
    p_at_k = hazard_sequence * survival_before
    p_never = survival[-1]
    offsets = np.arange(1, len(hazard_sequence) + 2)  # last is the "never" sentinel
    probs = np.concatenate([p_at_k, [p_never]])
    probs = probs / probs.sum()  # guard against float drift
    return offsets, probs


def sample_rival_pit_offsets(hazard_model, circuit: str, compound: str, tire_age_start: float,
                              lap_start: int, has_close_ahead: bool, has_close_behind: bool,
                              horizon: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """N samples of "laps from now the rival pits" (a sentinel value of
    horizon+1 means "doesn't pit within the window")."""
    hazard_seq = _pit_hazard_sequence(hazard_model, circuit, compound, tire_age_start,
                                        lap_start, has_close_ahead, has_close_behind, horizon)
    offsets, probs = _discrete_offset_distribution(hazard_seq)
    return rng.choice(offsets, size=n, p=probs)


def sample_sc_windows(hazard_df: pd.DataFrame, periods_df: pd.DataFrame, circuit: str,
                       lap_start: int, horizon: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """N samples of (start_offset, end_offset) for the first SC/VSC period within
    the window, as an (n, 2) array; a start_offset of horizon+1 means "no SC/VSC
    in this simulated window". SC and VSC are combined into one "track
    compromised" event (this project doesn't distinguish their effect on relative
    pace - both mean "everyone running the same controlled pace"). Start lap
    sampled from Stage 3's real hazard curve; duration sampled (with
    replacement) from the real historical distribution of SC/VSC period lengths
    for this circuit if there's enough of it (>=10 real periods), else pooled
    across all circuits."""
    hazard = hazard_df[(hazard_df["circuit"] == circuit) & (hazard_df["type"].isin(["SC", "VSC"]))
                        & (hazard_df["lap"] > lap_start) & (hazard_df["lap"] <= lap_start + horizon)]
    per_lap_hazard = hazard.groupby("lap")["smoothed_hazard"].sum().clip(upper=1.0)
    seq = np.array([per_lap_hazard.get(lap_start + k, 0.0) for k in range(1, horizon + 1)])

    offsets, probs = _discrete_offset_distribution(seq)
    start_offsets = rng.choice(offsets, size=n, p=probs)

    durations_this_circuit = periods_df[(periods_df["circuit"] == circuit)
                                          & periods_df["type"].isin(["SC", "VSC"])]["duration_laps"]
    duration_pool = durations_this_circuit if len(durations_this_circuit) >= 10 else \
        periods_df[periods_df["type"].isin(["SC", "VSC"])]["duration_laps"]
    sampled_durations = rng.choice(duration_pool.to_numpy(), size=n)

    end_offsets = np.where(start_offsets > horizon, start_offsets,
                            start_offsets + sampled_durations - 1)
    return np.stack([start_offsets, end_offsets], axis=1)


def _slope_intercept(circuit: str, compound: str, degradation_model: pd.DataFrame,
                      driver: str | None = None, driver_tire_management: pd.DataFrame | None = None
                      ) -> tuple[float, float, bool]:
    """A plain (slope, intercept, used_driver_specific) lookup - used instead of
    calling expected_laptime() per-element in the hot loop below, since that
    function's pandas boolean-mask filter makes it far too slow to call
    thousands of times per simulation (found directly: the first version of this
    function took long enough on a single 5,000-draw call that it was killed and
    rewritten, not guessed at from first principles).

    Step 2 of the Stage 5 accuracy-gap revisit: when `driver` and
    `driver_tire_management` are supplied and that (driver, circuit, compound)
    combination has a `reliable=True` row (data/derived/driver_tire_management.csv,
    Stage 1's per-driver refit vs. the circuit-wide baseline), uses the DRIVER'S
    OWN fitted slope instead of the circuit-wide average - the intercept always
    stays the circuit baseline's, since driver_tire_management.csv only re-fits
    the slope (see its own build notes), not a driver-specific intercept. Falls
    back to the circuit baseline (both slope and intercept) whenever no reliable
    driver-specific row exists - this is NOT an error case, most (driver,
    circuit, compound) combinations don't have enough laps to trust a personal
    fit, and falling back silently to the baseline is the documented, intended
    behavior, not a degraded state.
    """
    row = degradation_model[(degradation_model["circuit"] == circuit)
                              & (degradation_model["compound"] == compound)]
    if row.empty:
        raise KeyError(f"No fitted degradation model for circuit={circuit!r}, compound={compound!r}")
    r = row.iloc[0]
    slope, intercept = float(r["slope"]), float(r["intercept"])
    used_driver_specific = False
    if driver is not None and driver_tire_management is not None:
        d_row = driver_tire_management[
            (driver_tire_management["driver"] == driver) & (driver_tire_management["circuit"] == circuit)
            & (driver_tire_management["compound"] == compound) & (driver_tire_management["reliable"])
        ]
        if not d_row.empty:
            slope = float(d_row.iloc[0]["driver_slope"])
            used_driver_specific = True
    return slope, intercept, used_driver_specific


def _cumulative_pace_gain(circuit, own_tire_age, own_compound, own_stop_offset, candidate_compound,
                           rival_tire_age, rival_compound, rival_compound_after,
                           rival_pit_offsets, pit_loss_s, sc_windows, measurement_offset,
                           reference_laptime, degradation_model,
                           driver_a=None, driver_b=None, driver_tire_management=None) -> np.ndarray:
    """Per simulation draw: (rival's total elapsed time) - (own total elapsed
    time) over the window from now to that draw's measurement lap. Positive
    means the EGO car was net faster (gains ground - closes/reverses the gap).

    Vectorized over the N simulation draws (all the per-element work below is
    plain numpy arithmetic, not a Python-level call per draw) - only loops over
    the (small, ~10-20) lap offsets within the horizon.

    `driver_a`/`driver_b`/`driver_tire_management`: optional (Step 2 of the
    Stage 5 accuracy-gap revisit) - when supplied, each of the four degradation
    legs (own's current compound, own's new compound, rival's current compound,
    rival's assumed post-stop compound) independently uses that driver's own
    fitted slope where a reliable one exists, falling back to the circuit
    baseline leg-by-leg otherwise (see _slope_intercept).
    """
    n = len(rival_pit_offsets)
    own_slope, own_intercept, _ = _slope_intercept(circuit, own_compound, degradation_model,
                                                      driver_a, driver_tire_management)
    new_slope, new_intercept, _ = _slope_intercept(circuit, candidate_compound, degradation_model,
                                                      driver_a, driver_tire_management)
    rival_slope, rival_intercept, _ = _slope_intercept(circuit, rival_compound, degradation_model,
                                                          driver_b, driver_tire_management)
    rival_in_slope, rival_in_intercept, _ = _slope_intercept(circuit, rival_compound_after, degradation_model,
                                                                driver_b, driver_tire_management)

    # own_stop_offset == 0 means "pits at the current lap, right now" - that
    # stop happens BEFORE the k=1..horizon loop below (which only covers laps
    # AFTER the current one), so its cost is charged upfront here rather than
    # inside the loop, where a plain `k == own_stop_offset` check would never
    # match for k starting at 1 (found and fixed via a direct sanity check).
    own_total = np.full(n, pit_loss_s if own_stop_offset == 0 else 0.0)
    rival_total = np.zeros(n)
    max_k = int(measurement_offset.max())

    for k in range(1, max_k + 1):
        active = k <= measurement_offset
        if not active.any():
            continue
        sc_active = (k >= sc_windows[:, 0]) & (k <= sc_windows[:, 1]) & active

        if k <= own_stop_offset:
            own_lap_time = reference_laptime + own_slope * (own_tire_age + k) + own_intercept
        else:
            own_lap_time = reference_laptime + new_slope * (k - own_stop_offset) + new_intercept
        own_pace = np.where(sc_active, 0.0, own_lap_time)
        own_stop_cost = np.where(active & (k == own_stop_offset) & (own_stop_offset > 0), pit_loss_s, 0.0)
        own_total += np.where(active, own_pace, 0.0) + own_stop_cost

        rival_pitted_by_k = rival_pit_offsets <= k
        rival_lap_time_out = reference_laptime + rival_slope * (rival_tire_age + k) + rival_intercept
        rival_ages_in = np.maximum(k - rival_pit_offsets, 0)
        rival_lap_time_in = reference_laptime + rival_in_slope * rival_ages_in + rival_in_intercept
        rival_lap_time = np.where(rival_pitted_by_k, rival_lap_time_in, rival_lap_time_out)
        rival_pace = np.where(sc_active, 0.0, rival_lap_time)
        rival_stop_cost = np.where(active & (rival_pit_offsets == k), pit_loss_s, 0.0)
        rival_total += np.where(active, rival_pace, 0.0) + rival_stop_cost

    return rival_total - own_total


@dataclass
class SimulationResult:
    lookahead_laps: int
    compound: str
    n_simulations: int
    p_ahead: float
    mean_gap_seconds: float       # positive = ego ahead by this much, on average, at the measurement lap
    gap_std_seconds: float
    p_rival_pits_within_window: float
    p_sc_vsc_in_window: float


def simulate_candidate(
    circuit: str, current_lap: int, own_tire_age: float, own_compound: str,
    lookahead_laps: int, candidate_compound: str,
    rival_gap_seconds: float, rival_tire_age: float, rival_compound: str,
    reference_laptime: float, horizon_laps: int, n_simulations: int,
    degradation_model: pd.DataFrame, pit_loss_df: pd.DataFrame, hazard_model,
    sc_hazard_df: pd.DataFrame, sc_periods_df: pd.DataFrame,
    calibration_model=None,
    driver_a: str | None = None, driver_b: str | None = None,
    driver_tire_management: pd.DataFrame | None = None,
    rival_compound_after: str = DEFAULT_RIVAL_COMPOUND_AFTER,
    rival_in_traffic: bool = False, seed: int | None = None,
) -> SimulationResult:
    """
    `calibration_model`: the Step-1 revisit's fitted logistic regression
    (scripts/22_stage5_fit_calibration.py) mapping physics_final_gap +
    tire_age_a + tire_age_b + is_undercut_ahead to a REAL empirical success
    probability. When supplied, each draw's outcome is a Bernoulli sample from
    this calibrated probability (evaluated at that draw's own simulated
    physics_final_gap) instead of a hard physics_final_gap < 0 cutoff. The hard
    cutoff badly underperforms real outcomes, because it asks Stage 1's
    honestly-weak degradation slopes to carry the whole decision alone. When
    None (the pre-revisit default), falls back to the original hard
    cutoff, so old callers / the original validation run remain reproducible.

    `driver_a`/`driver_b`/`driver_tire_management`: Step 2 of the Stage 5
    accuracy-gap revisit - when supplied, uses each driver's own fitted
    degradation slope (data/derived/driver_tire_management.csv) in place of the
    circuit-wide baseline, wherever a reliable one exists (leg by leg - own
    current/new compound, rival's current/assumed compound - falling back to
    the baseline for any leg without a reliable driver-specific fit).
    """
    if lookahead_laps > horizon_laps:
        raise ValueError(f"lookahead_laps ({lookahead_laps}) must be <= horizon_laps ({horizon_laps})")

    rng = np.random.default_rng(seed)
    pit_loss_s = get_pit_loss(circuit, pit_loss_df)

    rival_pit_offsets = sample_rival_pit_offsets(
        hazard_model, circuit, rival_compound, rival_tire_age, current_lap,
        has_close_ahead=not rival_in_traffic, has_close_behind=True,  # ego is the "close" car from rival's side
        horizon=horizon_laps, n=n_simulations, rng=rng,
    )
    sc_windows = sample_sc_windows(sc_hazard_df, sc_periods_df, circuit, current_lap,
                                     horizon_laps, n_simulations, rng)

    measurement_offset = np.minimum(
        np.maximum(lookahead_laps, np.minimum(rival_pit_offsets, horizon_laps)) + MEASUREMENT_OFFSET_LAPS,
        horizon_laps,
    )

    pace_gain = _cumulative_pace_gain(
        circuit, own_tire_age, own_compound, lookahead_laps, candidate_compound,
        rival_tire_age, rival_compound, rival_compound_after, rival_pit_offsets, pit_loss_s,
        sc_windows, measurement_offset, reference_laptime, degradation_model,
        driver_a=driver_a, driver_b=driver_b, driver_tire_management=driver_tire_management,
    )
    final_gap = rival_gap_seconds - pace_gain  # positive = ego still behind at measurement lap

    if calibration_model is not None:
        tire_age_a_at_stop = own_tire_age + lookahead_laps
        features = pd.DataFrame({
            "physics_final_gap": final_gap,
            "tire_age_a": tire_age_a_at_stop,
            "tire_age_b": rival_tire_age,
            "is_undercut_ahead": int(rival_gap_seconds > 0),
        })
        p_success = calibration_model.predict_proba(features)[:, 1]
        ahead = rng.random(n_simulations) < p_success
    else:
        ahead = final_gap < 0

    return SimulationResult(
        lookahead_laps=lookahead_laps, compound=candidate_compound, n_simulations=n_simulations,
        p_ahead=float(ahead.mean()), mean_gap_seconds=float(-final_gap.mean()),
        gap_std_seconds=float(final_gap.std()),
        p_rival_pits_within_window=float((rival_pit_offsets <= horizon_laps).mean()),
        p_sc_vsc_in_window=float((sc_windows[:, 0] <= horizon_laps).mean()),
    )


def scan_strategies_montecarlo(
    circuit: str, current_lap: int, own_tire_age: float, own_compound: str,
    candidate_compounds: list[str], lookaheads: tuple[int, ...],
    rival_gap_seconds: float, rival_tire_age: float, rival_compound: str,
    reference_laptime: float, artifacts: dict,
    horizon_laps: int = 15, n_simulations: int = DEFAULT_N_SIMULATIONS,
    driver_a: str | None = None, driver_b: str | None = None,
    rival_in_traffic: bool = False, seed: int | None = None,
) -> pd.DataFrame:
    """Runs simulate_candidate() over every (lookahead, compound) combination and
    returns one row per option, ranked by p_ahead - the Monte Carlo counterpart
    to src/tool.py's deterministic option scan.

    Accuracy on real held-out attempts, after the Stage 5 accuracy-gap revisit
    (calibrated outcome rule + driver-specific tire management): 72.4%
    accuracy / 0.747 AUC, still slightly behind the deployed logistic
    regression's 74.1% / 0.765.
    """
    rows = []
    for lookahead in lookaheads:
        if lookahead > horizon_laps:
            continue
        for compound in candidate_compounds:
            result = simulate_candidate(
                circuit=circuit, current_lap=current_lap, own_tire_age=own_tire_age,
                own_compound=own_compound, lookahead_laps=lookahead, candidate_compound=compound,
                rival_gap_seconds=rival_gap_seconds, rival_tire_age=rival_tire_age,
                rival_compound=rival_compound, reference_laptime=reference_laptime,
                horizon_laps=horizon_laps, n_simulations=n_simulations,
                degradation_model=artifacts["degradation"], pit_loss_df=artifacts["pit_loss"],
                hazard_model=artifacts["pit_hazard_model"], sc_hazard_df=artifacts["hazard"],
                sc_periods_df=artifacts["sc_periods"], calibration_model=artifacts.get("calibration_model"),
                driver_a=driver_a, driver_b=driver_b,
                driver_tire_management=artifacts.get("driver_tire_management"),
                rival_in_traffic=rival_in_traffic, seed=seed,
            )
            rows.append({
                "lookahead_laps": lookahead, "compound": compound,
                "p_ahead": result.p_ahead, "mean_gap_seconds": result.mean_gap_seconds,
                "gap_std_seconds": result.gap_std_seconds,
                "p_rival_pits_within_window": result.p_rival_pits_within_window,
                "p_sc_vsc_in_window": result.p_sc_vsc_in_window,
            })
    return pd.DataFrame(rows).sort_values("p_ahead", ascending=False).reset_index(drop=True)
