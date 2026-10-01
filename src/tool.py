"""
The final callable tool: given a race state, output a ranked pit-strategy
recommendation (stay out / box now / box in N laps, and which compound) with the
reasoning behind it.

Ties together every stage:
  - Stage 1: expected_laptime() / pit_loss() to project pace forward
  - Stage 2 Part B: the fitted logistic regression to estimate P(gain position)
    against a specific rival, at the current state or N laps in the future -
    this stays the PRIMARY source for the recommendation's confidence number,
    because it's the best-validated predictor this project has (see below)
  - Stage 3: the SC/VSC hazard curve, surfaced as informational context (a high
    hazard in the next few laps is a reason to lean toward pitting now rather
    than waiting, since a Safety Car appearing later removes the choice)
  - Stage 4: the rain-probability rule and the (small, tested) temperature
    adjustment to the degradation baseline
  - Stage 5 (optional, `include_monte_carlo=True`): a real Monte Carlo
    simulation layer that samples the rival's pit timing from a fitted hazard
    model and Safety Car occurrence from Stage 3's hazard curve, instead of
    assuming the rival never pits. Retrospective validation on 286 real held-out
    attempts showed it does NOT beat the deployed logistic regression on raw
    accuracy (60.8% vs 74.8%) - so it's exposed as a
    supplementary, clearly-labeled capability (a real outcome distribution, and
    a genuine if imperfect answer to "should I wait"), not used to override the
    primary recommendation.
"""
from dataclasses import dataclass, field

import joblib
import pandas as pd

from src.config import COMPOUND_SOFTNESS_ORDER
from src.paths import MODELS_DIR
from src.stage1_degradation import expected_laptime, pit_loss as get_pit_loss
from src.stage5_montecarlo import scan_strategies_montecarlo

RAIN_FLAG_THRESHOLD = 0.30  # chosen low enough to give useful advance warning without flagging nearly every race
LOOKAHEAD_LAPS = (0, 1, 2, 3)  # "box now" through "box in 3 laps"

_MODEL_CACHE = {}


def _load_artifacts():
    if not _MODEL_CACHE:
        _MODEL_CACHE["degradation"] = pd.read_csv(MODELS_DIR / "degradation_model.csv")
        _MODEL_CACHE["pit_loss"] = pd.read_csv(MODELS_DIR / "pit_loss_model.csv")
        _MODEL_CACHE["reference"] = pd.read_csv(MODELS_DIR / "circuit_reference_laptimes.csv")
        _MODEL_CACHE["tire_life"] = pd.read_csv(MODELS_DIR.parent / "derived" / "tire_life_limits.csv")
        _MODEL_CACHE["hazard"] = pd.read_csv(MODELS_DIR / "sc_vsc_hazard_by_circuit_lap.csv")
        _MODEL_CACHE["undercut_model"] = joblib.load(MODELS_DIR / "undercut_model_logreg_deployed.joblib")
        _MODEL_CACHE["pit_hazard_model"] = joblib.load(MODELS_DIR / "pit_hazard_model.joblib")
        _MODEL_CACHE["sc_periods"] = pd.read_csv(MODELS_DIR.parent / "derived" / "sc_vsc_periods_autodetected.csv")
        _MODEL_CACHE["team_pit_speed"] = pd.read_csv(MODELS_DIR.parent / "derived" / "team_pit_speed.csv")
        # Stage 5 accuracy-gap revisit, Step 1: the calibrated outcome model that
        # replaced the Monte Carlo simulator's hard physics cutoff (60.8% -> 72.4%
        # accuracy on real held-out attempts). Step 2 (driver-
        # specific tire management) was tried and found to HURT accuracy (72.4% ->
        # 70.3%, back below the majority-class baseline) - not loaded here, since
        # only changes that earned their place are used by default, same standard
        # as every other feature in this project.
        _MODEL_CACHE["calibration_model"] = joblib.load(MODELS_DIR / "montecarlo_calibration_model.joblib")
    return _MODEL_CACHE


def _team_pit_speed(team: str | None, season: int | None, artifacts: dict) -> float:
    """Looks up the Stage 2 revisit's team pit-crew speed proxy (relative
    pit-lane transit time vs. that circuit's median, negative = faster).
    Defaults to 0.0 (an average team) when no team/season is supplied, so calls
    made without this info still work, just under a neutral assumption."""
    if team is None or season is None:
        return 0.0
    row = artifacts["team_pit_speed"][(artifacts["team_pit_speed"]["team"] == team)
                                        & (artifacts["team_pit_speed"]["year"] == season)]
    return float(row.iloc[0]["team_pit_speed_relative"]) if not row.empty else 0.0


@dataclass
class StrategyRecommendation:
    action: str                    # "box now" / "box in N laps" / "stay out"
    compound: str
    confidence: float | None       # P(gains/retains position), when a rival is specified - from Stage 2's logistic regression, the best-validated predictor (see module docstring)
    reasoning: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    all_options: pd.DataFrame = None
    monte_carlo_options: pd.DataFrame = None  # only populated when include_monte_carlo=True - supplementary, not authoritative (see module docstring)


def _tire_life_warning(circuit: str, compound: str, tire_age_at_stop: int, artifacts: dict) -> str | None:
    row = artifacts["tire_life"][(artifacts["tire_life"]["circuit"] == circuit)
                                   & (artifacts["tire_life"]["compound"] == compound)]
    if row.empty:
        return None
    p90 = row.iloc[0]["p90_stint_laps"]
    if tire_age_at_stop > p90:
        return (f"Tire age at this stop ({tire_age_at_stop} laps) exceeds the 90th-percentile real "
                f"stint length for {compound} at {circuit} ({p90:.0f} laps) - drivers rarely run this long "
                f"on this compound here.")
    return None


def _sc_hazard_context(circuit: str, current_lap: int, horizon: int, artifacts: dict) -> str | None:
    hazard = artifacts["hazard"]
    window = hazard[(hazard["circuit"] == circuit) & (hazard["type"].isin(["SC", "VSC"]))
                     & (hazard["lap"] >= current_lap) & (hazard["lap"] <= current_lap + horizon)]
    if window.empty:
        return None
    total_hazard = 1 - (1 - window.groupby("lap")["smoothed_hazard"].sum().clip(upper=1)).prod()
    if total_hazard > 0.15:
        return (f"Elevated SC/VSC risk at {circuit} over the next {horizon} laps (~{total_hazard*100:.0f}% "
                f"per this circuit's historical hazard curve) - a Safety Car appearing while you're "
                f"deciding removes the choice (an SC stop is close to free), which favors pitting sooner "
                f"rather than waiting.")
    return None


def recommend_strategy(
    circuit: str,
    current_lap: int,
    own_tire_age: int,
    own_compound: str,
    candidate_compounds: list[str] | None = None,
    reference_laptime: float | None = None,
    rival_gap_seconds: float | None = None,
    rival_tire_age: float | None = None,
    rival_compound: str | None = None,
    rival_in_traffic: bool = False,
    rain_probability: float = 0.0,
    include_monte_carlo: bool = False,
    own_team: str | None = None,
    season: int | None = None,
) -> StrategyRecommendation:
    artifacts = _load_artifacts()
    warnings = []
    reasoning = []

    team_pit_speed = _team_pit_speed(own_team, season, artifacts)
    if own_team is None:
        reasoning.append("No team specified - assuming an average pit-crew speed "
                          "(team_pit_speed_relative=0.0); pass own_team and season for a team-specific estimate.")

    if reference_laptime is None:
        ref_row = artifacts["reference"][artifacts["reference"]["circuit"] == circuit]
        if ref_row.empty:
            raise KeyError(f"No reference lap time available for circuit={circuit!r} - supply one explicitly.")
        reference_laptime = ref_row.iloc[0]["typical_reference_laptime"]
        reasoning.append(f"No live reference lap time supplied - using {circuit}'s historical typical "
                          f"pace ({reference_laptime:.1f}s) as the baseline.")

    if candidate_compounds is None:
        candidate_compounds = [c for c in ("SOFT", "MEDIUM", "HARD")
                               if not artifacts["degradation"][(artifacts["degradation"]["circuit"] == circuit)
                                                                 & (artifacts["degradation"]["compound"] == c)].empty]

    if rain_probability >= RAIN_FLAG_THRESHOLD:
        warnings.append(f"Rain probability ({rain_probability*100:.0f}%) is above the "
                        f"{RAIN_FLAG_THRESHOLD*100:.0f}% flag threshold - consider an intermediate/wet "
                        f"tire scenario alongside this dry-tire recommendation (full wet-weather "
                        f"strategy modeling is out of this tool's scope).")

    have_rival = rival_gap_seconds is not None and rival_tire_age is not None and rival_compound is not None

    options = []
    for lookahead in LOOKAHEAD_LAPS:
        stop_lap = current_lap + lookahead
        own_tire_age_at_stop = own_tire_age + lookahead
        for compound in candidate_compounds:
            row = {"lookahead_laps": lookahead, "stop_lap": stop_lap, "compound": compound}
            if have_rival:
                # Project the gap forward `lookahead` laps assuming the rival keeps
                # circulating on its current compound (doesn't pit in the interim) -
                # the same assumption Stage 2 Part A's physics threshold makes.
                projected_gap = rival_gap_seconds
                if lookahead > 0:
                    own_pace = sum(
                        expected_laptime(circuit, own_compound, own_tire_age + k, reference_laptime,
                                          artifacts["degradation"])
                        for k in range(lookahead)
                    )
                    rival_pace = sum(
                        expected_laptime(circuit, rival_compound, rival_tire_age + k, reference_laptime,
                                          artifacts["degradation"])
                        for k in range(lookahead)
                    )
                    projected_gap = rival_gap_seconds + (own_pace - rival_pace)
                features = pd.DataFrame([{
                    "gap_seconds": projected_gap,
                    "tire_age_a": own_tire_age_at_stop,
                    "tire_age_b": rival_tire_age + lookahead,
                    "pit_loss_seconds": get_pit_loss(circuit, artifacts["pit_loss"]),
                    "lap_a": stop_lap,
                    "a_in_traffic": int(rival_in_traffic),
                    "is_undercut_ahead": int(projected_gap > 0),
                    "compound_advantage": (COMPOUND_SOFTNESS_ORDER.get(rival_compound, 5)
                                            - COMPOUND_SOFTNESS_ORDER.get(compound, 5)),
                    "team_pit_speed_relative": team_pit_speed,
                    "gap_x_tire_age_b": projected_gap * (rival_tire_age + lookahead),
                }])
                row["success_probability"] = float(
                    artifacts["undercut_model"].predict_proba(features)[0, 1]
                )
                # Plausibility cap: the projection above assumes the rival keeps
                # circulating on its CURRENT compound for the whole lookahead
                # without pitting - reasonable for 1-3 laps, but the model's
                # strongest predictor is tire_age_b (rival's tire age), so without
                # a cap this projection mechanically favors "wait as long as
                # possible" purely because it keeps aging the rival's tires in the
                # simulation, not because waiting is actually a good idea. Flag
                # (don't silently drop) any lookahead where the rival's projected
                # tire age would exceed that compound's own real p90 stint length -
                # real drivers don't typically run a compound that long, so
                # assuming the rival still hasn't pitted by then is implausible.
                rival_p90 = artifacts["tire_life"][
                    (artifacts["tire_life"]["circuit"] == circuit)
                    & (artifacts["tire_life"]["compound"] == rival_compound)
                ]
                if not rival_p90.empty and (rival_tire_age + lookahead) > rival_p90.iloc[0]["p90_stint_laps"]:
                    row["plausible"] = False
                else:
                    row["plausible"] = True
            else:
                row["success_probability"] = None
                row["plausible"] = True
            row["tire_life_warning"] = _tire_life_warning(circuit, compound, own_tire_age_at_stop, artifacts)
            options.append(row)

    options_df = pd.DataFrame(options)

    if have_rival:
        plausible = options_df[options_df["plausible"]]
        if plausible.empty:
            plausible = options_df  # degenerate case: every option implausible, fall back to all
        best = plausible.loc[plausible["success_probability"].idxmax()]
        n_excluded = (~options_df["plausible"]).sum()
        reasoning.append(f"Compared {len(plausible)} (timing, compound) options against the specified "
                         f"rival using the Stage 2 logistic regression model (test-set AUC 0.76) - "
                         f"{best['success_probability']*100:.0f}% predicted chance of being "
                         f"ahead of the rival 3 laps after both have stopped.")
        if n_excluded:
            reasoning.append(f"Excluded {n_excluded} option(s) that assumed the rival stays out on its "
                             f"current tires past that compound's typical real stint length (implausible).")
    else:
        # No rival: minimize projected total time over a FIXED total number of
        # laps, split between staying out on the current compound for
        # `lookahead` more laps (at increasing tire age) and then running the
        # candidate compound for the rest of the horizon - so the lookahead
        # choice actually matters here too, not just the compound choice.
        HORIZON = 15
        for i, row in options_df.iterrows():
            lookahead, compound = int(row["lookahead_laps"]), row["compound"]
            stay_out_cost = sum(
                expected_laptime(circuit, own_compound, own_tire_age + k, reference_laptime,
                                  artifacts["degradation"])
                for k in range(lookahead)
            )
            new_tire_cost = sum(
                expected_laptime(circuit, compound, k, reference_laptime, artifacts["degradation"])
                for k in range(HORIZON - lookahead)
            )
            options_df.loc[i, "projected_total_time"] = (
                stay_out_cost + get_pit_loss(circuit, artifacts["pit_loss"]) + new_tire_cost
            )
        best = options_df.loc[options_df["projected_total_time"].idxmin()]
        reasoning.append(f"No rival specified - compared candidate (timing, compound) options by "
                         f"projected total time over the next {HORIZON} laps (current compound until "
                         f"the stop, pit loss, then the candidate compound) - no rival to gain/lose "
                         f"position against.")

    if best["tire_life_warning"]:
        warnings.append(best["tire_life_warning"])

    sc_note = _sc_hazard_context(circuit, current_lap, int(best["lookahead_laps"]) or 1, artifacts)
    if sc_note:
        reasoning.append(sc_note)

    action = "box now" if best["lookahead_laps"] == 0 else f"box in {int(best['lookahead_laps'])} laps"

    monte_carlo_options = None
    if include_monte_carlo and have_rival:
        # Supplementary only - see module docstring. Retrospective validation
        # showed this does NOT beat the logistic-regression-driven
        # recommendation above on real accuracy (60.8% vs 74.8%), so it's
        # reported alongside, not used to change `action`/`compound`/`confidence`.
        monte_carlo_options = scan_strategies_montecarlo(
            circuit=circuit, current_lap=current_lap, own_tire_age=own_tire_age,
            own_compound=own_compound, candidate_compounds=candidate_compounds,
            lookaheads=LOOKAHEAD_LAPS, rival_gap_seconds=rival_gap_seconds,
            rival_tire_age=rival_tire_age, rival_compound=rival_compound,
            reference_laptime=reference_laptime, artifacts=artifacts,
            rival_in_traffic=rival_in_traffic,
        )
        mc_best = monte_carlo_options.iloc[0]
        reasoning.append(
            f"Stage 5 Monte Carlo cross-check (supplementary, not authoritative - it runs slightly "
            f"behind the logistic regression above on real held-out accuracy): its "
            f"top-ranked option is {mc_best['compound']} at lookahead {int(mc_best['lookahead_laps'])} "
            f"with a simulated {mc_best['p_ahead']*100:.0f}% chance of being ahead at the measurement "
            f"lap (mean gap {mc_best['mean_gap_seconds']:+.1f}s), from {mc_best['p_rival_pits_within_window']*100:.0f}% "
            f"simulated probability the rival pits within the horizon and "
            f"{mc_best['p_sc_vsc_in_window']*100:.0f}% chance of a Safety Car/VSC in that window."
        )

    return StrategyRecommendation(
        action=action, compound=best["compound"],
        confidence=best.get("success_probability"),
        reasoning=reasoning, warnings=warnings, all_options=options_df,
        monte_carlo_options=monte_carlo_options,
    )
