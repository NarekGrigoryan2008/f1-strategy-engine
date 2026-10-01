# F1 Race Strategy Engine

A pit-strategy decision-support tool: given a race state (circuit, lap, tire
ages, gap to a rival, weather), it recommends **stay out / box now / box in N
laps** and **which compound**, with the reasoning behind the call.

It's built and validated against **real historical F1 undercut/overcut
outcomes** (2018-2026, 150 races, 20 circuits pulled from
[FastF1](https://github.com/theOehrly/Fast-F1)) rather than a purely
theoretical formula — the point is to be honest about how often it matches
reality, with the misses explained, not to overclaim accuracy in a sport with
real randomness (driver error, mechanical failure, other teams' unpredictable
calls).

## Project status

All six stages are built, run on real data, and validated — including Stage 5,
the Monte Carlo simulator, which was originally scoped as a stretch goal but
was completed once more time became available than originally planned for.

| Stage | Status |
|---|---|
| 0 — Data (150 races, 20 circuits, 2018-2026) | Complete |
| 1 — Tire degradation + pit-loss models | Complete. Low R² is a diagnosed real finding, not an unfit model — and a follow-up check found the fitted slopes don't hold up out-of-sample across years either (see below), so they're best read as descriptive of the pooled sample, not a forward-validated constant |
| 2 — Undercut/overcut classifier (physics + logistic regression) | Complete. 74.1% accuracy / 0.765 AUC deployed model, revisited with 2 new earned-their-place features and honest cross-validation |
| 3 — Safety Car / VSC detection | Complete. 24/150 races (16%) independently verified |
| 4 — Weather integration | Complete. Temperature-degradation effect tested and replicated on a held-out slice |
| 5 — Monte Carlo strategy simulator | **Complete** (not a placeholder) — integrated into the tool as `recommend_strategy(..., include_monte_carlo=True)`, a supplementary cross-check alongside the primary logistic-regression-driven recommendation |

## What it does

```python
from src.tool import recommend_strategy

rec = recommend_strategy(
    circuit="barcelona", current_lap=25,
    own_tire_age=18, own_compound="MEDIUM",
    rival_gap_seconds=1.5, rival_tire_age=18, rival_compound="MEDIUM",
)
print(rec.action, rec.compound, round(rec.confidence, 3))
# -> "box in 3 laps" "SOFT" 0.626
```

Add `include_monte_carlo=True` to also run the Stage 5 cross-check alongside the
primary (logistic-regression-driven) recommendation above — it doesn't change
`action`/`compound`/`confidence`, but surfaces its own ranked options and reasoning:

```python
rec = recommend_strategy(
    circuit="barcelona", current_lap=25,
    own_tire_age=18, own_compound="MEDIUM",
    rival_gap_seconds=1.5, rival_tire_age=18, rival_compound="MEDIUM",
    include_monte_carlo=True,
)
print(rec.monte_carlo_options.iloc[0][["compound", "lookahead_laps", "p_ahead"]])
# -> HARD, lookahead 0, p_ahead ~0.55 - a genuinely different top pick than the
#    primary recommendation above, a real example of the two models disagreeing
```

Run `python scripts/12_run_tool_examples.py` for real output across five
realistic scenarios (an undercut attempt, a defensive stop, a no-rival pure-pace
call, a rain-probability flag, and the same undercut attempt with the Stage 5
Monte Carlo cross-check enabled via `include_monte_carlo=True`).

## Headline findings

**Physics alone predicts real undercut/overcut outcomes worse than just guessing.**
A physics-only threshold (tire degradation math vs. pit-lane time loss) scores
**29.4% accuracy** on real, held-out 2025-2026 attempts — worse than always
predicting "success" (71.3% baseline). A logistic regression fit on 1,362 real
labeled attempts does modestly better — **74.1% accuracy, AUC 0.765 on the most
recent test split; a more honest cross-validated estimate across three
expanding-window folds is 72.3% ± 1.4%** (single-split numbers here carry real
sampling noise — checked directly, not assumed). The gap between
physics and the fitted model is the project's core result: **real F1 track
position through a pit sequence is far "stickier" than lap-time math alone
predicts** — pit-lane logistics, overtaking difficulty, and the fact both cars
pay a similar pit-loss cost matter more than tire physics for who ends up ahead.

![Physics vs. empirical model accuracy](charts/stage2_model_comparison.png)

**More machinery isn't automatically better — checked, not assumed, and then
diagnosed rather than left alone.** Stage 5 adds a real Monte Carlo simulator:
instead of assuming a rival never pits (the Stage 1-4 tool's known limitation),
it samples the rival's pit lap from a hazard model fit on 157,538 real
driver-laps, and Safety Car/VSC occurrence from Stage 3's real hazard curve.
First validation (same 286 held-out real attempts, same honest methodology as
Stage 2): **60.8% accuracy / AUC 0.617 — didn't beat the 74.1% logistic
regression, or even the 71.3% majority-class baseline.** Rather than stop there,
traced the cause: the simulator decided each outcome with a hard cutoff on
Stage 1's honestly-weak degradation slopes. Replaced that with a fitted,
calibrated probability (`P(success | physics_final_gap, tire_age_a, tire_age_b,
direction)`, sampled as a genuine per-draw Bernoulli outcome) — **72.4%
accuracy, AUC 0.747**, now beating the baseline and closing most (not all) of
the 13.3-point gap to logistic regression. Two further fixes were tried and
tested the same rigorous way: driver-specific tire management (measurably
*hurt* accuracy, 72.4% → 70.3% — dropped) and a per-lap "persistence" term
(checked first, found to be identically zero by construction, not just hard to
estimate — skipped). Kept in the tool as a supplementary, clearly-labeled
cross-check (`include_monte_carlo=True`) rather than replacing the primary
recommendation, since it's still narrowly behind the classifier.

![Physics vs. models vs. Monte Carlo, same test set](charts/stage5_model_comparison.png)

**Revisiting the classifier itself.** Went back and tested four new candidate
features individually (team pit-crew speed, out-lap traffic, a gap×tire-age
interaction, race progress as a fraction) rather than batch-adding them. Two
earned their place (team pit-crew speed — independently validated by a real
sanity check: Red Bull Racing shows up as the fastest crew in multiple separate
years, matching its well-known real-world reputation — and the gap×tire-age
interaction); two didn't (out-lap traffic wasn't significant; race progress
added nothing lap number didn't already capture). Also investigated the
model's one counterintuitive coefficient directly (does older own-tire-age
really *reduce* predicted success?) via variance inflation factors and a
refit-without-the-correlated-feature test — real effect, not a collinearity
artifact.

## Setup

```bash
py -3.11 -m venv venv          # Python 3.11 specifically - newer releases lag on data-science wheel availability
./venv/Scripts/pip install -r requirements.txt
```

## Running the pipeline

Scripts are numbered in the order they were actually built and run — each is a
real, executable step, not illustrative pseudocode. Re-running from scratch on a
fresh clone will re-pull all data from FastF1 (slow — FastF1 rate-limits API calls)
since `data/raw/` and `data/cache/` are gitignored as regenerable.
`data/derived/` and `data/models/` (the actual outputs) are committed, except
`data/models/pit_hazard_model.joblib` (134MB, over GitHub's file-size limit) -
regenerate it by running `14_stage5_fit_pit_hazard.py` against the already-
committed `data/derived/pit_hazard_dataset.csv` (quick; the slow part, pulling
the raw laps that dataset is built from, is already done).

| Script | What it does |
|---|---|
| `01_get_calendars.py` | Determines the 20 most-raced circuits 2018-2026 from real FastF1 calendars |
| `02_pull_race_data.py` | Pulls laps/weather/race-control/results for every (circuit, season) |
| `02b_migrate_race_dirs.py` | One-time fix for a doubleheader-collision bug |
| `02c_backfill_laptime_column.py` | Backfills the `Time` column needed for Stage 2's gap calculations |
| `03_stage1_degradation.py` | Fits tire degradation + pit-loss models (Stage 1) |
| `04_stage2_detect_attempts.py` | Detects real undercut/overcut attempts from lap data |
| `05_stage2_fit_model.py` | Fits + evaluates the logistic regression / LightGBM classifiers (Stage 2 Part B) |
| `06_stage2_flagship_chart.py` / `07_stage2_comparison_chart.py` | The two headline Stage 2 charts |
| `08_stage3_detect_sc.py` | Safety Car / VSC / red-flag detection + hazard curves (Stage 3) |
| `09_stage4_weather_pull.py` | Open-Meteo cross-reference against FastF1's weather (Stage 4) |
| `10_stage4_temperature_test.py` | Tests whether temperature increases degradation (Stage 4) |
| `11_derived_tire_life_and_incidents.py` | Tire life limits, driver tire management, incidents log |
| `12_run_tool_examples.py` | Runs the final tool on real example race states (incl. a Monte Carlo cross-check) |
| `13_stage5_build_pit_hazard.py` / `14_stage5_fit_pit_hazard.py` | Builds + fits the real rival pit-timing hazard model (Stage 5) |
| `15_stage5_sc_pit_loss.py` | Measures the real (found: nonexistent) SC/VSC pit-loss discount |
| `16_stage5_n_stability_test.py` | The N-simulations runtime/stability test behind the N=20,000 decision |
| `17_stage5_validate.py` | Retrospective validation of the Monte Carlo simulator against real outcomes |
| `18_stage5_comparison_chart.py` | The Stage 5 headline comparison chart |
| `19_stage2_vif_and_cv.py` | VIF/collinearity check + expanding-window CV for the Stage 2 classifier (found the counterintuitive `tire_age_a` coefficient is real, and that 74.8% overstated typical accuracy) |
| `20_stage2_new_features.py` | Builds candidate features for the Stage 2 classifier revisit (team pit-crew speed, out-lap traffic, race progress) |
| `21_stage2_test_features.py` | Tests each new feature individually, selects the final combined feature set (2 kept, 2 dropped) |
| `22_stage5_fit_calibration.py` | Fits the calibrated outcome-probability model that replaced the Monte Carlo simulator's hard physics cutoff (60.8% → 72.4% accuracy) |
| `23_stage1_cv_stability.py` | Expanding-window stability check on Stage 1's degradation fits — finds they don't hold up out-of-sample across years |
| `24_stage4_temperature_holdout_check.py` | Re-checks the temperature-degradation effect on the 2025-2026 holdout alone — replicates cleanly |

## Project structure

```
src/            importable package: data loading, per-stage models, the final tool
scripts/        the numbered pipeline above
data/raw/       per-race CSV pulls (gitignored, regenerable)
data/derived/   computed tables (attempts, incidents, periods, cross-references)
data/models/    fitted model artifacts (degradation curves, pit-loss constants,
                the deployed classifier, SC/VSC hazard curves)
charts/         diagnostic + portfolio charts
```

## Honest limitations

- **Compound labels (SOFT/MEDIUM/HARD) are relative to each race weekend, not a
  fixed physical compound** — pooling them across 2018-2026 blends different
  actual rubber compounds under the same label. Noted, not fixed.
- **Tire degradation is a real but small effect** relative to driver/traffic
  noise in a simple single-variable fit — most (circuit, compound) R² values are
  under 0.05 for dry compounds, reported honestly rather than chased with added
  model complexity that would violate the project's "deliberately simple" design.
  Checked further, not just left there: an expanding-window CV (fit on earlier
  years, test on later ones — same methodology used for Stage 2's classifier)
  found the fitted slopes do NOT hold up well over time — **median out-of-sample
  R² was negative in all 3 tested folds** (a line fit on earlier years predicts
  later years worse than just guessing their average), with weak, inconsistent
  slope stability across folds (correlation 0.65, 0.09, 0.01). Most likely tied
  to the compound-label instability above, compounding with already-thin
  per-group samples. **Read Stage 1's slopes as descriptive of the pooled
  2018-2026 sample, not a validated, forward-stable constant.**
- **Stage 4's temperature-degradation effect, by contrast, replicates cleanly**
  on an independent check: refit the exact same interaction model on the
  2025-2026 slice alone (never checked in isolation before) and got an
  essentially identical coefficient (+0.00115 vs. the original +0.00111
  sec/lap/°C, both p<0.0001) — real, stable signal, not a pooled-sample
  artifact. Same checking method as the Stage 1 finding above, opposite
  honest result — both reported with equal weight.
- **24/150 races (16%) have been manually verified against Wikipedia/race-control
  logs** for Stage 3 (SC/VSC detection) — up from an initial 6-race pass, mixing
  adversarial and random picks. Caught and fixed 3 real, understood issues (a
  race run entirely behind the Safety Car; a red flag fragmented by a lap-
  numbering gap during a full stoppage; two false-positive VSCs from apparent
  TrackStatus noise in a very recent race). Also found that some apparent
  "misses" were actually imprecision in the Wikipedia *summary*, not the
  detector — resolved by checking the primary race-control log directly. Still
  a sample, not an audit — the other 126 races (84%) haven't been individually
  checked.
- **Stage 5's Monte Carlo simulator still doesn't quite beat the simpler Stage 2
  model** on real held-out accuracy — 72.4% after a diagnosed, targeted fix
  (replacing a hard physics cutoff with a calibrated probability), up from an
  original 60.8%, against the classifier's 74.1%. Two further fixes were tried
  and rejected the same rigorous way (driver-specific tire management measurably
  hurt accuracy; a persistence term was checked and found to be identically
  zero by construction, not fit). Kept as a supplementary cross-check, not the
  primary decision engine.
- **The rival's post-stop compound is assumed** (defaults to MEDIUM) in the
  Monte Carlo simulation, since the real choice isn't knowable in advance — a
  documented simplification, not a fitted value.
