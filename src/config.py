"""
Project-wide constants that encode judgment calls made along the way. Centralized
here (rather than left as magic numbers scattered in each stage's script) so every
stage that needs one of these values uses the same definition.
"""

# --- Stage 1: lap cleaning thresholds ---
OUTLIER_STD_DEVS = 3.0  # drop laps more than this many std devs from their (circuit, compound) group mean

# --- Stage 2: undercut/overcut attempt detection + success rule ---
# An "attempt" is a pair of cars running close together where one pits before the
# other, per the project's success-rule definition (decided 2026-09-22):
UNDERCUT_GAP_THRESHOLD_S = 3.0  # cars must be within this gap (seconds) at the moment the first car pits, to count as a real head-to-head attempt (per project spec)
UNDERCUT_SUCCESS_WINDOW_LAPS = 3  # measure who's ahead this many laps after the LATER of the two stops completes
UNDERCUT_EXCLUDE_IF_WINDOW_UNAVAILABLE = True  # if the race ends before the measurement window is reached, drop the attempt from the labeled dataset rather than measuring early

# Ordinal softness ranking, low = softest. Only ever used to compare two compounds
# WITHIN the same race (e.g. "is car A now on a softer tire than car B") - a valid
# comparison regardless of the cross-year compound-label instability (SOFT in
# 2018 isn't necessarily SOFT in 2024), since both cars in a
# single attempt are always racing in the same event with the same nomenclature.
COMPOUND_SOFTNESS_ORDER = {
    "HYPERSOFT": 1, "ULTRASOFT": 2, "SUPERSOFT": 3, "SOFT": 4, "MEDIUM": 5,
    "HARD": 6, "INTERMEDIATE": 7, "WET": 8,
}

# Stage 2 Part B train/test split: chronological, not random. Holding out the most
# recent seasons tests genuine forward generalization (would this have worked on
# races the model never saw) rather than just interpolation within a shuffled mix
# of years - the more scientifically honest bar for a tool meant to validate
# against real outcomes.
UNDERCUT_MODEL_TEST_YEARS = (2025, 2026)
