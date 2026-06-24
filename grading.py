"""Single source of truth for the 7-tier grade ladder, BUY threshold, and colors.

Imported by debate.py, scoring.py, report.py, scout.py, gems.py so the thresholds
can never drift apart.
"""

# Minimum consensus score to count as a BUY — gates Telegram alerts, the KV upload,
# and what appears on the Sovereign Scout/Gems dashboard (single lever for all three).
BUY_THRESHOLD = 7.0

# Threshold below which a held position drops to TRIM/EXIT in hold-mode grading.
HOLD_THRESHOLD = 5.5

# (min_score, label), descending.
_LADDER = [
    (9.0, "CONVICTION BUY"),
    (8.0, "STRONG BUY"),
    (6.5, "BUY"),
    (5.0, "HOLD"),
    (3.5, "SELL"),
    (2.0, "STRONG SELL"),
]

# Hold-mode ladder — applied to current portfolio holdings. The vocabulary changes
# from BUY/SELL (entry decision) to ADD/HOLD/TRIM/EXIT (already-own decision).
_HOLD_LADDER = [
    (7.0, "ADD"),
    (5.5, "HOLD"),
    (3.5, "TRIM"),
]


def grade(score: float) -> str:
    """7-tier grading scale (entry mode — used for scout/gems)."""
    for threshold, label in _LADDER:
        if score >= threshold:
            return label
    return "AVOID"


def grade_hold(score: float) -> str:
    """4-tier hold-mode grading scale — for stocks already in the portfolio.

    The question is "should I keep this," not "should I buy this today." A late-cycle
    macro view shouldn't push a multi-year compounder to SELL, so the threshold for
    HOLD is lower (5.5) and the labels are ADD/HOLD/TRIM/EXIT.
    """
    for threshold, label in _HOLD_LADDER:
        if score >= threshold:
            return label
    return "EXIT"


GRADE_COLORS = {
    "CONVICTION BUY": "bold bright_green",
    "STRONG BUY":     "bold green",
    "BUY":            "green",
    "HOLD":           "yellow",
    "SELL":           "red",
    "STRONG SELL":    "bold red",
    "AVOID":          "bold bright_red",
    # Hold-mode labels
    "ADD":            "bold green",
    "TRIM":           "red",
    "EXIT":           "bold red",
}
