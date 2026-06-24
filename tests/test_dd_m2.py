"""Tests for M2 refactors: grade ladder, scoring gates, valuation smoke."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from grading import grade, BUY_THRESHOLD
import scoring
import fair_value


# ── grade ladder (single source of truth) ───────────────────────────────────

@pytest.mark.parametrize("score,label", [
    (9.5, "CONVICTION BUY"), (9.0, "CONVICTION BUY"),
    (8.0, "STRONG BUY"), (6.5, "BUY"), (5.0, "HOLD"),
    (3.5, "SELL"), (2.0, "STRONG SELL"), (1.0, "AVOID"), (0.0, "AVOID"),
])
def test_grade_ladder(score, label):
    assert grade(score) == label

def test_buy_threshold_value():
    assert BUY_THRESHOLD == 7.0
    # The single source of truth is re-exported everywhere.
    from scout import BUY_THRESHOLD as s
    from gems import BUY_THRESHOLD as g
    assert s == g == BUY_THRESHOLD


# ── consensus_gap_adjust: thin-coverage gate ─────────────────────────────────

def test_consensus_gap_skipped_on_thin_coverage():
    _, details = scoring.consensus_gap_adjust(7.0, 100.0, 130.0, num_analysts=2)
    assert details["applied"] is False

def test_consensus_gap_applied_with_enough_analysts():
    adj, details = scoring.consensus_gap_adjust(7.0, 100.0, 130.0, num_analysts=10)
    assert details["applied"] is True
    assert adj > 7.0  # 30% upside vs consensus nudges the score up

def test_consensus_gap_applied_when_count_unknown():
    _, details = scoring.consensus_gap_adjust(7.0, 100.0, 130.0)
    assert details["applied"] is True


# ── valuation engine smoke (no exceptions; sane composite) ───────────────────

def _financial_dossier(roe):
    return {
        "financials": {
            "ratios_ttm": {"roe": roe, "shares_out": 10.0},
            "balance": [{"stockholders_equity": 100.0, "goodwill": 0,
                         "intangible_assets": 0, "total_debt": 0, "cash": 0}],
            "income": [{}], "cashflow": [{}],
        },
        "macro": {"yield_curve_spread": 1.0},
        "profile": {"sector": "Financial Services", "industry": "Banks"},
        "quote": {"price": 12.0},
    }

def test_classify_archetype_financial():
    info = fair_value.classify_archetype(_financial_dossier(16.0))
    assert isinstance(info, dict) and info.get("archetype")

def test_compute_fair_values_does_not_raise():
    out = fair_value.compute_fair_values(_financial_dossier(16.0))
    assert isinstance(out, dict)
    assert "composite_fair_value" in out
    assert out.get("valuation_failed") is not True
