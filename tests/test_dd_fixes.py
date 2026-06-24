"""Regression tests for the code-review fixes (units, JSON parsing, debate contamination).

Run from the repo root:  python -m pytest tests/ -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from fair_value import _norm_margin, _cagr, _value_financial
from llm import extract_json
from debate import _live_scores


# ── _norm_margin: always treat input as a percentage (dossier _pct output) ──────

def test_norm_margin_percentage():
    assert _norm_margin(65.4) == pytest.approx(0.654)

def test_norm_margin_sub_one_percent_not_misread():
    # The old abs(v)>1 heuristic read 0.5 (=0.5%) as 0.5 (=50%). Must be 0.005 now.
    assert _norm_margin(0.5) == pytest.approx(0.005)

def test_norm_margin_none():
    assert _norm_margin(None) is None


# ── _cagr: reject non-positive endpoints (no complex numbers) ───────────────────

def test_cagr_basic():
    assert _cagr(100, 200, 1) == pytest.approx(1.0)

def test_cagr_negative_end_is_none():
    # (-50/100)**(1/3) is a complex number in Python — must be guarded.
    assert _cagr(100, -50, 3) is None

def test_cagr_zero_start_is_none():
    assert _cagr(0, 100, 1) is None


# ── _value_financial: ROE is a percentage, not a fraction (D-C1, 100x bug) ──────

def _financial_dossier(roe: float) -> dict:
    # tangible book = equity - goodwill - intangibles = 100; shares = 10 -> TBV/sh = 10
    return {
        "financials": {
            "ratios_ttm": {"roe": roe, "shares_out": 10.0},
            "balance": [{
                "stockholders_equity": 100.0, "goodwill": 0, "intangible_assets": 0,
                "total_debt": 0, "cash": 0,
            }],
            "income": [{}],
            "cashflow": [{}],
        },
        "macro": {"yield_curve_spread": 1.0},
        "profile": {"sector": "Financial Services", "industry": "Banks"},
        "quote": {"price": 12.0},
    }

def test_value_financial_weak_roe_low_ptbv():
    # ROE 4% -> bottom tier target 0.6 -> fair value = 10 * 0.6 = 6.0
    fv = _value_financial(_financial_dossier(4.0))["primary"]["fair_value"]
    assert fv == pytest.approx(6.0)

def test_value_financial_strong_roe_high_ptbv():
    # ROE 16% -> top tier target 1.8 -> fair value = 10 * 1.8 = 18.0
    fv = _value_financial(_financial_dossier(16.0))["primary"]["fair_value"]
    assert fv == pytest.approx(18.0)

def test_value_financial_roe_tiers_differ():
    # The whole point: a weak bank and a strong bank must NOT get the same target.
    weak   = _value_financial(_financial_dossier(4.0))["primary"]["fair_value"]
    strong = _value_financial(_financial_dossier(16.0))["primary"]["fair_value"]
    assert strong > weak


# ── extract_json: robust to fences, arrays, internal backticks ──────────────────

def test_extract_json_object():
    assert extract_json('{"score": 9}') == {"score": 9}

def test_extract_json_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

def test_extract_json_array_of_objects_returns_object():
    # brace-first scan yields the inner object — debate callers require a dict.
    assert extract_json('[{"score": 9}]') == {"score": 9}

def test_extract_json_pure_array():
    assert extract_json('[1, 2, 3]') == [1, 2, 3]

def test_extract_json_internal_backticks_preserved():
    # The old global ``` strip could corrupt string values; leading/trailing only now.
    assert extract_json('{"note": "see ```x```"}') == {"note": "see ```x```"}


# ── _live_scores: exclude failed agents' fabricated 5.0 from consensus stats ─────

def test_live_scores_excludes_failed():
    scores  = {"A": 2.0, "B": 8.0, "FAILED": 5.0}
    results = {"A": {"score": 2.0}, "B": {"score": 8.0}, "FAILED": {"_failed": True}}
    live = _live_scores(scores, results)
    assert "FAILED" not in live
    # spread reflects real opinions (8-2=6), not shrunk by the fabricated 5.0
    assert max(live.values()) - min(live.values()) == pytest.approx(6.0)

def test_live_scores_all_failed_falls_back_to_full():
    scores  = {"A": 5.0, "B": 5.0}
    results = {"A": {"_failed": True}, "B": {"_failed": True}}
    # never return empty — fall back to the full set so downstream math doesn't crash
    assert _live_scores(scores, results) == scores
