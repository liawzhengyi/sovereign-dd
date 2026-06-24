"""Tests for the quantified risk/reward layer and its scoring integration."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import risk_reward as rr_mod
from risk_reward import compute_risk_reward, llm_cross_check, compact
import scoring


# ── fixture helper ───────────────────────────────────────────────────────────

def _dossier(**over):
    """Minimal valid dossier: $100 stock, composite FV $150, clean balance sheet.

    Defaults land in LOW risk / HIGH reward (upside +47%, downside to floor ~22%).
    """
    d = {
        "quote": {"price": 100.0},
        "technicals": {"52w_low": 70.0, "rsi_14": 55.0, "pct_from_52w_high": -15.0},
        "valuation": {
            "dcf_iv_per_share": 140.0,
            "analyst_consensus": {"target_mean": 160.0, "num_analysts": 12},
        },
        "fair_values": {
            "composite_fair_value": 150.0,
            "primary_fair_value": 145.0,
            "secondary_methods": [{"method": "EV/FCF", "fair_value": 155.0}],
            "archetype": {"archetype": "ASSET_LIGHT_GROWTH", "confidence": "HIGH"},
            "blind_spot_flags": [],
        },
        "financials": {
            "ratios_ttm": {
                "roic": 24.6, "wacc": 0.13, "net_margin": 25.0,
                "debt_equity": 30.0, "current_ratio": 2.5, "ebitda": 5e9,
                "beta": 1.2, "short_pct": 2.0, "eps_revision_momentum": 0.03,
                "shares_out": 1e9,
            },
            "balance": [{"total_debt": 1e9, "cash": 3e9}],
        },
        "insiders": {"net_insider_usd": 0, "significant_sells": 0},
        "insider_sentiment_mspr": {"avg_mspr_3m": None},
        "earnings_surprises": [{"beat_quality": "BEAT"}] * 4,
    }
    d.update(over)
    return d


# ── applied / not-applied gates ──────────────────────────────────────────────

def test_clean_low_risk_high_reward():
    rr = compute_risk_reward(_dossier())
    assert rr["applied"] is True
    assert rr["risk_tier"] == "LOW" and rr["reward_tier"] == "HIGH"
    assert rr["quadrant"] == "LOW_RISK_HIGH_REWARD"
    assert rr["upside_source"] == "fv_blend"
    assert rr["adjustment"] >= 0.75


def test_no_price_not_applied():
    rr = compute_risk_reward(_dossier(quote={}, technicals={}))
    assert rr["applied"] is False and "price" in rr["reason"]


def test_fv_error_shape_no_crash():
    d = _dossier(fair_values={"error": "boom", "composite_fair_value": None})
    rr = compute_risk_reward(d)
    # falls back to analyst_only upside (12 analysts) — still applied
    assert rr["applied"] is True and rr["upside_source"] == "analyst_only"
    assert rr["dampened_by"] == rr_mod.CONFIDENCE_DAMPEN


def test_no_fv_and_thin_coverage_not_applied():
    d = _dossier(fair_values={"composite_fair_value": None})
    d["valuation"]["analyst_consensus"]["num_analysts"] = 2
    rr = compute_risk_reward(d)
    assert rr["applied"] is False


def test_no_floor_candidates_not_applied():
    d = _dossier(
        technicals={},  # no 52w_low
        fair_values={"composite_fair_value": 150.0},  # no primary/secondary
    )
    d["valuation"]["dcf_iv_per_share"] = None
    rr = compute_risk_reward(d)
    assert rr["applied"] is False and "floor" in rr["reason"]


def test_garbage_input_never_raises():
    assert compute_risk_reward({})["applied"] is False
    assert compute_risk_reward({"quote": {"price": "not-a-number"}})["applied"] is False


# ── matrix + dampener ────────────────────────────────────────────────────────

def test_kicker_total_is_one_point_zero():
    # upside +47%, downside floored low -> rr_ratio >= 3 with a high floor
    d = _dossier()
    d["technicals"]["52w_low"] = 95.0  # tight floor -> small downside -> big ratio
    rr = compute_risk_reward(d)
    assert rr["risk_tier"] == "LOW" and rr["rr_ratio"] >= 3.0
    assert rr["adjustment"] == 1.0


def test_medium_confidence_dampens():
    d = _dossier()
    d["fair_values"]["archetype"]["confidence"] = "MEDIUM"
    d["technicals"]["52w_low"] = 95.0
    rr = compute_risk_reward(d)
    assert rr["dampened_by"] == 0.6
    assert rr["adjustment"] == round(1.0 * 0.6, 2)


def test_high_risk_low_reward_is_max_penalty():
    d = _dossier()
    d["fair_values"]["composite_fair_value"] = 95.0   # below price -> low reward
    d["valuation"]["analyst_consensus"]["target_mean"] = 98.0
    d["financials"]["ratios_ttm"].update({
        "roic": 5.0, "wacc": 0.13,        # quality +1.0
        "net_margin": -3.0,               # quality +1.0
        "beta": 2.5,                      # market +0.75
        "short_pct": 20.0,                # market +1.0
        "current_ratio": 0.8,             # leverage +0.5
        "debt_equity": 200.0,             # leverage +0.5
    })
    d["financials"]["balance"] = [{"total_debt": 30e9, "cash": 1e9}]  # nde ~5.8 +1.5
    rr = compute_risk_reward(d)
    assert rr["risk_tier"] == "HIGH" and rr["reward_tier"] == "LOW"
    assert rr["adjustment"] == -0.90


def test_diagonal_is_neutral():
    # LOW/LOW: clean balance sheet, FV below price
    d = _dossier()
    d["fair_values"]["composite_fair_value"] = 100.0
    d["valuation"]["analyst_consensus"]["target_mean"] = 102.0
    rr = compute_risk_reward(d)
    assert (rr["risk_tier"], rr["reward_tier"]) == ("LOW", "LOW")
    assert rr["adjustment"] == 0.0


# ── upside / downside math ───────────────────────────────────────────────────

def test_downside_min_clamp_when_support_above_price():
    d = _dossier()
    d["technicalss"] = None  # ignore
    d["technicals"]["52w_low"] = 120.0  # support above price
    rr = compute_risk_reward(d)
    assert rr["downside_pct"] == rr_mod.DOWNSIDE_MIN * 100


def test_downside_max_clamp():
    d = _dossier()
    d["technicals"]["52w_low"] = 1.0
    d["valuation"]["dcf_iv_per_share"] = 6.0
    d["fair_values"]["primary_fair_value"] = 6.0
    d["fair_values"]["secondary_methods"] = []
    rr = compute_risk_reward(d)
    assert rr["downside_pct"] == rr_mod.DOWNSIDE_MAX * 100


def test_negative_upside_means_zero_ratio():
    d = _dossier()
    d["fair_values"]["composite_fair_value"] = 60.0
    d["valuation"]["analyst_consensus"]["target_mean"] = 70.0
    rr = compute_risk_reward(d)
    assert rr["rr_ratio"] == 0.0 and rr["reward_tier"] == "LOW"


def test_net_cash_floor_lifts_support():
    d = _dossier()
    d["financials"]["balance"] = [{"total_debt": 0, "cash": 90e9}]  # $90/share net cash
    rr = compute_risk_reward(d)
    assert rr["floor_sources"]["net_cash_ps"] == 90.0
    assert rr["downside_floor"] >= 90.0


# ── risk_index scale guards ──────────────────────────────────────────────────

def test_roic_percent_vs_wacc_fraction_scale():
    ok = compute_risk_reward(_dossier())  # roic 24.6% vs wacc 13% -> no flag
    assert not any("ROIC" in c for c in ok["risk_components"])
    d = _dossier()
    d["financials"]["ratios_ttm"]["roic"] = 8.0  # 8% < 13% -> flag
    bad = compute_risk_reward(d)
    assert any("ROIC" in c for c in bad["risk_components"])


def test_total_debt_none_is_safe():
    d = _dossier()
    d["financials"]["balance"] = [{"total_debt": None, "cash": 3e9}]
    rr = compute_risk_reward(d)
    assert rr["applied"] is True
    assert not any("net_debt" in c for c in rr["risk_components"])


def test_two_large_beats_flag():
    d = _dossier(earnings_surprises=[
        {"beat_quality": "LARGE_BEAT"}, {"beat_quality": "BEAT"},
        {"beat_quality": "LARGE_BEAT"}, {"beat_quality": "MISS"},
    ])
    rr = compute_risk_reward(d)
    assert any("LARGE_BEAT" in c for c in rr["risk_components"])


def test_cycle_never_affects_risk_index():
    a = compute_risk_reward(_dossier())
    b = compute_risk_reward(_dossier(cycle_type="CYCLICAL", macro={"regime": "LATE_CYCLE"}))
    assert a["risk_index"] == b["risk_index"]


# ── compact + cross-check ────────────────────────────────────────────────────

def test_compact_shape():
    c = compact(compute_risk_reward(_dossier()))
    assert set(c) == {"applied", "rr_ratio", "upside_pct", "downside_pct",
                      "risk_index", "risk_tier", "reward_tier", "quadrant", "adjustment"}
    nc = compact({"applied": False, "reason": "x"})
    assert nc == {"applied": False, "reason": "x"}


def test_llm_cross_check():
    rr = {"rr_ratio": 2.0}
    xc = llm_cross_check(rr, bull_target=140.0, bear_floor=80.0, price=100.0)
    assert xc["llm_rr"] == 2.0 and xc["divergent"] is False
    xc2 = llm_cross_check(rr, bull_target=400.0, bear_floor=99.0, price=100.0)
    assert xc2["divergent"] is True
    assert llm_cross_check(rr, None, 80.0, 100.0) is None
    assert llm_cross_check(rr, "junk", 80.0, 100.0) is None


# ── scoring integration ──────────────────────────────────────────────────────

def _result(score=7.0):
    return {
        "consensus_score": score, "confidence": "HIGH",
        "cycle_position": {"phase": "MID"}, "moat_composite": 8.0,
        "catalyst": "Major product cycle ramping through 2026",
    }


def test_apply_adjustments_has_risk_reward_trace():
    out = scoring.apply_adjustments(7.0, _result(), _dossier())
    tr = out["score_adjustments"]["risk_reward"]
    assert tr["applied"] is True and "result" in tr
    assert out["risk_reward"]["applied"] is True
    assert set(out["risk_reward"]) == set(compact(tr))


def test_consensus_gap_superseded_in_scout_mode():
    out = scoring.apply_adjustments(7.0, _result(), _dossier(), is_holding=False)
    cg = out["score_adjustments"]["consensus_gap"]
    assert cg["applied"] is False and "superseded" in cg["reason"]


def test_consensus_gap_survives_when_layer_off():
    d = _dossier(fair_values={"composite_fair_value": None}, technicals={})
    d["valuation"]["dcf_iv_per_share"] = None
    out = scoring.apply_adjustments(7.0, _result(), d, is_holding=False)
    cg = out["score_adjustments"]["consensus_gap"]
    assert "superseded" not in (cg.get("reason") or "")


def test_hold_mode_consensus_still_skipped():
    out = scoring.apply_adjustments(7.0, _result(), _dossier(), is_holding=True)
    assert out["score_adjustments"]["consensus_gap"]["reason"] == "skipped in hold-mode"
    assert out["score_adjustments"]["risk_reward"]["applied"] is True


def test_score_moves_by_adjustment():
    d = _dossier()
    out = scoring.apply_adjustments(7.0, _result(), d)
    tr = out["score_adjustments"]
    rr_step = tr["risk_reward"]
    assert rr_step["result"] == round(
        min(10.0, max(1.0, tr["cycle_position"]["result"] + rr_step["adjustment"])), 2
    )


def test_banger_uses_rr_condition():
    rr_ok = {"applied": True, "rr_ratio": 2.5}
    res = {"consensus_score": 8.0, "catalyst": "Datacenter ramp + new product cycle"}
    d = _dossier()
    d["insiders"] = {"buy_count": 3, "sell_count": 0, "cluster_buying": True}
    b = scoring.banger_check(res, d, risk_reward=rr_ok)
    assert any("rr_ratio" in c for c in b["conditions_met"])
    rr_bad = {"applied": True, "rr_ratio": 1.2}
    b2 = scoring.banger_check(res, d, risk_reward=rr_bad)
    assert b2["is_banger"] is False
    assert any("rr" in c.lower() for c in b2["conditions_failed"])


def test_banger_legacy_fallback_without_layer():
    res = {"consensus_score": 8.0, "catalyst": "Major catalyst incoming soon"}
    d = _dossier()
    d["insiders"] = {"buy_count": 3, "sell_count": 0, "cluster_buying": True}
    b = scoring.banger_check(res, d, risk_reward=None)
    # legacy FV-floor path: composite 150 >= 0.7*100 -> condition met
    assert b["is_banger"] is True


def test_position_size_quadrant_boost_and_risk_halving():
    lowhigh = {"applied": True, "quadrant": "LOW_RISK_HIGH_REWARD", "risk_tier": "LOW", "reward_tier": "HIGH"}
    s1 = scoring.position_size(7.0, "HIGH", False, "SECULAR", "MID", 8, "HIGH", risk_reward=lowhigh)
    s0 = scoring.position_size(7.0, "HIGH", False, "SECULAR", "MID", 8, "HIGH", risk_reward=None)
    assert s1["basis_pct"] >= s0["basis_pct"]
    assert any("low-risk/high-reward" in m for m in s1["modifiers"])

    risky = {"applied": True, "quadrant": "HIGH_RISK_MED_REWARD", "risk_tier": "HIGH", "reward_tier": "MED"}
    s2 = scoring.position_size(7.0, "HIGH", False, "SECULAR", "MID", 8, "HIGH", risk_reward=risky)
    assert s2["basis_pct"] == pytest.approx(s0["basis_pct"] * 0.5, abs=0.01)


def test_safe_wrapper_includes_layer_fallback():
    out = scoring._safe_apply_adjustments(7.0, None, None)  # forces an exception
    assert out["score_adjustments"].get("error")
    assert out["risk_reward"]["applied"] is False
