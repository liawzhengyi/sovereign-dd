"""Tests for the BUY confirmation gate (verify.py).

Covers the Stage-1 deterministic quality gate (truth table), the Stage-2
red-team normalization, and verify_buy orchestration/routing incl. the
cost-saving short-circuit (no LLM call on a Stage-1 reject) and fail-open.
"""

import asyncio

import llm
import verify
from verify import quality_gate, verify_buy


def _run(coro):
    return asyncio.run(coro)


def _good_result(**over):
    """A clean BUY that passes every Stage-1 check."""
    r = {
        "consensus_score": 7.6, "consensus_grade": "STRONG BUY", "confidence": "HIGH",
        "converged": True, "score_spread": 1.2, "failed_agents": [], "data_confidence": "HIGH",
        "risk_reward": {"applied": True, "risk_index": 3.0, "rr_ratio": 2.5, "risk_tier": "LOW",
                        "upside_pct": 40.0, "downside_pct": 16.0,
                        "llm_cross_check": {"divergent": False}},
        "majority_thesis": "Wide moat at a discount.", "key_swing_factor": "Margin expansion.",
        "catalyst": "New product ramp.", "asymmetry_ratio": "3:1",
        "agent_final_scores": {"A": 7.5, "B": 7.7},
    }
    r.update(over)
    return r


_DOSSIER = {
    "profile": {"name": "Test Co", "market_cap_bn": 10.0},
    "quote": {"price": 100.0},
    "financials": {"ratios_ttm": {"pe": 20, "roic": 0.15, "debt_equity": 0.4}},
    "fair_values": {"composite_fair_value": 140.0},
}


def _patch_llm(monkeypatch, *, returns=None, raises=None, track=None):
    async def stub(system, user, **kwargs):
        if track is not None:
            track.append(True)
        if raises is not None:
            raise raises
        return returns
    monkeypatch.setattr(llm, "call_gemini_async", stub)


# ── Stage 1: quality_gate truth table ───────────────────────────────────────────

def test_quality_gate_clean_passes():
    ok, reasons = quality_gate(_good_result())
    assert ok and reasons == []


def test_quality_gate_not_converged_fails():
    ok, reasons = quality_gate(_good_result(converged=False))
    assert not ok and any("converge" in r for r in reasons)


def test_quality_gate_wide_spread_fails(monkeypatch):
    monkeypatch.setenv("VERIFY_MAX_SPREAD", "2.0")
    ok, reasons = quality_gate(_good_result(score_spread=2.5))
    assert not ok and any("spread" in r for r in reasons)


def test_quality_gate_low_confidence_fails():
    ok, reasons = quality_gate(_good_result(confidence="LOW"))
    assert not ok and any("confidence LOW" in r for r in reasons)


def test_quality_gate_failed_agent_fails():
    ok, reasons = quality_gate(_good_result(failed_agents=["ValuationEngine"]))
    assert not ok and any("agent(s) failed" in r for r in reasons)


def test_quality_gate_high_risk_index_fails(monkeypatch):
    monkeypatch.setenv("VERIFY_MAX_RISK_INDEX", "6.0")
    rr = _good_result()["risk_reward"] | {"risk_index": 7.5}
    ok, reasons = quality_gate(_good_result(risk_reward=rr))
    assert not ok and any("risk index" in r for r in reasons)


def test_quality_gate_divergent_rr_fails():
    rr = _good_result()["risk_reward"] | {"llm_cross_check": {"divergent": True}}
    ok, reasons = quality_gate(_good_result(risk_reward=rr))
    assert not ok and any("diverge" in r for r in reasons)


def test_quality_gate_low_data_confidence_fails():
    ok, reasons = quality_gate(_good_result(data_confidence="LOW"))
    assert not ok and any("data confidence LOW" in r for r in reasons)


def test_quality_gate_no_rr_still_passes():
    # Missing R:R is not a disqualifier on its own.
    ok, reasons = quality_gate(_good_result(risk_reward={"applied": False}))
    assert ok and reasons == []


# ── verify_buy orchestration / routing ──────────────────────────────────────────

def test_stage1_reject_skips_llm_call(monkeypatch):
    calls = []
    _patch_llm(monkeypatch, raises=AssertionError("LLM must not be called"), track=calls)
    v = _run(verify_buy("AAA", _good_result(converged=False), _DOSSIER))
    assert v["confirmed"] is False
    assert v["stage"] == 1 and v["verdict"] == "REJECTED_STAGE1"
    assert calls == []  # Stage-1 reject must not spend a call


def test_stage2_confirm(monkeypatch):
    _patch_llm(monkeypatch, returns='{"verdict":"CONFIRM","verification_score":8.2,'
                                    '"confirms_buy":true,"strongest_bear_point":"none material",'
                                    '"falsification_findings":[]}')
    v = _run(verify_buy("AAA", _good_result(), _DOSSIER))
    assert v["confirmed"] is True and v["verdict"] == "CONFIRM"
    assert v["verification_score"] == 8.2


def test_stage2_veto_rejects(monkeypatch):
    _patch_llm(monkeypatch, returns='{"verdict":"VETO","verification_score":3.0,'
                                    '"confirms_buy":false,"strongest_bear_point":"Guidance cut 2 days ago",'
                                    '"falsification_findings":["Q3 guide cut 15% (PR, Jun 12)"]}')
    v = _run(verify_buy("AAA", _good_result(), _DOSSIER))
    assert v["confirmed"] is False and v["verdict"] == "VETO"
    assert "Guidance cut" in v["strongest_bear_point"]
    assert v["falsification_findings"]


def test_stage2_downgrade_rejects(monkeypatch):
    _patch_llm(monkeypatch, returns='{"verdict":"DOWNGRADE","verification_score":5.5,'
                                    '"confirms_buy":false,"strongest_bear_point":"Margins peaking",'
                                    '"falsification_findings":[]}')
    v = _run(verify_buy("AAA", _good_result(), _DOSSIER))
    assert v["confirmed"] is False and v["verdict"] == "DOWNGRADE"


def test_stage2_error_fail_open(monkeypatch):
    monkeypatch.setenv("VERIFY_FAIL_OPEN", "1")
    _patch_llm(monkeypatch, raises=RuntimeError("504 deadline"))
    v = _run(verify_buy("AAA", _good_result(), _DOSSIER))
    assert v["confirmed"] is True and v["verdict"] == "UNVERIFIED"
    assert "red-team unavailable" in v.get("note", "")


def test_stage2_error_fail_closed(monkeypatch):
    monkeypatch.setenv("VERIFY_FAIL_OPEN", "0")
    _patch_llm(monkeypatch, raises=RuntimeError("504 deadline"))
    v = _run(verify_buy("AAA", _good_result(), _DOSSIER))
    assert v["confirmed"] is False and v["verdict"] == "UNVERIFIED"


def test_garbage_verdict_treated_as_error(monkeypatch):
    monkeypatch.setenv("VERIFY_FAIL_OPEN", "1")
    _patch_llm(monkeypatch, returns='{"verdict":"MAYBE","verification_score":5}')
    v = _run(verify_buy("AAA", _good_result(), _DOSSIER))
    # Unrecognized verdict -> _error -> fail-open path
    assert v["verdict"] == "UNVERIFIED"


def test_disabled_passthrough(monkeypatch):
    monkeypatch.setenv("VERIFY_BUYS", "0")
    calls = []
    _patch_llm(monkeypatch, raises=AssertionError("must not call when disabled"), track=calls)
    v = _run(verify_buy("AAA", _good_result(converged=False), _DOSSIER))
    assert v["confirmed"] is True and v["verdict"] == "DISABLED"
    assert calls == []
