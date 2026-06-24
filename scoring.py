"""Post-debate scoring pipeline — applies adjustments to raw consensus scores."""

from __future__ import annotations

from risk_reward import compute_risk_reward, compact as _rr_compact, BANGER_MIN_RR


# ── Sector / industry → earnings durability mappings ──────────────────────────

_DURABILITY_BY_SECTOR: dict[str, int] = {
    # High durability — recurring/contractual revenue
    "Technology":             8,   # Many SaaS/platform businesses
    "Healthcare":             7,   # Mix of recurring and project
    "Communication Services": 7,
    "Consumer Defensive":     8,   # Essential goods, stable demand
    "Utilities":              9,   # Regulated, contractual
    "Financials":             6,   # Rate-sensitive but recurring fees
    # Medium durability
    "Industrials":            5,
    "Real Estate":            6,
    # Lower durability — commodity/cyclical
    "Consumer Cyclical":      4,
    "Energy":                 3,
    "Basic Materials":        3,
}

# Industry-level overrides (checked first)
_DURABILITY_BY_INDUSTRY: dict[str, int] = {
    # High durability industries
    "Software—Application":    9,
    "Software—Infrastructure": 9,
    "Software":                9,
    "Insurance":               9,
    "Insurance—Diversified":   9,
    "Insurance—Specialty":     9,
    "Information Technology Services": 8,
    "Internet Content & Information":  8,
    "Semiconductor Equipment & Materials": 7,
    "Semiconductors":          6,
    # Lower durability industries
    "Oil & Gas E&P":           2,
    "Oil & Gas Integrated":    3,
    "Oil & Gas Midstream":     5,
    "Gold":                    2,
    "Silver":                  2,
    "Copper":                  2,
    "Steel":                   3,
    "Coal":                    2,
    "Agricultural Inputs":     3,
}


def earnings_durability_adjust(
    raw_score: float,
    sector: str,
    industry: str,
) -> tuple[float, int, str]:
    """Apply earnings durability multiplier to raw debate score.

    Returns (adjusted_score, durability_score, revenue_label).

    Formula: adjusted = raw * (0.7 + 0.03 * durability_score)
    - durability 10 → ×1.0  (no penalty)
    - durability 5  → ×0.85
    - durability 1  → ×0.73
    """
    durability = (
        _DURABILITY_BY_INDUSTRY.get(industry)
        or _DURABILITY_BY_SECTOR.get(sector)
        or 6  # default: moderate durability
    )

    # Build a label
    if durability >= 9:
        label = "contractual/recurring"
    elif durability >= 7:
        label = "structural/stable"
    elif durability >= 5:
        label = "market-dependent"
    elif durability >= 3:
        label = "commodity/cyclical"
    else:
        label = "speculative"

    multiplier = 0.7 + 0.03 * durability
    adjusted = round(min(10.0, max(1.0, raw_score * multiplier)), 2)
    return adjusted, durability, label


# ── Analyst consensus positioning ─────────────────────────────────────────────

def consensus_gap_adjust(
    score: float,
    price: float | None,
    analyst_target_mean: float | None,
    num_analysts: int | None = None,
) -> tuple[float, dict]:
    """Adjust score based on gap between current price and analyst consensus target.

    Returns (adjusted_score, details_dict).
    """
    if not price or not analyst_target_mean or price <= 0:
        return score, {"applied": False, "reason": "no analyst data"}
    # Thin coverage: a 1-2 analyst target is noisy/stale — don't move the score on it
    # (otherwise quality compounders that trade above sparse targets get penalized).
    if num_analysts is not None and num_analysts < 5:
        return score, {"applied": False, "reason": f"thin analyst coverage ({num_analysts})"}

    gap_pct = (analyst_target_mean - price) / price * 100

    if gap_pct > 30:
        adj, label = 0.3, "STRONG UPSIDE vs consensus"
    elif gap_pct > 10:
        adj, label = 0.15, "MODERATE UPSIDE vs consensus"
    elif gap_pct > -5:
        adj, label = 0.0, "AT CONSENSUS"
    elif gap_pct >= -20:
        adj, label = -0.15, "ABOVE CONSENSUS (moderate)"
    else:
        adj, label = -0.3, "SIGNIFICANTLY ABOVE CONSENSUS"

    adjusted = round(min(10.0, max(1.0, score + adj)), 2)
    return adjusted, {
        "applied": True,
        "price": price,
        "target": analyst_target_mean,
        "gap_pct": round(gap_pct, 1),
        "adjustment": adj,
        "label": label,
    }


# ── Cycle position adjustment ──────────────────────────────────────────────────

def cycle_position_adjust(
    score: float,
    cycle_phase: str | None,
    cycle_type: str | None,
    moat_composite: float | None,
) -> tuple[float, dict]:
    """Adjust score based on business cycle positioning.

    Returns (adjusted_score, details_dict).

    Early cycle + secular business → +0.5 to +1.0
    Trough + durable moat (>=7)    → +0.5 (supercycle entry)
    Late/Peak cycle + cyclical     → -0.5 to -1.0
    """
    if not cycle_phase:
        return score, {"applied": False}

    phase = (cycle_phase or "").upper()
    ctype = (cycle_type or "HYBRID").upper()

    adj = 0.0
    reason = ""

    if phase in ("EARLY",):
        if ctype == "SECULAR":
            adj = 1.0
            reason = "early cycle + secular business — maximum cycle boost"
        elif ctype in ("HYBRID", "DEFENSIVE"):
            adj = 0.5
            reason = "early cycle — moderate boost"
        else:  # CYCLICAL
            adj = 0.3
            reason = "early cycle — small boost even for cyclical"

    elif phase == "TROUGH":
        if moat_composite and moat_composite >= 7:
            adj = 0.5
            reason = f"cyclical trough + strong moat ({moat_composite:.1f}) — supercycle entry signal"
        else:
            adj = 0.2
            reason = "at trough — small speculative boost"

    elif phase == "MID":
        adj = 0.0
        reason = "mid-cycle — no adjustment"

    elif phase in ("LATE", "PEAK"):
        if ctype == "CYCLICAL":
            adj = -1.0
            reason = f"{phase.lower()} cycle + cyclical business — full penalty"
        elif ctype == "HYBRID":
            adj = -0.5
            reason = f"{phase.lower()} cycle + hybrid business — moderate penalty"
        else:
            adj = -0.2
            reason = f"{phase.lower()} cycle — small penalty even for secular"

    adjusted = round(min(10.0, max(1.0, score + adj)), 2)
    return adjusted, {
        "applied": adj != 0.0,
        "cycle_phase": phase,
        "cycle_type": ctype,
        "adjustment": adj,
        "reason": reason,
    }


# ── Data confidence penalty ────────────────────────────────────────────────────

def data_confidence_adjust(score: float, data_confidence: str) -> tuple[float, dict]:
    """Penalize score when data quality is LOW.

    Returns (adjusted_score, details_dict).
    """
    if data_confidence == "LOW":
        adjusted = round(max(1.0, score - 0.5), 2)
        return adjusted, {"applied": True, "adjustment": -0.5, "reason": "data quality LOW"}
    return score, {"applied": False}


# ── Banger detector ────────────────────────────────────────────────────────────

def banger_check(result: dict, dossier: dict, risk_reward: dict | None = None) -> dict:
    """Check if a stock qualifies as an asymmetric 'BANGER' opportunity.

    Four conditions must ALL be met:
    1. Adjusted score >= 7.5
    2. Catalyst present (from debate output)
    3. Computed risk/reward ratio >= 2:1 (falls back to the legacy
       fair-value-floor test when the risk_reward layer isn't applied)
    4. Insider net buying (buy_count > sell_count)

    Returns {"is_banger": bool, "conditions_met": list, "conditions_failed": list, "reason": str}
    """
    conditions_met = []
    conditions_failed = []

    # Condition 1: high adjusted score
    score = result.get("consensus_score", 0)
    if score >= 7.5:
        conditions_met.append(f"score {score:.1f} >= 7.5")
    else:
        conditions_failed.append(f"score {score:.1f} < 7.5")

    # Condition 2: catalyst present
    catalyst = result.get("catalyst", "") or ""
    cycle_pos = result.get("cycle_position", {})
    cycle_phase = (cycle_pos.get("phase") or "").upper() if isinstance(cycle_pos, dict) else ""
    if catalyst and len(catalyst.strip()) > 10:
        conditions_met.append("specific catalyst identified")
    elif cycle_phase in ("EARLY", "TROUGH"):
        conditions_met.append(f"cycle at {cycle_phase} (favorable entry)")
    else:
        conditions_failed.append("no specific catalyst identified")

    # Condition 3: asymmetric risk/reward. The computed rr_ratio (risk_reward.py)
    # is strictly more informative than the old FV-floor test — it measures both
    # the upside AND a conservative downside floor. Legacy FV-floor kept verbatim
    # as the fallback so old saved results / missing-data names behave as before.
    if risk_reward and risk_reward.get("applied"):
        rr_ratio = risk_reward.get("rr_ratio") or 0.0
        if rr_ratio >= BANGER_MIN_RR:
            conditions_met.append(f"rr_ratio {rr_ratio:.1f} >= {BANGER_MIN_RR:.1f} (computed asymmetry)")
        else:
            conditions_failed.append(f"insufficient risk/reward asymmetry (rr {rr_ratio:.1f} < {BANGER_MIN_RR:.1f})")
    else:
        price = (dossier.get("quote") or {}).get("price")
        dcf_iv = (dossier.get("valuation") or {}).get("dcf_iv_per_share")
        # Core principle: Python does the math. Prefer the Python-computed composite
        # (dossier["fair_values"]) and the dossier DCF over the debate's
        # fair_value_composite, which may be the LLM moderator's number (see debate.py).
        def _f(v):
            try:
                return float(v) if v is not None else None
            except (ValueError, TypeError):
                return None
        py_composite  = _f((dossier.get("fair_values") or {}).get("composite_fair_value"))
        dcf_iv        = _f(dcf_iv)
        llm_composite = _f(result.get("fair_value_composite"))
        if py_composite is not None:
            floor_iv, source = py_composite, "Python FV composite"
        elif dcf_iv is not None:
            floor_iv, source = dcf_iv, "DCF IV"
        else:
            floor_iv, source = llm_composite, "LLM FV composite"
        if price and floor_iv and price > 0 and floor_iv >= price * 0.7:
            conditions_met.append(f"{source} ${floor_iv:.2f} >= 70% of price ${price:.2f}")
        else:
            conditions_failed.append("insufficient fair value floor support")

    # Condition 4: insider net buying
    insiders = dossier.get("insiders") or {}
    buy_count = insiders.get("buy_count", 0)
    sell_count = insiders.get("sell_count", 0)
    cluster = insiders.get("cluster_buying", False)
    if cluster or buy_count > sell_count:
        detail = "cluster buying" if cluster else f"net buying ({buy_count}B/{sell_count}S)"
        conditions_met.append(detail)
    else:
        conditions_failed.append(f"no insider net buying ({buy_count}B/{sell_count}S)")

    is_banger = len(conditions_failed) == 0

    if is_banger:
        reason = "All 4 conditions met: " + "; ".join(conditions_met)
    else:
        reason = "Failed: " + "; ".join(conditions_failed)

    return {
        "is_banger": is_banger,
        "conditions_met": conditions_met,
        "conditions_failed": conditions_failed,
        "reason": reason,
    }


# ── Position sizing guidance ───────────────────────────────────────────────────

def position_size(
    score: float,
    confidence: str,
    is_banger: bool,
    cycle_type: str | None,
    cycle_phase: str | None,
    durability_score: int,
    data_confidence: str,
    risk_reward: dict | None = None,
) -> dict:
    """Map adjusted score to suggested portfolio allocation %.

    Returns {"range": "1-2%", "basis_pct": float, "reasoning": str, "modifiers": list}
    """
    # Base allocation from score
    if score >= 9.0:
        base = 0.05   # 4-6%
        label = "4-6%"
    elif score >= 8.0:
        base = 0.035  # 3-4%
        label = "3-4%"
    elif score >= 6.5:
        base = 0.015  # 1-2%
        label = "1-2%"
    elif score >= 5.0:
        base = 0.005  # 0.5%
        label = "0.5%"
    else:
        base = 0.0
        label = "0%"

    modifiers = []
    multiplier = 1.0

    # Halve for low conviction
    if confidence == "LOW":
        multiplier *= 0.5
        modifiers.append("halved: low conviction")

    # Halve for commodity/cyclical at late cycle
    phase = (cycle_phase or "").upper()
    ctype = (cycle_type or "HYBRID").upper()
    if ctype == "CYCLICAL" and phase in ("LATE", "PEAK"):
        multiplier *= 0.5
        modifiers.append("halved: cyclical + late cycle")

    # Halve for low earnings durability
    if durability_score < 5:
        multiplier *= 0.5
        modifiers.append(f"halved: low earnings durability ({durability_score}/10)")

    # Halve for low data confidence
    if data_confidence == "LOW":
        multiplier *= 0.5
        modifiers.append("halved: low data confidence")

    # Risk/reward quadrant: size up the prize quadrant, size down un-afforded risk
    # (HIGH risk with HIGH reward is partially afforded — 0.75x instead of 0.5x).
    rrq = risk_reward or {}
    if rrq.get("applied"):
        if rrq.get("quadrant") == "LOW_RISK_HIGH_REWARD":
            multiplier *= 1.25
            modifiers.append("1.25× low-risk/high-reward quadrant")
        elif rrq.get("risk_tier") == "HIGH":
            f = 0.75 if rrq.get("reward_tier") == "HIGH" else 0.5
            multiplier *= f
            modifiers.append(f"{f}×: HIGH risk tier")

    # BANGER bonus: allow 1.5x
    if is_banger:
        multiplier *= 1.5
        modifiers.append("1.5× BANGER bonus")

    # Cap final_pct to the tier ceiling, and make `reasoning` reflect the ACTUAL
    # sizing (the prior code printed "× 1.50" even when the cap clamped it lower).
    tier_cap = {0.05: 0.06, 0.035: 0.04, 0.015: 0.02, 0.005: 0.01, 0.0: 0.0}
    cap = tier_cap.get(base, base * 2)
    uncapped = base * multiplier
    capped = min(cap, uncapped)
    final_pct = round(capped * 100, 2)
    if multiplier == 1.0:
        reasoning = label + " base"
    elif capped < uncapped:
        reasoning = label + f" base (capped at {cap * 100:.0f}% tier ceiling)"
    else:
        reasoning = label + f" base × {multiplier:.2f}"
    if modifiers:
        reasoning += " (" + ", ".join(modifiers) + ")"

    return {
        "range": label,
        "basis_pct": final_pct,
        "reasoning": reasoning,
        "modifiers": modifiers,
    }


# ── Portfolio overlap adjustment ───────────────────────────────────────────────

def portfolio_overlap_adjust(
    score: float,
    sector: str,
    portfolio_sectors: dict[str, str],  # {ticker: sector}
) -> tuple[float, dict]:
    """Adjust score based on portfolio sector overlap.

    Returns (adjusted_score, details_dict).
    """
    if not portfolio_sectors:
        return score, {"applied": False}

    all_sectors = list(portfolio_sectors.values())
    sector_count = sum(1 for s in all_sectors if s == sector)
    total = len(all_sectors)

    overlap_ratio = sector_count / total if total > 0 else 0

    if overlap_ratio > 0.7:
        adj = -0.5
        flag = "REDUNDANT_EXPOSURE"
    elif overlap_ratio < 0.20 and sector not in all_sectors:
        adj = 0.15
        flag = "DIVERSIFICATION_VALUE"
    else:
        adj = 0.0
        flag = None

    # Concentration warning
    concentration_warning = None
    if total > 0:
        sector_pcts = {}
        for s in all_sectors:
            sector_pcts[s] = sector_pcts.get(s, 0) + 1
        for s, cnt in sector_pcts.items():
            if cnt / total > 0.4:
                concentration_warning = f"Portfolio already >40% {s}"

    adjusted = round(min(10.0, max(1.0, score + adj)), 2)
    return adjusted, {
        "applied": adj != 0.0,
        "sector_overlap_count": sector_count,
        "sector_overlap_ratio": round(overlap_ratio, 2),
        "adjustment": adj,
        "flag": flag,
        "concentration_warning": concentration_warning,
    }


# ── Grade function ─────────────────────────────────────────────────────────────

from grading import grade, grade_hold  # re-exported for callers that do `from scoring import grade`


# ── Master pipeline orchestrator ───────────────────────────────────────────────

def apply_adjustments(
    raw_score: float,
    result: dict,
    dossier: dict,
    portfolio_sectors: dict[str, str] | None = None,
    is_holding: bool = False,
) -> dict:
    """Run the full post-debate scoring pipeline.

    Input: raw_score from LLM debate consensus, full result dict, full dossier dict.
    Output: enriched dict with adjusted score, grade, all adjustment details, banger, position guidance.

    Pipeline order:
    1. Earnings durability multiplier
    2. Analyst consensus gap adjustment (superseded by risk/reward when it applies)
    3. Cycle position adjustment
    4. Quantified risk/reward matrix (risk_reward.py)
    5. Data confidence penalty
    6. Portfolio overlap (if portfolio provided)
    7. Grade assignment
    8. Banger detection
    9. Position sizing
    """
    profile = dossier.get("profile") or {}
    sector = profile.get("sector") or "Unknown"
    industry = profile.get("industry") or ""  # may not be in profile, will default gracefully
    quote = dossier.get("quote") or {}
    price = quote.get("price")
    valuation = dossier.get("valuation") or {}
    _consensus = valuation.get("analyst_consensus") or {}
    analyst_target = _consensus.get("target_mean")
    num_analysts = _consensus.get("num_analysts")
    dq = dossier.get("data_quality") or {}
    data_confidence = dq.get("data_confidence", "HIGH")
    cycle_type = dossier.get("cycle_type")

    # Extract cycle phase from debate result (set by CatalystHunter synthesis)
    cycle_pos = result.get("cycle_position") or {}
    if isinstance(cycle_pos, dict):
        cycle_phase = cycle_pos.get("phase")
    else:
        cycle_phase = None

    # Extract moat composite from debate result (set by StructuralEdge synthesis)
    # Coerce to float — LLMs sometimes return the string "null" or "7.5"
    _mc = result.get("moat_composite")
    try:
        moat_composite = float(_mc) if _mc is not None and str(_mc).lower() != "null" else None
    except (ValueError, TypeError):
        moat_composite = None

    # Quantified risk/reward — computed once, reused by the matrix step, banger
    # gate, and position sizing. Returns {"applied": False, ...} on missing data.
    rr = compute_risk_reward(dossier)

    score = raw_score
    adjustments = {"raw": raw_score, "is_holding": is_holding}

    # 1. Earnings durability — for holdings, floor the multiplier at 0.92 so a
    # semicon compounder (durability 6 → ×0.88 normally) doesn't get docked just
    # for being in a "market-dependent" bucket once we already own it.
    pre_score = score
    score, durability_score, durability_label = earnings_durability_adjust(score, sector, industry)
    if is_holding and pre_score > 0:
        floored_mult = max(score / pre_score, 0.92)
        score = round(min(10.0, max(1.0, pre_score * floored_mult)), 2)
    adjustments["earnings_durability"] = {
        "score": durability_score,
        "label": durability_label,
        "result": score,
        "hold_floor_applied": is_holding,
    }

    # 2. Analyst consensus — skipped entirely for holdings. Thin analyst coverage
    # on a quality compounder routinely flags the stock as "above consensus" and
    # docks the score; that's an entry-decision signal, not a hold-decision one.
    if is_holding:
        adjustments["consensus_gap"] = {"applied": False, "reason": "skipped in hold-mode", "result": score}
    elif rr.get("applied"):
        # The risk/reward layer already blends the analyst-target gap into its
        # upside measure (0.3 weight) — applying both would double-count it.
        adjustments["consensus_gap"] = {"applied": False, "reason": "superseded by risk_reward layer", "result": score}
    else:
        score, consensus_details = consensus_gap_adjust(score, price, analyst_target, num_analysts)
        adjustments["consensus_gap"] = {**consensus_details, "result": score}

    # 3. Cycle position — halved for holdings. Late-cycle macro is a real risk
    # but shouldn't unilaterally turn a held compounder into a SELL.
    score_before_cycle = score
    score, cycle_details = cycle_position_adjust(score, cycle_phase, cycle_type, moat_composite)
    if is_holding and cycle_details.get("applied"):
        full_delta = score - score_before_cycle
        halved_delta = full_delta * 0.5
        score = round(min(10.0, max(1.0, score_before_cycle + halved_delta)), 2)
        cycle_details = {
            **cycle_details,
            "adjustment": round(halved_delta, 2),
            "reason": (cycle_details.get("reason") or "") + " (halved for hold-mode)",
        }
    adjustments["cycle_position"] = {**cycle_details, "result": score}

    # 4. Quantified risk/reward matrix — full strength in BOTH scout and hold mode:
    # for holdings this IS the "remaining reward affords the risk" logic (a held
    # name with little upside left and elevated risk should feel trim pressure).
    if rr.get("applied"):
        score = round(min(10.0, max(1.0, score + rr["adjustment"])), 2)
    adjustments["risk_reward"] = {**rr, "result": score}

    # 5. Data confidence
    score, dc_details = data_confidence_adjust(score, data_confidence)
    adjustments["data_confidence"] = {**dc_details, "result": score}

    # 6. Portfolio overlap (optional)
    if portfolio_sectors is not None:
        score, overlap_details = portfolio_overlap_adjust(score, sector, portfolio_sectors)
        adjustments["portfolio_overlap"] = {**overlap_details, "result": score}

    # 7. Grade — hold ladder (ADD/HOLD/TRIM/EXIT) for holdings, entry ladder otherwise.
    final_grade = grade_hold(score) if is_holding else grade(score)
    adjustments["final"] = score

    # 8. Banger detection
    result_for_banger = dict(result)
    result_for_banger["consensus_score"] = score  # use adjusted score for banger check
    banger = banger_check(result_for_banger, dossier, risk_reward=rr)

    # 9. Position sizing
    confidence = result.get("confidence", "MEDIUM")
    sizing = position_size(
        score=score,
        confidence=confidence,
        is_banger=banger["is_banger"],
        cycle_type=cycle_type,
        cycle_phase=cycle_phase,
        durability_score=durability_score,
        data_confidence=data_confidence,
        risk_reward=rr,
    )

    return {
        "adjusted_score":    score,
        "consensus_grade":   final_grade,
        "score_adjustments": adjustments,
        "banger":            banger,
        "position_guidance": sizing,
        "risk_reward":       _rr_compact(rr),
    }


def _safe_apply_adjustments(raw_score, result, dossier, portfolio_sectors=None, is_holding=False):
    """Wrapper around apply_adjustments that never raises — degrades gracefully on error."""
    try:
        return apply_adjustments(raw_score, result, dossier, portfolio_sectors, is_holding=is_holding)
    except Exception as e:
        fallback_grade = grade_hold(raw_score) if is_holding else grade(raw_score)
        return {
            "adjusted_score":    raw_score,
            "consensus_grade":   fallback_grade,
            "score_adjustments": {"raw": raw_score, "is_holding": is_holding, "error": str(e)},
            "banger":            {"is_banger": False, "conditions_met": [], "conditions_failed": ["pipeline error"], "reason": str(e)},
            "position_guidance": {"range": "N/A", "basis_pct": 0.0, "reasoning": "scoring pipeline error", "modifiers": []},
            "risk_reward":       {"applied": False, "reason": "pipeline error"},
        }
