"""Quantified risk/reward layer — deterministic asymmetry math from the dossier.

Computes, with NO LLM involvement:
  - upside_pct:   distance from price to a blended fair-value/analyst target
  - downside_pct: distance from price to a conservative support floor
  - rr_ratio:     upside/downside (the asymmetry the agents previously only narrated)
  - risk_index:   0-10 composite of balance-sheet / quality / market / insider / data flags
  - adjustment:   bounded score delta from a risk x reward matrix (neutral diagonal)

Design rules:
  - Missing data NEVER penalizes — the layer returns {"applied": False} instead
    (the validator's LOW-confidence penalty already covers bad-data cases).
  - Cycle phase is deliberately EXCLUDED from risk_index: scoring.cycle_position_adjust
    already prices it, and double-counting would re-create the old over-penalizing bias.
  - Known accepted overlaps (documented, not mitigated): RSI/short-interest also inform
    the MarketStructure agent's raw score (same class of overlap as sector durability vs
    FundamentalForensics); ADR names tend to carry both validator warnings and FV blind
    spots, so the data group is capped at 1.5/10 to avoid triple-charging.
"""

from __future__ import annotations

# ── Upside ──
ANALYST_MIN_COVERAGE = 5      # min analysts before target_mean is trusted (mirrors consensus_gap_adjust)
FV_BLEND_WEIGHT      = 0.70   # weight on composite-FV upside when blending with the analyst gap
ANALYST_BLEND_WEIGHT = 0.30   # weight on analyst-target gap in the blend
UPSIDE_CAP   = 1.50           # cap upside at +150% — beyond this the number is noise
UPSIDE_FLOOR = -0.95          # sanity floor for deeply-overvalued names

# ── Downside (support floor) ──
FLOOR_FUNDAMENTAL_WEIGHT = 0.50  # blend weight: most conservative fundamental FV anchor
FLOOR_MARKET_WEIGHT      = 0.50  # blend weight: 52-week-low market floor
DOWNSIDE_MIN = 0.08   # nothing is riskless — floor the downside estimate at 8%
DOWNSIDE_MAX = 0.80   # beyond -80% the floor estimate is noise
FV_SANITY_LO = 0.05   # discard FVs below 5% of price (ratio-not-price guard, mirrors debate._sanitize_fv)
FV_SANITY_HI = 20.0   # discard FVs above 20x price

# ── Risk tiers (risk_index 0-10) ──
RISK_LOW_MAX = 2.0    # risk_index <= 2.0 -> LOW
RISK_MED_MAX = 4.5    # risk_index <= 4.5 -> MED, else HIGH

# ── Reward tiers (on upside fraction) ──
REWARD_HIGH_MIN = 0.35   # upside >= 35% -> HIGH
REWARD_MED_MIN  = 0.15   # upside >= 15% -> MED, else LOW

# ── Matrix (risk_tier, reward_tier) -> score adjustment; NEUTRAL diagonal:
#    high reward "affords" high risk — neither boosted nor penalized.
MATRIX = {
    ("LOW", "HIGH"): 0.75, ("LOW", "MED"): 0.40, ("MED", "HIGH"): 0.40,
    ("LOW", "LOW"): 0.0,   ("MED", "MED"): 0.0,  ("HIGH", "HIGH"): 0.0,
    ("MED", "LOW"): -0.40, ("HIGH", "MED"): -0.40, ("HIGH", "LOW"): -0.90,
}
RR_KICKER_MIN_RATIO = 3.0   # rr_ratio >= 3:1 AND LOW risk earns an extra kicker
RR_KICKER           = 0.25  # max total adjustment = 0.75 + 0.25 = 1.0
CONFIDENCE_DAMPEN   = 0.6   # shrink the adjustment when the FV inputs are shaky
BLIND_SPOT_DAMPEN_COUNT = 2 # blind_spot_flags >= this triggers the dampener

# Banger gate: computed asymmetry replaces the legacy FV-floor condition
BANGER_MIN_RR = 2.0


def _f(x):
    """Coerce to float, returning None on any failure (LLM/API fields can be strings)."""
    try:
        if x is None or isinstance(x, bool):
            return None
        v = float(x)
        return v if v == v else None  # NaN guard
    except (TypeError, ValueError):
        return None


def _sane_fv(v, price):
    """A fair-value candidate is usable iff it's a positive price-like number."""
    v = _f(v)
    if v is None or v <= 0 or price is None or price <= 0:
        return None
    return v if (FV_SANITY_LO * price) <= v <= (FV_SANITY_HI * price) else None


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _risk_index(dossier, fv, blind_spots, arch_conf):
    """0-10 composite of risk flags. Missing data never adds points.

    Returns (risk_index, components) where components are human-readable strings.
    """
    ratios = (dossier.get("financials") or {}).get("ratios_ttm") or {}
    balance = (dossier.get("financials") or {}).get("balance") or []
    bal0 = balance[0] if balance and isinstance(balance[0], dict) else {}
    insiders = dossier.get("insiders") or {}
    technicals = dossier.get("technicals") or {}
    mspr = dossier.get("insider_sentiment_mspr") or {}
    surprises = dossier.get("earnings_surprises") or []

    components = []
    groups = {}

    def add(group, cap, pts, label):
        before = groups.get(group, 0.0)
        gain = min(cap - before, pts)
        if gain > 0:
            groups[group] = before + gain
            components.append(f"{group}: {label} (+{gain:g})")

    # leverage (cap 2.5)
    total_debt = _f(bal0.get("total_debt"))
    cash = _f(bal0.get("cash"))
    ebitda = _f(ratios.get("ebitda"))
    if total_debt is not None:
        net_debt = total_debt - (cash or 0.0)
        if ebitda is not None and ebitda > 0:
            nde = net_debt / ebitda
            if nde > 3.0:
                add("leverage", 2.5, 1.5, f"net_debt/EBITDA {nde:.1f}x > 3")
            elif nde > 2.0:
                add("leverage", 2.5, 0.75, f"net_debt/EBITDA {nde:.1f}x > 2")
        elif total_debt > 0 and ebitda is not None and ebitda <= 0:
            add("leverage", 2.5, 1.5, "debt with negative EBITDA")
    de = _f(ratios.get("debt_equity"))
    if de is not None and de > 150:  # percent scale: 150 = 1.5x
        add("leverage", 2.5, 0.5, f"debt/equity {de:.0f}%")
    cr = _f(ratios.get("current_ratio"))
    if cr is not None and cr < 1.0:
        add("leverage", 2.5, 0.5, f"current ratio {cr:.2f}")

    # quality (cap 2.0) — NB: roic is percent-scale, wacc is a fraction
    roic = _f(ratios.get("roic"))
    wacc = _f(ratios.get("wacc"))
    if roic is not None and wacc is not None and (roic / 100.0) < wacc:
        add("quality", 2.0, 1.0, f"ROIC {roic:.1f}% < WACC {wacc * 100:.1f}%")
    nm = _f(ratios.get("net_margin"))
    if nm is not None and nm < 0:
        add("quality", 2.0, 1.0, f"negative net margin {nm:.1f}%")

    # earnings quality (cap 1.5)
    large_beats = sum(
        1 for s in surprises[:4]
        if isinstance(s, dict) and s.get("beat_quality") == "LARGE_BEAT"
    )
    if large_beats >= 2:
        add("earnings", 1.5, 0.75, f"{large_beats} LARGE_BEATs in last 4Q (one-time items?)")
    erm = _f(ratios.get("eps_revision_momentum"))
    if erm is not None and erm < -0.05:
        add("earnings", 1.5, 0.75, f"EPS revisions falling ({erm:+.2f})")

    # market (cap 2.5)
    beta = _f(ratios.get("beta"))
    if beta is not None and beta > 2.0:
        add("market", 2.5, 0.75, f"beta {beta:.1f}")
    short = _f(ratios.get("short_pct"))
    if short is not None:
        if short > 15:
            add("market", 2.5, 1.0, f"short interest {short:.1f}%")
        elif short > 8:
            add("market", 2.5, 0.5, f"short interest {short:.1f}%")
    rsi = _f(technicals.get("rsi_14"))
    from_high = _f(technicals.get("pct_from_52w_high"))
    if rsi is not None and from_high is not None and rsi > 75 and from_high > -2.0:
        add("market", 2.5, 0.75, f"euphoric: RSI {rsi:.0f} at 52w high")

    # insider (cap 1.5) — significant_sells is an int count in the dossier
    net_usd = _f(insiders.get("net_insider_usd"))
    sig_sells = _f(insiders.get("significant_sells")) or 0
    if net_usd is not None:
        if net_usd < -10_000_000 and sig_sells >= 3:
            add("insider", 1.5, 1.0, f"heavy insider selling (${net_usd / 1e6:.0f}M net)")
        elif net_usd < 0 and sig_sells >= 2:
            add("insider", 1.5, 0.5, "net insider selling")
    avg_mspr = _f(mspr.get("avg_mspr_3m"))
    if avg_mspr is not None and avg_mspr < -50:
        add("insider", 1.5, 0.5, f"MSPR 3m avg {avg_mspr:.0f}")

    # data (cap 1.5) — FV-layer signals only; validator's data_confidence keeps its own -0.5
    if (dossier.get("financials") or {}).get("ratios_ttm", {}).get("adr_mismatch"):
        add("data", 1.5, 0.5, "ADR/FX mismatch")
    if len(blind_spots) >= 2:
        add("data", 1.5, 0.5, f"{len(blind_spots)} FV blind spots")
    if arch_conf == "LOW" or fv.get("error"):
        add("data", 1.5, 0.5, "low-confidence valuation")

    return min(10.0, round(sum(groups.values()), 2)), components


def compute_risk_reward(dossier: dict) -> dict:
    """Deterministic risk/reward profile from an existing dossier. Never raises."""
    try:
        return _compute(dossier)
    except Exception as e:  # a layer bug must never break scoring or prompts
        return {"applied": False, "reason": f"risk_reward error: {e}"}


def _compute(dossier: dict) -> dict:
    quote = dossier.get("quote") or {}
    technicals = dossier.get("technicals") or {}
    price = _f(quote.get("price")) or _f(technicals.get("price"))
    if not price or price <= 0:
        return {"applied": False, "reason": "no price"}

    fv = dossier.get("fair_values") or {}
    valuation = dossier.get("valuation") or {}
    consensus = valuation.get("analyst_consensus") or {}
    ratios = (dossier.get("financials") or {}).get("ratios_ttm") or {}

    arch = fv.get("archetype")
    arch_conf = (arch or {}).get("confidence") if isinstance(arch, dict) else None
    blind_spots = fv.get("blind_spot_flags") or []
    if not isinstance(blind_spots, list):
        blind_spots = []

    # ── Upside ──────────────────────────────────────────────────────────────
    composite = _sane_fv(fv.get("composite_fair_value"), price)
    target = _sane_fv(consensus.get("target_mean"), price)
    n_analysts = _f(consensus.get("num_analysts"))
    covered = n_analysts is not None and n_analysts >= ANALYST_MIN_COVERAGE

    fv_up = (composite - price) / price if composite else None
    an_up = (target - price) / price if (target and covered) else None

    if fv_up is not None and an_up is not None:
        upside, upside_source = FV_BLEND_WEIGHT * fv_up + ANALYST_BLEND_WEIGHT * an_up, "fv_blend"
    elif fv_up is not None:
        upside, upside_source = fv_up, "fv_only"
    elif an_up is not None:
        upside, upside_source = an_up, "analyst_only"
    else:
        return {"applied": False, "reason": "no composite FV and no trusted analyst target"}
    upside = _clamp(upside, UPSIDE_FLOOR, UPSIDE_CAP)

    # ── Downside (support floor) ────────────────────────────────────────────
    fund_candidates = [
        c for c in (
            [_sane_fv(valuation.get("dcf_iv_per_share"), price),
             _sane_fv(fv.get("primary_fair_value"), price)]
            + [_sane_fv(m.get("fair_value"), price)
               for m in (fv.get("secondary_methods") or []) if isinstance(m, dict)]
        ) if c is not None
    ]
    fundamental_floor = min(fund_candidates) if fund_candidates else None
    low_52w = _f(technicals.get("52w_low"))
    market_floor = low_52w if (low_52w and low_52w > 0) else None

    if fundamental_floor is not None and market_floor is not None:
        support = FLOOR_FUNDAMENTAL_WEIGHT * fundamental_floor + FLOOR_MARKET_WEIGHT * market_floor
    elif fundamental_floor is not None:
        support = fundamental_floor
    elif market_floor is not None:
        support = market_floor
    else:
        return {"applied": False, "reason": "no downside floor candidates"}

    # Cash-rich hard floor: net cash per share can't be argued away by multiples
    bal = (dossier.get("financials") or {}).get("balance") or []
    bal0 = bal[0] if bal and isinstance(bal[0], dict) else {}
    cash = _f(bal0.get("cash"))
    shares = _f(ratios.get("shares_out"))
    net_cash_ps = None
    if cash is not None and shares and shares > 0:
        ncps = (cash - (_f(bal0.get("total_debt")) or 0.0)) / shares
        if ncps > 0:
            net_cash_ps = ncps
            support = max(support, net_cash_ps)

    downside = _clamp((price - support) / price, DOWNSIDE_MIN, DOWNSIDE_MAX)
    rr_ratio = round(max(upside, 0.0) / downside, 2)

    # ── Risk index + tiers ──────────────────────────────────────────────────
    risk_index, components = _risk_index(dossier, fv, blind_spots, arch_conf)
    risk_tier = "LOW" if risk_index <= RISK_LOW_MAX else ("MED" if risk_index <= RISK_MED_MAX else "HIGH")
    reward_tier = "HIGH" if upside >= REWARD_HIGH_MIN else ("MED" if upside >= REWARD_MED_MIN else "LOW")

    # ── Matrix adjustment ───────────────────────────────────────────────────
    base = MATRIX[(risk_tier, reward_tier)]
    if rr_ratio >= RR_KICKER_MIN_RATIO and risk_tier == "LOW":
        base += RR_KICKER
    dampen = CONFIDENCE_DAMPEN if (
        arch_conf in ("MEDIUM", "LOW")
        or len(blind_spots) >= BLIND_SPOT_DAMPEN_COUNT
        or upside_source == "analyst_only"
    ) else 1.0
    adjustment = round(base * dampen, 2)

    return {
        "applied": True,
        "upside_pct": round(upside * 100, 1),
        "downside_pct": round(downside * 100, 1),
        "rr_ratio": rr_ratio,
        "risk_index": risk_index,
        "risk_tier": risk_tier,
        "reward_tier": reward_tier,
        "quadrant": f"{risk_tier}_RISK_{reward_tier}_REWARD",
        "adjustment": adjustment,
        "dampened_by": dampen,
        "upside_source": upside_source,
        "downside_floor": round(support, 2),
        "floor_sources": {
            "fundamental": round(fundamental_floor, 2) if fundamental_floor else None,
            "market_52w_low": round(market_floor, 2) if market_floor else None,
            "net_cash_ps": round(net_cash_ps, 2) if net_cash_ps else None,
        },
        "risk_components": components,
    }


def compact(rr: dict) -> dict:
    """Small summary for the top-level result field — full trace stays in score_adjustments."""
    if not rr.get("applied"):
        return {"applied": False, "reason": rr.get("reason", "")}
    keep = ("applied", "rr_ratio", "upside_pct", "downside_pct", "risk_index",
            "risk_tier", "reward_tier", "quadrant", "adjustment")
    return {k: rr[k] for k in keep}


def llm_cross_check(rr: dict, bull_target, bear_floor, price) -> dict | None:
    """Compare the agents' bull/bear price targets against the computed R:R.

    Returns {"bull_target","bear_floor","llm_rr","divergent"} or None when unusable.
    """
    price = _f(price)
    bull = _f(bull_target)
    bear = _f(bear_floor)
    if not price or price <= 0 or bull is None or bear is None or bull <= 0 or bear <= 0:
        return None
    llm_rr = round(max((bull - price) / price, 0.0) / max((price - bear) / price, 0.01), 2)
    computed = _f(rr.get("rr_ratio")) or 0.0
    divergent = abs(llm_rr - computed) > max(1.0, 0.75 * max(computed, 1.0))
    return {
        "bull_target": bull,
        "bear_floor": bear,
        "llm_rr": llm_rr,
        "divergent": divergent,
    }
