"""Archetype-based fair value engine."""

from __future__ import annotations


# ── Archetype constants ───────────────────────────────────────────────────────

ARCHETYPE_ASSET_LIGHT    = "ASSET_LIGHT_GROWTH"
ARCHETYPE_CYCLICAL       = "CAPITAL_INTENSIVE_CYCLICAL"
ARCHETYPE_FINANCIAL      = "FINANCIAL_INSTITUTION"
ARCHETYPE_INFRASTRUCTURE = "ASSET_HEAVY_INFRASTRUCTURE"
ARCHETYPE_EARLY_STAGE    = "EARLY_STAGE"
ARCHETYPE_MATURE         = "MATURE_COMPOUNDER"

# EV/FCF target multiple for mature compounders, indexed by GICS sector.
# High-quality compounder sectors (Technology, Healthcare) command premium multiples;
# capital-heavy or regulated sectors (Energy, Utilities) trade at lower FCF yields.
# Adjusted upward vs DCF terminal multiples to reflect quality premium.
_MATURE_FCF_MULTIPLE: dict[str, float] = {
    "Technology": 28,
    "Communication Services": 22,
    "Healthcare": 22,
    "Consumer Cyclical": 20,
    "Consumer Defensive": 20,
    "Industrials": 18,
    "Energy": 12,
    "Utilities": 14,
    "Financials": 14,
    "Real Estate": 16,
    "Basic Materials": 14,
}
_DEFAULT_MATURE_FCF_MULTIPLE = 20


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(val, fallback=None):
    """Return val if not None and not NaN-like, else fallback."""
    if val is None:
        return fallback
    try:
        if isinstance(val, float) and (val != val):  # NaN check
            return fallback
    except Exception:
        pass
    return val


def _norm_margin(v):
    """Normalise a margin to a fraction (0.654 for 65.4%).
    All call sites pass ratios produced by dossier.py _pct(), which always returns
    a percentage (e.g. 65.4), so divide unconditionally. The previous abs(v)>1
    heuristic silently misread legitimately sub-1% margins (e.g. 0.5% -> 0.5).
    """
    if v is None:
        return None
    try:
        return float(v) / 100
    except (ValueError, TypeError):
        return None


def _cagr(start, end, years):
    """Compound annual growth rate. Returns None if inputs invalid."""
    try:
        if start and end is not None and years > 0 and start > 0 and end > 0:
            return (end / start) ** (1 / years) - 1
    except Exception:
        pass
    return None


def _get_income(dossier: dict, idx: int = 0) -> dict:
    """Safely retrieve income statement at index."""
    try:
        return dossier["financials"]["income"][idx] or {}
    except (KeyError, IndexError, TypeError):
        return {}


def _get_balance(dossier: dict, idx: int = 0) -> dict:
    """Safely retrieve balance sheet at index."""
    try:
        return dossier["financials"]["balance"][idx] or {}
    except (KeyError, IndexError, TypeError):
        return {}


def _get_cashflow(dossier: dict, idx: int = 0) -> dict:
    """Safely retrieve cash flow statement at index."""
    try:
        return dossier["financials"]["cashflow"][idx] or {}
    except (KeyError, IndexError, TypeError):
        return {}


def _get_ratios(dossier: dict) -> dict:
    """Safely retrieve TTM ratios."""
    try:
        return dossier["financials"]["ratios_ttm"] or {}
    except (KeyError, TypeError):
        return {}


def _get_macro(dossier: dict) -> dict:
    """Safely retrieve macro data."""
    try:
        return dossier["macro"] or {}
    except (KeyError, TypeError):
        return {}


def _get_profile(dossier: dict) -> dict:
    """Safely retrieve profile data."""
    try:
        return dossier["profile"] or {}
    except (KeyError, TypeError):
        return {}


def _get_price(dossier: dict):
    """Safely retrieve current price."""
    try:
        return _safe(dossier["quote"]["price"])
    except (KeyError, TypeError):
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def classify_archetype(dossier: dict) -> dict:
    """Classify the stock into one of 6 archetypes.

    Check order matters — first match wins:
    1. EARLY_STAGE
    2. FINANCIAL_INSTITUTION
    3. ASSET_HEAVY_INFRASTRUCTURE
    4. ASSET_LIGHT_GROWTH
    5. CAPITAL_INTENSIVE_CYCLICAL
    6. MATURE_COMPOUNDER (default)
    """
    profile  = _get_profile(dossier)
    ratios   = _get_ratios(dossier)
    income0  = _get_income(dossier, 0)
    income1  = _get_income(dossier, 1)
    cf0      = _get_cashflow(dossier, 0)

    sector    = _safe(profile.get("sector"), "")
    yf_sector = _safe(profile.get("yf_sector"), "")   # GICS sector from yfinance
    industry  = _safe(profile.get("industry"), "")

    net_income    = _safe(income0.get("net_income"))
    fcf           = _safe(cf0.get("free_cash_flow")) or _safe(ratios.get("fcf"))
    revenue_ttm   = _safe(ratios.get("revenue_ttm"))
    gross_margin  = _norm_margin(ratios.get("gross_margin"))
    capex         = _safe(cf0.get("capex"))

    # ── Revenue growth YoY ────────────────────────────────────────────────────
    rev0 = _safe(income0.get("revenue"))
    rev1 = _safe(income1.get("revenue"))
    revenue_growth_yoy = None
    if rev0 is not None and rev1 is not None and rev1 > 0:
        revenue_growth_yoy = (rev0 - rev1) / rev1

    # ── Capex intensity ───────────────────────────────────────────────────────
    capex_intensity = None
    if capex is not None and revenue_ttm is not None and revenue_ttm > 0:
        capex_intensity = abs(capex) / revenue_ttm

    # ─────────────────────────────────────────────────────────────────────────
    # 1. EARLY_STAGE
    # ─────────────────────────────────────────────────────────────────────────
    early_stage_net_negative  = (net_income is not None and net_income < 0)
    early_stage_fcf_negative  = (fcf is not None and fcf < 0)
    early_stage_sector_match  = sector in ("Healthcare", "Biotechnology")
    early_stage_small_rev     = (revenue_ttm is not None and revenue_ttm < 100_000_000)

    if early_stage_net_negative and early_stage_fcf_negative and (early_stage_sector_match or early_stage_small_rev):
        confidence = "HIGH" if (net_income is not None and fcf is not None) else "MEDIUM"
        reason_parts = []
        if early_stage_sector_match:
            reason_parts.append(f"sector={sector}")
        if early_stage_small_rev:
            reason_parts.append(f"revenue_ttm={revenue_ttm:,.0f}")
        reasoning = (
            f"Pre-profit company with negative net income ({net_income}) and FCF ({fcf})"
            + (f"; {', '.join(reason_parts)}" if reason_parts else "")
            + "."
        )
        return {
            "archetype": ARCHETYPE_EARLY_STAGE,
            "confidence": confidence,
            "reasoning": reasoning,
            "secondary_archetype": None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 2. FINANCIAL_INSTITUTION
    # ─────────────────────────────────────────────────────────────────────────
    _FINANCIAL_INDUSTRIES = ("Bank", "Insurance", "Asset Management", "Capital Markets", "Mortgage")
    is_financial_sector   = (sector == "Financial Services") or (yf_sector == "Financial Services")
    is_financial_industry = any(kw in industry for kw in _FINANCIAL_INDUSTRIES)

    if is_financial_sector or is_financial_industry:
        confidence = "HIGH" if is_financial_sector else "MEDIUM"
        reasoning = (
            f"Financial institution classification: sector={sector!r}, industry={industry!r}."
        )
        return {
            "archetype": ARCHETYPE_FINANCIAL,
            "confidence": confidence,
            "reasoning": reasoning,
            "secondary_archetype": None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 3. ASSET_HEAVY_INFRASTRUCTURE
    # ─────────────────────────────────────────────────────────────────────────
    _INFRA_INDUSTRIES = ("REIT", "Utility", "Utilities", "Telecom")
    is_realestate_sector  = (sector == "Real Estate") or (yf_sector == "Real Estate")
    is_infra_industry     = any(kw in industry for kw in _INFRA_INDUSTRIES)

    if is_realestate_sector or is_infra_industry:
        confidence = "HIGH" if is_realestate_sector else "MEDIUM"
        reasoning = (
            f"Asset-heavy infrastructure: sector={sector!r}, industry={industry!r}."
        )
        return {
            "archetype": ARCHETYPE_INFRASTRUCTURE,
            "confidence": confidence,
            "reasoning": reasoning,
            "secondary_archetype": None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 4. ASSET_LIGHT_GROWTH
    # ─────────────────────────────────────────────────────────────────────────
    _GROWTH_SECTORS = ("Technology", "Communication Services")
    high_gross_margin  = (gross_margin is not None and gross_margin > 0.60)
    # Check both Finnhub industry AND yfinance GICS sector — Finnhub uses non-standard names
    # (e.g. "Semiconductors" for NVDA, "Internet Content & Information" for META) that don't
    # match the GICS strings used here; yf_sector carries the canonical classification.
    growth_sector      = (sector in _GROWTH_SECTORS) or (yf_sector in _GROWTH_SECTORS)
    strong_rev_growth  = (revenue_growth_yoy is not None and revenue_growth_yoy > 0.10)

    if high_gross_margin and growth_sector and strong_rev_growth:
        confidence = "HIGH"
        reasoning = (
            f"Asset-light growth: gross_margin={gross_margin:.1%}, "
            f"sector={sector!r}, revenue_growth_yoy={revenue_growth_yoy:.1%}."
        )
        return {
            "archetype": ARCHETYPE_ASSET_LIGHT,
            "confidence": confidence,
            "reasoning": reasoning,
            "secondary_archetype": None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 5. CAPITAL_INTENSIVE_CYCLICAL
    # ─────────────────────────────────────────────────────────────────────────
    _CYCLICAL_SECTORS = ("Energy", "Basic Materials", "Industrials")
    is_cyclical_sector    = (sector in _CYCLICAL_SECTORS) or (yf_sector in _CYCLICAL_SECTORS)
    is_capex_heavy        = (capex_intensity is not None and capex_intensity > 0.12)

    if is_cyclical_sector or is_capex_heavy:
        confidence = "HIGH" if is_cyclical_sector else "MEDIUM"
        reasoning_parts = []
        if is_cyclical_sector:
            reasoning_parts.append(f"sector={sector!r}")
        if is_capex_heavy:
            reasoning_parts.append(f"capex_intensity={capex_intensity:.2f}")
        reasoning = "Capital-intensive cyclical: " + ", ".join(reasoning_parts) + "."
        secondary = ARCHETYPE_MATURE if (not is_cyclical_sector and is_capex_heavy) else None
        return {
            "archetype": ARCHETYPE_CYCLICAL,
            "confidence": confidence,
            "reasoning": reasoning,
            "secondary_archetype": secondary,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 5b. ASSET_LIGHT_GROWTH — relaxed pass (MEDIUM confidence)
    # Catches hybrid tech companies (e.g. hardware+services) that miss the strict
    # 60% margin / 10% growth thresholds but are clearly not cyclicals or mature.
    # ─────────────────────────────────────────────────────────────────────────
    moderate_gross_margin = (gross_margin is not None and gross_margin > 0.40)
    positive_rev_growth   = (revenue_growth_yoy is not None and revenue_growth_yoy > 0)

    if moderate_gross_margin and growth_sector and positive_rev_growth:
        return {
            "archetype": ARCHETYPE_ASSET_LIGHT,
            "confidence": "MEDIUM",
            "reasoning": (
                f"Asset-light growth (relaxed): gross_margin={gross_margin:.1%}, "
                f"sector={sector!r}" + (f"/yf={yf_sector!r}" if yf_sector and yf_sector != sector else "")
                + f", revenue_growth_yoy={revenue_growth_yoy:.1%}."
            ),
            "secondary_archetype": ARCHETYPE_MATURE,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 6. MATURE_COMPOUNDER (default)
    # ─────────────────────────────────────────────────────────────────────────
    # Determine if we fell through due to missing data or genuine fit
    missing_key_data = (gross_margin is None or revenue_growth_yoy is None)
    confidence = "LOW" if missing_key_data else "MEDIUM"
    reasoning = (
        "Defaulted to mature compounder — no earlier archetype conditions matched"
        + (" (some key metrics were None)" if missing_key_data else "")
        + "."
    )
    return {
        "archetype": ARCHETYPE_MATURE,
        "confidence": confidence,
        "reasoning": reasoning,
        "secondary_archetype": None,
    }


def compute_fair_values(dossier: dict) -> dict:
    """Main entry point: classify archetype, run the matching engine, return unified result."""

    _ENGINES = {
        ARCHETYPE_ASSET_LIGHT:    _value_asset_light,
        ARCHETYPE_CYCLICAL:       _value_cyclical,
        ARCHETYPE_FINANCIAL:      _value_financial,
        ARCHETYPE_INFRASTRUCTURE: _value_infrastructure,
        ARCHETYPE_EARLY_STAGE:    _value_early_stage,
        ARCHETYPE_MATURE:         _value_mature_compounder,
    }

    archetype_info: dict = {}  # init before try — prevents UnboundLocalError in handler
    try:
        archetype_info = classify_archetype(dossier)
        archetype_key  = archetype_info["archetype"]
        engine = _ENGINES[archetype_key]
        result = engine(dossier)
    except Exception as exc:
        # Don't kill the batch on one ticker's valuation bug, but make it LOUD:
        # log the traceback and flag the failure so report/notify treat it as a red
        # flag rather than a benign "valuation unavailable".
        import traceback
        print(f"  [fair_value] valuation FAILED for {dossier.get('ticker', '?')}: {exc}")
        traceback.print_exc()
        return {
            "archetype": archetype_info,
            "primary_method": None,
            "primary_fair_value": None,
            "secondary_methods": [],
            "composite_fair_value": None,
            "margin_of_safety": None,
            "archetype_metrics": {},
            "blind_spot_flags": [],
            "invalid_methods": [],
            "valuation_failed": True,
            "error": str(exc),
        }

    primary       = result.get("primary") or {}
    secondary_raw = result.get("secondary") or []
    composite     = _safe(result.get("composite"))
    price         = _get_price(dossier)

    # Backstop sanity check: composite FV > 5× price or < 4% of price likely reflects
    # a currency or unit error. With FX conversion now active, genuine hits are rare
    # (deep-value microcaps, early-stage, etc.) so preserve the value but flag it.
    _backstop_flags = result.get("blind_spots") or []
    if composite is not None and price is not None and price > 0:
        _ratio = composite / price
        if _ratio > 5.0 or _ratio < 0.04:
            _backstop_flags = list(_backstop_flags) + [
                f"composite_fv_extreme_ratio:{_ratio:.1f}x — verify for currency/unit errors"
            ]

    primary_fv = _safe(primary.get("fair_value"))
    # Compute margin of safety from composite if available, otherwise fall back to
    # primary alone — better than returning None when composite is unavailable.
    margin_of_safety = None
    _mos_base = composite if composite is not None else primary_fv
    if _mos_base is not None and price is not None and _mos_base != 0:
        margin_of_safety = (_mos_base - price) / _mos_base

    secondary_methods = [
        {"method": s.get("method"), "fair_value": _safe(s.get("fair_value"))}
        for s in secondary_raw
        if s
    ]

    return {
        "archetype": archetype_info,
        "primary_method": primary.get("method"),
        "primary_fair_value": primary_fv,
        "secondary_methods": secondary_methods,
        "composite_fair_value": composite,
        "margin_of_safety": margin_of_safety,
        "archetype_metrics": result.get("key_metrics") or {},
        "blind_spot_flags": _backstop_flags,
        "invalid_methods": result.get("invalid") or [],
    }


# ── Private archetype engines ─────────────────────────────────────────────────

def _value_asset_light(dossier: dict) -> dict:
    """Archetype 1 — Asset-Light SaaS / Digital Platforms."""
    ratios  = _get_ratios(dossier)
    income0 = _get_income(dossier, 0)
    income1 = _get_income(dossier, 1)
    cf0     = _get_cashflow(dossier, 0)
    balance = _get_balance(dossier, 0)

    price       = _get_price(dossier)
    shares_out  = _safe(ratios.get("shares_out"))
    revenue_ttm = _safe(ratios.get("revenue_ttm"))
    gross_margin= _norm_margin(ratios.get("gross_margin"))
    # Prefer cashflow statement FCF so FCF and SBC are from the same reporting period.
    # ratios_ttm FCF is yfinance TTM and may be depressed by a recent CapEx ramp while
    # cashflow SBC is from the latest annual — mixing them distorts sbc_adjusted_fcf.
    fcf         = _safe(cf0.get("free_cash_flow")) or _safe(ratios.get("fcf"))
    sbc         = _safe(cf0.get("stock_based_compensation"))
    total_debt  = _safe(balance.get("total_debt"), 0)
    cash        = _safe(balance.get("cash"), 0)

    # Revenue growth YoY
    rev0 = _safe(income0.get("revenue"))
    rev1 = _safe(income1.get("revenue"))
    revenue_growth_yoy = None
    if rev0 is not None and rev1 is not None and rev1 > 0:
        revenue_growth_yoy = (rev0 - rev1) / rev1

    # SBC metrics
    sbc_pct_revenue = None
    if sbc is not None and revenue_ttm is not None and revenue_ttm > 0:
        sbc_pct_revenue = sbc / revenue_ttm

    sbc_adjusted_fcf = fcf
    if fcf is not None and sbc is not None:
        sbc_adjusted_fcf = fcf - sbc

    sbc_adjusted_fcf_margin = None
    if sbc_adjusted_fcf is not None and revenue_ttm is not None and revenue_ttm > 0:
        sbc_adjusted_fcf_margin = sbc_adjusted_fcf / revenue_ttm

    # Rule of 40 — require BOTH legs. Estimating a missing FCF margin as 0 is
    # optimistic for money-losing hypergrowth (true FCF can be deeply negative),
    # so refuse partial credit and let target_multiple fall through to the floor.
    rule_of_40 = None
    rev_growth_pct = (revenue_growth_yoy * 100) if revenue_growth_yoy is not None else None
    fcf_margin_pct = (sbc_adjusted_fcf_margin * 100) if sbc_adjusted_fcf_margin is not None else None
    if rev_growth_pct is not None and fcf_margin_pct is not None:
        rule_of_40 = rev_growth_pct + fcf_margin_pct

    # R&D yield: latest gross_profit_growth / prior year R&D spend
    rd_yield = None
    gp0  = _safe(income0.get("gross_profit"))
    gp1  = _safe(income1.get("gross_profit"))
    rd1  = _safe(income1.get("research_development"))
    if gp0 is not None and gp1 is not None and rd1 is not None and rd1 > 0:
        gross_profit_growth = gp0 - gp1
        rd_yield = gross_profit_growth / rd1

    # ── Hypergrowth override (fwd revenue growth >50% + gross margin >60%) ───
    # Uses EV/NTM Revenue multiples calibrated to 2024-2026 AI cycle comps.
    # Negative FCF companies use 100% NTM Revenue; positive FCF blends 70/30.
    fwd_rev_growth = _safe(ratios.get("fwd_revenue_growth"))
    hypergrowth_fv = None
    hypergrowth_active = False
    if fwd_rev_growth is not None and fwd_rev_growth > 0.50 and gross_margin is not None and gross_margin > 0.60:
        hypergrowth_active = True
        ntm_multiple = 40 if fwd_rev_growth > 1.0 else 25
        if revenue_ttm is not None and revenue_ttm > 0 and shares_out is not None and shares_out > 0:
            ntm_rev = revenue_ttm * (1 + fwd_rev_growth)
            net_debt_hg = (total_debt or 0) - (cash or 0)
            ev_ntm = ntm_multiple * ntm_rev
            hypergrowth_fv = (ev_ntm - net_debt_hg) / shares_out

    # ── Primary: EV/FCF (SBC-adjusted) ───────────────────────────────────────
    if rule_of_40 is not None and rule_of_40 >= 40:
        target_multiple = 25
    elif rule_of_40 is not None and rule_of_40 >= 20:
        target_multiple = 18
    else:
        target_multiple = 12

    primary_fv = None
    net_debt_al = (total_debt or 0) - (cash or 0)
    primary_assumptions = {
        "method": "EV/FCF (SBC-adjusted)",
        "target_multiple": target_multiple,
        "rule_of_40": rule_of_40,
        "sbc_adjusted_fcf": sbc_adjusted_fcf,
        "net_debt": net_debt_al,
    }
    if sbc_adjusted_fcf is not None and shares_out is not None and shares_out > 0:
        # FCF × multiple gives enterprise value; subtract net debt to get equity value.
        # A negative net_debt (net cash position) correctly inflates the equity value.
        equity_value = sbc_adjusted_fcf * target_multiple - net_debt_al
        if equity_value > 0:
            primary_fv = equity_value / shares_out

    primary = {
        "method": "EV/FCF (SBC-adjusted)",
        "fair_value": primary_fv,
        "assumptions": primary_assumptions,
    }

    # ── Secondary: Price/Sales ────────────────────────────────────────────────
    if gross_margin is not None and gross_margin > 0.70:
        ps_multiple = 8
    elif gross_margin is not None and gross_margin > 0.60:
        ps_multiple = 5
    else:
        ps_multiple = 3

    secondary_fv = None
    if revenue_ttm is not None and revenue_ttm > 0 and shares_out is not None and shares_out > 0:
        secondary_fv = (ps_multiple * revenue_ttm) / shares_out

    secondary = [{
        "method": "Price/Sales",
        "fair_value": secondary_fv,
        "assumptions": {
            "ps_multiple": ps_multiple,
            "gross_margin": gross_margin,
            "revenue_ttm": revenue_ttm,
        },
    }]

    # ── Composite ─────────────────────────────────────────────────────────────
    if hypergrowth_active and hypergrowth_fv is not None:
        # Hypergrowth: NTM Revenue method dominates.
        # If FCF is negative or unavailable, use 100% NTM Revenue (FCF multiple is misleading).
        # If FCF is positive, blend 70% NTM Revenue + 30% FCF-based primary.
        fcf_positive = sbc_adjusted_fcf is not None and sbc_adjusted_fcf > 0
        if fcf_positive and primary_fv is not None:
            composite = hypergrowth_fv * 0.70 + primary_fv * 0.30
        else:
            composite = hypergrowth_fv
    elif primary_fv is not None and secondary_fv is not None:
        composite = primary_fv * 0.60 + secondary_fv * 0.40
    elif primary_fv is not None:
        composite = primary_fv
    elif secondary_fv is not None:
        composite = secondary_fv
    else:
        composite = None

    # ── Blind spots ───────────────────────────────────────────────────────────
    blind_spots = ["NRR_NOT_COMPUTED"]
    if sbc_pct_revenue is not None and sbc_pct_revenue > 0.15:
        blind_spots.append("SBC_DILUTION")

    return {
        "primary": primary,
        "secondary": secondary,
        "composite": composite,
        "key_metrics": {
            "sbc_pct_revenue": sbc_pct_revenue,
            "sbc_adjusted_fcf": sbc_adjusted_fcf,
            "sbc_adjusted_fcf_margin": sbc_adjusted_fcf_margin,
            "rule_of_40": rule_of_40,
            "rd_yield": rd_yield,
            "revenue_growth_yoy": revenue_growth_yoy,
            "gross_margin": gross_margin,
            "hypergrowth_active": hypergrowth_active,
            "hypergrowth_fv": hypergrowth_fv,
            "fwd_rev_growth": fwd_rev_growth,
        },
        "blind_spots": blind_spots,
        "invalid": ["STANDARD_DCF_UNRELIABLE", "EV_FCF_BEFORE_SBC_MISLEADING"],
    }


def _value_cyclical(dossier: dict) -> dict:
    """Archetype 2 — Capital-Intensive Cyclicals."""
    ratios  = _get_ratios(dossier)
    balance = _get_balance(dossier, 0)
    cf0     = _get_cashflow(dossier, 0)
    profile = _get_profile(dossier)

    price         = _get_price(dossier)
    shares_out    = _safe(ratios.get("shares_out"))
    revenue_ttm   = _safe(ratios.get("revenue_ttm"))
    current_pe    = _safe(ratios.get("pe"))
    sector        = _safe(profile.get("sector"), "")
    industry      = _safe(profile.get("industry"), "")

    total_assets       = _safe(balance.get("total_assets"))
    current_liabilities= _safe(balance.get("current_liabilities"))
    total_debt         = _safe(balance.get("total_debt"), 0)
    cash               = _safe(balance.get("cash"), 0)
    inventory          = _safe(balance.get("inventory"))
    capex              = _safe(cf0.get("capex"))

    # Normalized earnings: average net_income over up to 4 years
    net_incomes = []
    for i in range(4):
        inc = _get_income(dossier, i)
        ni  = _safe(inc.get("net_income"))
        if ni is not None:
            net_incomes.append(ni)
    normalized_earnings = (sum(net_incomes) / len(net_incomes)) if net_incomes else None

    # Cost of revenue for latest year (for DIO)
    inc0   = _get_income(dossier, 0)
    cogs   = _safe(inc0.get("cost_of_revenue"))

    # Normalized P/E
    normalized_eps = None
    if normalized_earnings is not None and shares_out is not None and shares_out > 0:
        normalized_eps = normalized_earnings / shares_out

    normalized_pe = None
    if normalized_eps is not None and price is not None and normalized_eps > 0:
        normalized_pe = price / normalized_eps

    # Peak earnings trap
    peak_earnings_trap = False
    if current_pe is not None and normalized_pe is not None and normalized_pe > 0:
        if current_pe < normalized_pe * 0.5:
            peak_earnings_trap = True

    # Days Inventory Outstanding
    dio = None
    if inventory is not None and cogs is not None and cogs > 0:
        dio = inventory / (cogs / 365)

    # Capex intensity
    capex_intensity = None
    if capex is not None and revenue_ttm is not None and revenue_ttm > 0:
        capex_intensity = abs(capex) / revenue_ttm

    # EV / Invested Capital. Invested capital = equity + interest-bearing debt
    # (matches dossier._compute_roic). The prior `total_assets - current_liabilities`
    # wrongly counted long-term debt on the asset side, inflating invested capital.
    ev_ic = None
    invested_capital = None
    stockholders_equity_ic = _safe(balance.get("stockholders_equity"))
    if stockholders_equity_ic is not None:
        invested_capital = stockholders_equity_ic + (total_debt or 0)
    market_cap = (price * shares_out) if (price is not None and shares_out is not None) else None
    ev = None
    if market_cap is not None:
        ev = market_cap + (total_debt or 0) - (cash or 0)
    if ev is not None and invested_capital is not None and invested_capital > 0:
        ev_ic = ev / invested_capital

    # ── Primary: Normalized mid-cycle P/E ────────────────────────────────────
    if "Semiconductor" in industry:
        mid_cycle_pe_target = 18
    elif sector == "Industrials":
        mid_cycle_pe_target = 16
    else:
        mid_cycle_pe_target = 14  # Energy / Materials

    primary_fv = None
    if normalized_eps is not None:
        primary_fv = normalized_eps * mid_cycle_pe_target

    primary = {
        "method": "Normalized Mid-Cycle P/E",
        "fair_value": primary_fv,
        "assumptions": {
            "normalized_earnings": normalized_earnings,
            "normalized_eps": normalized_eps,
            "mid_cycle_pe_target": mid_cycle_pe_target,
            "years_averaged": len(net_incomes),
        },
    }

    # ── Secondary: EV/IC ──────────────────────────────────────────────────────
    target_ev_ic_multiple = 1.5
    secondary_fv = None
    if invested_capital is not None and shares_out is not None and shares_out > 0:
        secondary_fv = (invested_capital * target_ev_ic_multiple) / shares_out

    secondary = [{
        "method": "EV/Invested Capital",
        "fair_value": secondary_fv,
        "assumptions": {
            "invested_capital": invested_capital,
            "target_ev_ic_multiple": target_ev_ic_multiple,
        },
    }]

    # ── Composite ─────────────────────────────────────────────────────────────
    if primary_fv is not None and secondary_fv is not None:
        composite = primary_fv * 0.65 + secondary_fv * 0.35
    elif primary_fv is not None:
        composite = primary_fv
    elif secondary_fv is not None:
        composite = secondary_fv
    else:
        composite = None

    # ── Blind spots ───────────────────────────────────────────────────────────
    blind_spots = ["BOOK_TO_BILL_NOT_COMPUTED"]
    if peak_earnings_trap:
        blind_spots.append("PEAK_EARNINGS_TRAP")
    if len(net_incomes) == 1:
        blind_spots.append("SINGLE_YEAR_NORMALIZATION — only 1 earnings year; mid-cycle P/E may reflect peak or trough")

    return {
        "primary": primary,
        "secondary": secondary,
        "composite": composite,
        "key_metrics": {
            "normalized_earnings": normalized_earnings,
            "normalized_pe": normalized_pe,
            "current_pe": current_pe,
            "peak_earnings_trap": peak_earnings_trap,
            "dio": dio,
            "capex_intensity": capex_intensity,
            "ev_ic": ev_ic,
        },
        "blind_spots": blind_spots,
        "invalid": ["STANDARD_DCF_UNRELIABLE_AT_CYCLE_EXTREMES"],
    }


def _value_financial(dossier: dict) -> dict:
    """Archetype 3 — Financial Institutions."""
    ratios  = _get_ratios(dossier)
    balance = _get_balance(dossier, 0)
    macro   = _get_macro(dossier)

    price          = _get_price(dossier)
    shares_out     = _safe(ratios.get("shares_out"))
    roe            = _safe(ratios.get("roe"))
    yield_curve    = _safe(macro.get("yield_curve_spread"))

    stockholders_equity = _safe(balance.get("stockholders_equity"), 0)
    goodwill            = _safe(balance.get("goodwill"), 0)
    intangible_assets   = _safe(balance.get("intangible_assets"), 0)

    # Tangible book value
    tangible_book = stockholders_equity - (goodwill or 0) - (intangible_assets or 0)

    tangible_book_per_share = None
    if shares_out is not None and shares_out > 0 and tangible_book is not None:
        tangible_book_per_share = tangible_book / shares_out

    ptbv = None
    if tangible_book_per_share is not None and tangible_book_per_share > 0 and price is not None:
        ptbv = price / tangible_book_per_share

    # Fair P/TBV target based on ROE
    # ROE arrives as a percentage (e.g. 12.5 == 12.5%) — see dossier.py _pct().
    fair_ptbv_target = 0.6  # default / low ROE
    if roe is not None:
        if roe >= 15:
            fair_ptbv_target = 1.8
        elif roe >= 12:
            fair_ptbv_target = 1.3
        elif roe >= 8:
            fair_ptbv_target = 0.9
        else:
            fair_ptbv_target = 0.6

    # ── Primary: P/TBV ───────────────────────────────────────────────────────
    primary_fv = None
    if tangible_book_per_share is not None:
        primary_fv = tangible_book_per_share * fair_ptbv_target

    primary = {
        "method": "P/TBV",
        "fair_value": primary_fv,
        "assumptions": {
            "tangible_book_per_share": tangible_book_per_share,
            "fair_ptbv_target": fair_ptbv_target,
            "roe": roe,
        },
    }

    # ── Blind spots ───────────────────────────────────────────────────────────
    blind_spots = ["DDM_REQUIRES_DIVIDEND_DATA", "DURATION_MISMATCH_CHECK_YIELD_CURVE"]
    if yield_curve is not None and yield_curve < 0:
        blind_spots.append("RATE_SENSITIVE")

    return {
        "primary": primary,
        "secondary": [],
        "composite": primary_fv,
        "key_metrics": {
            "tangible_book_per_share": tangible_book_per_share,
            "ptbv": ptbv,
            "roe": roe,
            "fair_ptbv_target": fair_ptbv_target,
            "dividend_yield": None,
        },
        "blind_spots": blind_spots,
        "invalid": ["EV_FCF_INVALID_FOR_FINANCIALS", "STANDARD_DCF_INVALID_FOR_FINANCIALS", "DDM_INVALID_NO_DIVIDEND_DATA"],
    }


def _value_infrastructure(dossier: dict) -> dict:
    """Archetype 4 — Asset-Heavy Infrastructure (REITs, Utilities)."""
    ratios  = _get_ratios(dossier)
    balance = _get_balance(dossier, 0)
    cf0     = _get_cashflow(dossier, 0)
    profile = _get_profile(dossier)

    price       = _get_price(dossier)
    shares_out  = _safe(ratios.get("shares_out"))
    ev_ebitda   = _safe(ratios.get("ev_ebitda"))
    ebitda      = _safe(ratios.get("ebitda"))
    total_debt  = _safe(balance.get("total_debt"), 0)
    cash        = _safe(balance.get("cash"), 0)
    industry    = _safe(profile.get("industry"), "")

    operating_cf = _safe(cf0.get("operating_cf"))
    capex        = _safe(cf0.get("capex"))

    # AFFO estimate: operating_cf - maintenance capex.
    # Maintenance fraction varies by sub-type: telecoms/cable spend most capex on
    # ongoing network maintenance; REITs spend far less (tenants handle maintenance).
    affo_estimate = None
    if operating_cf is not None:
        if "Telecom" in industry or "Telecom Services" in industry or "Cable" in industry:
            maintenance_ratio = 0.60
        elif "Utility" in industry or "Utilities" in industry:
            maintenance_ratio = 0.45
        elif "REIT" in industry:
            maintenance_ratio = 0.20
        else:
            maintenance_ratio = 0.25
        maintenance_capex = (abs(capex) * maintenance_ratio) if capex is not None else 0
        affo_estimate = operating_cf - maintenance_capex

    affo_per_share = None
    if affo_estimate is not None and shares_out is not None and shares_out > 0:
        affo_per_share = affo_estimate / shares_out

    affo_yield = None
    market_cap = (price * shares_out) if (price is not None and shares_out is not None) else None
    if affo_estimate is not None and market_cap is not None and market_cap > 0:
        affo_yield = affo_estimate / market_cap

    # EV/EBITDA check
    ev_ebitda_expensive = (ev_ebitda is not None and ev_ebitda > 18)

    # Net debt / EBITDA
    net_debt_ebitda = None
    if total_debt is not None and ebitda is not None and ebitda > 0:
        net_debt_ebitda = total_debt / ebitda

    # Detect sub-type target multiple
    if "REIT" in industry:
        target_affo_multiple = 18
    elif "Utility" in industry or "Utilities" in industry:
        target_affo_multiple = 15
    elif "Telecom" in industry:
        target_affo_multiple = 12
    else:
        target_affo_multiple = 16

    # ── Primary: P/AFFO ───────────────────────────────────────────────────────
    primary_fv = None
    if affo_per_share is not None:
        primary_fv = affo_per_share * target_affo_multiple

    primary = {
        "method": "P/AFFO",
        "fair_value": primary_fv,
        "assumptions": {
            "affo_estimate": affo_estimate,
            "affo_per_share": affo_per_share,
            "target_affo_multiple": target_affo_multiple,
            "industry": industry,
        },
    }

    # ── Secondary: EV/(EBITDA - CapEx) ───────────────────────────────────────
    secondary_fv = None
    net_debt_infra = (total_debt or 0) - (cash or 0)
    secondary_assumptions: dict = {"ev_ebitda": ev_ebitda, "ebitda": ebitda, "capex": capex, "net_debt": net_debt_infra}
    if ebitda is not None and capex is not None and shares_out is not None and shares_out > 0:
        ebitda_minus_capex = ebitda - abs(capex)
        if ebitda_minus_capex > 0:
            # EV = (EBITDA - CapEx) × EV multiple; subtract net debt -> equity value.
            # This is an ENTERPRISE multiple and must NOT reuse the equity P/AFFO
            # multiple (the prior code aliased them, mixing equity and EV bases).
            # Values are approximate — calibrate against sector comps (TODO). Also note
            # EBITDA is TTM while CapEx is the latest ANNUAL figure (period mismatch).
            if "REIT" in industry:
                target_ev_multiple = 20
            elif "Utility" in industry or "Utilities" in industry:
                target_ev_multiple = 16
            elif "Telecom" in industry:
                target_ev_multiple = 13
            else:
                target_ev_multiple = 16
            equity_value = ebitda_minus_capex * target_ev_multiple - net_debt_infra
            if equity_value > 0:
                secondary_fv = equity_value / shares_out
            secondary_assumptions["ebitda_minus_capex"] = ebitda_minus_capex
            secondary_assumptions["target_ev_multiple"] = target_ev_multiple
            secondary_assumptions["period_mismatch"] = "EBITDA=TTM, CapEx=annual"

    secondary = [{
        "method": "EV/(EBITDA-CapEx)",
        "fair_value": secondary_fv,
        "assumptions": secondary_assumptions,
    }]

    # ── Composite ─────────────────────────────────────────────────────────────
    if primary_fv is not None and secondary_fv is not None:
        composite = primary_fv * 0.70 + secondary_fv * 0.30
    elif primary_fv is not None:
        composite = primary_fv
    elif secondary_fv is not None:
        composite = secondary_fv
    else:
        composite = None

    return {
        "primary": primary,
        "secondary": secondary,
        "composite": composite,
        "key_metrics": {
            "affo_estimate": affo_estimate,
            "affo_per_share": affo_per_share,
            "affo_yield": affo_yield,
            "ev_ebitda": ev_ebitda,
            "ev_ebitda_expensive": ev_ebitda_expensive,
            "net_debt_ebitda": net_debt_ebitda,
            "target_affo_multiple": target_affo_multiple,
        },
        "blind_spots": ["RATE_SENSITIVE", "AFFO_IS_ESTIMATED"],
        "invalid": ["STANDARD_DCF_UNRELIABLE_DUE_TO_LEVERAGE"],
    }


def _value_early_stage(dossier: dict) -> dict:
    """Archetype 5 — Early-Stage / Pre-Profit."""
    ratios  = _get_ratios(dossier)
    balance = _get_balance(dossier, 0)
    cf0     = _get_cashflow(dossier, 0)
    profile = _get_profile(dossier)
    income0 = _get_income(dossier, 0)

    price       = _get_price(dossier)
    shares_out  = _safe(ratios.get("shares_out"))
    revenue_ttm = _safe(ratios.get("revenue_ttm"))
    sector      = _safe(profile.get("sector"), "")
    market_cap_bn = _safe(profile.get("market_cap_bn"))

    free_cash_flow = _safe(cf0.get("free_cash_flow")) or _safe(ratios.get("fcf"))
    cash           = _safe(balance.get("cash"), 0)
    total_debt     = _safe(balance.get("total_debt"), 0)

    # Monthly burn
    monthly_burn = None
    if free_cash_flow is not None and free_cash_flow < 0:
        monthly_burn = abs(free_cash_flow) / 12

    # Cash runway months
    cash_runway_months = None
    if cash is not None and monthly_burn is not None and monthly_burn > 0:
        cash_runway_months = cash / monthly_burn

    # Revenue run rate: approximate from annual income
    revenue_run_rate = _safe(income0.get("revenue"))

    # EV/Revenue
    ev_revenue = None
    if market_cap_bn is not None and revenue_ttm is not None and revenue_ttm > 0:
        ev_num = market_cap_bn * 1e9 + (total_debt or 0) - (cash or 0)
        ev_revenue = ev_num / revenue_ttm

    # Runway signal — distinguish "not burning" from "truly critical"
    if free_cash_flow is None:
        runway_signal = "DATA_UNAVAILABLE"
    elif free_cash_flow >= 0:
        runway_signal = "NOT_BURNING"
    elif cash_runway_months is None:
        runway_signal = "CRITICAL"
    elif cash_runway_months >= 18:
        runway_signal = "ADEQUATE"
    elif cash_runway_months >= 6:
        runway_signal = "WARNING"
    else:
        runway_signal = "CRITICAL"

    # ── Primary: cash runway signal (not a price target) ─────────────────────
    primary = {
        "method": "Cash Runway Signal",
        "fair_value": None,
        "assumptions": {
            "monthly_burn": monthly_burn,
            "cash_runway_months": cash_runway_months,
            "runway_signal": runway_signal,
            "cash": cash,
        },
    }

    # ── Secondary: EV/Revenue ─────────────────────────────────────────────────
    if sector == "Technology":
        target_ev_rev = 8
    elif sector == "Healthcare":
        target_ev_rev = 6
    else:
        target_ev_rev = 4

    secondary_fv = None
    if revenue_ttm is not None and revenue_ttm > 0 and shares_out is not None and shares_out > 0:
        secondary_fv = (target_ev_rev * revenue_ttm) / shares_out

    secondary = [{
        "method": "EV/Revenue",
        "fair_value": secondary_fv,
        "assumptions": {
            "target_ev_rev": target_ev_rev,
            "revenue_ttm": revenue_ttm,
            "ev_revenue_current": ev_revenue,
            "sector": sector,
        },
    }]

    # ── Composite: secondary only ─────────────────────────────────────────────
    composite = secondary_fv

    # ── Blind spots ───────────────────────────────────────────────────────────
    blind_spots = ["DCF_AND_PE_MEANINGLESS"]
    if cash_runway_months is None:
        blind_spots.append("DILUTION_RISK_UNKNOWN")
    elif cash_runway_months < 18:
        blind_spots.append("DILUTION_RISK")

    return {
        "primary": primary,
        "secondary": secondary,
        "composite": composite,
        "key_metrics": {
            "monthly_burn": monthly_burn,
            "cash_runway_months": cash_runway_months,
            "runway_signal": runway_signal,
            "revenue_run_rate": revenue_run_rate,
            "ev_revenue": ev_revenue,
        },
        "blind_spots": blind_spots,
        "invalid": ["STANDARD_DCF_MEANINGLESS", "PE_RATIO_MEANINGLESS"],
    }


def _value_mature_compounder(dossier: dict) -> dict:
    """Archetype 6 — Mature Compounders."""
    ratios   = _get_ratios(dossier)
    balance  = _get_balance(dossier, 0)
    cf0      = _get_cashflow(dossier, 0)
    profile  = _get_profile(dossier)
    valuation= dossier.get("valuation") or {}

    shares_out  = _safe(ratios.get("shares_out"))
    roic        = _safe(ratios.get("roic"))
    sector      = _safe(profile.get("sector"), "") or _safe(profile.get("yf_sector"), "")
    fcf         = _safe(ratios.get("fcf")) or _safe(cf0.get("free_cash_flow"))

    # Latest net income from income[0]
    inc0       = _get_income(dossier, 0)
    net_income = _safe(inc0.get("net_income"))

    # FCF conversion
    fcf_conversion = None
    if fcf is not None and net_income is not None and net_income > 0:
        fcf_conversion = fcf / net_income

    # ROIC - WACC spread (assume 8% WACC). roic is a percentage (e.g. 20.0),
    # so the spread is in percentage points.
    roic_wacc_spread = None
    if roic is not None:
        roic_wacc_spread = roic - 8

    # Share count trend: not available from dossier
    share_count_trend = None

    # Organic growth (revenue CAGR)
    income_list = []
    try:
        income_list = dossier["financials"]["income"] or []
    except (KeyError, TypeError):
        pass

    organic_growth = None
    if len(income_list) >= 2:
        rev_latest = _safe(income_list[0].get("revenue") if income_list[0] else None)
        rev_oldest = _safe(income_list[-1].get("revenue") if income_list[-1] else None)
        years = len(income_list) - 1
        organic_growth = _cagr(rev_oldest, rev_latest, years)

    # ── Primary: DCF from existing valuation ─────────────────────────────────
    dcf_iv = _safe(valuation.get("dcf_iv_per_share"))
    primary = {
        "method": "DCF (existing)",
        "fair_value": dcf_iv,
        "assumptions": valuation.get("dcf_assumptions") or {},
    }

    # ── Secondary: EV/FCF yield ───────────────────────────────────────────────
    # Sector-based FCF multiple — Technology/Healthcare command premium multiples;
    # Energy/Utilities/Financials trade at lower FCF yields.
    ev_fcf_multiple = _MATURE_FCF_MULTIPLE.get(sector, _DEFAULT_MATURE_FCF_MULTIPLE)
    total_debt_mc   = _safe(balance.get("total_debt"), 0)
    cash_mc         = _safe(balance.get("cash"), 0)
    net_debt_mc     = (total_debt_mc or 0) - (cash_mc or 0)

    secondary_fv = None
    if fcf is not None and shares_out is not None and shares_out > 0:
        # FCF × multiple = enterprise value; subtract net debt for equity value per share.
        equity_value_mc = fcf * ev_fcf_multiple - net_debt_mc
        if equity_value_mc > 0:
            secondary_fv = equity_value_mc / shares_out

    secondary = [{
        "method": f"EV/FCF ({ev_fcf_multiple}x, {sector or 'default'})",
        "fair_value": secondary_fv,
        "assumptions": {
            "fcf": fcf,
            "ev_fcf_multiple": ev_fcf_multiple,
            "sector": sector,
            "net_debt": net_debt_mc,
        },
    }]

    # ── Composite ─────────────────────────────────────────────────────────────
    if dcf_iv is not None and secondary_fv is not None:
        composite = dcf_iv * 0.55 + secondary_fv * 0.45
    elif dcf_iv is not None:
        composite = dcf_iv
    elif secondary_fv is not None:
        composite = secondary_fv
    else:
        composite = None

    # ── Blind spots ───────────────────────────────────────────────────────────
    blind_spots = ["SHARE_COUNT_HISTORY_NOT_AVAILABLE"]
    if roic is not None and roic < 0:
        blind_spots.append("NEGATIVE_ROIC — value destruction; MATURE_COMPOUNDER archetype may not apply")
    if organic_growth is not None and roic is not None:
        # organic_growth is a fraction (0.03 == 3%); roic is a percentage (15 == 15%).
        if organic_growth < 0.03 and roic > 15:
            blind_spots.append("FINANCIAL_ENGINEERING")

    return {
        "primary": primary,
        "secondary": secondary,
        "composite": composite,
        "key_metrics": {
            "fcf_conversion": fcf_conversion,
            "roic_wacc_spread": roic_wacc_spread,
            "share_count_trend": share_count_trend,
            "organic_growth": organic_growth,
            "roic": roic,
            "dcf_iv_per_share": dcf_iv,
        },
        "blind_spots": blind_spots,
        "invalid": [],
    }
