"""
pillar_scoring.py — 5-pillar pre-debate quality scorer from Finviz fundament data.

Each pillar returns a float score in [1.0, 10.0]. compute_composite() aggregates
them into a weighted composite for candidate ranking before the LLM debate stage.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHOKEPOINT_INDUSTRIES: dict[str, int] = {
    # Score 9-10: Irreplaceable infrastructure
    "semiconductor equipment": 9,
    "semiconductor manufacturing": 9,
    "electronic manufacturing services": 9,
    "specialty chemicals": 8,
    "medical devices": 8,
    "diagnostic equipment": 8,
    "defense electronics": 8,
    "aerospace components": 8,
    # Score 7-8: High switching costs / network effects
    "enterprise software": 8,
    "application software": 7,
    "data & analytics": 8,
    "payment processing": 9,
    "financial data services": 8,
    "cloud infrastructure": 9,
    "cybersecurity": 7,
    "industrial automation": 7,
    "testing & measurement": 8,
    "scientific instruments": 7,
    # Score 6-7: Moderate competitive advantage
    "healthcare services": 6,
    "managed care": 6,
    "insurance": 6,
    "specialty retail": 5,
    "home improvement retail": 6,
    "waste management": 7,
    "water utilities": 7,
    "distribution": 6,
    "trucking": 5,
    "railroad": 8,
    "airport services": 7,
    # Score 3-5: Commodity / low moat
    "oil & gas": 4,
    "steel": 3,
    "aluminum": 3,
    "mining": 3,
    "commodity chemicals": 3,
    "apparel": 4,
    "restaurants": 5,
    "hotels": 4,
    "airlines": 3,
    "auto manufacturers": 4,
    "homebuilders": 4,
}

# ---------------------------------------------------------------------------
# Private parsing helpers
# ---------------------------------------------------------------------------

def _pct(val: str | None) -> float | None:
    if val is None or str(val).strip() in ("", "-"):
        return None
    s = str(val).strip().replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _split_3y_5y(val: str | None) -> tuple[float | None, float | None]:
    if val is None or str(val).strip() in ("", "-"):
        return (None, None)
    parts = str(val).strip().split()
    if len(parts) >= 2:
        three_y = _pct(parts[0])
        five_y = _pct(parts[1])
        return (three_y, five_y)
    elif len(parts) == 1:
        return (None, _pct(parts[0]))
    return (None, None)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _float(v: str | None) -> float | None:
    if v is None or str(v).strip() in ("", "-"):
        return None
    try:
        return float(str(v).strip())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Pillar scorers
# ---------------------------------------------------------------------------

def score_financial_physics(fundament: dict) -> float:
    """Measure capital efficiency and profitability depth.

    Uses ROIC, Gross Margin, Oper. Margin, Profit Margin.
    Returns a score in [1.0, 10.0].
    """
    roic = _pct(fundament.get("ROIC"))
    gross_margin = _pct(fundament.get("Gross Margin"))
    oper_margin = _pct(fundament.get("Oper. Margin"))
    profit_margin = _pct(fundament.get("Profit Margin"))

    if all(v is None for v in (roic, gross_margin, oper_margin, profit_margin)):
        return 5.0

    pts = 0.0

    # ROIC
    if roic is not None:
        if roic >= 30:
            pts += 3
        elif roic >= 15:
            pts += 2
        elif roic >= 8:
            pts += 1

    # Gross Margin
    if gross_margin is not None:
        if gross_margin >= 60:
            pts += 2
        elif gross_margin >= 40:
            pts += 1

    # Oper. Margin
    if oper_margin is not None:
        if oper_margin >= 25:
            pts += 2
        elif oper_margin >= 15:
            pts += 1

    # Profit Margin
    if profit_margin is not None:
        if profit_margin >= 20:
            pts += 2
        elif profit_margin >= 10:
            pts += 1

    score = max(1.0, (pts / 9) * 10)
    return _clamp(score, 1.0, 10.0)


def score_moat_proxy(fundament: dict) -> float:
    """Measure durable advantage signals.

    Uses Gross Margin, ROIC, Debt/Eq, Recom, P/E vs Forward P/E.
    Returns a score in [1.0, 10.0].
    """
    gross_margin = _pct(fundament.get("Gross Margin"))
    roic = _pct(fundament.get("ROIC"))
    recom_raw = fundament.get("Recom")
    pe_raw = fundament.get("P/E")
    fwd_pe_raw = fundament.get("Forward P/E")

    # Parse Debt/Eq and Recom as plain floats (not percentages)
    debt_eq = _float(fundament.get("Debt/Eq"))
    recom = _float(recom_raw)
    pe = _float(pe_raw)
    fwd_pe = _float(fwd_pe_raw)

    if all(v is None for v in [gross_margin, roic, debt_eq, recom, fwd_pe, pe]):
        return 5.0

    pts = 0.0

    # Gross Margin
    if gross_margin is not None:
        if gross_margin >= 50:
            pts += 2.5
        elif gross_margin >= 35:
            pts += 1.5

    # ROIC
    if roic is not None:
        if roic >= 20:
            pts += 2.5
        elif roic >= 10:
            pts += 1.5

    # Debt/Eq
    if debt_eq is not None:
        if debt_eq < 0.5:
            pts += 1.5
        elif debt_eq < 1.5:
            pts += 1.0

    # Recom (analyst consensus; lower = stronger buy)
    if recom is not None:
        if recom <= 1.5:
            pts += 1.5
        elif recom <= 2.5:
            pts += 1.0

    return _clamp(pts, 1.0, 10.0)


def score_temporal(fundament: dict) -> float:
    """Measure growth velocity and momentum.

    Uses EPS past 5Y, Sales past 5Y, EPS Q/Q, Sales Q/Q.
    Returns a score in [1.0, 10.0].
    """
    _, eps_5y = _split_3y_5y(fundament.get("EPS past 3/5Y"))
    _, sales_5y = _split_3y_5y(fundament.get("Sales past 3/5Y"))
    eps_qq = _pct(fundament.get("EPS Q/Q"))
    sales_qq = _pct(fundament.get("Sales Q/Q"))

    if all(v is None for v in [eps_5y, sales_5y, eps_qq, sales_qq]):
        return 5.0

    pts = 0.0

    # EPS 5Y CAGR
    if eps_5y is not None:
        if eps_5y >= 20:
            pts += 3
        elif eps_5y >= 10:
            pts += 2
        elif eps_5y >= 5:
            pts += 1
        if eps_5y < 0:
            pts -= 1.5

    # Sales 5Y CAGR
    if sales_5y is not None:
        if sales_5y >= 15:
            pts += 2
        elif sales_5y >= 8:
            pts += 1.5
        elif sales_5y >= 3:
            pts += 1
        if sales_5y < 0:
            pts -= 1.5

    # EPS Q/Q
    if eps_qq is not None:
        if eps_qq >= 15:
            pts += 2
        elif eps_qq >= 5:
            pts += 1.5

    # Sales Q/Q
    if sales_qq is not None:
        if sales_qq >= 10:
            pts += 2
        elif sales_qq >= 3:
            pts += 1.5

    score = max(1.0, (pts / 9) * 10)

    # Forward PEG adjustment — applied after base score to avoid distorting normalization.
    # Skipped for hypergrowth names (EPS next Y >= 50%) where PEG is unreliable.
    eps_next_y_t = _pct(fundament.get("EPS next Y"))
    fwd_pe_t = _float(fundament.get("Forward P/E"))
    fwd_peg_t = None
    if fwd_pe_t is not None and eps_next_y_t is not None and eps_next_y_t > 0:
        fwd_peg_t = fwd_pe_t / eps_next_y_t
    is_hypergrowth_t = eps_next_y_t is not None and eps_next_y_t >= 50

    if fwd_peg_t is not None and not is_hypergrowth_t:
        if fwd_peg_t < 1.0:
            score += 1.5   # market underpricing NTM growth
        elif fwd_peg_t < 2.0:
            score += 0.75  # fairly valued on growth
        elif fwd_peg_t > 3.0 and (eps_5y is None or eps_5y < 20):
            score -= 1.0   # expensive and low historical growth

    return _clamp(score, 1.0, 10.0)


def score_management(fundament: dict) -> float:
    """Measure capital allocation discipline.

    Uses Insider Trans, Inst Own, ROE, ROA.
    Returns a score in [1.0, 10.0].
    """
    insider_trans = _pct(fundament.get("Insider Trans"))
    inst_own = _pct(fundament.get("Inst Own"))
    roe = _pct(fundament.get("ROE"))
    roa = _pct(fundament.get("ROA"))

    if all(v is None for v in (insider_trans, inst_own, roe, roa)):
        return 5.0

    pts = 0.0

    # Insider Trans
    if insider_trans is not None:
        if insider_trans > 5:
            pts += 3
        elif insider_trans > 0:
            pts += 1.5
        elif insider_trans < -10:
            pts -= 1.5

    # Inst Own (30-75% = sweet spot)
    if inst_own is not None:
        if 30 <= inst_own <= 75:
            pts += 2
        elif inst_own < 20:
            pts += 1

    # ROE
    if roe is not None:
        if roe >= 20:
            pts += 2.5
        elif roe >= 12:
            pts += 1.5

    # ROA
    if roa is not None:
        if roa >= 15:
            pts += 2
        elif roa >= 8:
            pts += 1.5

    score = _clamp(1 + (pts / 9.5) * 9, 1.0, 10.0)
    return score


def score_chokepoint_proxy(fundament: dict) -> float:
    """Estimate value-chain criticality from industry classification.

    Matches against CHOKEPOINT_INDUSTRIES using case-insensitive partial matching.
    Returns a score in [1.0, 10.0].
    """
    industry = fundament.get("Industry", "") or ""
    ind_lower = industry.lower()

    best_score: int | None = None
    for key, val in CHOKEPOINT_INDUSTRIES.items():
        if key in ind_lower:
            if best_score is None or val > best_score:
                best_score = val

    score = float(best_score) if best_score is not None else 5.5
    return _clamp(score, 1.0, 10.0)


# ---------------------------------------------------------------------------
# Composite aggregator
# ---------------------------------------------------------------------------

def compute_composite(fundament: dict) -> dict:
    """Compute all 5 pillar scores and weighted composite from ticker_fundament() data.

    Returns:
        {
          "financial_physics": float,   # 1-10
          "moat_proxy": float,          # 1-10
          "temporal": float,            # 1-10
          "management": float,          # 1-10
          "chokepoint_proxy": float,    # 1-10
          "composite": float,           # 1-10 weighted
          "ticker": str | None,         # from fundament if present
        }
    """
    fp = score_financial_physics(fundament)
    mp = score_moat_proxy(fundament)
    tm = score_temporal(fundament)
    mg = score_management(fundament)
    cp = score_chokepoint_proxy(fundament)

    weights = (0.25, 0.20, 0.20, 0.15, 0.20)
    scores = (fp, mp, tm, mg, cp)
    composite = sum(s * w for s, w in zip(scores, weights))
    composite = _clamp(composite, 1.0, 10.0)

    return {
        "financial_physics": fp,
        "moat_proxy": mp,
        "temporal": tm,
        "management": mg,
        "chokepoint_proxy": cp,
        "composite": composite,
        "ticker": fundament.get("Ticker") or fundament.get("ticker"),
    }
