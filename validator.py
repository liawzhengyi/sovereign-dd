"""Data quality validator — cross-references dossier metrics across sources."""

from __future__ import annotations


def _flag_divergence(
    warnings: list[str],
    metric_name: str,
    sources: dict[str, float | None],
    threshold: float = 0.25,
) -> None:
    """Compare numeric values from multiple sources. Appends a warning if any pair diverges > threshold."""
    valid = {name: val for name, val in sources.items() if val is not None and val > 0}
    if len(valid) < 2:
        return  # need at least 2 sources to compare

    values = list(valid.values())
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            a, b = values[i], values[j]
            names = list(valid.keys())
            divergence = abs(a - b) / max(a, b)
            if divergence > threshold:
                warnings.append(
                    f"{metric_name} divergence: {names[i]}={a:.1f} vs {names[j]}={b:.1f} "
                    f"({divergence:.0%} gap — verify before trusting)"
                )


def validate_dossier(dossier: dict) -> dict:
    """Cross-reference key financial metrics across data sources.

    Returns {"warnings": [...], "data_confidence": "HIGH"|"MEDIUM"|"LOW"}

    Checks:
    1. PE cross-check: yfinance vs computed from market_cap/net_income
    2. P/S sanity: computed vs reported
    3. Foreign stock / ADR flag
    4. Extreme value sanity checks
    5. Missing forward PE warning
    6. Analyst consensus vs price (trading above consensus)
    7. EPS beat quality flag (large beats may be one-time items)
    8. Upcoming earnings staleness warning
    """
    warnings: list[str] = []

    profile = dossier.get("profile") or {}
    fin = dossier.get("financials") or {}
    ratios = fin.get("ratios_ttm") or {}
    valuation = dossier.get("valuation") or {}
    quote = dossier.get("quote") or {}

    # 1. PE CROSS-CHECK (yfinance vs first-principles)
    yf_pe = ratios.get("pe")

    mcap_bn = profile.get("market_cap_bn") or 0
    mcap = mcap_bn * 1e9
    income = fin.get("income") or []
    net_income = income[0].get("net_income") if income else None
    computed_pe: float | None = None
    if net_income and net_income > 0 and mcap > 0:
        computed_pe = round(mcap / net_income, 2)

    _flag_divergence(warnings, "Trailing PE",
                     {"yfinance": yf_pe, "computed": computed_pe},
                     threshold=0.25)

    # 2. P/S SANITY CHECK
    reported_ps = ratios.get("ps")
    revenue_ttm = ratios.get("revenue_ttm")
    if mcap > 0 and revenue_ttm and revenue_ttm > 0:
        computed_ps = round(mcap / revenue_ttm, 2)
        _flag_divergence(warnings, "P/S ratio",
                         {"reported": reported_ps, "computed": computed_ps},
                         threshold=0.30)

    # 3. FOREIGN STOCK / ADR FLAG
    country = profile.get("country") or ""
    exchange = profile.get("exchange") or ""
    adr_mismatch = (fin.get("ratios_ttm") or {}).get("adr_mismatch", False)
    if country and country.upper() not in ("US", "USA", "UNITED STATES", ""):
        if adr_mismatch:
            warnings.append(
                f"FOREIGN STOCK / ADR ({country}, exchange: {exchange}) — financial statements converted "
                f"to USD where currency differs; P/B and P/S nulled. "
                f"Forward PE was nulled if growth implied >100%. Verify all multiples independently."
            )
        else:
            warnings.append(
                f"FOREIGN STOCK ({country}, exchange: {exchange}) — ADR share structure "
                f"may distort PE/EPS ratios. Verify all valuation multiples independently."
            )

    # 4. EXTREME VALUE SANITY CHECKS
    pe = yf_pe or computed_pe
    if pe is not None:
        if pe < 0:
            warnings.append(f"Negative PE ({pe:.1f}) — company is loss-making; PE-based valuation unreliable")
        elif pe > 300:
            warnings.append(f"Extreme PE ({pe:.1f}) — likely data error or highly speculative earnings; verify")

    fwd_pe = ratios.get("fwd_pe")
    if fwd_pe is not None and (fwd_pe < 0 or fwd_pe > 200):
        warnings.append(f"Extreme forward PE ({fwd_pe:.1f}) — verify against second source before using in thesis")

    # 5. MISSING FORWARD PE
    if fwd_pe is None and pe is not None and pe > 0:
        warnings.append(
            "Forward PE unavailable — agents should look up NTM consensus EPS estimate via web research"
        )

    # 6. ANALYST CONSENSUS VS PRICE
    price = quote.get("price")
    target_mean = (valuation.get("analyst_consensus") or {}).get("target_mean")
    if price and target_mean and price > 0:
        gap = (target_mean - price) / price
        if gap < -0.10:
            warnings.append(
                f"TRADING ABOVE CONSENSUS: ${price:.2f} vs analyst target ${target_mean:.2f} "
                f"({gap:.1%}) — BUY thesis requires explicit justification why sell-side is wrong"
            )

    # 7. LARGE EPS BEAT FLAG
    surprises = dossier.get("earnings_surprises") or []
    large_beats = [s for s in surprises[:4] if s.get("beat_quality") == "LARGE_BEAT"]
    if len(large_beats) >= 2:
        warnings.append(
            f"LARGE EPS BEATS in {len(large_beats)} of last 4 quarters (>50% surprise each) — "
            "verify whether beats reflect recurring earnings power or one-time items before trusting EPS trend"
        )

    # 8. UPCOMING EARNINGS STALENESS
    try:
        from datetime import datetime, timezone
        upcoming = (dossier.get("earnings_calendar") or {}).get("upcoming") or []
        if upcoming:
            next_date_str = upcoming[0].get("date", "")
            if next_date_str:
                next_dt = datetime.strptime(next_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                days_away = (next_dt - datetime.now(timezone.utc)).days
                if 0 <= days_away <= 5:
                    warnings.append(
                        f"EARNINGS IN {days_away} DAY(S) ({next_date_str}) — dossier data may be stale; "
                        "verify current consensus EPS estimate before finalizing thesis"
                    )
    except Exception:
        pass

    # Assign data confidence
    if not warnings:
        confidence = "HIGH"
    elif len(warnings) >= 3:
        confidence = "LOW"
    else:
        confidence = "MEDIUM"

    return {"warnings": warnings, "data_confidence": confidence}
