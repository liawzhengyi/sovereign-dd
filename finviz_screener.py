"""finviz_screener.py — Finviz-based stock screening using finvizfinance library.

Runs 10 specialized screens against the ~8,000 stock Finviz universe:
  - 7 Hidden Gem screens (small/mid-cap focus)
  - 3 Banger screens (any cap, quality/value/growth factors)

Entry points:
  run_finviz_screens()     → {screen_name: [tickers]}
  get_unique_candidates()  → deduplicated sorted ticker list
  enrich_candidates()      → {ticker: {finviz_fundamentals}}
"""

import time
from finvizfinance.screener.overview import Overview
from finvizfinance.quote import finvizfinance as FinvizQuote


# ── Screen definitions ─────────────────────────────────────────────────────────
#
# Screens that target Small + Mid cap are expressed as a list of filter dicts
# (one per cap size). They are run as separate API calls and merged — this
# yields more results than a single combined cap filter.
#
# All screens include "Country": "USA".

_SCREENS: list[dict] = [
    # ── Hidden Gem screens ─────────────────────────────────────────────────────

    {
        "name": "capital_light_compounder",
        "filters_list": [
            {
                "Market Cap.": "Small ($300mln to $2bln)",
                "Return on Investment": "Over +15%",
                "Operating Margin": "Over 15%",
                "Gross Margin": "Over 50%",
                "Country": "USA",
            },
            {
                "Market Cap.": "Mid ($2bln to $10bln)",
                "Return on Investment": "Over +15%",
                "Operating Margin": "Over 15%",
                "Gross Margin": "Over 50%",
                "Country": "USA",
            },
        ],
    },
    {
        "name": "sticky_revenue_machine",
        "filters_list": [
            {
                "Market Cap.": "Small ($300mln to $2bln)",
                "Gross Margin": "Over 60%",
                "EPS growthpast 5 years": "Over 15%",
                "Country": "USA",
            },
            {
                "Market Cap.": "Mid ($2bln to $10bln)",
                "Gross Margin": "Over 60%",
                "EPS growthpast 5 years": "Over 15%",
                "Country": "USA",
            },
        ],
    },
    {
        "name": "under_the_radar_growth",
        "filters_list": [
            {
                "Market Cap.": "Small ($300mln to $2bln)",
                "Sales growthpast 5 years": "Over 15%",
                "InstitutionalOwnership": "Under 60%",
                "Country": "USA",
            },
        ],
    },
    {
        "name": "moat_and_margin",
        "filters_list": [
            {
                "Market Cap.": "Small ($300mln to $2bln)",
                "Gross Margin": "Over 60%",
                "Return on Equity": "Over +15%",
                "Operating Margin": "Over 20%",
                "Country": "USA",
            },
            {
                "Market Cap.": "Mid ($2bln to $10bln)",
                "Gross Margin": "Over 60%",
                "Return on Equity": "Over +15%",
                "Operating Margin": "Over 20%",
                "Country": "USA",
            },
        ],
    },
    {
        "name": "insider_conviction",
        "filters_list": [
            {
                "Market Cap.": "Small ($300mln to $2bln)",
                "InsiderTransactions": "Positive (>0%)",
                "Return on Investment": "Over +10%",
                "Country": "USA",
            },
            {
                "Market Cap.": "Mid ($2bln to $10bln)",
                "InsiderTransactions": "Positive (>0%)",
                "Return on Investment": "Over +10%",
                "Country": "USA",
            },
        ],
    },
    {
        "name": "value_dislocation",
        "filters_list": [
            {
                "Market Cap.": "Small ($300mln to $2bln)",
                "Price/Free Cash Flow": "Under 20",
                "PEG": "Under 2",
                "Country": "USA",
            },
            {
                "Market Cap.": "Mid ($2bln to $10bln)",
                "Price/Free Cash Flow": "Under 20",
                "PEG": "Under 2",
                "Country": "USA",
            },
        ],
    },
    {
        "name": "scale_compounder",
        "filters_list": [
            {
                "Market Cap.": "Mid ($2bln to $10bln)",
                "Return on Investment": "Over +20%",
                "Net Profit Margin": "Over 10%",
                "Country": "USA",
            },
        ],
    },

    # ── Banger screens (any cap — no market cap filter) ────────────────────────

    {
        "name": "obvious_quality_fair_price",
        "filters_list": [
            {
                "Return on Investment": "Over +20%",
                "Gross Margin": "Over 50%",
                "PEG": "Under 2",
                "Country": "USA",
            },
        ],
    },
    {
        "name": "cash_flow_king",
        "filters_list": [
            {
                "Price/Free Cash Flow": "Under 25",
                "Return on Equity": "Over +20%",
                "InsiderTransactions": "Positive (>0%)",
                "Country": "USA",
            },
        ],
    },
    {
        "name": "growth_at_reasonable_price",
        "filters_list": [
            {
                "Forward P/E": "Under 30",
                "EPS growthqtr over qtr": "Over 15%",
                "Gross Margin": "Over 50%",
                "Country": "USA",
            },
        ],
    },
    {
        "name": "ai_hypergrowth_momentum",
        "filters_list": [
            {
                "EPS growthqtr over qtr": "Over 25%",
                "Gross Margin": "Over 60%",
                "Sales growthqtr over qtr": "Over 20%",
                "Country": "USA",
            },
        ],
    },
]


# ── Core helpers ───────────────────────────────────────────────────────────────

def _run_single_screen(name: str, filters: dict, verbose: bool = False) -> list[str]:
    """Run one Finviz screen with the given filters. Returns list of ticker strings."""
    screener = Overview()
    screener.set_filter(filters_dict=filters)
    df = screener.screener_view()
    if df is None or df.empty:
        return []
    return df["Ticker"].dropna().astype(str).tolist()


def run_finviz_screens(verbose: bool = False) -> dict[str, list[str]]:
    """Run all 10 Finviz screens. Returns {screen_name: [ticker, ...], ...}.

    Screens that cover multiple cap sizes are run as separate API calls and
    their results are merged before deduplication.  A 1.5-second sleep is
    inserted between every individual API call to avoid rate limiting.
    """
    results: dict[str, list[str]] = {}
    first_call = True

    for screen in _SCREENS:
        name = screen["name"]
        filters_list = screen["filters_list"]
        merged: list[str] = []
        seen_for_screen: set[str] = set()

        for filters in filters_list:
            if not first_call:
                time.sleep(1.5)
            first_call = False

            try:
                tickers = _run_single_screen(name, filters, verbose=verbose)
                for t in tickers:
                    if t not in seen_for_screen:
                        seen_for_screen.add(t)
                        merged.append(t)
            except Exception as e:
                print(f"  [screen] {name}: error — {e}")

        results[name] = merged

        if verbose:
            print(f"  [screen] {name}: {len(merged)} tickers")

    return results


# ── Public API ─────────────────────────────────────────────────────────────────

def get_unique_candidates(screen_results: dict[str, list[str]]) -> list[str]:
    """Deduplicate and sort all tickers returned across all screens.

    Args:
        screen_results: dict returned by run_finviz_screens().

    Returns:
        Alphabetically sorted list of unique ticker strings.
    """
    seen: set[str] = set()
    for tickers in screen_results.values():
        for t in tickers:
            seen.add(t)
    return sorted(seen)


def enrich_candidates(tickers: list[str]) -> dict[str, dict]:
    """Fetch 70+ Finviz fundamentals for each ticker via finvizfinance.

    Processes tickers sequentially with a 0.5-second sleep between calls.
    If a ticker lookup fails, an empty dict is stored for that ticker.

    Args:
        tickers: List of ticker strings (e.g. from get_unique_candidates()).

    Returns:
        {ticker: {metric: value, ...}, ...}
    """
    result: dict[str, dict] = {}
    for ticker in tickers:
        try:
            f = FinvizQuote(ticker)
            result[ticker] = f.ticker_fundament()
        except Exception:
            result[ticker] = {}
        time.sleep(0.5)
    return result
