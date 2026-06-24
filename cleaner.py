"""Batch metadata cleaner — one grounded Gemini call per scout batch.

Resolves per-ticker ground-truth metadata (GICS sector, ADR status, financials
currency) before the expensive 5-agent debates run.  This catches data-source
mismatches that no static heuristic can reliably detect — e.g. ADRs that
yfinance labels as quoteType=EQUITY, or tickers whose yfinance sector field
is empty / mis-mapped from Finnhub's non-standard industry strings.

Returned dict is keyed by upper-case ticker symbol:

  {
    "TSM":  {"canonical_sector": "Technology", "is_adr": True,  "financials_currency": "TWD"},
    "AAPL": {"canonical_sector": "Technology", "is_adr": False, "financials_currency": "USD"},
    ...
  }

Fail-safe contract: any error (API failure, bad JSON, timeout) returns {} so the
pipeline falls through to existing heuristics without interruption.  Never raises.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Accepted GICS sector strings — must match the set used by the archetype classifier
# in fair_value.py so that overrides translate directly into correct archetype routing.
_GICS_SECTORS = frozenset({
    "Technology",
    "Communication Services",
    "Healthcare",
    "Consumer Discretionary",
    "Consumer Staples",
    "Financials",
    "Financial Services",   # some yfinance datasets use this variant
    "Real Estate",
    "Utilities",
    "Energy",
    "Industrials",
    "Materials",
    "Basic Materials",      # yfinance variant for Materials
    "Consumer Cyclical",    # yfinance variant for Consumer Discretionary
    "Consumer Defensive",   # yfinance variant for Consumer Staples
})

_SYSTEM = """\
You are a financial data quality specialist.  Your job is to return accurate,
ground-truth metadata for a list of US-listed stock tickers.  Use Google Search
to verify when you are not certain.

For EVERY ticker in the list, return exactly these three fields:

  canonical_sector   — The correct GICS sector.  Must be one of:
                       Technology, Communication Services, Healthcare,
                       Consumer Discretionary, Consumer Staples, Financials,
                       Real Estate, Utilities, Energy, Industrials, Materials

  is_adr             — true  if this ticker is an American Depositary Receipt
                       representing shares of a non-US company.
                       false for all ordinary US-domiciled companies.

  financials_currency — ISO 4217 code for the currency in which the underlying
                        company files its financial statements.  "USD" for almost
                        all US companies; "TWD" for TSMC (TSM); "JPY" for Toyota
                        (TM); "HKD" for Alibaba (BABA); etc.

Return a single JSON object — ticker as key, object with the three fields as value.
No explanation, no markdown, no extra text.  Only the JSON.
"""

_USER_TEMPLATE = """\
Return the metadata JSON for these {n} tickers: {tickers}

Required output format (fill in correct values — do not copy this example verbatim):
{{
  "AAPL": {{"canonical_sector": "Technology",             "is_adr": false, "financials_currency": "USD"}},
  "TSM":  {{"canonical_sector": "Technology",             "is_adr": true,  "financials_currency": "TWD"}},
  "JPM":  {{"canonical_sector": "Financials",             "is_adr": false, "financials_currency": "USD"}},
  "T":    {{"canonical_sector": "Communication Services", "is_adr": false, "financials_currency": "USD"}}
}}
"""


async def clean_ticker_batch(tickers: list[str]) -> dict[str, dict]:
    """Return ground-truth metadata overrides for a batch of tickers.

    Makes a single grounded Gemini call for the whole batch — cheap at 1 call
    regardless of batch size (vs one call per ticker).

    Returns a dict of {TICKER: {canonical_sector, is_adr, financials_currency}}.
    Returns {} on any failure — the pipeline falls through to existing heuristics.
    """
    if not tickers:
        return {}

    try:
        from llm import call_gemini_async, extract_json
    except ImportError:
        log.warning("[cleaner] llm module unavailable — skipping metadata clean")
        return {}

    upper_tickers = [t.upper() for t in tickers]
    ticker_list   = ", ".join(upper_tickers)
    prompt        = _USER_TEMPLATE.format(n=len(upper_tickers), tickers=ticker_list)

    print(f"  [cleaner] Grounded metadata scan: {ticker_list}")
    try:
        text = await call_gemini_async(
            _SYSTEM,
            prompt,
            model="gemma-4-31b-it",  # same model as rest of pipeline; grounding does the heavy lifting
            grounding=True,
            temperature=0.0,          # deterministic — this is fact retrieval, not generation
        )
        raw = extract_json(text)
        if not isinstance(raw, dict):
            log.warning("[cleaner] Response was not a JSON object — skipping overrides")
            return {}
    except Exception as exc:
        log.warning(f"[cleaner] Gemini call failed ({exc}) — proceeding without overrides")
        return {}

    # ── Validate and normalise each entry ────────────────────────────────────
    result: dict[str, dict] = {}
    for raw_ticker, entry in raw.items():
        t = raw_ticker.upper()
        if not isinstance(entry, dict):
            continue

        cleaned: dict = {}

        sector = str(entry.get("canonical_sector", "")).strip()
        if sector in _GICS_SECTORS:
            cleaned["canonical_sector"] = sector

        if "is_adr" in entry:
            cleaned["is_adr"] = bool(entry["is_adr"])

        currency = str(entry.get("financials_currency", "")).strip().upper()
        if len(currency) == 3 and currency.isalpha():
            cleaned["financials_currency"] = currency

        if cleaned:
            result[t] = cleaned

    # ── Summary log ──────────────────────────────────────────────────────────
    adrs     = sorted(t for t, m in result.items() if m.get("is_adr"))
    non_usd  = {t: m["financials_currency"] for t, m in result.items()
                if m.get("financials_currency", "USD") != "USD"}
    print(
        f"  [cleaner] {len(result)}/{len(upper_tickers)} tickers resolved"
        + (f" | ADRs: {adrs}"    if adrs    else "")
        + (f" | non-USD: {non_usd}" if non_usd else "")
    )
    return result
