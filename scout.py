"""Scout mode — quantitative screener → Gemma triage → full debate.

Every run:
  1. All 7 screener calls fire simultaneously (Yahoo Finance free API, no key needed)
  2. One grounded Gemma call picks the 12 most interesting from the combined pool
  3. Full 6-agent debate on those 12 picks — all run in parallel (max 4 concurrent)
"""

import asyncio
import json
import os
import re
import requests
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from grading import BUY_THRESHOLD  # re-exported (upload_kv does `from scout import BUY_THRESHOLD`)

# ── Continuous-mode knobs (override via env) ───────────────────────────────────
SCOUT_HISTORY_FILE   = Path("output/scout_history.json")
SCOUT_NOTIFIED_FILE  = Path("output/scout_notified.json")
SCOUT_SEEN_FILE      = Path("output/scout_seen.json")  # triage rotation ledger
SCOUT_COOLDOWN_HOURS        = int(os.getenv("SCOUT_COOLDOWN_HOURS", "48"))
SCOUT_NOTIFY_COOLDOWN_HOURS = int(os.getenv("SCOUT_NOTIFY_COOLDOWN_HOURS", "168"))  # 7 days
SCOUT_DEBATE_COUNT   = int(os.getenv("SCOUT_DEBATE_COUNT", "6"))
SCOUT_MAX_LOOPS      = int(os.getenv("SCOUT_MAX_LOOPS", "3"))
SCOUT_MIN_MCAP       = int(os.getenv("SCOUT_MIN_MCAP", "300000000"))   # universe floor
SCOUT_WINDOW_SIZE    = int(os.getenv("SCOUT_WINDOW_SIZE", "600"))      # candidates shown to triage per run
SCOUT_DRY_RUN        = os.getenv("SCOUT_DRY_RUN", "") == "1"           # stop after triage (testing)

# Fraction of the triage window reserved for lens-tagged candidates (live market
# action) — the rest is pure staleness rotation through the full universe.
TAG_RESERVE_FRACTION = 0.25


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically (tmp file + os.replace) so a crash mid-write can't
    corrupt the dedup/notify window file — a corrupt file loads as {} and would
    re-debate and re-alert everything."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load_history() -> dict:
    """Load {ticker: {ts, score, grade}} from disk. Returns {} if missing or corrupt."""
    try:
        if SCOUT_HISTORY_FILE.exists():
            return json.loads(SCOUT_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_history(history: dict) -> None:
    """Persist scout history to disk."""
    try:
        _atomic_write_text(SCOUT_HISTORY_FILE, json.dumps(history, indent=2))
    except Exception as e:
        print(f"  [scout] Warning: could not save history: {e}")


def _recently_scouted(history: dict) -> set[str]:
    """Return set of tickers analyzed within SCOUT_COOLDOWN_HOURS."""
    cutoff = datetime.now(timezone.utc).timestamp() - SCOUT_COOLDOWN_HOURS * 3600
    return {ticker for ticker, entry in history.items() if entry.get("ts", 0) >= cutoff}


def _load_notified() -> dict:
    """Load {ticker: {ts, score, grade}} Telegram notification history. Returns {} if missing."""
    try:
        if SCOUT_NOTIFIED_FILE.exists():
            return json.loads(SCOUT_NOTIFIED_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_notified(notified: dict) -> None:
    """Persist Telegram notification history to disk."""
    try:
        _atomic_write_text(SCOUT_NOTIFIED_FILE, json.dumps(notified, indent=2))
    except Exception as e:
        print(f"  [scout] Warning: could not save notify history: {e}")


def _recently_notified(notified: dict) -> set[str]:
    """Return set of tickers Telegram-alerted within SCOUT_NOTIFY_COOLDOWN_HOURS."""
    cutoff = datetime.now(timezone.utc).timestamp() - SCOUT_NOTIFY_COOLDOWN_HOURS * 3600
    return {ticker for ticker, entry in notified.items() if entry.get("ts", 0) >= cutoff}


def _load_seen() -> dict:
    """Load the triage rotation ledger {ticker: {last_shown, shown, picked}}."""
    try:
        if SCOUT_SEEN_FILE.exists():
            return json.loads(SCOUT_SEEN_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_seen(seen: dict) -> None:
    """Persist the triage rotation ledger to disk (atomic)."""
    try:
        _atomic_write_text(SCOUT_SEEN_FILE, json.dumps(seen, indent=2))
    except Exception as e:
        print(f"  [scout] Warning: could not save rotation ledger: {e}")


def _update_seen(seen: dict, shown: list[dict], picked: set[str], now: float) -> dict:
    """Record a triage round in the ledger: every windowed ticker was shown;
    Gemma's picks additionally count as picked. Mutates and returns ``seen``."""
    for c in shown:
        e = seen.setdefault(c["ticker"], {"last_shown": 0, "shown": 0, "picked": 0})
        e["last_shown"] = now
        e["shown"] = e.get("shown", 0) + 1
    for t in picked:
        e = seen.setdefault(t, {"last_shown": now, "shown": 0, "picked": 0})
        e["picked"] = e.get("picked", 0) + 1
    return seen


def _select_window(candidates: list[dict], seen: dict, window_size: int) -> list[dict]:
    """Pick the rotating triage window from the candidate universe.

    ~TAG_RESERVE_FRACTION of the window goes to lens-tagged names (today's live
    market action), ordered stalest-first then most-tagged. The rest is pure
    rotation: stalest ``last_shown`` first (never-shown = 0 sorts first), with
    market cap as the tiebreak. Pure function — no I/O.
    """
    def last_shown(c: dict) -> float:
        return (seen.get(c["ticker"]) or {}).get("last_shown", 0)

    tagged = [c for c in candidates if c.get("lenses")]
    reserve = min(int(window_size * TAG_RESERVE_FRACTION), len(tagged))
    tagged_sorted = sorted(
        tagged, key=lambda c: (last_shown(c), -len(c.get("lenses") or []), -(c.get("mcap_b") or 0))
    )
    window = tagged_sorted[:reserve]
    chosen = {c["ticker"] for c in window}

    rest = sorted(
        (c for c in candidates if c["ticker"] not in chosen),
        key=lambda c: (last_shown(c), -(c.get("mcap_b") or 0)),
    )
    window += rest[: max(0, window_size - len(window))]
    return window


# Yahoo Finance predefined screener API — no key required
YF_SCREENER_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
YF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# ── Screener lens definitions ──────────────────────────────────────────────────

SCREENER_LENSES: list[dict] = [
    # ── Cross-sector / factor lenses ───────────────────────────────────────────
    {
        "name": "value",
        "desc": "Large-cap undervalued stocks — low P/E, strong fundamentals",
        "scrId": "undervalued_large_caps",
        "count": 250,
    },
    {
        "name": "growth",
        "desc": "Technology growth stocks with strong revenue momentum",
        "scrId": "growth_technology_stocks",
        "count": 250,
    },
    {
        "name": "momentum",
        "desc": "Most actively traded — high volume, momentum plays",
        "scrId": "most_actives",
        "count": 250,
    },
    {
        "name": "small_cap",
        "desc": "Small-cap gainers — hidden gems with asymmetric upside",
        "scrId": "small_cap_gainers",
        "count": 250,
    },
    {
        "name": "aggressive_small_cap",
        "desc": "Aggressive small-caps — high-risk, high-reward growth",
        "scrId": "aggressive_small_caps",
        "count": 250,
    },
    {
        "name": "contrarian",
        "desc": "Day losers — oversold names with potential reversal setups",
        "scrId": "day_losers",
        "count": 250,
    },
    {
        "name": "macro_tailwind",
        "desc": "Undervalued growth — cyclical and macro-sensitive opportunities",
        "scrId": "undervalued_growth_stocks",
        "count": 250,
    },
    {
        "name": "breakout",
        "desc": "Day gainers — strong price action with near-term catalysts",
        "scrId": "day_gainers",
        "count": 250,
    },
    {
        "name": "quality",
        "desc": "Portfolio anchors — quality large-caps with durable franchises",
        "scrId": "portfolio_anchors",
        "count": 250,
    },
    # NB: the 10 ms_* sector lenses were removed 2026-06-12 — they existed only to
    # approximate full-market coverage, which _fetch_universe() now provides by
    # construction. The remaining factor lenses act as signal TAGS on the universe.
]


# ── Yahoo Finance screener helpers ─────────────────────────────────────────────

# NYSE (NYQ), Nasdaq tiers (NMS/NGM/NCM), AMEX (ASE) — listed-only, no OTC.
_UNIVERSE_EXCHANGES = ["NMS", "NYQ", "NGM", "NCM", "ASE"]


def _fetch_universe() -> list[dict] | None:
    """THE candidate universe: every US-listed common stock above SCOUT_MIN_MCAP,
    via yfinance's custom screener (handles Yahoo cookie/crumb internally).

    ~3,900 names at the $300M default floor, paginated 250/page (~16 requests).
    Returns None on failure or an implausibly small result so the caller can fall
    back to the predefined-lens union — a Yahoo API change must never kill scout.
    """
    import time as _time
    import yfinance as yf
    try:
        q = yf.EquityQuery("and", [
            yf.EquityQuery("is-in", ["exchange", *_UNIVERSE_EXCHANGES]),
            yf.EquityQuery("gt", ["intradaymarketcap", SCOUT_MIN_MCAP]),
        ])
        out: list[dict] = []
        seen: set[str] = set()
        offset, total = 0, None
        while offset < 10_000:
            r = yf.screen(q, offset=offset, size=250,
                          sortField="intradaymarketcap", sortAsc=False)
            if total is None:
                total = int(r.get("total") or 0)
            quotes = r.get("quotes") or []
            if not quotes:
                break
            for item in quotes:
                sym = (item.get("symbol") or "").upper().strip()
                if not sym or sym in seen:
                    continue
                if item.get("quoteType") != "EQUITY":
                    continue
                if not re.match(r"^[A-Z]{1,5}$", sym):
                    continue
                seen.add(sym)
                out.append({
                    "ticker": sym,
                    "name":   item.get("longName") or item.get("shortName") or sym,
                    "mcap_b": round((item.get("marketCap") or 0) / 1e9, 2),
                    "price":  item.get("regularMarketPrice") or 0,
                    "volume": item.get("regularMarketVolume") or 0,
                    "lenses": [],
                })
            offset += 250
            if offset >= total:
                break
            _time.sleep(0.2)
        if len(out) < 500:
            print(f"  [scout] Universe fetch returned only {len(out)} names — "
                  f"falling back to predefined lenses")
            return None
        return out
    except Exception as e:
        print(f"  [scout] Universe fetch failed ({e}) — falling back to predefined lenses")
        return None


def _yf_screen(lens: dict) -> tuple[dict, list[dict]]:
    """Call Yahoo Finance predefined screener for one lens. Returns (lens, results)."""
    try:
        r = requests.get(
            YF_SCREENER_URL,
            params={
                "formatted": "false",
                "scrIds": lens["scrId"],
                "count": lens.get("count", 50),
                "region": "US",
                "lang": "en-US",
            },
            headers=YF_HEADERS,
            timeout=20,
        )
        if not r.ok:
            print(f"  [scout] YF screener HTTP {r.status_code} ({lens['name']})")
            return lens, []
        data = r.json()
        quotes = (
            data.get("finance", {})
                .get("result", [{}])[0]
                .get("quotes", [])
        )
        return lens, quotes if isinstance(quotes, list) else []
    except Exception as e:
        print(f"  [scout] YF screener error ({lens['name']}): {e}")
        return lens, []


async def _screen_lenses_raw() -> list[tuple[dict, list[dict]]]:
    """Run all factor lenses in parallel. Returns [(lens, raw_quotes), ...]."""
    return list(await asyncio.gather(*[
        asyncio.to_thread(_yf_screen, lens) for lens in SCREENER_LENSES
    ]))


def _candidates_from_lenses(raw: list[tuple[dict, list[dict]]], skip: set[str]) -> list[dict]:
    """FALLBACK universe builder (when _fetch_universe fails): the deduplicated
    union of the predefined lenses — the pre-2026-06-12 behavior. A ticker hit by
    several lenses carries all of them as tags."""
    by_ticker: dict[str, dict] = {}

    for lens, items in raw:
        lname = lens.get("name", "")
        for item in items:
            sym = (item.get("symbol") or "").upper().strip()
            if not sym or sym in skip:
                continue
            if sym in by_ticker:
                if lname and lname not in by_ticker[sym]["lenses"]:
                    by_ticker[sym]["lenses"].append(lname)
                continue
            # Only plain US equity tickers (no ETFs like SPY, BRK.B, etc.)
            if not re.match(r'^[A-Z]{1,5}$', sym):
                continue
            # Skip very low market cap (< $100M) — too speculative for debates
            mcap = item.get("marketCap") or 0
            if mcap < 100_000_000:
                continue
            by_ticker[sym] = {
                "ticker":   sym,
                "name":     item.get("longName") or item.get("shortName") or sym,
                "mcap_b":   round(mcap / 1e9, 2),
                "price":    item.get("regularMarketPrice") or item.get("ask") or 0,
                "volume":   item.get("regularMarketVolume") or 0,
                "lenses":   [lname] if lname else [],
            }

    return list(by_ticker.values())


def _merge_lens_tags(universe: list[dict], raw: list[tuple[dict, list[dict]]],
                     skip: set[str]) -> list[dict]:
    """Tag universe candidates with factor-lens membership. A lens hit that is
    NOT in the universe (e.g. a sub-floor day-gainer) is appended as an extra
    candidate — lens hits are interesting by definition."""
    by_ticker = {c["ticker"]: c for c in universe}

    for lens, items in raw:
        lname = lens.get("name", "")
        if not lname:
            continue
        for item in items:
            sym = (item.get("symbol") or "").upper().strip()
            if not sym or sym in skip or not re.match(r'^[A-Z]{1,5}$', sym):
                continue
            if sym in by_ticker:
                if lname not in by_ticker[sym]["lenses"]:
                    by_ticker[sym]["lenses"].append(lname)
                continue
            mcap = item.get("marketCap") or 0
            if mcap < 100_000_000:
                continue
            by_ticker[sym] = {
                "ticker":   sym,
                "name":     item.get("longName") or item.get("shortName") or sym,
                "mcap_b":   round(mcap / 1e9, 2),
                "price":    item.get("regularMarketPrice") or item.get("ask") or 0,
                "volume":   item.get("regularMarketVolume") or 0,
                "lenses":   [lname],
            }

    return list(by_ticker.values())


# ── Gemma triage ───────────────────────────────────────────────────────────────

def _compute_matched_filters(ratios: dict, sector: str = "") -> list[str]:
    matched = []
    fwd_rev_growth = ratios.get("fwd_revenue_growth") or 0
    gross_margin   = ratios.get("gross_margin") or 0

    if fwd_rev_growth >= 0.50:
        matched.append(f"Rev {fwd_rev_growth*100:.0f}%")
        if gross_margin >= 60:
            matched.append(f"GM {gross_margin:.0f}%")
        if (ratios.get("eps_acceleration") or 0) > 0:
            matched.append("EPS↑")
        if any(s in (sector or "") for s in ["Technology", "Software", "Semiconductor", "Communication"]):
            matched.append("AI/Tech")
    else:
        fwd_peg = ratios.get("fwd_peg")
        if fwd_peg and fwd_peg < 1.5:
            matched.append(f"Fwd PEG {fwd_peg:.1f}")
        rule40 = ratios.get("rule_of_40") or 0
        if rule40 >= 40:
            matched.append(f"R40={rule40:.0f}")
        if gross_margin > 50:
            matched.append(f"GM {gross_margin:.0f}%")
        fcf_yield = ratios.get("fcf_yield") or 0
        if fcf_yield > 0.05:
            matched.append(f"FCF {fcf_yield*100:.1f}%")

    roic = ratios.get("roic") or 0
    if roic > 15:
        matched.append(f"ROIC {roic:.0f}%")

    return matched


TRIAGE_SYSTEM = """You are a senior equity analyst with deep experience across all market caps and sectors.
You have access to live market data via Google Search. Your job is to identify the most
compelling investment opportunities from a pre-screened candidate list.

CANDIDATE SELECTION — apply the appropriate path based on revenue growth rate:

PATH B — HYPERGROWTH CANDIDATES (revenue growth >= 50%):
  Do NOT use forward PEG to evaluate these — it understates quality for fast growers.
  Prioritize when at least 2 of the following 3 fundamental conditions are met:
  - Revenue Q/Q acceleration (this quarter faster than last quarter)
  - Gross margin >= 60% and stable or expanding
  - EPS estimates trending UP vs prior quarter (positive revision momentum)
  Sector bonus (not required): AI infrastructure, cloud, custom silicon, datacenter, cybersecurity — structural tailwind adds conviction.
  A high absolute multiple (30-60x forward earnings) does NOT disqualify a hypergrowth name.

PATH A — STANDARD GROWTH CANDIDATES (revenue growth < 50%):
  Forward PEG IS a valid screening signal here.
  Prioritize when:
  - Forward PEG < 1.5: market underpricing NTM growth
  - Rule of 40 >= 40 AND gross margin > 50%: execution quality
  - FCF yield > 5% with stable or growing revenue: capital-efficient compounder
  Also consider: clear near-term catalysts, insider buying, sector tailwinds, small/mid-cap asymmetric upside.

DEPRIORITIZE across all paths:
  - Revenue growth declining for 2+ consecutive quarters with no explained catalyst
  - Heavy net insider selling (>$10M in 90 days) not attributable to planned 10b5-1 sales
  - Earnings estimates being cut while stock is near 52-week high (distribution phase)

Bias toward less-covered names where genuine alpha exists. Spread picks across at least 3 different lenses."""


def _build_triage_prompt(candidates: list[dict], portfolio: set[str], debate_count: int = 12) -> str:
    lines = [
        f"Below is a rotating window of {len(candidates)} stocks drawn from the FULL US-listed "
        f"market (every NYSE/Nasdaq/AMEX common stock above ${SCOUT_MIN_MCAP/1e6:.0f}M market cap). "
        "Some carry factor-lens tags (momentum, value, breakout, ...) — live market action; "
        "untagged names are EQUALLY valid picks. Judge on research, not name recognition.\n",
        "Use Google Search to research current market conditions and identify which of these "
        "represent the most compelling opportunities RIGHT NOW. Consider earnings season, "
        "sector rotation, macro trends, and any recent news or catalysts.\n",
        f"EXCLUDE tickers already in the portfolio: {', '.join(sorted(portfolio)) or 'none'}.\n",
        "CANDIDATE WINDOW:",
        f"{'TICKER':<8} {'NAME':<38} {'MCAP($B)':<10} {'LENS TAGS':<30}",
        "-" * 88,
    ]
    for c in candidates:
        tags = ", ".join(c.get("lenses") or []) or "—"
        lines.append(
            f"{c['ticker']:<8} {c['name'][:37]:<38} "
            f"{c['mcap_b']:<10.2f} {tags[:29]:<30}"
        )
    lines += [
        f"\nSelect EXACTLY {debate_count} tickers from this list that represent the best risk-adjusted "
        "opportunities based on your web research. Prefer less-covered names with asymmetric upside. "
        "When lens tags are present, spread picks across different tags — but a great untagged "
        "name always beats a mediocre tagged one.",
        "\nReturn your answer as a JSON object with this exact structure:",
        '{"picks": [',
        '  {"ticker": "SYM", "lens": "value", "rationale": "one concise sentence why"},',
        '  ...',
        ']}',
        "Return ONLY the JSON object. No other text.",
    ]
    return "\n".join(lines)


async def _triage_with_gemma(
    candidates: list[dict],
    portfolio: set[str],
    verbose: bool = True,
    debate_count: int = 6,
) -> list[dict] | None:
    """Returns the validated picks, or None when triage itself failed (LLM error
    or unparseable output) — None tells the caller the window was never
    evaluated, so it must not be marked shown in the rotation ledger."""
    from llm import call_gemini_async, extract_json

    if not candidates:
        return []

    prompt = _build_triage_prompt(candidates, portfolio, debate_count=debate_count)

    if verbose:
        print(f"  [scout] Triaging {len(candidates)} candidates with grounded Gemma...")

    try:
        text = await call_gemini_async(TRIAGE_SYSTEM, prompt, grounding=True, temperature=0.3)
    except Exception as e:
        print(f"  [scout] Triage LLM call failed: {e}")
        print("  [scout] Skipping run — window will be re-shown next run")
        return None

    try:
        parsed = extract_json(text)
        picks = parsed.get("picks", []) if isinstance(parsed, dict) else []
        valid_syms = {c["ticker"] for c in candidates}
        valid = [
            p for p in picks
            if isinstance(p, dict)
            and p.get("ticker", "").upper() in valid_syms
            and p.get("ticker", "").upper() not in portfolio
        ]
        if verbose:
            print(f"  [scout] Gemma selected: {[p['ticker'] for p in valid]}")
        return valid[:debate_count]
    except Exception as e:
        print(f"  [scout] Triage parse error: {e}\n  Raw: {text[:300]}")
        print("  [scout] Skipping run — refusing to debate/alert randomly-selected "
              "tickers on a triage parse failure. Window will be re-shown next run.")
        return None


# ── Main entry point ───────────────────────────────────────────────────────────

async def run_scout(
    max_tickers: int | None = None,
    portfolio: list[str] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Full scout pipeline:
      1. All screener lenses fire simultaneously
      2. Gemma triage picks SCOUT_DEBATE_COUNT candidates (grounded)
      3. Full 5-agent debate on all picks in parallel (max 4 concurrent)

    ``max_tickers`` is an optional extra cap on top of SCOUT_DEBATE_COUNT —
    leave it None so the triage count is the single knob (passing a smaller
    value silently discards picks Gemma already selected).

    Configurable via env vars:
      SCOUT_DEBATE_COUNT   — tickers to debate per run (default 6)
      SCOUT_MAX_LOOPS      — max debate convergence loops (default 3)
      SCOUT_COOLDOWN_HOURS — hours before re-analyzing a ticker (default 48)

    Returns list of BUY discovery dicts (score >= BUY_THRESHOLD only).
    """
    from dossier import build as build_dossier
    from debate import run as run_debate

    portfolio_set = {t.upper() for t in (portfolio or [])}

    # Load dedup history — skip tickers analyzed within SCOUT_COOLDOWN_HOURS
    history = _load_history()
    recently = _recently_scouted(history)
    if verbose and recently:
        print(f"  [scout] Skipping {len(recently)} recently-analyzed ticker(s): "
              f"{', '.join(sorted(recently))}")

    # Phase 1 — full-market universe + factor lenses, fetched concurrently
    if verbose:
        print(f"\n+----------------------------------------------+")
        print(f"|  SOVEREIGN SCOUT — full-market screen        |")
        print(f"|  Universe fetch + {len(SCREENER_LENSES)} factor lenses...        |")
        print(f"+----------------------------------------------+")

    skip = portfolio_set | recently
    universe, raw_lenses = await asyncio.gather(
        asyncio.to_thread(_fetch_universe),
        _screen_lenses_raw(),
    )

    if universe:
        universe = [c for c in universe if c["ticker"] not in skip]
        candidates = _merge_lens_tags(universe, raw_lenses, skip)
    else:
        # Yahoo custom screener unavailable — predefined-lens union (legacy mode)
        candidates = _candidates_from_lenses(raw_lenses, skip)

    if verbose:
        tagged = sum(1 for c in candidates if c["lenses"])
        print(f"  Universe: {len(candidates)} candidates "
              f"({'full-market' if universe else 'LENS FALLBACK'}, {tagged} lens-tagged, "
              f"excluding {len(recently)} in cooldown)")

    if not candidates:
        if verbose:
            print("  [scout] No new candidates — all tickers within cooldown window")
        return []

    # Phase 1b — rotating triage window (stalest-first + tagged reserve)
    seen_ledger = _load_seen()
    window = _select_window(candidates, seen_ledger, SCOUT_WINDOW_SIZE)
    if verbose:
        never_shown = sum(
            1 for c in window if (seen_ledger.get(c["ticker"]) or {}).get("last_shown", 0) == 0
        )
        w_tagged = sum(1 for c in window if c["lenses"])
        print(f"  Window: {len(window)} shown to triage "
              f"({w_tagged} tagged, {never_shown} never shown before)")

    # Phase 2 — Gemma triage (one grounded call)
    picks = await _triage_with_gemma(
        window, portfolio_set, verbose=verbose, debate_count=SCOUT_DEBATE_COUNT
    )

    if picks is None:
        # Triage itself failed (LLM error / unparseable output) — the window
        # was never evaluated, so leave the ledger untouched: the same names
        # get first shot at the next run instead of silently skipping ~600
        # stocks a full rotation cycle.
        return []

    # Record the round in the rotation ledger BEFORE any early return — shown
    # names rotate out of the window even when triage picks nothing.
    _update_seen(
        seen_ledger, window,
        {p["ticker"].upper() for p in picks},
        datetime.now(timezone.utc).timestamp(),
    )
    _save_seen(seen_ledger)

    if not picks:
        print("  [scout] Triage returned no picks")
        return []

    if max_tickers:
        picks = picks[:max_tickers]

    if SCOUT_DRY_RUN:
        print(f"\n  [scout] DRY RUN — triage picked: "
              f"{', '.join(p['ticker'] for p in picks)} — skipping debates")
        return []

    # Phase 2b — metadata cleaner: one grounded call for the whole batch.
    # Resolves canonical GICS sector, ADR status, and financials currency before
    # the debates run so the dossier builder has clean inputs for every ticker.
    from cleaner import clean_ticker_batch
    batch_meta = await clean_ticker_batch([p["ticker"] for p in picks])

    if verbose:
        print(f"\n+----------------------------------------------+")
        print(f"|  SOVEREIGN SCOUT — running debates           |")
        tickers_str = ", ".join(p["ticker"] for p in picks)
        print(f"|  Picks: {tickers_str:<39}|")
        print(f"|  {len(picks)} debates · max {SCOUT_MAX_LOOPS} loop(s) · 4 concurrent     |")
        print(f"+----------------------------------------------+")

    # Phase 3 — parallel debates (max 4 concurrent — matches number of API keys)
    out_dir = Path("output/scouts")
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(4)
    history_lock = asyncio.Lock()

    async def _debate_one(pick: dict) -> dict | None:
        ticker    = pick["ticker"].upper()
        lens      = pick.get("lens", "")
        rationale = pick.get("rationale", "")
        async with sem:
            try:
                if verbose:
                    print(f"\n  [scout] Analyzing {ticker} ({lens})...")
                    if rationale:
                        print(f"          Gemma rationale: {rationale[:100]}")

                dossier = await build_dossier(ticker, verbose=False,
                                              meta=batch_meta.get(ticker, {}))
                result  = await run_debate(ticker, dossier, verbose=False, max_loops=SCOUT_MAX_LOOPS)

                score = result.get("consensus_score", 0)
                grade = result.get("consensus_grade", "HOLD")

                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"{ticker}_{ts}.json"
                _ratios  = dossier.get("financials", {}).get("ratios_ttm", {})
                _path    = "B" if (_ratios.get("fwd_revenue_growth") or 0) >= 0.50 else "A"
                _matched = _compute_matched_filters(_ratios, dossier.get("profile", {}).get("sector", ""))
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"result": result, "dossier": dossier, "meta": {"path": _path, "matched_filters": _matched}}, f, indent=2, default=str)

                if verbose:
                    _tag = ""
                    if score >= BUY_THRESHOLD:
                        _tag = (" ← BUY ✓ CONFIRMED" if result.get("confirmed", True)
                                else " ← BUY ⚠ UNDER REVIEW")
                    print(f"  [scout] {ticker} → {score:.2f}/10 [{grade}]{_tag}")

                # Record in history regardless of grade (prevents re-analysis in cooldown window)
                # Save immediately so a mid-run crash doesn't lose completed tickers
                async with history_lock:
                    history[ticker] = {
                        "ts":    datetime.now(timezone.utc).timestamp(),
                        "score": round(score, 2),
                        "grade": grade,
                    }
                    _save_history(history)

                if score >= BUY_THRESHOLD:
                    return {
                        "ticker":           ticker,
                        "score":            round(score, 2),
                        "grade":            grade,
                        "confidence":       result.get("confidence", ""),
                        "thesis":           result.get("majority_thesis", ""),
                        "score_rationale":  result.get("score_rationale", ""),
                        "dissent":          result.get("dissent", ""),
                        "key_swing_factor": result.get("key_swing_factor", ""),
                        "catalyst":         result.get("catalyst", ""),
                        "asymmetry_ratio":  result.get("asymmetry_ratio", ""),
                        "rr":               (result.get("risk_reward") or {}).get("rr_ratio"),
                        "risk":             (result.get("risk_reward") or {}).get("risk_tier"),
                        "banger":           result.get("banger", {}),
                        "position_guidance": result.get("position_guidance", {}),
                        "cycle_position":   result.get("cycle_position", {}),
                        "path":             _path,
                        "matched_filters":  _matched,
                        "scout_lens":       lens,
                        "gemma_rationale":  rationale,
                        "analyzed_at":      ts,
                        "output_file":      str(out_path),
                        "confirmed":        result.get("confirmed", True),
                        "verification":     result.get("verification", {}),
                    }
                return None
            except Exception as e:
                print(f"  [scout] {ticker} failed: {e}")
                # Mark in history so this ticker isn't re-queued until the cooldown expires.
                # Prevents quota-exhausted tickers from being re-debated every run.
                async with history_lock:
                    history[ticker] = {
                        "ts":    datetime.now(timezone.utc).timestamp(),
                        "score": 0.0,
                        "grade": "FAILED",
                    }
                    _save_history(history)
                return None

    results = await asyncio.gather(*[_debate_one(p) for p in picks])
    discoveries = [r for r in results if r is not None]

    if verbose:
        print(f"\n  Scout complete: {len(discoveries)} BUY signal(s) "
              f"from {len(picks)} debated · history now has {len(history)} ticker(s)")

    return discoveries
