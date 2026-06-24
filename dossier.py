"""Data dossier builder â€" async, parallel fetches per ticker, shared macro cache."""

import asyncio
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import requests
import yfinance as yf
from dotenv import load_dotenv

from cache import cached
from live_events import emit_live

load_dotenv()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
FRED_KEY    = os.getenv("FRED_API_KEY", "")
FMP_KEY     = os.getenv("FMP_API_KEY", "")
_av_keys = [k.strip() for k in os.getenv("ALPHA_VANTAGE_API_KEYS", os.getenv("ALPHA_VANTAGE_API_KEY", "")).split(",") if k.strip()]

FH = "https://finnhub.io/api/v1"

_av_idx = 0
_av_lock = threading.Lock()      # serialize AV key rotation + rate-limit enforcement
_av_last_call: float = 0.0       # timestamp of most recent AV request
_AV_MIN_INTERVAL = 12.0          # seconds between calls (5 RPM limit = 1 per 12s)

_fmp_lock = threading.Lock()     # FMP rate-limit: 10 RPM free tier
_fmp_last_call: float = 0.0
_FMP_MIN_INTERVAL = 6.0          # seconds between calls (10 RPM = 1 per 6s)


# ── Cycle type classification ──────────────────────────────────────────────────
_CYCLICAL_SECTORS = {"Energy", "Basic Materials", "Consumer Cyclical", "Real Estate"}
_SECULAR_SECTORS  = {"Technology", "Healthcare", "Communication Services"}
_DEFENSIVE_SECTORS = {"Consumer Defensive", "Utilities"}


def _cycle_type(sector: str) -> str:
    """Classify a sector as SECULAR, CYCLICAL, DEFENSIVE, or HYBRID."""
    if sector in _SECULAR_SECTORS:  return "SECULAR"
    if sector in _CYCLICAL_SECTORS: return "CYCLICAL"
    if sector in _DEFENSIVE_SECTORS: return "DEFENSIVE"
    return "HYBRID"


def _detect_regime(macro: dict) -> str:
    """Classify the current macro regime from FRED indicators.

    Returns one of: EXPANSION | PEAK | LATE_CYCLE | RECESSION | INFLATIONARY | MID_CYCLE
    """
    fed     = macro.get("fed_funds_rate")
    cpi     = macro.get("cpi_yoy")
    unemp   = macro.get("unemployment")
    vix     = macro.get("vix")
    spread  = macro.get("yield_curve_spread")  # 10Y - 2Y

    # Priority order matters — most diagnostic condition first
    if spread is not None and spread < 0:
        return "LATE_CYCLE"          # inverted yield curve is the strongest signal
    if cpi is not None and cpi > 4.5:
        return "INFLATIONARY"
    if unemp is not None and unemp > 5.5:
        return "RECESSION"
    if fed is not None and unemp is not None and fed > 4.5 and unemp < 4.5:
        return "PEAK"                # tight labor, elevated rates
    if vix is not None and vix < 16:
        return "EXPANSION"           # low volatility, risk-on
    return "MID_CYCLE"


async def _fetch_and_emit(ticker: str, coro, source_name: str):
    """Await coro then fire a FETCH_DONE live event for visual dossier progress."""
    result = await coro
    await emit_live(ticker, {"type": "FETCH_DONE", "source": source_name})
    return result

# Macro data (FRED + VIX) is identical for all tickers â€" fetch once per run
_macro_cache: dict = {}
_macro_fetched = False
_macro_async_lock = asyncio.Lock()


# â"€â"€ Sync HTTP helpers (run inside asyncio.to_thread) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def _fh(path: str, params: dict = None) -> dict | list:
    try:
        p = {"token": FINNHUB_KEY, **(params or {})}
        r = requests.get(f"{FH}{path}", params=p, timeout=15)
        if r.status_code == 429:
            import random
            time.sleep(5 + random.uniform(0, 3))
            r = requests.get(f"{FH}{path}", params=p, timeout=15)
        return r.json() if r.ok else {}
    except requests.exceptions.RequestException as e:
        print(f"  [dossier] Finnhub {path} failed: {e}")
        return {}



def _fred(series: str) -> float | None:
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series, "api_key": FRED_KEY, "file_type": "json",
                    "sort_order": "desc", "limit": 1},
            timeout=10,
        )
        obs = r.json().get("observations", [])
        return float(obs[0]["value"]) if obs and obs[0]["value"] != "." else None
    except Exception:
        return None


def _fred_cpi_yoy() -> float | None:
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "CPIAUCSL", "api_key": FRED_KEY, "file_type": "json",
                    "sort_order": "desc", "limit": 13},
            timeout=10,
        )
        obs = r.json().get("observations", [])
        vals = [float(o["value"]) for o in obs if o["value"] != "."]
        if len(vals) >= 12:
            return round((vals[0] - vals[-1]) / vals[-1] * 100, 1)
        return None
    except Exception:
        return None


def _safe_float(val) -> float | None:
    """Parse AV overview strings ('12.77', 'None', '-') to float or None."""
    try:
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _safe_div(a, b) -> float | None:
    try:
        return round(a / b, 4) if a is not None and b is not None and b != 0 else None
    except (TypeError, ZeroDivisionError):
        return None


def _safe_sub(a, b) -> float | None:
    try:
        return round(a - b, 4) if a is not None and b is not None else None
    except TypeError:
        return None


def _av(function: str, params: dict = None) -> dict:
    global _av_idx, _av_last_call
    if not _av_keys:
        return {}
    with _av_lock:
        key = _av_keys[_av_idx % len(_av_keys)]
        _av_idx += 1
        # Enforce 5 RPM: sleep inside the lock so concurrent threads queue up
        wait = _AV_MIN_INTERVAL - (time.time() - _av_last_call)
        if wait > 0:
            time.sleep(wait)
        _av_last_call = time.time()
    try:
        p = {"function": function, "apikey": key, **(params or {})}
        r = requests.get("https://www.alphavantage.co/query", params=p, timeout=15)
        return r.json() if r.ok else {}
    except requests.exceptions.RequestException as e:
        print(f"  [dossier] AlphaVantage {function} failed: {e}")
        return {}


def _fmp_estimates(ticker: str) -> dict:
    """Fetch annual analyst consensus from FMP stable API (250 req/day free tier).

    Returns NTM EPS avg, NTM revenue avg, computed forward growth rates,
    and analyst count. Uses the two closest future fiscal years to compute growth.
    Falls back to {} on any error or missing key.
    """
    global _fmp_last_call
    if not FMP_KEY:
        return {}
    with _fmp_lock:
        wait = _FMP_MIN_INTERVAL - (time.time() - _fmp_last_call)
        if wait > 0:
            time.sleep(wait)
        _fmp_last_call = time.time()
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        r = requests.get(
            "https://financialmodelingprep.com/stable/analyst-estimates",
            params={"symbol": ticker, "period": "annual", "apikey": FMP_KEY},
            timeout=10,
        )
        if not r.ok:
            return {}
        data = r.json()
        if not isinstance(data, list) or not data:
            return {}
        data.sort(key=lambda x: x.get("date", ""))
        future = [e for e in data if e.get("date", "") >= today]
        if not future:
            return {}
        ntm = future[0]
        ntm_idx = data.index(ntm)
        if ntm_idx == 0:
            return {}
        prior = data[ntm_idx - 1]
        ntm_rev   = ntm.get("revenueAvg")
        prior_rev = prior.get("revenueAvg")
        ntm_eps   = ntm.get("epsAvg")
        prior_eps = prior.get("epsAvg")
        result: dict = {
            "fwd_eps_ntm":        ntm_eps,
            "fwd_rev_ntm":        ntm_rev,
            "num_analysts_eps":   ntm.get("numAnalystsEps"),
            "num_analysts_rev":   ntm.get("numAnalystsRevenue"),
            "ntm_date":           ntm.get("date"),
        }
        if ntm_rev and prior_rev and prior_rev > 0:
            result["fwd_rev_growth"] = round((ntm_rev - prior_rev) / prior_rev, 4)
        if ntm_eps is not None and prior_eps and prior_eps != 0:
            result["fwd_eps_growth"] = round((ntm_eps - prior_eps) / abs(prior_eps), 4)
        return result
    except Exception as e:
        print(f"  [dossier] FMP estimates {ticker} failed: {e}")
        return {}


def _get_vix() -> float | None:
    try:
        return float(yf.Ticker("^VIX").history(period="2d")["Close"].iloc[-1])
    except Exception:
        return None


def _technicals(ticker: str) -> dict:
    try:
        try:
            hist = yf.Ticker(ticker).history(period="1y")
        except Exception:
            time.sleep(5)
            hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty:
            return {}
        close = hist["Close"]
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])
        price = float(close.iloc[-1])

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = float(100 - 100 / (1 + rs.iloc[-1]))

        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = float((ema12 - ema26).iloc[-1])
        signal_line = float((ema12 - ema26).ewm(span=9).mean().iloc[-1])

        w52_high = float(hist["High"].max())
        w52_low = float(hist["Low"].min())
        pct_from_high = round((price - w52_high) / w52_high * 100, 1)

        return {
            "price": round(price, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "above_sma50": price > sma50,
            "above_sma200": price > sma200,
            "rsi_14": round(rsi, 1),
            "macd_line": round(macd_line, 3),
            "macd_signal": round(signal_line, 3),
            "macd_bullish": macd_line > signal_line,
            "52w_high": round(w52_high, 2),
            "52w_low": round(w52_low, 2),
            "pct_from_52w_high": pct_from_high,
        }
    except Exception as e:
        return {"error": str(e)}


def _yf_financials(ticker: str) -> dict:
    try:
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
        except Exception:
            time.sleep(5)
            t = yf.Ticker(ticker)
            info = t.info or {}

        def _pct(v):
            return round(v * 100, 2) if v is not None else None

        def _r(v, n=2):
            return round(v, n) if v is not None else None

        # For ADRs / foreign stocks, yfinance mixes underlying share counts with ADR-level price.
        # Detection uses three independent signals:
        #   1. quoteType == "ADR" — explicit yfinance flag
        #   2. sharesOutstanding / floatShares > 2 — implicit ratio heuristic
        #   3. financialCurrency != currency — financials in foreign currency while price is USD
        #      (catches NYSE-listed foreign companies like TSM that yfinance labels as EQUITY)
        shares_out = info.get("sharesOutstanding") or 0
        float_shares = info.get("floatShares") or 0
        _fin_currency = info.get("financialCurrency") or "USD"
        _price_currency = info.get("currency") or "USD"
        _is_adr = info.get("quoteType", "").upper() == "ADR"
        _is_share_ratio_mismatch = float_shares > 0 and shares_out > 0 and shares_out / float_shares > 2
        _is_fx_mismatch = _price_currency == "USD" and _fin_currency != "USD"
        _is_adr_mismatch = _is_adr or _is_share_ratio_mismatch or _is_fx_mismatch
        # Only use floatShares when the ratio heuristic fires — for fx-mismatch ADRs (TSM),
        # sharesOutstanding is already in ADR units and pairs correctly with the USD ADR price.
        _safe_shares = float_shares if _is_share_ratio_mismatch else shares_out

        # P/B and P/S from yfinance are computed as price / (metric / sharesOutstanding).
        # For ADRs this produces wrong values; null them out and let agents use web research.
        _pb = None if _is_adr_mismatch else _r(info.get("priceToBook"))
        _ps = None if _is_adr_mismatch else _r(info.get("priceToSalesTrailing12Months"))
        _fcf_ps = None  # always compute from safe_shares below if fcf available
        _fcf = info.get("freeCashflow")
        if _fcf and _safe_shares:
            _fcf_ps = _r(_fcf / _safe_shares)

        ratios = {
            "pe":            _r(info.get("trailingPE")),
            "fwd_pe":        _r(info.get("forwardPE")),
            "pb":            _pb,
            "ps":            _ps,
            "ev_ebitda":     _r(info.get("enterpriseToEbitda")),
            "gross_margin":  _pct(info.get("grossMargins")),
            "net_margin":    _pct(info.get("profitMargins")),
            "roe":           _pct(info.get("returnOnEquity")),
            "roa":           _pct(info.get("returnOnAssets")),
            "debt_equity":   _r(info.get("debtToEquity")),
            "current_ratio": _r(info.get("currentRatio")),
            "fcf":           _fcf,
            "fcf_per_share": _fcf_ps,
            "revenue_ttm":   info.get("totalRevenue"),
            "ebitda":        info.get("ebitda"),
            "beta":          _r(info.get("beta")),
            "shares_out":    _safe_shares,
            "short_pct":     _pct(info.get("shortPercentOfFloat")),
            "adr_mismatch":  _is_adr_mismatch,  # flag for downstream consumers
            "op_margin":     _pct(info.get("operatingMargins")),
            "trailing_eps":  info.get("trailingEps"),
            "forward_eps":   info.get("forwardEps"),
        }

        analyst = {
            "target_mean":    _r(info.get("targetMeanPrice")),
            "target_high":    _r(info.get("targetHighPrice")),
            "target_low":     _r(info.get("targetLowPrice")),
            "num_analysts":   info.get("numberOfAnalystOpinions"),
            "recommendation": info.get("recommendationKey", ""),
        }

        income = []
        try:
            fin = t.financials
            if fin is not None and not fin.empty:
                for col in fin.columns[:4]:
                    rev = fin.loc["Total Revenue", col] if "Total Revenue" in fin.index else None
                    gp  = fin.loc["Gross Profit", col] if "Gross Profit" in fin.index else None
                    oi  = fin.loc["Operating Income", col] if "Operating Income" in fin.index else None
                    ni  = fin.loc["Net Income", col] if "Net Income" in fin.index else None
                    rd  = fin.loc["Research And Development", col] if "Research And Development" in fin.index else None
                    cor = fin.loc["Cost Of Revenue", col] if "Cost Of Revenue" in fin.index else None
                    income.append({
                        "date": str(col.date()),
                        "revenue": int(rev) if rev is not None and str(rev) != "nan" else None,
                        "gross_profit": int(gp) if gp is not None and str(gp) != "nan" else None,
                        "operating_income": int(oi) if oi is not None and str(oi) != "nan" else None,
                        "net_income": int(ni) if ni is not None and str(ni) != "nan" else None,
                        "research_development": int(rd) if rd is not None and str(rd) != "nan" else None,
                        "cost_of_revenue": int(cor) if cor is not None and str(cor) != "nan" else None,
                    })
        except Exception as e:
            print(f"  [dossier] income statement parse failed: {e}")

        balance = []
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                for col in bs.columns[:2]:
                    _bs = lambda key, c=col: int(bs.loc[key, c]) if key in bs.index and str(bs.loc[key, c]) != "nan" else None
                    balance.append({
                        "date": str(col.date()),
                        "total_assets": _bs("Total Assets"),
                        "total_debt": _bs("Total Debt"),
                        "stockholders_equity": _bs("Stockholders Equity"),
                        "cash": _bs("Cash And Cash Equivalents"),
                        "current_assets": _bs("Current Assets"),
                        "current_liabilities": _bs("Current Liabilities"),
                        "goodwill": _bs("Goodwill"),
                        "intangible_assets": _bs("Other Intangible Assets"),
                        "inventory": _bs("Inventory"),
                    })
        except Exception as e:
            print(f"  [dossier] balance sheet parse failed: {e}")

        cashflow = []
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                for col in cf.columns[:2]:
                    _cf = lambda key, c=col: int(cf.loc[key, c]) if key in cf.index and str(cf.loc[key, c]) != "nan" else None
                    op = _cf("Operating Cash Flow") or _cf("Cash Flow From Continuing Operating Activities")
                    capex = _cf("Capital Expenditure")
                    fcf_val = ((op + capex) if op is not None and capex is not None
                               else (op if op is not None else None))
                    sbc = _cf("Stock Based Compensation")
                    cashflow.append({
                        "date": str(col.date()),
                        "operating_cf": op,
                        "capex": capex,
                        "free_cash_flow": fcf_val,
                        "stock_based_compensation": sbc,
                    })
        except Exception as e:
            print(f"  [dossier] cashflow parse failed: {e}")

        # Fresh NTM consensus from Yahoo analyst estimates — more current than info dict.
        # info['earningsGrowth'] / info['revenueGrowth'] can lag 6-12 months; these
        # attributes parse the live analyst consensus page and update daily/weekly.
        estimates: dict = {}
        try:
            ee = t.earnings_estimate
            if ee is not None and not ee.empty:
                def _ee_val(idx, col):
                    try:
                        return float(ee.loc[idx, col]) if idx in ee.index and col in ee.columns and ee.loc[idx, col] is not None else None
                    except (TypeError, ValueError):
                        return None

                estimates["fwd_eps_growth"]          = _ee_val("+1y", "growth")
                estimates["fwd_eps_ntm"]             = _ee_val("+1y", "avg")
                estimates["est_eps_current_q"]       = _ee_val("0q",  "avg")
                estimates["est_eps_current_q_growth"] = _ee_val("0q", "growth")
                estimates["est_eps_next_q"]          = _ee_val("+1q", "avg")
                estimates["est_eps_next_q_growth"]   = _ee_val("+1q", "growth")
                # Strip None values so _first_not_none chains work cleanly
                estimates = {k: v for k, v in estimates.items() if v is not None}
        except Exception:
            pass
        try:
            re_est = t.revenue_estimate
            if re_est is not None and not re_est.empty and "+1y" in re_est.index:
                row = re_est.loc["+1y"]
                if "growth" in re_est.columns and row["growth"] is not None:
                    try:
                        estimates["fwd_rev_growth"] = float(row["growth"])
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass
        try:
            et = t.eps_trend
            if et is not None and not et.empty and "+1y" in et.index:
                row = et.loc["+1y"]
                cur = float(row["current"]) if "current" in et.columns and row["current"] is not None else None
                ago30 = float(row["30daysAgo"]) if "30daysAgo" in et.columns and row["30daysAgo"] is not None else None
                if cur is not None and ago30 is not None and ago30 != 0:
                    estimates["eps_revision_momentum"] = round((cur - ago30) / abs(ago30), 4)
        except Exception:
            pass

        return {"ratios": ratios, "analyst": analyst,
                "income": income, "balance": balance, "cashflow": cashflow,
                "industry": info.get("industry", ""),
                "sector": info.get("sector", ""),
                "company_name": info.get("longName") or info.get("shortName", ""),
                "market_cap": info.get("marketCap"),
                "fwd_revenue_growth": info.get("revenueGrowth"),
                "fwd_earnings_growth": info.get("earningsGrowth"),
                "previous_close": info.get("previousClose"),
                "financials_currency": _fin_currency if _is_fx_mismatch else None,
                "estimates": estimates}
    except Exception as e:
        return {"error": str(e), "ratios": {}, "analyst": {},
                "income": [], "balance": [], "cashflow": [], "industry": "", "sector": ""}


SECTOR_TERMINAL = {
    "Technology": 25, "Healthcare": 20, "Consumer Cyclical": 18,
    "Communication Services": 20, "Financials": 12, "Industrials": 15,
    "Energy": 10, "Utilities": 12, "Consumer Defensive": 16,
    "Real Estate": 14, "Basic Materials": 12,
}

# Maximum blended growth rate allowed in DCF by GICS sector.
# High-growth sectors (Tech, Comms) legitimately sustain 35-40% near-term growth;
# capping them at 25% systematically undervalues compounders like NVDA or META.
# Defensive and capital-constrained sectors get tighter caps.
SECTOR_GROWTH_CAP = {
    "Technology": 0.40,
    "Communication Services": 0.35,
    "Healthcare": 0.30,
    "Consumer Cyclical": 0.25,
    "Industrials": 0.20,
    "Energy": 0.20,
    "Financials": 0.15,
    "Utilities": 0.10,
    "Consumer Defensive": 0.15,
    "Real Estate": 0.15,
    "Basic Materials": 0.15,
}
_DEFAULT_GROWTH_CAP = 0.25


def _compute_roic(yf_fin: dict) -> float | None:
    """ROIC from yfinance: NOPAT / (Equity + Debt). Returns percentage or None."""
    try:
        income = yf_fin.get("income") or []
        balance = yf_fin.get("balance") or []
        if not income or not balance:
            return None
        op_income = income[0].get("operating_income")
        equity = balance[0].get("stockholders_equity")
        debt = balance[0].get("total_debt") or 0
        if op_income is None or equity is None:
            return None
        invested_capital = equity + debt
        if invested_capital <= 0:
            return None
        nopat = op_income * 0.79
        return round(nopat / invested_capital * 100, 2)
    except Exception:
        return None


def _dynamic_dcf(
    fcf: float | None,
    income: list,
    beta: float | None,
    treasury_10y: float | None,
    sector: str,
    shares_out: int | None,
    fwd_revenue_growth: float | None = None,
    fwd_earnings_growth: float | None = None,
    net_debt: float | None = None,
) -> tuple[float | None, dict]:
    """Dynamic DCF using blended growth (historical CAGR + forward analyst estimates),
    CAPM discount rate, and sector-mapped terminal multiple.
    Returns (iv_per_share, assumptions_dict).
    """
    if not fcf or fcf <= 0:
        return None, {}
    try:
        # Growth rate: blend historical revenue CAGR with forward analyst estimates
        revenues = [yr.get("revenue") for yr in income if yr.get("revenue")]
        hist_cagr = None
        if len(revenues) >= 2:
            hist_cagr = (revenues[0] / revenues[-1]) ** (1 / (len(revenues) - 1)) - 1
            rev_years = len(revenues)
        else:
            rev_years = 0

        fwd_growth = fwd_revenue_growth if fwd_revenue_growth is not None else fwd_earnings_growth

        # Use sector-specific growth cap so high-growth sectors (Technology, Communication Services)
        # are not artificially suppressed by a one-size-fits-all 25% ceiling.
        growth_cap = SECTOR_GROWTH_CAP.get(sector, _DEFAULT_GROWTH_CAP)

        if hist_cagr is not None and fwd_growth is not None:
            growth = max(0.02, min(hist_cagr * 0.5 + fwd_growth * 0.5, growth_cap))
            growth_method = "blended"
        elif fwd_growth is not None:
            growth = max(0.02, min(float(fwd_growth), growth_cap))
            growth_method = "forward"
        elif hist_cagr is not None:
            growth = max(0.02, min(float(hist_cagr), growth_cap))
            growth_method = "historical"
        else:
            growth = 0.08
            growth_method = "default"
            rev_years = 0

        # Discount rate: CAPM (risk-free + beta x ERP), clamped to 7-20%
        risk_free = (treasury_10y or 4.3) / 100
        beta_val = max(beta, 0.5) if beta and beta > 0 else 1.0
        discount = max(0.07, min(risk_free + beta_val * 0.055, 0.20))

        # Terminal multiple: sector-mapped
        terminal_mult = SECTOR_TERMINAL.get(sector, 15)

        years = 5
        pv = 0.0
        cf = fcf
        for i in range(1, years + 1):
            cf *= (1 + growth)
            pv += cf / (1 + discount) ** i
        terminal_val = cf * terminal_mult / (1 + discount) ** years
        total_pv = pv + terminal_val

        equity_value = total_pv - (net_debt or 0)
        if equity_value <= 0:
            return None, {"note": "negative_equity_value", "total_pv": round(total_pv, 0), "net_debt": round(net_debt or 0, 0)}
        iv = (round(equity_value / shares_out, 2)
              if shares_out and shares_out > 0
              else round(equity_value, 0))

        assumptions = {
            "growth_rate_pct":    round(growth * 100, 1),
            "growth_method":      growth_method,
            "hist_cagr_pct":      round(hist_cagr * 100, 1) if hist_cagr is not None else None,
            "fwd_growth_pct":     round(fwd_growth * 100, 1) if fwd_growth is not None else None,
            "discount_rate_pct":  round(discount * 100, 1),
            "terminal_multiple":  terminal_mult,
            "years":              years,
            "method":             "blended growth CAGR + CAPM",
            "revenue_years_used": rev_years,
        }
        return iv, assumptions
    except Exception:
        return None, {}


def _compute_change_pct(technicals: dict, yf_fin: dict) -> float | None:
    """Compute daily % change from yfinance when Finnhub quote is unavailable."""
    price = technicals.get("price") if isinstance(technicals, dict) else None
    prev_close = yf_fin.get("previous_close")
    if price is not None and prev_close is not None and prev_close > 0:
        return round((price - prev_close) / prev_close * 100, 4)
    return None


def _apply_fx_conversion(yf_fin: dict, currency: str, verbose: bool = False) -> float | None:
    """Convert non-USD financial statement values to USD using live FX from yfinance.

    Returns the rate used (USD per 1 unit of local currency, e.g. ~0.031 for TWD),
    or None if the rate could not be fetched.
    """
    try:
        fx_ticker = yf.Ticker(f"{currency}USD=X")
        fx_rate = 0.0
        # fast_info is not a dict — use history for reliable rate fetch
        hist = fx_ticker.history(period="1d")
        if not hist.empty:
            fx_rate = float(hist["Close"].iloc[-1])
        if fx_rate <= 0:
            # Fallback: info dict
            info_price = fx_ticker.info.get("regularMarketPrice") or 0
            fx_rate = float(info_price)
        if fx_rate <= 0:
            return None
    except Exception:
        return None

    if verbose:
        print(f"  [dossier] FX {currency}→USD: {fx_rate:.6f}  (converting all financial statements)")

    def _conv_stmt(entries: list) -> list:
        out = []
        for entry in entries:
            converted = {}
            for k, v in entry.items():
                converted[k] = v * fx_rate if (k != "date" and isinstance(v, (int, float))) else v
            out.append(converted)
        return out

    for key in ("income", "balance", "cashflow"):
        if yf_fin.get(key):
            yf_fin[key] = _conv_stmt(yf_fin[key])

    # Convert absolute-dollar fields in ratios TTM; leave ratios/percentages untouched
    ratios = yf_fin.get("ratios", {})
    for field in ("fcf", "revenue_ttm", "ebitda"):
        if ratios.get(field) is not None:
            ratios[field] = ratios[field] * fx_rate
    # Recompute fcf_per_share from the now-USD fcf and ADR share count
    if ratios.get("fcf") is not None and ratios.get("shares_out"):
        ratios["fcf_per_share"] = round(ratios["fcf"] / ratios["shares_out"], 4)

    return fx_rate


def _fetch_peer(peer_ticker: str) -> dict | None:
    try:
        info = yf.Ticker(peer_ticker).info
        return {
            "ticker":       peer_ticker,
            "pe":           round(info.get("trailingPE") or 0, 1),
            "fwd_pe":       round(info.get("forwardPE") or 0, 1),
            "ev_ebitda":    round(info.get("enterpriseToEbitda") or 0, 1),
            "rev_growth":   round((info.get("revenueGrowth") or 0) * 100, 1),
            "gross_margin": round((info.get("grossMargins") or 0) * 100, 1),
        }
    except Exception:
        return None


def _latest_filing(ticker: str) -> dict:
    try:
        filings = _fh("/stock/filings", {"symbol": ticker})
        if isinstance(filings, list) and filings:
            latest = next((f for f in filings if f.get("form") in ("10-K", "10-Q")), filings[0])
            return {
                "form": latest.get("form"),
                "filed_date": latest.get("filedDate"),
                "period": latest.get("reportDate"),
                "url": latest.get("reportUrl", ""),
            }
    except Exception as e:
        return {"error": str(e)}
    return {}


# â"€â"€ Async macro cache (fetched once per run, shared across all tickers) â"€â"€â"€â"€â"€â"€â"€

async def _get_macro() -> dict:
    global _macro_cache, _macro_fetched
    async with _macro_async_lock:
        if _macro_fetched:
            return _macro_cache
        # All 5 FRED series + VIX in parallel
        fed, cpi, unemp, t10, t2, vix = await asyncio.gather(
            asyncio.to_thread(_fred, "FEDFUNDS"),
            asyncio.to_thread(_fred_cpi_yoy),
            asyncio.to_thread(_fred, "UNRATE"),
            asyncio.to_thread(_fred, "DGS10"),
            asyncio.to_thread(_fred, "DGS2"),
            asyncio.to_thread(_get_vix),
        )
        spread = round(t10 - t2, 2) if t10 and t2 else None
        _macro_cache = {
            "fed_funds_rate":     fed,
            "cpi_yoy":            cpi,
            "unemployment":       unemp,
            "treasury_10y":       t10,
            "treasury_2y":        t2,
            "vix":                round(vix, 1) if vix else None,
            "yield_curve_spread": spread,
        }
        _macro_fetched = True
        return _macro_cache


# â"€â"€ Main async builder â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

async def build(ticker: str, verbose: bool = True, meta: dict | None = None) -> dict:
    """Build the full data dossier for a ticker. Async — all sources fetched in parallel.

    meta — optional per-ticker override dict from cleaner.clean_ticker_batch().
           Keys: canonical_sector, is_adr, financials_currency.
           Applied after raw data is fetched but before any assembly, so all
           downstream logic (archetype classification, ADR nulling, etc.) sees
           corrected values without knowing about the override layer.
    """
    ticker = ticker.upper()
    if verbose:
        print(f"\n[dossier] Building dossier for {ticker} (parallel fetch)...")

    dossier: dict = {"ticker": ticker, "built_at": datetime.now(timezone.utc).isoformat()}

    since      = (datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d")
    from_date  = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    to_date    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    since_yr   = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
    fwd_30     = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")

    # â"€â"€ Batch 1: everything that doesn't depend on another result â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    await emit_live(ticker, {"type": "DOSSIER_START"})

    (
        profile_raw,
        quote_raw,
        technicals,
        yf_fin,
        earnings_raw,
        av_overview_raw,
        insiders_raw,
        fh_news_raw,
        sec_raw,
        macro,
        rec_trends_raw,
        insider_sent_raw,
        usa_spending_raw,
        earnings_cal_raw,
        fmp_estimates_raw,
    ) = await asyncio.gather(
        _fetch_and_emit(ticker, asyncio.to_thread(cached, f"fh:profile:{ticker}",       72,  _fh, "/stock/profile2", {"symbol": ticker}), "profile"),
        _fetch_and_emit(ticker, asyncio.to_thread(cached, f"fh:quote:{ticker}",          1,  _fh, "/quote", {"symbol": ticker}), "quote"),
        _fetch_and_emit(ticker, asyncio.to_thread(cached, f"fh:tech:{ticker}",           1,  _technicals, ticker), "technicals"),
        _fetch_and_emit(ticker, asyncio.to_thread(cached, f"yf:fin:{ticker}",           12,  _yf_financials, ticker), "financials"),
        _fetch_and_emit(ticker, asyncio.to_thread(cached, f"av:EARNINGS:{ticker}",      24,  _av, "EARNINGS", {"symbol": ticker}), "earnings"),
        _fetch_and_emit(ticker, asyncio.to_thread(cached, f"av:OVERVIEW:{ticker}",      24,  _av, "OVERVIEW", {"symbol": ticker}), "av_overview"),
        _fetch_and_emit(ticker, asyncio.to_thread(cached, f"fh:insiders:{ticker}",      12,  _fh, "/stock/insider-transactions", {"symbol": ticker, "from": since}), "insiders"),
        _fetch_and_emit(ticker, asyncio.to_thread(cached, f"fh:news:{ticker}",           2,  _fh, "/company-news", {"symbol": ticker, "from": from_date, "to": to_date}), "news"),
        _fetch_and_emit(ticker, asyncio.to_thread(cached, f"sec:filing:{ticker}",       48,  _latest_filing, ticker), "sec_filing"),
        _fetch_and_emit(ticker, _get_macro(), "macro"),
        _fetch_and_emit(ticker, asyncio.to_thread(cached, f"fh:rec:{ticker}",           12,  _fh, "/stock/recommendation", {"symbol": ticker}), "rec_trends"),
        _fetch_and_emit(ticker, asyncio.to_thread(cached, f"fh:insider_sent:{ticker}",  12,  _fh, "/stock/insider-sentiment", {"symbol": ticker, "from": since_yr, "to": to_date}), "insider_sentiment"),
        _fetch_and_emit(ticker, asyncio.to_thread(cached, f"fh:usa_spending:{ticker}",  24,  _fh, "/stock/usa-spending", {"symbol": ticker, "from": since, "to": to_date}), "usa_spending"),
        _fetch_and_emit(ticker, asyncio.to_thread(cached, f"fh:earnings_cal:{ticker}",   6,  _fh, "/calendar/earnings", {"symbol": ticker, "from": to_date, "to": fwd_30}), "earnings_cal"),
        _fetch_and_emit(ticker, asyncio.to_thread(cached, f"fmp:estimates:{ticker}",     6,  _fmp_estimates, ticker), "fmp_estimates"),
    )

    # ── Metadata overrides (from cleaner.clean_ticker_batch) ─────────────────
    # Applied here — after raw data arrives but before any assembly — so all
    # downstream logic sees corrected values transparently.
    _meta = meta or {}
    if _meta.get("canonical_sector"):
        # Override both yf_fin["sector"] (used for yf_sector / archetype) and
        # profile_raw's industry field so the profile section picks it up.
        yf_fin["sector"] = _meta["canonical_sector"]
        if verbose:
            print(f"  [dossier] {ticker}: sector overridden to {_meta['canonical_sector']!r} (cleaner)")
    if _meta.get("is_adr"):
        # Force ADR flag regardless of what yfinance quoteType or share-ratio heuristic found
        yf_fin.setdefault("ratios", {})["adr_mismatch"] = True
        if verbose:
            print(f"  [dossier] {ticker}: ADR flag forced True (cleaner)")
    if _meta.get("financials_currency") and _meta.get("financials_currency") != "USD":
        currency = _meta["financials_currency"]
        yf_fin["financials_currency"] = currency
        if verbose:
            print(f"  [dossier] {ticker}: financials_currency={currency} (cleaner)")
        _fx_rate = _apply_fx_conversion(yf_fin, currency, verbose=verbose)
        if _fx_rate:
            yf_fin["fx_rate_to_usd"] = _fx_rate
        elif verbose:
            print(f"  [dossier] {ticker}: FX rate unavailable for {currency} — financial statements remain in local currency")
    elif yf_fin.get("financials_currency"):
        # yfinance-native FX detection (no cleaner required — covers single-ticker runs)
        currency = yf_fin["financials_currency"]
        if verbose:
            print(f"  [dossier] {ticker}: financials_currency={currency} (yfinance — applying FX conversion)")
        _fx_rate = _apply_fx_conversion(yf_fin, currency, verbose=verbose)
        if _fx_rate:
            yf_fin["fx_rate_to_usd"] = _fx_rate
        elif verbose:
            print(f"  [dossier] {ticker}: FX rate unavailable for {currency} — financial statements remain in local currency")

    # ── Profile ───────────────────────────────────────────────────────────────
    sector = profile_raw.get("finnhubIndustry") or yf_fin.get("sector") or "Unknown"
    dossier["profile"] = {
        "name":                 profile_raw.get("name") or yf_fin.get("company_name") or ticker,
        "sector":               sector,
        "yf_sector":            yf_fin.get("sector", ""),   # GICS sector (used for archetype classification)
        "industry":             yf_fin.get("industry", ""),
        "exchange":             profile_raw.get("exchange", ""),
        # For ADR stocks, Finnhub/yfinance return the local-exchange market cap in the
        # local currency. Compute from USD price × ADR shares instead (always USD-correct).
        "market_cap_bn":        round(
            (quote_raw.get("c")
             or (technicals.get("price") if isinstance(technicals, dict) else None)
             or yf_fin.get("previous_close") or 0)
            * (yf_fin.get("ratios", {}).get("shares_out") or 0) / 1e9
            if yf_fin.get("ratios", {}).get("adr_mismatch")
            else (profile_raw.get("marketCapitalization") or (yf_fin.get("market_cap") or 0) / 1e6 or 0) / 1000,
            2,
        ),
        "ipo_date":             profile_raw.get("ipo", ""),
        "employees":            profile_raw.get("employeeTotal", ""),
        "country":              profile_raw.get("country", ""),
        "website":              profile_raw.get("weburl", ""),
        "financials_currency":  yf_fin.get("financials_currency") or "USD",
        "fx_rate_to_usd":       yf_fin.get("fx_rate_to_usd"),   # non-None only for converted ADRs
    }
    dossier["cycle_type"] = _cycle_type(sector)

    # ── Quote ──â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    dossier["quote"] = {
        "price":      quote_raw.get("c") or (technicals.get("price") if isinstance(technicals, dict) else None),
        "change":     quote_raw.get("d"),
        "change_pct": quote_raw.get("dp") or _compute_change_pct(technicals, yf_fin),
        "high":       quote_raw.get("h"),
        "low":        quote_raw.get("l"),
        "open":       quote_raw.get("o"),
        "prev_close": quote_raw.get("pc") or yf_fin.get("previous_close"),
    }

    # â"€â"€ Technicals â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    dossier["technicals"] = technicals

    # â"€â"€ Financials â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    # FMP v3 API deprecated for keys created after Aug 2025 — yfinance is the sole source
    fmp_income, fmp_balance, fmp_cashflow = [], [], []

    yf_r = yf_fin.get("ratios", {})

    # AV OVERVIEW fields (returned as strings; missing/invalid → None via _safe_float)
    av_fwd_pe      = _safe_float(av_overview_raw.get("ForwardPE"))
    av_pb          = _safe_float(av_overview_raw.get("PriceToBookRatio"))
    av_trailing_pe = _safe_float(av_overview_raw.get("PERatio"))

    _pe_trailing = yf_r.get("pe") or av_trailing_pe
    _pe_forward_raw = yf_r.get("fwd_pe")

    # Prefer AV OVERVIEW forward PE — it's derived from analyst consensus estimates and is
    # correctly adjusted for ADR share structure (fixes the 6.25x vs 12.77x MFG discrepancy).
    if av_fwd_pe:
        _fwd_pe_clean: float | None = av_fwd_pe
        if verbose and _pe_forward_raw and _pe_forward_raw > 0:
            divergence = abs(av_fwd_pe - _pe_forward_raw) / max(av_fwd_pe, _pe_forward_raw)
            if divergence > 0.10:
                print(f"  [dossier] {ticker}: AV forward PE {av_fwd_pe}x overrides "
                      f"yfinance/FMP {_pe_forward_raw}x ({divergence:.0%} divergence)")
    else:
        # Fallback: yfinance/FMP forward PE with sanity check.
        # For ADR stocks (where yfinance PE data may mix local-currency EPS with USD
        # price), apply a strict 100% implied-growth threshold — mismatch is currency.
        # For domestic stocks, never null: high-growth names legitimately show 100-200%
        # implied EPS improvement between trailing and forward PE (NVDA, TSLA, PLTR).
        _is_adr = yf_r.get("adr_mismatch", False)
        _threshold = 1.0 if _is_adr else float("inf")
        _fwd_pe_clean = _pe_forward_raw
        if _pe_trailing and _pe_forward_raw and _pe_trailing > 0 and _pe_forward_raw > 0:
            _implied_growth = _pe_trailing / _pe_forward_raw - 1
            if _implied_growth > _threshold:
                _fwd_pe_clean = None
                if verbose:
                    print(f"  [dossier] {ticker}: forward PE ({_pe_forward_raw:.1f}x) vs trailing "
                          f"({_pe_trailing:.1f}x) implies {_implied_growth:.0%} YoY growth — "
                          f"likely ADR/FX mismatch, nulling fwd_pe")

    # Pre-compute growth/valuation derived metrics for ratios_ttm
    _income_list = yf_fin.get("income", [])
    _ttm_rev_growth_pct = None
    if len(_income_list) >= 2:
        _ri0 = _income_list[0].get("revenue")
        _ri1 = _income_list[1].get("revenue")
        if _ri0 and _ri1 and _ri1 != 0:
            _ttm_rev_growth_pct = round((_ri0 - _ri1) / abs(_ri1) * 100, 2)
    _op_margin_pct = yf_r.get("op_margin")  # already percentage from _pct()
    _r40 = round(_ttm_rev_growth_pct + _op_margin_pct, 2) if (
        _ttm_rev_growth_pct is not None and _op_margin_pct is not None
    ) else None

    _trailing_eps = yf_r.get("trailing_eps") or 0
    _forward_eps  = yf_r.get("forward_eps") or 0
    _implied_ntm_growth = _safe_div(_forward_eps - _trailing_eps, abs(_trailing_eps)) if _trailing_eps else None
    # Forward growth: FMP analyst consensus (live, 250 req/day free) is the primary source.
    # Falls back to yfinance t.earnings_estimate (daily Yahoo consensus), then to stale
    # yfinance info dict (earningsGrowth/revenueGrowth can lag 6-12 months).
    _fmp_est  = fmp_estimates_raw if isinstance(fmp_estimates_raw, dict) else {}
    _yf_est   = yf_fin.get("estimates", {})
    def _first_not_none(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    _fwd_earnings_growth = _first_not_none(
        _fmp_est.get("fwd_eps_growth"),
        _yf_est.get("fwd_eps_growth"),
        yf_fin.get("fwd_earnings_growth"),
    )
    _fwd_revenue_growth = _first_not_none(
        _fmp_est.get("fwd_rev_growth"),
        _yf_est.get("fwd_rev_growth"),
        yf_fin.get("fwd_revenue_growth"),
    )
    _eps_revision_momentum = _yf_est.get("eps_revision_momentum")  # yfinance eps_trend, no FMP equivalent on free tier

    # WACC — computed from existing data, zero new API calls.
    # Ke = risk_free + beta × 5.5% ERP (Damodaran US). Kd = 5% pre-tax (investment-grade default).
    _wacc = None
    _beta_w = yf_r.get("beta")
    _rf_w   = (macro.get("treasury_10y") or 0) / 100
    # yfinance returns debtToEquity as a percentage (e.g. 30.27 = 30.27% = 0.30x ratio)
    _de_w   = (yf_r.get("debt_equity") or 0) / 100
    if _beta_w is not None and _rf_w > 0 and 0 < _beta_w < 5:
        _ke = _rf_w + _beta_w * 0.055
        _dv = _de_w / (1 + _de_w) if _de_w > 0 else 0.0
        _wacc = round((1 - _dv) * _ke + _dv * 0.05 * 0.79, 4)

    dossier["financials"] = {
        "income":   yf_fin.get("income")   or fmp_income,
        "balance":  yf_fin.get("balance")  or fmp_balance,
        "cashflow": yf_fin.get("cashflow") or fmp_cashflow,
        "ratios_ttm": {
            "pe":            _pe_trailing,
            "fwd_pe":        _fwd_pe_clean,
            "pb":            yf_r.get("pb") or av_pb,
            "ps":            yf_r.get("ps"),
            "ev_ebitda":     yf_r.get("ev_ebitda"),
            "gross_margin":  yf_r.get("gross_margin"),
            "net_margin":    yf_r.get("net_margin"),
            "roe":           yf_r.get("roe"),
            "roic":          _compute_roic(yf_fin),
            "roa":           yf_r.get("roa"),
            "debt_equity":   yf_r.get("debt_equity"),
            "current_ratio": yf_r.get("current_ratio"),
            "fcf_per_share": yf_r.get("fcf_per_share"),
            "fcf":           yf_r.get("fcf"),
            "revenue_ttm":   yf_r.get("revenue_ttm"),
            "ebitda":        yf_r.get("ebitda"),
            "beta":          yf_r.get("beta"),
            "short_pct":     yf_r.get("short_pct"),
            "adr_mismatch":  yf_r.get("adr_mismatch", False),
            "shares_out":    yf_r.get("shares_out"),
            # Growth & valuation metrics
            "fwd_revenue_growth":      _fwd_revenue_growth,
            "fwd_earnings_growth":     _fwd_earnings_growth,
            "fwd_peg":                 _safe_div(_fwd_pe_clean, (_fwd_earnings_growth or 0) * 100),
            "fcf_yield":               _safe_div(yf_r.get("fcf"), yf_fin.get("market_cap")),
            "rule_of_40":              _r40,
            "implied_ntm_growth":      _implied_ntm_growth,
            "eps_acceleration":        _safe_sub(_fwd_earnings_growth, _implied_ntm_growth),
            "eps_revision_momentum":    _eps_revision_momentum,
            "wacc":                     _wacc,
            "fwd_eps_ntm":              _fmp_est.get("fwd_eps_ntm"),
            "fwd_rev_ntm":              _fmp_est.get("fwd_rev_ntm"),
            "num_analysts_eps":         _fmp_est.get("num_analysts_eps"),
            "est_eps_current_q":        _yf_est.get("est_eps_current_q"),
            "est_eps_current_q_growth": _yf_est.get("est_eps_current_q_growth"),
            "est_eps_next_q":           _yf_est.get("est_eps_next_q"),
            "est_eps_next_q_growth":    _yf_est.get("est_eps_next_q_growth"),
        },
    }

    # ── Valuation ──â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    yf_analyst = yf_fin.get("analyst", {})
    fmp_dcf_price = None   # FMP v3 dead
    fmp_targets: list = [] # FMP v3 dead

    fcf_val = yf_r.get("fcf")
    if fcf_val is None:
        cf_list = yf_fin.get("cashflow", [])
        if cf_list and cf_list[0].get("free_cash_flow") is not None:
            fcf_val = cf_list[0]["free_cash_flow"]
    shares_out = yf_r.get("shares_out")
    # Prefer GICS sector from yfinance over Finnhub's non-standard industry strings
    # (e.g. Finnhub returns "Semiconductors" for NVDA, not the GICS "Technology" that
    # SECTOR_GROWTH_CAP keys on — yf_fin["sector"] was already overridden by cleaner if needed)
    gics_sector = yf_fin.get("sector") or sector
    _balance = dossier["financials"].get("balance") or []
    _b0 = _balance[0] if _balance else {}
    _net_debt = (_b0.get("total_debt") or 0) - (_b0.get("cash") or 0)
    computed_dcf, dcf_assumptions = _dynamic_dcf(
        fcf_val,
        income=dossier["financials"].get("income", []),
        beta=yf_r.get("beta"),
        treasury_10y=macro.get("treasury_10y"),
        sector=gics_sector,
        shares_out=shares_out,
        fwd_revenue_growth=_fwd_revenue_growth,
        fwd_earnings_growth=_fwd_earnings_growth,
        net_debt=_net_debt,
    )

    dossier["valuation"] = {
        "dcf_price":        fmp_dcf_price,
        "dcf_iv_per_share": computed_dcf,
        "dcf_assumptions":  dcf_assumptions,
        "analyst_consensus": yf_analyst or {},
        "analyst_targets":   fmp_targets,
    }

    # â"€â"€ Earnings surprises â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    quarterly = earnings_raw.get("quarterlyEarnings", [])[:8]
    _surprises = []
    for e in quarterly:
        sp_raw = e.get("surprisePercentage")
        try:
            sp_f = float(sp_raw) if sp_raw is not None else None
        except (TypeError, ValueError):
            sp_f = None
        _surprises.append({
            "date":          e.get("fiscalDateEnding"),
            "reported_eps":  e.get("reportedEPS"),
            "estimated_eps": e.get("estimatedEPS"),
            "surprise_pct":  sp_raw,
            # >50% surprise often signals a one-time item, not durable earnings power
            "beat_quality":  ("LARGE_BEAT" if sp_f is not None and sp_f > 50 else
                              "BEAT"        if sp_f is not None and sp_f > 0 else
                              "MISS"        if sp_f is not None and sp_f < 0 else None),
        })
    dossier["earnings_surprises"] = _surprises

    # â"€â"€ Insider transactions â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    txns  = insiders_raw.get("data", []) if isinstance(insiders_raw, dict) else []
    buys  = [t for t in txns if t.get("transactionType") == "P - Purchase"]
    sells = [t for t in txns if t.get("transactionType") == "S - Sale"]

    # Cluster detection: 3+ insiders buying within a 14-day window
    def _has_cluster(transactions: list, window_days: int = 14) -> bool:
        from datetime import datetime
        dates = []
        for t in transactions:
            d = t.get("transactionDate", "")[:10]
            try:
                dates.append(datetime.strptime(d, "%Y-%m-%d"))
            except ValueError:
                continue
        if len(dates) < 3:
            return False
        dates.sort()
        for i in range(len(dates) - 2):
            if (dates[i + 2] - dates[i]).days <= window_days:
                return True
        return False

    # Significant transactions: value > $100K (uses transactionPrice from Finnhub)
    def _tx_value(t: dict) -> float:
        shares = abs(t.get("change", 0) or t.get("share", 0) or 0)
        price  = t.get("transactionPrice") or 0
        return shares * price

    significant_buys  = [t for t in buys  if _tx_value(t) >= 100_000]
    significant_sells = [t for t in sells if _tx_value(t) >= 100_000]
    buyer_names = list({t.get("name", "") for t in buys if t.get("name")})
    total_buy_usd  = round(sum(_tx_value(t) for t in buys))
    total_sell_usd = round(sum(_tx_value(t) for t in sells))

    dossier["insiders"] = {
        "buy_count":        len(buys),
        "sell_count":       len(sells),
        "net_shares":       sum(t.get("share", 0) for t in buys) - sum(t.get("share", 0) for t in sells),
        "cluster_buying":   _has_cluster(buys),
        "significant_buys": len(significant_buys),
        "significant_sells": len(significant_sells),
        "buyer_roles":      buyer_names[:5],
        "total_buy_usd":    total_buy_usd,
        "total_sell_usd":   total_sell_usd,
        "net_insider_usd":  total_buy_usd - total_sell_usd,
        "recent":           txns[:10],
    }

    # â"€â"€ News â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    fh_news = fh_news_raw[:10] if isinstance(fh_news_raw, list) else []
    dossier["news"] = {
        "finnhub": [
            {"date": n.get("datetime"), "headline": n.get("headline"),
             "source": n.get("source"), "summary": n.get("summary", "")}
            for n in fh_news
        ],
    }

    # â"€â"€ Analyst recommendation trends â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    _rec_list = rec_trends_raw if isinstance(rec_trends_raw, list) else []
    _rec_latest = _rec_list[0] if _rec_list else {}
    dossier["recommendation_trends"] = {
        "period":      _rec_latest.get("period"),
        "strong_buy":  _rec_latest.get("strongBuy"),
        "buy":         _rec_latest.get("buy"),
        "hold":        _rec_latest.get("hold"),
        "sell":        _rec_latest.get("sell"),
        "strong_sell": _rec_latest.get("strongSell"),
    } if _rec_latest else {}

    # â"€â"€ Insider sentiment (MSPR — Money-flow Smart Purchasing Ratio) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    _sent_data = insider_sent_raw.get("data", []) if isinstance(insider_sent_raw, dict) else []
    _sent_recent = _sent_data[-3:] if _sent_data else []
    dossier["insider_sentiment_mspr"] = {
        "monthly": [
            {
                "year":     s.get("year"),
                "month":    s.get("month"),
                "mspr":     s.get("mspr"),     # positive = net buying pressure
                "change":   s.get("change"),
                "purchase": s.get("purchase"),
                "sales":    s.get("sales"),
            }
            for s in _sent_recent
        ],
        "avg_mspr_3m": (round(sum(s.get("mspr") or 0 for s in _sent_recent) / len(_sent_recent), 4)
                        if _sent_recent else None),
    }

    # â"€â"€ Government contracts (USA Spending) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    # Finnhub returns {"data": [...], "symbol": "..."} for this endpoint
    _contracts = (usa_spending_raw.get("data", []) if isinstance(usa_spending_raw, dict)
                  else usa_spending_raw if isinstance(usa_spending_raw, list) else [])
    dossier["government_contracts"] = {
        "count":       len(_contracts),
        "total_value": sum(c.get("totalValue", 0) or 0 for c in _contracts),
        "recent":      _contracts[:5],
    }

    # â"€â"€ Upcoming earnings calendar â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    _ec_list = (earnings_cal_raw.get("earningsCalendar", [])
                if isinstance(earnings_cal_raw, dict) else [])
    _ec_upcoming = [e for e in _ec_list if e.get("epsActual") is None][:3]
    dossier["earnings_calendar"] = {
        "upcoming": [
            {
                "date":             e.get("date"),
                "hour":             e.get("hour"),   # "bmo" or "amc"
                "eps_estimate":     e.get("epsEstimate"),
                "revenue_estimate": e.get("revenueEstimate"),
                "quarter":          e.get("quarter"),
                "year":             e.get("year"),
            }
            for e in _ec_upcoming
        ],
    }

    # â"€â"€ SEC filing â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    dossier["sec_filing"] = sec_raw

    # â"€â"€ Macro (shared cache) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    dossier["macro"] = {**macro, "regime": _detect_regime(macro)}

    # ── Batch 2: peer comps (needs sector from profile) â€" 4 peers in parallel â"€
    # Try Finnhub /stock/peers first (actual industry peers), fall back to sector defaults
    SECTOR_PEERS = {
        "Technology":              ["NVDA", "AMD", "INTC", "QCOM", "AVGO"],
        "Energy":                  ["XOM", "CVX", "COP", "SLB", "EOG"],
        "Utilities":               ["NEE", "EXC", "D", "SO", "AEP"],
        "Financials":              ["JPM", "BAC", "GS", "MS", "C"],
        "Healthcare":              ["UNH", "CVS", "CI", "HUM", "ELV"],
        "Consumer Cyclical":       ["AMZN", "HD", "MCD", "NKE", "SBUX"],
        "Consumer Defensive":      ["WMT", "PG", "KO", "PEP", "COST"],
        "Industrials":             ["GE", "HON", "MMM", "CAT", "DE"],
        "Communication Services":  ["GOOGL", "META", "NFLX", "DIS", "CMCSA"],
        "Real Estate":             ["AMT", "PLD", "CCI", "EQIX", "PSA"],
        "Basic Materials":         ["LIN", "APD", "ECL", "SHW", "NEM"],
    }
    fh_peers_raw = await asyncio.to_thread(_fh, "/stock/peers", {"symbol": ticker})
    if isinstance(fh_peers_raw, list) and len(fh_peers_raw) > 1:
        peers = [p for p in fh_peers_raw if p != ticker and re.match(r'^[A-Z]{1,5}$', p)][:4]
    else:
        peers = [p for p in SECTOR_PEERS.get(sector, ["SPY", "QQQ", "DIA", "IWM"]) if p != ticker][:4]
    peer_results = await asyncio.gather(*[_fetch_and_emit(ticker, asyncio.to_thread(_fetch_peer, p), "peers") for p in peers])
    dossier["peer_comps"] = [r for r in peer_results if r]

    # Cross-validate key metrics across data sources
    try:
        from validator import validate_dossier
        dossier["data_quality"] = validate_dossier(dossier)
        if verbose and dossier["data_quality"]["warnings"]:
            print(f"  [dossier] {ticker}: data quality warnings: {dossier['data_quality']['warnings']}")
    except Exception as e:
        dossier["data_quality"] = {"warnings": [], "data_confidence": "HIGH"}
        if verbose:
            print(f"  [dossier] {ticker}: validator error (skipped): {e}")

    # ── Archetype-based fair value ──────────────────────────────────────────────
    try:
        from fair_value import compute_fair_values
        dossier["fair_values"] = compute_fair_values(dossier)
        if verbose:
            fv = dossier["fair_values"]
            arch = (fv.get("archetype") or {}).get("archetype", "?")
            cfv = fv.get("composite_fair_value")
            print(f"  [dossier] {ticker}: archetype={arch}, fair_value={cfv}")
    except Exception as e:
        dossier["fair_values"] = {"error": str(e), "composite_fair_value": None}
        if verbose:
            print(f"  [dossier] {ticker}: fair_value error (skipped): {e}")

    await emit_live(ticker, {"type": "DOSSIER_DONE"})
    if verbose:
        print(f"  [dossier] {ticker} done. {len(str(dossier)):,} chars.")
    return dossier
