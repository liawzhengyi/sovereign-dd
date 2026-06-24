"""Sovereign DD — Consensus-Based Stock Rating Multi-Agent Framework.

Usage:
    python main.py TICKER                          # single ticker
    python main.py TICKER --save                   # single ticker, save JSON
    python main.py --portfolio --save --notify     # all portfolio tickers + Telegram summary
    python main.py --scout --save --notify         # scout mode + Telegram BUY alerts
    python main.py --gems [--save] [--notify]      # gems pipeline + Telegram BUY alerts
    python main.py --scout --gems [--save] [--notify]  # scout + gems concurrently

Portfolio tickers are read from the PORTFOLIO_TICKERS env var (comma-separated).
"""

import asyncio
import json
import os
import sys
os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from datetime import datetime, timezone
from pathlib import Path

from dossier import build as build_dossier
from debate import run as run_debate
from report import render, console


# ── Helpers ───────────────────────────────────────────────────────────────────

def _env_tickers() -> list[str]:
    raw = os.getenv("PORTFOLIO_TICKERS", "")
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _live_tickers() -> list[str]:
    """Fetch the live holdings list from Sovereign Eye KV (positions:daryl).

    This is the source of truth — what the user actually holds, edited from the
    dashboard. Returns [] (so the caller falls back to PORTFOLIO_TICKERS) if the
    endpoint is unconfigured, unreachable, or empty. Never fatal.
    """
    base   = os.getenv("SOVEREIGN_EYE_URL", "").rstrip("/")
    secret = os.getenv("DD_UPLOAD_SECRET", "")
    if not base or not secret:
        return []
    try:
        import requests
        r = requests.get(
            f"{base}/api/dd/positions",
            headers={"Authorization": f"Bearer {secret}"},
            timeout=15,
        )
        if not r.ok:
            console.print(f"[dim]  [portfolio] live positions HTTP {r.status_code} — using PORTFOLIO_TICKERS[/dim]")
            return []
        tickers = r.json().get("tickers", [])
        return [str(t).strip().upper() for t in tickers if str(t).strip()]
    except Exception as e:
        console.print(f"[dim]  [portfolio] live positions fetch failed ({e}) — using PORTFOLIO_TICKERS[/dim]")
        return []


def _portfolio_tickers() -> list[str]:
    """Live dashboard holdings, falling back to the PORTFOLIO_TICKERS env var."""
    live = _live_tickers()
    if live:
        console.print(f"[dim]  [portfolio] using {len(live)} live holdings from dashboard[/dim]")
        return live
    tickers = _env_tickers()
    if tickers:
        console.print("[dim]  [portfolio] live positions unavailable — using PORTFOLIO_TICKERS env[/dim]")
        return tickers
    console.print("[red]No live positions and PORTFOLIO_TICKERS env var is empty or not set[/red]")
    sys.exit(1)


def _save_result(ticker: str, result: dict, dossier: dict, subdir: str = "") -> Path:
    base = Path("output") / subdir if subdir else Path("output")
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = base / f"{ticker}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"result": result, "dossier": dossier}, f, indent=2, default=str)
    return out_path


# ── Single ticker ─────────────────────────────────────────────────────────────

async def _run_single(ticker: str, save: bool = False, is_holding: bool = False):
    from cleaner import clean_ticker_batch
    # A dashboard-triggered re-run of a CURRENT holding must grade on the hold
    # ladder (ADD/HOLD/TRIM/EXIT), not the entry ladder — otherwise the same
    # stock shows e.g. SELL here but TRIM on the portfolio screen.
    if not is_holding and ticker in set(_live_tickers()):
        is_holding = True
        console.print(f"[dim]  [portfolio] {ticker} is a current holding — switching to hold-mode[/dim]")
    mode_tag = " (hold-mode)" if is_holding else ""
    console.rule(f"[bold blue]Sovereign DD — {ticker}{mode_tag}[/bold blue]")
    try:
        meta_map = await asyncio.wait_for(clean_ticker_batch([ticker]), timeout=30.0)
    except asyncio.TimeoutError:
        console.print("[dim]  [cleaner] timeout — proceeding without metadata override[/dim]")
        meta_map = {}
    dossier = await build_dossier(ticker, verbose=True, meta=meta_map.get(ticker, {}))
    result  = await run_debate(ticker, dossier, verbose=True, is_holding=is_holding)
    console.rule("[bold]FINAL REPORT[/bold]")
    render(result, dossier)
    if save:
        out_path = _save_result(ticker, result, dossier)
        console.print(f"[dim]Saved to {out_path}[/dim]")
    return result, dossier


# ── Batch mode — multiple tickers, shared key pool, bounded concurrency ───────

async def _run_batch(tickers: list[str], save: bool = False):
    """Run multiple tickers concurrently within a single process.

    All asyncio tasks share the same llm.py key-rotation state
    (_key_cooldowns, _key_daily_exhausted, _api_semaphore), so the load
    balancer can actually coordinate across concurrent requests instead of
    each ticker independently stampeding the same keys.

    Concurrency is capped at the number of API keys so we never send more
    simultaneous requests than we have keys to serve them.
    """
    from llm import _keys as _api_keys
    from cleaner import clean_ticker_batch
    tickers   = [t.upper() for t in tickers]
    # Cap at 4 concurrent debates regardless of key count — each debate fires
    # 5 agent calls simultaneously, so 4 debates = up to 20 in-flight API calls
    # across 9 keys ≈ 2–3 calls/key/burst, well under the 15 RPM per-key limit.
    n_keys    = len(_api_keys)
    max_concurrent = min(4, n_keys)
    sem       = asyncio.Semaphore(max_concurrent)

    console.rule(
        f"[bold blue]Sovereign DD — Batch ({len(tickers)} tickers, "
        f"≤{max_concurrent} concurrent)[/bold blue]"
    )

    try:
        batch_meta = await asyncio.wait_for(clean_ticker_batch(tickers), timeout=45.0)
    except asyncio.TimeoutError:
        console.print("[dim]  [cleaner] timeout — proceeding without metadata override[/dim]")
        batch_meta = {}
    if batch_meta:
        console.print(f"[dim]Cleaner resolved metadata for: {', '.join(batch_meta.keys())}[/dim]")

    async def _one(ticker: str):
        async with sem:
            try:
                dossier = await build_dossier(ticker, verbose=False, meta=batch_meta.get(ticker, {}))
                result  = await run_debate(ticker, dossier, verbose=False)
                render(result, dossier)
                if save:
                    out_path = _save_result(ticker, result, dossier)
                    console.print(f"[dim]{ticker}: saved to {out_path}[/dim]")
                return ticker, result, dossier
            except Exception as exc:
                console.print(f"[red]{ticker}: failed — {exc}[/red]")
                return ticker, None, None

    results = await asyncio.gather(*[_one(t) for t in tickers])

    console.rule("[bold]Batch complete[/bold]")
    for ticker, result, _ in results:
        if result:
            score = result.get("consensus_score", "?")
            grade = result.get("consensus_grade", "?")
            console.print(f"  {ticker:<6}  score={score}  grade={grade}")
        else:
            console.print(f"  [red]{ticker:<6}  FAILED[/red]")


# ── Portfolio mode — all tickers in parallel (max 3 concurrent) ───────────────

async def _run_portfolio(save: bool = False, notify: bool = False):
    from cleaner import clean_ticker_batch
    tickers = _portfolio_tickers()
    console.rule(
        f"[bold blue]Sovereign DD — Portfolio scan "
        f"({len(tickers)} tickers, 3 concurrent)[/bold blue]"
    )

    try:
        portfolio_meta = await asyncio.wait_for(clean_ticker_batch(tickers), timeout=45.0)
    except asyncio.TimeoutError:
        console.print("[dim]  [cleaner] timeout — proceeding without metadata override[/dim]")
        portfolio_meta = {}
    if portfolio_meta:
        console.print(f"[dim]Cleaner resolved metadata for: {', '.join(portfolio_meta.keys())}[/dim]")

    sem = asyncio.Semaphore(3)

    async def _analyze_one(ticker: str) -> dict:
        async with sem:
            try:
                console.rule(f"[blue]{ticker}[/blue]")
                dossier = await build_dossier(ticker, verbose=True, meta=portfolio_meta.get(ticker, {}))
                # Portfolio screen is by definition reviewing current holdings —
                # frame the debate as hold-vs-trim-vs-exit, not as a fresh entry.
                result  = await run_debate(ticker, dossier, verbose=True, is_holding=True)
                console.rule(f"[bold]{ticker} FINAL REPORT[/bold]")
                render(result, dossier)
                if save:
                    out_path = _save_result(ticker, result, dossier)
                    console.print(f"[dim]Saved to {out_path}[/dim]")
                return {
                    "ticker": ticker,
                    "score":  result.get("consensus_score", 0),
                    "grade":  result.get("consensus_grade", "?"),
                }
            except Exception as e:
                console.print(f"[red]  {ticker} failed: {e}[/red]")
                return {"ticker": ticker, "score": 0, "grade": "ERROR"}

    summaries = list(await asyncio.gather(*[_analyze_one(t) for t in tickers]))

    if notify and summaries:
        from notify import alert_portfolio_summary
        alert_portfolio_summary(summaries)
        console.print("[dim]Portfolio summary sent to Telegram[/dim]")

    return summaries


# ── Scout mode ────────────────────────────────────────────────────────────────

async def _run_gems(save: bool = False, notify: bool = False):
    from gems import run_gems
    discoveries = await run_gems(verbose=True)

    if notify:
        from notify import alert_buy_signal, alert_scout_summary, alert_dd_result, alert_watchlist
        confirmed = [d for d in discoveries if d.get("confirmed", True)]
        review    = [d for d in discoveries if not d.get("confirmed", True)]
        for d in confirmed:
            alert_buy_signal(d)
            if d.get("output_file"):
                try:
                    with open(d["output_file"], encoding="utf-8") as f:
                        data = json.load(f)
                    alert_dd_result(data["result"])
                except Exception as e:
                    console.print(f"[dim]  [notify] DD detail unavailable for {d.get('ticker','?')}: {e}[/dim]")
        for d in review:
            alert_watchlist(d)
        if confirmed:
            alert_scout_summary(confirmed)
        console.print(f"[dim]Gems alerts sent ({len(confirmed)} confirmed, {len(review)} under review)[/dim]")

    return discoveries


async def _run_scout(save: bool = False, notify: bool = False):
    from scout import run_scout, _load_notified, _save_notified, _recently_notified
    # Exclude current holdings from scout picks. Live positions are the source of
    # truth; the env var is only the fallback — don't gate the exclusion on it.
    portfolio = _live_tickers() or _env_tickers()
    discoveries = await run_scout(portfolio=portfolio, verbose=True)

    if notify:
        from notify import alert_scout_summary, alert_buy_signal, alert_dd_result, alert_watchlist

        notified   = _load_notified()
        suppressed = _recently_notified(notified)
        alerted    = []
        # Under-review BUYs go to the watchlist. We do NOT record them in the
        # notified ledger: the 48h scout_history cooldown already prevents a
        # re-debate (and so re-alert) within that window, and recording them
        # would suppress a legitimate CONFIRMED alert if the ticker later clears.
        review     = [d for d in discoveries if not d.get("confirmed", True)]

        for d in discoveries:
            if not d.get("confirmed", True):
                continue  # routed to the watchlist loop below
            ticker = d["ticker"]
            if ticker in suppressed:
                console.print(
                    f"[dim]  [notify] {ticker} already alerted within "
                    f"{os.getenv('SCOUT_NOTIFY_COOLDOWN_HOURS', '168')}h cooldown — skipping[/dim]"
                )
                continue
            alert_buy_signal(d)
            if d.get("output_file"):
                try:
                    with open(d["output_file"], encoding="utf-8") as f:
                        data = json.load(f)
                    alert_dd_result(data["result"])
                except Exception as e:
                    console.print(f"[dim]  [notify] DD detail unavailable for {d.get('ticker','?')}: {e}[/dim]")
            notified[ticker] = {
                "ts":    datetime.now(timezone.utc).timestamp(),
                "score": d["score"],
                "grade": d["grade"],
            }
            _save_notified(notified)
            alerted.append(d)

        for d in review:
            alert_watchlist(d)

        alert_scout_summary(alerted)
        console.print(f"[dim]Scout alerts sent ({len(alerted)} confirmed BUY(s), "
                      f"{len(review)} under review)[/dim]")

    return discoveries


# ── Entry point ───────────────────────────────────────────────────────────────

async def _main():
    args = sys.argv[1:]

    save           = "--save"      in args
    notify         = "--notify"    in args
    portfolio_mode = "--portfolio" in args
    scout_mode     = "--scout"     in args
    gems_mode      = "--gems"      in args
    hold_mode      = "--hold"      in args  # single-ticker test of hold-mode scoring
    positional     = [a for a in args if not a.startswith("--")]

    if portfolio_mode and scout_mode and gems_mode:
        await asyncio.gather(
            _run_portfolio(save=save, notify=notify),
            _run_scout(save=save, notify=notify),
            _run_gems(save=save, notify=notify),
        )
    elif scout_mode and gems_mode:
        await asyncio.gather(
            _run_scout(save=save, notify=notify),
            _run_gems(save=save, notify=notify),
        )
    elif portfolio_mode and scout_mode:
        await asyncio.gather(
            _run_portfolio(save=save, notify=notify),
            _run_scout(save=save, notify=notify),
        )
    elif portfolio_mode and gems_mode:
        await asyncio.gather(
            _run_portfolio(save=save, notify=notify),
            _run_gems(save=save, notify=notify),
        )
    elif portfolio_mode:
        await _run_portfolio(save=save, notify=notify)
    elif scout_mode:
        await _run_scout(save=save, notify=notify)
    elif gems_mode:
        await _run_gems(save=save, notify=notify)
    elif len(positional) == 1:
        await _run_single(positional[0].upper(), save=save, is_holding=hold_mode)
    elif len(positional) > 1:
        await _run_batch(positional, save=save)
    else:
        console.print(
            "[red]Usage:[/red]\n"
            "  python main.py TICKER [--save]\n"
            "  python main.py TICKER1 TICKER2 ... [--save]  # concurrent batch, shared key pool\n"
            "  python main.py --portfolio [--save] [--notify]\n"
            "  python main.py --scout [--save] [--notify]\n"
            "  python main.py --gems [--save] [--notify]\n"
            "  python main.py --scout --gems [--save] [--notify]\n"
            "  python main.py --portfolio --gems [--save] [--notify]\n"
            "  python main.py --portfolio --scout --gems [--save] [--notify]\n"
            "  python main.py --portfolio --scout [--save] [--notify]"
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())
    try:
        from cache import print_cache_stats
        print_cache_stats()
    except Exception:
        pass
