"""Shared live-event emitter — used by both debate.py and dossier.py.

Batching strategy: accumulate events in memory per ticker and flush to KV
as a full replacement every FLUSH_EVERY events (or immediately on DONE).
This changes ~35 individual read+write KV pairs per run into ~7 write-only
operations, cutting live-event KV usage by ~5x on both reads and writes.

Opt-in via LIVE_EVENTS=1: a full scout+gems run debates ~30 tickers, each
emitting ~35-55 events → ~280 write-only KV ops/run. Six 4-hourly crons plus
the portfolio scan = ~1,700 writes/day, which blows the Cloudflare KV free-tier
1,000 writes/day limit (observed: the upload step then 403s with "KV put()
limit exceeded"). Nobody watches an unattended cron live (KV read analytics: 44
reads vs 1,819 writes over 2.5 days), so the stream is pure waste there. Only
user-triggered single-ticker analyses set LIVE_EVENTS=1 (see analyze.yml); every
scheduled run leaves it unset and emits nothing.
"""

import asyncio
import os
import time

import requests

_LIVE_URL    = os.getenv("SOVEREIGN_EYE_URL", "https://sovereign-eye.pages.dev")
_LIVE_SECRET = os.getenv("DD_UPLOAD_SECRET", "")

FLUSH_EVERY = 5  # write to KV every N events


def _live_on() -> bool:
    """Live streaming is opt-in. Requires both the upload secret (to reach KV)
    and LIVE_EVENTS truthy — read at call time so a per-invocation env var
    (LIVE_EVENTS=1 python main.py TICKER) takes effect without re-import."""
    if not _LIVE_SECRET:
        return False
    return os.getenv("LIVE_EVENTS", "0").strip().lower() in ("1", "true", "yes", "on")

# In-memory buffer: ticker → [events].  Accumulated for the lifetime of the
# process (single GitHub Actions run), so memory growth is bounded.
_buffer: dict[str, list] = {}


def _post_batch(url: str, payload: dict, headers: dict) -> None:
    """Sync POST — runs in a background thread. Failures silently swallowed."""
    try:
        requests.post(url, json=payload, headers=headers, timeout=8)
    except Exception:
        pass


async def emit_live(ticker: str, event: dict) -> None:
    """Buffer event and flush to sovereign-eye KV when batch threshold is reached.

    The KV endpoint now accepts a full event list (replaces the stored array)
    so each flush is 1 KV write with no preceding read, vs the previous design
    of 1 read + 1 write per single event.

    DONE events always trigger an immediate flush regardless of batch size.
    """
    if not _live_on():
        return
    event["ts"] = time.time()

    buf = _buffer.setdefault(ticker, [])
    buf.append({**event, "_idx": len(buf)})

    is_done = event.get("type") == "DONE"
    if is_done or len(buf) % FLUSH_EVERY == 0:
        snapshot = buf.copy()
        headers = {
            "Authorization": f"Bearer {_LIVE_SECRET}",
            "Content-Type": "application/json",
        }
        payload = {
            "ticker": ticker,
            "events": snapshot,
            "done": is_done,
        }
        asyncio.ensure_future(
            asyncio.to_thread(_post_batch, f"{_LIVE_URL}/api/dd/live", payload, headers)
        )
