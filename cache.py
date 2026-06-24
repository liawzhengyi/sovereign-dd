"""Supabase-backed API response cache for dossier.py fetches.

Gracefully degrades to direct API calls when SUPABASE_URL / SUPABASE_KEY
are not set — zero behaviour change for unconfigured environments.

Zero new dependencies: uses requests (already in requirements.txt).

Usage:
    from cache import cached

    result = cached("fh:profile:MSFT", ttl_hours=72, fn=_fh,
                    "/stock/profile2", {"symbol": "MSFT"})
"""

import os
import threading
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

# Hit/miss metrics (approximate; call print_cache_stats() at end of a run).
_HITS = 0
_MISSES = 0
_stats_lock = threading.Lock()


def cache_stats() -> tuple[int, int]:
    """Return (hits, misses) so far this process."""
    return _HITS, _MISSES


def print_cache_stats() -> None:
    total = _HITS + _MISSES
    rate = (100 * _HITS / total) if total else 0
    print(f"  [cache] {_HITS} hit / {_MISSES} miss ({rate:.0f}% hit rate)")

_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
_KEY = os.getenv("SUPABASE_KEY", "")
_ENABLED = bool(_URL and _KEY)

_HEADERS = {
    "apikey":        _KEY,
    "Authorization": f"Bearer {_KEY}",
    "Content-Type":  "application/json",
}


def _sb_get(key: str, ttl_hours: float):
    """Return cached payload if a fresh row exists, else None."""
    if not _ENABLED:
        return None
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=ttl_hours)).isoformat()
    try:
        r = requests.get(
            f"{_URL}/rest/v1/api_cache",
            params={
                "cache_key": f"eq.{key}",
                "fetched_at": f"gte.{cutoff}",
                "select":    "payload",
                "limit":     "1",
            },
            headers=_HEADERS,
            timeout=5,
        )
        if r.ok:
            rows = r.json()
            if rows:
                return rows[0]["payload"]
    except Exception:
        pass
    return None


def _sb_set(key: str, payload) -> None:
    """Upsert payload into cache (merge-duplicates on primary key)."""
    if not _ENABLED:
        return
    try:
        requests.post(
            f"{_URL}/rest/v1/api_cache",
            json={
                "cache_key":  key,
                "payload":    payload,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
            headers={**_HEADERS, "Prefer": "resolution=merge-duplicates"},
            timeout=5,
        )
    except Exception:
        pass


def cached(key: str, ttl_hours: float, fn, *args, **kwargs):
    """Return a cached result or call fn(*args, **kwargs) and cache the result.

    Thread-safe for reads; concurrent cache misses may double-fetch (harmless
    for idempotent API reads — last write wins on upsert). Empty / falsy
    responses (rate-limit messages, network errors) are not cached so the
    next call retries the real API.
    """
    global _HITS, _MISSES
    hit = _sb_get(key, ttl_hours)
    if hit is not None:
        with _stats_lock:
            _HITS += 1
        return hit
    with _stats_lock:
        _MISSES += 1
    result = fn(*args, **kwargs)
    if result:
        _sb_set(key, result)
    return result
