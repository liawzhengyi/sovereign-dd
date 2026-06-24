"""Tests for the live-event KV-write gate (live_events.py).

Regression guard for the 2026-06-15 incident: unattended scout/gems crons
streamed ~280 write-only KV ops/run (~1,700/day across 6 crons + portfolio),
blowing the Cloudflare KV free-tier 1,000 writes/day limit and 403-ing the
upload step. Live streaming is now opt-in via LIVE_EVENTS — only user-watched
single-ticker analyses enable it.
"""

import asyncio

import live_events
from live_events import emit_live, _live_on


def _run(coro):
    return asyncio.run(coro)


def _patch(monkeypatch, *, secret, live_events_env):
    monkeypatch.setattr(live_events, "_LIVE_SECRET", secret)
    if live_events_env is None:
        monkeypatch.delenv("LIVE_EVENTS", raising=False)
    else:
        monkeypatch.setenv("LIVE_EVENTS", live_events_env)


# ── _live_on truth table ─────────────────────────────────────────────────────

def test_off_when_env_unset(monkeypatch):
    _patch(monkeypatch, secret="sek", live_events_env=None)
    assert _live_on() is False


def test_off_when_env_zero(monkeypatch):
    _patch(monkeypatch, secret="sek", live_events_env="0")
    assert _live_on() is False


def test_off_when_secret_missing_even_if_enabled(monkeypatch):
    # No upload secret → can't reach KV regardless of the flag.
    _patch(monkeypatch, secret="", live_events_env="1")
    assert _live_on() is False


def test_on_with_truthy_values(monkeypatch):
    for val in ("1", "true", "TRUE", "yes", "on", " On "):
        _patch(monkeypatch, secret="sek", live_events_env=val)
        assert _live_on() is True, val


# ── emit_live respects the gate (no network when off) ────────────────────────

def _spy_post(monkeypatch):
    calls = []
    # to_thread would actually run _post_batch in a thread; stub it so a leak
    # is observable without hitting the network.
    async def fake_to_thread(fn, *a, **k):
        calls.append((fn, a, k))
    monkeypatch.setattr(live_events.asyncio, "to_thread", fake_to_thread)
    return calls


def test_emit_does_not_post_when_off(monkeypatch):
    _patch(monkeypatch, secret="sek", live_events_env="0")
    live_events._buffer.clear()
    calls = _spy_post(monkeypatch)
    # Emit a full flush-worth of events plus a DONE — none should post.
    for i in range(FLUSH := 6):
        _run(emit_live("AAA", {"type": "R1_SCORE", "i": i}))
    _run(emit_live("AAA", {"type": "DONE"}))
    assert calls == []
    # Buffer is never touched when gated off.
    assert "AAA" not in live_events._buffer


def test_emit_posts_on_flush_threshold_when_on(monkeypatch):
    _patch(monkeypatch, secret="sek", live_events_env="1")
    live_events._buffer.clear()
    calls = _spy_post(monkeypatch)
    # FLUSH_EVERY events → exactly one flush.
    for i in range(live_events.FLUSH_EVERY):
        _run(emit_live("BBB", {"type": "R1_SCORE", "i": i}))
    assert len(calls) == 1


def test_emit_posts_immediately_on_done_when_on(monkeypatch):
    _patch(monkeypatch, secret="sek", live_events_env="1")
    live_events._buffer.clear()
    calls = _spy_post(monkeypatch)
    _run(emit_live("CCC", {"type": "START"}))   # below threshold, no flush
    assert calls == []
    _run(emit_live("CCC", {"type": "DONE"}))    # DONE forces a flush
    assert len(calls) == 1
