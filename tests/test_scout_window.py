"""Tests for the scout rotating triage window + rotation ledger (scout.py)."""

import json

import scout
from scout import _select_window, _update_seen


def _mk(ticker, mcap=1.0, lenses=None):
    return {
        "ticker": ticker, "name": ticker, "mcap_b": mcap,
        "price": 1.0, "volume": 1, "lenses": lenses or [],
    }


def test_window_caps_size():
    cands = [_mk(f"T{i}") for i in range(50)]
    assert len(_select_window(cands, {}, 10)) == 10


def test_window_smaller_universe_returns_all():
    cands = [_mk(f"T{i}") for i in range(5)]
    assert len(_select_window(cands, {}, 600)) == 5


def test_empty_ledger_prefers_bigger_mcap():
    cands = [_mk("SMALL", mcap=0.5), _mk("BIG", mcap=100.0), _mk("MID", mcap=10.0)]
    window = _select_window(cands, {}, 2)
    assert [c["ticker"] for c in window] == ["BIG", "MID"]


def test_tag_reserve_slots_go_to_tagged():
    tagged   = [_mk(f"TAG{i}", mcap=0.1, lenses=["momentum"]) for i in range(5)]
    untagged = [_mk(f"UNT{i}", mcap=100.0) for i in range(20)]
    window = _select_window(tagged + untagged, {}, 8)
    # reserve = int(8 * 0.25) = 2 — first two slots are tagged despite tiny mcap
    assert all(c["lenses"] for c in window[:2])
    assert len(window) == 8


def test_more_tags_win_reserve_tiebreak():
    a = _mk("ONE", lenses=["momentum"])
    b = _mk("TWO", lenses=["momentum", "value", "breakout"])
    window = _select_window([a, b] + [_mk(f"U{i}") for i in range(10)], {}, 4)
    assert window[0]["ticker"] == "TWO"  # reserve=1, most-tagged first


def test_never_shown_sorts_before_recently_shown():
    seen = {"FAMOUS": {"last_shown": 1000.0, "shown": 3, "picked": 0}}
    cands = [_mk("FAMOUS", mcap=1000.0), _mk("OBSCURE", mcap=0.4)]
    window = _select_window(cands, seen, 1)
    assert window[0]["ticker"] == "OBSCURE"


def test_rotation_two_rounds_disjoint():
    cands = [_mk(f"T{i:02d}", mcap=50 - i) for i in range(30)]
    seen: dict = {}
    w1 = _select_window(cands, seen, 10)
    _update_seen(seen, w1, set(), now=1000.0)
    w2 = _select_window(cands, seen, 10)
    assert {c["ticker"] for c in w1}.isdisjoint({c["ticker"] for c in w2})


def test_update_seen_math():
    seen: dict = {}
    shown = [_mk("AAA"), _mk("BBB")]
    _update_seen(seen, shown, {"AAA"}, now=42.0)
    assert seen["AAA"] == {"last_shown": 42.0, "shown": 1, "picked": 1}
    assert seen["BBB"] == {"last_shown": 42.0, "shown": 1, "picked": 0}
    _update_seen(seen, [_mk("AAA")], set(), now=99.0)
    assert seen["AAA"]["shown"] == 2 and seen["AAA"]["last_shown"] == 99.0


def test_save_and_load_seen_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(scout, "SCOUT_SEEN_FILE", tmp_path / "scout_seen.json")
    seen = {"XYZ": {"last_shown": 1.0, "shown": 2, "picked": 1}}
    scout._save_seen(seen)
    assert scout._load_seen() == seen
    # atomic write leaves no temp file behind
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_seen_corrupt_returns_empty(tmp_path, monkeypatch):
    path = tmp_path / "scout_seen.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(scout, "SCOUT_SEEN_FILE", path)
    assert scout._load_seen() == {}


# ── Triage failure must not burn the window (504 zombie of 2026-06-11) ─────────

def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_triage_llm_failure_returns_none(monkeypatch):
    import llm

    async def boom(*a, **k):
        raise RuntimeError("504 DEADLINE_EXCEEDED")

    monkeypatch.setattr(llm, "call_gemini_async", boom)
    result = _run(scout._triage_with_gemma([_mk("AAA")], set(), verbose=False))
    assert result is None


def test_triage_parse_failure_returns_none(monkeypatch):
    import llm

    async def garbage(*a, **k):
        return "not json at all"

    monkeypatch.setattr(llm, "call_gemini_async", garbage)
    monkeypatch.setattr(llm, "extract_json", lambda t: (_ for _ in ()).throw(ValueError("no JSON")))
    result = _run(scout._triage_with_gemma([_mk("AAA")], set(), verbose=False))
    assert result is None


def test_triage_legit_empty_returns_list(monkeypatch):
    import llm

    async def empty_picks(*a, **k):
        return '{"picks": []}'

    monkeypatch.setattr(llm, "call_gemini_async", empty_picks)
    monkeypatch.setattr(llm, "extract_json", lambda t: {"picks": []})
    result = _run(scout._triage_with_gemma([_mk("AAA")], set(), verbose=False))
    assert result == []


def test_triage_valid_picks_returned(monkeypatch):
    import llm

    async def ok(*a, **k):
        return "json"

    monkeypatch.setattr(llm, "call_gemini_async", ok)
    monkeypatch.setattr(
        llm, "extract_json",
        lambda t: {"picks": [{"ticker": "AAA", "reason": "x"}, {"ticker": "ZZZ", "reason": "y"}]},
    )
    result = _run(scout._triage_with_gemma([_mk("AAA")], set(), verbose=False, debate_count=5))
    assert [p["ticker"] for p in result] == ["AAA"]  # ZZZ not in window → dropped
