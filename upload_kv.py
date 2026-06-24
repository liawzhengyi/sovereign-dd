"""Upload sovereign-dd output JSONs to Sovereign Eye via the /api/dd/upload endpoint."""

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from scout import BUY_THRESHOLD

# Only upload scout files written in the last 2 hours — prevents re-uploading
# the entire accumulated history on every run (each run adds 12 files; without
# this filter the puts-per-upload grows unboundedly and blows the KV free tier).
SCOUT_UPLOAD_WINDOW_SECS = 2 * 3600

# Filenames to skip in output/ — not ticker results
_SKIP_FILENAMES = {"scout_history.json", "scout_notified.json", "gems_history.json", "scout_seen.json"}

UPLOAD_URL    = os.getenv("SOVEREIGN_EYE_URL", "https://sovereign-eye.pages.dev")
UPLOAD_SECRET = os.getenv("DD_UPLOAD_SECRET", "")
SUPABASE_URL  = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY", "")

HEADERS = {
    "Authorization": f"Bearer {UPLOAD_SECRET}",
    "Content-Type": "application/json",
}


def _extract_archetype(dossier: dict) -> str | None:
    fv = dossier.get("fair_values", {})
    arch_obj = fv.get("archetype")
    if isinstance(arch_obj, dict):
        return arch_obj.get("archetype")
    return None


def _supabase_insert(table: str, rows: list[dict]) -> None:
    """POST rows to a Supabase table via REST API. Silently skips if not configured."""
    if not SUPABASE_URL or not SUPABASE_KEY or not rows:
        return
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal,resolution=ignore-duplicates",
            },
            json=rows,
            timeout=15,
        )
        if resp.status_code not in (200, 201):
            print(f"  [supabase] {table} insert returned {resp.status_code}: {resp.text[:120]}")
    except Exception as e:
        print(f"  [supabase] {table} insert failed: {e}")


def _sanitize(obj):
    """Recursively replace NaN/inf floats with None so requests can serialize the payload.

    Uses a try/except around math.isnan so numpy.float64 (and other numeric subtypes
    that aren't isinstance float in numpy 2.x) are handled correctly.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    try:
        if math.isnan(obj) or math.isinf(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [upload] Could not read {path.name}: {e}")
        return None


def _scout_card(ticker: str, result: dict, meta: dict | None = None) -> dict:
    """Build the dd:scouts card shape from a debate result. Shared by scout
    collection and portfolio/triggered-analysis results so a card always reflects
    the latest run regardless of which pipeline produced it."""
    meta = meta or {}
    return {
        "ticker":            ticker,
        "score":             round(result.get("consensus_score", 0), 2),
        "grade":             result.get("consensus_grade", "?"),
        "conf":              result.get("confidence", ""),
        "thesis":            result.get("majority_thesis", "")[:200],
        "key_swing":         result.get("key_swing_factor", "")[:150],
        "analyzed_at":       result.get("built_at", ""),
        "catalyst":          result.get("catalyst", ""),
        "asymmetry_ratio":   result.get("asymmetry_ratio", ""),
        "rr":                (result.get("risk_reward") or {}).get("rr_ratio"),
        "risk":              (result.get("risk_reward") or {}).get("risk_tier"),
        "banger":            result.get("banger", {}),
        "position_guidance": result.get("position_guidance", {}),
        "cycle_position":    result.get("cycle_position", {}),
        "matched_filters":   meta.get("matched_filters", []),
        "path":              meta.get("path", "A"),
    }


def collect_portfolio_results(output_dir: Path) -> tuple[list, dict, list, list]:
    """Collect output/*.json ticker result files.

    Returns (results_list, index_dict, scout_cards, reconcile_remove):
      - scout_cards: portfolio/triggered tickers scoring >= BUY_THRESHOLD, as scout
        cards, so a manual `analyze` trigger (or a portfolio run) refreshes the
        ticker's Scout card with the latest run.
      - reconcile_remove: tickers analyzed this run that scored BELOW threshold —
        their stale Scout/Gems card (if any) is removed so Scout stays a clean
        >= threshold board.
    """
    # Dedupe by ticker — keep only the newest file per ticker (filename timestamp
    # sorts ascending, so the last one wins). Prevents writing the same dd:TICKER key
    # multiple times per upload (burns the free-tier KV write budget) and avoids
    # loading every historical file into memory.
    latest: dict[str, Path] = {}
    for path in sorted(output_dir.glob("*.json")):
        if path.name in _SKIP_FILENAMES:
            continue
        fname_ticker = path.stem.split("_")[0].upper()
        latest[fname_ticker] = path

    results = []
    index = {}
    scout_cards = []
    reconcile_remove = []
    for _, path in sorted(latest.items()):
        data = load_json(path)
        if not data:
            continue
        result = data.get("result", {})
        if not result.get("ticker"):
            continue  # skip non-ticker files (e.g. history files)
        ticker = result["ticker"]
        results.append({"key": f"dd:{ticker}", "value": data})
        index[ticker] = {
            "score":   result.get("consensus_score", 0),
            "grade":   result.get("consensus_grade", "?"),
            "conf":    result.get("confidence", ""),
            "updated": result.get("built_at", ""),
            "loops":   result.get("loops_run", 0),
            "spread":  result.get("score_spread", 0),
            "rr":      (result.get("risk_reward") or {}).get("rr_ratio"),
            "risk":    (result.get("risk_reward") or {}).get("risk_tier"),
        }
        # Reflect this run on the Scout board: >= threshold refreshes/creates the
        # card; below threshold removes any stale card for the ticker.
        # Hold-mode portfolio runs never populate the Scout board — that list is for
        # new-buy signals on tickers we DON'T already own. We still reconcile-remove
        # stale scout cards for held tickers so the board stays clean.
        if result.get("mode") == "hold":
            reconcile_remove.append(ticker)
        elif result.get("consensus_score", 0) >= BUY_THRESHOLD and result.get("confirmed", True):
            scout_cards.append(_scout_card(ticker, result, data.get("meta", {})))
        else:
            # Below threshold OR failed the confirmation gate — keep off the Scout board.
            reconcile_remove.append(ticker)

    return results, index, scout_cards, reconcile_remove


def collect_scout_results(scout_dir: Path) -> list:
    """Collect BUY-signal scout files written in the last 2h. Returns discoveries_list only.

    Individual scout:TICKER KV keys are intentionally NOT written — nothing in the dashboard
    reads them and they consume ~12 of the 1,000 free-tier KV writes per run.
    Only dd:scouts (the accumulated BUY list) is written via the payload 'scouts' field.
    """
    if not scout_dir.exists():
        return []

    cutoff = time.time() - SCOUT_UPLOAD_WINDOW_SECS
    discoveries = []

    # Deduplicate by ticker — keep only the newest file per ticker
    latest: dict[str, Path] = {}
    for path in scout_dir.glob("*.json"):
        if path.stat().st_mtime < cutoff:
            continue
        ticker = path.stem.split("_")[0].upper()
        if ticker not in latest or path.stat().st_mtime > latest[ticker].stat().st_mtime:
            latest[ticker] = path

    for ticker, path in sorted(latest.items()):
        data = load_json(path)
        if not data:
            continue
        result = data.get("result", {})
        score  = result.get("consensus_score", 0)
        grade  = result.get("consensus_grade", "?")

        if score >= BUY_THRESHOLD and result.get("confirmed", True):
            discoveries.append(_scout_card(ticker, result, data.get("meta", {})))

    return discoveries


GEMS_UPLOAD_WINDOW_SECS = 26 * 3600  # gems runs once/day — look back 26h


def collect_gems_results(gems_dir: Path) -> list:
    """Collect BUY-signal gems files written in the last 26h.

    Same dedup-by-ticker logic as collect_scout_results — keep newest file per ticker.
    Returns a list of discovery dicts for the 'gems' field in the upload payload.
    """
    if not gems_dir.exists():
        return []

    cutoff = time.time() - GEMS_UPLOAD_WINDOW_SECS

    # Deduplicate by ticker — keep only the newest file per ticker
    latest: dict[str, Path] = {}
    for path in gems_dir.glob("*.json"):
        if path.stat().st_mtime < cutoff:
            continue
        ticker = path.stem.split("_")[0].upper()
        if ticker not in latest or path.stat().st_mtime > latest[ticker].stat().st_mtime:
            latest[ticker] = path

    discoveries = []
    for ticker, path in sorted(latest.items()):
        data = load_json(path)
        if not data:
            continue
        result = data.get("result", {})
        score  = result.get("consensus_score", 0)
        grade  = result.get("consensus_grade", "?")

        if score >= BUY_THRESHOLD and result.get("confirmed", True):
            discoveries.append({
                "ticker":            ticker,
                "score":             round(score, 2),
                "grade":             grade,
                "conf":              result.get("confidence", ""),
                "thesis":            result.get("majority_thesis", "")[:200],
                "key_swing":         result.get("key_swing_factor", "")[:150],
                "analyzed_at":       result.get("built_at", ""),
                "catalyst":          result.get("catalyst", ""),
                "asymmetry_ratio":   result.get("asymmetry_ratio", ""),
                "rr":                (result.get("risk_reward") or {}).get("rr_ratio"),
                "risk":              (result.get("risk_reward") or {}).get("risk_tier"),
                "banger":            result.get("banger", {}),
                "position_guidance": result.get("position_guidance", {}),
                "cycle_position":    result.get("cycle_position", {}),
                "fair_value_composite": result.get("fair_value_composite"),
                "entry_assessment":  result.get("entry_assessment", ""),
            })

    return discoveries


def collect_watchlist_results(output_dir: Path) -> list:
    """Collect BUYs that crossed BUY_THRESHOLD but FAILED the confirmation gate
    (confirmed is False), from both the scouts and gems output dirs. These are the
    'Under Review' cards for dd:watchlist.

    Uses the same recency windows as the confirmed collectors so a stale reject
    ages off the board naturally; dedupes to the newest file per ticker.
    """
    out = []
    for src_dir, window in [(output_dir / "scouts", SCOUT_UPLOAD_WINDOW_SECS),
                            (output_dir / "gems",   GEMS_UPLOAD_WINDOW_SECS)]:
        if not src_dir.exists():
            continue
        cutoff = time.time() - window
        latest: dict[str, Path] = {}
        for path in src_dir.glob("*.json"):
            if path.stat().st_mtime < cutoff:
                continue
            ticker = path.stem.split("_")[0].upper()
            if ticker not in latest or path.stat().st_mtime > latest[ticker].stat().st_mtime:
                latest[ticker] = path
        for ticker, path in sorted(latest.items()):
            data = load_json(path)
            if not data:
                continue
            result = data.get("result", {})
            # Only true confirmation-gate rejects: scored a BUY but confirmed is False.
            if result.get("consensus_score", 0) < BUY_THRESHOLD:
                continue
            if result.get("confirmed", True):
                continue
            card = _scout_card(ticker, result, data.get("meta", {}))
            card["verification"] = result.get("verification", {})
            out.append(card)
    return out


def main():
    if not UPLOAD_SECRET:
        print("[upload] Missing DD_UPLOAD_SECRET — skipping upload")
        sys.exit(0)

    output_dir = Path("output")
    scout_dir  = output_dir / "scouts"

    print("\n[upload] Collecting portfolio results...")
    portfolio_results, index, portfolio_cards, reconcile_remove = collect_portfolio_results(output_dir)
    print(f"  {len(portfolio_results)} ticker file(s) found "
          f"({len(portfolio_cards)} >= threshold, {len(reconcile_remove)} below)")

    print("[upload] Collecting scout results (last 2h)...")
    discoveries = collect_scout_results(scout_dir)
    print(f"  {len(discoveries)} BUY signal(s)")

    # Merge portfolio/triggered cards into the scout payload (dedupe by ticker;
    # a dedicated scout result wins over a portfolio card for the same ticker).
    if portfolio_cards:
        seen = {d["ticker"] for d in discoveries}
        discoveries = discoveries + [c for c in portfolio_cards if c["ticker"] not in seen]
        # A ticker that now qualifies must not also be in the removal list.
        qualifying = {d["ticker"] for d in discoveries}
        reconcile_remove = [t for t in reconcile_remove if t not in qualifying]

    gems_dir = output_dir / "gems"
    print("[upload] Collecting gems results (last 26h)...")
    gems_discoveries = collect_gems_results(gems_dir)
    print(f"  {len(gems_discoveries)} gems BUY signal(s)")

    print("[upload] Collecting watchlist (confirmation-gate rejects)...")
    watchlist = collect_watchlist_results(output_dir)
    print(f"  {len(watchlist)} under-review")

    # Include scout/gems history files for KV backup (cache-miss recovery)
    print("[upload] Collecting scout history for KV backup...")
    histories: dict[str, dict | None] = {
        "scout_history": None, "scout_notified": None, "gems_history": None,
        "scout_seen": None,
    }
    for fname, varname in [
        ("scout_history.json",  "scout_history"),
        ("scout_notified.json", "scout_notified"),
        ("gems_history.json",   "gems_history"),
        ("scout_seen.json",     "scout_seen"),
    ]:
        path = output_dir / fname
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                histories[varname] = data
                print(f"  {len(data)} ticker(s) in {fname}")
        except Exception as e:
            print(f"  [upload] Could not read {fname}: {e}")
    scout_history  = histories["scout_history"]
    scout_notified = histories["scout_notified"]
    gems_history   = histories["gems_history"]
    scout_seen     = histories["scout_seen"]

    # A 0-signal run still MUST upload: the dedup/notify histories grew this run,
    # and skipping the POST would leave the KV backup stale (a later cache miss
    # would then restore an old window → re-debates and possible re-alerts) and
    # skip the dd:meta heartbeat the health screen watches.
    if (not portfolio_results and not index and not discoveries
            and not gems_discoveries and not reconcile_remove and not watchlist
            and not scout_history and not scout_notified and not gems_history
            and not scout_seen):
        print("[upload] Nothing to upload.")
        return

    if reconcile_remove:
        print(f"[upload] Reconcile: removing {len(reconcile_remove)} below-threshold "
              f"card(s) from Scout/Gems: {', '.join(reconcile_remove)}")

    payload = {
        "results":          portfolio_results,
        "index":            index,
        "scouts":           discoveries if discoveries else None,
        "gems":             gems_discoveries if gems_discoveries else None,
        "watchlist":        watchlist if watchlist else None,
        "reconcile_remove": reconcile_remove if reconcile_remove else None,
        "scout_history":    scout_history,
        "scout_notified":   scout_notified,
        "gems_history":     gems_history,
        "scout_seen":       scout_seen,
    }

    payload = _sanitize(payload)

    url = f"{UPLOAD_URL}/api/dd/upload"
    print(f"\n[upload] POSTing {len(portfolio_results)} key(s) + {len(discoveries)} scout + {len(gems_discoveries)} gems BUY(s) + history to {url}...")

    try:
        r = requests.post(url, headers=HEADERS, json=payload, timeout=60)
        data = r.json()
        if data.get("ok"):
            print(f"  Success — {len(data.get('written', []))} key(s) written")
        else:
            print(f"  Partial/failed: {data}")
            if data.get("failed"):
                sys.exit(1)
    except Exception as e:
        print(f"  [upload] Request failed: {e}")
        sys.exit(1)

    # ── Supabase: persist DD history ──────────────────────────────
    if SUPABASE_URL and SUPABASE_KEY:
        print("\n[supabase] Writing DD history + scout history...")
        dd_rows = []
        for item in portfolio_results:
            raw = item.get("value", {})
            result  = raw.get("result", {})
            dossier = raw.get("dossier", {})
            if not result.get("ticker"):
                continue
            price     = dossier.get("quote", {}).get("price")
            result_fv = result.get("fair_value_composite")
            comp_fv   = dossier.get("fair_values", {}).get("composite_fair_value")
            try:
                mos = round((result_fv - price) / price, 4) if price and result_fv else None
            except Exception:
                mos = None
            dd_rows.append(_sanitize({
                "ticker":       result["ticker"],
                # run_at is NOT NULL in Supabase — a missing built_at would reject the
                # whole batch insert, so fall back to upload time.
                "run_at":       dossier.get("built_at") or result.get("built_at")
                                or datetime.now(timezone.utc).isoformat(),
                "price":        price,
                "composite_fv": comp_fv,
                "result_fv":    result_fv,
                "mos":          mos,
                "score":        result.get("consensus_score"),
                "grade":        result.get("consensus_grade"),
                "confidence":   result.get("confidence"),
                "archetype":    _extract_archetype(dossier),
                "agent_scores": result.get("agent_final_scores", {}),
                "thesis":       (result.get("majority_thesis") or "")[:500],
                "swing":        (result.get("key_swing_factor") or "")[:300],
                "is_banger":    bool(result.get("banger", {}).get("is_banger")),
                "full_result":  {k: v for k, v in result.items() if k != "transcript"},
            }))
        if dd_rows:
            _supabase_insert("dd_history", dd_rows)
            print(f"  {len(dd_rows)} DD row(s) inserted")

        # Scout history
        scout_rows = []
        for s in (discoveries or []):
            scout_rows.append({
                "ticker":        s["ticker"],
                "score":         s.get("score"),
                "grade":         s.get("grade"),
                "sector":        s.get("sector"),
                "path":          s.get("path"),
                "filters":       s.get("matched_filters", []),
                "thesis":        (s.get("thesis") or "")[:300],
                "discovered_at": datetime.now(timezone.utc).isoformat(),
            })
        if scout_rows:
            _supabase_insert("scout_history", scout_rows)
            print(f"  {len(scout_rows)} scout row(s) inserted")

        # Gems history (requires a Supabase `gems_history` table; insert no-ops with a
        # warning if it doesn't exist — see DATA_CONTRACT.md).
        gems_rows = []
        for g in (gems_discoveries or []):
            gems_rows.append({
                "ticker":        g["ticker"],
                "score":         g.get("score"),
                "grade":         g.get("grade"),
                "thesis":        (g.get("thesis") or "")[:300],
                "catalyst":      (g.get("catalyst") or "")[:300],
                "fair_value":    g.get("fair_value_composite"),
                "discovered_at": datetime.now(timezone.utc).isoformat(),
            })
        if gems_rows:
            _supabase_insert("gems_history", gems_rows)
            print(f"  {len(gems_rows)} gems row(s) inserted")


if __name__ == "__main__":
    main()
