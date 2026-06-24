"""Gems pipeline — Finviz screening → pillar scoring → Gemma function-calling triage → debate.

Flow:
  1. Run 10 Finviz screens → unique candidate list
  2. Enrich candidates with ticker_fundament() data
  3. Compute pillar scores for all candidates
  4. Gemma triage with function-calling tools (rank/compare/score) → select top N
  5. Full debate on selected picks (max 4 concurrent)
  6. Return BUY signals only (score >= BUY_THRESHOLD)
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from grading import BUY_THRESHOLD

GEMS_HISTORY_FILE   = Path("output/gems_history.json")
GEMS_COOLDOWN_HOURS = int(os.getenv("GEMS_COOLDOWN_HOURS", "72"))
GEMS_DEBATE_COUNT   = int(os.getenv("GEMS_DEBATE_COUNT", "6"))
GEMS_MAX_LOOPS      = int(os.getenv("GEMS_MAX_LOOPS", "3"))


# ── History helpers ────────────────────────────────────────────────────────────

def _load_history() -> dict:
    """Load {ticker: {ts, score, grade}} from disk. Returns {} if missing or corrupt."""
    try:
        if GEMS_HISTORY_FILE.exists():
            return json.loads(GEMS_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_history(history: dict) -> None:
    """Persist gems history to disk (atomic — a corrupt file loads as {} and
    would re-debate everything, same failure mode the scout helper guards)."""
    from scout import _atomic_write_text
    try:
        _atomic_write_text(GEMS_HISTORY_FILE, json.dumps(history, indent=2))
    except Exception as e:
        print(f"  [gems] Warning: could not save history: {e}")


def _recently_analyzed(history: dict) -> set[str]:
    """Return set of tickers analyzed within GEMS_COOLDOWN_HOURS."""
    cutoff = datetime.now(timezone.utc).timestamp() - GEMS_COOLDOWN_HOURS * 3600
    return {ticker for ticker, entry in history.items() if entry.get("ts", 0) >= cutoff}


# ── Triage system prompt + prompt builder ─────────────────────────────────────

TRIAGE_SYSTEM = """You are a systematic equity analyst specializing in identifying under-the-radar
quality businesses. You have access to pre-computed pillar scores for every candidate.

Use the available tools to:
1. rank_candidates() — identify which pillars have the strongest cohort
2. compute_pillar_scores() — deep-dive individual tickers you find interesting
3. compare_candidates() — side-by-side comparisons before finalizing picks

Your goal: select the {debate_count} highest-conviction hidden gems. Prioritize:
- HIGH composite scores (7.0+) from multiple strong pillars simultaneously
- Chokepoint businesses (chokepoint_proxy >= 7.0) with good financials
- Capital-efficient compounders (financial_physics >= 7.0, moat_proxy >= 6.5)
- Growth acceleration stories (temporal >= 7.0) at reasonable valuations
- Management quality signals (management >= 7.0) combined with moat

Avoid: single-pillar outliers, purely momentum-driven scores, sectors with structural headwinds.
Return a JSON list of your final picks with reasoning."""


def _build_triage_prompt(
    candidates: list[dict],
    scores: dict[str, dict],
    recently_analyzed: set[str],
    debate_count: int = 6,
) -> str:
    """Build the triage user prompt with candidate universe summary."""
    # Top 20 by composite score
    scored = [
        (c, scores.get(c["ticker"], {}))
        for c in candidates
    ]
    scored.sort(key=lambda x: x[1].get("composite", 0), reverse=True)
    top20 = scored[:20]

    lines = [
        f"You have a universe of {len(candidates)} pre-screened US equities with pillar scores "
        f"already computed. Pillar scores are available for every candidate via the tools.\n",
        f"TOP 20 BY COMPOSITE SCORE:",
        f"{'TICKER':<8} {'COMPOSITE':<10} {'FIN_PHYS':<10} {'MOAT':<8} {'TEMPORAL':<10} {'INDUSTRY':<30}",
        "-" * 76,
    ]
    for c, s in top20:
        lines.append(
            f"{c['ticker']:<8} {s.get('composite', 0):<10.2f} "
            f"{s.get('financial_physics', 0):<10.2f} "
            f"{s.get('moat_proxy', 0):<8.2f} "
            f"{s.get('temporal', 0):<10.2f} "
            f"{str(c.get('fundament', {}).get('Industry', '—'))[:29]:<30}"
        )

    if recently_analyzed:
        lines.append(
            f"\nEXCLUDE these recently analyzed tickers (within cooldown): "
            f"{', '.join(sorted(recently_analyzed))}"
        )

    lines += [
        f"\nUse the available tools to explore the full universe further before deciding.",
        f"Select EXACTLY {debate_count} picks as your highest-conviction hidden gems.",
        f"\nReturn your answer as a JSON object with this exact structure:",
        '{"picks": [',
        '  {"ticker": "SYM", "pillar_rationale": "why this ticker stands out across pillars", "conviction": "HIGH|MEDIUM"},',
        '  ...',
        ']}',
        "Return ONLY the JSON object. No other text.",
    ]
    return "\n".join(lines)


# ── Triage function ────────────────────────────────────────────────────────────

async def _triage_with_tools(
    candidates: list[dict],
    scores: dict[str, dict],
    recently_analyzed: set[str],
    verbose: bool = True,
    debate_count: int = 6,
) -> list[dict]:
    """Run Gemma function-calling triage to select top candidates."""
    from llm import call_gemini_with_tools_async, extract_json
    from calc_tools import TRIAGE_TOOLS, execute_tool_call

    if not candidates:
        return []

    # Build candidates dict for tool executor (ticker -> fundament)
    cand_dict = {c["ticker"]: c.get("fundament", {}) for c in candidates}

    prompt = _build_triage_prompt(candidates, scores, recently_analyzed, debate_count)

    if verbose:
        print(f"  [gems] Triaging {len(candidates)} candidates with function-calling Gemma...")

    def tool_executor(fn_name: str, fn_args: dict, **kwargs) -> Any:
        return execute_tool_call(fn_name, fn_args, kwargs["candidates"], kwargs["scores"])

    try:
        text = await call_gemini_with_tools_async(
            system=TRIAGE_SYSTEM.format(debate_count=debate_count),
            user=prompt,
            tools=TRIAGE_TOOLS,
            tool_executor=tool_executor,
            tool_executor_kwargs={"candidates": cand_dict, "scores": scores},
            temperature=0.2,
            max_tool_turns=8,
        )
    except Exception as e:
        print(f"  [gems] Triage LLM call failed: {e}")
        print("  [gems] Falling back to top-N by composite score")
        sorted_cands = sorted(
            candidates,
            key=lambda c: scores.get(c["ticker"], {}).get("composite", 0),
            reverse=True,
        )
        valid = [
            {"ticker": c["ticker"], "pillar_rationale": "top composite score (LLM fallback)", "conviction": "MEDIUM"}
            for c in sorted_cands
            if c["ticker"] not in recently_analyzed
        ]
        return valid[:debate_count]

    try:
        parsed = extract_json(text)
        picks_raw = parsed.get("picks", []) if isinstance(parsed, dict) else []
        valid_syms = {c["ticker"] for c in candidates}
        valid = [
            p for p in picks_raw
            if isinstance(p, dict)
            and p.get("ticker", "").upper() in valid_syms
            and p.get("ticker", "").upper() not in recently_analyzed
        ]
        if verbose:
            print(f"  [gems] Gemma selected: {[p['ticker'] for p in valid]}")
        return valid[:debate_count]
    except Exception as e:
        print(f"  [gems] Triage parse error: {e}\n  Raw: {text[:300]}")
        # Fallback: pick top-N by composite score
        sorted_cands = sorted(candidates, key=lambda c: scores.get(c["ticker"], {}).get("composite", 0), reverse=True)
        valid = [
            {"ticker": c["ticker"], "pillar_rationale": "top composite score", "conviction": "MEDIUM"}
            for c in sorted_cands
            if c["ticker"] not in recently_analyzed
        ]
        return valid[:debate_count]


# ── Main entry point ───────────────────────────────────────────────────────────

async def run_gems(
    verbose: bool = True,
) -> list[dict]:
    """Full gems pipeline:
      1. Finviz screens (10 screens)
      2. Enrich with fundament data
      3. Compute pillar scores
      4. Gemma function-calling triage
      5. Full debates on selected picks

    Configurable via env vars:
      GEMS_DEBATE_COUNT   — tickers to debate per run (default 6)
      GEMS_MAX_LOOPS      — max debate convergence loops (default 3)
      GEMS_COOLDOWN_HOURS — hours before re-analyzing a ticker (default 72)

    Returns list of BUY discovery dicts (score >= BUY_THRESHOLD only).
    """
    from finviz_screener import run_finviz_screens, get_unique_candidates, enrich_candidates
    from pillar_scoring import compute_composite
    from dossier import build as build_dossier
    from debate import run as run_debate

    history = _load_history()
    recently = _recently_analyzed(history)
    if verbose and recently:
        print(f"  [gems] Skipping {len(recently)} recently-analyzed ticker(s): "
              f"{', '.join(sorted(recently))}")

    # Phase 1 — Screen
    if verbose:
        print("\n+----------------------------------------------+")
        print("|  SOVEREIGN GEMS — Finviz screen              |")
        print("|  Running 10 screens...                       |")
        print("+----------------------------------------------+")

    screen_results = await asyncio.to_thread(run_finviz_screens)
    raw_tickers = get_unique_candidates(screen_results)

    # get_unique_candidates returns a flat list of ticker strings
    raw_candidates = [{"ticker": t} for t in raw_tickers]

    # Exclude recently analyzed
    raw_candidates = [c for c in raw_candidates if c["ticker"] not in recently]

    if verbose:
        print(f"  [gems] {len(raw_candidates)} unique candidates after dedup and cooldown exclusion")

    if not raw_candidates:
        if verbose:
            print("  [gems] No new candidates — all tickers within cooldown window")
        return []

    # Phase 2 — Enrich
    tickers = [c["ticker"] for c in raw_candidates]
    try:
        fundament_map = await asyncio.to_thread(enrich_candidates, tickers)
    except Exception as e:
        print(f"  [gems] Enrichment failed: {e}")
        return []

    # Attach fundament to each candidate
    for c in raw_candidates:
        c["fundament"] = fundament_map.get(c["ticker"], {})

    # Filter out candidates with no fundament data
    candidates = [c for c in raw_candidates if c["fundament"]]

    if verbose:
        print(f"  [gems] {len(candidates)} candidates with fundament data")

    if not candidates:
        if verbose:
            print("  [gems] No candidates with fundament data — aborting")
        return []

    # Phase 3 — Pillar scores
    scores: dict[str, dict] = {}
    for c in candidates:
        try:
            scores[c["ticker"]] = compute_composite(c["fundament"])
        except Exception:
            scores[c["ticker"]] = {
                "composite": 5.0,
                "financial_physics": 5.0,
                "moat_proxy": 5.0,
                "temporal": 5.0,
                "management": 5.0,
                "chokepoint_proxy": 5.0,
            }

    if verbose:
        top5 = sorted(scores.items(), key=lambda x: x[1].get("composite", 0), reverse=True)[:5]
        print(f"  [gems] Top 5 by composite: " + ", ".join(f"{t}={s.get('composite', 0):.1f}" for t, s in top5))

    # Phase 4 — Triage
    picks = await _triage_with_tools(
        candidates, scores, recently, verbose=verbose, debate_count=GEMS_DEBATE_COUNT
    )
    if not picks:
        print("  [gems] Triage returned no picks")
        return []

    # Metadata cleaner — same pre-debate pass scout does, so gems dossiers get
    # ADR/currency/sector corrections too (small caps misclassify the most).
    from cleaner import clean_ticker_batch
    try:
        batch_meta = await asyncio.wait_for(
            clean_ticker_batch([p["ticker"] for p in picks]), timeout=45.0
        )
    except Exception:
        batch_meta = {}

    if verbose:
        print(f"\n+----------------------------------------------+")
        print(f"|  SOVEREIGN GEMS — running debates            |")
        tickers_str = ", ".join(p["ticker"] for p in picks)
        print(f"|  Picks: {tickers_str[:39]:<39}|")
        print(f"|  {len(picks)} debates · max {GEMS_MAX_LOOPS} loop(s) · 4 concurrent     |")
        print(f"+----------------------------------------------+")

    # Phase 5 — Debates (max 4 concurrent)
    out_dir = Path("output/gems")
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(4)
    history_lock = asyncio.Lock()

    async def _debate_one(pick: dict) -> dict | None:
        ticker         = pick["ticker"].upper()
        pillar_rationale = pick.get("pillar_rationale", "")
        conviction     = pick.get("conviction", "MEDIUM")
        async with sem:
            try:
                if verbose:
                    print(f"\n  [gems] Analyzing {ticker} ({conviction})...")
                    if pillar_rationale:
                        print(f"         Gemma rationale: {pillar_rationale[:100]}")

                dossier = await build_dossier(ticker, verbose=False,
                                              meta=batch_meta.get(ticker, {}))
                result  = await run_debate(ticker, dossier, verbose=False, max_loops=GEMS_MAX_LOOPS)

                score = result.get("consensus_score", 0)
                grade = result.get("consensus_grade", "HOLD")

                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"{ticker}_{ts}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"result": result, "dossier": dossier}, f, indent=2, default=str)

                if verbose:
                    _tag = ""
                    if score >= BUY_THRESHOLD:
                        _tag = (" ← BUY ✓ CONFIRMED" if result.get("confirmed", True)
                                else " ← BUY ⚠ UNDER REVIEW")
                    print(f"  [gems] {ticker} → {score:.2f}/10 [{grade}]{_tag}")

                # Record in history regardless of grade (prevents re-analysis in cooldown window)
                async with history_lock:
                    history[ticker] = {
                        "ts":    datetime.now(timezone.utc).timestamp(),
                        "score": round(score, 2),
                        "grade": grade,
                    }
                    _save_history(history)

                if score >= BUY_THRESHOLD:
                    return {
                        "ticker":                ticker,
                        "score":                 round(score, 2),
                        "grade":                 grade,
                        "confidence":            result.get("confidence", ""),
                        "thesis":                result.get("majority_thesis", ""),
                        "score_rationale":       result.get("score_rationale", ""),
                        "dissent":               result.get("dissent", ""),
                        "key_swing_factor":      result.get("key_swing_factor", ""),
                        "catalyst":              result.get("catalyst", ""),
                        "asymmetry_ratio":       result.get("asymmetry_ratio", ""),
                        "rr":                    (result.get("risk_reward") or {}).get("rr_ratio"),
                        "risk":                  (result.get("risk_reward") or {}).get("risk_tier"),
                        "banger":                result.get("banger", {}),
                        "position_guidance":     result.get("position_guidance", {}),
                        "cycle_position":        result.get("cycle_position", {}),
                        "gems_composite_score":  scores.get(ticker, {}).get("composite", 0),
                        "gems_pillar_rationale": pillar_rationale,
                        "scout_lens":            "gems",
                        "gemma_rationale":       pillar_rationale,
                        "analyzed_at":           ts,
                        "output_file":           str(out_path),
                        "confirmed":             result.get("confirmed", True),
                        "verification":          result.get("verification", {}),
                    }
                return None
            except Exception as e:
                print(f"  [gems] {ticker} failed: {e}")
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
        print(f"\n  Gems complete: {len(discoveries)} BUY signal(s) "
              f"from {len(picks)} debated · history now has {len(history)} ticker(s)")

    return discoveries
