"""Debate orchestrator â€" async parallel rounds, grounded R1, dynamic convergence, synthesis or moderator."""

import asyncio
from statistics import mean

import requests

from agents import (
    AGENTS, moderator_prompt, research_prompt, round1_prompt,
    round2_prompt, round3_prompt, synthesis_prompt,
)
from llm import call_gemini_async, extract_json
from live_events import emit_live
import scoring

CONVERGENCE_THRESHOLD = 2.5
MAX_LOOPS = 3


from grading import grade as _grade, grade_hold as _grade_hold, BUY_THRESHOLD


def _live_scores(scores: dict, results: dict) -> dict:
    """Return only the scores of agents that produced a real result.

    A failed agent carries a fabricated 5.0 (see the _r1_agent/_r3_agent fallbacks).
    Letting those into spread/mean/convergence math invents a fake 'neutral' opinion
    that shrinks the spread and can fake convergence, so exclude them from all
    consensus statistics. Falls back to the full set only if every agent failed.
    """
    live = {a: s for a, s in scores.items() if not (results.get(a) or {}).get("_failed")}
    return live if live else dict(scores)


# â"€â"€ Per-agent async helpers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

async def _r1_agent(agent: str, ticker: str, dossier: dict, company_name: str, is_holding: bool = False) -> tuple[str, dict]:
    """Run grounded research then scored analysis for one agent. Returns (agent, result)."""
    try:
        sys_r, usr_r = research_prompt(agent, ticker, company_name)
        print(f"  [debate] R1-research / {agent}...")
        web_summary = await call_gemini_async(sys_r, usr_r, grounding=True, max_output_tokens=8192)

        sys_p, usr_p = round1_prompt(agent, ticker, dossier, web_summary, is_holding=is_holding)
        print(f"  [debate] R1-analysis / {agent}...")
        text = await call_gemini_async(sys_p, usr_p, max_output_tokens=16384)
        result = extract_json(text)
        if not isinstance(result, dict):
            raise ValueError(f"expected JSON object, got {type(result).__name__}")
        result["agent"] = agent
        result["web_research"] = web_summary
        return agent, result
    except Exception as e:
        print(f"  [debate] {agent} failed in R1: {e}")
        return agent, {"agent": agent, "score": 5.0, "grade": "HOLD", "thesis": "", "web_research": "", "_failed": True}


async def _r1_emit(agent: str, ticker: str, dossier: dict, company_name: str, is_holding: bool = False) -> tuple[str, dict]:
    """Wraps _r1_agent â€" emits R1_SCORE live event as soon as this agent completes."""
    pair = await _r1_agent(agent, ticker, dossier, company_name, is_holding=is_holding)
    await emit_live(ticker, {
        "type": "R1_SCORE",
        "agent": pair[0],
        "score": pair[1].get("score", 5.0),
    })
    return pair


async def _r2_agent(agent: str, ticker: str, scores: dict, all_r1: list, loop: int, target: str = "") -> tuple[str, dict]:
    """Cross-examination for one agent. Returns (agent, result)."""
    try:
        sys_p, usr_p = round2_prompt(agent, ticker, scores[agent], all_r1, loop, target)
        print(f"  [debate] R2-{loop} / {agent} -> challenges {target}...")
        text = await call_gemini_async(sys_p, usr_p, max_output_tokens=8192, thinking_level=None)
        result = extract_json(text)
        if not isinstance(result, dict):
            raise ValueError(f"expected JSON object, got {type(result).__name__}")
        result["agent"] = agent
        result["target_agent"] = target  # enforce assignment; prevent LLM from redirecting
        return agent, result
    except Exception as e:
        print(f"  [debate] {agent} failed in R2-{loop}: {e}")
        return agent, {"agent": agent, "target_agent": target, "challenge": "", "_failed": True}


async def _r2_emit(agent: str, ticker: str, scores: dict, all_r1: list, loop: int, target: str = "") -> tuple[str, dict]:
    """Wraps _r2_agent â€" emits R2_CHALLENGE live event as soon as this agent completes."""
    pair = await _r2_agent(agent, ticker, scores, all_r1, loop, target)
    await emit_live(ticker, {
        "type": "R2_CHALLENGE",
        "agent": pair[0],
        "target": pair[1].get("target_agent", target),
        "challenge": (pair[1].get("challenge", "") or "")[:60],
        "loop": loop,
    })
    return pair


async def _r3_agent(
    agent: str, ticker: str, scores: dict,
    r2_results: dict, all_r2: list, loop: int,
) -> tuple[str, dict]:
    """Rebuttal & revised score for one agent. Returns (agent, result)."""
    try:
        challenges = [r for r in r2_results.values() if r.get("target_agent") == agent]
        sys_p, usr_p = round3_prompt(agent, ticker, scores[agent], challenges, all_r2, loop)
        print(f"  [debate] R3-{loop} / {agent}...")
        text = await call_gemini_async(sys_p, usr_p, max_output_tokens=8192, thinking_level=None)
        result = extract_json(text)
        if not isinstance(result, dict):
            raise ValueError(f"expected JSON object, got {type(result).__name__}")
        result["agent"] = agent
        return agent, result
    except Exception as e:
        print(f"  [debate] {agent} failed in R3-{loop}: {e}")
        return agent, {"agent": agent, "revised_score": scores.get(agent, 5.0), "final_thesis": "", "_failed": True}


async def _r3_emit(
    agent: str, ticker: str, scores: dict,
    r2_results: dict, all_r2: list, loop: int,
) -> tuple[str, dict]:
    """Wraps _r3_agent â€" emits R3_DELTA live event as soon as this agent completes."""
    pair = await _r3_agent(agent, ticker, scores, r2_results, all_r2, loop)
    prev    = scores.get(pair[0], 5.0)
    revised = float(pair[1].get("revised_score", prev))
    delta   = round(revised - prev, 2)
    await emit_live(ticker, {
        "type": "R3_DELTA",
        "agent": pair[0],
        "score": revised,
        "delta": delta,
        "loop": loop,
    })
    return pair


# â"€â"€ Main async orchestrator â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def _extract_ve_fair_value(transcript: list) -> float | None:
    """Extract ValuationEngine's fair_value_estimate from transcript R1 output."""
    for entry in transcript:
        if not isinstance(entry, dict):
            continue
        if entry.get("agent") == "ValuationEngine" and str(entry.get("round", "")).startswith("1"):
            fva = entry.get("fair_value_assessment")
            if isinstance(fva, dict) and fva.get("fair_value_estimate") is not None:
                try:
                    return float(fva["fair_value_estimate"])
                except (TypeError, ValueError):
                    pass
    return None


def _sanitize_fv(fv: float | None, price: float | None) -> float | None:
    """Reject a fair value that looks like a ratio/multiple rather than a dollar price.

    The synthesis LLM occasionally outputs a P/E or P/TBV ratio when it should output
    a dollar price (e.g. 2.15 for JPM instead of $200). Discard values that are less
    than 1% or more than 2000% of the current stock price.
    Lower bound 1% (not 5%) to avoid discarding legitimate distressed/turnaround FVs.
    """
    if fv is None or fv <= 0:
        return None
    if price and price > 0:
        ratio = fv / price
        if ratio < 0.01 or ratio > 20.0:
            return None
    return fv


async def run(ticker: str, dossier: dict, verbose: bool = True, max_loops: int | None = None, is_holding: bool = False) -> dict:
    """Run the full debate asynchronously. Returns the final consensus dict.

    When ``is_holding`` is True, the debate is framed as a hold-vs-trim-vs-exit
    decision on a current portfolio position rather than a buy-vs-pass decision
    on a new candidate. Agent prompts get a hold-mode preamble, the scoring
    pipeline softens its entry-time penalties, and the final grade is drawn from
    the ADD/HOLD/TRIM/EXIT ladder.
    """
    transcript: list[dict] = []
    company_name = dossier.get("profile", {}).get("name", ticker)
    effective_max_loops = max_loops if max_loops is not None else MAX_LOOPS
    _final_grader = _grade_hold if is_holding else _grade

    # Signal to frontend that the debate has started
    await emit_live(ticker, {"type": "START"})

    # â"€â"€ ROUND 1 â€" All agents in parallel (each: research -> analysis) â"€â"€â"€â"€â"€â"€â"€â"€â"€
    if verbose:
        print("\n+----------------------------------------------+")
        print("|  ROUND 1 -- Grounded Research & Assessment   |")
        print(f"|  ({len(AGENTS)} agents running in parallel)              |")
        print("+----------------------------------------------+")

    r1_pairs = await asyncio.gather(
        *[_r1_emit(a, ticker, dossier, company_name, is_holding=is_holding) for a in AGENTS]
    )

    r1_results: dict[str, dict] = {}
    for agent, result in r1_pairs:
        r1_results[agent] = result
        transcript.append(result)

    if verbose:
        for agent in AGENTS:
            r = r1_results[agent]
            score   = r.get("score", "?")
            grade   = r.get("grade", "")
            thesis  = r.get("thesis", "")[:80]
            finding = r.get("web_finding", "")[:60]
            print(f"    {agent:<14} -> {score:>4}  [{grade}]  {thesis}...")
            print(f"    {'':14}   web: {finding}...")

    scores    = {a: float(r1_results[a].get("score", 5.0)) for a in AGENTS}
    scores_r1 = dict(scores)
    all_r1    = list(r1_results.values())

    # Collect R1 score breakdowns as fallback
    r1_breakdowns = {a: r1_results[a].get("score_breakdown") for a in AGENTS}
    agent_breakdowns = r1_breakdowns  # will be overwritten if loops run

    # â"€â"€ DEBATE LOOPS â€" R2 + R3 in parallel per round â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    loops_run = 0
    r3_results: dict[str, dict] = {}
    prev_spread: float | None = None

    for loop in range(1, effective_max_loops + 1):
        loops_run = loop

        if verbose:
            print(f"\n+----------------------------------------------+")
            print(f"|  LOOP {loop} / ROUND 2 -- Cross-Examination         |")
            print(f"|  ({len(AGENTS)} agents running in parallel)              |")
            print(f"+----------------------------------------------+")

        _sorted = sorted(AGENTS, key=lambda a: scores[a])
        n = len(_sorted)
        r2_targets = {}
        for i in range(n // 2):
            r2_targets[_sorted[i]] = _sorted[n - 1 - i]
            r2_targets[_sorted[n - 1 - i]] = _sorted[i]
        if n % 2 == 1:
            mid = _sorted[n // 2]
            _mean = sum(scores[a] for a in AGENTS) / n
            _bull_dist = scores[_sorted[-1]] - _mean
            _bear_dist = _mean - scores[_sorted[0]]
            r2_targets[mid] = _sorted[-1] if _bull_dist >= _bear_dist else _sorted[0]

        r2_pairs = await asyncio.gather(
            *[_r2_emit(a, ticker, scores, all_r1, loop, r2_targets[a]) for a in AGENTS]
        )
        r2_results: dict[str, dict] = {a: r for a, r in r2_pairs}
        for _, r in r2_pairs:
            transcript.append(r)

        if verbose:
            for agent in AGENTS:
                r         = r2_results[agent]
                target    = r.get("target_agent", "?")
                challenge = r.get("challenge", "")[:70]
                print(f"    {agent:<14} -> challenges {target}: {challenge}...")

        all_r2 = list(r2_results.values())

        if verbose:
            print(f"\n+----------------------------------------------+")
            print(f"|  LOOP {loop} / ROUND 3 -- Rebuttal & Revision       |")
            print(f"|  ({len(AGENTS)} agents running in parallel)              |")
            print(f"+----------------------------------------------+")

        r3_pairs = await asyncio.gather(
            *[_r3_emit(a, ticker, scores, r2_results, all_r2, loop) for a in AGENTS]
        )
        r3_results = {a: r for a, r in r3_pairs}
        for _, r in r3_pairs:
            transcript.append(r)

        if verbose:
            for agent in AGENTS:
                r       = r3_results[agent]
                prev    = scores[agent]
                revised = float(r.get("revised_score", prev))
                delta   = revised - prev
                arrow   = "^" if delta > 0 else ("v" if delta < 0 else "-")
                print(f"    {agent:<14} -> {prev} {arrow} {revised}  (D {delta:+.1f})")

        scores     = {a: float(r3_results[a].get("revised_score", scores[a])) for a in AGENTS}

        # Collect per-agent score breakdowns from R3 (overwrite each loop)
        agent_breakdowns = {
            a: r3_results[a].get("revised_breakdown") or r3_results[a].get("score_breakdown")
            for a in AGENTS if a in r3_results
        }

        live_scores = _live_scores(scores, r3_results)
        score_vals  = list(live_scores.values())
        spread      = max(score_vals) - min(score_vals)

        if verbose:
            avg = mean(score_vals)
            print(f"\n  Scores after loop {loop}: {scores}")
            print(f"  Spread: {spread:.2f}  Mean: {avg:.2f}  Threshold: {CONVERGENCE_THRESHOLD}")

        if spread <= CONVERGENCE_THRESHOLD:
            if verbose:
                print(f"  OK Converged in {loop} loop(s).")
            break
        elif prev_spread is not None and 0 <= (prev_spread - spread) < 0.1:
            # Narrowing too slowly to justify another loop. A WIDENING spread
            # (prev_spread - spread < 0) is NOT a stall — agents are still moving, so
            # fall through and run another loop instead of escalating to the moderator.
            if verbose:
                print(f"  ! Stalled ({prev_spread:.2f} -> {spread:.2f}), escalating to moderator early.")
            break
        elif loop < effective_max_loops:
            if verbose:
                print(f"  X Not converged -- running loop {loop + 1}...")
            all_r1 = [
                {
                    "agent": a,
                    "score": r3_results[a].get("revised_score", scores[a]),
                    "thesis": r3_results[a].get("final_thesis", ""),
                    "rebuttal": r3_results[a].get("rebuttal", ""),
                    "concessions": r3_results[a].get("concessions", ""),
                }
                for a in AGENTS
            ]

        prev_spread = spread

    # â"€â"€ CONSENSUS â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    if verbose:
        print(f"\n+----------------------------------------------+")
        print(f"|  CONSENSUS                                    |")
        print(f"+----------------------------------------------+")

    # Consensus statistics over agents that actually produced results — a failed
    # agent's fabricated 5.0 must not contaminate spread/mean/convergence.
    _final_results = r3_results if r3_results else r1_results
    failed_agents  = [a for a in AGENTS if (_final_results.get(a) or {}).get("_failed")]
    live_scores    = _live_scores(scores, _final_results)
    score_vals     = list(live_scores.values())
    spread         = max(score_vals) - min(score_vals)
    avg            = mean(score_vals)

    final_positions = [r3_results[a] for a in AGENTS if a in r3_results] or all_r1

    if spread <= CONVERGENCE_THRESHOLD:
        if verbose:
            print(f"  [OK] Scores converged (spread={spread:.2f}) -- synthesizing...")
        sys_p, usr_p = synthesis_prompt(ticker, final_positions)
        print(f"  [debate] Synthesis...")
        text = await call_gemini_async(sys_p, usr_p, max_output_tokens=6144)
        moderator_result = extract_json(text)
        if not isinstance(moderator_result, dict):
            moderator_result = {}
        moderator_result.setdefault("consensus_score", round(avg, 2))
        moderator_result.setdefault("consensus_grade", _final_grader(avg))
        moderator_result.setdefault("confidence", "HIGH" if spread <= 1.0 else "MEDIUM")
    else:
        if verbose:
            print(f"  [!] Did not converge after {effective_max_loops} loops (spread={spread:.2f}) -- calling moderator...")
        sys_p, usr_p = moderator_prompt(ticker, transcript, loops_run, spread, threshold=CONVERGENCE_THRESHOLD)
        print(f"  [debate] Moderator...")
        text = await call_gemini_async(sys_p, usr_p, max_output_tokens=8192)
        moderator_result = extract_json(text)
        if not isinstance(moderator_result, dict):
            moderator_result = {}

    # â"€â"€ Post-debate scoring pipeline â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    # If agents failed (rate-limit/truncation/parse), the consensus rests on fewer
    # opinions — surface it and never claim HIGH confidence.
    if failed_agents:
        print(f"  [debate] WARNING: {len(failed_agents)}/{len(AGENTS)} agents failed: {failed_agents}")
        if len(failed_agents) >= 2:
            moderator_result["confidence"] = "LOW"
        elif moderator_result.get("confidence") == "HIGH":
            moderator_result["confidence"] = "MEDIUM"

    raw_score = moderator_result.get("consensus_score", round(avg, 2))
    scoring_output = scoring._safe_apply_adjustments(
        raw_score=raw_score,
        result=moderator_result,
        dossier=dossier,
        is_holding=is_holding,
    )
    # Merge adjusted values back into moderator_result so the transcript captures them
    moderator_result["raw_consensus_score"]  = raw_score
    moderator_result["consensus_score"]      = scoring_output["adjusted_score"]
    moderator_result["consensus_grade"]      = scoring_output["consensus_grade"]
    moderator_result["score_adjustments"]    = scoring_output["score_adjustments"]
    moderator_result["banger"]               = scoring_output["banger"]
    moderator_result["position_guidance"]    = scoring_output["position_guidance"]
    moderator_result["risk_reward"]          = scoring_output.get("risk_reward", {"applied": False})

    transcript.append(moderator_result)

    # Signal consensus grade and completion to the frontend
    consensus_grade = moderator_result.get("consensus_grade", _final_grader(avg))
    await emit_live(ticker, {"type": "CONSENSUS", "grade": consensus_grade})
    await emit_live(ticker, {"type": "DONE"})

    if verbose:
        cs   = moderator_result.get("consensus_score", avg)
        conf = moderator_result.get("confidence", "?")
        print(f"\n  CONSENSUS: {cs:.2f} / 10  [{consensus_grade}]  confidence={conf}")
        print(f"  {moderator_result.get('majority_thesis', '')[:120]}...")

    _price = (dossier.get("quote") or {}).get("price")
    _dossier_fv = (dossier.get("fair_values") or {}).get("composite_fair_value")
    _llm_fv = _sanitize_fv(
        moderator_result.get("fair_value_composite") or _extract_ve_fair_value(transcript),
        _price,
    )

    # Cross-check the agents' bull/bear price targets against the computed R:R —
    # a large divergence flags either an over-excited debate or a stale FV engine.
    _rr = moderator_result.get("risk_reward") or {}
    if _rr.get("applied"):
        from risk_reward import llm_cross_check

        def _coerce(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        # Synthesis may return null targets — fall back to CatalystHunter's R1
        # bull_target/bear_floor (the agent explicitly asked to price the range).
        def _target(field):
            v = _coerce(moderator_result.get(field))
            if v is None:
                for t in transcript:
                    if isinstance(t, dict) and t.get("agent") == "CatalystHunter" and t.get(field) is not None:
                        v = _coerce(t.get(field))
                        break
            return _sanitize_fv(v, _price)

        _xc = llm_cross_check(_rr, _target("bull_target"), _target("bear_floor"), _price)
        if _xc:
            _rr["llm_cross_check"] = _xc

    result_out = {
        "ticker":               ticker,
        "raw_consensus_score":  moderator_result.get("raw_consensus_score", round(avg, 2)),
        "consensus_score":      moderator_result.get("consensus_score", round(avg, 2)),
        "consensus_grade":      moderator_result.get("consensus_grade", _final_grader(avg)),
        "mode":                 "hold" if is_holding else "scout",
        "confidence":           moderator_result.get("confidence", "MEDIUM"),
        "majority_thesis":      moderator_result.get("majority_thesis", ""),
        "dissent":              moderator_result.get("dissent", ""),
        "key_swing_factor":     moderator_result.get("key_swing_factor", ""),
        "score_rationale":      moderator_result.get("score_rationale", ""),
        "catalyst":             moderator_result.get("catalyst", ""),
        "asymmetry_ratio":      moderator_result.get("asymmetry_ratio", ""),
        "risk_reward":          moderator_result.get("risk_reward", {"applied": False}),
        "moat_composite":       moderator_result.get("moat_composite"),
        "cycle_position":       moderator_result.get("cycle_position", {}),
        "fair_value_composite": _llm_fv or _dossier_fv,
        "entry_assessment":     moderator_result.get("entry_assessment", ""),
        "data_confidence":      dossier.get("data_quality", {}).get("data_confidence", "HIGH"),
        "score_adjustments":    moderator_result.get("score_adjustments", {}),
        "banger":               moderator_result.get("banger", {}),
        "position_guidance":    moderator_result.get("position_guidance", {}),
        "agent_r1_scores":      scores_r1,
        "agent_final_scores":   scores,
        "score_decomposition":  agent_breakdowns,
        "score_spread":         round(spread, 2),
        "loops_run":            loops_run,
        "converged":            spread <= CONVERGENCE_THRESHOLD,
        "failed_agents":        failed_agents,
        "transcript":           transcript,
    }

    # ── BUY confirmation gate ──────────────────────────────────────────────────
    # A scout/gems BUY gets a second round of scrutiny (deterministic quality gate
    # + adversarial red-team) before it is surfaced. Hold-mode (portfolio) results
    # and sub-threshold scores are not gated. verify_buy never raises.
    if not is_holding and result_out["consensus_score"] >= BUY_THRESHOLD:
        import verify
        _ver = await verify.verify_buy(ticker, result_out, dossier)
        result_out["verification"] = _ver
        result_out["confirmed"]    = _ver.get("confirmed", True)
        if verbose:
            print(f"  [verify] {ticker} BUY → "
                  f"{'CONFIRMED' if result_out['confirmed'] else 'UNDER REVIEW'} "
                  f"({_ver.get('verdict', '?')})")

    return result_out
