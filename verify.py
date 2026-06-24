"""BUY-signal confirmation gate — a second round of scrutiny before a debate
result is surfaced as an actionable BUY.

Every scout/gems result that crosses BUY_THRESHOLD passes through verify_buy()
before it reaches Telegram or the dashboard:

  Stage 1 — quality_gate (pure Python, 0 LLM calls):
      Reject BUYs whose INTERNAL signals are shaky — agents didn't converge,
      the risk/reward cross-check is divergent, risk index is high, confidence
      is LOW, an agent failed, or the dossier itself is low-confidence. These
      are all already computed by debate.run(), so this stage is free.

  Stage 2 — red_team (one grounded LLM call):
      Survivors face an adversarial prosecutor whose only job is to KILL the
      bull thesis using fresh Google-grounded research (recent guidance cuts,
      downgrades, deteriorating fundamentals, litigation, accounting flags).
      Returns CONFIRM / DOWNGRADE / VETO.

A BUY is `confirmed` only if it passes Stage 1 AND the prosecutor returns CONFIRM.
Rejected BUYs are not dropped — the caller routes them to an "Under Review"
watchlist with the reason attached.

Fail-safe contract (mirrors cleaner.py): never raises. On a Stage-2 error the
behavior is governed by VERIFY_FAIL_OPEN (default: confirm, so a transient API
blip never demotes a real BUY — no worse than the pre-gate behavior).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone


# ── Config (read at call time so env changes take effect without re-import) ──────
def _enabled() -> bool:
    return os.getenv("VERIFY_BUYS", "1").strip().lower() in ("1", "true", "yes", "on")


def _fail_open() -> bool:
    return os.getenv("VERIFY_FAIL_OPEN", "1").strip().lower() in ("1", "true", "yes", "on")


def _max_spread() -> float:
    try:
        return float(os.getenv("VERIFY_MAX_SPREAD", "2.0"))
    except ValueError:
        return 2.0


def _max_risk_index() -> float:
    try:
        return float(os.getenv("VERIFY_MAX_RISK_INDEX", "6.0"))
    except ValueError:
        return 6.0


_RED_TEAM_TIMEOUT = 90.0  # seconds — wraps the single grounded call


# ── Stage 1: deterministic quality gate (0 LLM calls) ───────────────────────────

def quality_gate(result: dict) -> tuple[bool, list[str]]:
    """Return (passes, reasons). `passes` is True only when every internal-quality
    check holds; `reasons` lists each failure in human-readable form.

    Reads only fields debate.run() already computed — costs nothing.
    """
    reasons: list[str] = []

    # Agents must have actually agreed — a wide final spread means low conviction.
    spread = result.get("score_spread")
    if not result.get("converged", False):
        reasons.append(f"agents did not converge (spread {spread})")
    elif isinstance(spread, (int, float)) and spread > _max_spread():
        reasons.append(f"score spread {spread} > {_max_spread()}")

    # Confidence + full panel.
    if result.get("confidence") == "LOW":
        reasons.append("debate confidence LOW")
    failed = result.get("failed_agents") or []
    if failed:
        reasons.append(f"{len(failed)} agent(s) failed: {', '.join(failed)}")

    # Risk/reward layer.
    rr = result.get("risk_reward") or {}
    if rr.get("applied"):
        ri = rr.get("risk_index")
        if isinstance(ri, (int, float)) and ri > _max_risk_index():
            reasons.append(f"risk index {ri} > {_max_risk_index()}")
        xc = rr.get("llm_cross_check") or {}
        if xc.get("divergent") is True:
            reasons.append("agents' bull/bear targets diverge from computed R:R")
    # (no R:R = not a disqualifier on its own; many valid BUYs lack a clean FV)

    # Dossier data quality.
    if result.get("data_confidence") == "LOW":
        reasons.append("dossier data confidence LOW")

    return (len(reasons) == 0, reasons)


# ── Stage 2: adversarial red-team prosecutor (1 grounded LLM call) ───────────────

RED_TEAM_SYSTEM = """\
You are a forensic short-seller and adversarial investment skeptic. A multi-agent
research panel has rated the stock below a BUY. Your ONLY job is to try to PROVE
THEM WRONG before real money is committed.

Use Google Search to hunt for DISCONFIRMING evidence the panel may have missed or
under-weighted — recent guidance cuts, analyst downgrades, earnings misses,
deteriorating margins or cash flow, rising debt, dilution, insider selling,
litigation, regulatory/accounting red flags, competitive threats, or a valuation
that already prices in perfection. Attack the single load-bearing assumption (the
"key swing factor") hardest.

Be intellectually honest: if after genuine effort you CANNOT find a thesis-breaking
flaw, you must CONFIRM. Do not invent weakness, and do not rubber-stamp.

Verdicts:
  CONFIRM   — the bull thesis survives scrutiny; no material disconfirming evidence.
  DOWNGRADE — real concerns that materially weaken conviction, but not fatal.
  VETO      — concrete disconfirming evidence that breaks the thesis.

Output STRICT JSON only — no markdown, no prose outside the JSON:
{
  "verdict": "CONFIRM" | "DOWNGRADE" | "VETO",
  "verification_score": <0-10 number, your independent conviction in the BUY>,
  "confirms_buy": <true only if verdict is CONFIRM>,
  "strongest_bear_point": "<the single most damaging finding, one sentence>",
  "falsification_findings": ["<concrete disconfirming fact + source/date>", "..."]
}
"""

_RED_TEAM_USER = """\
TICKER: {ticker}{name}
Current price: {price}   Market cap: {mcap}

PANEL VERDICT TO STRESS-TEST:
  Consensus: {score}/10 [{grade}]  (confidence {confidence})
  Bull thesis: {thesis}
  Key swing factor (attack this hardest): {swing}
  Stated catalyst: {catalyst}
  Stated asymmetry: {asymmetry}
  Risk/reward: {rr}

KEY FUNDAMENTALS (from the panel's dossier):
{fundamentals}

AGENT FINAL SCORES: {agent_scores}

Do your own fresh research. Try to break this BUY. Return the JSON verdict.
"""


def _fmt(v, suffix: str = "") -> str:
    if v is None or v == "":
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2f}{suffix}"
    return f"{v}{suffix}"


def _build_fundamentals(dossier: dict) -> str:
    """Compact, prosecutor-relevant slice of the dossier (kept small on purpose)."""
    fin = (dossier.get("financials") or {})
    r = (fin.get("ratios_ttm") or {})
    fv = (dossier.get("fair_values") or {})
    rows = [
        ("P/E (ttm)",        _fmt(r.get("pe"))),
        ("Fwd P/E",          _fmt(r.get("fwd_pe"))),
        ("P/S",              _fmt(r.get("ps"))),
        ("EV/EBITDA",        _fmt(r.get("ev_ebitda"))),
        ("ROIC",             _fmt(r.get("roic"))),
        ("Debt/Equity",      _fmt(r.get("debt_equity"))),
        ("Net margin",       _fmt(r.get("net_margin"))),
        ("Rev growth (fwd)", _fmt(r.get("fwd_revenue_growth"))),
        ("FCF/share",        _fmt(r.get("fcf_per_share"))),
        ("Composite FV",     _fmt(fv.get("composite_fair_value"))),
    ]
    return "\n".join(f"  {k}: {v}" for k, v in rows)


def _build_user_prompt(ticker: str, result: dict, dossier: dict) -> str:
    profile = (dossier.get("profile") or {})
    quote = (dossier.get("quote") or {})
    rr = (result.get("risk_reward") or {})
    rr_str = "n/a"
    if rr.get("applied"):
        rr_str = (f"{_fmt(rr.get('rr_ratio'))}:1 · {rr.get('risk_tier','?')} risk · "
                  f"upside {_fmt(rr.get('upside_pct'))}% / downside {_fmt(rr.get('downside_pct'))}%")
    name = profile.get("name")
    return _RED_TEAM_USER.format(
        ticker=ticker,
        name=f" ({name})" if name else "",
        price=_fmt(quote.get("price")),
        mcap=_fmt(profile.get("market_cap_bn") or result.get("market_cap_bn"), "B"),
        score=_fmt(result.get("consensus_score")),
        grade=result.get("consensus_grade", "?"),
        confidence=result.get("confidence", "?"),
        thesis=(result.get("majority_thesis") or "")[:700],
        swing=(result.get("key_swing_factor") or "")[:300],
        catalyst=(result.get("catalyst") or "")[:300],
        asymmetry=result.get("asymmetry_ratio") or "n/a",
        rr=rr_str,
        fundamentals=_build_fundamentals(dossier),
        agent_scores=result.get("agent_final_scores", {}),
    )


async def red_team(ticker: str, result: dict, dossier: dict) -> dict:
    """Run the adversarial prosecutor. Returns a normalized verdict dict, or
    {"_error": "..."} on failure (verify_buy decides fail-open vs fail-closed)."""
    try:
        from llm import call_gemini_async, extract_json
    except ImportError:
        return {"_error": "llm module unavailable"}

    prompt = _build_user_prompt(ticker, result, dossier)
    try:
        text = await asyncio.wait_for(
            call_gemini_async(RED_TEAM_SYSTEM, prompt, grounding=True, temperature=0.2),
            timeout=_RED_TEAM_TIMEOUT,
        )
        raw = extract_json(text)
    except Exception as exc:
        return {"_error": str(exc)}

    if not isinstance(raw, dict):
        return {"_error": "red-team response was not a JSON object"}

    verdict = str(raw.get("verdict", "")).upper().strip()
    if verdict not in ("CONFIRM", "DOWNGRADE", "VETO"):
        return {"_error": f"unrecognized verdict {verdict!r}"}

    try:
        vscore = float(raw.get("verification_score")) if raw.get("verification_score") is not None else None
    except (TypeError, ValueError):
        vscore = None

    findings = raw.get("falsification_findings")
    if not isinstance(findings, list):
        findings = [str(findings)] if findings else []

    return {
        "verdict": verdict,
        "verification_score": vscore,
        "confirms_buy": verdict == "CONFIRM",
        "strongest_bear_point": str(raw.get("strongest_bear_point", ""))[:400],
        "falsification_findings": [str(f)[:300] for f in findings[:6]],
    }


# ── Orchestration ───────────────────────────────────────────────────────────────

async def verify_buy(ticker: str, result: dict, dossier: dict) -> dict:
    """Two-stage confirmation gate. Returns a verification block (always includes
    a `confirmed` bool). Never raises.

    Stage 1 failure short-circuits — no LLM call is spent on an internally-shaky
    BUY; it goes straight to the watchlist with concrete reasons.
    """
    now = datetime.now(timezone.utc).isoformat()

    if not _enabled():
        return {"confirmed": True, "stage": 0, "verdict": "DISABLED",
                "reasons": [], "checked_at": now}

    stage1_pass, reasons = quality_gate(result)
    if not stage1_pass:
        return {
            "confirmed": False,
            "stage": 1,
            "stage1_pass": False,
            "verdict": "REJECTED_STAGE1",
            "reasons": reasons,
            "verification_score": None,
            "strongest_bear_point": reasons[0] if reasons else "",
            "falsification_findings": [],
            "checked_at": now,
        }

    rt = await red_team(ticker, result, dossier)

    if "_error" in rt:
        fail_open = _fail_open()
        return {
            "confirmed": fail_open,
            "stage": 2,
            "stage1_pass": True,
            "verdict": "UNVERIFIED",
            "reasons": [] if fail_open else ["red-team unavailable; fail-closed"],
            "verification_score": None,
            "strongest_bear_point": "",
            "falsification_findings": [],
            "note": f"red-team unavailable ({rt['_error']})",
            "checked_at": now,
        }

    return {
        "confirmed": rt["confirms_buy"],
        "stage": 2,
        "stage1_pass": True,
        "verdict": rt["verdict"],
        "reasons": [] if rt["confirms_buy"] else [rt["strongest_bear_point"]],
        "verification_score": rt["verification_score"],
        "strongest_bear_point": rt["strongest_bear_point"],
        "falsification_findings": rt["falsification_findings"],
        "checked_at": now,
    }
