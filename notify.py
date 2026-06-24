"""Telegram notifications for sovereign-dd — routes to specific topics."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# Topic (thread) IDs — send to the relevant channel
TOPIC_TRADE_ALERTS  = os.getenv("TELEGRAM_TOPIC_TRADE_ALERTS", "")
TOPIC_DEEP_DIVES    = os.getenv("TELEGRAM_TOPIC_DEEP_DIVES", "")
TOPIC_SCAN_RESULTS  = os.getenv("TELEGRAM_TOPIC_SCAN_RESULTS", "")
# Under-Review watchlist — confirmed-gate rejects. Falls back to Scan Results
# until a dedicated topic ID is set in TELEGRAM_TOPIC_WATCHLIST.
TOPIC_WATCHLIST     = os.getenv("TELEGRAM_TOPIC_WATCHLIST", "") or TOPIC_SCAN_RESULTS

GRADE_EMOJI = {
    "CONVICTION BUY": "🟢🟢🟢",
    "STRONG BUY":     "🟢🟢",
    "BUY":            "🟢",
    "HOLD":           "🟡",
    "SELL":           "🔴",
    "STRONG SELL":    "🔴🔴",
    "AVOID":          "🔴🔴🔴",
    # Hold-mode labels (ADD/HOLD/TRIM/EXIT) — mirror GRADE_COLORS in grading.py
    "ADD":            "🟢🟢",
    "TRIM":           "🔴",
    "EXIT":           "🔴🔴",
}

CONF_EMOJI = {"HIGH": "⭐⭐⭐", "MEDIUM": "⭐⭐", "LOW": "⭐"}


_TG_MAX = 4096


def _send(message: str, topic_id: str = "") -> bool:
    """Send a single message to the Telegram bot (caller must ensure len ≤ 4096)."""
    if not BOT_TOKEN or not CHAT_ID:
        print("  [notify] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — skipping")
        return False
    payload: dict = {
        "chat_id":    CHAT_ID,
        "text":       message,
        "parse_mode": "HTML",
    }
    if topic_id:
        try:
            payload["message_thread_id"] = int(topic_id)
        except ValueError:
            print(f"  [notify] Invalid topic_id '{topic_id}' — sending to main chat")
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json=payload, timeout=15,
        )
        if not r.ok:
            print(f"  [notify] Telegram error {r.status_code}: {r.text[:200]}")
        return r.ok
    except Exception as e:
        print(f"  [notify] Telegram request failed: {e}")
        return False


def _split_send(message: str, topic_id: str = "") -> bool:
    """Send message, splitting into ≤4096-char chunks at newline boundaries."""
    if len(message) <= _TG_MAX:
        return _send(message, topic_id)

    parts: list[str] = []
    remaining = message
    while remaining:
        if len(remaining) <= _TG_MAX:
            parts.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, _TG_MAX)
        if split_at <= 0:
            split_at = _TG_MAX
        parts.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")

    ok = True
    for i, part in enumerate(parts):
        if i > 0:
            part = "↩ <i>(continued)</i>\n" + part
        ok = _send(part, topic_id) and ok
    return ok


def alert_buy_signal(d: dict) -> bool:
    """Send a BUY signal alert for a scout discovery → Trade Alerts topic."""
    emoji = GRADE_EMOJI.get(d["grade"], "")
    conf  = CONF_EMOJI.get(d.get("confidence", ""), "")
    lens      = d.get("scout_lens", "")
    rationale = d.get("gemma_rationale", "")
    score_rat = d.get("score_rationale", "")
    dissent   = d.get("dissent", "")
    catalyst  = d.get("catalyst", "")
    asymmetry = d.get("asymmetry_ratio", "")
    banger    = d.get("banger", {})
    pos       = d.get("position_guidance", {})
    cycle_pos = d.get("cycle_position", {})

    filters   = d.get("matched_filters", [])
    path      = d.get("path", "")

    lens_tag   = f" · <code>{lens}</code>" if lens else ""
    path_tag   = f" [PATH {path}]" if path else ""
    banger_tag = "\n🔥 <b>BANGER</b> — " + banger.get("reason","")[:150] if isinstance(banger, dict) and banger.get("is_banger") else ""
    penalty    = score_rat or dissent
    filter_line = f"<b>Filters met:</b> {' · '.join(filters)}\n" if filters else ""

    msg = (
        f"{emoji} <b>BUY SIGNAL — {d['ticker']}</b>{path_tag}{lens_tag}\n"
        f"<b>Score:</b> {d['score']:.1f}/10 · {d['grade']} · {conf}\n"
        + filter_line
        + (f"<b>Gemma flagged:</b> <i>{rationale[:350]}</i>\n" if rationale else "")
        + (f"\n<b>Catalyst:</b> <i>{catalyst[:400]}</i>\n" if catalyst else "")
        + (f"<b>Asymmetry:</b> {asymmetry}\n" if asymmetry else "")
        + (f"<b>R/R:</b> {d['rr']:.1f}:1 ({d.get('risk','?')} risk)\n" if d.get("rr") is not None else "")
        + (f"<b>Cycle:</b> {cycle_pos.get('regime','')} — {cycle_pos.get('phase','')}\n" if isinstance(cycle_pos, dict) and cycle_pos.get("phase") else "")
        + f"\n<b>Bull case:</b> <i>{d['thesis'][:600]}</i>\n\n"
        + (f"<b>Why not higher:</b> <i>{penalty[:450]}</i>\n\n" if penalty else "")
        + f"<b>Key factor:</b> {d.get('key_swing_factor', '—')[:300]}\n"
        + (f"<b>Position:</b> {pos.get('range','?')} ({pos.get('reasoning','')[:200]})\n" if isinstance(pos, dict) and pos.get("range") else "")
        + banger_tag
        + f"\n⏰ {d['analyzed_at']}"
    )
    return _split_send(msg, TOPIC_TRADE_ALERTS)


def alert_watchlist(d: dict) -> bool:
    """Send an 'Under Review' alert for a BUY that crossed the threshold but
    failed the confirmation gate → Watchlist topic (falls back to Scan Results).

    Leads with the reason it was held back (Stage-1 quality reasons or the
    red-team's strongest bear point) so it reads as a near-miss, not a signal."""
    v = d.get("verification", {}) or {}
    verdict  = v.get("verdict", "?")
    vscore   = v.get("verification_score")
    bear     = v.get("strongest_bear_point", "")
    reasons  = v.get("reasons", []) or []
    findings = v.get("falsification_findings", []) or []

    if v.get("stage") == 1:
        held = "Failed internal quality checks — " + "; ".join(reasons)
    else:
        held = bear or (reasons[0] if reasons else "Flagged by red-team review")

    conf        = CONF_EMOJI.get(d.get("confidence", ""), "")
    lens        = d.get("scout_lens", "")
    lens_tag    = f" · <code>{lens}</code>" if lens else ""
    vscore_line = f" · verify {vscore:.1f}/10" if isinstance(vscore, (int, float)) else ""

    findings_block = ""
    if findings:
        findings_block = ("\n<b>Disconfirming findings:</b>\n"
                          + "\n".join(f"• <i>{str(f)[:200]}</i>" for f in findings[:4]) + "\n")

    msg = (
        f"⚠️ <b>UNDER REVIEW — {d['ticker']}</b>{lens_tag}\n"
        f"<b>Score:</b> {d['score']:.1f}/10 · {d.get('grade','?')} · {conf} · <b>{verdict}</b>{vscore_line}\n"
        f"<i>Crossed the BUY line but failed the confirmation gate — not a confirmed signal.</i>\n\n"
        f"<b>Why held back:</b> {held[:500]}\n"
        + findings_block
        + (f"<b>R/R:</b> {d['rr']:.1f}:1 ({d.get('risk','?')} risk)\n" if d.get("rr") is not None else "")
        + f"\n<b>Bull case (unconfirmed):</b> <i>{d.get('thesis','')[:450]}</i>\n"
        + f"\n⏰ {d.get('analyzed_at','')}"
    )
    return _split_send(msg, TOPIC_WATCHLIST)


def alert_dd_result(result: dict) -> bool:
    """Send a single DD result to the Deep Dives topic."""
    ticker = result.get("ticker", "?")
    score  = result.get("consensus_score", 0)
    grade  = result.get("consensus_grade", "HOLD")
    conf   = result.get("confidence", "")
    emoji  = GRADE_EMOJI.get(grade, "")
    cconf  = CONF_EMOJI.get(conf, "")

    agent_lines = []
    r1 = result.get("agent_r1_scores", {})
    rf = result.get("agent_final_scores", {})
    for agent, s1 in r1.items():
        sf    = rf.get(agent, s1)
        delta = sf - s1
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "→")
        agent_lines.append(f"  {agent:<14} {s1:.1f} {arrow} {sf:.1f}")

    agents_block = "\n".join(agent_lines)
    thesis = result.get("majority_thesis", "")[:700]

    catalyst = result.get("catalyst", "")
    banger = result.get("banger", {})
    banger_line = f"\n🔥 <b>BANGER</b> — {banger.get('reason','')[:250]}" if isinstance(banger, dict) and banger.get("is_banger") else ""

    rrd = result.get("risk_reward") or {}
    rr_line = (
        f"<b>R/R:</b> {rrd['rr_ratio']:.1f}:1 ({rrd.get('risk_tier','?')} risk)\n"
        if rrd.get("applied") and rrd.get("rr_ratio") is not None else ""
    )

    msg = (
        f"{emoji} <b>SOVEREIGN DD — {ticker}</b>\n"
        f"<b>Score:</b> {score:.2f}/10 · {grade} · {cconf}\n"
        + rr_line + "\n"
        f"<pre>{agents_block}</pre>\n\n"
        + (f"<b>Catalyst:</b> <i>{catalyst[:400]}</i>\n\n" if catalyst else "")
        + f"<i>{thesis}</i>"
        + banger_line
    )
    return _split_send(msg, TOPIC_DEEP_DIVES)


def alert_portfolio_summary(results: list[dict]) -> bool:
    """Send a pre-market portfolio scan summary → Scan Results topic."""
    sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
    lines = ["📊 <b>SOVEREIGN DD — Pre-Market Portfolio Scan</b>\n"]
    for r in sorted_results:
        emoji = GRADE_EMOJI.get(r.get("grade", ""), "")
        score = r.get("score", 0)
        grade = r.get("grade", "?")
        lines.append(f"{emoji} <b>{r['ticker']:<6}</b>  {score:.1f}/10  {grade}")
    lines.append("\n🕐 Analysis complete — check Deep Dives for full reports")
    return _split_send("\n".join(lines), TOPIC_SCAN_RESULTS)


def alert_scout_summary(discoveries: list[dict]) -> bool:
    """Send scout run summary → Scan Results topic."""
    if not discoveries:
        msg = "🔍 <b>SOVEREIGN SCOUT</b>\n\nNo BUY signals found in today's scan."
    else:
        lines = [f"🔍 <b>SOVEREIGN SCOUT — {len(discoveries)} signal(s) found</b>\n"]
        for d in discoveries:
            emoji = GRADE_EMOJI.get(d["grade"], "")
            lines.append(f"{emoji} <b>{d['ticker']}</b>  {d['score']:.1f}/10")
        lines.append("\nFull reports sent to Deep Dives ↑")
        msg = "\n".join(lines)
    return _split_send(msg, TOPIC_SCAN_RESULTS)
