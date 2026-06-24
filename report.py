"""Rich terminal report renderer."""

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

from grading import GRADE_COLORS, grade as _grade

AGENT_COLORS = {
    "StructuralEdge":       "bright_yellow",
    "FundamentalForensics": "cyan",
    "ValuationEngine":      "bright_green",
    "CatalystHunter":       "magenta",
    "MarketStructure":      "bright_blue",
    "Moderator":            "white",
}


def _score_bar(score: float, width: int = 30) -> str:
    filled = int(round(score / 10 * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.1f}/10"


def render(result: dict, dossier: dict) -> None:
    ticker = result["ticker"]
    score = result["consensus_score"]
    grade = result["consensus_grade"]
    profile = dossier.get("profile", {})
    quote = dossier.get("quote", {})

    grade_color = GRADE_COLORS.get(grade, "white")

    # ── Header ─────────────────────────────────────────────────────────────
    price = quote.get("price") or 0
    chg = quote.get("change_pct") or 0
    chg_str = f"+{chg:.2f}%" if chg >= 0 else f"{chg:.2f}%"
    chg_color = "green" if chg >= 0 else "red"

    header = Text()
    header.append(f"  {ticker}  ", style="bold white on dark_blue")
    header.append(f"  {profile.get('name', ticker)}\n", style="bold white")
    header.append(f"  {profile.get('sector', '')}  ·  ", style="dim")
    header.append(f"${price:.2f}  ", style="bold white")
    header.append(chg_str, style=chg_color)
    header.append(f"  ·  Market Cap ${profile.get('market_cap_bn', 0):.1f}B", style="dim")

    console.print(Panel(header, title="[bold]SOVEREIGN DD[/bold]", border_style="blue"))

    # ── Consensus Score ────────────────────────────────────────────────────
    score_text = Text()
    score_text.append(f"\n  CONSENSUS SCORE  ", style="bold white")
    score_text.append(f"{score:.2f} / 10\n", style=f"bold {grade_color}")

    # Show raw score if it differs from adjusted
    raw_score = result.get("raw_consensus_score")
    if raw_score is not None and abs(raw_score - score) >= 0.01:
        score_text.append(f"  (raw: {raw_score:.2f} → adjusted: {score:.2f})\n", style="dim")

    score_text.append(f"  {_score_bar(score)}\n", style=grade_color)
    score_text.append(f"\n  GRADE: ", style="bold white")
    score_text.append(f"{grade}", style=f"bold {grade_color}")
    score_text.append(f"   CONFIDENCE: {result.get('confidence', '?')}", style="yellow")

    # Banger tag
    banger = result.get("banger", {})
    if isinstance(banger, dict) and banger.get("is_banger"):
        score_text.append(f"   🔥 BANGER\n", style="bold bright_yellow")
    else:
        score_text.append(f"\n", style="")

    # Position guidance
    pos = result.get("position_guidance", {})
    if isinstance(pos, dict) and pos.get("range"):
        score_text.append(f"  Suggested allocation: {pos['range']}  ({pos.get('reasoning', '')})\n", style="dim")

    score_text.append(f"\n  {result.get('majority_thesis', '')}\n", style="white")

    console.print(Panel(score_text, title="[bold]Investment Verdict[/bold]",
                        border_style=grade_color))

    # ── Catalyst & Asymmetry ───────────────────────────────────────────────────
    catalyst = result.get("catalyst", "")
    asymmetry = result.get("asymmetry_ratio", "")
    rr = result.get("risk_reward") or {}
    cycle_pos = result.get("cycle_position", {})
    moat_comp = result.get("moat_composite")

    if catalyst or asymmetry or cycle_pos or moat_comp or rr.get("applied"):
        parts = []
        if catalyst:
            parts.append(f"[bold]Catalyst:[/bold] {escape(catalyst[:200])}")
        if asymmetry:
            parts.append(f"[bold]Asymmetry:[/bold] {escape(str(asymmetry))}")
        if rr.get("applied"):
            parts.append(
                f"[bold]R/R (computed):[/bold] {rr.get('rr_ratio', 0):.1f}:1"
                f"  ·  risk {rr.get('risk_tier','?')} ({rr.get('risk_index','?')}/10)"
                f"  ·  reward {rr.get('reward_tier','?')}"
            )
        if isinstance(cycle_pos, dict) and cycle_pos.get("phase"):
            parts.append(f"[bold]Cycle:[/bold] {escape(cycle_pos.get('regime', ''))} — {escape(cycle_pos.get('phase', ''))}"
                         f"  ({escape(cycle_pos.get('evidence', '')[:100])})")
        if moat_comp is not None:
            parts.append(f"[bold]Moat:[/bold] {moat_comp:.1f}/10" if isinstance(moat_comp, (int, float)) else f"[bold]Moat:[/bold] {moat_comp}")
        console.print(Panel("\n".join(parts), title="[bold]Opportunity Profile[/bold]", border_style="bright_yellow"))

    # ── Fair Value & Entry ─────────────────────────────────────────────────────
    fv = result.get("fair_value_composite")
    entry = result.get("entry_assessment", "")
    price_now = quote.get("price") or 0

    if fv or entry:
        fv_parts = []
        if fv is not None:
            try:
                fv_float = float(fv)
                mos = (fv_float - price_now) / fv_float * 100 if (fv_float > 0 and price_now > 0) else None
                if mos is not None:
                    mos_color = "green" if mos >= 15 else ("yellow" if mos >= 0 else "red")
                    mos_str = f"{mos:+.1f}%"
                    fv_parts.append(
                        f"[bold]Fair Value:[/bold] ${fv_float:.2f}  "
                        f"[bold]MoS:[/bold] [{mos_color}]{mos_str}[/{mos_color}] vs ${price_now:.2f}"
                    )
                else:
                    fv_parts.append(f"[bold]Fair Value:[/bold] ${fv_float:.2f}")
            except (ValueError, TypeError):
                fv_parts.append(f"[bold]Fair Value:[/bold] {escape(str(fv))}")
        if entry:
            entry_color = "green" if "ENTER_NOW" in str(entry).upper() else (
                "yellow" if "WAIT" in str(entry).upper() else "red"
            )
            fv_parts.append(f"[bold]Entry:[/bold] [{entry_color}]{escape(str(entry))}[/{entry_color}]")
        if fv_parts:
            console.print(Panel("\n".join(fv_parts), title="[bold]Fair Value & Entry[/bold]", border_style="cyan"))

    # ── Agent Score Table ──────────────────────────────────────────────────
    loops = result.get("loops_run", 1)
    table = Table(title=f"Agent Scores — R1 → Final ({loops} debate loop(s))",
                  box=box.SIMPLE_HEAVY, show_header=True,
                  header_style="bold white")
    table.add_column("Agent", style="bold", width=14)
    table.add_column("R1", justify="right", width=6)
    table.add_column("Final", justify="right", width=6)
    table.add_column("Δ", justify="right", width=6)
    table.add_column("Bar", width=22)
    table.add_column("Grade", width=12)

    r1 = result.get("agent_r1_scores", {})
    rf = result.get("agent_final_scores", result.get("agent_r3_scores", {}))

    for agent in r1:
        s1 = r1[agent]
        sf = rf.get(agent, s1)
        delta = sf - s1
        color = AGENT_COLORS.get(agent, "white")
        delta_str = f"+{delta:.1f}" if delta > 0 else (f"{delta:.1f}" if delta != 0 else "  —")
        delta_color = "green" if delta > 0 else ("red" if delta < 0 else "dim")
        table.add_row(
            Text(agent, style=color),
            f"{s1:.1f}",
            Text(f"{sf:.1f}", style="bold"),
            Text(delta_str, style=delta_color),
            Text(_score_bar(sf, 14), style=GRADE_COLORS.get(_grade(sf), "white")),
            Text(_grade(sf), style=GRADE_COLORS.get(_grade(sf), "white")),
        )

    console.print(table)

    # ── Score Decomposition ────────────────────────────────────────────────────
    decomp = result.get("score_decomposition", {})
    adj_dict = result.get("score_adjustments", {})
    if decomp or adj_dict:
        dtable = Table(title="Score Decomposition", box=box.SIMPLE_HEAVY,
                       show_header=True, header_style="bold white")
        dtable.add_column("Agent", style="bold", width=14)
        dtable.add_column("StructMoat", justify="right", width=10)
        dtable.add_column("FundQual", justify="right", width=9)
        dtable.add_column("ValGap", justify="right", width=8)
        dtable.add_column("CatalRisk", justify="right", width=9)
        dtable.add_column("MktStruct", justify="right", width=9)

        for agent, bd in decomp.items():
            if not isinstance(bd, dict):
                continue
            color = AGENT_COLORS.get(agent, "white")
            dtable.add_row(
                Text(agent, style=color),
                f"{bd.get('structural_moat', '—')}",
                f"{bd.get('fundamental_quality', '—')}",
                f"{bd.get('valuation_gap', '—')}",
                f"{bd.get('catalyst_risk', '—')}",
                f"{bd.get('market_structure', '—')}",
            )

        if adj_dict:
            adj_lines = []
            if "earnings_durability" in adj_dict:
                ed = adj_dict["earnings_durability"]
                adj_lines.append(f"  Earnings durability: {ed.get('label','?')} ({ed.get('score','?')}/10) → {ed.get('result', 0.0):.2f}")
            if adj_dict.get("consensus_gap", {}).get("applied"):
                cg = adj_dict["consensus_gap"]
                adj_lines.append(f"  Consensus gap: {cg.get('gap_pct', 0.0):.1f}% ({cg.get('label','?')}) → {cg.get('result', 0.0):.2f}")
            if adj_dict.get("cycle_position", {}).get("applied"):
                cp = adj_dict["cycle_position"]
                adj_lines.append(f"  Cycle position: {cp.get('reason','?')} ({cp.get('adjustment',0):+.1f}) → {cp.get('result', 0.0):.2f}")
            if adj_dict.get("risk_reward", {}).get("applied"):
                rra = adj_dict["risk_reward"]
                adj_lines.append(
                    f"  Risk/reward: {rra.get('quadrant','?')} ({rra.get('adjustment',0):+.2f}) → {rra.get('result', 0.0):.2f}"
                )
            if adj_dict.get("data_confidence", {}).get("applied"):
                adj_lines.append(f"  Data confidence penalty: -0.5")

            if adj_lines:
                console.print("\n[dim]Adjustments:[/dim]")
                for line in adj_lines:
                    console.print(f"[dim]{line}[/dim]")

        if decomp:
            console.print(dtable)

    # ── Key Factors ────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold]Key Swing Factor:[/bold] {escape(str(result.get('key_swing_factor', '—')))}\n\n"
        f"[bold]Rationale:[/bold] {escape(str(result.get('score_rationale', '—')))}\n\n"
        f"[bold]Dissent:[/bold] {escape(str(result.get('dissent', '—')))}",
        title="[bold]Debate Summary[/bold]", border_style="dim white",
    ))

    # ── Macro Snapshot ─────────────────────────────────────────────────────
    macro = dossier.get("macro", {})
    macro_str = (
        f"Fed Funds: {macro.get('fed_funds_rate', '?')}%  ·  "
        f"10Y: {macro.get('treasury_10y', '?')}%  ·  "
        f"2Y: {macro.get('treasury_2y', '?')}%  ·  "
        f"Spread: {macro.get('yield_curve_spread', '?')}  ·  "
        f"CPI: {macro.get('cpi_yoy', '?')}  ·  "
        f"VIX: {macro.get('vix', '?')}"
    )
    console.print(f"\n[dim]Macro:[/dim] {macro_str}")
    converged = result.get("converged", True)
    loops = result.get("loops_run", "?")
    conv_str = f"Converged in {loops} loop(s)" if converged else f"Moderator invoked after {loops} loop(s)"
    console.print(f"[dim]{conv_str}  ·  Score spread: {result.get('score_spread', '?')}[/dim]\n")


