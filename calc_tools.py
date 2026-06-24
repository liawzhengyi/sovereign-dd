"""Gemini function-calling tool declarations for gems triage."""

from __future__ import annotations

from typing import Any

from google.genai import types


# ── Tool callables ──────────────────────────────────────────────────────────

def compute_pillar_scores(ticker: str, candidates: dict[str, dict]) -> dict:
    """Compute pillar scores for a ticker from pre-loaded fundament data.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL')
        candidates: dict mapping ticker -> fundament dict (pre-loaded Finviz data)

    Returns:
        Pillar score dict from pillar_scoring.compute_composite(), or
        {"error": "ticker not found"} if ticker not in candidates.
    """
    from pillar_scoring import compute_composite
    fundament = (
        candidates.get(ticker.upper())
        or candidates.get(ticker.lower())
        or candidates.get(ticker)
    )
    if fundament is None:
        return {"error": f"ticker '{ticker}' not found in candidates"}
    return compute_composite(fundament)


def rank_candidates(
    candidates: dict[str, dict],
    scores: dict[str, dict],
    metric: str,
    top_n: int = 10,
    direction: str = "desc",
) -> list[dict]:
    """Rank candidates by a specific pillar metric.

    Args:
        candidates: dict mapping ticker -> fundament dict
        scores: dict mapping ticker -> compute_composite() result
        metric: one of 'composite', 'financial_physics', 'moat_proxy',
                'temporal', 'management', 'chokepoint_proxy'
        top_n: number of results to return (default 10)
        direction: 'desc' (highest first) or 'asc' (lowest first)

    Returns:
        List of {ticker, score, industry, sector} dicts sorted by metric.
    """
    valid_metrics = {
        "composite", "financial_physics", "moat_proxy",
        "temporal", "management", "chokepoint_proxy"
    }
    if metric not in valid_metrics:
        metric = "composite"

    rows = []
    for ticker, score_dict in scores.items():
        val = score_dict.get(metric)
        if val is None:
            continue
        fundament = candidates.get(ticker, {})
        rows.append({
            "ticker": ticker,
            "score": round(float(val), 2),
            "industry": fundament.get("Industry", ""),
            "sector": fundament.get("Sector", ""),
        })

    reverse = (direction != "asc")
    rows.sort(key=lambda r: r["score"], reverse=reverse)
    return rows[:top_n]


def compare_candidates(
    ticker_a: str,
    ticker_b: str,
    candidates: dict[str, dict],
    scores: dict[str, dict],
) -> dict:
    """Side-by-side comparison of two tickers across all pillar scores.

    Returns:
        Dict with both tickers' full score breakdowns and fundament highlights,
        or error if either ticker not found.
    """
    result = {}
    for ticker in (ticker_a.upper(), ticker_b.upper()):
        fundament = candidates.get(ticker) or candidates.get(ticker.lower())
        score_dict = scores.get(ticker) or scores.get(ticker.lower())
        if fundament is None or score_dict is None:
            result[ticker] = {"error": f"ticker '{ticker}' not found"}
        else:
            result[ticker] = {
                "scores": score_dict,
                "sector": fundament.get("Sector", ""),
                "industry": fundament.get("Industry", ""),
                "market_cap": fundament.get("Market Cap", ""),
                "gross_margin": fundament.get("Gross Margin", ""),
                "roic": fundament.get("ROIC", ""),
                "eps_5y": fundament.get("EPS past 3/5Y", ""),
            }
    return result


# ── Gemini SDK Tool declarations ────────────────────────────────────────────

TRIAGE_TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="compute_pillar_scores",
            description="Compute the 5 pillar quality scores for a given stock ticker.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "ticker": types.Schema(
                        type="STRING",
                        description="Stock ticker symbol, e.g. 'AAPL'",
                    ),
                },
                required=["ticker"],
            ),
        ),
        types.FunctionDeclaration(
            name="rank_candidates",
            description="Rank all screened candidates by a pillar metric and return top N.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "metric": types.Schema(
                        type="STRING",
                        description=(
                            "Pillar to rank by. One of: composite, financial_physics, "
                            "moat_proxy, temporal, management, chokepoint_proxy"
                        ),
                    ),
                    "top_n": types.Schema(
                        type="INTEGER",
                        description="How many results to return (default 10)",
                    ),
                    "direction": types.Schema(
                        type="STRING",
                        description="'desc' for highest first (default), 'asc' for lowest first",
                    ),
                },
                required=["metric"],
            ),
        ),
        types.FunctionDeclaration(
            name="compare_candidates",
            description="Compare two candidate tickers side-by-side across all pillar scores.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "ticker_a": types.Schema(
                        type="STRING",
                        description="First ticker symbol",
                    ),
                    "ticker_b": types.Schema(
                        type="STRING",
                        description="Second ticker symbol",
                    ),
                },
                required=["ticker_a", "ticker_b"],
            ),
        ),
    ]
)


# ── Tool executor ────────────────────────────────────────────────────────────

def execute_tool_call(
    fn_name: str,
    fn_args: dict,
    candidates: dict[str, dict],
    scores: dict[str, dict],
) -> Any:
    """Dispatch a model-issued function call to the correct Python callable.

    Args:
        fn_name: Name of the function the model wants to call.
        fn_args: Arguments dict from the model's function call.
        candidates: The full candidates dict (ticker -> fundament).
        scores: Pre-computed pillar scores (ticker -> score dict).

    Returns:
        JSON-serializable result, or {"error": "unknown tool"} for unknown names.
    """
    if fn_name == "compute_pillar_scores":
        return compute_pillar_scores(
            ticker=fn_args.get("ticker", ""),
            candidates=candidates,
        )
    elif fn_name == "rank_candidates":
        return rank_candidates(
            candidates=candidates,
            scores=scores,
            metric=fn_args.get("metric", "composite"),
            top_n=int(fn_args.get("top_n", 10)),
            direction=fn_args.get("direction", "desc"),
        )
    elif fn_name == "compare_candidates":
        return compare_candidates(
            ticker_a=fn_args.get("ticker_a", ""),
            ticker_b=fn_args.get("ticker_b", ""),
            candidates=candidates,
            scores=scores,
        )
    else:
        return {"error": f"unknown tool '{fn_name}'"}
