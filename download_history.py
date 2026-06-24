"""Restore scout history from Sovereign Eye KV when GitHub Actions cache is cold.

Called in CI before the scout run when actions/cache/restore reports a cache miss.
Writes output/scout_history.json and output/scout_notified.json so the dedup
window survives cache evictions, repository transfers, or manual cache clears.

Non-fatal — if the endpoint is unreachable, the scout run proceeds with empty history.
"""

import json
import os
import sys
from pathlib import Path

import requests

SOVEREIGN_EYE_URL = os.getenv("SOVEREIGN_EYE_URL", "https://sovereign-eye.pages.dev")
DD_UPLOAD_SECRET  = os.getenv("DD_UPLOAD_SECRET", "")


def main():
    if not DD_UPLOAD_SECRET:
        print("[history] DD_UPLOAD_SECRET not set — skipping KV restore")
        sys.exit(0)

    url = f"{SOVEREIGN_EYE_URL}/api/dd/history"
    print(f"[history] Fetching history from {url}...")
    try:
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {DD_UPLOAD_SECRET}"},
            timeout=30,
        )
        if not r.ok:
            print(f"[history] HTTP {r.status_code} — {r.text[:200]} — starting with empty history")
            sys.exit(0)
        data = r.json()
    except Exception as e:
        print(f"[history] Request failed: {e} — starting with empty history")
        sys.exit(0)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for field, fname, label in [
        ("history",  "scout_history.json",  "history"),
        ("notified", "scout_notified.json", "notify history"),
        ("gems",     "gems_history.json",   "gems history"),
        ("seen",     "scout_seen.json",     "rotation ledger"),
    ]:
        payload = data.get(field, {})
        if payload:
            (output_dir / fname).write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            print(f"[history] Restored {fname}  ({len(payload)} tickers)")
        else:
            print(f"[history] No {label} in KV — starting fresh")


if __name__ == "__main__":
    main()
