"""One-off Telegram forum-topic organizer for the Sovereign Engine group.

Renames the existing topics to a clean scheme and creates a new "Under Review"
topic for the BUY-confirmation-gate watchlist. Renaming changes DISPLAY NAMES
only — message_thread_ids are unchanged, so notify.py routing is unaffected.

Reads credentials from the environment (never hard-code them):
    TELEGRAM_BOT_TOKEN              (required — bot must be admin w/ Manage Topics)
    TELEGRAM_CHAT_ID               (required — the supergroup id)
    TELEGRAM_TOPIC_TRADE_ALERTS   (message_thread_id to rename)
    TELEGRAM_TOPIC_DEEP_DIVES
    TELEGRAM_TOPIC_SCAN_RESULTS

Run it where those values exist (locally, or as a GitHub Actions workflow_dispatch
that injects the repo secrets). After it prints the new "Under Review" topic id,
add it as the TELEGRAM_TOPIC_WATCHLIST GitHub secret so scheduled runs route
under-review alerts to the new topic (until then notify.py falls back to Scan
Results automatically).

Telegram Bot API: editForumTopic / createForumTopic.
Note: the Bot API cannot LIST or REORDER topics — it can only act on ids it is
given (+ General). Forward a message from any other topic to capture its id.
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

# Emoji in topic names crash the cp1252 Windows console on print — force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BOT = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# (env var holding the thread id) -> new display name
RENAMES = [
    ("TELEGRAM_TOPIC_TRADE_ALERTS", "🟢 Trade Alerts"),
    ("TELEGRAM_TOPIC_DEEP_DIVES",   "📊 Deep Dives"),
    ("TELEGRAM_TOPIC_SCAN_RESULTS", "🛰️ Scan Results"),
    ("TELEGRAM_TOPIC_EARNINGS",     "📅 Earnings"),
    ("TELEGRAM_TOPIC_SYSTEM",       "⚙️ System"),
]
NEW_TOPIC_NAME = "🔎 Under Review"


def _api(method: str, payload: dict) -> dict:
    r = requests.post(f"https://api.telegram.org/bot{BOT}/{method}", json=payload, timeout=15)
    try:
        data = r.json()
    except Exception:
        data = {"ok": False, "description": r.text[:200]}
    return data


def main() -> int:
    if not BOT or not CHAT:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in the environment.")
        return 2

    print(f"Group: {CHAT}\n")

    # 1. Rename existing topics.
    for env_var, new_name in RENAMES:
        thread_id = os.getenv(env_var, "").strip()
        if not thread_id:
            print(f"SKIP  {env_var} not set — can't rename to {new_name!r}")
            continue
        res = _api("editForumTopic", {
            "chat_id": CHAT,
            "message_thread_id": int(thread_id),
            "name": new_name,
        })
        ok = res.get("ok")
        print(f"{'OK   ' if ok else 'FAIL '} rename thread {thread_id} -> {new_name!r}"
              + ("" if ok else f"  ({res.get('description')})"))

    # 2. Create the new Under Review topic.
    res = _api("createForumTopic", {"chat_id": CHAT, "name": NEW_TOPIC_NAME})
    if res.get("ok"):
        new_id = res["result"]["message_thread_id"]
        print(f"\nOK    created {NEW_TOPIC_NAME!r} — message_thread_id = {new_id}")
        print(f"      → add this as the GitHub secret TELEGRAM_TOPIC_WATCHLIST = {new_id}")
    else:
        print(f"\nFAIL  createForumTopic ({res.get('description')})")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
