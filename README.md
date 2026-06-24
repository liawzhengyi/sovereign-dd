# Sovereign DD

Multi-agent stock due-diligence engine. Screens tickers, builds a fundamentals/news/
filings dossier, runs a 5-agent LLM debate, computes archetype valuations + pillar
scores, and uploads results to the **sovereign-eye** dashboard (Cloudflare KV + Supabase).

For the two-repo picture and the JSON/KV contract, see `sovereign-eye/ARCHITECTURE.md`
and `sovereign-eye/DATA_CONTRACT.md`.

## Modules

- `main.py` — entry point / orchestration (`--portfolio`, `--scout`, `--gems`, or a single ticker).
- `dossier.py` — fundamentals/macro/news/filings → one dossier.
- `agents.py` + `debate.py` — the 5-agent debate (research → R1 → R2 → R3 → consensus).
- `fair_value.py` — archetype valuations; `scoring.py` / `pillar_scoring.py` — scoring.
- `scout.py` / `gems.py` — screeners (Yahoo / Finviz) → triage → debate.
- `llm.py` — Gemini/Gemma client (multi-key rotation, free-tier quota handling).
- `upload_kv.py` — pushes results to sovereign-eye; `notify.py` — Telegram alerts.

## Run

```bash
cp .env.example .env        # fill in keys (gitignored)
pip install -r requirements.txt
python main.py AAPL                       # single ticker
python main.py --portfolio --save --notify
python main.py --scout --gems --save --notify
python -m pytest tests/ -q                # tests
```

## CI

`.github/workflows/analyze.yml` (weekday pre-market) and `scout.yml` (every 4h) run the
engine and upload to sovereign-eye. All keys are GitHub Actions secrets. Both workflows
Telegram-alert on failure. See `SECRETS.md` for secret handling + rotation.
