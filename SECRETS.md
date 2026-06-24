# Secrets

## Where they live
- **Local:** `sovereign-dd/.env` and `sovereign-eye/.dev.vars` — both gitignored, never commit.
- **CI:** GitHub Actions secrets (see `.github/workflows/*.yml`).
- **Production (eye):** Cloudflare Pages → Settings → Environment variables.

## Inventory
Gemini (×N), Finnhub, FMP, FRED, Alpha Vantage, Tavily (eye), **Supabase service-role key**,
`DD_UPLOAD_SECRET` (shared dd⇄eye), `DASHBOARD_PASSWORD` (eye), `GH_TOKEN` (eye → dispatch).

## Action items
- [ ] **Rotate the Supabase service-role key** (flagged in review; it sat in `.env`). After rotating, update GitHub Actions secrets + Cloudflare Pages env, and verify it was never committed: `git log -S 'sb_secret' --all`.
- [ ] `GH_TOKEN` should be a fine-grained PAT scoped to **Actions: Read & Write** on the sovereign-dd repo only.
- [ ] Rotate API keys roughly quarterly.

## Hygiene
Consider a pre-commit hook that blocks staged content matching `sb_secret`, `AIza`,
`github_pat_`, or `(key|secret|token)\s*=\s*[A-Za-z0-9_-]{16,}`.
