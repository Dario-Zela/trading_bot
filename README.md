# trading_bot

Self-improving LLM-driven stock trading bot. Designed to run unattended on GitHub Actions. Paper-only until a strategy is explicitly graduated to live.

See [`sparknotes.md`](./sparknotes.md) for the full design discussion and rationale.

## Status

**Wave 1 — skeleton.** A rule-based control strategy runs end-to-end in Shadow tier (no real or paper orders placed anywhere; entry/exit prices recorded from market data, P&L computed on paper). One US-region cron. Email summary at end of day. No LLM involvement yet.

See the build plan in `sparknotes.md` for what arrives in subsequent waves.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Fill in RESEND_API_KEY if you want emails locally
```

## Running manually

```bash
# Morning entry (decide picks, record entry prices)
python -m trading_bot.pipeline entry --region us

# Evening exit (record exit prices, compute P&L, email summary)
python -m trading_bot.pipeline exit --region us
```

## Project layout

```
.
├── sparknotes.md             # design discussion
├── pyproject.toml
├── .github/workflows/        # cron-scheduled CI workflows
├── src/trading_bot/          # Python source
│   ├── pipeline.py           # entry point
│   ├── tools/                # reusable analysis primitives
│   ├── strategy/             # strategy runtime code
│   ├── executor/             # broker-tier adapters
│   ├── state/                # ledger / predictions writers
│   └── notify/               # email
├── strategies/               # strategy configs + prompts (data, LLM-evolvable)
└── state/                    # runtime ledger, predictions, decisions (committed by CI)
```
