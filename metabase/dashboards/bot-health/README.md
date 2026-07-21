# Bot Health dashboard

Six saved questions that together give you a one-glance view of
whether the trading bot is earning, where the losses are coming from,
and which strategies are being eaten by fees.

## Import (manual — takes ~5 minutes)

In Metabase, for each `.sql` file in this directory:

1. Top-right **+ New → SQL query**
2. Paste the SQL body (skip the header comments)
3. Click **Run** to preview
4. Click the **Visualization** icon (bottom-left of results) and pick the
   type noted in the file header (Line / Bar / Scatter)
5. Save with a name matching the filename's title (e.g. `bot: weekly net pnl`)
6. When Metabase asks which collection, pick **Our analytics** or make
   a new **Bot Health** collection.

Then build the dashboard:

1. Top-right **+ New → Dashboard** → name it "Bot Health"
2. Click **+** in an empty area → add each saved question as a tile
3. Suggested layout:
   - Row 1: `weekly-net-pnl` full-width
   - Row 2: `cumulative-pnl-by-strategy` (half) + `top-strategies-30d` (half)
   - Row 3: `exit-reason-attribution` (half) + `fee-drag-by-strategy` (half)
   - Row 4: `recent-trades-scatter` full-width

## Files

| # | File | Question | Chart |
|---|---|---|---|
| 01 | `01-weekly-net-pnl.sql` | Weekly PnL trajectory (net / gross / fees) | Line |
| 02 | `02-cumulative-pnl-by-strategy.sql` | Cumulative PnL per strategy | Line (multi-series) |
| 03 | `03-top-strategies-30d.sql` | Top strategies over last 30 days | Bar (horizontal) |
| 04 | `04-exit-reason-attribution.sql` | PnL by exit reason (system-wide) | Bar |
| 05 | `05-fee-drag-by-strategy.sql` | Fee-share-of-gross per strategy | Bar (horizontal, sorted) |
| 06 | `06-recent-trades-scatter.sql` | Individual trades in last 30 days | Scatter |

## Alerts worth setting up

Once questions are saved, on each one's page: **three-dot menu → Get
alerts**.

- **Fee drag** — alert when any row's `fee_share_pct > 80`. Signals
  a strategy the evolution loop should tune (raise `cost_gate_multiplier`).
- **Top strategies 30d** — alert when the top row's `net_pnl_gbp < 0`.
  Signals the best strategy is losing money — evolution loop is
  probably about to demote/reshuffle.
- **Weekly net PnL** — alert on 2 consecutive weeks negative. Signals
  a real drawdown, not a bad day.

## Extending

Model for adding more:

1. Write the SQL locally, test in Metabase's SQL editor
2. Once it works, save the file here with a header block matching the
   existing pattern (question type, viz, axes, dashboard tile intent)
3. Commit — this directory is version control for what you'd otherwise
   only have in Metabase's H2 database (which is ephemeral in Codespaces).
