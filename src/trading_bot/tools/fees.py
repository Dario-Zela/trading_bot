"""Broker fee model — single source of truth.

Models the costs that move money out of the account independent of the
price move itself. The price-side P&L is `(exit - entry) * qty` in the
account currency; the *net* P&L the user actually sees is that minus
the fees this module computes.

Schedule (Trading 212 Invest account, May 2026):
- **FX fee: 0.15% per leg** — buy + sell, on any trade in a non-account
  currency (we run a GBP account, so this hits USD/EUR/CHF/etc).
- **UK Stamp Duty Reserve Tax: 0.5%** — on **purchases only** of LSE
  shares. Exempt: gilts, bonds, ETFs, AIM-listed stocks.
- **PTM Levy: £1.50** — flat, both buy + sell, but only for orders
  >£10,000. (Threshold = T212_PTM_THRESHOLD_GBP)
- **French FTT: 0.4%** — on purchases of French shares with market cap
  > €1bn. (We don't track market cap so this defaults to applying on
  any .PA listing; flag conservatively.)
- **Italian FTT: 0.1%** — on purchases of Italian shares.
- **US SEC fee: 0.00278%** — on sells.
- **US FINRA fee: $0.000195/share** — on sells.

Alpaca paper is commission-free with no FX (USD account). For
shadow-cost parity with where the bot will eventually run live, we
ALSO apply the T212 FX-fee model to Alpaca trades — so Alpaca's
ledger P&L reflects what the same trade would have cost on T212
live. Stamp duty + FTTs don't apply (US-only universe).

Sources:
- https://helpcentre.trading212.com/hc/en-us/articles/360018909758-What-is-the-FX-fee-Invest-Stocks-ISA
- https://helpcentre.trading212.com/hc/en-us/articles/360007081637-What-are-the-applicable-stock-exchange-fees
- https://helpcentre.trading212.com/hc/en-us/articles/11471996799517-What-are-the-fees-in-the-Invest-and-ISAs
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from trading_bot.tools.fx import to_gbp_multiplier


# ---------------------------------------------------------------------------
# Fee schedule constants
# ---------------------------------------------------------------------------

# T212 FX fee — applied per leg (buy + sell each), as a fraction of the
# notional converted into GBP.
T212_FX_FEE_PER_LEG = 0.0015                    # 0.15%

# UK stamp duty — purchases only, LSE non-AIM non-ETF.
UK_STAMP_DUTY_RATE = 0.005                      # 0.5%

# PTM levy — flat per leg, but only when notional exceeds threshold.
T212_PTM_LEVY_GBP = 1.50
T212_PTM_THRESHOLD_GBP = 10_000.0

# French FTT — purchases only, large-cap French shares.
FRENCH_FTT_RATE = 0.004                         # 0.4%

# Italian FTT — purchases only.
ITALIAN_FTT_RATE = 0.001                        # 0.1%

# US regulatory fees — sells only, tiny.
US_SEC_FEE_RATE = 0.0000278                     # 0.00278% of sell notional
US_FINRA_PER_SHARE_USD = 0.000195               # per share, sell side


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradeContext:
    """Everything the fee model needs to know about a trade.

    `tier` selects between live-broker and shadow models. `currency`,
    `exchange`, `instrument_type` come from the broker's instrument
    record at order time (T212) or are inferred for Alpaca/shadow.

    **Notional fields are GBP.** Callers convert from native to GBP
    BEFORE building this context — it's clearer that way, and lets
    compute_fees() stay a pure rate-application step independent of
    the broker's internal price representation. For T212 (prices
    already GBP-converted at fill time) the values are passed through;
    for Alpaca (prices in USD) the caller multiplies by the USD→GBP
    spot before constructing the context.
    """
    tier: str                       # 'alpaca-paper' | 'trading212-paper' | 'shadow'
    currency: str                   # 'GBP' / 'USD' / 'EUR' / ... — instrument's native currency (drives FX fee)
    exchange: str                   # 'LSE' / 'NYSE' / 'NASDAQ' / 'XPAR' / 'XAMS' / ...
    instrument_type: str            # 'share' / 'etf' / 'aim' / 'bond' / 'gilt' / 'crypto'
    entry_notional_gbp: float       # qty * entry_price, in GBP
    exit_notional_gbp: float        # qty * exit_price, in GBP
    quantity: float                 # shares filled


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class FeeBreakdown:
    """Per-trade fee breakdown. All amounts in GBP."""
    fx_fee_entry_gbp: float = 0.0
    fx_fee_exit_gbp: float = 0.0
    stamp_duty_gbp: float = 0.0
    ptm_levy_gbp: float = 0.0
    french_ftt_gbp: float = 0.0
    italian_ftt_gbp: float = 0.0
    us_sec_fee_gbp: float = 0.0
    us_finra_fee_gbp: float = 0.0

    @property
    def total_gbp(self) -> float:
        return (
            self.fx_fee_entry_gbp + self.fx_fee_exit_gbp + self.stamp_duty_gbp
            + self.ptm_levy_gbp + self.french_ftt_gbp + self.italian_ftt_gbp
            + self.us_sec_fee_gbp + self.us_finra_fee_gbp
        )

    def as_dict(self) -> dict[str, float]:
        """Compact dict for storage on TradeRecord.fees_breakdown.
        Drops zero entries so the row stays readable on the dashboard."""
        d = asdict(self)
        return {k: round(v, 4) for k, v in d.items() if v != 0.0}


# ---------------------------------------------------------------------------
# T212 ticker → exchange / instrument-type inference
# ---------------------------------------------------------------------------

# T212 ticker suffix convention: SYMBOL[<venue letter>]_EQ
# (e.g. AAPL_US_EQ, BARCl_EQ, ABNa_EQ, SAPd_EQ, ASMLn_EQ).
_T212_VENUE_SUFFIX_TO_EXCHANGE = {
    "l": "LSE",      # London Stock Exchange
    "a": "XAMS",     # Amsterdam (Euronext)
    "p": "XPAR",     # Paris (Euronext)
    "d": "XETR",     # Deutsche Börse / Xetra
    "m": "XMIL",     # Milan
    "e": "XMAD",     # Madrid
    "n": "XAMS",     # Some T212 listings use 'n' for Euronext (e.g. ASML)
    "i": "XLON",     # Sometimes used for international LSE
    "h": "XHEL",     # Helsinki
    "s": "XSWX",     # Swiss
}


def t212_exchange_from_ticker(t212_ticker: str) -> str:
    """Infer the exchange code from a T212 internal ticker. Returns
    '' if it can't classify. Used to assign stamp-duty / FTT rules."""
    if not t212_ticker:
        return ""
    s = t212_ticker.strip()
    if "_US_EQ" in s:
        return "NYSE"        # NYSE / NASDAQ collapsed — both share the same US fee model
    if s.endswith("_EQ") and len(s) >= 5:
        # Suffix letter is the char right before "_EQ"
        suffix = s[-4]
        return _T212_VENUE_SUFFIX_TO_EXCHANGE.get(suffix, "")
    return ""


# yfinance ticker suffixes → (exchange, currency). Bare tickers (no dot)
# default to US.
_YF_SUFFIX_TO_VENUE = {
    ".L":  ("LSE",  "GBP"),
    ".IL": ("LSE",  "USD"),     # GDR / dollar-denominated LSE line
    ".DE": ("XETR", "EUR"),
    ".F":  ("XETR", "EUR"),
    ".PA": ("XPAR", "EUR"),
    ".AS": ("XAMS", "EUR"),
    ".MI": ("XMIL", "EUR"),
    ".MC": ("XMAD", "EUR"),
    ".SW": ("XSWX", "CHF"),
    ".HE": ("XHEL", "EUR"),
    ".OL": ("XOSL", "NOK"),
    ".ST": ("XSTO", "SEK"),
    ".T":  ("XTKS", "JPY"),
    ".HK": ("XHKG", "HKD"),
    ".TO": ("XTSE", "CAD"),
    ".AX": ("XASX", "AUD"),
}


def yf_ticker_classify(yf_ticker: str) -> tuple[str, str]:
    """Map a yfinance ticker to (exchange_code, currency_code). Bare
    tickers (no dot) default to NYSE/USD."""
    if not yf_ticker or "." not in yf_ticker:
        return ("NYSE", "USD")
    for suffix, venue in _YF_SUFFIX_TO_VENUE.items():
        if yf_ticker.endswith(suffix):
            return venue
    return ("", "USD")


def t212_instrument_type(inst_type: str) -> str:
    """Map T212's `type` field on the instrument record to our internal
    instrument-type categories (used to decide stamp-duty exemptions).

    T212 values seen in the wild: STOCK, ETF, ETC (commodity), CRYPTO, BOND.
    """
    t = (inst_type or "").upper().strip()
    if t == "STOCK":   return "share"
    if t == "ETF":     return "etf"
    if t == "ETC":     return "etf"      # commodity ETFs — same exemption
    if t == "BOND":    return "bond"
    if t == "GILT":    return "gilt"
    if t == "CRYPTO":  return "crypto"
    return "share"     # safest default — overcharges by 0.5% on ETPs we miss-classify


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

# Exchanges with UK stamp duty on share purchases. LSE main market only.
_UK_LSE_EXCHANGES = {"LSE", "LSE_INTL", "LSEAIM"}     # LSEAIM kept as code, see exemption below
_UK_AIM_EXCHANGES = {"LSEAIM", "AIM"}

# Instruments exempt from UK stamp duty even on the LSE main market.
_STAMP_DUTY_EXEMPT_TYPES = {"etf", "etn", "bond", "gilt", "trust"}

# French / Italian exchange codes (T212 uses ISO MIC codes for some)
_FRENCH_EXCHANGES = {"XPAR", "EPA", "PA"}
_ITALIAN_EXCHANGES = {"XMIL", "BIT", "MI"}

# US exchanges where SEC/FINRA fees apply
_US_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "XNYS", "XNAS", "XASE"}


def compute_fees(ctx: TradeContext) -> FeeBreakdown:
    """Return the FeeBreakdown a real broker would charge on this
    round-trip trade.

    For `tier='alpaca-paper'` we apply the T212 shadow-fee model so the
    bot's tracked P&L matches what the same trade would cost on T212
    live. Alpaca paper itself charges nothing; the divergence is
    intentional and explains why bot P&L < Alpaca broker view by
    design.
    """
    b = FeeBreakdown()
    ccy = (ctx.currency or "GBP").upper()
    exch = (ctx.exchange or "").upper()
    typ = (ctx.instrument_type or "share").lower()

    # Notionals are already in GBP per the TradeContext contract.
    entry_gbp = ctx.entry_notional_gbp
    exit_gbp = ctx.exit_notional_gbp

    # --- FX fees (T212 + Alpaca-shadow) ---
    if ccy != "GBP":
        b.fx_fee_entry_gbp = entry_gbp * T212_FX_FEE_PER_LEG
        b.fx_fee_exit_gbp = exit_gbp * T212_FX_FEE_PER_LEG

    # --- UK stamp duty (purchases only, LSE non-ETF non-AIM) ---
    if exch in _UK_LSE_EXCHANGES and exch not in _UK_AIM_EXCHANGES and typ not in _STAMP_DUTY_EXEMPT_TYPES:
        b.stamp_duty_gbp = entry_gbp * UK_STAMP_DUTY_RATE

    # --- PTM levy (flat fee, both legs, only above threshold) ---
    if entry_gbp > T212_PTM_THRESHOLD_GBP:
        b.ptm_levy_gbp += T212_PTM_LEVY_GBP
    if exit_gbp > T212_PTM_THRESHOLD_GBP:
        b.ptm_levy_gbp += T212_PTM_LEVY_GBP

    # --- French FTT (purchases only) ---
    if exch in _FRENCH_EXCHANGES and typ not in _STAMP_DUTY_EXEMPT_TYPES:
        b.french_ftt_gbp = entry_gbp * FRENCH_FTT_RATE

    # --- Italian FTT (purchases only) ---
    if exch in _ITALIAN_EXCHANGES and typ not in _STAMP_DUTY_EXEMPT_TYPES:
        b.italian_ftt_gbp = entry_gbp * ITALIAN_FTT_RATE

    # --- US SEC + FINRA (sells only) ---
    if exch in _US_EXCHANGES:
        b.us_sec_fee_gbp = exit_gbp * US_SEC_FEE_RATE
        # FINRA is per share, in USD. Convert to GBP.
        mult = to_gbp_multiplier("USD") or 0.79
        b.us_finra_fee_gbp = ctx.quantity * US_FINRA_PER_SHARE_USD * mult

    return b


# ---------------------------------------------------------------------------
# LLM-facing helpers (used by strategies to score candidate trades)
# ---------------------------------------------------------------------------

def estimate_round_trip_cost_pct(
    *,
    tier: str,
    currency: str,
    exchange: str,
    instrument_type: str,
    notional_gbp: float,
    quantity: float | None = None,
) -> dict[str, Any]:
    """Estimate the round-trip cost of a trade BEFORE it's executed.
    Returns a dict with:
      - total_pct: cost as a fraction of notional (multiply by 100 for %)
      - total_gbp: cost in GBP
      - breakdown: dict of {fee_name: gbp_amount}
      - note: one-line summary the LLM can quote in its rationale

    Used by strategies to score candidates net of fees. Assumes
    entry_notional ≈ exit_notional (true for short-term trades where
    the price hasn't moved much yet).
    """
    if not quantity:
        # Rough fallback — we don't know fill price yet, so guess from
        # notional (works for the FINRA per-share fee, which is tiny anyway)
        quantity = notional_gbp / 50.0     # placeholder $50 avg

    # For estimation we treat entry == exit notional (flat round-trip).
    # Small over-estimate on losers, small under-estimate on winners —
    # close enough for trade scoring.
    b = FeeBreakdown()
    ccy = (currency or "GBP").upper()
    exch = (exchange or "").upper()
    typ = (instrument_type or "share").lower()
    if ccy != "GBP":
        b.fx_fee_entry_gbp = notional_gbp * T212_FX_FEE_PER_LEG
        b.fx_fee_exit_gbp = notional_gbp * T212_FX_FEE_PER_LEG
    if exch in _UK_LSE_EXCHANGES and exch not in _UK_AIM_EXCHANGES and typ not in _STAMP_DUTY_EXEMPT_TYPES:
        b.stamp_duty_gbp = notional_gbp * UK_STAMP_DUTY_RATE
    if notional_gbp > T212_PTM_THRESHOLD_GBP:
        b.ptm_levy_gbp = 2 * T212_PTM_LEVY_GBP    # both legs
    if exch in _FRENCH_EXCHANGES and typ not in _STAMP_DUTY_EXEMPT_TYPES:
        b.french_ftt_gbp = notional_gbp * FRENCH_FTT_RATE
    if exch in _ITALIAN_EXCHANGES and typ not in _STAMP_DUTY_EXEMPT_TYPES:
        b.italian_ftt_gbp = notional_gbp * ITALIAN_FTT_RATE
    if exch in _US_EXCHANGES:
        b.us_sec_fee_gbp = notional_gbp * US_SEC_FEE_RATE
        mult = to_gbp_multiplier("USD") or 0.79
        b.us_finra_fee_gbp = quantity * US_FINRA_PER_SHARE_USD * mult

    total_gbp = b.total_gbp
    total_pct = (total_gbp / notional_gbp) if notional_gbp > 0 else 0.0

    parts = []
    if b.stamp_duty_gbp:    parts.append(f"stamp 0.5%")
    if b.fx_fee_entry_gbp:  parts.append(f"FX 0.30% rt")
    if b.french_ftt_gbp:    parts.append(f"FR-FTT 0.4%")
    if b.italian_ftt_gbp:   parts.append(f"IT-FTT 0.1%")
    if b.ptm_levy_gbp:      parts.append(f"PTM £{b.ptm_levy_gbp:.0f}")
    note = (
        f"~{total_pct*100:.2f}% round-trip cost on £{notional_gbp:,.0f} "
        f"({', '.join(parts) or 'no fees'})"
    )
    return {
        "total_pct": round(total_pct, 6),
        "total_gbp": round(total_gbp, 4),
        "breakdown": b.as_dict(),
        "note": note,
    }


# ---------------------------------------------------------------------------
# Short summary the LLM can read once at session start
# ---------------------------------------------------------------------------

FEE_SCHEDULE_BRIEF = """
**T212 Invest fee schedule** (applied to live trades; Alpaca paper
trades carry the same schedule as a shadow cost so the ledger
reflects what live execution would have cost):

- **FX fee: 0.15% per leg** on non-GBP trades → 0.30% round-trip
- **UK stamp duty: 0.5%** on LSE share PURCHASES (not ETFs, not AIM)
- **PTM levy: £1.50** per leg on trades > £10,000 each
- **French FTT: 0.4%** on French large-cap purchases
- **Italian FTT: 0.1%** on Italian purchases
- US SEC/FINRA fees on sells are negligible (~0.003% combined)

Practical implication: a UK LSE share needs ≥ 0.6% expected move to
break even; a US trade on T212 needs ≥ 0.35%. ETFs and AIM stocks
on the LSE skip stamp duty and only carry small spread costs.
""".strip()
