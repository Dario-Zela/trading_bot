"""Per-slot Alpaca credential loader.

Each Tier-1 strategy is bound to a numbered slot (1, 2, 3, ...) via its config.
Credentials for slot N live in env vars `ALPACA_API_KEY__N` and `ALPACA_API_SECRET__N`.
The market-data API doesn't care which slot is used; we default to slot 1 there.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AlpacaCreds:
    slot: int
    api_key: str
    api_secret: str

    @property
    def trading_base_url(self) -> str:
        # Paper trading endpoint; live would be https://api.alpaca.markets
        return "https://paper-api.alpaca.markets"


def load_slot_creds(slot: int) -> AlpacaCreds:
    """Read credentials for the given slot from env vars."""
    key_var = f"ALPACA_API_KEY__{slot}"
    secret_var = f"ALPACA_API_SECRET__{slot}"
    api_key = os.environ.get(key_var)
    api_secret = os.environ.get(secret_var)
    if not api_key or not api_secret:
        raise RuntimeError(
            f"Alpaca slot {slot} credentials not set. "
            f"Expected env vars {key_var} and {secret_var}."
        )
    return AlpacaCreds(slot=slot, api_key=api_key, api_secret=api_secret)


def load_data_creds() -> AlpacaCreds:
    """Credentials used for read-only market data calls (news, bars).
    Defaults to slot 1; any valid keypair would work."""
    for slot in (1, 2, 3, 4, 5):
        key_var = f"ALPACA_API_KEY__{slot}"
        secret_var = f"ALPACA_API_SECRET__{slot}"
        if os.environ.get(key_var) and os.environ.get(secret_var):
            return AlpacaCreds(
                slot=slot,
                api_key=os.environ[key_var],
                api_secret=os.environ[secret_var],
            )
    raise RuntimeError(
        "No Alpaca slot credentials found. Set at least ALPACA_API_KEY__1 / "
        "ALPACA_API_SECRET__1."
    )
