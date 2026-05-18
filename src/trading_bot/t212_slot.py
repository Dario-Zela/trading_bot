"""Per-slot Trading212 demo (paper) credential loader.

Mirrors the Alpaca slot pattern: each Tier-1.5 strategy bound to a numbered
slot via its config, credentials in env vars `T212_API_KEY__N`. T212 uses
a single bearer-style API key per account; no separate secret.

To get a key: switch the Trading212 Invest account to Practice mode
(Settings → API → "Demo" toggle), then generate a key. The key is
account-specific, so each slot needs its own Practice account if you want
isolated strategy P&L.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


_DEMO_BASE_URL = "https://demo.trading212.com/api/v0"
_LIVE_BASE_URL = "https://live.trading212.com/api/v0"


@dataclass(frozen=True)
class T212Creds:
    slot: int
    api_key: str
    is_demo: bool = True

    @property
    def base_url(self) -> str:
        return _DEMO_BASE_URL if self.is_demo else _LIVE_BASE_URL


def load_slot_creds(slot: int, *, demo: bool = True) -> T212Creds:
    """Read credentials for the given T212 slot from env vars."""
    key_var = f"T212_API_KEY__{slot}"
    api_key = os.environ.get(key_var)
    if not api_key:
        raise RuntimeError(
            f"Trading212 slot {slot} credentials not set. "
            f"Expected env var {key_var}."
        )
    return T212Creds(slot=slot, api_key=api_key, is_demo=demo)
