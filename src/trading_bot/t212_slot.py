"""Per-slot Trading212 demo (paper) credential loader.

Mirrors the Alpaca slot pattern: each Tier-1.5 strategy bound to a numbered
slot via its config, credentials in env vars.

T212 generates **two** values per key: an API key and an API secret. The
T212 help centre is explicit about this, but the most popular Python
clients only send `Authorization: <api_key>`. The two sources disagree,
so we support both schemes:

- If both `T212_API_KEY__N` and `T212_API_SECRET__N` are set, we use
  HTTP Basic auth (`Authorization: Basic <base64(key:secret)>`).
- If only `T212_API_KEY__N` is set, we send `Authorization: <api_key>`
  (single-header) — works with the older T212 API client behaviour.

To get a key: switch the Trading212 Invest account to Practice mode
(Settings → API → "Demo" toggle), then generate a key — the app shows
the secret only once, so copy it immediately.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass


_DEMO_BASE_URL = "https://demo.trading212.com/api/v0"
_LIVE_BASE_URL = "https://live.trading212.com/api/v0"


@dataclass(frozen=True)
class T212Creds:
    slot: int
    api_key: str
    api_secret: str | None = None
    is_demo: bool = True

    @property
    def base_url(self) -> str:
        return _DEMO_BASE_URL if self.is_demo else _LIVE_BASE_URL

    def auth_header(self) -> str:
        """Build the Authorization header value. Prefers HTTP Basic when a
        secret is present; falls back to raw key (the older single-token
        scheme some T212 clients still use)."""
        if self.api_secret:
            token = base64.b64encode(
                f"{self.api_key}:{self.api_secret}".encode()
            ).decode()
            return f"Basic {token}"
        return self.api_key


def load_slot_creds(slot: int, *, demo: bool = True) -> T212Creds:
    """Read credentials for the given T212 slot from env vars."""
    key_var = f"T212_API_KEY__{slot}"
    secret_var = f"T212_API_SECRET__{slot}"
    api_key = os.environ.get(key_var)
    if not api_key:
        raise RuntimeError(
            f"Trading212 slot {slot} credentials not set. "
            f"Expected env var {key_var} (and optionally {secret_var})."
        )
    api_secret = os.environ.get(secret_var) or None
    return T212Creds(slot=slot, api_key=api_key, api_secret=api_secret, is_demo=demo)
