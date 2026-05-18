"""One-shot Trading212 demo API smoke test.

Hits read-only endpoints with whatever credentials are configured for a
T212 slot and prints the outcome. Used to validate that T212_API_KEY__N
(and optionally T212_API_SECRET__N) are wired up correctly *before* any
strategy starts placing orders.

Triggered via the `t212-smoke-test.yml` workflow_dispatch action. No
state is written; pure read-only probe.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import requests

from trading_bot.t212_slot import load_slot_creds


log = logging.getLogger(__name__)


_PROBES = [
    ("GET", "/equity/account/cash",  "account cash balance"),
    ("GET", "/equity/account/info",  "account info"),
    ("GET", "/equity/portfolio",     "open positions"),
]


def run(slot: int, *, demo: bool = True) -> int:
    creds = load_slot_creds(slot, demo=demo)
    auth = creds.auth_header()
    scheme = "Basic (key+secret)" if auth.startswith("Basic ") else "single-header (key only)"

    print(f"T212 smoke test")
    print(f"  slot:    {slot}")
    print(f"  base:    {creds.base_url}")
    print(f"  scheme:  {scheme}")
    print(f"  secret:  {'present' if creds.api_secret else 'absent'}")
    print()

    headers = {"Authorization": auth, "Content-Type": "application/json"}
    failures = 0

    for method, path, label in _PROBES:
        url = f"{creds.base_url}{path}"
        try:
            response = requests.request(method, url, headers=headers, timeout=15)
        except requests.RequestException as e:
            print(f"  {method} {path:30s}  ERROR: {e}")
            failures += 1
            continue

        status = response.status_code
        body_preview = response.text[:300].replace("\n", " ")
        ok_marker = "OK " if response.ok else "FAIL"
        print(f"  {ok_marker} {method} {path:30s}  → {status} ({label})")
        if not response.ok:
            print(f"        body: {body_preview}")
            failures += 1
        else:
            # Pretty-print the OK response so we can see what came back
            try:
                parsed = response.json()
                preview = json.dumps(parsed, indent=2)[:500]
                print(f"        body: {preview}")
            except json.JSONDecodeError:
                print(f"        body: {body_preview}")
        print()

    print(f"Result: {len(_PROBES) - failures}/{len(_PROBES)} probes succeeded")
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trading_bot.t212_smoke")
    parser.add_argument("--slot", type=int, default=1)
    parser.add_argument("--live", action="store_true",
                        help="Probe the live API instead of demo (use with caution)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return run(args.slot, demo=not args.live)


if __name__ == "__main__":
    sys.exit(main())
