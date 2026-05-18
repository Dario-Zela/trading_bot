#!/usr/bin/env python3
"""Provision cron-job.org schedules that trigger GitHub Actions workflows.

GitHub Actions' built-in cron is unreliable — schedules silently drop
during high-load windows (top-of-hour especially). We observed two
missed firings in a single day on 2026-05-18. This script sets up
cron-job.org as an independent trigger that POSTs to GitHub's REST API
at the right times. GH Actions schedules stay as a redundant fallback
but the cron-job.org triggers are now the primary path.

Why cron-job.org: free, independent infrastructure, per-job timezones
(handles DST automatically — no need for our own dst-sync to update
cron strings), reliable SLA.

Usage:
    export GH_PAT=github_pat_...           # see https://github.com/settings/personal-access-tokens/new
    export CRON_API_KEY=<from console.cron-job.org → API>
    python scripts/setup_cron_jobs.py

Idempotent: lists existing jobs, deletes any titled 'trading_bot — ...',
re-creates from the SCHEDULES list. Re-run safely after editing this file.
"""
from __future__ import annotations

import json
import os
import sys
import time

import requests


REPO = "Dario-Zela/trading_bot"
CRON_BASE = "https://api.cron-job.org"

# All times are in the market's local timezone — cron-job.org handles DST
# per timezone, so we don't have to maintain UTC offsets ourselves.
SCHEDULES: list[dict] = [
    # US — NYSE 09:30–16:00 ET. Entry 5 min after open, exit 30 min before close.
    {
        "name": "pipeline-us entry",
        "workflow": "pipeline-us.yml",
        "inputs": {"mode": "entry"},
        "tz": "America/New_York",
        "hour": 9, "minute": 35,
        "wdays": [1, 2, 3, 4, 5],
    },
    {
        "name": "pipeline-us exit",
        "workflow": "pipeline-us.yml",
        "inputs": {"mode": "exit"},
        "tz": "America/New_York",
        "hour": 15, "minute": 30,
        "wdays": [1, 2, 3, 4, 5],
    },
    # UK-EU — LSE 08:00–16:30 BST/GMT. Entry 5 min after open, exit 30 min before close.
    {
        "name": "pipeline-uk-eu entry",
        "workflow": "pipeline-uk-eu.yml",
        "inputs": {"mode": "entry"},
        "tz": "Europe/London",
        "hour": 8, "minute": 35,
        "wdays": [1, 2, 3, 4, 5],
    },
    {
        "name": "pipeline-uk-eu exit",
        "workflow": "pipeline-uk-eu.yml",
        "inputs": {"mode": "exit"},
        "tz": "Europe/London",
        "hour": 16, "minute": 0,
        "wdays": [1, 2, 3, 4, 5],
    },
    # Weekly meta-jobs — pure UTC schedules, no inputs.
    {
        "name": "weekly-evolution",
        "workflow": "weekly-evolution.yml",
        "inputs": {},
        "tz": "UTC",
        "hour": 9, "minute": 0,
        "wdays": [6],  # Saturday
    },
    {
        "name": "weekly-macro",
        "workflow": "weekly-macro.yml",
        "inputs": {},
        "tz": "UTC",
        "hour": 17, "minute": 0,
        "wdays": [0],  # Sunday
    },
]

JOB_TITLE_PREFIX = "trading_bot — "


def _env_or_die(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"Set the {name} env var before running this script.")
    return v


def _cron_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


# cron-job.org's free-tier API rate-limits at ~1 req/sec. We add a small
# pause between requests and retry on HTTP 429 with linear backoff.
_REQ_PAUSE_S = 2.0
_MAX_RETRIES = 5


def _request_with_retry(method: str, url: str, **kw) -> requests.Response:
    """Wrap requests with a 429-aware retry loop. The cron-job.org API
    returns 429 with no body; we just back off and try again."""
    last: requests.Response | None = None
    for attempt in range(_MAX_RETRIES):
        r = requests.request(method, url, timeout=15, **kw)
        last = r
        if r.status_code != 429:
            return r
        wait = _REQ_PAUSE_S * (attempt + 2)
        print(f"  ... rate-limited (attempt {attempt + 1}/{_MAX_RETRIES}); sleeping {wait:.0f}s")
        time.sleep(wait)
    return last  # type: ignore[return-value]


def list_jobs(api_key: str) -> list[dict]:
    r = _request_with_retry("GET", f"{CRON_BASE}/jobs", headers=_cron_headers(api_key))
    r.raise_for_status()
    return r.json().get("jobs", [])


def delete_job(api_key: str, job_id: int) -> None:
    r = _request_with_retry(
        "DELETE", f"{CRON_BASE}/jobs/{job_id}", headers=_cron_headers(api_key)
    )
    r.raise_for_status()


def create_job(api_key: str, gh_pat: str, spec: dict) -> int:
    body_obj: dict = {"ref": "main"}
    if spec.get("inputs"):
        body_obj["inputs"] = spec["inputs"]

    payload = {
        "job": {
            "title": JOB_TITLE_PREFIX + spec["name"],
            "url": (
                f"https://api.github.com/repos/{REPO}/actions/workflows/"
                f"{spec['workflow']}/dispatches"
            ),
            "enabled": True,
            "saveResponses": True,
            "requestMethod": 1,  # POST
            "schedule": {
                "timezone": spec["tz"],
                "expiresAt": 0,
                "hours": [spec["hour"]],
                "minutes": [spec["minute"]],
                "mdays": [-1],
                "months": [-1],
                "wdays": spec["wdays"],
            },
            "extendedData": {
                "headers": {
                    "Authorization": f"Bearer {gh_pat}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "Content-Type": "application/json",
                },
                "body": json.dumps(body_obj),
            },
        }
    }

    r = _request_with_retry(
        "PUT",
        f"{CRON_BASE}/jobs",
        headers=_cron_headers(api_key),
        json=payload,
    )
    if not r.ok:
        print(f"  FAILED to create {spec['name']}: {r.status_code} {r.text[:300]}")
        r.raise_for_status()
    return r.json().get("jobId")


def main() -> int:
    gh_pat = _env_or_die("GH_PAT")
    cron_key = _env_or_die("CRON_API_KEY")

    print("Listing existing cron-job.org jobs...")
    existing = list_jobs(cron_key)
    to_delete = [j for j in existing if (j.get("title") or "").startswith(JOB_TITLE_PREFIX)]
    if to_delete:
        print(f"Deleting {len(to_delete)} existing trading_bot job(s):")
        for j in to_delete:
            delete_job(cron_key, j["jobId"])
            print(f"  - deleted '{j['title']}' (jobId={j['jobId']})")
    else:
        print("No existing trading_bot jobs to delete.")

    print(f"\nCreating {len(SCHEDULES)} fresh schedule(s):")
    for i, spec in enumerate(SCHEDULES):
        if i > 0:
            time.sleep(_REQ_PAUSE_S)  # cron-job.org free-tier pace
        wdays_human = ",".join(("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")[d] for d in spec["wdays"])
        jid = create_job(cron_key, gh_pat, spec)
        print(
            f"  + {spec['name']:<25}  {wdays_human:<19} {spec['hour']:02d}:{spec['minute']:02d} "
            f"{spec['tz']:<20}  jobId={jid}"
        )

    print(
        "\nDone. Verify at https://console.cron-job.org/jobs — each row shows its "
        "'Next execution' time. The first firings will trigger via GitHub's REST API; "
        "if anything 401s, the PAT lacks `Actions: Read and write` on this repo."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
