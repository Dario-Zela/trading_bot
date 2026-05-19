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
    # Daily — runs ~1h before UK-EU entry to give the brief time to land.
    {
        "name": "daily-news-brief",
        "workflow": "daily-news-brief.yml",
        "inputs": {},
        "tz": "Europe/London",
        "hour": 7, "minute": 30,
        "wdays": [1, 2, 3, 4, 5],  # Mon-Fri
    },
    # Early-morning grader — scores any open prediction whose target_date
    # has passed. Runs at 05:00 UTC: after US close (~21:00 UTC prior day),
    # after Asia overnight, and ~1.5h before the morning UK-EU news brief
    # so the "marking the homework" section reads the freshest verdicts.
    {
        "name": "grade-predictions",
        "workflow": "grade-predictions.yml",
        "inputs": {},
        "tz": "UTC",
        "hour": 5, "minute": 0,
        "wdays": [1, 2, 3, 4, 5, 6, 0],  # every day
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
    # Phase 7 — Archive trim. Runs Sunday 04:00 UTC, before grade-predictions
    # (05:00) and the daily-news-brief (06:30 UK = 06:30 UTC in winter).
    # Compresses news/macro editions older than 90 days into state/archive/*.tar.gz.
    {
        "name": "archive-trim",
        "workflow": "archive-trim.yml",
        "inputs": {},
        "tz": "UTC",
        "hour": 4, "minute": 0,
        "wdays": [0],  # Sunday
    },
    # Phase 8D — midday trailing-stop pass. Split by region because the
    # mid-sessions are hours apart. UK-EU mid-LSE-session is around
    # 12:00 UK; US mid-NYSE-session is around 12:30 ET. Each workflow
    # talks to only its own broker.
    {
        "name": "midday-trail-uk-eu",
        "workflow": "midday-trail-uk-eu.yml",
        "inputs": {},
        "tz": "Europe/London",
        "hour": 12, "minute": 0,
        "wdays": [1, 2, 3, 4, 5],
    },
    {
        "name": "midday-trail-us",
        "workflow": "midday-trail-us.yml",
        "inputs": {},
        "tz": "America/New_York",
        "hour": 12, "minute": 30,
        "wdays": [1, 2, 3, 4, 5],
    },
    # Phase 9F — nightly bot-health check. 22:00 UTC sits after the day's
    # last live workflow (US exit at 15:30 ET = 19:30/20:30 UTC) but
    # before tomorrow's grade-predictions at 05:00, so the email lands
    # while we're awake.
    {
        "name": "health-check",
        "workflow": "health-check.yml",
        "inputs": {},
        "tz": "UTC",
        "hour": 22, "minute": 0,
        "wdays": [1, 2, 3, 4, 5, 6, 0],   # every day
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


def _spec_fingerprint(spec: dict) -> tuple:
    """A normalized fingerprint we can compare against existing-job fields
    from cron-job.org's list response. Captures everything the user might
    plausibly change: workflow file, schedule time, timezone, wdays."""
    return (
        spec["workflow"],
        spec.get("tz", ""),
        int(spec.get("hour", 0)),
        int(spec.get("minute", 0)),
        tuple(sorted(spec.get("wdays", []))),
    )


def _existing_fingerprint(job: dict) -> tuple | None:
    """Extract the same fingerprint from an existing cron-job.org job
    record. Returns None if the job isn't shaped like one we'd manage
    (e.g., missing fields after a manual edit)."""
    url = job.get("url") or ""
    # url is like https://api.github.com/repos/X/Y/actions/workflows/{wf}/dispatches
    workflow = ""
    if "/actions/workflows/" in url:
        try:
            workflow = url.split("/actions/workflows/")[1].split("/")[0]
        except IndexError:
            pass
    sch = job.get("schedule") or {}
    hours = sch.get("hours") or []
    minutes = sch.get("minutes") or []
    wdays = sch.get("wdays") or []
    if not workflow or not hours or not minutes:
        return None
    return (
        workflow,
        sch.get("timezone") or "",
        int(hours[0]),
        int(minutes[0]),
        tuple(sorted(wdays)),
    )


def main() -> int:
    gh_pat = _env_or_die("GH_PAT")
    cron_key = _env_or_die("CRON_API_KEY")

    print("Listing existing cron-job.org jobs...")
    existing = list_jobs(cron_key)
    existing_by_title = {
        (j.get("title") or ""): j
        for j in existing
        if (j.get("title") or "").startswith(JOB_TITLE_PREFIX)
    }
    desired_titles = {JOB_TITLE_PREFIX + s["name"] for s in SCHEDULES}

    # Three-way diff: keep / update / create / delete.
    to_skip: list[tuple[str, int]] = []     # (name, existing_jobId)
    to_update: list[tuple[dict, int]] = []  # (spec, existing_jobId_to_delete_first)
    to_create: list[dict] = []
    to_delete_orphan: list[dict] = [        # ours-but-no-longer-in-SCHEDULES
        j for title, j in existing_by_title.items() if title not in desired_titles
    ]

    for spec in SCHEDULES:
        title = JOB_TITLE_PREFIX + spec["name"]
        existing_job = existing_by_title.get(title)
        if existing_job is None:
            to_create.append(spec)
            continue
        ex_fp = _existing_fingerprint(existing_job)
        if ex_fp is not None and ex_fp == _spec_fingerprint(spec):
            to_skip.append((spec["name"], existing_job["jobId"]))
        else:
            to_update.append((spec, existing_job["jobId"]))

    n_actions = len(to_update) + len(to_create) + len(to_delete_orphan)

    print(
        f"\nDiff: {len(to_skip)} unchanged · {len(to_create)} to create · "
        f"{len(to_update)} to update · {len(to_delete_orphan)} orphan(s) to remove"
    )
    if to_skip:
        for name, jid in to_skip:
            print(f"  = {name:<25} jobId={jid} (unchanged)")

    if n_actions == 0:
        print("\nNothing to do. Re-run is a no-op.")
        return 0

    def _paced(idx: int) -> None:
        if idx > 0:
            time.sleep(_REQ_PAUSE_S)

    # Deletes first: orphans, then the to-update old versions.
    delete_ops = [(j["title"], j["jobId"]) for j in to_delete_orphan] + [
        (JOB_TITLE_PREFIX + spec["name"], jid) for spec, jid in to_update
    ]
    if delete_ops:
        print(f"\nDeleting {len(delete_ops)} job(s):")
        for i, (title, jid) in enumerate(delete_ops):
            _paced(i)
            delete_job(cron_key, jid)
            print(f"  - {title} (jobId={jid})")
        time.sleep(_REQ_PAUSE_S * 2)  # grace period before create burst

    create_ops = [spec for spec, _jid in to_update] + to_create
    if create_ops:
        print(f"\nCreating {len(create_ops)} job(s):")
        for i, spec in enumerate(create_ops):
            _paced(i)
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
