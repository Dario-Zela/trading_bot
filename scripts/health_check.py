"""Phase 9F — nightly bot-health check.

Flags anything that looks wrong without scrolling through GH Actions
manually. Sends a single summary email if there's anything to flag;
silent if all good.

Checks:
- Recent workflow runs that failed
- Ledger staleness (no new entries / exits in N days vs expected)
- Predictions log corruption (unparseable JSONL rows)
- Halt file state (Phase 8F)
- State directory growth (Phase 7 archive sanity)

Run by `.github/workflows/health-check.yml` daily.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from trading_bot.state.paths import STATE_ROOT, ledger_path, predictions_path


log = logging.getLogger("health_check")


@dataclass
class HealthFinding:
    severity: str          # "info" | "warning" | "error"
    category: str          # short label
    message: str           # one-line human-readable

    def __str__(self) -> str:
        icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(self.severity, "•")
        return f"{icon} [{self.category}] {self.message}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nightly bot-health check.")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "Dario-Zela/trading_bot"))
    parser.add_argument("--ledger-stale-days", type=int, default=3,
                        help="Warn if the ledger has no new entries in this many days "
                        "(default 3; Mon-Fri only).")
    parser.add_argument("--send-email", action="store_true",
                        help="Send a summary email if findings exist. Without this flag, "
                        "just print findings to stdout.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    findings: list[HealthFinding] = []
    findings.extend(_check_workflow_failures(args.repo))
    findings.extend(_check_ledger_staleness(args.ledger_stale_days))
    findings.extend(_check_predictions_corruption())
    findings.extend(_check_halt_state())
    findings.extend(_check_state_growth())

    if not findings:
        print("All clear — no health findings.")
        return 0

    print(f"\nHealth findings: {len(findings)} (severity counts: "
          f"{sum(1 for f in findings if f.severity == 'error')} error, "
          f"{sum(1 for f in findings if f.severity == 'warning')} warning, "
          f"{sum(1 for f in findings if f.severity == 'info')} info)\n")
    for f in findings:
        print(f)
    print()

    if args.send_email:
        _send_summary_email(findings)

    # Exit non-zero only on errors so warnings don't fail the CI job
    if any(f.severity == "error" for f in findings):
        return 1
    return 0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_workflow_failures(repo: str) -> list[HealthFinding]:
    """Query GitHub's API for workflow runs in the last 24h. Flag any
    failed / cancelled runs (excluding manual-cancel workflows like
    midday-trail when nothing was queued)."""
    out: list[HealthFinding] = []
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        out.append(HealthFinding("info", "workflows",
            "GITHUB_TOKEN not set — workflow-status check skipped"))
        return out
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    try:
        r = requests.get(
            f"https://api.github.com/repos/{repo}/actions/runs",
            params={"created": f">={cutoff[:10]}", "per_page": 50},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            timeout=15,
        )
    except requests.RequestException as e:
        out.append(HealthFinding("warning", "workflows", f"API fetch failed: {e}"))
        return out
    if not r.ok:
        out.append(HealthFinding("warning", "workflows", f"API returned {r.status_code}: {r.text[:120]}"))
        return out
    body = r.json() or {}
    runs = body.get("workflow_runs") or []
    bad = [run for run in runs if run.get("conclusion") in ("failure", "cancelled", "timed_out")]
    # Don't flag pages-build-deployment churn or our own midday-trail
    # cancellation patterns
    bad = [run for run in bad if "pages-build" not in (run.get("name") or "")]
    if bad:
        names = ", ".join(sorted({run.get("name", "?") for run in bad}))
        out.append(HealthFinding(
            "error", "workflows",
            f"{len(bad)} failed/cancelled runs in last 24h: {names}",
        ))
    return out


def _check_ledger_staleness(threshold_days: int) -> list[HealthFinding]:
    """Most recent entry_date in the ledger; warn if older than threshold."""
    p = ledger_path()
    if not p.exists():
        return [HealthFinding("info", "ledger", "ledger.jsonl not yet created")]
    latest_iso: str = ""
    try:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ed = rec.get("entry_date", "")
                if ed and ed > latest_iso:
                    latest_iso = ed
    except OSError as e:
        return [HealthFinding("error", "ledger", f"could not read ledger: {e}")]
    if not latest_iso:
        return [HealthFinding("warning", "ledger", "no entries in ledger")]
    try:
        latest = date.fromisoformat(latest_iso)
    except ValueError:
        return [HealthFinding("warning", "ledger", f"latest entry_date unparseable: {latest_iso}")]
    age = (date.today() - latest).days
    if age > threshold_days:
        return [HealthFinding(
            "warning", "ledger",
            f"most recent entry was {age} days ago ({latest_iso}); expected ≤ {threshold_days}",
        )]
    return []


def _check_predictions_corruption() -> list[HealthFinding]:
    """Flag unparseable rows in the prediction stores: the primary flat file
    (state/predictions.jsonl — drives IC / metrics / the demotion gate, and
    is rewritten in place by reflection so most at risk of a torn write) plus
    the per-source files under state/predictions/ (news, macro)."""
    out: list[HealthFinding] = []
    files = [predictions_path()] + sorted((STATE_ROOT / "predictions").glob("*.jsonl"))
    for p in files:
        if not p.exists():
            continue
        bad_lines = 0
        try:
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        bad_lines += 1
        except OSError as e:
            out.append(HealthFinding("error", "predictions",
                f"{p.name}: could not read ({e})"))
            continue
        if bad_lines:
            out.append(HealthFinding("error", "predictions",
                f"{p.name}: {bad_lines} unparseable rows"))
    return out


def _check_halt_state() -> list[HealthFinding]:
    """Surface kill-switch state. Halted = error severity so it's prominent
    in the email."""
    try:
        from trading_bot.state.halt import is_halted
        halted, rec = is_halted()
    except Exception as e:
        return [HealthFinding("warning", "kill-switch", f"halt check failed: {e}")]
    if halted:
        reason = (rec.reason if rec else "(no record)")[:200]
        return [HealthFinding("error", "kill-switch",
            f"KILL SWITCH ENGAGED — live-tier entries blocked. Reason: {reason}")]
    return []


def _check_state_growth() -> list[HealthFinding]:
    """Sanity-check on docs/state size. If docs/news has > 365 day-dirs,
    archive-trim hasn't run for a year."""
    out: list[HealthFinding] = []
    news_dir = Path("docs") / "news"
    if news_dir.exists():
        n_dirs = sum(1 for c in news_dir.iterdir() if c.is_dir())
        if n_dirs > 200:
            out.append(HealthFinding(
                "warning", "archive",
                f"docs/news has {n_dirs} edition dirs — archive-trim may need attention",
            ))
    macro_dir = Path("docs") / "macro"
    if macro_dir.exists():
        n_dirs = sum(1 for c in macro_dir.iterdir() if c.is_dir())
        if n_dirs > 60:
            out.append(HealthFinding(
                "warning", "archive",
                f"docs/macro has {n_dirs} edition dirs — archive-trim may need attention",
            ))
    return out


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def _send_summary_email(findings: list[HealthFinding]) -> None:
    """Email a single summary if any findings exist."""
    try:
        from trading_bot.notify.email import send_summary_email, _escape
    except Exception as e:
        log.warning("email helpers unavailable: %s", e)
        return

    severity_count = {
        "error": sum(1 for f in findings if f.severity == "error"),
        "warning": sum(1 for f in findings if f.severity == "warning"),
        "info": sum(1 for f in findings if f.severity == "info"),
    }
    icon = "🔴" if severity_count["error"] else ("🟡" if severity_count["warning"] else "🔵")
    subject = (
        f"[trading-bot] Health {icon} — "
        f"{severity_count['error']}E / {severity_count['warning']}W / {severity_count['info']}I"
    )

    lines = ["Bot-health check found these:\n"]
    for f in findings:
        lines.append(f"  - {f.severity.upper():<7} [{f.category}] {f.message}")
    text_body = "\n".join(lines)

    rows_html = "".join(
        f'<tr>'
        f'<td style="padding:4px 10px;font-size:11px;letter-spacing:0.05em;text-transform:uppercase;color:{_severity_color(f.severity)};font-weight:700;">{f.severity}</td>'
        f'<td style="padding:4px 10px;font-size:13px;color:#374151;"><strong>{_escape(f.category)}</strong></td>'
        f'<td style="padding:4px 10px;font-size:13px;color:#111827;">{_escape(f.message)}</td>'
        f'</tr>'
        for f in findings
    )
    html_body = f"""<!DOCTYPE html><html lang="en"><body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,Segoe UI,Roboto,sans-serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#f3f4f6;padding:20px 0;">
<tr><td align="center"><table role="presentation" width="640" cellpadding="0" cellspacing="0" border="0" style="background:#ffffff;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
<tr><td style="padding:24px 28px 8px;border-bottom:1px solid #e5e7eb;">
<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.05em;color:#6b7280;font-weight:600;">Bot health</div>
<div style="font-size:20px;font-weight:700;color:#111827;letter-spacing:-0.01em;margin-top:4px;">{icon} {len(findings)} finding(s)</div>
</td></tr>
<tr><td style="padding:16px 28px 24px;font-size:13px;color:#374151;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">{rows_html}</table>
</td></tr>
<tr><td style="padding:18px 28px 22px;border-top:1px solid #e5e7eb;background:#fafafa;font-size:11px;color:#9ca3af;">
Auto-generated by the nightly health check.
</td></tr>
</table></td></tr></table></body></html>"""
    try:
        send_summary_email(subject=subject, body_text=text_body, body_html=html_body)
        log.info("Sent health summary email (%d findings)", len(findings))
    except Exception as e:
        log.warning("Health email send failed: %s", e)


def _severity_color(sev: str) -> str:
    return {"error": "#dc2626", "warning": "#b45309", "info": "#6b7280"}.get(sev, "#6b7280")


if __name__ == "__main__":
    sys.exit(main())
