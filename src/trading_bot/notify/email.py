from __future__ import annotations

import logging
import os
from datetime import date

import requests


_BREVO_URL = "https://api.brevo.com/v3/smtp/email"
_BREVO_SENDER_NAME = "trading-bot"
log = logging.getLogger(__name__)


def send_summary_email(
    *,
    subject: str,
    body_markdown: str,
    body_html: str | None = None,
) -> None:
    """Send a summary email via Brevo's transactional API. Reads BREVO_API_KEY,
    NOTIFY_EMAIL_TO, NOTIFY_EMAIL_FROM from env. Silently logs and returns if
    BREVO_API_KEY is not set (e.g., local dev without email configured).

    Brevo's free tier allows 300 emails/day and supports single-sender
    verification — no domain required. Verify the sender address (NOTIFY_EMAIL_FROM)
    once in the Brevo dashboard before sending.
    """
    api_key = os.environ.get("BREVO_API_KEY")
    to_addr = os.environ.get("NOTIFY_EMAIL_TO")
    from_addr = os.environ.get("NOTIFY_EMAIL_FROM")

    if not api_key:
        log.warning("BREVO_API_KEY not set — skipping email")
        return
    if not to_addr or not from_addr:
        log.warning("NOTIFY_EMAIL_TO / NOTIFY_EMAIL_FROM not set — skipping email")
        return

    payload = {
        "sender": {"name": _BREVO_SENDER_NAME, "email": from_addr},
        "to": [{"email": to_addr}],
        "subject": subject,
        "textContent": body_markdown,
        "htmlContent": body_html or _markdown_to_basic_html(body_markdown),
    }

    response = requests.post(
        _BREVO_URL,
        headers={
            "accept": "application/json",
            "api-key": api_key,
            "content-type": "application/json",
        },
        json=payload,
        timeout=15,
    )
    if not response.ok:
        log.error("Brevo send failed: %s %s", response.status_code, response.text)
        response.raise_for_status()


def _markdown_to_basic_html(md: str) -> str:
    """Minimal markdown → HTML for email bodies. We don't need full fidelity here —
    headings, paragraphs, and code blocks are sufficient for daily summaries."""
    lines = md.split("\n")
    out: list[str] = ["<html><body style='font-family: -apple-system, sans-serif;'>"]
    in_code = False
    for line in lines:
        if line.startswith("```"):
            if in_code:
                out.append("</pre>")
                in_code = False
            else:
                out.append("<pre style='background:#f5f5f5;padding:8px;'>")
                in_code = True
            continue
        if in_code:
            out.append(_escape(line))
            continue
        if line.startswith("# "):
            out.append(f"<h1>{_escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{_escape(line[3:])}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{_escape(line[4:])}</h3>")
        elif line.startswith("- "):
            out.append(f"<li>{_escape(line[2:])}</li>")
        elif line.strip() == "":
            out.append("<br>")
        else:
            out.append(f"<p>{_escape(line)}</p>")
    out.append("</body></html>")
    return "\n".join(out)


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_daily_summary(
    *,
    run_date: date,
    region: str,
    entries: dict[str, list[dict]],
    exits: dict[str, list[dict]],
) -> tuple[str, str]:
    """Build the daily summary subject + markdown body.

    entries: {strategy_id: [trade records that were opened today]}
    exits:   {strategy_id: [trade records that were closed today]}
    """
    subject = f"[trading-bot] {region.upper()} daily summary — {run_date.isoformat()}"

    lines: list[str] = []
    lines.append(f"# Daily summary — {run_date.isoformat()} ({region.upper()})")
    lines.append("")

    if not entries and not exits:
        lines.append("No activity today.")
        return subject, "\n".join(lines)

    all_strategy_ids = sorted(set(entries) | set(exits))
    for strategy_id in all_strategy_ids:
        lines.append(f"## {strategy_id}")
        lines.append("")

        strategy_exits = exits.get(strategy_id, [])
        if strategy_exits:
            total_pnl = sum(float(t.get("pnl_gbp") or 0.0) for t in strategy_exits)
            lines.append(f"**Closed positions: {len(strategy_exits)} — total P&L £{total_pnl:+.2f}**")
            lines.append("")
            lines.append("| Ticker | Entry | Exit | P&L £ | P&L % | Reason |")
            lines.append("|---|---|---|---|---|---|")
            for t in strategy_exits:
                lines.append(
                    "| {ticker} | {entry:.2f} | {exit:.2f} | {pnl:+.2f} | {pct:+.2f}% | {reason} |".format(
                        ticker=t["ticker"],
                        entry=float(t["entry_price"]),
                        exit=float(t["exit_price"]),
                        pnl=float(t["pnl_gbp"]),
                        pct=float(t["pnl_pct"]),
                        reason=t.get("exit_reason") or "scheduled",
                    )
                )
            lines.append("")

        strategy_entries = entries.get(strategy_id, [])
        if strategy_entries:
            lines.append(f"**Opened positions: {len(strategy_entries)}**")
            lines.append("")
            lines.append("| Ticker | Entry | Allocation % | Thesis |")
            lines.append("|---|---|---|---|")
            for t in strategy_entries:
                lines.append(
                    "| {ticker} | {entry:.2f} | {alloc:.1f}% | {thesis} |".format(
                        ticker=t["ticker"],
                        entry=float(t["entry_price"]),
                        alloc=float(t["allocation_pct"]),
                        thesis=(t.get("thesis") or "").replace("|", "\\|"),
                    )
                )
            lines.append("")

    return subject, "\n".join(lines)
