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
    body_text: str,
    body_html: str,
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
        "textContent": body_text,
        "htmlContent": body_html,
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


_DASHBOARD_URL = "https://dario-zela.github.io/trading_bot/"
_REPO_URL = "https://github.com/Dario-Zela/trading_bot"


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_daily_summary(
    *,
    run_date: date,
    region: str,
    entries: dict[str, list[dict]],
    exits: dict[str, list[dict]],
) -> tuple[str, str, str]:
    """Build the daily summary subject + plain-text body + HTML body.

    entries: {strategy_id: [trade records that were opened today]}
    exits:   {strategy_id: [trade records that were closed today]}
    """
    subject = f"[trading-bot] {region.upper()} daily summary — {run_date.isoformat()}"
    text_body = _render_text_summary(run_date, region, entries, exits)
    html_body = _render_html_summary(run_date, region, entries, exits)
    return subject, text_body, html_body


def _render_text_summary(
    run_date: date,
    region: str,
    entries: dict[str, list[dict]],
    exits: dict[str, list[dict]],
) -> str:
    """Plain-text fallback body. Email clients show this if HTML is blocked."""
    lines: list[str] = [f"Daily summary — {run_date.isoformat()} ({region.upper()})", ""]
    if not entries and not exits:
        lines.append("No activity today.")
        return "\n".join(lines)

    for strategy_id in sorted(set(entries) | set(exits)):
        lines.append(f"## {strategy_id}")
        for t in exits.get(strategy_id, []):
            lines.append(
                f"  {t['ticker']}  "
                f"${float(t['entry_price']):.2f} → ${float(t['exit_price']):.2f}  "
                f"£{float(t['pnl_gbp']):+.2f} ({float(t['pnl_pct']):+.2f}%)  "
                f"[{t.get('exit_reason', 'scheduled')}]"
            )
        for t in entries.get(strategy_id, []):
            lines.append(
                f"  [OPEN] {t['ticker']} @ ${float(t['entry_price']):.2f}  "
                f"alloc {float(t['allocation_pct']):.1f}%"
            )
        lines.append("")
    return "\n".join(lines)


def _render_html_summary(
    run_date: date,
    region: str,
    entries: dict[str, list[dict]],
    exits: dict[str, list[dict]],
) -> str:
    """Inline-CSS HTML body for the daily summary. 600px-wide, Gmail/Outlook safe."""
    total_pnl = sum(
        float(t.get("pnl_gbp") or 0.0)
        for trades in exits.values()
        for t in trades
    )
    total_closed = sum(len(v) for v in exits.values())
    n_strategies = len(set(exits) | set(entries))

    is_positive = total_pnl >= 0
    headline_bg = "linear-gradient(135deg,#f0fdf4 0%,#dcfce7 100%)" if is_positive else "linear-gradient(135deg,#fef2f2 0%,#fee2e2 100%)"
    headline_border = "#bbf7d0" if is_positive else "#fecaca"
    headline_text_color = "#15803d" if is_positive else "#b91c1c"
    headline_sub_color = "#166534" if is_positive else "#991b1b"
    headline_sign = "+" if total_pnl > 0 else ("−" if total_pnl < 0 else "")
    headline_amount = f"{headline_sign}£{abs(total_pnl):.2f}"

    weekday_name = run_date.strftime("%a %d %b %Y")
    region_label = region.upper().replace("-", "/")

    header_html = f"""
    <tr>
      <td style="padding:20px 28px 16px;border-bottom:1px solid #e5e7eb;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
          <tr>
            <td style="font-size:18px;font-weight:700;color:#111827;letter-spacing:-0.01em;">
              ⚡ trading-bot
            </td>
            <td align="right" style="font-size:13px;color:#6b7280;">
              {_escape(region_label)} · {_escape(weekday_name)}
            </td>
          </tr>
        </table>
      </td>
    </tr>
    """

    headline_html = f"""
    <tr>
      <td style="padding:24px 28px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:{headline_bg};border:1px solid {headline_border};border-radius:8px;">
          <tr>
            <td style="padding:20px 24px;">
              <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.05em;color:{headline_text_color};font-weight:600;margin-bottom:6px;">
                Total P&amp;L today
              </div>
              <div style="font-size:32px;font-weight:800;color:{headline_text_color};line-height:1;letter-spacing:-0.02em;">
                {headline_amount}
              </div>
              <div style="font-size:13px;color:{headline_sub_color};margin-top:8px;">
                {total_closed} trade{'s' if total_closed != 1 else ''} closed across {n_strategies} strateg{'ies' if n_strategies != 1 else 'y'}
              </div>
            </td>
          </tr>
        </table>
      </td>
    </tr>
    """

    strategy_blocks: list[str] = []
    for strategy_id in sorted(set(entries) | set(exits)):
        strategy_blocks.append(
            _render_strategy_section(
                strategy_id=strategy_id,
                opened=entries.get(strategy_id, []),
                closed=exits.get(strategy_id, []),
            )
        )

    footer_html = f"""
    <tr>
      <td style="padding:24px 28px 24px;border-top:1px solid #e5e7eb;background:#fafafa;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
          <tr>
            <td>
              <a href="{_DASHBOARD_URL}" style="display:inline-block;padding:10px 18px;background:#2563eb;color:#ffffff;text-decoration:none;border-radius:6px;font-size:13px;font-weight:600;box-shadow:0 1px 2px rgba(37,99,235,0.3);">
                Open dashboard →
              </a>
            </td>
            <td align="right" style="font-size:11px;color:#9ca3af;">
              <a href="{_REPO_URL}" style="color:#6b7280;text-decoration:none;">github.com/Dario-Zela/trading_bot</a>
            </td>
          </tr>
        </table>
      </td>
    </tr>
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f3f4f6;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#f3f4f6;">
  <tr>
    <td align="center" style="padding:24px 12px;">
      <table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0" style="max-width:600px;width:100%;background:#ffffff;border-radius:8px;overflow:hidden;border:1px solid #e5e7eb;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
        {header_html}
        {headline_html}
        {''.join(strategy_blocks)}
        {footer_html}
      </table>
    </td>
  </tr>
</table>
</body>
</html>"""


def _render_strategy_section(
    *, strategy_id: str, opened: list[dict], closed: list[dict]
) -> str:
    """One per-strategy block within the email body."""
    pnl = sum(float(t.get("pnl_gbp") or 0.0) for t in closed)
    avg_pct = (sum(float(t.get("pnl_pct") or 0.0) for t in closed) / len(closed)) if closed else 0.0
    wins = sum(1 for t in closed if float(t.get("pnl_gbp") or 0.0) > 0)
    pnl_color = "#15803d" if pnl >= 0 else "#b91c1c"
    avg_color = "#15803d" if avg_pct >= 0 else "#b91c1c"
    pnl_sign = "+" if pnl > 0 else ("−" if pnl < 0 else "")
    avg_sign = "+" if avg_pct > 0 else ("−" if avg_pct < 0 else "")

    stat_row = f"""
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:16px;">
      <tr>
        <td width="33%" style="padding:10px 12px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;">
          <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#6b7280;font-weight:500;">P&amp;L</div>
          <div style="font-size:18px;font-weight:700;color:{pnl_color};letter-spacing:-0.01em;">{pnl_sign}£{abs(pnl):.2f}</div>
        </td>
        <td width="4"></td>
        <td width="33%" style="padding:10px 12px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;">
          <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#6b7280;font-weight:500;">Avg %</div>
          <div style="font-size:18px;font-weight:700;color:{avg_color};letter-spacing:-0.01em;">{avg_sign}{abs(avg_pct):.2f}%</div>
        </td>
        <td width="4"></td>
        <td width="33%" style="padding:10px 12px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;">
          <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#6b7280;font-weight:500;">Hit rate</div>
          <div style="font-size:18px;font-weight:700;color:#111827;letter-spacing:-0.01em;">{wins} / {len(closed)}</div>
        </td>
      </tr>
    </table>
    """

    closed_html_parts: list[str] = []
    if closed:
        closed_html_parts.append(
            '<div style="font-size:12px;text-transform:uppercase;letter-spacing:0.05em;color:#6b7280;font-weight:600;margin:0 0 8px;">Closed positions</div>'
        )
        for t in closed:
            closed_html_parts.append(_render_trade_card(t))

    opened_html_parts: list[str] = []
    if opened:
        opened_html_parts.append(
            '<div style="font-size:12px;text-transform:uppercase;letter-spacing:0.05em;color:#6b7280;font-weight:600;margin:14px 0 8px;">Open positions</div>'
        )
        for t in opened:
            opened_html_parts.append(_render_open_card(t))

    return f"""
    <tr>
      <td style="padding:0 28px 12px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
          <tr>
            <td>
              <h2 style="margin:0 0 4px;font-size:16px;color:#111827;font-weight:600;">{_escape(strategy_id)}</h2>
              <div style="margin-bottom:12px;">
                <span style="display:inline-block;font-size:10px;text-transform:uppercase;letter-spacing:0.05em;padding:2px 7px;border-radius:3px;background:#f1f5f9;color:#475569;font-weight:600;">{_escape((closed[0]['tier'] if closed else opened[0]['tier']).upper()) if (closed or opened) else 'SHADOW'}</span>
              </div>
            </td>
          </tr>
        </table>
        {stat_row}
        {''.join(closed_html_parts)}
        {''.join(opened_html_parts)}
      </td>
    </tr>
    """


def _reason_badge_style(reason: str) -> tuple[str, str]:
    if reason == "take_profit":
        return ("#dcfce7", "#15803d")
    if reason == "stop":
        return ("#fee2e2", "#b91c1c")
    return ("#f1f5f9", "#475569")


def _render_trade_card(t: dict) -> str:
    """One closed-trade card with three labeled sections."""
    pnl = float(t.get("pnl_gbp") or 0.0)
    pnl_pct = float(t.get("pnl_pct") or 0.0)
    is_win = pnl >= 0
    pnl_color = "#15803d" if is_win else "#b91c1c"
    pnl_sign = "+" if pnl > 0 else ("−" if pnl < 0 else "")
    pct_sign = "+" if pnl_pct > 0 else ("−" if pnl_pct < 0 else "")
    reason = t.get("exit_reason") or "scheduled"
    badge_bg, badge_fg = _reason_badge_style(reason)

    thesis = _escape(t.get("thesis") or "(no thesis recorded)")
    outcome = _escape(t.get("outcome_notes") or "(no outcome analysis yet — populated by the reflection agent in Wave 6)")
    risks = _escape(t.get("risks_observed") or "(no risks flagged for this trade)")

    return f"""
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:8px;border:1px solid #e5e7eb;border-radius:6px;">
      <tr>
        <td style="padding:12px 14px;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
            <tr>
              <td style="font-size:14px;font-weight:700;color:#111827;">{_escape(t['ticker'])}</td>
              <td align="right" style="font-size:14px;font-weight:700;color:{pnl_color};">{pnl_sign}£{abs(pnl):.2f} ({pct_sign}{abs(pnl_pct):.2f}%)</td>
            </tr>
            <tr>
              <td colspan="2" style="font-size:12px;color:#6b7280;padding-top:2px;padding-bottom:8px;">
                ${float(t['entry_price']):.2f} → ${float(t['exit_price']):.2f} ·
                <span style="display:inline-block;font-size:10px;padding:1px 6px;border-radius:3px;background:{badge_bg};color:{badge_fg};">{_escape(reason)}</span>
              </td>
            </tr>
            <tr>
              <td colspan="2" style="padding-top:8px;border-top:1px solid #f3f4f6;font-size:12px;color:#374151;line-height:1.5;">
                <strong style="display:block;font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#6b7280;font-weight:600;margin-bottom:3px;">Why entered</strong>
                <p style="margin:0 0 10px;">{thesis}</p>

                <strong style="display:block;font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#0369a1;font-weight:600;margin-bottom:3px;">What happened</strong>
                <p style="margin:0 0 10px;">{outcome}</p>

                <strong style="display:block;font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#b45309;font-weight:600;margin-bottom:3px;">⚠ Risks observed</strong>
                <p style="margin:0;">{risks}</p>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
    """


def _render_open_card(t: dict) -> str:
    """Open-position card (no exit data yet)."""
    return f"""
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:8px;border:1px solid #e5e7eb;border-radius:6px;background:#fafbfc;">
      <tr>
        <td style="padding:10px 14px;font-size:13px;color:#374151;">
          <strong style="color:#111827;">{_escape(t['ticker'])}</strong> ·
          entry ${float(t['entry_price']):.2f} · alloc {float(t['allocation_pct']):.1f}%
        </td>
      </tr>
    </table>
    """
