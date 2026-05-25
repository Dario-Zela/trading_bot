"""Regression: the slate-table renderer crashed when a strategy's metrics
had `ic` (or n_trades) set to None — `m.get("ic", 0.0)` only fires the
default for a MISSING key, but compute_all_metrics writes None when there
are no graded predictions, and `None:.+.2f}` raises TypeError. That
killed both the editorial render AND the email send (nested in the same
try)."""
from __future__ import annotations

from trading_bot.meta.evolution_v2 import _render_slate_table, EvolutionEdition


def _mk_edition(snapshot_rows):
    return EvolutionEdition(
        week_end="2026-05-23",
        editorial_md="",
        reports=[],
        snapshot_rows=snapshot_rows,
        action_log=[],
    )


def test_slate_table_handles_none_ic():
    rows = [
        {"id": "x", "region": "us", "tier": "shadow",
         "metrics": {"total_pnl_gbp": None, "hit_rate": None,
                     "n_trades": None, "ic": None}},
    ]
    html = _render_slate_table(_mk_edition(rows))
    # No crash + IC printed as 0.00 (the None-coerced default)
    assert "IC +0.00" in html or "IC -0.00" in html or "IC 0.00" in html


def test_slate_table_renders_populated_row():
    rows = [
        {"id": "macro-aligned", "region": "uk-eu", "tier": "alpaca-paper",
         "metrics": {"total_pnl_gbp": 12.34, "hit_rate": 0.55,
                     "n_trades": 100, "ic": 0.17}},
    ]
    html = _render_slate_table(_mk_edition(rows))
    assert "macro-aligned" in html
    assert "£+12.34" in html
    assert "55% hit" in html
    assert "IC +0.17" in html
    assert "across 100 trades" in html


def test_slate_table_handles_missing_metrics_dict():
    rows = [{"id": "x", "region": "us", "tier": "shadow"}]    # no metrics key
    html = _render_slate_table(_mk_edition(rows))
    assert "x" in html        # didn't crash


# -----------------------------------------------------------------------------
# Phase 11: research → action gap parsing
# -----------------------------------------------------------------------------

def test_parse_implication_fits_existing():
    from trading_bot.meta.evolution_v2 import _parse_implication
    sid, kind = _parse_implication("fits existing: momentum-trader")
    assert sid == "momentum-trader"
    assert kind == "tune-or-keep"


def test_parse_implication_spawn_candidate_with_parent():
    from trading_bot.meta.evolution_v2 import _parse_implication
    sid, kind = _parse_implication(
        "spawn-candidate: filing-drift variant of news-reactive — score each candidate ..."
    )
    assert sid == "news-reactive"
    assert kind == "spawn-variant"


def test_parse_implication_out_of_scope_returns_empty():
    from trading_bot.meta.evolution_v2 import _parse_implication
    sid, kind = _parse_implication("out of scope: methodological warning ...")
    assert sid == ""
    assert kind == ""


def test_grade_alignment_spawn_gap_when_no_spawn_action():
    """Research said spawn a news-reactive variant; agent didn't. Should
    surface as GAP — this is the exact scenario the section exists to
    track. Without this the WebSearch brief degrades into decoration."""
    from trading_bot.meta.evolution_v2 import _grade_research_alignment
    verdict, note = _grade_research_alignment(
        expected_sid="news-reactive",
        expected_kind="spawn-variant",
        actions_by_sid={
            "news-reactive": [{"action": "demote", "applied": True, "region": "us"}],
            "momentum-trader": [{"action": "tune", "applied": True, "region": None}],
        },
    )
    assert verdict == "GAP"
    assert "spawn-variant" in note


def test_grade_alignment_acted_when_tune_matches():
    from trading_bot.meta.evolution_v2 import _grade_research_alignment
    verdict, note = _grade_research_alignment(
        expected_sid="momentum-trader",
        expected_kind="tune-or-keep",
        actions_by_sid={
            "momentum-trader": [{"action": "tune", "applied": True, "region": None}],
        },
    )
    assert verdict == "ACTED"
    assert "tune" in note


# -----------------------------------------------------------------------------
# Phase 11: decision grading
# -----------------------------------------------------------------------------

def test_grade_demote_call_stayed_bad_is_good():
    from trading_bot.meta.evolution_v2 import _grade_one_decision
    g = _grade_one_decision(
        "demote",
        pre={"ic": -0.10, "total_pnl_gbp": -50},
        post={"ic": -0.15, "total_pnl_gbp": -120},
    )
    assert g["verdict"] == "GOOD"


def test_grade_demote_call_strong_rebound_is_bad():
    from trading_bot.meta.evolution_v2 import _grade_one_decision
    g = _grade_one_decision(
        "demote",
        pre={"ic": -0.10, "total_pnl_gbp": -50},
        post={"ic": 0.20, "total_pnl_gbp": 200},
    )
    assert g["verdict"] == "BAD"


def test_grade_tune_improving_metrics_is_good():
    from trading_bot.meta.evolution_v2 import _grade_one_decision
    g = _grade_one_decision(
        "tune",
        pre={"ic": -0.05, "total_pnl_gbp": -50},
        post={"ic": 0.10, "total_pnl_gbp": 100},
    )
    assert g["verdict"] == "GOOD"


def test_grade_keep_with_deterioration_is_bad():
    from trading_bot.meta.evolution_v2 import _grade_one_decision
    g = _grade_one_decision(
        "keep",
        pre={"ic": 0.05, "total_pnl_gbp": 50},
        post={"ic": -0.20, "total_pnl_gbp": -200},
    )
    assert g["verdict"] == "BAD"
