from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from trading_bot.state.paths import predictions_path


@dataclass
class PredictionRecord:
    """One graded prediction. Wave 1 only logs the picks actually traded;
    later waves will log the full 200-prediction set per strategy per day."""

    strategy_id: str
    region: str
    prediction_date: str  # ISO date
    ticker: str
    predicted_class: str  # strong_up / mild_up / flat / mild_down / strong_down
    predicted_return_pct: float
    conviction: float
    rationale: str = ""

    actual_return_pct: float | None = None
    actual_class: str | None = None
    was_traded: bool = False

    # Per-prediction context that explains what inputs the strategy saw
    # when it produced this row. Used by the tool-attribution layer to
    # measure IC contribution per input — when this strategy got the
    # daily news brief vs not, what did its IC look like? Reflection
    # post-grade by the same LLM that reads it in subsequent prompts.
    # Empty / None for strategies that don't yet emit it.
    tools_used: list[str] = field(default_factory=list)
    # Free-form one-line reflection on whether prediction held up.
    # Populated by the daily reflect_predictions_on_day pass.
    reflection: str = ""

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def append_prediction(record: PredictionRecord) -> None:
    path = predictions_path()
    with path.open("a") as f:
        f.write(json.dumps(asdict(record)) + "\n")
