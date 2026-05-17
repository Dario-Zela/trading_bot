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

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def append_prediction(record: PredictionRecord) -> None:
    path = predictions_path()
    with path.open("a") as f:
        f.write(json.dumps(asdict(record)) + "\n")
