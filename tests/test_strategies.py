from __future__ import annotations

import json
import random
from pathlib import Path

from stck.data import HistoricalData
from stck.formulas import FormulaFactory, TradingFormula
from stck.strategies import SavedStrategy, future_orders, load_strategy, replay_strategy, save_strategy


def test_trading_formula_serialization_round_trip():
    history = HistoricalData({"AAA": [1.0, 1.1, 1.2, 1.3]})
    factory = FormulaFactory(rng=random.Random(9), max_depth=2, data_length=len(history))
    formula = factory.create(priority=2)
    payload = formula.to_dict()
    restored = TradingFormula.from_dict(json.loads(json.dumps(payload)))
    assert restored.describe() == formula.describe()
    assert restored.priority == formula.priority
    assert restored.evaluation_windows == formula.evaluation_windows


def test_saved_strategy_future_orders(tmp_path: Path):
    base_data = HistoricalData({"AAA": [10, 11, 12]})
    extended_data = HistoricalData({"AAA": [10, 11, 12, 13, 14]})
    formula = TradingFormula.from_dict(
        {
            "priority": 1,
            "evaluation_windows": [2, 3],
            "root": {"type": "moving_average", "window": 1},
        }
    )
    strategy = SavedStrategy(
        ticker="AAA",
        generation=3,
        formula=formula,
        training_length=len(base_data),
        initial_cash=1_000.0,
    )
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir()
    path = strategies_dir / "saved.json"
    save_strategy(path, strategy)
    loaded = load_strategy(path)
    assert loaded.ticker == "AAA"
    assert loaded.training_length == len(base_data)
    orders = list(future_orders(loaded, extended_data))
    assert len(orders) == len(extended_data) - len(base_data)
    # Returned structure should align with replay output
    replay = replay_strategy(loaded, extended_data)
    assert orders == replay[len(base_data) :]
