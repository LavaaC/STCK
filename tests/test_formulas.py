from __future__ import annotations

import random

from stck.data import HistoricalData
from stck.formulas import (
    BinaryNode,
    FormulaFactory,
    MovingAverageNode,
    PriceNode,
    TradingFormula,
)
from stck.portfolio import PortfolioBacktester, TickerAllocation


def test_formula_evaluation_and_mutation():
    history = HistoricalData({"AAA": [1.0, 1.2, 1.1, 1.3, 1.25]})
    factory = FormulaFactory(rng=random.Random(123), max_depth=2, data_length=len(history))
    formula = factory.create(priority=1)
    price_history = history.history_for("AAA", 2)
    value = formula.evaluate(price_history)
    mutated = factory.mutate(formula)
    assert isinstance(value, float)
    assert mutated is not formula
    assert mutated.priority in {formula.priority - 1, formula.priority, formula.priority + 1}
    assert formula.evaluation_windows
    assert mutated.evaluation_windows


def test_backtester_allocates_cash_and_updates_equity():
    data = HistoricalData({"AAA": [10, 11, 12, 13]})
    formula = TradingFormula(root=MovingAverageNode(window=1), priority=1)
    allocation = TickerAllocation(ticker="AAA", formula=formula, priority=1)
    backtester = PortfolioBacktester(data=data, allocations=[allocation], initial_cash=1_000)
    result = backtester.run()

    # Equity should increase as prices go up
    assert result.equity_curve[0] <= result.equity_curve[-1]
    # Final cash should increase because only a capped portion is invested.
    assert result.cash_curve[-1] > result.cash_curve[0]


def test_factory_clamps_complexity() -> None:
    factory = FormulaFactory(rng=random.Random(42))
    leaf = PriceNode()
    complex_root: BinaryNode = BinaryNode(op=lambda a, b: a + b, left=leaf, right=leaf, name="add")
    for _ in range(6):
        complex_root = BinaryNode(
            op=lambda a, b: a * b,
            left=complex_root,
            right=BinaryNode(op=lambda a, b: a + b, left=leaf, right=leaf, name="add"),
            name="mul",
        )
    formula = TradingFormula(root=complex_root, priority=3, evaluation_windows=[10, 20])
    clamped = factory.clamp_complexity(formula, max_complexity=5)
    assert clamped.complexity() <= 5
    assert clamped.priority == formula.priority
    assert clamped.evaluation_windows == formula.evaluation_windows
