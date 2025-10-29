from __future__ import annotations

import random

from stck.data import HistoricalData
from stck.formulas import FormulaFactory, MovingAverageNode, TradingFormula
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
    # Cash should decrease once invested
    assert result.cash_curve[0] >= result.cash_curve[-1]

