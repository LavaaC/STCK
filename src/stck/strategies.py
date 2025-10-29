"""Utilities for persisting and replaying evolved trading strategies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from .data import HistoricalData
from .formulas import TradingFormula
from .portfolio import PortfolioBacktester, TickerAllocation, TradeOrder


@dataclass
class SavedStrategy:
    """Snapshot of a ticker-specific trading strategy for later replay."""

    ticker: str
    generation: int
    formula: TradingFormula
    training_length: int
    initial_cash: float

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "generation": self.generation,
            "training_length": self.training_length,
            "initial_cash": self.initial_cash,
            "formula": self.formula.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SavedStrategy":
        return cls(
            ticker=str(payload["ticker"]),
            generation=int(payload.get("generation", 0)),
            training_length=int(payload.get("training_length", 0)),
            initial_cash=float(payload.get("initial_cash", 0.0)),
            formula=TradingFormula.from_dict(payload["formula"]),
        )

    def clone(self) -> "SavedStrategy":
        return SavedStrategy(
            ticker=self.ticker,
            generation=self.generation,
            formula=self.formula.clone(),
            training_length=self.training_length,
            initial_cash=self.initial_cash,
        )


def save_strategy(path: str | Path, strategy: SavedStrategy) -> Path:
    """Serialize a saved strategy to ``path`` and return the resulting :class:`Path`."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(strategy.to_dict(), fh, indent=2)
    return output


def load_strategy(path: str | Path) -> SavedStrategy:
    """Load a saved strategy from ``path``."""

    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return SavedStrategy.from_dict(payload)


def replay_strategy(strategy: SavedStrategy, data: HistoricalData) -> Sequence[List[TradeOrder]]:
    """Run the strategy over ``data`` and return the per-day orders."""

    allocation = TickerAllocation(ticker=strategy.ticker, formula=strategy.formula.clone(), priority=strategy.formula.priority)
    backtester = PortfolioBacktester(
        data=data,
        allocations=[allocation],
        initial_cash=strategy.initial_cash,
    )
    return backtester.run().orders


def future_orders(strategy: SavedStrategy, data: HistoricalData) -> Sequence[List[TradeOrder]]:
    """Return the orders generated after the original training window."""

    if len(data) <= strategy.training_length:
        return []
    orders = replay_strategy(strategy, data)
    return orders[strategy.training_length :]
