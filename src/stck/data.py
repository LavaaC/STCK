"""Data structures for handling historical price information."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class HistoricalData:
    """Container for synchronized historical price series.

    The class validates that each ticker has the same number of observations.
    Prices should be ordered chronologically.
    """

    prices: Dict[str, Sequence[float]]

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        lengths = {len(series) for series in self.prices.values()}
        if not lengths:
            raise ValueError("HistoricalData requires at least one ticker")
        if len(lengths) != 1:
            raise ValueError("All price series must share the same length")

    @property
    def tickers(self) -> Iterable[str]:
        return self.prices.keys()

    def __len__(self) -> int:
        return len(next(iter(self.prices.values())))

    def history_for(self, ticker: str, index: int) -> "PriceHistory":
        series = self.prices[ticker]
        if index < 0 or index >= len(series):
            raise IndexError("index out of range for price series")
        return PriceHistory(ticker=ticker, prices=list(series[: index + 1]))

    def tail(self, length: int) -> "HistoricalData":
        """Return a new :class:`HistoricalData` containing the trailing observations."""

        if length <= 0:
            raise ValueError("length must be positive")
        length = min(length, len(self))
        return HistoricalData({ticker: series[-length:] for ticker, series in self.prices.items()})

    def slice(self, start: int, end: int | None = None) -> "HistoricalData":
        """Return a sliced view of the historical data between ``start`` and ``end``."""

        if start < 0:
            raise ValueError("start must be non-negative")
        if end is None:
            end = len(self)
        if start >= end:
            raise ValueError("start must be less than end")
        end = min(end, len(self))
        return HistoricalData({ticker: series[start:end] for ticker, series in self.prices.items()})


@dataclass(frozen=True)
class PriceHistory:
    """Represents the price evolution of a ticker up to a specific point."""

    ticker: str
    prices: List[float]

    @property
    def current(self) -> float:
        return self.prices[-1]

    def tail(self, length: int) -> List[float]:
        if length <= 0:
            return []
        return self.prices[-length:]

    def percent_change(self, periods: int) -> float:
        if periods <= 0 or periods >= len(self.prices):
            return 0.0
        prev = self.prices[-periods - 1]
        if prev == 0:
            return 0.0
        return (self.current - prev) / prev

    def rolling_mean(self, window: int) -> float:
        if window <= 0:
            return self.current
        values = self.tail(window)
        if not values:
            return self.current
        return sum(values) / len(values)

    def rolling_std(self, window: int) -> float:
        values = self.tail(window)
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance**0.5
