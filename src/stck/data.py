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
        if 0 in lengths:
            raise ValueError("Price series must contain at least one observation")
        if len(lengths) != 1:
            raise ValueError("All price series must share the same length")

    @property
    def tickers(self) -> Iterable[str]:
        return self.prices.keys()

    def __len__(self) -> int:
        return len(next(iter(self.prices.values())))

    def history_for(self, ticker: str, index: int) -> "PriceHistory":
        if ticker not in self.prices:
            available = ", ".join(sorted(self.prices))
            raise KeyError(f"Ticker '{ticker}' not found in historical data. Available: {available}")
        series = self.prices[ticker]
        if index < 0 or index >= len(series):
            raise IndexError("index out of range for price series")
        return PriceHistory(ticker=ticker, prices=series, end_index=index)

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


@dataclass(slots=True)
class PriceHistory:
    """Represents the price evolution of a ticker up to a specific point."""

    ticker: str
    prices: Sequence[float]
    end_index: int

    def advance_to(self, index: int) -> None:
        if index < 0 or index >= len(self.prices):
            raise IndexError("index out of range for price history advance")
        self.end_index = index

    @property
    def current(self) -> float:
        return self.prices[self.end_index]

    def tail(self, length: int) -> List[float]:
        if length <= 0:
            return []
        start = max(0, self.end_index - length + 1)
        return list(self.prices[start : self.end_index + 1])

    def rolling_min(self, window: int) -> float:
        if window <= 0:
            return min(self.prices[: self.end_index + 1])
        start = max(0, self.end_index - window + 1)
        if start > self.end_index:
            return min(self.prices[: self.end_index + 1])
        return min(self.prices[start : self.end_index + 1])

    def rolling_max(self, window: int) -> float:
        if window <= 0:
            return max(self.prices[: self.end_index + 1])
        start = max(0, self.end_index - window + 1)
        if start > self.end_index:
            return max(self.prices[: self.end_index + 1])
        return max(self.prices[start : self.end_index + 1])

    def percent_change(self, periods: int) -> float:
        if periods <= 0 or periods > self.end_index:
            return 0.0
        prev_index = self.end_index - periods
        prev = self.prices[prev_index]
        if prev == 0:
            return 0.0
        return (self.current - prev) / prev

    def rolling_mean(self, window: int) -> float:
        if window <= 0:
            return self.current
        start = max(0, self.end_index - window + 1)
        window_values = self.prices[start : self.end_index + 1]
        if not window_values:
            return self.current
        return sum(window_values) / len(window_values)

    def rolling_std(self, window: int) -> float:
        start = max(0, self.end_index - window + 1)
        window_values = self.prices[start : self.end_index + 1]
        if len(window_values) <= 1:
            return 0.0
        mean = sum(window_values) / len(window_values)
        variance = sum((v - mean) ** 2 for v in window_values) / (len(window_values) - 1)
        return variance**0.5

    def exponential_moving_average(self, window: int) -> float:
        if window <= 0:
            return self.current
        alpha = 2.0 / (window + 1)
        ema = self.prices[0]
        for price in self.prices[1 : self.end_index + 1]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def relative_strength_index(self, window: int) -> float:
        if window <= 0 or self.end_index <= 0:
            return 50.0
        deltas = [
            self.prices[i] - self.prices[i - 1]
            for i in range(1, self.end_index + 1)
        ]
        window = min(window, len(deltas))
        if window == 0:
            return 50.0
        recent = deltas[-window:]
        gains = sum(delta for delta in recent if delta > 0) / window
        losses = sum(-delta for delta in recent if delta < 0) / window
        if losses == 0:
            return 100.0 if gains > 0 else 50.0
        if gains == 0:
            return 0.0
        rs = gains / losses
        return 100.0 - (100.0 / (1 + rs))

    def macd(self, fast: int, slow: int) -> float:
        if fast <= 0:
            fast = 1
        if slow <= fast:
            slow = fast + 1
        fast_ema = self.exponential_moving_average(fast)
        slow_ema = self.exponential_moving_average(slow)
        return fast_ema - slow_ema

    def bollinger_band_width(self, window: int, multiplier: float = 2.0) -> float:
        if window <= 0:
            window = 1
        mean = self.rolling_mean(window)
        std = self.rolling_std(window)
        upper = mean + multiplier * std
        lower = mean - multiplier * std
        if mean == 0:
            return 0.0
        return (upper - lower) / mean
