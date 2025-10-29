"""Portfolio management and backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .data import HistoricalData, PriceHistory
from .formulas import TradingFormula


@dataclass
class PortfolioState:
    """Represents the portfolio at a single point in time."""

    cash: float
    holdings: Dict[str, float] = field(default_factory=dict)

    def value(self, prices: Dict[str, float]) -> float:
        equity = self.cash
        for ticker, shares in self.holdings.items():
            equity += shares * prices.get(ticker, 0.0)
        return equity

    def clone(self) -> "PortfolioState":
        return PortfolioState(cash=self.cash, holdings=dict(self.holdings))


@dataclass
class TickerAllocation:
    ticker: str
    formula: TradingFormula
    priority: int = 0
    weight: float = 1.0

    def desired_fraction(self, history) -> float:
        fraction = self.formula.desired_fraction(history)
        return max(0.0, min(1.0, fraction))

    def weighted_signal(self, history) -> float:
        fraction = self.desired_fraction(history)
        return fraction * max(0.0, self.weight)


@dataclass
class BacktestResult:
    allocations: List[Dict[str, float]]
    equity_curve: List[float]
    cash_curve: List[float]
    value_breakdown: List[Dict[str, float]]
    value_percentages: List[Dict[str, float]]
    tickers: List[str]
    start_index: int = 0

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1]

    @property
    def final_cash(self) -> float:
        return self.cash_curve[-1]

    def top_stock(self) -> tuple[str, float] | None:
        if not self.value_breakdown:
            return None
        final_values = {
            ticker: value
            for ticker, value in self.value_breakdown[-1].items()
            if ticker != "CASH"
        }
        if not final_values:
            return None
        ticker = max(final_values, key=final_values.get)
        return ticker, final_values[ticker]

    def max_drawdown(self) -> float:
        peak = self.equity_curve[0]
        max_dd = 0.0
        for value in self.equity_curve:
            peak = max(peak, value)
            drawdown = (peak - value) / peak if peak else 0.0
            max_dd = max(max_dd, drawdown)
        return max_dd


class PortfolioBacktester:
    """Simulate a portfolio driven by ticker-specific trading formulas."""

    def __init__(self, data: HistoricalData, allocations: List[TickerAllocation], initial_cash: float = 100_000.0) -> None:
        if not allocations:
            raise ValueError("At least one ticker allocation is required")
        self.data = data
        self.allocations = allocations
        self.initial_cash = initial_cash
        self._sorted_allocations = sorted(self.allocations, key=lambda a: a.priority, reverse=True)
        self._tracked_tickers = sorted({allocation.ticker for allocation in self._sorted_allocations})
        self._price_series = {ticker: self.data.prices[ticker] for ticker in self._tracked_tickers}
        self._signals_buffer: Dict[str, float] = {}

    def _current_prices(self, index: int) -> Dict[str, float]:  # pragma: no cover - retained for API compatibility
        return {ticker: series[index] for ticker, series in self._price_series.items()}

    def run(self, *, start_index: int = 0) -> BacktestResult:
        if start_index < 0 or start_index >= len(self.data):
            raise ValueError("start_index must be within the price history range")

        portfolio = PortfolioState(cash=self.initial_cash)
        allocations_history: List[Dict[str, float]] = []
        equity_curve: List[float] = []
        cash_curve: List[float] = []
        value_breakdown: List[Dict[str, float]] = []
        value_percentages: List[Dict[str, float]] = []
        tracked_tickers = self._tracked_tickers
        tracked_with_cash = tracked_tickers + ["CASH"]

        prices_buffer = {ticker: 0.0 for ticker in tracked_tickers}
        breakdown_buffer = {ticker: 0.0 for ticker in tracked_tickers}
        percentage_buffer = {ticker: 0.0 for ticker in tracked_with_cash}
        histories: Dict[str, PriceHistory] = {
            ticker: PriceHistory(ticker=ticker, prices=series, end_index=start_index)
            for ticker, series in self._price_series.items()
        }

        for index in range(start_index, len(self.data)):
            for ticker, series in self._price_series.items():
                prices_buffer[ticker] = series[index]
                histories[ticker].advance_to(index)

            total_equity = portfolio.value(prices_buffer)
            desired = self._desired_allocations(index, total_equity, histories)
            portfolio = self._rebalance(portfolio, prices_buffer, desired)

            allocations_history.append({ticker: shares for ticker, shares in portfolio.holdings.items()})
            equity = portfolio.value(prices_buffer)
            equity_curve.append(equity)
            cash_curve.append(portfolio.cash)

            for ticker in tracked_tickers:
                breakdown_buffer[ticker] = (
                    portfolio.holdings.get(ticker, 0.0) * prices_buffer[ticker]
                )
            breakdown = breakdown_buffer.copy()
            breakdown["CASH"] = portfolio.cash
            value_breakdown.append(breakdown)
            if equity > 0:
                for ticker, value in breakdown.items():
                    percentage_buffer[ticker] = value / equity
            else:
                for ticker in breakdown:
                    percentage_buffer[ticker] = 0.0
            value_percentages.append(dict(percentage_buffer))

        return BacktestResult(
            allocations=allocations_history,
            equity_curve=equity_curve,
            cash_curve=cash_curve,
            value_breakdown=value_breakdown,
            value_percentages=value_percentages,
            tickers=tracked_with_cash,
            start_index=start_index,
        )

    def _desired_allocations(
        self, index: int, total_equity: float, histories: Dict[str, PriceHistory]
    ) -> Dict[str, float]:
        signals = self._signals_buffer
        signals.clear()
        for allocation in self._sorted_allocations:
            history = histories[allocation.ticker]
            weighted = allocation.weighted_signal(history)
            if weighted > 0:
                signals[allocation.ticker] = signals.get(allocation.ticker, 0.0) + weighted

        total_signal = sum(signals.values())
        if total_signal <= 1e-9:
            return {}

        desired_values = {
            ticker: (signal / total_signal) * total_equity for ticker, signal in signals.items()
        }

        allocation_cap = 0.10 * total_equity
        capped_values = {
            ticker: min(value, allocation_cap) for ticker, value in desired_values.items()
        }
        return capped_values

    def _rebalance(
        self,
        portfolio: PortfolioState,
        prices: Dict[str, float],
        desired_values: Dict[str, float],
    ) -> PortfolioState:
        new_portfolio = portfolio.clone()
        # First, sell holdings that exceed desired values or belong to tickers without allocations.
        for ticker, shares in list(new_portfolio.holdings.items()):
            price = prices.get(ticker, 0.0)
            current_value = shares * price
            target_value = desired_values.get(ticker, 0.0)
            if current_value > target_value + 1e-8:
                value_to_sell = current_value - target_value
                shares_to_sell = value_to_sell / price if price else 0.0
                new_portfolio.holdings[ticker] = max(0.0, shares - shares_to_sell)
                new_portfolio.cash += value_to_sell
        # Remove zero holdings to keep state tidy
        new_portfolio.holdings = {t: s for t, s in new_portfolio.holdings.items() if s > 1e-9}

        # Then, buy up to desired values using available cash.
        for ticker, target_value in desired_values.items():
            price = prices.get(ticker, 0.0)
            current_shares = new_portfolio.holdings.get(ticker, 0.0)
            current_value = current_shares * price
            if target_value > current_value + 1e-8 and price > 0:
                value_to_buy = min(target_value - current_value, new_portfolio.cash)
                shares_to_buy = value_to_buy / price
                new_portfolio.cash -= value_to_buy
                new_portfolio.holdings[ticker] = current_shares + shares_to_buy
        return new_portfolio
