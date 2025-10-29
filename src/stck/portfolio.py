"""Portfolio management and backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .data import HistoricalData
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

    def desired_fraction(self, history) -> float:
        fraction = self.formula.desired_fraction(history)
        return max(0.0, min(1.0, fraction))


@dataclass
class BacktestResult:
    allocations: List[Dict[str, float]]
    equity_curve: List[float]
    cash_curve: List[float]
    orders: List[List["TradeOrder"]]

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1]

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

    def _current_prices(self, index: int) -> Dict[str, float]:
        return {ticker: series[index] for ticker, series in self.data.prices.items()}

    def run(self) -> BacktestResult:
        portfolio = PortfolioState(cash=self.initial_cash)
        allocations_history: List[Dict[str, float]] = []
        equity_curve: List[float] = []
        cash_curve: List[float] = []
        orders_curve: List[List[TradeOrder]] = []

        for index in range(len(self.data)):
            prices = self._current_prices(index)
            total_equity = portfolio.value(prices)
            desired = self._desired_allocations(index, total_equity)
            portfolio, orders = self._rebalance(portfolio, prices, desired)

            allocations_history.append({ticker: shares for ticker, shares in portfolio.holdings.items()})
            equity_curve.append(portfolio.value(prices))
            cash_curve.append(portfolio.cash)
            orders_curve.append(orders)

        return BacktestResult(
            allocations=allocations_history,
            equity_curve=equity_curve,
            cash_curve=cash_curve,
            orders=orders_curve,
        )

    def _desired_allocations(self, index: int, total_equity: float) -> Dict[str, float]:
        priority_order = sorted(self.allocations, key=lambda a: a.priority, reverse=True)
        remaining_equity = total_equity
        desired_values: Dict[str, float] = {}
        for allocation in priority_order:
            history = self.data.history_for(allocation.ticker, index)
            fraction = allocation.desired_fraction(history)
            target_value = min(fraction * total_equity, remaining_equity)
            desired_values[allocation.ticker] = target_value
            remaining_equity = max(0.0, remaining_equity - target_value)
        return desired_values

    def _rebalance(
        self,
        portfolio: PortfolioState,
        prices: Dict[str, float],
        desired_values: Dict[str, float],
    ) -> tuple[PortfolioState, List["TradeOrder"]]:
        new_portfolio = portfolio.clone()
        orders: List[TradeOrder] = []

        # Determine desired trade for each ticker exactly once per day.
        tickers = set(desired_values) | set(new_portfolio.holdings)
        for ticker in sorted(tickers):
            price = prices.get(ticker, 0.0)
            if price <= 0:
                continue
            current_shares = new_portfolio.holdings.get(ticker, 0.0)
            target_value = desired_values.get(ticker, 0.0)
            target_shares = target_value / price if price else 0.0
            delta_shares = target_shares - current_shares

            if abs(delta_shares) <= 1e-8:
                continue

            if delta_shares > 0:
                max_affordable_shares = new_portfolio.cash / price
                shares_to_buy = min(delta_shares, max_affordable_shares)
                if shares_to_buy <= 1e-9:
                    continue
                value = shares_to_buy * price
                new_portfolio.cash -= value
                new_portfolio.holdings[ticker] = current_shares + shares_to_buy
                orders.append(
                    TradeOrder(ticker=ticker, action="buy", shares=shares_to_buy, price=price, value=value)
                )
            else:
                shares_to_sell = min(-delta_shares, current_shares)
                if shares_to_sell <= 1e-9:
                    continue
                value = shares_to_sell * price
                new_portfolio.cash += value
                remaining_shares = current_shares - shares_to_sell
                if remaining_shares <= 1e-9:
                    new_portfolio.holdings.pop(ticker, None)
                else:
                    new_portfolio.holdings[ticker] = remaining_shares
                orders.append(
                    TradeOrder(ticker=ticker, action="sell", shares=shares_to_sell, price=price, value=value)
                )

        # Clean up small residual holdings for consistency
        new_portfolio.holdings = {t: s for t, s in new_portfolio.holdings.items() if s > 1e-9}
        return new_portfolio, orders


@dataclass(frozen=True)
class TradeOrder:
    ticker: str
    action: str  # "buy" or "sell"
    shares: float
    price: float
    value: float
