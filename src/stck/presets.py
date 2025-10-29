"""Preset trading formulas for specific tickers.

The simulator now seeds each supported ticker with a deterministic formula so that
every symbol begins its evolutionary journey from a unique starting point. The
population is still expanded through mutation, but the base formula provides a
distinct personality per ticker.
"""

from __future__ import annotations

import math
import operator
from typing import Callable, Dict, Optional

from .formulas import (
    BinaryNode,
    MovingAverageNode,
    MomentumNode,
    PriceNode,
    TradingFormula,
    UnaryNode,
    VolatilityNode,
)


def _trend_following() -> TradingFormula:
    fast = MovingAverageNode(window=10)
    slow = MovingAverageNode(window=30)
    spread = BinaryNode(op=operator.sub, left=fast, right=slow, name="sub")
    signal = UnaryNode(op=math.tanh, operand=spread, name="tanh")
    root = BinaryNode(op=operator.add, left=signal, right=MomentumNode(lookback=5), name="add")
    return TradingFormula(root=root, priority=1, evaluation_windows=[60, 120, 180])


def _volatility_scalp() -> TradingFormula:
    price_vs_avg = BinaryNode(
        op=operator.sub,
        left=PriceNode(),
        right=MovingAverageNode(window=20),
        name="sub",
    )
    volatility_bias = BinaryNode(
        op=operator.add,
        left=VolatilityNode(window=15),
        right=MomentumNode(lookback=3),
        name="add",
    )
    root = BinaryNode(op=operator.mul, left=price_vs_avg, right=volatility_bias, name="mul")
    return TradingFormula(root=root, priority=0, evaluation_windows=[45, 90, 180])


def _mean_reversion() -> TradingFormula:
    short_avg = MovingAverageNode(window=5)
    long_avg = MovingAverageNode(window=50)
    difference = BinaryNode(op=operator.sub, left=short_avg, right=long_avg, name="sub")
    compressed = UnaryNode(op=lambda x: math.tanh(x) * 0.8, operand=difference, name="scaled_tanh")
    volatility = VolatilityNode(window=25)
    root = BinaryNode(op=operator.sub, left=compressed, right=volatility, name="sub")
    return TradingFormula(root=root, priority=-1, evaluation_windows=[30, 120, 240])


def _momentum_breakout() -> TradingFormula:
    momentum_fast = MomentumNode(lookback=4)
    momentum_slow = MomentumNode(lookback=20)
    trend = BinaryNode(op=operator.sub, left=momentum_fast, right=momentum_slow, name="sub")
    price_vs_avg = BinaryNode(
        op=operator.sub,
        left=PriceNode(),
        right=MovingAverageNode(window=40),
        name="sub",
    )
    root = BinaryNode(op=operator.add, left=trend, right=price_vs_avg, name="add")
    return TradingFormula(root=root, priority=2, evaluation_windows=[90, 180, 270])


def _volatility_compression() -> TradingFormula:
    vol_short = VolatilityNode(window=10)
    vol_long = VolatilityNode(window=50)
    compression = BinaryNode(op=operator.sub, left=vol_long, right=vol_short, name="sub")
    momentum = UnaryNode(op=math.tanh, operand=MomentumNode(lookback=8), name="tanh")
    root = BinaryNode(op=operator.add, left=compression, right=momentum, name="add")
    return TradingFormula(root=root, priority=1, evaluation_windows=[60, 150, 300])


def _range_trader() -> TradingFormula:
    price = PriceNode()
    upper_band = BinaryNode(
        op=operator.add,
        left=MovingAverageNode(window=25),
        right=VolatilityNode(window=25),
        name="add",
    )
    lower_band = BinaryNode(
        op=operator.sub,
        left=MovingAverageNode(window=25),
        right=VolatilityNode(window=25),
        name="sub",
    )
    midpoint = BinaryNode(op=operator.sub, left=upper_band, right=lower_band, name="sub")
    root = BinaryNode(op=operator.sub, left=midpoint, right=price, name="sub")
    return TradingFormula(root=root, priority=0, evaluation_windows=[45, 120, 200])


PRESET_BUILDERS: Dict[str, Callable[[], TradingFormula]] = {
    "AAPL": _trend_following,
    "MSFT": _volatility_scalp,
    "GOOGL": _mean_reversion,
    "AMZN": _momentum_breakout,
    "NVDA": _volatility_compression,
    "TSLA": _range_trader,
}


def get_preset_formula(ticker: str) -> Optional[TradingFormula]:
    """Return a preset formula for ``ticker`` if one is defined."""

    builder = PRESET_BUILDERS.get(ticker.upper())
    if builder is None:
        return None
    return builder()


__all__ = ["get_preset_formula", "PRESET_BUILDERS"]
