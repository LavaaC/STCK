"""Trading formula representations and utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Protocol

from .data import PriceHistory


class FormulaNode(Protocol):

    """Interface for nodes that can be evaluated and mutated."""


    def evaluate(self, history: PriceHistory) -> float:
        ...

    def clone(self) -> "FormulaNode":
        ...

    def mutate(self, rng: random.Random) -> "FormulaNode":
        ...

    def complexity(self) -> int:
        ...

    def describe(self) -> str:
        ...




# Indicator nodes -----------------------------------------------------------------


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-9:
        return 0.0
    return numerator / denominator


@dataclass(frozen=True)
class PriceNode:
    def evaluate(self, history: PriceHistory) -> float:
        return history.current

    def clone(self) -> "PriceNode":
        return PriceNode()

    def mutate(self, rng: random.Random) -> "PriceNode":  # pragma: no cover - nothing to mutate
        return self

    def complexity(self) -> int:
        return 1

    def describe(self) -> str:
        return "price"


@dataclass(frozen=True)
class MovingAverageNode:
    window: int

    def evaluate(self, history: PriceHistory) -> float:
        return history.rolling_mean(self.window)

    def clone(self) -> "MovingAverageNode":
        return MovingAverageNode(window=self.window)

    def mutate(self, rng: random.Random) -> "MovingAverageNode":
        delta = rng.choice([-2, -1, 1, 2])
        new_window = max(1, self.window + delta)
        return MovingAverageNode(window=new_window)

    def complexity(self) -> int:
        return 1

    def describe(self) -> str:
        return f"sma({self.window})"


@dataclass(frozen=True)
class MomentumNode:
    lookback: int

    def evaluate(self, history: PriceHistory) -> float:
        return history.percent_change(self.lookback)

    def clone(self) -> "MomentumNode":
        return MomentumNode(lookback=self.lookback)

    def mutate(self, rng: random.Random) -> "MomentumNode":
        delta = rng.choice([-2, -1, 1, 2])
        new_lookback = max(1, self.lookback + delta)
        return MomentumNode(lookback=new_lookback)

    def complexity(self) -> int:
        return 1

    def describe(self) -> str:
        return f"momentum({self.lookback})"


@dataclass(frozen=True)
class VolatilityNode:
    window: int

    def evaluate(self, history: PriceHistory) -> float:
        return history.rolling_std(self.window)

    def clone(self) -> "VolatilityNode":
        return VolatilityNode(window=self.window)

    def mutate(self, rng: random.Random) -> "VolatilityNode":
        delta = rng.choice([-2, -1, 1, 2])
        new_window = max(1, self.window + delta)
        return VolatilityNode(window=new_window)

    def complexity(self) -> int:
        return 1

    def describe(self) -> str:
        return f"vol({self.window})"


# Arithmetic nodes ----------------------------------------------------------------


@dataclass(frozen=True)
class UnaryNode:
    op: Callable[[float], float]
    operand: FormulaNode
    name: str

    def evaluate(self, history: PriceHistory) -> float:
        return self.op(self.operand.evaluate(history))

    def clone(self) -> "UnaryNode":
        return UnaryNode(op=self.op, operand=self.operand.clone(), name=self.name)

    def mutate(self, rng: random.Random) -> "UnaryNode":
        return UnaryNode(op=self.op, operand=self.operand.mutate(rng), name=self.name)

    def complexity(self) -> int:
        return 1 + self.operand.complexity()

    def describe(self) -> str:
        return f"{self.name}({self.operand.describe()})"


@dataclass(frozen=True)
class BinaryNode:
    op: Callable[[float, float], float]
    left: FormulaNode
    right: FormulaNode
    name: str

    def evaluate(self, history: PriceHistory) -> float:
        return self.op(self.left.evaluate(history), self.right.evaluate(history))

    def clone(self) -> "BinaryNode":
        return BinaryNode(op=self.op, left=self.left.clone(), right=self.right.clone(), name=self.name)

    def mutate(self, rng: random.Random) -> "BinaryNode":
        if rng.random() < 0.5:
            return BinaryNode(op=self.op, left=self.left.mutate(rng), right=self.right.clone(), name=self.name)
        return BinaryNode(op=self.op, left=self.left.clone(), right=self.right.mutate(rng), name=self.name)

    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()

    def describe(self) -> str:
        return f"{self.name}({self.left.describe()}, {self.right.describe()})"


# Trading formula wrapper ----------------------------------------------------------


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-max(-60, min(60, x))))


@dataclass
class TradingFormula:
    """Represents a tree of indicator/operation nodes producing a trading signal."""

    root: FormulaNode
    priority: int = 0
    evaluation_windows: List[int] = field(default_factory=list)

    def evaluate(self, history: PriceHistory) -> float:
        return self.root.evaluate(history)

    def desired_fraction(self, history: PriceHistory) -> float:
        """Convert raw signal to a long-only allocation fraction between 0 and 1."""

        signal = self.evaluate(history)
        return _sigmoid(signal)

    def mutate(self, rng: random.Random) -> "TradingFormula":
        return TradingFormula(
            root=self.root.mutate(rng),
            priority=self.priority,
            evaluation_windows=list(self.evaluation_windows),
        )

    def complexity(self) -> int:
        return self.root.complexity()

    def describe(self) -> str:
        return self.root.describe()

    def clone(self) -> "TradingFormula":
        return TradingFormula(
            root=self.root.clone(),
            priority=self.priority,
            evaluation_windows=list(self.evaluation_windows),
        )


# Factories -----------------------------------------------------------------------


UNARY_OPERATIONS: Dict[str, Callable[[float], float]] = {
    "tanh": math.tanh,
    "sqrt_abs": lambda x: math.sqrt(abs(x)),
}

BINARY_OPERATIONS: Dict[str, Callable[[float, float], float]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": _safe_divide,
}


INDICATOR_FACTORIES: List[Callable[[random.Random], FormulaNode]] = [
    lambda rng: PriceNode(),
    lambda rng: MovingAverageNode(window=rng.randint(2, 20)),
    lambda rng: MomentumNode(lookback=rng.randint(1, 20)),
    lambda rng: VolatilityNode(window=rng.randint(2, 20)),
]


@dataclass
class FormulaFactory:
    """Factory capable of generating and mutating trading formulas."""

    rng: random.Random = field(default_factory=random.Random)
    max_depth: int = 4
    min_window: int = 30
    max_window: int = 252
    window_count_range: tuple[int, int] = (2, 4)
    data_length: int = 252

    def _random_indicator(self) -> FormulaNode:
        return self.rng.choice(INDICATOR_FACTORIES)(self.rng)

    def _random_unary(self, operand: FormulaNode) -> FormulaNode:
        name, op = self.rng.choice(list(UNARY_OPERATIONS.items()))
        return UnaryNode(op=op, operand=operand, name=name)

    def _random_binary(self, left: FormulaNode, right: FormulaNode) -> FormulaNode:
        name, op = self.rng.choice(list(BINARY_OPERATIONS.items()))
        return BinaryNode(op=op, left=left, right=right, name=name)

    def _build_tree(self, depth: int) -> FormulaNode:
        if depth == 0:
            return self._random_indicator()
        choice = self.rng.random()
        if choice < 0.3:
            return self._random_indicator()
        if choice < 0.55:
            operand = self._build_tree(depth - 1)
            return self._random_unary(operand)
        left = self._build_tree(depth - 1)
        right = self._build_tree(depth - 1)
        return self._random_binary(left, right)

    def _random_windows(self) -> List[int]:
        if self.data_length <= 0:
            return [1]
        min_count, max_count = self.window_count_range
        count = self.rng.randint(min_count, max_count)
        windows: List[int] = []
        min_allowed = max(1, min(self.min_window, self.data_length))
        max_allowed = max(min_allowed, min(self.max_window, self.data_length))
        for _ in range(count):
            length = self.rng.randint(min_allowed, max_allowed)
            windows.append(length)
        return sorted(set(windows)) or [max_allowed]

    def _mutate_windows(self, windows: List[int]) -> List[int]:
        if not windows:
            return self._random_windows()
        new_windows = sorted(set(max(1, min(w, self.data_length)) for w in windows))
        # randomly adjust existing window lengths
        index = self.rng.randrange(len(new_windows))
        delta = self.rng.choice([-30, -15, -5, 5, 15, 30])
        new_value = new_windows[index] + delta
        min_allowed = max(1, min(self.min_window, self.data_length))
        max_allowed = max(min_allowed, min(self.max_window, self.data_length))
        new_value = max(min_allowed, min(new_value, max_allowed))
        new_windows[index] = new_value

        # occasionally add or remove a window to diversify time frames
        action = self.rng.random()
        if action < 0.2 and len(new_windows) > self.window_count_range[0]:
            remove_index = self.rng.randrange(len(new_windows))
            new_windows.pop(remove_index)
        elif action > 0.8 and len(new_windows) < self.window_count_range[1]:
            extra = self.rng.randint(min_allowed, max_allowed)
            new_windows.append(extra)

        return sorted(set(new_windows)) or self._random_windows()

    def create(self, priority: int = 0) -> TradingFormula:
        depth = self.rng.randint(1, self.max_depth)
        root = self._build_tree(depth)
        windows = self._random_windows()
        return TradingFormula(root=root, priority=priority, evaluation_windows=windows)

    def mutate(self, formula: TradingFormula, mutate_priority: bool = True) -> TradingFormula:
        mutated_root = formula.root.mutate(self.rng)
        priority = formula.priority
        if mutate_priority and self.rng.random() < 0.3:
            delta = self.rng.choice([-1, 1])
            priority = formula.priority + delta
        windows = self._mutate_windows(list(formula.evaluation_windows))
        return TradingFormula(root=mutated_root, priority=priority, evaluation_windows=windows)
