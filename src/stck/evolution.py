"""Evolution engine for trading formulas."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from .presets import get_preset_formula

from .data import HistoricalData
from .formulas import FormulaFactory, TradingFormula
from .portfolio import PortfolioBacktester, TickerAllocation


@dataclass
class EvolutionConfig:
    population_size: int = 12
    generations: int = 0
    initial_cash: float = 100_000.0
    window_count_range: tuple[int, int] = (2, 4)
    min_window: int = 30
    max_window: int = 252
    top_survivor_fraction: float = 0.1
    bottom_death_fraction: float = 0.1


@dataclass
class WindowPerformance:
    window: int
    final_equity: float
    max_drawdown: float


@dataclass
class FormulaPerformance:
    formula: TradingFormula
    ticker: str
    windows: List[WindowPerformance]

    @property
    def average_final_equity(self) -> float:
        return sum(w.final_equity for w in self.windows) / len(self.windows)

    @property
    def average_max_drawdown(self) -> float:
        return sum(w.max_drawdown for w in self.windows) / len(self.windows)

    @property
    def score(self) -> float:
        penalty = 1.0 + self.average_max_drawdown
        return self.average_final_equity / penalty


@dataclass
class PopulationMetrics:
    generation: int
    top_equity: float
    top10_mean: float
    top20_mean: float
    average_equity: float


@dataclass
class GenerationReport:
    generation: int
    ticker_performances: Dict[str, List[FormulaPerformance]]
    metrics: PopulationMetrics
    best_performance: FormulaPerformance


class EvolutionEngine:
    """Manage populations of formulas per ticker and evolve them."""

    def __init__(self, data: HistoricalData, config: EvolutionConfig | None = None, rng: random.Random | None = None) -> None:
        self.data = data
        self.config = config or EvolutionConfig()
        self.rng = rng or random.Random()
        self.factory = FormulaFactory(
            rng=self.rng,
            min_window=self.config.min_window,
            max_window=self.config.max_window,
            window_count_range=self.config.window_count_range,
            data_length=len(self.data),
        )

    def initialize_population(self, tickers: List[str]) -> Dict[str, List[TradingFormula]]:
        population: Dict[str, List[TradingFormula]] = {}
        for ticker in tickers:
            formulas: List[TradingFormula] = []
            preset = get_preset_formula(ticker)
            if preset is not None:
                formulas.append(preset.clone())
            while len(formulas) < self.config.population_size:
                if formulas:
                    parent = self.rng.choice(formulas)
                    formulas.append(self.factory.mutate(parent))
                else:
                    formulas.append(self.factory.create(priority=0))
            population[ticker] = formulas
        return population

    def evolve(
        self, population: Dict[str, List[TradingFormula]], generation: int = 0
    ) -> tuple[Dict[str, List[TradingFormula]], GenerationReport]:
        ticker_performances: Dict[str, List[FormulaPerformance]] = {}
        new_population: Dict[str, List[TradingFormula]] = {}
        all_performances: List[FormulaPerformance] = []

        for ticker, formulas in population.items():
            performances = self._evaluate_ticker(ticker, formulas)
            ticker_performances[ticker] = performances
            all_performances.extend(performances)
            survivors = self._select_survivors(performances)
            new_population[ticker] = self._repopulate(survivors)

        metrics = self._compute_population_metrics(all_performances, generation)
        best_performance = max(all_performances, key=lambda p: p.average_final_equity)
        report = GenerationReport(
            generation=generation,
            ticker_performances=ticker_performances,
            metrics=metrics,
            best_performance=best_performance,
        )
        return new_population, report

    # Internal helpers -----------------------------------------------------------------

    def _evaluate_ticker(self, ticker: str, formulas: List[TradingFormula]) -> List[FormulaPerformance]:
        performances: List[FormulaPerformance] = []
        for formula in formulas:
            windows = formula.evaluation_windows or [len(self.data)]
            window_results: List[WindowPerformance] = []
            for window in windows:
                window_data = self.data.tail(window)
                allocation = TickerAllocation(ticker=ticker, formula=formula, priority=formula.priority)
                backtester = PortfolioBacktester(
                    data=window_data,
                    allocations=[allocation],
                    initial_cash=self.config.initial_cash,
                )
                result = backtester.run()
                window_results.append(
                    WindowPerformance(
                        window=len(window_data),
                        final_equity=result.final_equity,
                        max_drawdown=result.max_drawdown(),
                    )
                )
            performances.append(FormulaPerformance(formula=formula, ticker=ticker, windows=window_results))
        return performances

    def _compute_population_metrics(
        self, performances: List[FormulaPerformance], generation: int
    ) -> PopulationMetrics:
        if not performances:
            return PopulationMetrics(
                generation=generation,
                top_equity=0.0,
                top10_mean=0.0,
                top20_mean=0.0,
                average_equity=0.0,
            )

        sorted_by_equity = sorted(performances, key=lambda p: p.average_final_equity, reverse=True)

        def mean_for_fraction(fraction: float) -> float:
            count = max(1, int(round(len(sorted_by_equity) * fraction)))
            count = min(count, len(sorted_by_equity))
            selected = sorted_by_equity[:count]
            return sum(p.average_final_equity for p in selected) / len(selected)

        average_equity = sum(p.average_final_equity for p in sorted_by_equity) / len(sorted_by_equity)
        return PopulationMetrics(
            generation=generation,
            top_equity=sorted_by_equity[0].average_final_equity,
            top10_mean=mean_for_fraction(0.1),
            top20_mean=mean_for_fraction(0.2),
            average_equity=average_equity,
        )

    def _select_survivors(self, performances: List[FormulaPerformance]) -> List[TradingFormula]:
        performances.sort(key=lambda p: p.score, reverse=True)
        count = len(performances)
        if count == 0:
            return []

        top_count = max(1, int(round(count * self.config.top_survivor_fraction)))
        bottom_count = max(1, int(round(count * self.config.bottom_death_fraction)))
        if top_count + bottom_count > count:
            bottom_count = max(0, count - top_count)

        survivors = [perf.formula.clone() for perf in performances[:top_count]]
        middle = performances[top_count : count - bottom_count]

        if middle:
            middle_count = len(middle)
            added_from_middle = False
            for idx, perf in enumerate(middle):
                probability = 1.0 - (idx / max(1, middle_count))
                if self.rng.random() < probability:
                    survivors.append(perf.formula.clone())
                    added_from_middle = True
            if not added_from_middle:
                survivors.append(middle[0].formula.clone())

        return survivors

    def _repopulate(self, survivors: List[TradingFormula]) -> List[TradingFormula]:
        population: List[TradingFormula] = []
        while len(population) < self.config.population_size:
            if survivors:
                parent = self.rng.choice(survivors)
                population.append(self.factory.mutate(parent))
            else:
                population.append(self.factory.create())
        return population
