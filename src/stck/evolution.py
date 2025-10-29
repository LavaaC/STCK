"""Evolution engine for multi-asset trading formulas."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence

from .data import HistoricalData
from .formulas import FormulaFactory, TradingFormula
from .portfolio import PortfolioBacktester, TickerAllocation
from .tickers import POPULAR_ETFS


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
    min_tickers: int = 5
    initial_ticker_count: int = 7
    initial_etf_count: int = 1


@dataclass
class MemberAsset:
    ticker: str
    formula: TradingFormula
    weight: float
    is_etf: bool = False

    def clone(self) -> "MemberAsset":
        return MemberAsset(
            ticker=self.ticker,
            formula=self.formula.clone(),
            weight=self.weight,
            is_etf=self.is_etf,
        )


@dataclass
class PortfolioMember:
    assets: List[MemberAsset] = field(default_factory=list)

    def clone(self) -> "PortfolioMember":
        return PortfolioMember(assets=[asset.clone() for asset in self.assets])

    def tickers(self) -> List[str]:
        return [asset.ticker for asset in self.assets]

    def ticker_count(self) -> int:
        return sum(1 for asset in self.assets if not asset.is_etf)

    def etf_count(self) -> int:
        return sum(1 for asset in self.assets if asset.is_etf)

    def allocations(self) -> List[TickerAllocation]:
        return [
            TickerAllocation(
                ticker=asset.ticker,
                formula=asset.formula,
                priority=0,
                weight=asset.weight,
            )
            for asset in self.assets
        ]

    def describe_formulas(self) -> List[str]:
        lines = []
        for asset in self.assets:
            lines.append(
                f"{asset.ticker} (weight {asset.weight:.2f}) -> {asset.formula.describe()}"
            )
        return lines


@dataclass
class MemberPerformance:
    member: PortfolioMember
    final_equity: float
    percent_gain: float
    max_drawdown: float


@dataclass
class PopulationMetrics:
    generation: int
    top_percent_gain: float
    top10_mean_percent: float
    top20_mean_percent: float
    average_percent_gain: float


@dataclass
class GenerationReport:
    generation: int
    performances: List[MemberPerformance]
    metrics: PopulationMetrics
    best_member: MemberPerformance | None


class EvolutionEngine:
    """Manage populations of multi-asset trading formulas and evolve them."""

    def __init__(
        self,
        data: HistoricalData,
        config: EvolutionConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
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
        self._etf_universe = set(POPULAR_ETFS)
        self.available_tickers: List[str] = []
        self.available_etfs: List[str] = []

    # Population management ---------------------------------------------------------

    def initialize_population(self, tickers: List[str]) -> List[PortfolioMember]:
        self.available_tickers = [t for t in tickers if t not in self._etf_universe]
        self.available_etfs = [t for t in tickers if t in self._etf_universe]

        if len(self.available_tickers) < self.config.min_tickers:
            raise ValueError(
                "Not enough equity tickers available to satisfy minimum ticker requirement"
            )

        population: List[PortfolioMember] = []
        for _ in range(self.config.population_size):
            population.append(self._create_initial_member())
        return population

    def evolve(
        self, population: List[PortfolioMember], generation: int = 0
    ) -> tuple[List[PortfolioMember], GenerationReport]:
        performances = [self._evaluate_member(member) for member in population]
        sorted_performances = sorted(
            performances, key=lambda performance: performance.percent_gain, reverse=True
        )
        metrics = self._compute_population_metrics(sorted_performances, generation)
        survivors = self._select_survivors(sorted_performances)
        new_population = self._repopulate(sorted_performances, survivors)
        best_member = sorted_performances[0] if sorted_performances else None
        report = GenerationReport(
            generation=generation,
            performances=sorted_performances,
            metrics=metrics,
            best_member=best_member,
        )
        return new_population, report

    # Internal helpers -------------------------------------------------------------

    def _create_initial_member(self) -> PortfolioMember:
        member = PortfolioMember()
        ticker_choices = self._random_selection(
            self.available_tickers, self.config.initial_ticker_count
        )
        etf_choices = self._random_selection(
            self.available_etfs, self.config.initial_etf_count
        )

        for ticker in ticker_choices:
            member.assets.append(self._create_asset(ticker, is_etf=False))
        for ticker in etf_choices:
            member.assets.append(self._create_asset(ticker, is_etf=True))

        self._ensure_min_requirements(member)
        return member

    def _create_asset(self, ticker: str, *, is_etf: bool) -> MemberAsset:
        formula = self.factory.create(priority=0)
        weight = self._random_weight()
        return MemberAsset(ticker=ticker, formula=formula, weight=weight, is_etf=is_etf)

    def _random_weight(self) -> float:
        return round(self.rng.uniform(0.05, 1.0), 3)

    def _random_selection(self, universe: Sequence[str], count: int) -> List[str]:
        if not universe:
            return []
        count = max(0, min(count, len(universe)))
        return self.rng.sample(universe, count)

    def _evaluate_member(self, member: PortfolioMember) -> MemberPerformance:
        subset_prices = {ticker: self.data.prices[ticker] for ticker in member.tickers()}
        subset_data = HistoricalData(subset_prices)
        allocations = member.allocations()
        backtester = PortfolioBacktester(
            data=subset_data, allocations=allocations, initial_cash=self.config.initial_cash
        )
        result = backtester.run()
        final_equity = result.final_equity
        percent_gain = (final_equity - self.config.initial_cash) / self.config.initial_cash * 100.0
        return MemberPerformance(
            member=member,
            final_equity=final_equity,
            percent_gain=percent_gain,
            max_drawdown=result.max_drawdown(),
        )

    def _compute_population_metrics(
        self, performances: List[MemberPerformance], generation: int
    ) -> PopulationMetrics:
        if not performances:
            return PopulationMetrics(
                generation=generation,
                top_percent_gain=0.0,
                top10_mean_percent=0.0,
                top20_mean_percent=0.0,
                average_percent_gain=0.0,
            )

        def mean_for_fraction(fraction: float) -> float:
            count = max(1, int(round(len(performances) * fraction)))
            count = min(count, len(performances))
            selected = performances[:count]
            return sum(p.percent_gain for p in selected) / len(selected)

        average_percent = sum(p.percent_gain for p in performances) / len(performances)
        return PopulationMetrics(
            generation=generation,
            top_percent_gain=performances[0].percent_gain,
            top10_mean_percent=mean_for_fraction(0.1),
            top20_mean_percent=mean_for_fraction(0.2),
            average_percent_gain=average_percent,
        )

    def _select_survivors(self, performances: List[MemberPerformance]) -> List[PortfolioMember]:
        count = len(performances)
        if count == 0:
            return []

        top_count = max(1, math.ceil(count * self.config.top_survivor_fraction))
        bottom_count = max(1, math.ceil(count * self.config.bottom_death_fraction))

        if top_count + bottom_count > count:
            bottom_count = max(0, count - top_count)

        target_removals = max(bottom_count, math.ceil(count * 0.5))
        target_removals = min(target_removals, count - top_count)
        survivors_target = count - target_removals

        top_survivors = performances[:top_count]
        middle_end = count - bottom_count
        middle_segment = performances[top_count:middle_end]
        additional_removals = max(0, target_removals - bottom_count)
        middle_survivor_count = max(0, len(middle_segment) - additional_removals)
        middle_survivors = middle_segment[:middle_survivor_count]

        selected = top_survivors + middle_survivors
        if len(selected) > survivors_target:
            selected = selected[:survivors_target]
        return [perf.member for perf in selected]

    def _repopulate(
        self,
        performances: List[MemberPerformance],
        survivors: List[PortfolioMember],
    ) -> List[PortfolioMember]:
        new_population: List[PortfolioMember] = []

        if not performances or not survivors:
            while len(new_population) < self.config.population_size:
                new_population.append(self._create_initial_member())
            return new_population

        for survivor in survivors:
            if len(new_population) >= self.config.population_size:
                break
            new_population.append(survivor)
            if len(new_population) >= self.config.population_size:
                break
            clone = survivor.clone()
            mutated_clone = self._mutate_clone(clone)
            new_population.append(mutated_clone)

        while len(new_population) < self.config.population_size:
            parent = self.rng.choice(survivors)
            clone = parent.clone()
            mutated_clone = self._mutate_clone(clone)
            new_population.append(mutated_clone)

        return new_population[: self.config.population_size]

    def _mutate_member(self, parent: PortfolioMember) -> PortfolioMember:
        child = parent.clone()
        return self._mutate_clone(child)

    def _mutate_clone(self, clone: PortfolioMember) -> PortfolioMember:
        if not clone.assets:
            return self._create_initial_member()

        action = self.rng.random()
        if action < 0.4:
            asset = self.rng.choice(clone.assets)
            asset.formula = self.factory.mutate(asset.formula)
        elif action < 0.7:
            asset = self.rng.choice(clone.assets)
            scale = self.rng.uniform(0.8, 1.2)
            asset.weight = max(0.05, round(asset.weight * scale, 3))
        elif action < 0.85:
            self._add_random_asset(clone)
        else:
            self._remove_random_asset(clone)

        self._ensure_min_requirements(clone)
        return clone

    def _add_random_asset(self, member: PortfolioMember) -> bool:
        existing = set(member.tickers())
        available_tickers = [t for t in self.available_tickers if t not in existing]
        available_etfs = [t for t in self.available_etfs if t not in existing]

        choices: List[tuple[str, bool]] = []
        for ticker in available_tickers:
            choices.append((ticker, False))
        for ticker in available_etfs:
            choices.append((ticker, True))

        if not choices:
            return False

        ticker, is_etf = self.rng.choice(choices)
        member.assets.append(self._create_asset(ticker, is_etf=is_etf))
        return True

    def _remove_random_asset(self, member: PortfolioMember) -> bool:
        if not member.assets:
            return False

        ticker_assets = [asset for asset in member.assets if not asset.is_etf]
        removable: List[MemberAsset] = []
        if len(ticker_assets) > self.config.min_tickers:
            removable = list(member.assets)
        else:
            removable = [asset for asset in member.assets if asset.is_etf]

        if not removable:
            return False

        asset = self.rng.choice(removable)
        member.assets.remove(asset)
        return True

    def _ensure_min_requirements(self, member: PortfolioMember) -> None:
        existing = set(member.tickers())
        ticker_pool = [t for t in self.available_tickers if t not in existing]
        while member.ticker_count() < self.config.min_tickers and ticker_pool:
            ticker = ticker_pool.pop(self.rng.randrange(len(ticker_pool)))
            member.assets.append(self._create_asset(ticker, is_etf=False))
            existing.add(ticker)

        if not member.assets:
            fallback = self._random_selection(self.available_tickers, 1)
            for ticker in fallback:
                member.assets.append(self._create_asset(ticker, is_etf=False))

