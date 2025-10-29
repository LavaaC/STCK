"""Evolution engine for multi-asset trading formulas."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .data import HistoricalData
from .formulas import FormulaFactory, TradingFormula
from .portfolio import BacktestResult, PortfolioBacktester, TickerAllocation
from .tickers import POPULAR_ETFS


@dataclass
class EvolutionConfig:
    population_size: int = 50
    generations: int = 0
    initial_cash: float = 100_000.0
    window_count_range: tuple[int, int] = (2, 4)
    min_window: int = 30
    max_window: int = 252
    top_survivor_fraction: float = 0.1
    bottom_death_fraction: float = 0.1
    min_tickers: int = 10
    initial_ticker_count: int = 13
    initial_etf_count: int = 2
    formula_mutation_chance: float = 0.75
    formula_mutation_attempts: int = 2
    weight_mutation_chance: float = 0.6
    asset_mutation_chance: float = 0.65
    backtest_years: int = 5
    trading_days_per_year: int = 252
    target_ticker_low: int = 12
    target_ticker_high: int = 13
    max_formula_complexity: int = 28
    complexity_target: int = 18
    complexity_penalty_per_node: float = 100.0


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
        return format_member_portfolio(self)


def format_member_portfolio(member: PortfolioMember) -> List[str]:
    """Return human-readable strings describing a member's holdings."""

    lines: List[str] = []
    for asset in member.assets:
        lines.append(
            f"{asset.ticker} (weight {asset.weight:.2f}) -> {asset.formula.describe()}"
        )
    return lines


@dataclass
class MemberPerformance:
    member: PortfolioMember
    final_cash: float
    cash_percent_gain: float
    final_equity: float
    backtest: BacktestResult
    complexity: int
    complexity_penalty: float
    score: float


@dataclass
class PopulationMetrics:
    generation: int
    top_final_cash: float
    top10_mean_final_cash: float
    top20_mean_final_cash: float
    average_final_cash: float
    top_cash_percent_gain: float
    average_cash_percent_gain: float


@dataclass
class GenerationReport:
    generation: int
    performances: List[MemberPerformance]
    metrics: PopulationMetrics
    best_member: MemberPerformance | None
    transition: "PopulationTransitionSummary" | None = None


@dataclass
class PopulationTransitionSummary:
    """Details about how the population changed between generations."""

    top_preserved_count: int
    bottom_culled_count: int
    middle_strategy: str
    survivor_count: int
    clones_created: int


@dataclass
class _SelectionOutcome:
    survivors: List[PortfolioMember]
    top_preserved_count: int
    bottom_culled_count: int
    middle_strategy: str


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
        self._backtest_length = min(
            len(self.data),
            max(1, self.config.backtest_years * self.config.trading_days_per_year),
        )
        self._backtest_start_index = max(0, len(self.data) - self._backtest_length)
        self._etf_universe = set(POPULAR_ETFS)
        self.available_tickers: List[str] = []
        self.available_etfs: List[str] = []
        self._elite_member: PortfolioMember | None = None
        self._elite_signature: Tuple[Tuple[str, float, bool, str], ...] | None = None

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
            performances, key=lambda p: (p.score, p.final_cash), reverse=True
        )
        metrics = self._compute_population_metrics(performances, generation)
        selection = self._select_survivors(sorted_performances)
        new_population, clones_created = self._repopulate(
            sorted_performances, selection.survivors
        )
        best_member = sorted_performances[0] if sorted_performances else None
        if best_member is not None:
            self._elite_member = best_member.member.clone()
            self._elite_signature = self._member_signature(best_member.member)
        transition = PopulationTransitionSummary(
            top_preserved_count=selection.top_preserved_count,
            bottom_culled_count=selection.bottom_culled_count,
            middle_strategy=selection.middle_strategy,
            survivor_count=len(selection.survivors),
            clones_created=clones_created,
        )
        report = GenerationReport(
            generation=generation,
            performances=sorted_performances,
            metrics=metrics,
            best_member=best_member,
            transition=transition,
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
        formula = self.factory.clamp_complexity(
            formula, self.config.max_formula_complexity
        )
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
        result = backtester.run(start_index=self._backtest_start_index)
        final_equity = result.final_equity
        final_cash = result.final_cash
        cash_gain = final_cash - self.config.initial_cash
        percent_gain = (cash_gain / self.config.initial_cash) * 100.0
        total_complexity = sum(asset.formula.complexity() for asset in member.assets)
        complexity_penalty = self._complexity_penalty(total_complexity)
        score = final_cash - complexity_penalty
        return MemberPerformance(
            member=member,
            final_cash=final_cash,
            cash_percent_gain=percent_gain,
            final_equity=final_equity,
            backtest=result,
            complexity=total_complexity,
            complexity_penalty=complexity_penalty,
            score=score,
        )

    def _compute_population_metrics(
        self, performances: List[MemberPerformance], generation: int
    ) -> PopulationMetrics:
        if not performances:
            return PopulationMetrics(
                generation=generation,
                top_final_cash=0.0,
                top10_mean_final_cash=0.0,
                top20_mean_final_cash=0.0,
                average_final_cash=0.0,
                top_cash_percent_gain=0.0,
                average_cash_percent_gain=0.0,
            )

        sorted_by_cash = sorted(performances, key=lambda p: p.final_cash, reverse=True)

        def mean_cash_for_fraction(fraction: float) -> float:
            count = max(1, int(round(len(sorted_by_cash) * fraction)))
            count = min(count, len(sorted_by_cash))
            selected = sorted_by_cash[:count]
            return sum(p.final_cash for p in selected) / len(selected)

        average_cash = sum(p.final_cash for p in performances) / len(performances)
        average_percent_gain = sum(p.cash_percent_gain for p in performances) / len(performances)
        return PopulationMetrics(
            generation=generation,
            top_final_cash=sorted_by_cash[0].final_cash,
            top10_mean_final_cash=mean_cash_for_fraction(0.1),
            top20_mean_final_cash=mean_cash_for_fraction(0.2),
            average_final_cash=average_cash,
            top_cash_percent_gain=sorted_by_cash[0].cash_percent_gain,
            average_cash_percent_gain=average_percent_gain,
        )

    def _complexity_penalty(self, complexity: int) -> float:
        if self.config.complexity_penalty_per_node <= 0:
            return 0.0
        excess = max(0, complexity - self.config.complexity_target)
        return excess * self.config.complexity_penalty_per_node

    def _select_survivors(self, performances: List[MemberPerformance]) -> _SelectionOutcome:
        count = len(performances)
        if count == 0:
            return _SelectionOutcome(
                survivors=[],
                top_preserved_count=0,
                bottom_culled_count=0,
                middle_strategy="No members available; repopulating from scratch.",
            )

        top_target = max(1, math.ceil(count * self.config.top_survivor_fraction))
        bottom_target = max(1, math.ceil(count * self.config.bottom_death_fraction))
        kill_target = max(bottom_target, math.ceil(count * 0.5))

        if top_target + bottom_target > count:
            bottom_target = max(0, count - top_target)

        kill_target = min(count, max(kill_target, bottom_target))

        top_indices = list(range(min(top_target, count)))
        bottom_indices = list(range(max(0, count - bottom_target), count))

        middle_start = len(top_indices)
        middle_end = max(middle_start, count - len(bottom_indices))
        middle_indices = list(range(middle_start, middle_end))

        kills_needed = max(0, kill_target - len(bottom_indices))
        kills_from_middle: List[int] = []

        if middle_indices and kills_needed > 0:
            weighted_pool = [
                (idx, (idx - middle_start + 1)) for idx in middle_indices
            ]
            available = list(weighted_pool)
            while available and len(kills_from_middle) < kills_needed:
                total_weight = sum(weight for _, weight in available)
                pick = self.rng.uniform(0, total_weight)
                cumulative = 0.0
                chosen_index = None
                for i, (candidate, weight) in enumerate(available):
                    cumulative += weight
                    if pick <= cumulative:
                        chosen_index = i
                        break
                if chosen_index is None:
                    chosen_index = len(available) - 1
                candidate, _ = available.pop(chosen_index)
                kills_from_middle.append(candidate)

        kill_set = set(bottom_indices + kills_from_middle)
        survivors: List[PortfolioMember] = [
            performances[i].member for i in range(count) if i not in kill_set
        ]

        middle_strategy = (
            "Protected top {protected} performers, culled {bottom} from the bottom tier, "
            "and removed {middle} additional members from the middle with weighted odds."
        ).format(
            protected=len(top_indices),
            bottom=len(bottom_indices),
            middle=len(kills_from_middle),
        )

        return _SelectionOutcome(
            survivors=survivors,
            top_preserved_count=min(len(top_indices), len(survivors)),
            bottom_culled_count=len(bottom_indices),
            middle_strategy=middle_strategy,
        )

    def _repopulate(
        self,
        performances: List[MemberPerformance],
        survivors: List[PortfolioMember],
    ) -> tuple[List[PortfolioMember], int]:
        new_population: List[PortfolioMember] = list(survivors)
        clones_created = 0

        if not performances and not survivors:
            while len(new_population) < self.config.population_size:
                new_population.append(self._create_initial_member())
            return new_population, clones_created

        parents = list(survivors)
        if not parents:
            top_seed = max(1, min(len(performances), self.config.population_size))
            parents = [performances[i].member for i in range(top_seed)]

        initial_clones: List[PortfolioMember] = []
        for survivor in parents:
            child = self._mutate_member(survivor)
            initial_clones.append(child)
        clones_created += len(initial_clones)
        new_population.extend(initial_clones)

        if (
            self._elite_member is not None
            and self._elite_signature is not None
            and len(new_population) < self.config.population_size
        ):
            elite_signature = self._elite_signature
            already_present = any(
                self._member_signature(member) == elite_signature for member in new_population
            )
            if not already_present:
                new_population.insert(0, self._elite_member.clone())

        while len(new_population) < self.config.population_size:
            if parents:
                parent = self.rng.choice(parents)
                child = self._mutate_member(parent)
                new_population.append(child)
                clones_created += 1
            else:
                new_population.append(self._create_initial_member())

        return new_population[: self.config.population_size], clones_created

    def _member_signature(self, member: PortfolioMember) -> Tuple[Tuple[str, float, bool, str], ...]:
        return tuple(
            (asset.ticker, asset.weight, asset.is_etf, asset.formula.describe())
            for asset in member.assets
        )

    def _mutate_formula(self, formula: TradingFormula) -> TradingFormula:
        mutated = self.factory.mutate(formula)
        return self.factory.clamp_complexity(mutated, self.config.max_formula_complexity)

    def _mutate_member(self, parent: PortfolioMember) -> PortfolioMember:
        child = parent.clone()
        if not child.assets:
            return self._create_initial_member()

        mutated = False

        for _ in range(self.config.formula_mutation_attempts):
            if child.assets and self.rng.random() < self.config.formula_mutation_chance:
                asset = self.rng.choice(child.assets)
                asset.formula = self._mutate_formula(asset.formula)
                mutated = True

        if child.assets and self.rng.random() < self.config.weight_mutation_chance:
            asset = self.rng.choice(child.assets)
            scale = self.rng.uniform(0.8, 1.25)
            asset.weight = max(0.05, round(asset.weight * scale, 3))
            mutated = True

        if self.rng.random() < self.config.asset_mutation_chance:
            change_performed = False
            ticker_count = child.ticker_count()
            if ticker_count < self.config.target_ticker_low:
                add_bias = 0.9
            elif ticker_count > self.config.target_ticker_high:
                add_bias = 0.25
            else:
                add_bias = 0.5
            add_first = not child.assets or self.rng.random() < add_bias
            if add_first:
                change_performed = self._add_random_asset(child)
                if (
                    change_performed
                    and child.ticker_count() < self.config.target_ticker_low
                    and self.rng.random() < 0.35
                ):
                    # Occasionally add a second asset to increase diversification.
                    change_performed = (
                        self._add_random_asset(child) or change_performed
                    )
                if not change_performed:
                    change_performed = self._remove_random_asset(child)
            else:
                change_performed = self._remove_random_asset(child)
                if not change_performed:
                    change_performed = self._add_random_asset(child)
            mutated = mutated or change_performed

        if not mutated and child.assets:
            asset = self.rng.choice(child.assets)
            asset.formula = self._mutate_formula(asset.formula)

        self._ensure_min_requirements(child)
        return child

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

