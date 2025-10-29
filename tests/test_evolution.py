"""Unit tests for the evolutionary engine."""

from __future__ import annotations

import random

from stck.data import HistoricalData
from stck.evolution import (
    EvolutionConfig,
    EvolutionEngine,
    MemberPerformance,
    PortfolioMember,
)


def _build_mock_data() -> HistoricalData:
    prices: dict[str, list[float]] = {}
    base_symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
    for idx, symbol in enumerate(base_symbols):
        prices[symbol] = [100 + idx + day for day in range(60)]
    prices["SPY"] = [120 + day * 0.5 for day in range(60)]
    return HistoricalData(prices)


def _default_config() -> EvolutionConfig:
    return EvolutionConfig(
        population_size=4,
        min_tickers=5,
        initial_ticker_count=5,
        initial_etf_count=1,
    )


def test_evolution_produces_reports() -> None:
    data = _build_mock_data()
    engine = EvolutionEngine(data=data, config=_default_config(), rng=random.Random(7))
    tickers = list(data.tickers)
    population = engine.initialize_population(tickers)

    assert len(population) == engine.config.population_size
    for member in population:
        assert isinstance(member, PortfolioMember)
        assert member.ticker_count() >= engine.config.min_tickers

    evolved, report = engine.evolve(population, generation=0)
    assert len(evolved) == engine.config.population_size
    assert report.metrics.top_percent_gain >= report.metrics.top10_mean_percent
    assert report.metrics.average_percent_gain <= report.metrics.top_percent_gain
    assert report.best_member is not None
    assert report.best_member.member.ticker_count() >= engine.config.min_tickers
    assert report.transition is not None
    assert 0 <= report.transition.survivor_count <= engine.config.population_size
    assert report.transition.bottom_culled_count >= 0


def test_mutation_preserves_minimum_tickers() -> None:
    data = _build_mock_data()
    engine = EvolutionEngine(data=data, config=_default_config(), rng=random.Random(3))
    population = engine.initialize_population(list(data.tickers))
    parent = population[0]
    child = engine._mutate_member(parent)  # type: ignore[attr-defined]
    assert child.ticker_count() >= engine.config.min_tickers
    assert len(set(child.tickers())) == len(child.tickers())


def test_repopulate_keeps_survivors() -> None:
    data = _build_mock_data()
    engine = EvolutionEngine(data=data, config=_default_config(), rng=random.Random(11))
    population = engine.initialize_population(list(data.tickers))
    performances = [engine._evaluate_member(member) for member in population]  # type: ignore[attr-defined]
    performances.sort(key=lambda p: p.percent_gain, reverse=True)
    selection = engine._select_survivors(performances)  # type: ignore[attr-defined]
    survivors = selection.survivors
    survivor_snapshots = [member.clone() for member in survivors]

    original_mutate = engine._mutate_member  # type: ignore[attr-defined]
    mutation_parents: list[PortfolioMember] = []

    def tracking_mutation(parent: PortfolioMember) -> PortfolioMember:
        mutation_parents.append(parent)
        return original_mutate(parent)  # type: ignore[misc]

    engine._mutate_member = tracking_mutation  # type: ignore[assignment]

    new_population, clones_created = engine._repopulate(  # type: ignore[attr-defined]
        performances, survivors
    )

    assert len(new_population) == engine.config.population_size
    assert clones_created >= len(survivors)
    assert len(mutation_parents) == clones_created
    for survivor, snapshot in zip(survivors, survivor_snapshots):
        assert survivor in new_population
        assert survivor.describe_formulas() == snapshot.describe_formulas()
    assert all(parent in survivors for parent in mutation_parents)
    clone_count = sum(1 for member in new_population if member not in survivors)
    assert clone_count + len(survivors) == engine.config.population_size


def test_selection_culls_half_even_and_odd_populations() -> None:
    data = _build_mock_data()
    engine = EvolutionEngine(data=data, config=_default_config(), rng=random.Random(13))

    even_performances = [
        MemberPerformance(
            member=PortfolioMember(),
            final_equity=engine.config.initial_cash * (1 + gain / 100.0),
            percent_gain=gain,
            max_drawdown=0.0,
        )
        for gain in [60, 55, 50, 45, 40, 35]
    ]
    even_outcome = engine._select_survivors(even_performances)  # type: ignore[attr-defined]
    assert len(even_outcome.survivors) == len(even_performances) // 2
    assert all(
        survivor is perf.member
        for survivor, perf in zip(even_outcome.survivors, even_performances)
    )
    assert even_outcome.bottom_culled_count >= 1

    odd_performances = [
        MemberPerformance(
            member=PortfolioMember(),
            final_equity=engine.config.initial_cash * (1 + gain / 100.0),
            percent_gain=gain,
            max_drawdown=0.0,
        )
        for gain in [70, 60, 50, 40, 30]
    ]
    odd_outcome = engine._select_survivors(odd_performances)  # type: ignore[attr-defined]
    assert len(odd_outcome.survivors) == len(odd_performances) // 2
    assert all(
        survivor is perf.member
        for survivor, perf in zip(odd_outcome.survivors, odd_performances)
    )
    assert odd_outcome.bottom_culled_count >= 1


def test_best_equity_curve_never_regresses() -> None:
    data = _build_mock_data()
    engine = EvolutionEngine(data=data, config=_default_config(), rng=random.Random(19))
    population = engine.initialize_population(list(data.tickers))

    best_equities: list[float] = []
    generations = 4
    for generation in range(generations):
        population, report = engine.evolve(population, generation=generation)
        assert report.best_member is not None
        best_equities.append(report.best_member.final_equity)

    assert best_equities == sorted(best_equities)
