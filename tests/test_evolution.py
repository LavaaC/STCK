"""Unit tests for the evolutionary engine."""

from __future__ import annotations

import random

from unittest.mock import patch

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


def test_mutation_preserves_minimum_tickers() -> None:
    data = _build_mock_data()
    engine = EvolutionEngine(data=data, config=_default_config(), rng=random.Random(3))
    population = engine.initialize_population(list(data.tickers))
    parent = population[0]
    child = engine._mutate_member(parent)  # type: ignore[attr-defined]
    assert child.ticker_count() >= engine.config.min_tickers
    assert len(set(child.tickers())) == len(child.tickers())


def test_survivor_selection_keeps_top_half() -> None:
    data = _build_mock_data()
    config = EvolutionConfig(
        population_size=10,
        min_tickers=5,
        initial_ticker_count=5,
        initial_etf_count=1,
    )
    engine = EvolutionEngine(data=data, config=config, rng=random.Random(5))
    population = engine.initialize_population(list(data.tickers))

    performances = []
    for idx, member in enumerate(population):
        gain = float(idx)
        performances.append(
            MemberPerformance(
                member=member,
                final_equity=engine.config.initial_cash * (1 + gain / 100.0),
                percent_gain=gain,
                max_drawdown=0.0,
            )
        )

    performances.sort(key=lambda perf: perf.percent_gain, reverse=True)
    survivors = engine._select_survivors(performances)  # type: ignore[attr-defined]

    assert len(survivors) == 5
    top_member = performances[0].member
    bottom_member = performances[-1].member
    assert top_member in survivors
    assert bottom_member not in survivors
    survivor_gains = [perf.percent_gain for perf in performances if perf.member in survivors]
    assert survivor_gains == [9.0, 8.0, 7.0, 6.0, 5.0]


def test_survivor_selection_removes_half_for_odd_population() -> None:
    data = _build_mock_data()
    config = EvolutionConfig(
        population_size=9,
        min_tickers=5,
        initial_ticker_count=5,
        initial_etf_count=1,
    )
    engine = EvolutionEngine(data=data, config=config, rng=random.Random(13))
    population = engine.initialize_population(list(data.tickers))

    performances = []
    for idx, member in enumerate(population):
        gain = float(idx)
        performances.append(
            MemberPerformance(
                member=member,
                final_equity=engine.config.initial_cash * (1 + gain / 100.0),
                percent_gain=gain,
                max_drawdown=0.0,
            )
        )

    performances.sort(key=lambda perf: perf.percent_gain, reverse=True)
    survivors = engine._select_survivors(performances)  # type: ignore[attr-defined]

    assert len(survivors) == 4
    survivor_gains = [perf.percent_gain for perf in performances if perf.member in survivors]
    assert survivor_gains == [8.0, 7.0, 6.0, 5.0]


def test_repopulate_preserves_survivors_and_mutates_clones() -> None:
    data = _build_mock_data()
    engine = EvolutionEngine(data=data, config=_default_config(), rng=random.Random(11))
    population = engine.initialize_population(list(data.tickers))
    performances = [engine._evaluate_member(member) for member in population]  # type: ignore[attr-defined]
    performances.sort(key=lambda p: p.percent_gain, reverse=True)
    survivors = engine._select_survivors(performances)  # type: ignore[attr-defined]
    survivor_snapshots = {id(member): member.clone() for member in survivors}

    with patch.object(engine, "_mutate_clone", wraps=engine._mutate_clone) as mutate_mock:
        new_population = engine._repopulate(performances, survivors)  # type: ignore[attr-defined]

    assert len(new_population) == engine.config.population_size
    assert mutate_mock.call_count >= len(survivors)

    for survivor in survivors:
        assert sum(1 for member in new_population if member is survivor) == 1
        snapshot = survivor_snapshots[id(survivor)]
        assert survivor.describe_formulas() == snapshot.describe_formulas()
        assert [asset.weight for asset in survivor.assets] == [
            asset.weight for asset in snapshot.assets
        ]

    clone_count = len([member for member in new_population if member not in survivors])
    assert clone_count == engine.config.population_size - len(survivors)


def test_best_equity_curve_is_monotonic() -> None:
    data = _build_mock_data()
    engine = EvolutionEngine(data=data, config=_default_config(), rng=random.Random(19))
    population = engine.initialize_population(list(data.tickers))

    population, report0 = engine.evolve(population, generation=0)
    population, report1 = engine.evolve(population, generation=1)

    assert report1.best_member is not None
    assert report0.best_member is not None
    assert report1.best_member.final_equity >= report0.best_member.final_equity
