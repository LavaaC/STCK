"""Unit tests for the evolutionary engine."""

from __future__ import annotations

import random

from stck.data import HistoricalData
from stck.evolution import EvolutionConfig, EvolutionEngine, PortfolioMember


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
    new_population, clones_created = engine._repopulate(  # type: ignore[attr-defined]
        performances, survivors
    )

    assert len(new_population) == engine.config.population_size
    for survivor in survivors:
        assert survivor in new_population
    if len(survivors) < engine.config.population_size:
        assert any(member not in survivors for member in new_population)
    assert clones_created >= 0
