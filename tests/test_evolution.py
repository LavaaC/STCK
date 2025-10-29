from __future__ import annotations

import random

from stck.data import HistoricalData
from stck.evolution import EvolutionConfig, EvolutionEngine


def test_evolution_repopulates_population():
    data = HistoricalData({"AAA": [10, 12, 11, 13, 12, 14]})
    engine = EvolutionEngine(
        data=data,
        config=EvolutionConfig(population_size=4),
        rng=random.Random(5),
    )
    population = engine.initialize_population(["AAA"])
    original_formulas = list(population["AAA"])
    evolved, report = engine.evolve(population, generation=0)
    assert "AAA" in evolved
    assert len(evolved["AAA"]) == 4
    # Ensure at least one formula differs from the original population to confirm mutation occurred
    assert any(f.describe() != o.describe() for f, o in zip(evolved["AAA"], original_formulas))
    # Report should carry metrics for visualization
    assert report.metrics.top_equity >= report.metrics.top10_mean >= 0


def test_selection_elitism_and_culling():
    data = HistoricalData({"AAA": [10, 12, 11, 13, 12, 14, 13, 15]})
    engine = EvolutionEngine(data=data, config=EvolutionConfig(population_size=10), rng=random.Random(42))
    population = engine.initialize_population(["AAA"])
    performances = engine._evaluate_ticker("AAA", population["AAA"])
    survivors = engine._select_survivors(performances)
    assert len(survivors) >= 1
    # Top performer should always survive
    top_description = max(performances, key=lambda p: p.average_final_equity).formula.describe()
    assert any(s.describe() == top_description for s in survivors)
    # Bottom performer should be culled
    bottom_description = min(performances, key=lambda p: p.average_final_equity).formula.describe()
    assert all(s.describe() != bottom_description for s in survivors)


def test_repopulate_recovers_from_empty_survivors():
    data = HistoricalData({"AAA": [10, 11, 12, 13, 14]})
    engine = EvolutionEngine(data=data, config=EvolutionConfig(population_size=3), rng=random.Random(0))
    population = engine._repopulate([])
    assert len(population) == 3
    assert all(isinstance(formula.priority, int) for formula in population)
