import itertools

from stck.ui import determine_generation_limit, generation_sequence


def test_determine_generation_limit_prefers_requested() -> None:
    assert determine_generation_limit(10, 5) == 5
    assert determine_generation_limit(10, 0) == 0


def test_determine_generation_limit_infinite_when_zero() -> None:
    assert determine_generation_limit(0, None) is None
    assert determine_generation_limit(-1, None) is None
    assert determine_generation_limit(12, None) == 12


def test_generation_sequence_handles_infinite() -> None:
    finite = list(generation_sequence(3))
    assert finite == [0, 1, 2]
    infinite = generation_sequence(None)
    first_three = list(itertools.islice(infinite, 3))
    assert first_three == [0, 1, 2]
