from stck.presets import PRESET_BUILDERS, get_preset_formula


def test_get_preset_formula_known_ticker() -> None:
    for ticker in PRESET_BUILDERS:
        formula = get_preset_formula(ticker)
        assert formula is not None
        assert formula.evaluation_windows == sorted(formula.evaluation_windows)
        assert formula.describe()  # non-empty description


def test_get_preset_formula_unknown_ticker() -> None:
    assert get_preset_formula("UNKNOWN") is None
