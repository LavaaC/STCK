from __future__ import annotations

import pytest

from stck.data import HistoricalData


def test_historical_data_rejects_empty_series():
    with pytest.raises(ValueError, match="must contain at least one observation"):
        HistoricalData({"AAA": []})


def test_history_for_unknown_ticker_raises_keyerror():
    data = HistoricalData({"AAA": [1, 2, 3]})
    with pytest.raises(KeyError) as exc:
        data.history_for("BBB", 0)
    assert "BBB" in str(exc.value)
    assert "AAA" in str(exc.value)


def test_price_history_indicators_cover_common_metrics():
    data = HistoricalData({"AAA": [10, 11, 12, 11, 13, 14, 15, 14, 16, 17]})
    history = data.history_for("AAA", len(data) - 1)

    ema = history.exponential_moving_average(5)
    rsi = history.relative_strength_index(5)
    macd_value = history.macd(3, 6)
    band_width = history.bollinger_band_width(5)

    assert isinstance(ema, float)
    assert 0.0 <= rsi <= 100.0
    assert isinstance(macd_value, float)
    assert band_width >= 0.0
