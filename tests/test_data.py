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
