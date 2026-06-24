import pytest

from oscarlic.bd import BDError, bd_rate_linear


def test_constant_half_rate_is_minus_50_percent():
    result = bd_rate_linear(
        anchor_rate=[0.1, 0.2, 0.3],
        anchor_metric=[0.3, 0.2, 0.1],
        candidate_rate=[0.05, 0.1, 0.15],
        candidate_metric=[0.3, 0.2, 0.1],
    )
    assert result["bd_rate_percent"] == pytest.approx(-50.0, abs=0.05)
    assert result["extrapolation"] is False


def test_no_overlap_rejected():
    with pytest.raises(BDError):
        bd_rate_linear([1, 2], [0.1, 0.2], [1, 2], [0.3, 0.4])
