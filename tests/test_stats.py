import numpy as np
import pytest

from cryptoflow.stats import bayesian_ab, msprt


def test_msprt_rejects_empty_snapshots():
    with pytest.raises(ValueError, match="non-empty"):
        msprt([], sigma=1.0)


def test_msprt_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        msprt([(10, 1.0, 10, 1.1)], sigma=1.0, alpha=1.0)


def test_bayesian_rejects_too_small_samples():
    with pytest.raises(ValueError, match="at least 2"):
        bayesian_ab(np.array([1.0]), np.array([2.0]))


def test_bayesian_zero_variance_is_finite():
    res = bayesian_ab(np.array([2.0, 2.0, 2.0]), np.array([3.0, 3.0, 3.0]))
    assert res.effect_std == 0.0
    assert res.credible_interval_95 == (1.0, 1.0)
    assert np.isfinite(res.expected_loss_launch)
    assert np.isfinite(res.expected_loss_hold)
