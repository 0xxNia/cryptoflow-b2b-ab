import numpy as np

from cryptoflow.agents import simulate


def test_simulation_is_reproducible_for_same_inputs():
    params = {"fee_change_pct": -20.0}
    first = simulate(params, scenario_id="fee_reduction")
    second = simulate(params, scenario_id="fee_reduction")

    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a.robot.name == b.robot.name
        assert a.group == b.group
        assert np.isclose(a.trades_change_mean, b.trades_change_mean)
        assert np.isclose(a.churn_change_mean, b.churn_change_mean)
