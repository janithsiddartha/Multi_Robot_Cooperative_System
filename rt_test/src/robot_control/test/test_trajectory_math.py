import numpy as np

from robot_control.handover.trajectory import (
    smoothstep7,
    find_feasible_duration
)


def test_smooth_7th_order_bounds():
    q_start = np.array([0.0])
    q_target = np.array([1.0])
    v0, _, _, _ = smoothstep7(q_start, q_target, 0.0, 1.0)
    v1, _, _, _ = smoothstep7(q_start, q_target, 0.5, 1.0)
    v2, _, _, _ = smoothstep7(q_start, q_target, 1.0, 1.0)
    assert np.allclose(v0, q_start)
    assert np.allclose(v2, q_target)


def test_calculate_duration_is_positive():
    q_start = np.deg2rad(np.array([0.0, 10.0, 20.0, 30.0, 40.0]))
    q_target = np.deg2rad(np.array([90.0, 45.0, 15.0, 60.0, 80.0]))

    v = np.full(5, np.deg2rad(10.0))
    a = np.full(5, np.deg2rad(100.0))
    j = np.full(5, np.deg2rad(500.0))

    duration = find_feasible_duration(q_start, q_target, 1.0, v, a, j)

    assert duration >= 1.0
    assert np.isfinite(duration)
