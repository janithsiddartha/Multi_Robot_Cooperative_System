import numpy as np

from robot_control.trajectory_math import (
    calculate_duration,
    generate_joint_trajectory,
    smooth_7th_order,
)


def test_smooth_7th_order_bounds():
    values = [smooth_7th_order(0.0), smooth_7th_order(0.5), smooth_7th_order(1.0)]
    assert np.all(np.isfinite(values))
    assert values[0] == 0.0
    assert values[-1] == 1.0


def test_generate_joint_trajectory_shapes():
    q_start = np.deg2rad(np.array([0.0, 10.0, 20.0, 30.0, 40.0]))
    q_target = np.deg2rad(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))

    time, trajectory = generate_joint_trajectory(q_start, q_target, dt=0.02)

    assert len(time) == len(trajectory)
    assert time[0] == 0.0
    assert np.all(np.isfinite(trajectory))
    assert np.allclose(trajectory[-1], q_target, atol=1e-9)


def test_calculate_duration_is_positive():
    q_start = np.deg2rad(np.array([0.0, 10.0, 20.0, 30.0, 40.0]))
    q_target = np.deg2rad(np.array([90.0, 45.0, 15.0, 60.0, 80.0]))

    duration = calculate_duration(q_start, q_target)

    assert duration >= 1.0
    assert np.isfinite(duration)
