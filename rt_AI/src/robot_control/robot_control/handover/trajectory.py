import numpy as np

def smoothstep7(q0, qf, t, T):
    """
    7th-order smooth trajectory.
    Zero velocity, acceleration and jerk
    at both boundaries.
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    q0 = np.asarray(q0, dtype=float)
    qf = np.asarray(qf, dtype=float)
    s = np.clip(t / T, 0.0, 1.0)
    S = (
        35*s**4
        -84*s**5
        +70*s**6
        -20*s**7
    )
    dS = (
        140*s**3
        -420*s**4
        +420*s**5
        -140*s**6
    )
    ddS = (
        420*s**2
        -1680*s**3
        +2100*s**4
        -840*s**5
    )
    dddS = (
        840*s
        -5040*s**2
        +8400*s**3
        -4200*s**4
    )
    dq = qf - q0
    q = q0 + dq * S
    velocity = dq * dS / T
    acceleration = dq * ddS / (T**2)
    jerk = dq * dddS / (T**3)
    return q, velocity, acceleration, jerk

def check_trajectory_limits(
    q0,
    qf,
    T,
    velocity_limits,
    acceleration_limits,
    jerk_limits,
    samples=200,
):
    times = np.linspace(0.0, T, samples)
    velocity_limits = np.asarray(velocity_limits)
    acceleration_limits = np.asarray(acceleration_limits)
    jerk_limits = np.asarray(jerk_limits)
    max_velocity = 0.0
    max_acceleration = 0.0
    max_jerk = 0.0
    for t in times:
        _, dq, ddq, dddq = smoothstep7(q0, qf, t, T)
        max_velocity = max(max_velocity, float(np.max(np.abs(dq))))
        max_acceleration = max(max_acceleration, float(np.max(np.abs(ddq))))
        max_jerk = max(max_jerk, float(np.max(np.abs(dddq))))
        if np.any(np.abs(dq) > velocity_limits):
            return False
        if np.any(np.abs(ddq) > acceleration_limits):
            return False
        if np.any(np.abs(dddq) > jerk_limits):
            return False
    return True

def find_feasible_duration(
    q0,
    qf,
    initial_duration,
    velocity_limits,
    acceleration_limits,
    jerk_limits,
    max_duration=30.0,
):
    T = float(initial_duration)
    while T <= max_duration:
        if check_trajectory_limits(
            q0,
            qf,
            T,
            velocity_limits,
            acceleration_limits,
            jerk_limits,
        ):
            return T
        T *= 1.10
    return None
