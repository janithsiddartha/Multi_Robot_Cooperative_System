import numpy as np
from robot_control.fk_math import fk
from robot_control.robot_kinematics import JOINT_LIMITS_RAD

# ============================================================
# NUMERICAL JACOBIAN
# ============================================================

def numerical_jacobian(q, delta=1e-6):
    q = np.asarray(q, dtype=float)
    J = np.zeros((3, 5))
    for i in range(5):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[i] += delta
        q_minus[i] -= delta

        p_plus, _ = fk(q_plus)
        p_minus, _ = fk(q_minus)

        J[:, i] = (p_plus - p_minus) / (2.0 * delta)
    return J


# ============================================================
# JOINT CENTER COST
# ============================================================

def posture_cost(q):
    q_min = JOINT_LIMITS_RAD[:, 0]
    q_max = JOINT_LIMITS_RAD[:, 1]
    q_mid = 0.5 * (q_min + q_max)
    q_range = q_max - q_min
    normalized_error = ((q - q_mid) / (0.5 * q_range))
    return np.sum(normalized_error ** 2)


# ============================================================
# DISTANCE FROM JOINT LIMITS
# ============================================================

def joint_limit_margin(q):
    q_min = JOINT_LIMITS_RAD[:, 0]
    q_max = JOINT_LIMITS_RAD[:, 1]
    margin_low = q - q_min
    margin_high = q_max - q
    margin = np.minimum(margin_low, margin_high)
    return np.min(margin)


# ============================================================
# NUMERICAL GRADIENT OF POSTURE COST
# ============================================================

def posture_gradient(q, delta=1e-6):
    grad = np.zeros(5)
    for i in range(5):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[i] += delta
        q_minus[i] -= delta
        grad[i] = (posture_cost(q_plus) - posture_cost(q_minus)) / (2.0 * delta)
    return grad


# ============================================================
# DAMPED PSEUDOINVERSE
# ============================================================

def damped_pseudoinverse(J, damping=0.01):
    rows = J.shape[0]
    I = np.eye(rows)
    return J.T @ np.linalg.inv(J @ J.T + damping**2 * I)


# ============================================================
# SINGLE IK SOLVER
# ============================================================

def ik_single(
    target,
    q_init=None,
    max_iterations=500,
    tolerance=0.05,
    damping=0.5,
    step_size=0.7,
    posture_gain=0.01
):
    target = np.asarray(target, dtype=float)
    if q_init is None:
        q = np.zeros(5)
    else:
        q = np.asarray(q_init, dtype=float).copy()

    q = np.clip(q, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])

    for iteration in range(max_iterations):
        current_position, _ = fk(q)
        error = target - current_position
        error_norm = np.linalg.norm(error)

        if error_norm < tolerance:
            return {"success": True, "q": q, "iterations": iteration, "error": error_norm}

        J = numerical_jacobian(q)
        J_pinv = damped_pseudoinverse(J, damping)
        dq_task = J_pinv @ error
        
        N = np.eye(5) - J_pinv @ J
        grad_H = posture_gradient(q)
        dq_null = -posture_gain * N @ grad_H

        dq = dq_task + dq_null
        q += step_size * dq

        q = np.clip(q, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])

    final_position, _ = fk(q)
    final_error = np.linalg.norm(target - final_position)

    return {"success": final_error < tolerance, "q": q, "iterations": max_iterations, "error": final_error}


# ============================================================
# MULTI-START IK
# ============================================================

def ik(target, current_q=None, num_random_starts=30):
    target = np.asarray(target, dtype=float)
    candidates = []
    initial_guesses = [np.zeros(5)]

    if current_q is not None:
        initial_guesses.append(np.asarray(current_q, dtype=float))

    q_min = JOINT_LIMITS_RAD[:, 0]
    q_max = JOINT_LIMITS_RAD[:, 1]
    
    for _ in range(num_random_starts):
        initial_guesses.append(q_min + (q_max - q_min) * np.random.rand(5))

    for q_init in initial_guesses:
        result = ik_single(target, q_init=q_init)
        if not result["success"]:
            continue

        q_sol = result["q"]
        p_sol, _ = fk(q_sol)
        position_error = np.linalg.norm(target - p_sol)

        if position_error > 0.1:
            continue

        posture = posture_cost(q_sol)
        limit_margin = joint_limit_margin(q_sol)
        movement = np.linalg.norm(q_sol - np.asarray(current_q)) if current_q is not None else 0.0

        score = 1000.0 * position_error + 1.0 * posture + 0.1 * movement

        candidates.append({
            "q": q_sol,
            "error": position_error,
            "posture": posture,
            "limit_margin": limit_margin,
            "movement": movement,
            "score": score
        })

    if not candidates:
        return {"success": False, "q": None, "solutions": []}

    candidates.sort(key=lambda x: x["score"])
    best = candidates[0]

    return {"success": True, "q": best["q"], "error": best["error"], "solutions": candidates}
