import numpy as np
from robot_control.kinematics.fk_math import dh_joint_angles, mdh_matrix

def normalize_quaternion(q):
    """Normalize a quaternion [w, x, y, z]."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def quaternion_multiply(q1, q2):
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_inverse(q):
    """Inverse of a unit quaternion [w, x, y, z]."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    m = np.asarray(R, dtype=float)
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2,1] - m[1,2]) / S
        y = (m[0,2] - m[2,0]) / S
        z = (m[1,0] - m[0,1]) / S
    elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
        S = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
        w = (m[2,1] - m[1,2]) / S
        x = 0.25 * S
        y = (m[0,1] + m[1,0]) / S
        z = (m[0,2] + m[2,0]) / S
    elif m[1,1] > m[2,2]:
        S = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
        w = (m[0,2] - m[2,0]) / S
        x = (m[0,1] + m[1,0]) / S
        y = 0.25 * S
        z = (m[1,2] + m[2,1]) / S
    else:
        S = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
        w = (m[1,0] - m[0,1]) / S
        x = (m[0,2] + m[2,0]) / S
        y = (m[1,2] + m[2,1]) / S
        z = 0.25 * S
    return normalize_quaternion([w, x, y, z])

def rotation_error_vector(R_current, R_target):
    """Compute 3D orientation error vector using quaternions."""
    q_c = rotation_matrix_to_quaternion(R_current)
    q_d = rotation_matrix_to_quaternion(R_target)
    
    q_e = quaternion_multiply(q_d, quaternion_inverse(q_c))
    w_e = q_e[0]
    v_e = q_e[1:]
    
    v_norm = np.linalg.norm(v_e)
    if v_norm < 1e-6:
        # Small angle approximation
        e_R = 2.0 * np.sign(w_e if w_e != 0 else 1.0) * v_e
    else:
        theta = 2.0 * np.arctan2(v_norm, np.abs(w_e))
        u = v_e / v_norm
        e_R = theta * np.sign(w_e if w_e != 0 else 1.0) * u
        
    return e_R

def pose_error(current_position, current_rotation, target_position, target_rotation):
    """Compute 6D pose error [e_p, e_R]."""
    e_p = np.asarray(target_position) - np.asarray(current_position)
    e_R = rotation_error_vector(current_rotation, target_rotation)
    return np.concatenate((e_p, e_R))

def geometric_jacobian_and_fk(q, config):
    """Compute geometric Jacobian and FK simultaneously."""
    q = np.asarray(q, dtype=float)
    theta_dh = dh_joint_angles(q, config)
    T = np.eye(4)
    
    z_axes = []
    p_origins = []
    joint_indices = []
    multipliers = []
    
    for i in range(config.num_frames):
        alpha_i = np.deg2rad(config.alpha_deg[i])
        Ti = mdh_matrix(config.a[i], alpha_i, theta_dh[i], config.d[i])
        T = T @ Ti
        
        idx = config.theta_joint_index[i]
        if idx >= 0:
            z_axes.append(T[:3, 2].copy())
            p_origins.append(T[:3, 3].copy())
            joint_indices.append(idx)
            multipliers.append(config.theta_joint_multiplier[i])
            
    p_tcp = T[:3, 3] + config.tcp_offset
    R_tcp = T[:3, :3]
    
    J = np.zeros((6, config.num_joints))
    for z, p_orig, idx, mult in zip(z_axes, p_origins, joint_indices, multipliers):
        J[:3, idx] += mult * np.cross(z, p_tcp - p_orig)
        J[3:, idx] += mult * z
        
    return J, p_tcp, R_tcp

def singularity_metrics(J):
    """Compute minimum singular value and condition number."""
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    sigma_min = S[-1]
    sigma_max = S[0]
    cond = sigma_max / sigma_min if sigma_min > 1e-12 else float('inf')
    return sigma_min, cond

def adaptive_damping(sigma_min, min_manipulability=0.01):
    """Compute adaptive damping based on sigma_min."""
    if sigma_min >= min_manipulability:
        return 0.01  # Small nominal damping
    elif sigma_min > 1e-6:
        # Ramp up damping as we approach singularity
        return 0.01 + 0.1 * (1.0 - sigma_min / min_manipulability)**2
    else:
        return 0.2  # Maximum damping in severe singularity

def joint_limit_cost(q, config):
    """Compute cost that penalizes approaching joint limits."""
    q_min = config.joint_limits_rad[:, 0]
    q_max = config.joint_limits_rad[:, 1]
    
    margin_low = q - q_min
    margin_high = q_max - q
    
    # Simple quadratic barrier
    cost = 0.0
    for ml, mh in zip(margin_low, margin_high):
        m = min(ml, mh)
        if m < 0.1:
            cost += (0.1 - m)**2
    return cost

def ik_pose_single(target_position, target_rotation, config, q_init, max_iterations=100,
                  pos_tolerance=1.0, ori_tolerance_deg=15.0):
    """Solve IK for a single seed."""
    q = np.asarray(q_init, dtype=float).copy()
    q_min = config.joint_limits_rad[:, 0]
    q_max = config.joint_limits_rad[:, 1]
    
    q = np.clip(q, q_min, q_max)
    
    ori_tolerance_rad = np.deg2rad(ori_tolerance_deg)
    
    # For 5-DOF, we heavily weight position so it reaches the XYZ target
    # and just uses the remaining redundancy to minimize orientation.
    if config.num_joints == 5:
        W = np.diag([1.0, 1.0, 1.0, 0.001, 0.001, 0.001])
    else:
        W = np.eye(6)
        
    alpha = 0.5  # Step size
    
    best_error = float('inf')
    stagnation_count = 0
    
    for iteration in range(max_iterations):
        J, p_current, R_current = geometric_jacobian_and_fk(q, config)
        
        e = pose_error(p_current, R_current, target_position, target_rotation)
        pos_err_norm = np.linalg.norm(e[:3])
        ori_err_norm = np.linalg.norm(e[3:])
        
        total_err = pos_err_norm + ori_err_norm
        
        if pos_err_norm <= pos_tolerance and ori_err_norm <= ori_tolerance_rad:
            # Re-verify FK exactly
            return {"success": True, "q": q, "pos_error": pos_err_norm, "ori_error": ori_err_norm, "iterations": iteration}
            
        if total_err < best_error - 1e-4:
            best_error = total_err
            stagnation_count = 0
        else:
            stagnation_count += 1
            
        if stagnation_count > 10 or total_err > 1e4:
            break # Stalled or diverged
            
        sigma_min, _ = singularity_metrics(J)
        lam = adaptive_damping(sigma_min)
        
        if config.num_joints == 5:
            # Weighted DLS
            H = J.T @ W @ J + (lam**2) * np.eye(config.num_joints)
            dq = np.linalg.inv(H) @ J.T @ W @ e
        else:
            # Standard DLS
            H = J.T @ J + (lam**2) * np.eye(config.num_joints)
            dq = np.linalg.inv(H) @ J.T @ e
            
        q_next = q + alpha * dq
        q_next = np.clip(q_next, q_min, q_max)
        
        # Check step size to prevent infinite tiny steps
        if np.linalg.norm(q_next - q) < 1e-5:
            break
            
        q = q_next
        
    return {"success": False, "q": q, "pos_error": float('inf'), "ori_error": float('inf'), "iterations": max_iterations}

def validate_pose_solution(q, target_position, target_rotation, config, pos_tolerance=1.0, ori_tolerance_deg=15.0, min_margin_deg=1.0, min_manipulability=0.01):
    """Strict final validation of an IK solution."""
    _, p_fk, R_fk = geometric_jacobian_and_fk(q, config)
    
    pos_error = np.linalg.norm(target_position - p_fk)
    
    # Angle of R_d^T R_fk
    R_err = target_rotation.T @ R_fk
    tr = np.trace(R_err)
    val = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    ori_error_deg = np.rad2deg(np.arccos(val))
    
    q_min = config.joint_limits_rad[:, 0]
    q_max = config.joint_limits_rad[:, 1]
    
    if np.any(q < q_min - 1e-9) or np.any(q > q_max + 1e-9):
        return False, pos_error, ori_error_deg, 0.0, 0.0
        
    margin_deg = np.rad2deg(np.min(np.minimum(q - q_min, q_max - q)))
    
    J, _, _ = geometric_jacobian_and_fk(q, config)
    sigma_min, _ = singularity_metrics(J)
    
    if (pos_error <= pos_tolerance and 
        ori_error_deg <= ori_tolerance_deg and 
        margin_deg >= min_margin_deg and 
        sigma_min >= min_manipulability):
        return True, pos_error, ori_error_deg, margin_deg, sigma_min
        
    return False, pos_error, ori_error_deg, margin_deg, sigma_min

def ik_pose_multiple(target_position, target_rotation, config, seeds=None, pos_tolerance=1.0, ori_tolerance_deg=15.0, min_margin_deg=1.0, min_manipulability=0.01):
    """Run solver from multiple seeds and return best valid solution."""
    q_min = config.joint_limits_rad[:, 0]
    q_max = config.joint_limits_rad[:, 1]
    
    if seeds is None:
        seeds = []
        # Home
        if hasattr(config, "home_deg"):
            seeds.append(np.deg2rad(np.asarray(config.home_deg, dtype=float)))
        # Midpoint
        seeds.append(0.5 * (q_min + q_max))
        # Lower bias
        seeds.append(q_min + 0.1 * (q_max - q_min))
        # Upper bias
        seeds.append(q_max - 0.1 * (q_max - q_min))
        
        # Some deterministic pseudo-random seeds
        rng = np.random.RandomState(42)
        for _ in range(6):
            seeds.append(q_min + (q_max - q_min) * rng.rand(config.num_joints))
            
    valid_solutions = []
    
    for q_init in seeds:
        res = ik_pose_single(target_position, target_rotation, config, q_init,
                            pos_tolerance=pos_tolerance, ori_tolerance_deg=ori_tolerance_deg)
                            
        if res["success"]:
            q_sol = res["q"]
            is_valid, p_err, o_err, margin, sigma = validate_pose_solution(
                q_sol, target_position, target_rotation, config,
                pos_tolerance=pos_tolerance, ori_tolerance_deg=ori_tolerance_deg,
                min_margin_deg=min_margin_deg, min_manipulability=min_manipulability
            )
            
            if is_valid:
                # Check for duplicates
                is_duplicate = False
                for sol in valid_solutions:
                    if np.allclose(sol["q"], q_sol, atol=1e-3):
                        is_duplicate = True
                        break
                        
                if not is_duplicate:
                    # Deterministic cost: heavily penalize orientation and position error, then prefer margin and non-singularity
                    cost = 1000.0 * p_err + 100.0 * o_err - 10.0 * margin - 100.0 * sigma + joint_limit_cost(q_sol, config)
                    
                    valid_solutions.append({
                        "q": q_sol,
                        "position_error_mm": p_err,
                        "orientation_error_deg": o_err,
                        "joint_limit_margin_deg": margin,
                        "manipulability": sigma,
                        "cost": cost
                    })
                    
    if not valid_solutions:
        return None
        
    valid_solutions.sort(key=lambda x: x["cost"])
    return valid_solutions[0]
