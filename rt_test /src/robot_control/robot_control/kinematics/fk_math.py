import numpy as np

# ============================================================
# MODIFIED DH MATH
# ============================================================

def mdh_matrix(a_i, alpha_i, theta_i, d_i):
    ca = np.cos(alpha_i)
    sa = np.sin(alpha_i)
    ct = np.cos(theta_i)
    st = np.sin(theta_i)

    T = np.array([
        [ct,       -st,        0.0,       a_i],
        [st * ca,  ct * ca,   -sa,       -d_i * sa],
        [st * sa,  ct * sa,    ca,        d_i * ca],
        [0.0,       0.0,        0.0,       1.0]
    ])
    return T


def dh_joint_angles(q, config):
    theta_dh = np.zeros(7)
    for i in range(7):
        theta_dh[i] = np.deg2rad(config.theta_offset_deg[i])
        idx = config.theta_joint_index[i]
        if idx >= 0:
            theta_dh[i] += config.theta_joint_multiplier[i] * q[idx]
    return theta_dh


def fk(q, config):
    """
    Forward kinematics.
    Input: q in radians
    Returns: p (TCP position in mm), T (4x4 matrix)
    """
    q = np.asarray(q, dtype=float)
    theta_dh = dh_joint_angles(q, config)
    T = np.eye(4)

    for i in range(7):
        alpha_i = np.deg2rad(config.alpha_deg[i])
        Ti = mdh_matrix(config.a[i], alpha_i, theta_dh[i], config.d[i])
        T = T @ Ti

    p_dh = T[:3, 3].copy()
    p_tcp = p_dh + config.tcp_offset
    return p_tcp, T


def all_frame_positions(q, config):
    q = np.asarray(q, dtype=float)
    theta_dh = dh_joint_angles(q, config)
    T = np.eye(4)
    positions = []

    # Frame 0
    positions.append(T[:3, 3].copy())
    for i in range(7):
        Ti = mdh_matrix(config.a[i], np.deg2rad(config.alpha_deg[i]), theta_dh[i], config.d[i])
        T = T @ Ti
        positions.append(T[:3, 3].copy())

    return np.array(positions)
