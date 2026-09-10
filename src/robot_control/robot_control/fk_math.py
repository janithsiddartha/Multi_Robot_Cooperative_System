import numpy as np
from robot_control.robot_kinematics import alpha_deg, a, d, TCP_OFFSET

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


def dh_joint_angles(q):
    theta1, theta2, theta3, theta4, theta5 = q
    theta_dh = np.array([
        theta1,
        np.pi / 2.0 - theta2,
        -np.pi / 2.0 - theta3,
        theta4,
        np.pi / 2.0,
        -np.pi / 2.0 - theta5,
        np.pi
    ])
    return theta_dh


def fk(q):
    """
    Forward kinematics.
    Input: q in radians
    Returns: p (TCP position in mm), T (4x4 matrix)
    """
    q = np.asarray(q, dtype=float)
    theta_dh = dh_joint_angles(q)
    T = np.eye(4)

    for i in range(7):
        alpha_i = np.deg2rad(alpha_deg[i])
        Ti = mdh_matrix(a[i], alpha_i, theta_dh[i], d[i])
        T = T @ Ti

    p_dh = T[:3, 3].copy()
    p_tcp = p_dh + TCP_OFFSET
    return p_tcp, T


def all_frame_positions(q):
    q = np.asarray(q, dtype=float)
    theta_dh = dh_joint_angles(q)
    T = np.eye(4)
    positions = []

    # Frame 0
    positions.append(T[:3, 3].copy())
    for i in range(7):
        Ti = mdh_matrix(a[i], np.deg2rad(alpha_deg[i]), theta_dh[i], d[i])
        T = T @ Ti
        positions.append(T[:3, 3].copy())

    return np.array(positions)
