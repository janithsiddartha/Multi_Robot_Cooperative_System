import numpy as np

def euler_to_matrix(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def matrix_to_euler(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw])

def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n

def is_rotation_matrix(R, tol=1e-6):
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    return (
        np.allclose(R.T @ R, np.eye(3), atol=tol)
        and np.isclose(np.linalg.det(R), 1.0, atol=tol)
    )

def get_handover_relative_rotations():
    R_plus = np.array([
        [0.0,  1.0,  0.0],
        [1.0,  0.0,  0.0],
        [0.0,  0.0, -1.0],
    ])
    R_minus = np.array([
        [0.0, -1.0,  0.0],
        [-1.0, 0.0,  0.0],
        [0.0,  0.0, -1.0],
    ])
    return {
        +1: R_plus,
        -1: R_minus,
    }

def make_tcp_orientation(z_tcp, phi=0.0):
    x_obj = np.array([np.cos(phi), np.sin(phi), 0.0])
    if np.abs(np.dot(normalize(z_tcp), normalize(x_obj))) > 0.99:
        x_obj = np.array([0.0, np.cos(phi), np.sin(phi)])
    return make_tilted_tcp_orientation(z_tcp, x_object=x_obj)


def make_tilted_tcp_orientation(z_tcp, x_object=[1.0, 0.0, 0.0]):
    """
    Construct TCP rotation matrix given approach direction z_tcp
    and the desired object X axis alignment.
    """
    z = normalize(z_tcp)
    x_obj = normalize(x_object)
    
    # Project x_object onto plane perp to z
    x_proj = x_obj - np.dot(x_obj, z) * z
    if np.linalg.norm(x_proj) < 1e-6:
        raise ValueError("z_tcp and x_object are parallel")
        
    x = normalize(x_proj)
    y = normalize(np.cross(z, x))
    
    R = np.column_stack((x, y, z))
    if not is_rotation_matrix(R):
        raise RuntimeError("Invalid TCP rotation generated.")
    return R

def make_receiver_orientation(R_picker, branch):
    rotations = get_handover_relative_rotations()
    if branch not in rotations:
        raise ValueError("branch must be +1 or -1")
    R_receiver = R_picker @ rotations[branch]
    if not is_rotation_matrix(R_receiver):
        raise RuntimeError("Invalid receiver rotation generated.")
    return R_receiver

def validate_handover_orientation(R1, R2, tol=1e-6):
    if not is_rotation_matrix(R1):
        return False
    if not is_rotation_matrix(R2):
        return False
    x1 = R1[:, 0]
    y1 = R1[:, 1]
    z1 = R1[:, 2]
    x2 = R2[:, 0]
    y2 = R2[:, 1]
    z2 = R2[:, 2]
    return (
        np.isclose(np.dot(z1, z2), -1.0, atol=tol)
        and np.isclose(np.dot(x1, x2), 0.0, atol=tol)
        and np.isclose(np.dot(y1, y2), 0.0, atol=tol)
    )

def rotation_error(R_desired, R_actual):
    R_error = R_desired.T @ R_actual
    value = (np.trace(R_error) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)
    return float(np.arccos(value))

def manipulability(J, eps=1e-12):
    J = np.asarray(J, dtype=float)
    value = np.linalg.det(J @ J.T)
    return float(np.sqrt(max(value, eps)))

def singularity_cost(J, eps=1e-9):
    mu = manipulability(J)
    return 1.0 / (mu + eps)
