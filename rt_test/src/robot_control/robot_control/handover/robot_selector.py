import numpy as np
from dataclasses import dataclass


@dataclass
class HandoverCandidate:
    """A candidate handover configuration between two robots."""
    position: np.ndarray
    picker_tilt_deg: float
    receiver_tilt_deg: float
    picker_robot: int
    receiver_robot: int
    picker_rotation: np.ndarray
    receiver_rotation: np.ndarray
    picker_joints: object = None
    receiver_joints: object = None
    picker_manipulability: float = None
    receiver_manipulability: float = None
    trajectory_time: float = None
    minimum_clearance: float = None
    cost: float = None
    valid: bool = False
    waypoints_used: bool = False
    rejection_reason: str = None


def robot_selection_cost(
    current_position,
    target_position,
    current_q,
    target_q,
    orientation_error,
    manipulability_value,
    weights,
    joint_limit_margin=10.0,
    eps=1e-9,
    max_singularity_cost=1000.0,
):
    """Compute soft selection cost among FEASIBLE robots.

    All inputs must belong to a robot that has already passed
    hard constraints.  This function does NOT rescue infeasible robots.

    The singularity term  1/(mu + eps)  is clamped to
    *max_singularity_cost* to prevent it from numerically
    dominating the other terms.
    """
    distance_cost = np.linalg.norm(
        np.asarray(target_position)
        - np.asarray(current_position)
    )
    joint_cost = np.linalg.norm(
        np.asarray(target_q)
        - np.asarray(current_q)
    )
    singularity_cost_value = min(
        1.0 / (manipulability_value + eps),
        max_singularity_cost,
    )
    # Penalize being very close to joint limits (e.g. margin < 10 degrees)
    margin_cost = max(0.0, 10.0 - joint_limit_margin)
    
    total = (
        weights.get("distance", 1.0) * distance_cost
        + weights.get("joint", 0.5) * joint_cost
        + weights.get("orientation", 2.0) * orientation_error
        + weights.get("singularity", 0.1) * singularity_cost_value
        + weights.get("joint_limit", 5.0) * margin_cost
    )
    return float(total)
