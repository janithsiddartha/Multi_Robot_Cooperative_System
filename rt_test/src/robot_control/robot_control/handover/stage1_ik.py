"""Stage-1 position IK (XYZ only).

This is the same solver as ``robot_control.kinematics.ik_solver``:
target is a Cartesian point, no gripper / TCP-orientation constraints.

``task_ik`` is not used here. That solver searches approach yaw/tilt poses.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from robot_control.kinematics.fk_math import fk
from robot_control.kinematics.ik_solver import ik


@dataclass
class Stage1IKResult:
    """One feasible XYZ IK solution for a robot."""

    robot_id: str
    xyz_world: np.ndarray
    xyz_base: np.ndarray
    q_deg: np.ndarray
    q_rad: np.ndarray
    error_mm: float
    solution_count: int


def world_to_base(p_world, config):
    """Convert a world-frame XYZ point into the robot base frame."""
    p = np.asarray(p_world, dtype=float)
    if hasattr(config, "world_base_R") and hasattr(config, "world_base_xyz"):
        origin = np.asarray(config.world_base_xyz, dtype=float)
        return config.world_base_R.T @ (p - origin)
    return p


def solve_stage1_ik(
    robot_id,
    target_xyz_world,
    config,
    current_q_deg=None,
    num_random_starts=30,
):
    """Solve XYZ IK for one robot.

    Parameters
    ----------
    robot_id : str
        e.g. ``robot1``.
    target_xyz_world : array-like, shape (3,)
        Target point in the common world frame (mm).
    config : kinematics config
        Must provide DH / joint limits. World-base fields are used if present.
    current_q_deg : array-like, optional
        Current joints in degrees, used as an IK seed.

    Returns
    -------
    Stage1IKResult or None
        Best joint pose in degrees, or None if the point is unreachable.
    """
    target_world = np.asarray(target_xyz_world, dtype=float)
    target_base = world_to_base(target_world, config)

    current_q_rad = None
    if current_q_deg is not None:
        current_q_rad = np.deg2rad(np.asarray(current_q_deg, dtype=float))

    result = ik(
        target=target_base,
        config=config,
        current_q=current_q_rad,
        num_random_starts=num_random_starts,
    )

    if not result.get("success"):
        return None

    q_rad = np.asarray(result["q"], dtype=float)
    q_deg = np.rad2deg(q_rad)
    achieved, _ = fk(q_rad, config)
    error_mm = float(np.linalg.norm(target_base - achieved))

    return Stage1IKResult(
        robot_id=robot_id,
        xyz_world=target_world,
        xyz_base=target_base,
        q_deg=q_deg,
        q_rad=q_rad,
        error_mm=error_mm,
        # Stage 1 returns one selected joint pose.  The underlying solver may
        # evaluate many seeds to find that pose, but those candidates are not
        # separate pick/place poses for the user to choose from.
        solution_count=1,
    )


def robots_that_reach(
    target_xyz_world,
    robots: Dict[str, object],
    current_q_deg: Optional[Dict[str, object]] = None,
) -> Dict[str, Stage1IKResult]:
    """Return every robot whose workspace can reach this world XYZ.

    A robot is considered to own the point when Stage-1 IK succeeds.
    Intersection = more than one key in the returned dict.
    """
    current_q_deg = current_q_deg or {}
    reachable = {}
    for robot_id, config in robots.items():
        solution = solve_stage1_ik(
            robot_id,
            target_xyz_world,
            config,
            current_q_deg=current_q_deg.get(robot_id),
        )
        if solution is not None:
            reachable[robot_id] = solution
    return reachable
