"""Stage-1 pick / place planning.

Flow:
  1. User gives pick XYZ and place XYZ in the world frame.
  2. Each point is tested against every robot with Stage-1 XYZ IK.
  3. One reachable robot → selected automatically.
     Several (workspace intersection) → caller must choose.
  4. Joint poses come from ``stage1_ik`` (same solver as ``ik_solver``).
  5. If pick robot != place robot, use ``manual_handover`` q from handover.yaml.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml

from robot_control.handover.geometry import euler_to_matrix
from robot_control.handover.stage1_ik import Stage1IKResult, robots_that_reach
from robot_control.kinematics.fk_math import all_frame_positions


class MockKinematicsConfig:
    def __init__(self, yaml_data, world_base_xyz, world_base_rpy_deg):
        params = yaml_data["/**"]["ros__parameters"]
        kin = params["kinematics"]

        self.alpha_deg = kin["dh"]["alpha_deg"]
        self.a = kin["dh"]["a"]
        self.d = kin["dh"]["d"]
        self.theta_offset_deg = kin["dh"]["theta_offset_deg"]
        self.theta_joint_index = kin["dh"]["theta_joint_index"]
        self.theta_joint_multiplier = kin["dh"]["theta_joint_multiplier"]
        self.tcp_offset = np.array(kin["tcp"]["offset"], dtype=float)

        limits_min = kin["joints"]["limits_deg_min"]
        limits_max = kin["joints"]["limits_deg_max"]
        self.home_deg = kin["joints"]["home_deg"]
        self.joint_limits_deg = np.array(list(zip(limits_min, limits_max)))
        self.joint_limits_rad = np.deg2rad(self.joint_limits_deg)

        self.num_frames = len(self.alpha_deg)
        self.num_joints = len(self.home_deg)

        self.world_base_xyz = np.array(world_base_xyz, dtype=float)
        self.world_base_R = euler_to_matrix(np.deg2rad(world_base_rpy_deg))


def default_config_dir():
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config")
    )


def load_stage1_robots(config_dir=None):
    config_dir = config_dir or default_config_dir()
    with open(os.path.join(config_dir, "handover.yaml"), "r", encoding="utf-8") as handle:
        handover_cfg = yaml.safe_load(handle) or {}

    robots = {}
    robot_yaml_paths = {}
    for rid in ("robot1", "robot2", "robot3"):
        path = os.path.join(config_dir, f"{rid}.yaml")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            yaml_data = yaml.safe_load(handle)
        base = handover_cfg["robots"][rid]
        robots[rid] = MockKinematicsConfig(
            yaml_data,
            base["world_base_xyz"],
            base["world_base_rpy_deg"],
        )
        robot_yaml_paths[rid] = path

    return handover_cfg, robots, robot_yaml_paths


def lookup_manual_handover(handover_cfg, robot_a, robot_b):
    """Return (q_a_deg, q_b_deg) from manual_handover, or None."""
    table = handover_cfg.get("manual_handover") or {}
    forward = table.get(f"{robot_a}_{robot_b}")
    reverse = table.get(f"{robot_b}_{robot_a}")
    entry = forward or reverse
    if not entry:
        return None
    q_a = entry.get(f"{robot_a}_q")
    q_b = entry.get(f"{robot_b}_q")
    if q_a is None or q_b is None:
        return None
    return np.array(q_a, dtype=float), np.array(q_b, dtype=float)


def generate_interpolated_path(q_start, q_end, steps=50):
    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)
    return [q_start + s * (q_end - q_start) for s in np.linspace(0, 1, steps)]


def collision_check_path(path_q, config, safe_distance):
    for q in path_q:
        positions = all_frame_positions(np.deg2rad(q), config)
        n = len(positions)
        for i in range(n):
            for j in range(i + 2, n - 1):
                if np.linalg.norm(positions[i] - positions[j]) < safe_distance:
                    return False
    return True


def select_robot(reachable, chooser, label):
    """Auto-select a single reachable robot, otherwise ask ``chooser``.

    ``chooser(label, robot_ids)`` must return one id from ``robot_ids``.
    """
    robot_ids = list(reachable.keys())
    if not robot_ids:
        return None
    if len(robot_ids) == 1:
        return robot_ids[0]
    chosen = chooser(label, robot_ids)
    if chosen not in reachable:
        return None
    return chosen


@dataclass
class Stage1Plan:
    pick_robot: str
    place_robot: str
    pick: Stage1IKResult
    place: Stage1IKResult
    needs_handover: bool
    pick_handover_q: Optional[np.ndarray]
    place_handover_q: Optional[np.ndarray]


def build_stage1_plan(
    pick_xyz,
    place_xyz,
    robots,
    handover_cfg,
    chooser,
    current_q_deg=None,
):
    """Resolve robots, IK poses, and optional handover q.

    ``chooser(label, robot_ids)`` is called only for workspace intersections.
    """
    pick_opts = robots_that_reach(pick_xyz, robots, current_q_deg)
    if not pick_opts:
        raise RuntimeError("No robot workspace contains the pick XYZ.")

    pick_rid = select_robot(pick_opts, chooser, "pick")
    if pick_rid is None:
        raise RuntimeError("Pick robot was not selected.")

    place_opts = robots_that_reach(place_xyz, robots, current_q_deg)
    if not place_opts:
        raise RuntimeError("No robot workspace contains the place XYZ.")

    place_rid = select_robot(place_opts, chooser, "place")
    if place_rid is None:
        raise RuntimeError("Place robot was not selected.")

    pick_h_q = None
    place_h_q = None
    needs_handover = pick_rid != place_rid
    if needs_handover:
        handover = lookup_manual_handover(handover_cfg, pick_rid, place_rid)
        if handover is None:
            raise RuntimeError(
                f"No manual_handover entry for {pick_rid} <-> {place_rid} in handover.yaml"
            )
        pick_h_q, place_h_q = handover

    return Stage1Plan(
        pick_robot=pick_rid,
        place_robot=place_rid,
        pick=pick_opts[pick_rid],
        place=place_opts[place_rid],
        needs_handover=needs_handover,
        pick_handover_q=pick_h_q,
        place_handover_q=place_h_q,
    ), pick_opts, place_opts


def _cli_chooser(label, robot_ids):
    print(f"\n{label.upper()} XYZ is in the workspace of: {', '.join(robot_ids)}")
    print("Select a robot:")
    for index, rid in enumerate(robot_ids, start=1):
        print(f"  {index}) {rid}")
    raw = input("Choice [1]: ").strip() or "1"
    try:
        return robot_ids[int(raw) - 1]
    except (ValueError, IndexError):
        return robot_ids[0]


def main():
    print("===== STAGE 1 TASK PLANNING =====")
    handover_cfg, robots, _ = load_stage1_robots()
    safe_distance = handover_cfg.get("handover", {}).get("safe_distance", 50.0)

    pick_xyz_world = [100.0, 100.0, 200.0]
    place_xyz_world = [0.0, 150.0, 150.0]
    print(f"PICK XYZ = {pick_xyz_world}")
    print(f"PLACE XYZ = {place_xyz_world}")

    try:
        plan, pick_opts, place_opts = build_stage1_plan(
            pick_xyz_world,
            place_xyz_world,
            robots,
            handover_cfg,
            chooser=_cli_chooser,
        )
    except RuntimeError as exc:
        print(str(exc))
        return

    print("\nReachable for PICK:")
    for rid, sol in pick_opts.items():
        print(f"  {rid}: reachable, one XYZ pose selected (error={sol.error_mm:.3f} mm)")
    print("\nSELECTED PICK")
    print(f"robot = {plan.pick_robot}")
    print(f"q_deg = {plan.pick.q_deg}")
    print(f"position_error = {plan.pick.error_mm:.4f} mm")

    print("\nReachable for PLACE:")
    for rid, sol in place_opts.items():
        print(f"  {rid}: reachable, one XYZ pose selected (error={sol.error_mm:.3f} mm)")
    print("\nSELECTED PLACE")
    print(f"robot = {plan.place_robot}")
    print(f"q_deg = {plan.place.q_deg}")
    print(f"position_error = {plan.place.error_mm:.4f} mm")

    if not plan.needs_handover:
        print("\nSame robot for pick and place — no handover.")
        start_q = np.zeros(robots[plan.pick_robot].num_joints)
        pick_path = generate_interpolated_path(start_q, plan.pick.q_deg)
        place_path = generate_interpolated_path(plan.pick.q_deg, plan.place.q_deg)
        pick_ok = collision_check_path(pick_path, robots[plan.pick_robot], safe_distance)
        place_ok = collision_check_path(place_path, robots[plan.place_robot], safe_distance)
        print(f"start -> pick = {'PASS' if pick_ok else 'FAIL'}")
        print(f"pick -> place = {'PASS' if place_ok else 'FAIL'}")
        print("\nSTAGE 1 PLAN")
        print("VALID" if pick_ok and place_ok else "INVALID")
        return

    print("\nHANDOVER")
    print(f"{plan.pick_robot} q = {plan.pick_handover_q}")
    print(f"{plan.place_robot} q = {plan.place_handover_q}")

    start_pick = np.zeros(robots[plan.pick_robot].num_joints)
    start_place = np.zeros(robots[plan.place_robot].num_joints)
    pick_path = generate_interpolated_path(start_pick, plan.pick.q_deg)
    pick_h_path = generate_interpolated_path(plan.pick.q_deg, plan.pick_handover_q)
    place_h_path = generate_interpolated_path(start_place, plan.place_handover_q)
    place_path = generate_interpolated_path(plan.place_handover_q, plan.place.q_deg)

    print("\nPATH VALIDATION")
    pick_ok = collision_check_path(pick_path, robots[plan.pick_robot], safe_distance)
    pick_h_ok = collision_check_path(pick_h_path, robots[plan.pick_robot], safe_distance)
    place_h_ok = collision_check_path(place_h_path, robots[plan.place_robot], safe_distance)
    place_ok = collision_check_path(place_path, robots[plan.place_robot], safe_distance)
    print(f"start -> pick = {'PASS' if pick_ok else 'FAIL'}")
    print(f"pick -> handover = {'PASS' if pick_h_ok else 'FAIL'}")
    print(f"start -> handover = {'PASS' if place_h_ok else 'FAIL'}")
    print(f"handover -> place = {'PASS' if place_ok else 'FAIL'}")
    print("\nSTAGE 1 PLAN")
    print("VALID" if pick_ok and pick_h_ok and place_h_ok and place_ok else "INVALID")


if __name__ == "__main__":
    main()
