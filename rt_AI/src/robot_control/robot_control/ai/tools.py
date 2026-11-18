"""Typed robotics tool interface for AI agents.

Directly bridges natural language intent and AI reasoning to verified kinematics,
Stage 3 Test 2 trajectory generation, and safety gates without modifying
any existing robotics libraries.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from robot_control.ai.world_model import WorldModel
from robot_control.ai.robot_agent import Robot1Agent, Robot2Agent, Robot3Agent, RobotLogicalAgent
from robot_control.handover.stage1 import load_stage1_robots, lookup_manual_handover_with_pre
from robot_control.handover.stage3_test2 import build_stage3_test2_plan, plan_stage3_test2_paths
from robot_control.handover.stage3_test2_solver import resolve_test2_handover_cache_path, load_test2_handover_cache
from robot_control.handover.trajectory_cache import compute_config_fingerprint
from robot_control.kinematics.fk_math import all_frame_positions, fk


class RoboticsTools:
    """Tool-calling library connecting AI Coordinator to robotics execution."""

    def __init__(self, world_model: Optional[WorldModel] = None):
        self.world_model = world_model or WorldModel()
        self.agents: Dict[str, RobotLogicalAgent] = {
            "robot1": Robot1Agent(),
            "robot2": Robot2Agent(),
            "robot3": Robot3Agent(),
        }
        # Cache robotics config
        self._handover_cfg = None
        self._robots_cfg = None

    def _ensure_configs(self):
        if self._handover_cfg is None or self._robots_cfg is None:
            self._handover_cfg, self._robots_cfg, _ = load_stage1_robots()

    def get_fleet_status(self) -> Dict[str, Any]:
        """Query live positions, joint angles, and connection states of all robots."""
        return self.world_model.snapshot()

    def check_reachability(
        self,
        robot_id: str,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        role: str = "pick",
    ) -> Dict[str, Any]:
        """Check if a specific robot can reach Cartesian target [x, y, z] mm."""
        rid = robot_id.lower().strip()
        if rid not in self.agents:
            return {
                "success": False,
                "reachable": False,
                "error": f"Unknown robot identifier '{robot_id}'. Valid arms: robot1, robot2, robot3.",
            }
        eval_res = self.agents[rid].evaluate_point([x, y, z], target_yaw_deg=yaw, role=role)
        return {
            "success": True,
            "robot_id": rid,
            "target_xyz": [x, y, z],
            **eval_res,
        }

    def rank_robots_for_point(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        role: str = "pick",
    ) -> Dict[str, Any]:
        """Rank all 3 robots for reachability and suitability to reach [x, y, z]."""
        results = {}
        reachable_arms = []
        for rid, agent in self.agents.items():
            res = agent.evaluate_point([x, y, z], target_yaw_deg=yaw, role=role)
            results[rid] = res
            if res.get("reachable"):
                reachable_arms.append((res.get("suitability_score", 0.0), rid))

        reachable_arms.sort(key=lambda x: x[0], reverse=True)
        ranked_order = [rid for _, rid in reachable_arms]
        best_robot = ranked_order[0] if ranked_order else None

        return {
            "target_xyz": [x, y, z],
            "role": role,
            "results": results,
            "ranked_order": ranked_order,
            "best_robot": best_robot,
            "is_reachable_by_any": len(ranked_order) > 0,
            "explanation": (
                f"Best candidate for {role.upper()} at [{x}, {y}, {z}] is {best_robot.upper()} "
                f"(score: {results[best_robot]['suitability_score']})"
                if best_robot
                else f"No robot can reach [{x}, {y}, {z}] mm."
            ),
        }

    def evaluate_handover_need(
        self,
        pick_xyz: List[float],
        place_xyz: List[float],
        pick_yaw: float = 0.0,
        place_yaw: float = 0.0,
    ) -> Dict[str, Any]:
        """Determine whether direct single-robot transfer is feasible or if handover is required."""
        pick_rank = self.rank_robots_for_point(pick_xyz[0], pick_xyz[1], pick_xyz[2], yaw=pick_yaw, role="pick")
        place_rank = self.rank_robots_for_point(place_xyz[0], place_xyz[1], place_xyz[2], yaw=place_yaw, role="place")

        pick_candidates = pick_rank["ranked_order"]
        place_candidates = place_rank["ranked_order"]

        if not pick_candidates:
            return {
                "feasible": False,
                "error": f"Pick target [{pick_xyz[0]}, {pick_xyz[1]}, {pick_xyz[2]}] is unreachable by all robots.",
            }
        if not place_candidates:
            return {
                "feasible": False,
                "error": f"Place target [{place_xyz[0]}, {place_xyz[1]}, {place_xyz[2]}] is unreachable by all robots.",
            }

        common_robots = set(pick_candidates).intersection(set(place_candidates))
        needs_handover = len(common_robots) == 0

        if not needs_handover:
            # Single robot can perform both pick and place!
            # Pick the robot with highest combined score
            best_single = max(
                common_robots,
                key=lambda rid: (
                    pick_rank["results"][rid]["suitability_score"]
                    + place_rank["results"][rid]["suitability_score"]
                ),
            )
            return {
                "feasible": True,
                "needs_handover": False,
                "single_robot": best_single,
                "pick_robot": best_single,
                "place_robot": best_single,
                "pick_candidates": pick_candidates,
                "place_candidates": place_candidates,
                "explanation": (
                    f"Direct Single-Robot execution possible: {best_single.upper()} can reach both "
                    f"Pick [{pick_xyz[0]}, {pick_xyz[1]}, {pick_xyz[2]}] and "
                    f"Place [{place_xyz[0]}, {place_xyz[1]}, {place_xyz[2]}]. No handover needed."
                ),
            }
        else:
            # Collaborative handover is mandatory
            best_pick = pick_candidates[0]
            best_place = place_candidates[0]
            pair_key = f"{best_pick}_{best_place}"

            return {
                "feasible": True,
                "needs_handover": True,
                "pick_robot": best_pick,
                "place_robot": best_place,
                "pair_key": pair_key,
                "pick_candidates": pick_candidates,
                "place_candidates": place_candidates,
                "explanation": (
                    f"Collaborative handover required: Pick target is owned by {best_pick.upper()} "
                    f"and Place target is owned by {best_place.upper()}. "
                    f"Mating corridor must be established between {best_pick.upper()} and {best_place.upper()}."
                ),
            }

    def plan_handover_task(
        self,
        pick_xyz: List[float],
        place_xyz: List[float],
        pick_yaw: float = 0.0,
        place_yaw: float = 0.0,
        preferred_pick_robot: Optional[str] = None,
        preferred_place_robot: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate verified 9-phase collaborative Stage 3 Test 2 trajectory."""
        self._ensure_configs()
        cache_path = resolve_test2_handover_cache_path()
        fp = compute_config_fingerprint(self._robots_cfg, self._handover_cfg)
        test_poses = load_test2_handover_cache(cache_path, fp)

        chooser = None
        if preferred_pick_robot or preferred_place_robot:
            def _custom_chooser(role: str, candidates: List[str]) -> str:
                if role == "pick" and preferred_pick_robot in candidates:
                    return preferred_pick_robot
                if role == "place" and preferred_place_robot in candidates:
                    return preferred_place_robot
                return candidates[0]
            chooser = _custom_chooser

        try:
            plan, pick_cands, place_cands = build_stage3_test2_plan(
                pick_xyz=pick_xyz,
                place_xyz=place_xyz,
                robots=self._robots_cfg,
                handover_cfg=self._handover_cfg,
                chooser=chooser,
                pick_yaw_deg=pick_yaw,
                place_yaw_deg=place_yaw,
                use_auto_handover=True,
                test_poses=test_poses,
            )

            traj_res = plan_stage3_test2_paths(
                plan=plan,
                robots=self._robots_cfg,
                steps=12,
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Trajectory planning failed: {str(e)}",
            }

        # Build trajectory steps for visualization and hardware playback
        phase_definitions = [
            (1, "Phase 1: Approach Pre-Pick (+50mm) & Cartesian Descent (-Z)", traj_res.phase1_simultaneous),
            (2, "Phase 2: Settle & Gripper Grasp (Dwell 3.5s)", None),
            (3, "Phase 3: Cartesian Lift (+50mm) & Transit to Pre-Handover", traj_res.phase3_pick_to_pre_handover),
            (4, "Phase 4: Synchronized Advance along Handover Corridors", traj_res.phase4_advance_to_handover),
            (5, "Phase 5: Gripper Handover Transfer (Dwell 4.0s)", None),
            (6, "Phase 6: Synchronized Retraction along Reverse Corridors", traj_res.phase6_retract_to_pre_handover),
            (7, "Phase 7: Transfer to Pre-Place (+50mm) & Cartesian Descent (-Z)", traj_res.phase7_handover_to_place),
            (8, "Phase 8: Settle & Gripper Release (Dwell 3.5s)", None),
            (9, "Phase 9: Cartesian Retract (+50mm) & Simultaneous Return Home", traj_res.phase9_simultaneous_home),
        ]

        pick_r = plan.pick_robot
        place_r = plan.place_robot
        current_joint_state = {
            r: [float(v) for v in self._robots_cfg[r].home_deg]
            for r in ["robot1", "robot2", "robot3"]
        }

        steps_data = []
        step_counter = 0

        for phase_num, phase_title, step_list in phase_definitions:
            if step_list is not None:
                for step_idx, step in enumerate(step_list):
                    current_joint_state[pick_r] = [round(float(v), 2) for v in step.pick_q]
                    current_joint_state[place_r] = [round(float(v), 2) for v in step.place_q]

                    if pick_r == place_r:
                        g_act = "close" if phase_num in (2, 3, 4, 5, 6, 7) else "open"
                        grippers = {"robot1": "open", "robot2": "open", "robot3": "open"}
                        grippers[pick_r] = g_act
                    else:
                        g_pick = "close" if 2 <= phase_num < 5 else "open"
                        g_place = "close" if 5 <= phase_num < 8 else "open"
                        grippers = {"robot1": "open", "robot2": "open", "robot3": "open"}
                        grippers[pick_r] = g_pick
                        grippers[place_r] = g_place

                    frames = {}
                    for rid, q_deg in current_joint_state.items():
                        cfg = self._robots_cfg[rid]
                        f_w = all_frame_positions(q_deg, cfg)
                        _, T_tcp = fk(np.deg2rad(q_deg), cfg)
                        R_tcp = cfg.world_base_R @ T_tcp[:3, :3] if hasattr(cfg, "world_base_R") else T_tcp[:3, :3]
                        frames[rid] = {
                            "frames_world": f_w.tolist(),
                            "tcp_world": [round(float(c), 1) for c in f_w[-1]],
                            "tcp_R": R_tcp.tolist(),
                        }

                    steps_data.append({
                        "step_index": step_counter,
                        "phase_number": phase_num,
                        "phase_title": phase_title,
                        "joint_state": {k: list(v) for k, v in current_joint_state.items()},
                        "grippers": grippers,
                        "frames": frames,
                        "clearance": {
                            "min_distance_mm": round(float(step.min_clearance_mm), 1),
                            "is_safe": bool(step.min_clearance_mm >= 45.0),
                        },
                    })
                    step_counter += 1
            else:
                if pick_r == place_r:
                    g_act = "close" if phase_num in (2, 3, 4, 5, 6, 7) else "open"
                    grippers = {"robot1": "open", "robot2": "open", "robot3": "open"}
                    grippers[pick_r] = g_act
                else:
                    g_pick = "close" if 2 <= phase_num < 5 else "open"
                    g_place = "close" if 5 <= phase_num < 8 else "open"
                    grippers = {"robot1": "open", "robot2": "open", "robot3": "open"}
                    grippers[pick_r] = g_pick
                    grippers[place_r] = g_place

                frames = {}
                for rid, q_deg in current_joint_state.items():
                    cfg = self._robots_cfg[rid]
                    f_w = all_frame_positions(q_deg, cfg)
                    _, T_tcp = fk(np.deg2rad(q_deg), cfg)
                    R_tcp = cfg.world_base_R @ T_tcp[:3, :3] if hasattr(cfg, "world_base_R") else T_tcp[:3, :3]
                    frames[rid] = {
                        "frames_world": f_w.tolist(),
                        "tcp_world": [round(float(c), 1) for c in f_w[-1]],
                        "tcp_R": R_tcp.tolist(),
                    }

                steps_data.append({
                    "step_index": step_counter,
                    "phase_number": phase_num,
                    "phase_title": phase_title,
                    "joint_state": {k: list(v) for k, v in current_joint_state.items()},
                    "grippers": grippers,
                    "frames": frames,
                    "clearance": {
                        "min_distance_mm": round(float(traj_res.min_overall_clearance_mm), 1),
                        "is_safe": bool(traj_res.min_overall_clearance_mm >= 45.0),
                    },
                })
                step_counter += 1

        summary = {
            "task_id": f"task_{int(np.random.randint(10000, 99999))}",
            "route": f"{plan.pick_robot.upper()} ➔ {plan.place_robot.upper()}",
            "pick_robot": plan.pick_robot,
            "place_robot": plan.place_robot,
            "needs_handover": plan.needs_handover,
            "approach_mode": plan.approach_mode,
            "total_steps": len(steps_data),
            "min_clearance_mm": round(float(traj_res.min_overall_clearance_mm), 1),
            "all_phases_safe": bool(traj_res.min_overall_clearance_mm >= 45.0),
            "handover_xyz": (
                [round(float(c), 1) for c in plan.auto_handover_info.handover_position_world]
                if plan.auto_handover_info
                else None
            ),
        }

        return {
            "success": True,
            "summary": summary,
            "plan": plan,
            "trajectory_steps": steps_data,
        }

    def return_all_home(self) -> Dict[str, Any]:
        """Dispatches return home command for all physical arms."""
        self._ensure_configs()
        home_joints = {rid: list(cfg.home_deg) for rid, cfg in self._robots_cfg.items()}
        self.world_model.update_robot_telemetry(joint_state=home_joints)
        return {
            "success": True,
            "action": "home_all",
            "home_joints": home_joints,
            "message": "All robots commanded to mechanical home posture [0, 0, 0, ...].",
        }

    def emergency_stop(self) -> Dict[str, Any]:
        """Immediate software halt."""
        self.world_model.set_emergency_stop(True)
        return {
            "success": True,
            "action": "emergency_stop",
            "message": "EMERGENCY STOP ASSERTED: All robot motions preempted and halted.",
        }
