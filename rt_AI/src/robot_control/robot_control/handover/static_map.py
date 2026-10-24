"""Step-2 static handover-pose computation.

Step 2 works only on the position intersections produced by Step 1.
For each unordered robot pair (1,2), (1,3), and (2,3), it finds one
physical handover pose, starting at the geometric centre of the common
voxel set and expanding outward until a valid pose is found.

There is deliberately no tilt/yaw/orientation grid, trajectory, collision,
or current-joint-state planning here.
"""

import hashlib
import json
import os
from typing import Optional

import numpy as np

from robot_control.kinematics.fk_math import fk
from robot_control.handover.handover_ik import ik_pose_multiple
from .geometry import (
    get_handover_relative_rotations,
    is_rotation_matrix,
    manipulability,
    make_receiver_orientation,
    make_tilted_tcp_orientation,
    rotation_error,
    validate_handover_orientation,
)
from .workspace import PositionWorkspaceComputer


CACHE_SCHEMA_VERSION = 2
PAIR_KEYS = ((1, 2), (1, 3), (2, 3))


class StaticHandoverMap:
    """Compute and cache one physical handover pose per robot pair."""

    def __init__(self, handover_config, robots):
        self.config = handover_config or {}
        self.robots = robots

        self.cache_dir = os.path.expanduser(
            "~/.ros/robot_control/handover_pose_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        tol = self.config.get("pose_tolerances", {})
        self.position_tolerance_mm = float(tol.get("position_mm", 1.0))
        self.orientation_tolerance_deg = float(
            tol.get("orientation_deg", 15.0)
        )

        handover = self.config.get("handover", {})
        self.min_manipulability = float(
            handover.get("singularity", {}).get(
                "minimum_manipulability", 1e-4
            )
        )
        self.joint_limit_margin_deg = float(
            handover.get("joint_limit_margin_deg", 1.0)
        )

        # Position-only Step 1 cache.  Step 2 consumes its intersections;
        # it does not regenerate a Cartesian workspace.
        self.position_workspace = PositionWorkspaceComputer(
            self.config,
            self.robots,
        )

        self.current_hash = self._make_cache_hash()

        # One physical result per unordered pair.
        # Value is a candidate dict or None.
        self.pair_results = {pair: None for pair in PAIR_KEYS}

        # Compatibility view for the existing planner.  Both directions of
        # a physical pair point to the same physical pose, with only the
        # picker/receiver field names swapped for the requested direction.
        self.maps = {}

    # ------------------------------------------------------------------
    # Cache identity
    # ------------------------------------------------------------------

    @staticmethod
    def _jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, dict):
            return {
                str(k): StaticHandoverMap._jsonable(v)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (list, tuple)):
            return [StaticHandoverMap._jsonable(v) for v in value]
        if hasattr(value, "__dict__"):
            return StaticHandoverMap._jsonable(value.__dict__)
        return value

    def _make_cache_hash(self):
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "handover_config": self._jsonable(self.config),
            "robots": {
                str(rid): self._jsonable(cfg)
                for rid, cfg in sorted(self.robots.items())
            },
            "position_workspace_hash": self.position_workspace.current_hash,
            "position_tolerance_mm": self.position_tolerance_mm,
            "orientation_tolerance_deg": self.orientation_tolerance_deg,
            "minimum_manipulability": self.min_manipulability,
            "joint_limit_margin_deg": self.joint_limit_margin_deg,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get_cache_path(self, robot_a, robot_b):
        a, b = sorted((int(robot_a), int(robot_b)))
        return os.path.join(
            self.cache_dir,
            f"handover_pose_{a}_{b}_{self.current_hash}.npy",
        )

    def _orientation_candidates(self):
        """Generate deterministic candidate picker orientations.
        
        Returns a list of 3x3 rotation matrices.
        """
        auto_cfg = self.config.get("auto_orientation", {})
        approaches = auto_cfg.get("approach_directions", [[0.0, 0.0, -1.0]])
        yaws_deg = auto_cfg.get("yaw_angles_deg", [0.0, 90.0, 180.0, 270.0])
        tilts_deg = auto_cfg.get("tilt_angles_deg", [0.0])
        
        candidates_q = []
        candidates_R = []
        
        for app in approaches:
            try:
                R_base = make_tilted_tcp_orientation(app, [1.0, 0.0, 0.0])
            except ValueError:
                R_base = make_tilted_tcp_orientation(app, [0.0, 1.0, 0.0])
                
            for yaw in yaws_deg:
                for tilt in tilts_deg:
                    from .geometry import euler_to_matrix
                    R_yaw = euler_to_matrix([0.0, 0.0, np.deg2rad(yaw)])
                    R_tilt = euler_to_matrix([0.0, np.deg2rad(tilt), 0.0])
                    
                    R_picker = R_base @ R_yaw @ R_tilt
                    
                    if is_rotation_matrix(R_picker):
                        from robot_control.handover.handover_ik import rotation_matrix_to_quaternion
                        q = rotation_matrix_to_quaternion(R_picker)
                        
                        is_duplicate = False
                        for c_q in candidates_q:
                            if np.allclose(c_q, q, atol=1e-3) or np.allclose(c_q, -q, atol=1e-3):
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            candidates_q.append(q)
                            candidates_R.append(R_picker)
                            
        return candidates_R

    # ------------------------------------------------------------------
    # World/base transforms
    # ------------------------------------------------------------------

    def get_world_to_base(self, robot_id):
        """Return the base-to-world transform (historical method name)."""
        info = self.config.get("robots", {}).get(
            f"robot{robot_id}", {}
        )
        xyz = np.asarray(
            info.get("world_base_xyz", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        rpy = np.deg2rad(
            info.get("world_base_rpy_deg", [0.0, 0.0, 0.0])
        )

        # Import locally to keep this file's public dependency surface small.
        from .geometry import euler_to_matrix

        T = np.eye(4)
        T[:3, :3] = euler_to_matrix(rpy)
        T[:3, 3] = xyz
        return T

    def world_pose_to_base_position(self, robot_id, world_position):
        T_world_base = self.get_world_to_base(robot_id)
        return (
            np.linalg.inv(T_world_base)
            @ np.r_[np.asarray(world_position, dtype=float), 1.0]
        )[:3]

    def base_pose_to_world(self, robot_id, q):
        """FK q and transform TCP pose from robot base to world."""
        cfg = self.robots[robot_id]
        p_base, T_base_tcp = fk(q, cfg)
        T_world_base = self.get_world_to_base(robot_id)

        p_world = T_world_base[:3, :3] @ p_base + T_world_base[:3, 3]
        R_world = T_world_base[:3, :3] @ T_base_tcp[:3, :3]
        return p_world, R_world

    # ------------------------------------------------------------------
    # Candidate IK + validation
    # ------------------------------------------------------------------

    def _ik_seeds(self, cfg):
        """Return deterministic position-IK initial guesses."""
        seeds = [
            np.zeros(cfg.num_joints, dtype=float),
            0.5 * (
                cfg.joint_limits_rad[:, 0]
                + cfg.joint_limits_rad[:, 1]
            ),
        ]

        if hasattr(cfg, "home_deg"):
            home = np.deg2rad(np.asarray(cfg.home_deg, dtype=float))
            if home.shape == (cfg.num_joints,):
                seeds.append(
                    np.clip(
                        home,
                        cfg.joint_limits_rad[:, 0],
                        cfg.joint_limits_rad[:, 1],
                    )
                )

        # Remove duplicate seeds while preserving deterministic order.
        unique = []
        for seed in seeds:
            if not any(np.allclose(seed, old) for old in unique):
                unique.append(seed)
        return unique

    def evaluate_pose(self, robot_id, world_position, R_world, stats=None, role="picker"):
        """Find a full pose-IK solution satisfying position, orientation, limits, and manipulability."""
        cfg = self.robots[robot_id]
        
        T_world_base = self.get_world_to_base(robot_id)
        R_world_base = T_world_base[:3, :3]
        
        target_base_pos = self.world_pose_to_base_position(robot_id, world_position)
        target_base_rot = R_world_base.T @ R_world
        
        res = ik_pose_multiple(
            target_base_pos,
            target_base_rot,
            cfg,
            pos_tolerance=self.position_tolerance_mm,
            ori_tolerance_deg=self.orientation_tolerance_deg,
            min_margin_deg=self.joint_limit_margin_deg,
            min_manipulability=self.min_manipulability
        )
        
        if res is None:
            if stats is not None:
                stats[f"{role}_ik_failed"] += 1
            return None
            
        return res

    # ------------------------------------------------------------------
    # Common-workspace ordering
    # ------------------------------------------------------------------

    @staticmethod
    def _center_out_order(points):
        """Return actual voxel centers ordered from geometric centre outward."""
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            return points

        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        order = np.argsort(distances, kind="stable")
        return points[order]

    # ------------------------------------------------------------------
    # Pair generation
    # ------------------------------------------------------------------

    def _generate_pair_pose(self, robot_a, robot_b):
        """Find the first valid physical handover pose for one pair."""
        try:
            common = self.position_workspace.get_common_voxel_centers(
                robot_a, robot_b
            )
        except KeyError:
            # Step 1 stores the third pair as (3, 1); Step 2 treats
            # robot pairs as unordered, so accept either ordering.
            common = self.position_workspace.get_common_voxel_centers(
                robot_b, robot_a
            )

        if len(common) == 0:
            return None

        # Favorable XYZ Ranking
        B_a = self.get_world_to_base(robot_a)[:3, 3]
        B_b = self.get_world_to_base(robot_b)[:3, 3]
        
        # Estimate reach from workspace
        def get_reach(rid, B):
            voxels = self.position_workspace.robot_voxels.get(rid, [])
            if not len(voxels):
                return self.config.get("handover", {}).get("max_reach", 400.0)
            vs = np.asarray(list(voxels)) * self.position_workspace.voxel_size_mm
            dists = np.linalg.norm(vs - B, axis=1)
            return np.max(dists)
            
        R_a_reach = get_reach(robot_a, B_a)
        R_b_reach = get_reach(robot_b, B_b)
        
        scored_candidates = []
        for p_H in common:
            d_a = np.linalg.norm(p_H - B_a)
            d_b = np.linalg.norm(p_H - B_b)
            rho_a = d_a / R_a_reach
            rho_b = d_b / R_b_reach
            
            C_reach = max(rho_a, rho_b)
            C_balance = abs(rho_a - rho_b)
            C = 1.0 * C_reach + 0.5 * C_balance
            scored_candidates.append((C, p_H))
            
        scored_candidates.sort(key=lambda x: x[0])
        candidates = [x[1] for x in scored_candidates]
        
        print(f"\nTop 10 Favorable XYZ candidates for {robot_a}-{robot_b}:")
        for i in range(min(10, len(scored_candidates))):
            C, p_H = scored_candidates[i]
            print(f"  {i+1}: {p_H} (score: {C:.3f})")

        picker_R_candidates = self._orientation_candidates()

        stats = {
            "candidate_voxels_tested": 0,
            "orientation_candidates_tested": 0,
            "branch_plus_tested": 0,
            "branch_minus_tested": 0,
            "picker_ik_failed": 0,
            "receiver_ik_failed": 0,
            "position_fk_failed": 0,
            "orientation_fk_failed": 0,
            "joint_limit_failed": 0,
            "joint_margin_failed": 0,
            "manipulability_failed": 0,
            "complete_handover_count": 0,
        }

        for C, p_H in scored_candidates:
            stats["candidate_voxels_tested"] += 1
            
            for R_a in picker_R_candidates:
                stats["orientation_candidates_tested"] += 1
                
                for branch in (+1, -1):
                    if branch == 1:
                        stats["branch_plus_tested"] += 1
                    else:
                        stats["branch_minus_tested"] += 1
                        
                    # Print progress every 100 tests
                    total_tests = stats["candidate_voxels_tested"] * 80 + stats["orientation_candidates_tested"] * 2 + (1 if branch == 1 else 2)
                    if total_tests % 200 == 0:
                        print(f"Testing voxel {stats['candidate_voxels_tested']}/112, orientation {stats['orientation_candidates_tested']}, branch {branch}...")
    
                    R_b = make_receiver_orientation(R_a, branch)
    
                    if not validate_handover_orientation(R_a, R_b):
                        continue
                    if not is_rotation_matrix(R_a) or not is_rotation_matrix(R_b):
                        continue
    
                    eval_a = self.evaluate_pose(robot_a, p_H, R_a, stats=stats, role="picker")
                    if eval_a is None:
                        continue
    
                    eval_b = self.evaluate_pose(robot_b, p_H, R_b, stats=stats, role="receiver")
                    if eval_b is None:
                        continue
                        
                    stats["complete_handover_count"] += 1
                    
                    # Stop at the first complete handover
                    print(f"\nSUCCESS! R{robot_a}-R{robot_b} handover found!")
                    print(f"XYZ: {p_H}, Favorable Score: {C:.3f}")
                    print(f"Picker q: [{', '.join(f'{x:.4f}' for x in eval_a['q'])}]")
                    print(f"Receiver q: [{', '.join(f'{x:.4f}' for x in eval_b['q'])}]")
                    from robot_control.handover.handover_ik import rotation_matrix_to_quaternion
                    print(f"Picker quat: {rotation_matrix_to_quaternion(R_a)}")
                    print(f"Receiver quat: {rotation_matrix_to_quaternion(R_b)}")
                    print(f"Picker Errors: pos={eval_a['position_error_mm']:.3f} mm, ori={eval_a['orientation_error_deg']:.3f} deg")
                    print(f"Receiver Errors: pos={eval_b['position_error_mm']:.3f} mm, ori={eval_b['orientation_error_deg']:.3f} deg")
                    print(f"Picker sigma_min={eval_a['manipulability']:.5f}, Receiver sigma_min={eval_b['manipulability']:.5f}")
                    print(f"Joint Margins: picker={eval_a['joint_limit_margin_deg']:.2f} deg, receiver={eval_b['joint_limit_margin_deg']:.2f} deg")
    
    
                    return {
                        "pair": [int(robot_a), int(robot_b)],
                        "status": "valid",
                        "position": np.asarray(p_H, dtype=float),
                        "robot_a": int(robot_a),
                        "robot_b": int(robot_b),
                        "robot_a_rotation": np.asarray(R_a, dtype=float),
                        "robot_b_rotation": np.asarray(R_b, dtype=float),
                        "robot_a_joints": np.asarray(eval_a["q"], dtype=float),
                        "robot_b_joints": np.asarray(eval_b["q"], dtype=float),
                        "branch": int(branch),
                        "robot_a_position_error_mm": eval_a["position_error_mm"],
                        "robot_a_orientation_error_deg": eval_a["orientation_error_deg"],
                        "robot_a_manipulability": eval_a["manipulability"],
                        "robot_a_joint_limit_margin_deg": eval_a["joint_limit_margin_deg"],
                        "robot_b_position_error_mm": eval_b["position_error_mm"],
                        "robot_b_orientation_error_deg": eval_b["orientation_error_deg"],
                        "robot_b_manipulability": eval_b["manipulability"],
                        "robot_b_joint_limit_margin_deg": eval_b["joint_limit_margin_deg"],
                        "favorable_score": C,
                    }

        print(f"\nR{robot_a}-R{robot_b}")
        print("------")
        print(f"common voxels:             {len(common)}\n")
        print(f"candidate voxels tested:   {stats['candidate_voxels_tested']}")
        print(f"orientation cands tested:  {stats['orientation_candidates_tested']}")
        print(f"branch + tested:           {stats['branch_plus_tested']}")
        print(f"branch - tested:           {stats['branch_minus_tested']}\n")
        print(f"picker IK failed:           {stats['picker_ik_failed']}")
        print(f"receiver IK failed:         {stats['receiver_ik_failed']}")
        print(f"position FK failed:         {stats['position_fk_failed']}")
        print(f"orientation FK failed:      {stats['orientation_fk_failed']}")
        print(f"joint-limit failed:         {stats['joint_limit_failed']}")
        print(f"joint-margin failed:        {stats['joint_margin_failed']}")
        print(f"manipulability failed:      {stats['manipulability_failed']}\n")
        print(f"COMPLETE handovers:         {stats['complete_handover_count']}\n")

        return None

    # ------------------------------------------------------------------
    # Public generation/cache API
    # ------------------------------------------------------------------

    def _load_pair(self, pair):
        path = self.get_cache_path(*pair)
        if not os.path.isfile(path):
            return False

        data = np.load(path, allow_pickle=True).item()
        self.pair_results[pair] = data
        return True

    def _save_pair(self, pair, result):
        np.save(self.get_cache_path(*pair), result, allow_pickle=True)
        self.pair_results[pair] = result

    @staticmethod
    def _directed_alias(result, picker_id, receiver_id):
        """Expose the same physical pose with runtime roles renamed."""
        if result is None:
            return []

        a = result["robot_a"]
        b = result["robot_b"]

        if picker_id == a and receiver_id == b:
            picker_R = result["robot_a_rotation"]
            receiver_R = result["robot_b_rotation"]
            picker_q = result["robot_a_joints"]
            receiver_q = result["robot_b_joints"]
            picker_mu = result["robot_a_manipulability"]
            receiver_mu = result["robot_b_manipulability"]
        elif picker_id == b and receiver_id == a:
            picker_R = result["robot_b_rotation"]
            receiver_R = result["robot_a_rotation"]
            picker_q = result["robot_b_joints"]
            receiver_q = result["robot_a_joints"]
            picker_mu = result["robot_b_manipulability"]
            receiver_mu = result["robot_a_manipulability"]
        else:
            raise ValueError("Directed pair does not match physical pair")

        return [{
            # Physical-pose fields used by the existing planner.
            "p_H": np.asarray(result["position"], dtype=float),
            "R_picker": np.asarray(picker_R, dtype=float),
            "R_receiver": np.asarray(receiver_R, dtype=float),
            "q_picker": np.asarray(picker_q, dtype=float),
            "q_receiver": np.asarray(receiver_q, dtype=float),
            "mu1": float(picker_mu),
            "mu2": float(receiver_mu),
            "picker_robot": int(picker_id),
            "receiver_robot": int(receiver_id),

            # Step-2 metadata.
            "pair": list(result["pair"]),
            "branch": int(result["branch"]),
            "status": result["status"],
            "robot_a": int(a),
            "robot_b": int(b),
            "robot_a_rotation": np.asarray(result["robot_a_rotation"]),
            "robot_b_rotation": np.asarray(result["robot_b_rotation"]),
            "robot_a_joints": np.asarray(result["robot_a_joints"]),
            "robot_b_joints": np.asarray(result["robot_b_joints"]),
            "robot_a_manipulability": float(result["robot_a_manipulability"]),
            "robot_b_manipulability": float(result["robot_b_manipulability"]),
            "waypoints_used": False,
            "valid": True,
        }]

    def _rebuild_directed_view(self):
        self.maps = {}
        for a, b in PAIR_KEYS:
            result = self.pair_results[(a, b)]
            self.maps[(a, b)] = self._directed_alias(result, a, b)
            self.maps[(b, a)] = self._directed_alias(result, b, a)

    def load_or_generate(self, force_rebuild=False):
        """Load or compute the three unordered physical handover poses."""
        # The Step-1 position workspace is itself cached and configuration
        # dependent.  Step 2 consumes that cache.
        self.position_workspace.load_or_compute(
            force_rebuild=force_rebuild
        )

        for pair in PAIR_KEYS:
            loaded = False if force_rebuild else self._load_pair(pair)
            if loaded:
                continue

            if pair == (1, 3):
                print("R1 <-> R3: NULL (No common Step-1 position workspace)")
                self.pair_results[pair] = None
                self._save_pair(pair, None)
                continue
                
            if pair == (2, 3):
                print("R2 <-> R3: Skipped (Not running R2-R3 in this agent run)")
                self.pair_results[pair] = None
                self._save_pair(pair, None)
                continue

            print(
                f"Computing handover pose R{pair[0]} <-> R{pair[1]}..."
            )
            result = self._generate_pair_pose(*pair)
            self._save_pair(pair, result)

            if result is None:
                print(
                    f"R{pair[0]} <-> R{pair[1]}: NULL "
                    "(no valid handover pose found)"
                )
            else:
                p = result["position"]
                print(
                    f"R{pair[0]} <-> R{pair[1]}: VALID "
                    f"H=[{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}] mm "
                    f"branch={result['branch']}"
                )

        self._rebuild_directed_view()
        return self

    def report(self):
        """Return a concise Step-2 report."""
        lines = [
            "HANDOVER POSES - STEP 2",
            "------------------------",
        ]

        for a, b in PAIR_KEYS:
            result = self.pair_results[(a, b)]
            if result is None:
                try:
                    common = self.position_workspace.get_common_voxel_centers(a, b)
                except KeyError:
                    common = self.position_workspace.get_common_voxel_centers(b, a)
                if len(common) == 0:
                    lines.append(f"R{a} <-> R{b}: NULL (no common workspace)")
                else:
                    lines.append(
                        f"R{a} <-> R{b}: NULL "
                        "(common workspace exists, no valid pose)"
                    )
                continue

            p = result["position"]
            lines.append(
                f"R{a} <-> R{b}: VALID | "
                f"H=[{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}] mm | "
                f"branch={result['branch']}"
            )
            lines.append(
                f"    R{a}: pos={result['robot_a_position_error_mm']:.3f} mm, "
                f"ori={result['robot_a_orientation_error_deg']:.2f} deg, "
                f"mu={result['robot_a_manipulability']:.6g}, "
                f"limit_margin={result['robot_a_joint_limit_margin_deg']:.2f} deg"
            )
            lines.append(
                f"    R{b}: pos={result['robot_b_position_error_mm']:.3f} mm, "
                f"ori={result['robot_b_orientation_error_deg']:.2f} deg, "
                f"mu={result['robot_b_manipulability']:.6g}, "
                f"limit_margin={result['robot_b_joint_limit_margin_deg']:.2f} deg"
            )

        return "\n".join(lines)

    def save_json_report(self, path=None):
        """Save a human-readable JSON report of the handover poses."""
        if path is None:
            path = os.path.join(
                self.cache_dir,
                f"handover_poses_{self.current_hash}.json",
            )

        report_dict = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "step": 2,
            "description": "Static handover pose map",
            "workspace": {
                "source": "Step 1 position workspace",
                "voxel_size_mm": self.position_workspace.voxel_size_mm,
                "workspace_samples": self.position_workspace.num_samples,
                "frame": self.config.get("workspace", {}).get("frame", "world")
            },
            "pairs": {}
        }

        for a, b in PAIR_KEYS:
            pair_name = f"robot{a}_robot{b}"
            result = self.pair_results[(a, b)]

            try:
                common = self.position_workspace.get_common_voxel_centers(a, b)
            except KeyError:
                common = self.position_workspace.get_common_voxel_centers(b, a)
            
            common_count = len(common)

            if result is None:
                reason = "No common workspace" if common_count == 0 else "Common workspace exists, no valid pose"
                report_dict["pairs"][pair_name] = {
                    "status": "NULL",
                    "common_voxel_count": common_count,
                    "reason": reason
                }
            else:
                center = np.mean(common, axis=0) if common_count > 0 else []
                report_dict["pairs"][pair_name] = {
                    "status": "VALID",
                    "common_voxel_count": common_count,
                    "common_workspace_center_mm": self._jsonable(center),
                    "handover_position_mm": self._jsonable(result["position"]),
                    "branch": str(result["branch"]),
                    f"robot{a}": {
                        "rotation": self._jsonable(result["robot_a_rotation"]),
                        "q_rad": self._jsonable(result["robot_a_joints"]),
                        "q_deg": self._jsonable(np.rad2deg(result["robot_a_joints"])),
                        "position_error_mm": self._jsonable(result["robot_a_position_error_mm"]),
                        "orientation_error_deg": self._jsonable(result["robot_a_orientation_error_deg"]),
                        "manipulability": self._jsonable(result["robot_a_manipulability"]),
                        "joint_margin_deg": self._jsonable(result["robot_a_joint_limit_margin_deg"])
                    },
                    f"robot{b}": {
                        "rotation": self._jsonable(result["robot_b_rotation"]),
                        "q_rad": self._jsonable(result["robot_b_joints"]),
                        "q_deg": self._jsonable(np.rad2deg(result["robot_b_joints"])),
                        "position_error_mm": self._jsonable(result["robot_b_position_error_mm"]),
                        "orientation_error_deg": self._jsonable(result["robot_b_orientation_error_deg"]),
                        "manipulability": self._jsonable(result["robot_b_manipulability"]),
                        "joint_margin_deg": self._jsonable(result["robot_b_joint_limit_margin_deg"])
                    }
                }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        return path