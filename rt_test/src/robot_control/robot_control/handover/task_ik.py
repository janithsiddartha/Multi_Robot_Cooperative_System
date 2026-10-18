import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from robot_control.handover.geometry import euler_to_matrix
from robot_control.handover.handover_ik import (
    ik_pose_multiple, 
    validate_pose_solution,
    rotation_matrix_to_quaternion
)
from robot_control.handover.object_pick_ik import (
    solve_object_pick_ik,
    ObjectPickIKResult,
    generate_pick_orientations,
    parse_object_pick_target
)

@dataclass
class TaskIKCandidate:
    robot_id: str
    position: np.ndarray
    rotation: np.ndarray
    quaternion: np.ndarray
    joints: np.ndarray
    position_error: float
    orientation_error: float
    joint_limit_margin: float
    sigma_min: float
    approach_direction: str
    yaw: float
    tilt: float
    cost: float = 0.0

    @property
    def joints_deg(self) -> np.ndarray:
        return np.rad2deg(self.joints)

def _construct_base_rotation(z_hat: np.ndarray) -> np.ndarray:
    z_hat = z_hat / np.linalg.norm(z_hat)
    # Choose 'a' not parallel to z_hat
    a = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(a, z_hat)) > 0.99:
        a = np.array([0.0, 1.0, 0.0])
    
    x_prime = a - np.dot(a, z_hat) * z_hat
    x_hat = x_prime / np.linalg.norm(x_prime)
    y_hat = np.cross(z_hat, x_hat)
    
    R = np.column_stack((x_hat, y_hat, z_hat))
    return R

def generate_orientation_candidates(
    approaches: List[Tuple[float, float, float]],
    yaw_angles_deg: List[float],
    tilt_angles_deg: List[float]
) -> List[dict]:
    candidates = []
    
    if not tilt_angles_deg:
        tilt_angles_deg = [0.0]
        
    for approach in approaches:
        z_hat = np.array(approach, dtype=float)
        try:
            R_base = _construct_base_rotation(z_hat)
        except ValueError:
            continue
            
        for yaw in yaw_angles_deg:
            for tilt in tilt_angles_deg:
                # Local rotations: tilt around X, yaw around Z
                # R = R_base * R_x(tilt) * R_z(yaw)
                R_tilt = euler_to_matrix([np.deg2rad(tilt), 0.0, 0.0])
                R_yaw = euler_to_matrix([0.0, 0.0, np.deg2rad(yaw)])
                R_final = R_base @ R_tilt @ R_yaw
                
                candidates.append({
                    'rotation': R_final,
                    'approach_direction': str(approach),
                    'yaw': yaw,
                    'tilt': tilt
                })
                
    return candidates

def solve_task_ik(
    robot_id: str,
    target_position: np.ndarray,
    config,
    approaches: List[Tuple[float, float, float]] = None,
    yaw_angles_deg: List[float] = None,
    tilt_angles_deg: List[float] = None,
    pos_tolerance: float = 1.0,
    ori_tolerance_deg: float = 15.0
) -> List[TaskIKCandidate]:
    
    h_cfg = getattr(config, 'handover_cfg', None) or {}
    auto_ori = h_cfg.get('auto_orientation', {})
    pose_tols = h_cfg.get('pose_tolerances', {})

    if approaches is None:
        cfg_appr = auto_ori.get('approach_directions')
        if cfg_appr:
            approaches = [tuple(v) for v in cfg_appr]
        else:
            approaches = [
                (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)
            ]
    if yaw_angles_deg is None:
        cfg_yaw = auto_ori.get('yaw_angles_deg')
        if cfg_yaw:
            yaw_angles_deg = [float(v) for v in cfg_yaw]
        else:
            yaw_angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    if tilt_angles_deg is None:
        cfg_tilt = auto_ori.get('tilt_angles_deg')
        if cfg_tilt:
            tilt_angles_deg = [float(v) for v in cfg_tilt]
        else:
            tilt_angles_deg = [0, 10, -10, 20, -20]

    if pos_tolerance == 1.0 and 'position_mm' in pose_tols:
        pos_tolerance = float(pose_tols['position_mm'])
    if ori_tolerance_deg == 15.0 and 'orientation_deg' in pose_tols:
        ori_tolerance_deg = float(pose_tols['orientation_deg'])
        
    ori_candidates = generate_orientation_candidates(approaches, yaw_angles_deg, tilt_angles_deg)
    
    valid_candidates = []
    
    for ori in ori_candidates:
        R_d = ori['rotation']
        
        # We reuse the ik_pose_multiple seeds generator, but we extract all valid ones manually
        # to prevent averaging or dropping solutions
        
        q_min = config.joint_limits_rad[:, 0]
        q_max = config.joint_limits_rad[:, 1]
        
        seeds = []
        if hasattr(config, "home_deg"):
            seeds.append(np.deg2rad(np.asarray(config.home_deg, dtype=float)))
        seeds.append(0.5 * (q_min + q_max))
        seeds.append(q_min + 0.1 * (q_max - q_min))
        seeds.append(q_max - 0.1 * (q_max - q_min))
        
        rng = np.random.RandomState(42)
        for _ in range(6):
            seeds.append(q_min + (q_max - q_min) * rng.rand(config.num_joints))
            
        from robot_control.handover.handover_ik import ik_pose_single
        
        for q_init in seeds:
            res = ik_pose_single(
                target_position, R_d, config, q_init,
                pos_tolerance=pos_tolerance, 
                ori_tolerance_deg=ori_tolerance_deg
            )
            
            if res["success"]:
                q_sol = res["q"]
                is_valid, p_err, o_err, margin, sigma = validate_pose_solution(
                    q_sol, target_position, R_d, config,
                    pos_tolerance=pos_tolerance, ori_tolerance_deg=ori_tolerance_deg,
                    min_margin_deg=1.0, min_manipulability=1e-4
                )
                
                if is_valid:
                    # Prefer picking from above (-Z approach direction)
                    approach_cost = 0.0
                    if ori['approach_direction'] != str((0.0, 0.0, -1.0)):
                        approach_cost = 50.0  # Penalty for not picking from above
                        
                    # Deduplicate
                    is_duplicate = False
                    for sol in valid_candidates:
                        if np.allclose(sol.joints, q_sol, atol=1e-3):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        valid_candidates.append(TaskIKCandidate(
                            robot_id=robot_id,
                            position=target_position,
                            rotation=R_d,
                            quaternion=rotation_matrix_to_quaternion(R_d),
                            joints=q_sol,
                            position_error=p_err,
                            orientation_error=o_err,
                            joint_limit_margin=margin,
                            sigma_min=sigma,
                            approach_direction=ori['approach_direction'],
                            yaw=ori['yaw'],
                            tilt=ori['tilt'],
                            cost=p_err * 100 + o_err * 10 - margin + approach_cost - sigma
                        ))
                        
    # Sort candidates by cost (lowest first) so we prefer -Z approach and good margins
    valid_candidates.sort(key=lambda c: c.cost)
    return valid_candidates
