import numpy as np
import yaml
import sys

from robot_control.kinematics.robot_kinematics import KinematicsConfig
from robot_control.kinematics.fk_math import fk, dh_joint_angles, mdh_matrix
from robot_control.handover.handover_ik import (
    geometric_jacobian_and_fk, ik_pose_multiple, ik_pose_single, pose_error, 
    singularity_metrics, joint_limit_cost, rotation_matrix_to_quaternion
)
from robot_control.kinematics.ik_solver import ik_single, numerical_jacobian
from robot_control.handover.geometry import euler_to_matrix, rotation_error
from robot_control.handover.static_map import StaticHandoverMap
from robot_control.handover.workspace import PositionWorkspaceComputer

class MockNode:
    def __init__(self, data, name): 
        self.data = data
        self.name = name
    def get_name(self): return self.name
    def get_parameter(self, name):
        class Param:
            def __init__(self, val): self.value = val
        keys = name.split('.')
        val = self.data
        try:
            for k in keys: 
                if k == 'kinematics' and 'robot' in val:
                    val = val['kinematics']
                else:
                    val = val[k]
        except KeyError: return Param(None)
        return Param(val)
    def declare_parameter(self, name, descriptor=None): pass

def load_handover_config():
    with open('/home/iit/rt_test/src/robot_control/config/handover.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_robot_config(robot_id):
    path = f'/home/iit/rt_test/src/robot_control/config/robot{robot_id}.yaml'
    with open(path, 'r') as f:
        data = yaml.safe_load(f)['/**']['ros__parameters']
    return KinematicsConfig(MockNode(data, f"robot{robot_id}")), data

def main():
    print("Loading configs...")
    handover_cfg = load_handover_config()
    configs = {}
    for i in (1, 2, 3):
        configs[i], _ = get_robot_config(i)

    def get_world_to_base(robot_id):
        info = handover_cfg.get("robots", {}).get(f"robot{robot_id}", {})
        xyz = np.asarray(info.get("world_base_xyz", [0.0, 0.0, 0.0]), dtype=float)
        rpy = np.deg2rad(info.get("world_base_rpy_deg", [0.0, 0.0, 0.0]))
        T = np.eye(4)
        T[:3, :3] = euler_to_matrix(rpy)
        T[:3, 3] = xyz
        return T

    print("\n=== 1. WORLD/BASE TRANSFORM TEST ===")
    np.random.seed(42)
    for rid in (1, 2, 3):
        T_WB = get_world_to_base(rid)
        R_WB = T_WB[:3, :3]
        base_world = T_WB[:3, 3]

        max_pos_err = 0.0
        max_rot_err = 0.0

        for _ in range(10):
            p_world = np.random.uniform(-500, 500, 3)
            # random rotation matrix
            q = np.random.randn(4)
            q /= np.linalg.norm(q)
            # using quaternion to R is easy, but we can just use euler
            rpy = np.random.uniform(-np.pi, np.pi, 3)
            R_world = euler_to_matrix(rpy)

            # To base
            p_base = R_WB.T @ (p_world - base_world)
            R_base = R_WB.T @ R_world

            # Reconstruct
            p_world_recon = R_WB @ p_base + base_world
            R_world_recon = R_WB @ R_base

            pos_err = np.linalg.norm(p_world - p_world_recon)
            rot_err = np.linalg.norm(R_world - R_world_recon)
            
            max_pos_err = max(max_pos_err, pos_err)
            max_rot_err = max(max_rot_err, rot_err)

        print(f"R{rid} Max Pos Err: {max_pos_err:.2e}, Max Rot Err: {max_rot_err:.2e}")

    print("\n=== 2. JACOBIAN VERIFICATION ===")
    for rid in (1, 2, 3):
        cfg = configs[rid]
        max_jv_err = 0.0
        max_jw_err = 0.0
        max_rel_err = 0.0
        
        for _ in range(10):
            q = np.random.uniform(cfg.joint_limits_rad[:, 0], cfg.joint_limits_rad[:, 1])
            J_ana, _, _ = geometric_jacobian_and_fk(q, cfg)
            
            # Numeric 6D J
            delta = 1e-6
            J_num = np.zeros((6, cfg.num_joints))
            p0, T0 = fk(q, cfg)
            R0 = T0[:3, :3]
            for i in range(cfg.num_joints):
                q_plus = q.copy()
                q_plus[i] += delta
                p_plus, T_plus = fk(q_plus, cfg)
                R_plus = T_plus[:3, :3]
                
                dp = (p_plus - p0) / delta
                
                # angular velocity
                R_rel = R_plus @ R0.T
                angle = np.arccos(np.clip((np.trace(R_rel) - 1)/2, -1, 1))
                if angle > 1e-8:
                    rx = R_rel[2,1] - R_rel[1,2]
                    ry = R_rel[0,2] - R_rel[2,0]
                    rz = R_rel[1,0] - R_rel[0,1]
                    axis = np.array([rx, ry, rz])
                    norm = np.linalg.norm(axis)
                    if norm > 1e-8:
                        axis = axis / norm
                        dw = axis * angle / delta
                    else: dw = np.zeros(3)
                else:
                    dw = np.zeros(3)
                
                J_num[:3, i] = dp
                J_num[3:, i] = dw
                
            err = np.abs(J_ana - J_num)
            max_jv_err = max(max_jv_err, np.max(err[:3, :]))
            max_jw_err = max(max_jw_err, np.max(err[3:, :]))
            rel_err = np.linalg.norm(J_ana - J_num) / (np.linalg.norm(J_ana) + 1e-10)
            max_rel_err = max(max_rel_err, rel_err)

        print(f"R{rid} Jacobian max Jv err: {max_jv_err:.2e}, Jw err: {max_jw_err:.2e}, rel err: {max_rel_err:.2e}")

    print("\n=== 3, 4, 6. EXACT PICKER-POSE TEST AND FAILURE BREAKDOWN ===")
    
    # We will instantiate a dummy map just for its helper functions
    # since StaticHandoverMap loads caches and does lots of initialization.
    # Instead, we just borrow the fixed_picker_orientation and voxel grids.
    
    # Let's mock a StaticHandoverMap minimal instance
    s_map = StaticHandoverMap(handover_cfg, configs)
    
    # We must run the position workspace compute first to get voxels
    ws_comp = PositionWorkspaceComputer(handover_cfg, configs)
    ws_comp.compute()
    
    def test_voxels(robot_a, robot_b):
        common = ws_comp.get_common_voxel_centers(robot_a, robot_b)
        R_world = s_map.fixed_picker_orientation()
        
        cfg_a = configs[robot_a]
        T_WB = get_world_to_base(robot_a)
        R_WB = T_WB[:3, :3]
        
        target_base_rot = R_WB.T @ R_world
        
        pos_reachable = 0
        full_reachable = 0
        
        breakdown = {
            "no_seed_converged": 0,
            "pos_error_high": 0,
            "ori_error_high": 0,
            "joint_limit": 0,
            "joint_margin": 0,
            "singularity": 0,
            "numerical_fail": 0
        }
        
        printed_details = 0
        
        for p_world in common:
            target_base_pos = (np.linalg.inv(T_WB) @ np.r_[p_world, 1.0])[:3]
            
            # Test position IK
            pos_ik_success = False
            for seed in s_map._ik_seeds(cfg_a):
                res = ik_single(target_base_pos, cfg_a, q_init=seed, max_iterations=200, tolerance=5.0)
                if res.get("success"):
                    pos_ik_success = True
                    break
            
            if pos_ik_success: pos_reachable += 1
            
            # Test full pose IK
            res = ik_pose_multiple(
                target_base_pos, target_base_rot, cfg_a, 
                pos_tolerance=1.0, ori_tolerance_deg=15.0, 
                min_margin_deg=1.0, min_manipulability=0.01
            )
            
            if res is not None:
                full_reachable += 1
            else:
                # Classify failure
                # Run single seed to see why it fails
                fail_reason = "no_seed_converged"
                best_single = ik_pose_single(target_base_pos, target_base_rot, cfg_a, s_map._ik_seeds(cfg_a)[0])
                p_err = best_single["pos_error"]
                o_err = np.rad2deg(best_single["ori_error"])
                
                if not best_single["success"]:
                    if p_err > 1.0: fail_reason = "pos_error_high"
                    elif o_err > 15.0: fail_reason = "ori_error_high"
                    else: fail_reason = "numerical_fail"
                else:
                    # It succeeded but failed validation
                    q = best_single["q"]
                    J, _, _ = geometric_jacobian_and_fk(q, cfg_a)
                    sig, _ = singularity_metrics(J)
                    
                    q_min = cfg_a.joint_limits_rad[:, 0]
                    q_max = cfg_a.joint_limits_rad[:, 1]
                    margin = np.rad2deg(np.min(np.minimum(q - q_min, q_max - q)))
                    
                    if np.any(q < q_min) or np.any(q > q_max): fail_reason = "joint_limit"
                    elif margin < 1.0: fail_reason = "joint_margin"
                    elif sig < 0.01: fail_reason = "singularity"
                    else: fail_reason = "validation_mismatch"
                
                breakdown[fail_reason] += 1
                
                # Print details for first few
                if printed_details < 3:
                    print(f"\n[Detail] R{robot_a} Failed Voxel: p_world={p_world}")
                    print(f"         p_base={target_base_pos}")
                    print(f"         R_world=\n{R_world}")
                    print(f"         R_base=\n{target_base_rot}")
                    print(f"         best pos_err={p_err:.2f}mm, ori_err={o_err:.2f}deg, sigma={sig if 'sig' in locals() else 0:.4f}")
                    printed_details += 1

        print(f"\nR{robot_a} (Pair R{robot_a}-R{robot_b})")
        print(f"  Total common voxels: {len(common)}")
        print(f"  Position reachable: {pos_reachable}")
        print(f"  Full-pose reachable: {full_reachable}")
        print(f"  Failure breakdown: {breakdown}")

    test_voxels(1, 2)
    test_voxels(2, 3)

    print("\n=== 5. KNOWN VALID POSE TEST ===")
    for rid in (1, 2, 3):
        cfg = configs[rid]
        q_known = np.zeros(cfg.num_joints)
        if cfg.num_joints >= 3:
            q_known[1] = np.deg2rad(45.0)
            q_known[2] = np.deg2rad(-45.0)
            
        p_base, T_base = fk(q_known, cfg)
        R_base = T_base[:3, :3]
        
        # To world
        T_WB = get_world_to_base(rid)
        p_world = T_WB[:3, :3] @ p_base + T_WB[:3, 3]
        R_world = T_WB[:3, :3] @ R_base
        
        # Through map logic
        target_base_pos = (np.linalg.inv(T_WB) @ np.r_[p_world, 1.0])[:3]
        target_base_rot = T_WB[:3, :3].T @ R_world
        
        res = ik_pose_multiple(
            target_base_pos, target_base_rot, cfg, 
            seeds=[q_known], # inject the known seed
            pos_tolerance=1.0, ori_tolerance_deg=15.0, 
            min_margin_deg=1.0, min_manipulability=0.01
        )
        
        if res is not None:
            print(f"R{rid}: Recovered. pos_err={res['position_error_mm']:.2e}, ori_err={res['orientation_error_deg']:.2e}, sigma={res['manipulability']:.2e}, margin={res['joint_limit_margin_deg']:.2f}deg")
        else:
            print(f"R{rid}: FAILED TO RECOVER KNOWN POSE!")

if __name__ == '__main__':
    main()
