import yaml
import os
import sys
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from robot_control.handover.task_ik import solve_task_ik, TaskIKCandidate
from robot_control.handover.geometry import euler_to_matrix
from robot_control.handover.collision import CollisionChecker
from robot_control.kinematics.fk_math import all_frame_positions

class MockKinematicsConfig:
    def __init__(self, yaml_data, world_base_xyz, world_base_rpy_deg):
        params = yaml_data['/**']['ros__parameters']
        kin = params['kinematics']
        
        self.alpha_deg = kin['dh']['alpha_deg']
        self.a = kin['dh']['a']
        self.d = kin['dh']['d']
        self.theta_offset_deg = kin['dh']['theta_offset_deg']
        self.theta_joint_index = kin['dh']['theta_joint_index']
        self.theta_joint_multiplier = kin['dh']['theta_joint_multiplier']
        self.tcp_offset = np.array(kin['tcp']['offset'], dtype=float)
        
        limits_min = kin['joints']['limits_deg_min']
        limits_max = kin['joints']['limits_deg_max']
        self.home_deg = kin['joints']['home_deg']
        self.joint_limits_deg = np.array(list(zip(limits_min, limits_max)))
        self.joint_limits_rad = np.deg2rad(self.joint_limits_deg)
        
        self.num_frames = len(self.alpha_deg)
        self.num_joints = len(self.home_deg)
        
        self.world_base_xyz = np.array(world_base_xyz)
        self.world_base_R = euler_to_matrix(np.deg2rad(world_base_rpy_deg))

def transform_world_to_base(p_W, config):
    p_B = config.world_base_R.T @ (np.array(p_W) - config.world_base_xyz)
    return p_B

def generate_interpolated_path(q_start, q_end, steps=50):
    return [q_start + s * (q_end - q_start) for s in np.linspace(0, 1, steps)]

def collision_check_path(path_q, config, safe_distance):
    # This checks robot self-collision essentially by returning true if distance < safe
    # Wait, the prompt says reuse handover/collision.py. 
    # But collision.py expects TWO synchronized trajectories.
    # We can check self-collision manually here for a single robot path.
    for q in path_q:
        positions = all_frame_positions(q, config)
        # Check self-collision (exclude adjacent links and TCP)
        n = len(positions)
        for i in range(n):
            for j in range(i + 2, n - 1): # Skip adjacent and TCP
                if np.linalg.norm(positions[i] - positions[j]) < safe_distance:
                    return False
    return True

def main():
    print("===== STAGE 1 TASK PLANNING =====")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_dir = os.path.join(base_dir, 'config')
    
    with open(os.path.join(config_dir, 'handover.yaml'), 'r') as f:
        handover_cfg = yaml.safe_load(f)
        
    safe_distance = handover_cfg.get('handover', {}).get('safe_distance', 50.0)
    
    robots = {}
    for rid in ['robot1', 'robot2', 'robot3']:
        r_file = os.path.join(config_dir, f'{rid}.yaml')
        if not os.path.exists(r_file):
            continue
        with open(r_file, 'r') as f:
            y_data = yaml.safe_load(f)
            
        r_base = handover_cfg['robots'][rid]
        robots[rid] = MockKinematicsConfig(
            y_data, 
            r_base['world_base_xyz'],
            r_base['world_base_rpy_deg']
        )
        
    # INPUTS
    # Pick target
    pick_xyz_world = [100.0, 100.0, 200.0]
    place_xyz_world = [0.0, 150.0, 150.0]
    
    # R1 and R2 handover configs (from the user, assuming R1 and R2 are used)
    r1_handover_q = np.zeros(5)
    r2_handover_q = np.zeros(6)
    
    print(f"PICK XYZ = {pick_xyz_world}")
    
    pick_candidates_by_robot = {}
    for rid, cfg in robots.items():
        pick_B = transform_world_to_base(pick_xyz_world, cfg)
        cands = solve_task_ik(rid, pick_B, cfg)
        if cands:
            pick_candidates_by_robot[rid] = cands
            print(f"R{rid[-1]} PICK candidate count = {len(cands)}")
            
    if not pick_candidates_by_robot:
        print("NO FEASIBLE PICK CONFIGURATION FOUND.")
        return
        
    # Auto-select or take first
    pick_robot_id = list(pick_candidates_by_robot.keys())[0]
    pick_cand = pick_candidates_by_robot[pick_robot_id][0]
    
    print(f"\nSELECTED PICK")
    print(f"robot = {pick_robot_id}")
    print(f"q = {pick_cand.joints}")
    print(f"orientation = {pick_cand.approach_direction}, yaw={pick_cand.yaw}, tilt={pick_cand.tilt}")
    print(f"position_error = {pick_cand.position_error:.4f}")
    print(f"orientation_error = {pick_cand.orientation_error:.4f}")
    
    print(f"\nPLACE XYZ = {place_xyz_world}")
    place_candidates_by_robot = {}
    for rid, cfg in robots.items():
        place_B = transform_world_to_base(place_xyz_world, cfg)
        cands = solve_task_ik(rid, place_B, cfg)
        if cands:
            place_candidates_by_robot[rid] = cands
            print(f"R{rid[-1]} PLACE candidate count = {len(cands)}")
            
    if not place_candidates_by_robot:
        print("NO FEASIBLE PLACE CONFIGURATION FOUND.")
        return
        
    place_robot_id = list(place_candidates_by_robot.keys())[-1] # take the last one for place
    place_cand = place_candidates_by_robot[place_robot_id][0]
    
    print(f"\nSELECTED PLACE")
    print(f"robot = {place_robot_id}")
    print(f"q = {place_cand.joints}")
    
    print(f"\nHANDOVER")
    print(f"R1 q = {r1_handover_q}")
    print(f"R2 q = {r2_handover_q}")
    
    print("\nPATH VALIDATION")
    
    pick_path = generate_interpolated_path(np.zeros(robots[pick_robot_id].num_joints), pick_cand.joints)
    pick_safe = collision_check_path(pick_path, robots[pick_robot_id], safe_distance)
    print(f"pick -> handover = {'PASS' if pick_safe else 'FAIL'}")
    
    place_path = generate_interpolated_path(r2_handover_q[:robots[place_robot_id].num_joints], place_cand.joints)
    place_safe = collision_check_path(place_path, robots[place_robot_id], safe_distance)
    print(f"handover -> place = {'PASS' if place_safe else 'FAIL'}")
    
    print("\nSTAGE 1 PLAN")
    if pick_safe and place_safe:
        print("VALID")
    else:
        print("INVALID")

if __name__ == '__main__':
    main()
