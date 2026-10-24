import numpy as np
import pytest
import os
import yaml
from robot_control.kinematics.fk_math import fk
from robot_control.handover.geometry import make_tcp_orientation, make_receiver_orientation, validate_handover_orientation
from robot_control.handover.trajectory import smoothstep7
from robot_control.handover.workspace import WorkspaceSampler
from robot_control.handover.planner import HandoverPlanner, euler_to_matrix
from robot_control.nodes.handover_planner_node import YamlKinematicsConfig

def test_workspace_sampler_concept():
    # TEST: Ensure position vs pose distinction
    bounds = {'x_min': 0.0, 'x_max': 100.0, 'y_min': 0.0, 'y_max': 100.0, 'z_min': 0.0, 'z_max': 100.0}
    sampler = WorkspaceSampler(bounds, position_step=50.0, orientation_step_deg=90.0)
    
    positions = list(sampler.get_position_candidates())
    orientations = list(sampler.get_orientation_candidates())
    
    assert len(positions) == 27  # 3x3x3 grid
    assert len(orientations) == 4  # 0, 90, 180, 270 degrees
    
    # Check that points are correctly generated
    assert np.array_equal(positions[0], [0.0, 0.0, 0.0])

def test_handover_geometry():
    phi = 0.5
    z_approach = np.array([0, 0, -1])
    
    R1 = make_tcp_orientation(z_approach, phi)
    
    # TEST: Rotation matrix validity
    assert np.allclose(R1.T @ R1, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R1), 1.0, atol=1e-6)
    
    for branch in [+1, -1]:
        R2 = make_receiver_orientation(R1, branch)
        
        # TEST: Rotation matrix validity
        assert np.allclose(R2.T @ R2, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R2), 1.0, atol=1e-6)
        
        z1 = R1[:, 2]
        z2 = R2[:, 2]
        x1 = R1[:, 0]
        x2 = R2[:, 0]
        y1 = R1[:, 1]
        y2 = R2[:, 1]
        
        # TEST: opposing Z axes
        assert np.isclose(np.dot(z1, z2), -1.0)
        
        # TEST: orthogonal transverse axes
        assert np.isclose(np.dot(x1, x2), 0.0)
        assert np.isclose(np.dot(y1, y2), 0.0)
        
        # Validate composite
        assert validate_handover_orientation(R1, R2)

def test_trajectory_conditions():
    q0 = np.array([0.0, 0.0, 0.0])
    qf = np.array([1.0, -1.0, 0.5])
    T = 2.0
    
    q_start, v_start, a_start, j_start = smoothstep7(q0, qf, 0.0, T)
    q_end, v_end, a_end, j_end = smoothstep7(q0, qf, T, T)
    
    # TEST: Boundary conditions
    assert np.allclose(v_start, 0)
    assert np.allclose(a_start, 0)
    assert np.allclose(j_start, 0)
    
    assert np.allclose(v_end, 0)
    assert np.allclose(a_end, 0)
    assert np.allclose(j_end, 0)
    
    assert np.allclose(q_start, q0)
    assert np.allclose(q_end, qf)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
