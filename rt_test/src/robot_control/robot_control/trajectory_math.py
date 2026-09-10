import numpy as np
from robot_control.ik_solver import ik

# ============================================================
# 7TH-ORDER SMOOTH TRAJECTORY
# ============================================================

def s7(tau): return 35.0 * tau**4 - 84.0 * tau**5 + 70.0 * tau**6 - 20.0 * tau**7
def s7_dot(tau): return 140.0 * tau**3 - 420.0 * tau**4 + 420.0 * tau**5 - 140.0 * tau**6
def s7_ddot(tau): return 420.0 * tau**2 - 1680.0 * tau**3 + 2100.0 * tau**4 - 840.0 * tau**5
def s7_dddot(tau): return 840.0 * tau - 5040.0 * tau**2 + 8400.0 * tau**3 - 4200.0 * tau**4

# ============================================================
# JOINT TRAJECTORY
# ============================================================

def joint_trajectory(q_start, q_target, duration=3.0, dt=0.01):
    q_start = np.asarray(q_start, dtype=float)
    q_target = np.asarray(q_target, dtype=float)
    time = np.arange(0.0, duration + dt, dt)

    q_all, qdot_all, qddot_all, qdddot_all = [], [], [], []
    dq = q_target - q_start

    for t in time:
        tau = np.clip(t / duration, 0.0, 1.0)
        s, sd, sdd, sddd = s7(tau), s7_dot(tau), s7_ddot(tau), s7_dddot(tau)

        q_all.append(q_start + dq * s)
        qdot_all.append(dq / duration * sd)
        qddot_all.append(dq / duration**2 * sdd)
        qdddot_all.append(dq / duration**3 * sddd)

    return (time, np.array(q_all), np.array(qdot_all), np.array(qddot_all), np.array(qdddot_all))

# ============================================================
# PICK-AND-PLACE TRAJECTORY
# ============================================================

def pick_and_place(pick_xyz, place_xyz, current_q=None, approach_height=50.0, segment_time=2.0, dt=0.01):
    pick_xyz = np.asarray(pick_xyz, dtype=float)
    place_xyz = np.asarray(place_xyz, dtype=float)

    pre_pick = pick_xyz.copy(); pre_pick[2] += approach_height
    lift = pre_pick.copy()
    pre_place = place_xyz.copy(); pre_place[2] += approach_height
    retreat = pre_place.copy()

    waypoints = [pre_pick, pick_xyz, lift, pre_place, place_xyz, retreat]
    q_previous = np.zeros(5) if current_q is None else np.asarray(current_q, dtype=float)
    joint_waypoints = [q_previous]

    for point in waypoints:
        result = ik(point, current_q=q_previous)
        if not result["success"]:
            raise RuntimeError(f"IK failed for waypoint {point}")
        q_solution = result["q"]
        joint_waypoints.append(q_solution)
        q_previous = q_solution

    complete_time, complete_q, complete_qdot, complete_qddot, complete_qdddot = [], [], [], [], []
    time_offset = 0.0

    for i in range(len(joint_waypoints) - 1):
        q_start, q_target = joint_waypoints[i], joint_waypoints[i + 1]
        t, q, qdot, qddot, qdddot = joint_trajectory(q_start, q_target, duration=segment_time, dt=dt)

        if i > 0:
            t, q, qdot, qddot, qdddot = t[1:], q[1:], qdot[1:], qddot[1:], qdddot[1:]

        t = t + time_offset
        complete_time.append(t); complete_q.append(q); complete_qdot.append(qdot)
        complete_qddot.append(qddot); complete_qdddot.append(qdddot)
        time_offset += segment_time

    return {
        "time": np.concatenate(complete_time), "q": np.concatenate(complete_q),
        "q_dot": np.concatenate(complete_qdot), "q_ddot": np.concatenate(complete_qddot),
        "q_dddot": np.concatenate(complete_qdddot), "joint_waypoints": np.array(joint_waypoints),
        "cartesian_waypoints": np.array([pre_pick, pick_xyz, lift, pre_place, place_xyz, retreat])
    }
