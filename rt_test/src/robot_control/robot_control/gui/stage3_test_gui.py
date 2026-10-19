"""Stage 3 Test GUI - Experimental Multi-Robot Handover Control Station.

Adds:
1. Candidate Selection Strategy Selector:
   - TOP_OF_WORKSPACE (High Z)
   - CENTER_OF_WORKSPACE (Centroid)
2. Approach Strategy Status & Telemetry:
   - DUAL_SYNCHRONIZED (Simultaneous Cartesian corridors)
   - ASYMMETRIC (Non-corridor robot reaches handover first, corridor robot follows)
3. Dedicated experimental cache: `auto_handover_poses_test.json`.
"""

import os
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, List, Optional
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from ament_index_python.packages import get_package_share_directory

from robot_control.gui.stage3_gui import Stage3GUI
from robot_control.handover.stage1_5 import validate_execution_safety_gate
from robot_control.handover.stage3_test import (
    Stage3TestPlan,
    build_stage3_test_plan,
    plan_stage3_test_paths,
)
from robot_control.handover.stage3_test_solver import (
    AutoHandoverTestPose,
    compute_all_test_handover_poses,
    load_test_handover_cache,
    resolve_test_handover_cache_path,
)
from robot_control.handover.trajectory_cache import lookup_cached_target_ik


class Stage3TestGUI(Stage3GUI):
    """Stage 3 Test Experimental Control Station GUI Node."""

    def __init__(self, root: tk.Tk):
        self.test_strategy_var = tk.StringVar(value="TOP_OF_WORKSPACE")
        super().__init__(root)

        # Ensure test cache path is active and load available poses
        self.auto_handover_cache_path = resolve_test_handover_cache_path()
        self._load_initial_auto_handover_poses()

        # Update window title and stage badge
        self.root.title("Stage 3 Test - Multi-Robot Handover Experimental Station")
        if hasattr(self, 'stage_badge'):
            self.stage_badge.config(text="STAGE 3 TEST (EXPERIMENTAL)", bg="#D35400")

        # Inject testing controls into Handover Card
        self._inject_test_controls()
        self._update_handover_status_card()

    def _load_initial_auto_handover_poses(self):
        """Load test auto-handover poses from JSON cache without blocking UI."""
        poses = None
        try:
            if os.path.exists(self.auto_handover_cache_path):
                poses = load_test_handover_cache(self.auto_handover_cache_path, self.config_fingerprint)
        except Exception as e:
            self.get_logger().warn(f"Initial test auto-handover load: {e}")

        # Fallback to main auto_handover_poses.json if test cache empty or not created yet
        if not poses:
            try:
                from robot_control.handover.stage3_handover_solver import (
                    load_auto_handover_cache,
                    resolve_auto_handover_cache_path,
                )
                main_cache = resolve_auto_handover_cache_path()
                if os.path.exists(main_cache):
                    main_poses = load_auto_handover_cache(main_cache, self.config_fingerprint)
                    if main_poses:
                        poses = {}
                        for k, p in main_poses.items():
                            poses[k] = AutoHandoverTestPose(
                                pair_name=p.pair_name,
                                handover_position_world=p.handover_position_world,
                                standoff_distance_mm=p.standoff_distance_mm,
                                robot_a_id=p.robot_a_id,
                                robot_b_id=p.robot_b_id,
                                robot_a_handover_q_deg=p.robot_a_handover_q_deg,
                                robot_b_handover_q_deg=p.robot_b_handover_q_deg,
                                robot_a_pre_handover_q_deg=p.robot_a_pre_handover_q_deg,
                                robot_b_pre_handover_q_deg=p.robot_b_pre_handover_q_deg,
                                handover_clearance_mm=p.handover_clearance_mm,
                                pre_handover_clearance_mm=p.pre_handover_clearance_mm,
                                error_a_mm=p.error_a_mm,
                                error_b_mm=p.error_b_mm,
                                dot_z=p.dot_z,
                                dot_x=p.dot_x,
                                approach_angle_dev_deg=p.approach_angle_dev_deg,
                                approach_mode="ASYMMETRIC_PLACE_FIRST" if "robot2" in k else "DUAL_SYNCHRONIZED",
                                candidate_origin="TOP_OF_WORKSPACE",
                                corridor_a_valid=True,
                                corridor_b_valid=False if "robot2" in k else True,
                            )
                        self.get_logger().info(f"Loaded {len(poses)} baseline poses from {main_cache}.")
            except Exception as e:
                self.get_logger().warn(f"Fallback to main auto-handover cache: {e}")

        self.auto_handover_poses = poses if poses is not None else {}
        if self.auto_handover_poses:
            self.get_logger().info(f"Active auto-handover pairs: {list(self.auto_handover_poses.keys())}")

    def _inject_test_controls(self):
        """Add strategy selector and approach mode badge to the handover UI card."""
        if not hasattr(self, 'ho_summary_lbl'):
            return

        # Frame for testing strategy
        test_frame = tk.Frame(self.ho_summary_lbl.master, bg=self.card_color)
        test_frame.pack(fill="x", pady=(2, 2), before=self.ho_summary_lbl)

        tk.Label(
            test_frame, text="Candidate Strategy:", bg=self.card_color,
            fg="#2C3E50", font=("Helvetica", 8, "bold")
        ).pack(side="left", padx=(0, 4))

        def _on_strat_change():
            self.cached_plan = None
            strat = self.test_strategy_var.get()
            self._log(f"\n[TEST CONFIG] Switched candidate strategy to: {strat}")
            self._update_handover_status_card()

        tk.Radiobutton(
            test_frame, text="Top (High Z)", variable=self.test_strategy_var,
            value="TOP_OF_WORKSPACE", bg=self.card_color, font=("Helvetica", 8),
            activebackground=self.card_color, command=_on_strat_change
        ).pack(side="left")

        tk.Radiobutton(
            test_frame, text="Center (Centroid)", variable=self.test_strategy_var,
            value="CENTER_OF_WORKSPACE", bg=self.card_color, font=("Helvetica", 8),
            activebackground=self.card_color, command=_on_strat_change
        ).pack(side="left", padx=4)

        # Mode Indicator Badge
        mode_frame = tk.Frame(self.ho_summary_lbl.master, bg=self.card_color)
        mode_frame.pack(fill="x", pady=(1, 3), after=self.ho_metrics_lbl)

        tk.Label(
            mode_frame, text="Approach Mode:", bg=self.card_color,
            fg="#555555", font=("Helvetica", 8, "bold")
        ).pack(side="left")

        self.test_approach_mode_lbl = tk.Label(
            mode_frame, text="DUAL_SYNCHRONIZED", bg="#EAF2F8", fg="#2980B9",
            font=("Consolas", 8, "bold"), padx=5, pady=1
        )
        self.test_approach_mode_lbl.pack(side="left", padx=4)

        # Update Recompute button command to test recomputer
        if hasattr(self, 'btn_recompute_ho'):
            self.btn_recompute_ho.config(command=self._on_recompute_test_handover_clicked)

    def _preview_handover_in_3d(self, which: str = 'handover'):
        """Preview either handover or pre-handover pose in 3D scene (Manual or Auto)."""
        mode = self.handover_mode_var.get()
        pair_key = self.selected_ho_pair.get()
        parts = pair_key.split('_') if '_' in pair_key else ["robot1", "robot2"]
        r_a, r_b = parts[0], parts[1]
        inv_key = f"{r_b}_{r_a}"

        if mode == "manual":
            from robot_control.handover.stage1 import lookup_manual_handover_with_pre
            res = lookup_manual_handover_with_pre(self.handover_cfg, r_a, r_b)
            if res is None:
                self._log(f"⚠️ Cannot preview 3D: No manual handover found for {pair_key} in handover.yaml.")
                return
            q_a, q_b, pre_q_a, pre_q_b = res
            self.is_previewing_handover = True
            if which == 'pre_handover':
                self.preview_q = {
                    r_a: np.asarray(pre_q_a, dtype=float).copy(),
                    r_b: np.asarray(pre_q_b, dtype=float).copy(),
                }
                self._log(f"👁️ [3D PREVIEW] Showing MANUAL PRE-HANDOVER stance for {r_a.upper()} and {r_b.upper()}.")
            else:
                self.preview_q = {
                    r_a: np.asarray(q_a, dtype=float).copy(),
                    r_b: np.asarray(q_b, dtype=float).copy(),
                }
                self._log(f"👁️ [3D PREVIEW] Showing MANUAL HANDOVER stance for {r_a.upper()} and {r_b.upper()}.")
            self._update_3d()
            return

        # AUTO mode
        p = self.auto_handover_poses.get(pair_key) or self.auto_handover_poses.get(inv_key)
        if not p:
            # Try reloading from cache
            self._load_initial_auto_handover_poses()
            p = self.auto_handover_poses.get(pair_key) or self.auto_handover_poses.get(inv_key)

        if not p:
            self._log(f"⚠️ Cannot preview 3D: No solution available for {pair_key}. Click '⚡ Recompute' below.")
            return

        self.is_previewing_handover = True
        if which == 'pre_handover':
            self.preview_q = {
                p.robot_a_id: np.asarray(p.pre_handover_a_joint_angles_deg, dtype=float).copy(),
                p.robot_b_id: np.asarray(p.pre_handover_b_joint_angles_deg, dtype=float).copy(),
            }
            self._log(f"👁️ [3D PREVIEW] Showing PRE-HANDOVER stance for {p.robot_a_id.upper()} and {p.robot_b_id.upper()}. (No robot motion commanded)")
        else:
            self.preview_q = {
                p.robot_a_id: np.asarray(p.robot_a_joint_angles_deg, dtype=float).copy(),
                p.robot_b_id: np.asarray(p.robot_b_joint_angles_deg, dtype=float).copy(),
            }
            self._log(f"👁️ [3D PREVIEW] Showing HANDOVER stance for {p.robot_a_id.upper()} and {p.robot_b_id.upper()}. (No robot motion commanded)")

        self._update_3d()

    def _update_handover_status_card(self):
        """Update Handover card UI with test approach mode and metrics."""
        mode = self.handover_mode_var.get()
        pair_name = self.selected_ho_pair.get() if hasattr(self, 'selected_ho_pair') else "robot1_robot2"
        parts = pair_name.split('_') if '_' in pair_name else ["robot1", "robot2"]
        r_a, r_b = parts[0], parts[1]
        inv_key = f"{r_b}_{r_a}"

        if mode == "manual":
            super()._update_handover_status_card()
            if hasattr(self, 'test_approach_mode_lbl'):
                self.test_approach_mode_lbl.config(
                    text="MANUAL (handover.yaml)", bg="#FCF3CF", fg="#B7950B"
                )
            return

        pose = None
        if hasattr(self, 'auto_handover_poses') and self.auto_handover_poses:
            pose = self.auto_handover_poses.get(pair_name) or self.auto_handover_poses.get(inv_key)

        if pose is not None:
            super()._update_handover_status_card()
            mode_str = getattr(pose, 'approach_mode', 'DUAL_SYNCHRONIZED')
            origin_str = getattr(pose, 'candidate_origin', 'TOP_OF_WORKSPACE')

            if hasattr(self, 'test_approach_mode_lbl'):
                if mode_str == "DUAL_SYNCHRONIZED":
                    self.test_approach_mode_lbl.config(
                        text=f"DUAL SYNCHRONIZED ({origin_str})", bg="#E8F8F5", fg="#1ABC9C"
                    )
                else:
                    self.test_approach_mode_lbl.config(
                        text=f"ASYMMETRIC ({mode_str})", bg="#FEF9E7", fg="#F39C12"
                    )
            return

        super()._update_handover_status_card()

    def _on_recompute_test_handover_clicked(self):
        """Recompute test handover poses with chosen strategy."""
        if getattr(self, 'is_recomputing_handover', False):
            return

        self.is_recomputing_handover = True
        self.btn_recompute_ho.config(state="disabled", text="⏳ Solving...")
        self.ho_recomp_msg.config(text="Computing...", fg=self.accent_color)
        strat = self.test_strategy_var.get()
        self._log(f"\n⚡ Starting Stage 3 Test Handover Recomputation (Strategy: {strat})...")

        def progress_cb(msg: str):
            self.root.after(0, lambda m=msg: self._log(f"  [Solver] {m}"))

        def worker():
            try:
                poses = compute_all_test_handover_poses(
                    self.robots,
                    self.handover_cfg,
                    cache_path=self.auto_handover_cache_path,
                    force_recompute=True,
                    preferred_strategy=strat,
                    progress_callback=progress_cb,
                )

                def on_done():
                    self.is_recomputing_handover = False
                    self.auto_handover_poses = poses
                    self.btn_recompute_ho.config(state="normal", text="⚡ Recompute")
                    self.ho_recomp_msg.config(text="✓ Saved", fg=self.success_color)
                    self._update_handover_status_card()
                    self._update_3d()
                    self._log(f"✓ Test handover recomputation complete! {len(poses)} pairs saved to test cache.")

                self.root.after(0, on_done)
            except Exception as e:
                def on_err(err=str(e)):
                    self.is_recomputing_handover = False
                    self.btn_recompute_ho.config(state="normal", text="⚡ Recompute")
                    self.ho_recomp_msg.config(text="Error", fg=self.danger_color)
                    self._log(f"ERROR recomputing test handover poses: {err}")

                self.root.after(0, on_err)

        threading.Thread(target=worker, daemon=True).start()

    def _on_handover_mode_changed(self):
        self.cached_plan = None
        super()._on_handover_mode_changed()

    def _on_target_change(self, event=None):
        super()._on_target_change(event)
        if self.cached_plan is not None:
            try:
                cur_p = tuple(float(x.strip()) for x in self.pick_entry.get().split(','))
                cur_py = float(self.pick_yaw_entry.get().strip())
                cur_pl = tuple(float(x.strip()) for x in self.place_entry.get().split(','))
                cur_ply = float(self.place_yaw_entry.get().strip())
                strat = self.test_strategy_var.get()
                use_auto = (self.handover_mode_var.get() == "auto")
                key = (cur_p, cur_py, cur_pl, cur_ply, strat, use_auto)
                if key != self.cached_plan.get('target_key'):
                    self.cached_plan = None
            except Exception:
                self.cached_plan = None

    def _run_plan_thread(self, real_robot: bool = False):
        """Execute Stage 3 Test planning and execution with instant cache reuse like Stage 2.5."""
        self._log("=== STARTING STAGE 3 TEST TASK EXECUTION ===")
        mode_str = "REAL ROBOT HARDWARE" if real_robot else "3D SIMULATION PREVIEW"
        self._log(f"[MODE: {mode_str}] Strategy: {self.test_strategy_var.get()}")

        try:
            pick_xyz = [float(x.strip()) for x in self.pick_entry.get().split(',')]
            pick_yaw = float(self.pick_yaw_entry.get().strip())
            place_xyz = [float(x.strip()) for x in self.place_entry.get().split(',')]
            place_yaw = float(self.place_yaw_entry.get().strip())
        except Exception:
            self._log("ERROR: Invalid Pick/Place Coordinates or Yaw Angle.")
            return

        current_q = {rid: q.copy() for rid, q in self.live_q.items()}
        strat = self.test_strategy_var.get()
        use_auto = (self.handover_mode_var.get() == "auto")

        target_key = (tuple(pick_xyz), pick_yaw, tuple(place_xyz), place_yaw, strat, use_auto)

        cache_fp = self.cache_filepath if (hasattr(self, 'use_disk_cache_var') and self.use_disk_cache_var.get()) else None
        fp_hash = self.config_fingerprint if (hasattr(self, 'use_disk_cache_var') and self.use_disk_cache_var.get()) else None

        is_pick_cached = False
        is_place_cached = False
        if cache_fp and fp_hash:
            is_pick_cached = lookup_cached_target_ik(cache_fp, fp_hash, pick_xyz, pick_yaw) is not None
            is_place_cached = lookup_cached_target_ik(cache_fp, fp_hash, place_xyz, place_yaw) is not None

        if is_pick_cached and is_place_cached:
            self._log("\n⚡ [DISK CACHE HIT] Target IK loaded from disk in < 1 ms!")
        else:
            self._log("\n🔄 Solving kinematics and adaptive tilt angles across robots...")

        try:
            plan, pick_opts, place_opts = build_stage3_test_plan(
                pick_xyz=pick_xyz,
                place_xyz=place_xyz,
                robots=self.robots,
                handover_cfg=self.handover_cfg,
                chooser=self._chooser,
                pick_yaw_deg=pick_yaw,
                place_yaw_deg=place_yaw,
                use_auto_handover=use_auto,
                preferred_strategy=strat,
                cache_filepath=cache_fp,
                config_fingerprint=fp_hash,
                test_poses=self.auto_handover_poses,
            )
        except RuntimeError as exc:
            self._log(f"ERROR: {exc}")
            return

        self._set_badges(plan.pick.solver_type, plan.place.solver_type)

        self._log(f"\nPICK: {plan.pick_robot.upper()} (Solver: {plan.pick.solver_type})")
        self._log(f"PLACE: {plan.place_robot.upper()} (Solver: {plan.place.solver_type})")

        if plan.needs_handover:
            info = plan.auto_handover_info
            if info:
                self._log(f"HANDOVER: TEST AUTO {plan.pick_robot} <-> {plan.place_robot}")
                self._log(f"  Handover Position: {info.handover_position_world} mm (Origin: {plan.candidate_origin})")
                self._log(f"  Standoff: {info.standoff_distance_mm:.1f} mm | Clr: {info.handover_clearance_mm:.1f} mm")
            else:
                self._log(f"HANDOVER: MANUAL OVERRIDE (handover.yaml)")

        self._log("\nPlanning collision-free synchronized trajectories...")
        traj_result = plan_stage3_test_paths(
            plan, self.robots, safe_distance=self.safe_dist, steps=self.trajectory_steps, current_q=current_q
        )

        if traj_result.collision_avoidance_applied:
            self._log("! WARNING: Direct simultaneous path was near collision limit.")
            self._log("! Safe collision-avoidance staggering was automatically applied.")

        self._log(f"Min predicted body clearance: {traj_result.min_overall_clearance_mm:.1f} mm (Safe dist: {self.safe_dist:.1f} mm)")

        self.cached_plan = {
            'target_key': target_key,
            'plan': plan,
            'traj_result': traj_result,
        }
        is_cached = (is_pick_cached and is_place_cached)

        proceed = self._show_solution_dialog(
            plan, traj_result,
            is_cached=is_cached,
            real_robot=real_robot,
        )
        if not proceed:
            self._log("\n[ABORTED] Execution cancelled by operator in solution dialog.")
            self._set_status_lbl("Execution cancelled by user", self.warning_color)
            return

        self.is_simulating = True
        self.is_real_execution = real_robot
        self.current_q = {k: v.copy() for k, v in self.live_q.items()}

        pick_rid = plan.pick_robot
        place_rid = plan.place_robot

        self._set_control_mode(pick_rid, 'TRAJECTORY', real_robot=real_robot)
        if plan.needs_handover:
            self._set_control_mode(place_rid, 'TRAJECTORY', real_robot=real_robot)

        def step_simultaneous(step_list, phase_label, dwell=None):
            dwell = self.dwell_time_s if dwell is None else dwell
            self._log(f"\n>>> {phase_label}")
            for step in step_list:
                if pick_rid == place_rid:
                    self.current_q[pick_rid] = np.asarray(step.pick_q, dtype=float).copy()
                    self._send_joint_command(pick_rid, self.current_q[pick_rid], real_robot=real_robot)
                else:
                    self.current_q[pick_rid] = np.asarray(step.pick_q, dtype=float).copy()
                    self.current_q[place_rid] = np.asarray(step.place_q, dtype=float).copy()
                    self._send_joint_command(pick_rid, self.current_q[pick_rid], real_robot=real_robot)
                    self._send_joint_command(place_rid, self.current_q[place_rid], real_robot=real_robot)
                status = "safe" if step.is_safe else "warn"
                self._update_clearance_ui(step.min_clearance_mm, "MOTION ACTIVE", status)
                self.root.after(0, self._update_3d)
                time.sleep(self.step_interval_s)
            time.sleep(dwell)

        if plan.needs_handover:
            # PHASE 1: SIMULTANEOUS MOTION
            step_simultaneous(
                traj_result.phase1_simultaneous,
                f"1. SIMULTANEOUS: {pick_rid} moves to Pick, WHILE {place_rid} moves to Pre-Handover!",
                dwell=self.dwell_time_s,
            )

            # PHASE 2: GRASP AT PICK
            self._log(f"2. {pick_rid} waits {self.pick_wait_s}s and closes gripper. ({place_rid} waiting at Pre-Handover)")
            self._send_gripper_command(pick_rid, 'close', real_robot=real_robot)
            self._update_clearance_ui(self.live_clearance_mm, f"{place_rid} WAITING AT PRE-HANDOVER", "info")
            time.sleep(self.pick_wait_s)

            # PHASE 3: PICK ROBOT MOVES TO PRE-HANDOVER
            step_simultaneous(
                traj_result.phase3_pick_to_pre_handover,
                f"3. {pick_rid} moves to Pre-Handover ({place_rid} stationed at Pre-Handover).",
                dwell=self.dwell_time_s,
            )

            # PHASE 4: SYNCHRONIZED ADVANCE (Pre-Handover -> Handover)
            step_simultaneous(
                traj_result.phase4_advance_to_handover,
                f"4. ADVANCE: Both {pick_rid} and {place_rid} advance Pre-Handover -> Handover at same rate!",
                dwell=self.dwell_time_s,
            )

            # PHASE 5: CONFIRM & HANDOVER TRANSFER
            self._log("5. CONFIRMED: Both robots are at Handover position.")
            self._log(f"   Transfer: {place_rid} closes gripper, {pick_rid} opens gripper.")
            self._send_gripper_command(place_rid, 'close', real_robot=real_robot)
            self._send_gripper_command(pick_rid, 'open', real_robot=real_robot)
            if traj_result.phase4_advance_to_handover:
                self._update_clearance_ui(traj_result.phase4_advance_to_handover[-1].min_clearance_mm, "HANDOVER TRANSFER (GRIPPER)", "info")
            time.sleep(self.transfer_wait_s)

            # PHASE 6: SYNCHRONIZED RETRACT (Handover -> Pre-Handover)
            step_simultaneous(
                traj_result.phase6_retract_to_pre_handover,
                f"6. RETRACT: Both robots back away to Pre-Handover at same rate!",
                dwell=self.dwell_time_s,
            )

            # PHASE 7: PLACE ROBOT MOVES PRE-HANDOVER TO PLACE
            step_simultaneous(
                traj_result.phase7_handover_to_place,
                f"7. {place_rid} moves from Pre-Handover to Place target ({pick_rid} waiting at Pre-Handover).",
                dwell=self.dwell_time_s,
            )

            # PHASE 8: RELEASE OBJECT
            self._log(f"8. {place_rid} waits {self.place_wait_s}s, opens gripper to release object.")
            self._send_gripper_command(place_rid, 'open', real_robot=real_robot)
            time.sleep(self.place_wait_s)

            # PHASE 9: SIMULTANEOUS RETURN HOME
            step_simultaneous(
                traj_result.phase9_simultaneous_home,
                f"9. SIMULTANEOUS RETURN: Both {pick_rid} and {place_rid} return to Home!",
                dwell=self.dwell_time_s,
            )

        else:
            self._log("\nSame robot for pick and place — skipping handover.")
            step_simultaneous(traj_result.phase1_simultaneous, f"1. {pick_rid} moves to Pick.", dwell=self.dwell_time_s)
            self._log(f"2. {pick_rid} waits {self.pick_wait_s}s, closes gripper.")
            self._send_gripper_command(pick_rid, 'close', real_robot=real_robot)
            time.sleep(self.pick_wait_s)
            step_simultaneous(traj_result.phase7_handover_to_place, f"3. {pick_rid} moves to Place.", dwell=self.dwell_time_s)
            self._log(f"4. {pick_rid} waits {self.place_wait_s}s, releases object.")
            self._send_gripper_command(pick_rid, 'open', real_robot=real_robot)
            time.sleep(self.place_wait_s)
            step_simultaneous(traj_result.phase9_simultaneous_home, f"5. {pick_rid} returns Home.", dwell=self.dwell_time_s)

        self.current_q[pick_rid] = np.zeros(self.robots[pick_rid].num_joints)
        self.current_q[place_rid] = np.zeros(self.robots[place_rid].num_joints)

        # Reset hardware mode back to MANUAL
        self._set_control_mode(pick_rid, 'MANUAL', real_robot=real_robot)
        if plan.needs_handover:
            self._set_control_mode(place_rid, 'MANUAL', real_robot=real_robot)

        self._log("\n=== STAGE 3 TEST EXECUTION SUCCESS ===")
        self._update_clearance_ui(float("inf"), "STAGE 3 TEST COMPLETED", "safe")
        if not real_robot:
            self._set_status_lbl("✓ Simulation verified & ready", self.success_color)
            self._log("\n[READY] Click '⚡ EXECUTE (REAL ROBOT)' to directly run this exact plan on hardware!")
        else:
            self._set_status_lbl("✓ Real robot execution completed", self.success_color)

        self.is_simulating = False


def main(args=None):
    rclpy.init(args=args)
    root = tk.Tk()
    gui_node = Stage3TestGUI(root)

    def ros_spin():
        try:
            rclpy.spin(gui_node)
        except Exception:
            pass

    t = threading.Thread(target=ros_spin, daemon=True)
    t.start()

    try:
        root.mainloop()
    finally:
        gui_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
