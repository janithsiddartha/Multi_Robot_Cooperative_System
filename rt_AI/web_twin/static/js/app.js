/**
 * rt_AI 3D WEB DIGITAL TWIN — APPLICATION CONTROLLER
 * Full Stage 3 Test 2 Multi-Robot Planning + Robot GUI 1, 2, 3 Manual Jogging + 3D Inspector
 */

class WebTwinApp {
  constructor() {
    this.ws = null;
    this.scene = null;
    this.robots = {};
    this.laser = null;

    // Kinematics & visual configurations
    this.robotConfigs = {};
    this.currentJointAngles = {};
    this.activeJogRobot = 'robot1';
    this.activeGripperStates = { robot1: 'open', robot2: 'open', robot3: 'open' };

    // Simulation / Playback state
    this.trajectorySteps = [];
    this.currentStepIdx = 0;
    this.isPlaying = false;
    this.playbackTimer = null;
    this.playbackSpeed = 1.0;
    this.currentPlanSummary = null;
    this.currentPlanKey = null;

    // Active Pending Plan Request
    this.pendingPlan = null;

    this._initDomReferences();
    this._initEventListeners();
    this._connectWebSocket();
  }

  _initDomReferences() {
    this.dom = {
      wsStatusDot: document.getElementById('ws-status-dot'),
      wsStatusText: document.getElementById('ws-status-text'),
      canvas: document.getElementById('webgl-canvas'),

      // Sidebar Tabs
      modeNavBtns: document.querySelectorAll('.mode-nav-btn'),
      sidebarPanes: document.querySelectorAll('.sidebar-pane'),

      // Planning Inputs - Pick
      pickX: document.getElementById('input-pick-x'),
      pickY: document.getElementById('input-pick-y'),
      pickZ: document.getElementById('input-pick-z'),
      pickYaw: document.getElementById('input-pick-yaw'),
      pickYawSlider: document.getElementById('slider-pick-yaw'),
      pickBadge: document.getElementById('badge-pick-status'),

      // Planning Inputs - Place
      placeX: document.getElementById('input-place-x'),
      placeY: document.getElementById('input-place-y'),
      placeZ: document.getElementById('input-place-z'),
      placeYaw: document.getElementById('input-place-yaw'),
      placeYawSlider: document.getElementById('slider-place-yaw'),
      placeBadge: document.getElementById('badge-place-status'),

      // Primary Action Buttons (Stage 3 Test 2)
      btnSimulateTask: document.getElementById('btn-simulate-task'),
      btnExecuteTask: document.getElementById('btn-execute-task'),
      btnPlan: document.getElementById('btn-plan'),

      // Single-Robot Kinematics Subtabs & Direct IK (testGUI 1, 2, 3)
      jogSubtabBtns: document.querySelectorAll('.jog-subtab-btn'),
      jogSubpanes: document.querySelectorAll('.jog-subpane'),
      singleIkX: document.getElementById('input-single-ik-x'),
      singleIkY: document.getElementById('input-single-ik-y'),
      singleIkZ: document.getElementById('input-single-ik-z'),
      btnSolveSingleIk: document.getElementById('btn-solve-single-ik'),
      singleIkFeedback: document.getElementById('single-ik-feedback'),
      dhParamsTbody: document.getElementById('dh-params-tbody'),
      jogFkStatus: document.getElementById('jog-fk-status'),

      // Handover Telemetry Card
      hoCoords: document.getElementById('ho-coords'),
      hoMode: document.getElementById('ho-mode'),
      hoStandoff: document.getElementById('ho-standoff'),

      // Handover Inspector Controls
      selectInspectorPair: document.getElementById('select-inspector-pair'),
      btnInspectHandover: document.getElementById('btn-inspect-handover'),
      btnInspectPre: document.getElementById('btn-inspect-pre'),
      btnInspectHome: document.getElementById('btn-inspect-home'),

      // Manual Joint Jogger (Robot GUI 1, 2, 3)
      joggerPills: document.querySelectorAll('.jogger-pill'),
      joggerSlidersContainer: document.getElementById('jogger-sliders-container'),
      btnJogGripper: document.getElementById('btn-jog-gripper'),
      btnJogHome: document.getElementById('btn-jog-home'),
      jogTcpWorld: document.getElementById('jog-tcp-world'),

      // Timeline & Controls
      btnHomeAll: document.getElementById('btn-home'),
      btnPlayPause: document.getElementById('btn-play-pause'),
      btnReset: document.getElementById('btn-reset'),
      btnExecute: document.getElementById('btn-execute'),
      timelineSlider: document.getElementById('timeline-slider'),
      stepCounter: document.getElementById('timeline-step-counter'),
      phaseBadge: document.getElementById('timeline-phase-badge'),
      speedSelect: document.getElementById('select-playback-speed'),

      // Gizmo Mode Buttons
      btnGizmoTranslate: document.getElementById('btn-gizmo-translate'),
      btnGizmoRotate: document.getElementById('btn-gizmo-rotate'),

      // Terminal Log & Joint Tables
      terminalLog: document.getElementById('terminal-log'),
      jointR1Table: document.getElementById('joint-r1-table'),
      jointR2Table: document.getElementById('joint-r2-table'),
      jointR3Table: document.getElementById('joint-r3-table'),

      // Multi-Robot Chooser Modal
      chooserModal: document.getElementById('modal-robot-chooser'),
      chooserPickContainer: document.getElementById('chooser-pick-candidates'),
      chooserPlaceContainer: document.getElementById('chooser-place-candidates'),
      btnConfirmChoice: document.getElementById('btn-confirm-choice'),
      btnCancelChoice: document.getElementById('btn-cancel-choice'),

      // Solution Summary Modal (Stage 3 Test 2 Dialog)
      solutionModal: document.getElementById('modal-solution-summary'),
      summaryRoute: document.getElementById('summary-route'),
      summaryPickSolver: document.getElementById('summary-pick-solver'),
      summaryPlaceSolver: document.getElementById('summary-place-solver'),
      summaryHoMode: document.getElementById('summary-ho-mode'),
      summaryHoCoords: document.getElementById('summary-ho-coords'),
      summaryStandoff: document.getElementById('summary-standoff'),
      summaryClearance: document.getElementById('summary-clearance'),
      btnSummarySimulate: document.getElementById('btn-summary-simulate'),
      btnSummaryExecute: document.getElementById('btn-summary-execute'),
      btnSummaryDismiss: document.getElementById('btn-summary-dismiss'),

      // Real Hardware Safety Confirmation Modal
      safetyModal: document.getElementById('modal-safety-gate'),
      safetyClearanceVal: document.getElementById('modal-clearance-val'),
      btnConfirmExecute: document.getElementById('btn-confirm-execute'),
      btnCancelExecute: document.getElementById('btn-cancel-execute'),
    };
  }

  _initEventListeners() {
    // 3D Scene instantiation
    this.scene = new WebTwinScene(this.dom.canvas);
    this.laser = new DynamicClearanceLaser(this.scene.scene);

    // Sidebar Mode Nav Tabs
    this.dom.modeNavBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        this.dom.modeNavBtns.forEach((b) => b.classList.remove('active'));
        this.dom.sidebarPanes.forEach((p) => p.classList.remove('active'));
        btn.classList.add('active');
        const targetId = btn.getAttribute('data-tab');
        const pane = document.getElementById(targetId);
        if (pane) pane.classList.add('active');

        if (targetId === 'tab-jogger') {
          this._renderJoggerSliders();
        }
      });
    });

    // Synchronize 3D Gizmo drags back into input boxes
    this.scene.onTargetMovedCallback = (type, robCoord, yawDeg) => {
      if (type === 'pick') {
        this.dom.pickX.value = robCoord[0];
        this.dom.pickY.value = robCoord[1];
        this.dom.pickZ.value = robCoord[2];
        this.dom.pickYaw.value = Math.round(yawDeg);
        this.dom.pickYawSlider.value = Math.round(yawDeg);
        this._queryReachability('pick', robCoord, yawDeg);
      } else {
        this.dom.placeX.value = robCoord[0];
        this.dom.placeY.value = robCoord[1];
        this.dom.placeZ.value = robCoord[2];
        this.dom.placeYaw.value = Math.round(yawDeg);
        this.dom.placeYawSlider.value = Math.round(yawDeg);
        this._queryReachability('place', robCoord, yawDeg);
      }
    };

    // Debounced Reachability Scheduler (prevents thread pool queue starvation)
    let reachDebounceTimer = null;
    const scheduleReachability = (type, xyz, yaw) => {
      if (reachDebounceTimer) clearTimeout(reachDebounceTimer);
      reachDebounceTimer = setTimeout(() => {
        this._queryReachability(type, xyz, yaw);
      }, 450);
    };

    // Coordinate inputs onchange
    const onPickChange = () => {
      const xyz = [
        parseFloat(this.dom.pickX.value) || 0,
        parseFloat(this.dom.pickY.value) || 0,
        parseFloat(this.dom.pickZ.value) || 0,
      ];
      const yaw = parseFloat(this.dom.pickYaw.value) || 0;
      this.dom.pickYawSlider.value = yaw;
      this.scene.setPickTarget(xyz, yaw);
      scheduleReachability('pick', xyz, yaw);
    };

    const onPlaceChange = () => {
      const xyz = [
        parseFloat(this.dom.placeX.value) || 0,
        parseFloat(this.dom.placeY.value) || 0,
        parseFloat(this.dom.placeZ.value) || 0,
      ];
      const yaw = parseFloat(this.dom.placeYaw.value) || 0;
      this.dom.placeYawSlider.value = yaw;
      this.scene.setPlaceTarget(xyz, yaw);
      scheduleReachability('place', xyz, yaw);
    };

    [this.dom.pickX, this.dom.pickY, this.dom.pickZ, this.dom.pickYaw].forEach((el) => {
      el.addEventListener('input', onPickChange);
    });
    this.dom.pickYawSlider.addEventListener('input', (e) => {
      this.dom.pickYaw.value = e.target.value;
      onPickChange();
    });

    [this.dom.placeX, this.dom.placeY, this.dom.placeZ, this.dom.placeYaw].forEach((el) => {
      el.addEventListener('input', onPlaceChange);
    });
    this.dom.placeYawSlider.addEventListener('input', (e) => {
      this.dom.placeYaw.value = e.target.value;
      onPlaceChange();
    });

    // Preset Buttons
    document.querySelectorAll('.preset-pill').forEach((btn) => {
      btn.addEventListener('click', () => {
        const type = btn.getAttribute('data-type');
        const coords = JSON.parse(btn.getAttribute('data-coords'));
        const yaw = parseFloat(btn.getAttribute('data-yaw') || 0);

        if (type === 'pick') {
          this.dom.pickX.value = coords[0];
          this.dom.pickY.value = coords[1];
          this.dom.pickZ.value = coords[2];
          this.dom.pickYaw.value = yaw;
          this.dom.pickYawSlider.value = yaw;
          this.scene.setPickTarget(coords, yaw);
          this._queryReachability('pick', coords, yaw);
        } else {
          this.dom.placeX.value = coords[0];
          this.dom.placeY.value = coords[1];
          this.dom.placeZ.value = coords[2];
          this.dom.placeYaw.value = yaw;
          this.dom.placeYawSlider.value = yaw;
          this.scene.setPlaceTarget(coords, yaw);
          this._queryReachability('place', coords, yaw);
        }
      });
    });

    // Transform Gizmo Mode Toggles
    this.dom.btnGizmoTranslate.addEventListener('click', () => {
      this.scene.setGizmoMode('translate');
      this.dom.btnGizmoTranslate.classList.add('active');
      this.dom.btnGizmoRotate.classList.remove('active');
    });

    this.dom.btnGizmoRotate.addEventListener('click', () => {
      this.scene.setGizmoMode('rotate');
      this.dom.btnGizmoRotate.classList.add('active');
      this.dom.btnGizmoTranslate.classList.remove('active');
    });

    // Camera Preset Buttons
    document.querySelectorAll('.camera-btn[data-preset]').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.camera-btn[data-preset]').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        this.scene.setCameraPreset(btn.getAttribute('data-preset'));
      });
    });

    // Telemetry Tabs (Joints vs Log)
    document.querySelectorAll('.telemetry-tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.telemetry-tab').forEach((t) => t.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach((p) => p.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tab.getAttribute('data-target')).classList.add('active');
      });
    });

    // Primary Action Buttons (matching Stage 3 Test 2 GUI)
    if (this.dom.btnSimulateTask) {
      this.dom.btnSimulateTask.addEventListener('click', () => this.executeOrSimulateTask(false));
    }
    if (this.dom.btnExecuteTask) {
      this.dom.btnExecuteTask.addEventListener('click', () => this.executeOrSimulateTask(true));
    }
    if (this.dom.btnPlan) {
      this.dom.btnPlan.addEventListener('click', () => this.requestPlan({ autoSimulate: false, autoExecute: false }));
    }

    // Top Right Home All Button
    if (this.dom.btnHomeAll) {
      this.dom.btnHomeAll.addEventListener('click', () => {
        this.log('[System] Returning all physical robots to mechanical HOME...', 'info');
        this.send({ cmd: 'home_all' });
      });
    }

    // Playback Controls
    this.dom.btnPlayPause.addEventListener('click', () => this.togglePlayback());
    this.dom.btnReset.addEventListener('click', () => this.resetPlayback());
    this.dom.timelineSlider.addEventListener('input', (e) => {
      this.pausePlayback();
      this.applyStep(parseInt(e.target.value));
    });
    this.dom.speedSelect.addEventListener('change', (e) => {
      this.playbackSpeed = parseFloat(e.target.value);
      if (this.isPlaying) {
        this.pausePlayback();
        this.startPlayback();
      }
    });

    // Hardware Execution Button -> Safety Gate Modal
    this.dom.btnExecute.addEventListener('click', () => {
      if (!this.currentPlanSummary) {
        alert('Please plan and verify a trajectory before executing on real hardware.');
        return;
      }
      this.dom.safetyClearanceVal.textContent = `${this.currentPlanSummary.min_overall_clearance_mm} mm`;
      this.dom.safetyModal.classList.add('active');
    });

    this.dom.btnCancelExecute.addEventListener('click', () => {
      this.dom.safetyModal.classList.remove('active');
    });

    this.dom.btnConfirmExecute.addEventListener('click', () => {
      this.dom.safetyModal.classList.remove('active');
      this.log('🚀 Safety gate confirmed. Hardware trajectory streaming initiated...', 'success');
      this.startPlayback();
    });

    // Multi-Robot Chooser Modal Listeners
    this.dom.btnCancelChoice.addEventListener('click', () => {
      this.dom.chooserModal.classList.remove('active');
      this.pendingPlan = null;
      this.dom.planLoadingBanner.style.display = 'none';
      this.dom.btnPlan.disabled = false;
      this.dom.btnPlan.innerHTML = '<i class="fa fa-bolt"></i> Plan &amp; Verify Trajectory';
    });

    this.dom.btnConfirmChoice.addEventListener('click', () => {
      const selectedPick = document.querySelector('.pick-choice-card.selected');
      const selectedPlace = document.querySelector('.place-choice-card.selected');
      if (!selectedPick || !selectedPlace) {
        alert('Please select both a Pick robot and a Place robot.');
        return;
      }
      this.dom.chooserModal.classList.remove('active');
      if (this.pendingPlan) {
        this.pendingPlan.chosen_pick_robot = selectedPick.getAttribute('data-robot');
        this.pendingPlan.chosen_place_robot = selectedPlace.getAttribute('data-robot');
        this.dom.planLoadingBanner.style.display = 'flex';
        this.dom.planLoadingTitle.textContent = 'Generating Corridors...';
        this.dom.planLoadingSubtitle.textContent = `Planning ${this.pendingPlan.chosen_pick_robot.toUpperCase()} -> ${this.pendingPlan.chosen_place_robot.toUpperCase()} trajectory...`;
        this.send(this.pendingPlan);
        this.log(
          `Operator Chooser: Assigned Pick to ${this.pendingPlan.chosen_pick_robot}, Place to ${this.pendingPlan.chosen_place_robot}.`,
          'info'
        );
      }
    });

    // Solution Summary Modal (Stage 3 Test 2 Confirmation)
    this.dom.btnSummaryDismiss.addEventListener('click', () => {
      this.dom.solutionModal.classList.remove('active');
    });

    this.dom.btnSummarySimulate.addEventListener('click', () => {
      this.dom.solutionModal.classList.remove('active');
      this.applyStep(0);
      this.startPlayback();
    });

    this.dom.btnSummaryExecute.addEventListener('click', () => {
      this.dom.solutionModal.classList.remove('active');
      this.dom.safetyClearanceVal.textContent = `${this.currentPlanSummary.min_overall_clearance_mm} mm`;
      this.dom.safetyModal.classList.add('active');
    });

    // Manual Jogger Robot Pills
    this.dom.joggerPills.forEach((pill) => {
      pill.addEventListener('click', () => {
        this.dom.joggerPills.forEach((p) => p.classList.remove('active'));
        pill.classList.add('active');
        this.activeJogRobot = pill.getAttribute('data-robot');
        this._renderJoggerSliders();
        this._renderDhTable();
      });
    });

    // Single-Robot Subtabs (Sliders / Direct IK / DH Parameters from testGUI 1, 2, 3)
    if (this.dom.jogSubtabBtns) {
      this.dom.jogSubtabBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
          this.dom.jogSubtabBtns.forEach((b) => b.classList.remove('active'));
          this.dom.jogSubpanes.forEach((p) => p.classList.remove('active'));
          btn.classList.add('active');
          const targetId = btn.getAttribute('data-subtab');
          const pane = document.getElementById(targetId);
          if (pane) pane.classList.add('active');
          if (targetId === 'jog-sub-dh') {
            this._renderDhTable();
          }
        });
      });
    }

    // Direct Single-Robot Inverse Kinematics (testGUI 1, 2, 3)
    if (this.dom.btnSolveSingleIk) {
      this.dom.btnSolveSingleIk.addEventListener('click', () => {
        const x = parseFloat(this.dom.singleIkX.value) || 0;
        const y = parseFloat(this.dom.singleIkY.value) || 0;
        const z = parseFloat(this.dom.singleIkZ.value) || 0;
        this.dom.singleIkFeedback.textContent = `Solving IK for ${this.activeJogRobot.toUpperCase()}...`;
        this.dom.singleIkFeedback.style.color = '#00F0FF';
        this.log(`[Single IK] Solving IK for ${this.activeJogRobot.toUpperCase()} to target [${x}, ${y}, ${z}] mm...`, 'info');
        this.send({
          cmd: 'solve_single_ik',
          robot_id: this.activeJogRobot,
          target_xyz: [x, y, z],
        });
      });
    }

    // Manual Jogger Gripper Toggle
    this.dom.btnJogGripper.addEventListener('click', () => {
      const cur = this.activeGripperStates[this.activeJogRobot] || 'open';
      const nextState = cur === 'open' ? 'close' : 'open';
      this.send({
        cmd: 'toggle_gripper',
        robot_id: this.activeJogRobot,
        state: nextState,
      });
    });

    // Manual Jogger Home Selected Robot
    this.dom.btnJogHome.addEventListener('click', () => {
      this.log(`[Manual Jogger] Resetting ${this.activeJogRobot.toUpperCase()} to mechanical HOME...`, 'info');
      this.send({
        cmd: 'home_arm',
        robot_id: this.activeJogRobot,
      });
    });

    // Handover Inspector Buttons (like Stage 3 Test 2 GUI)
    this.dom.btnInspectHandover.addEventListener('click', () => {
      const pair = this.dom.selectInspectorPair.value;
      this.log(`[Handover Inspector] Requesting 3D Mated Handover posture for ${pair}...`, 'info');
      this.send({
        cmd: 'preview_handover',
        pair: pair,
        which: 'handover',
      });
    });

    this.dom.btnInspectPre.addEventListener('click', () => {
      const pair = this.dom.selectInspectorPair.value;
      this.log(`[Handover Inspector] Requesting Pre-Handover Stance for ${pair}...`, 'info');
      this.send({
        cmd: 'preview_handover',
        pair: pair,
        which: 'pre_handover',
      });
    });

    this.dom.btnInspectHome.addEventListener('click', () => {
      this.send({ cmd: 'home_all' });
    });
  }

  _connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    this.log(`Connecting to Digital Twin bridge at ${wsUrl}...`, 'info');
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this.dom.wsStatusDot.classList.remove('disconnected');
      this.dom.wsStatusText.textContent = 'ONLINE (60 FPS)';
      this.log('⚡ Connected to rt_AI Kinematics & Digital Twin Engine.', 'success');
    };

    this.ws.onclose = () => {
      this.dom.wsStatusDot.classList.add('disconnected');
      this.dom.wsStatusText.textContent = 'DISCONNECTED';
      this.log('⚠️ Digital Twin bridge disconnected. Reconnecting in 3s...', 'warn');
      setTimeout(() => this._connectWebSocket(), 3000);
    };

    this.ws.onerror = (err) => {
      this.log(`WebSocket Error: ${err.message || 'Connection lost'}`, 'error');
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this._onWebSocketMessage(data);
      } catch (e) {
        console.error('Error parsing WebSocket message:', e);
      }
    };
  }

  send(payload) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
    }
  }

  _queryReachability(type, xyz, yaw) {
    const badge = type === 'pick' ? this.dom.pickBadge : this.dom.placeBadge;
    badge.textContent = 'CHECKING...';
    badge.style.background = 'rgba(255, 255, 255, 0.08)';
    badge.style.color = '#94A3B8';

    this.send({
      cmd: 'check_reachability',
      target_type: type,
      target_xyz: xyz,
      target_yaw: yaw,
    });
  }

  _onWebSocketMessage(data) {
    // 1. Initial State
    if (data.type === 'initial_state') {
      this.robotConfigs = data.robots;
      this.currentJointAngles = data.current_q;
      this._setupRobots(data.robots);
      this._updateHandoverCard(data.handover_poses);
      if (data.clearance) {
        this.laser.update(
          data.clearance.min_distance_mm,
          data.clearance.endpoint_a,
          data.clearance.endpoint_b
        );
      }
      this._queryReachability('pick', data.default_pick, 0.0);
      this._queryReachability('place', data.default_place, 0.0);
      this._renderJoggerSliders();
      this.log(`Initialized 3 physical arms (R1, R2, R3). Kinematics verified.`, 'info');
    }

    // 2. Reachability Result
    else if (data.cmd === 'reachability_result') {
      const badge = data.target_type === 'pick' ? this.dom.pickBadge : this.dom.placeBadge;
      if (data.success) {
        const count = data.count;
        const robotsList = Object.keys(data.reachable_robots).join(', ');
        badge.textContent = data.is_fallback
          ? `FALLBACK (${robotsList})`
          : `REACHABLE (${robotsList})`;
        badge.style.background = data.is_fallback
          ? 'rgba(255, 184, 0, 0.15)'
          : 'rgba(0, 255, 157, 0.15)';
        badge.style.color = data.is_fallback ? '#FFB800' : '#00FF9D';
        badge.style.borderColor = data.is_fallback
          ? 'rgba(255, 184, 0, 0.3)'
          : 'rgba(0, 255, 157, 0.3)';
      } else {
        badge.textContent = 'OUT OF REACH';
        badge.style.background = 'rgba(255, 0, 85, 0.15)';
        badge.style.color = '#FF0055';
        badge.style.borderColor = 'rgba(255, 0, 85, 0.3)';
      }
    }

    // 3. Trajectory Plan Result
    else if (data.cmd === 'trajectory_planned') {
      this.dom.planLoadingBanner.style.display = 'none';
      if (this.dom.btnPlan) this.dom.btnPlan.disabled = false;
      if (this.dom.btnSimulateTask) this.dom.btnSimulateTask.disabled = false;
      if (this.dom.btnExecuteTask) this.dom.btnExecuteTask.disabled = false;
      if (this.dom.btnPlan) {
        this.dom.btnPlan.innerHTML = '<i class="fa fa-magnifying-glass"></i> Verify Kinematics &amp; Corridors';
      }

      if (!data.success) {
        this.log(`Planning Error: ${data.error}`, 'error');
        alert(`Planning failed: ${data.error}`);
        return;
      }

      // Check if interactive robot choice is required
      if (data.needs_choice) {
        this._showRobotChooserModal(data);
        return;
      }

      this.currentPlanSummary = data.plan_summary;
      this.currentPlanKey = this._getCurrentTargetKey();
      this.trajectorySteps = data.steps;
      this.currentStepIdx = 0;

      this.dom.timelineSlider.max = this.trajectorySteps.length - 1;
      this.dom.timelineSlider.value = 0;

      const ps = data.plan_summary;
      if (data.planning_logs) {
        data.planning_logs.forEach((line) => {
          this.log(line, 'info');
        });
      } else {
        this.log(
          `Plan generated: ${ps.pick_robot.toUpperCase()} -> ${ps.place_robot.toUpperCase()} | Min Clearance: ${ps.min_overall_clearance_mm} mm | Total Steps: ${ps.total_steps}`,
          'success'
        );
      }

      // Check if auto-execution or auto-simulation was requested
      if (this.pendingPlan && this.pendingPlan.autoExecute) {
        this.dom.safetyClearanceVal.textContent = `${ps.min_overall_clearance_mm} mm`;
        this.dom.safetyModal.classList.add('active');
      } else if (this.pendingPlan && this.pendingPlan.autoSimulate) {
        this.applyStep(0);
        this.startPlayback();
      } else {
        // Display the Stage 3 Test 2 Solution Summary Confirmation Dialog
        this._showSolutionSummaryDialog(ps);
      }
    }

    // 4. Return Home Result
    else if (data.cmd === 'home_result') {
      this.currentJointAngles = data.joint_state;
      this._updateJointTables(data.joint_state);
      this._syncJoggerSliders(data.joint_state);
      this._renderJoggerSliders();
      for (const [rid, frames] of Object.entries(data.frames)) {
        if (this.robots[rid]) {
          this.robots[rid].updatePose(frames.frames_world, frames.tcp_R, 'open');
        }
      }
      if (data.clearance) {
        this.laser.update(
          data.clearance.min_distance_mm,
          data.clearance.endpoint_a,
          data.clearance.endpoint_b
        );
      }
      this.log('✓ All robots returned to mechanical HOME positions.', 'success');
    }

    // 5. Handover Inspector 3D Preview Result
    else if (data.cmd === 'preview_handover_result') {
      this.currentJointAngles = data.joint_state;
      this._updateJointTables(data.joint_state);
      for (const [rid, frames] of Object.entries(data.frames)) {
        if (this.robots[rid]) {
          this.robots[rid].updatePose(frames.frames_world, frames.tcp_R, 'open');
        }
      }
      if (data.clearance) {
        this.laser.update(
          data.clearance.min_distance_mm,
          data.clearance.endpoint_a,
          data.clearance.endpoint_b
        );
      }
      this.dom.hoCoords.textContent = `[${data.handover_xyz.join(', ')}] mm`;
      this.dom.hoMode.textContent = `${data.approach_mode} (${data.standoff_mm}mm)`;
      this.dom.hoStandoff.textContent = `${data.standoff_mm} mm (Clearance: ${data.clearance.min_distance_mm} mm)`;
      this.log(`[Handover Preview] 3D posed ${data.pair} in ${data.which} posture.`, 'info');
    }

    // 6. Manual Joint Jogger Result
    else if (data.cmd === 'manual_jog_result') {
      this.currentJointAngles = data.joint_state;
      this._updateJointTables(data.joint_state);
      this._syncJoggerSliders(data.joint_state);
      for (const [rid, frames] of Object.entries(data.frames)) {
        if (this.robots[rid]) {
          const gState = this.activeGripperStates[rid] || 'open';
          this.robots[rid].updatePose(frames.frames_world, frames.tcp_R, gState);
        }
      }
      if (data.clearance) {
        this.laser.update(
          data.clearance.min_distance_mm,
          data.clearance.endpoint_a,
          data.clearance.endpoint_b
        );
      }
      if (data.frames[this.activeJogRobot]) {
        const tcp = data.frames[this.activeJogRobot].tcp_world;
        this.dom.jogTcpWorld.textContent = `X: ${tcp[0].toFixed(1)}, Y: ${tcp[1].toFixed(1)}, Z: ${tcp[2].toFixed(1)} mm`;
      }
    }

    // 7. Gripper Toggle Result
    else if (data.cmd === 'gripper_result') {
      this.activeGripperStates[data.robot_id] = data.state;
      if (this.robots[data.robot_id]) {
        this.robots[data.robot_id].setGripperState(data.state);
      }
      this.dom.btnJogGripper.innerHTML = `<i class="fa fa-hand"></i> Toggle Gripper (${data.state.toUpperCase()})`;
      this.log(`[Gripper] ${data.robot_id.toUpperCase()} gripper set to ${data.state.toUpperCase()}.`, 'info');
    }

    // 8. Single-Robot Direct IK Result (testGUI 1, 2, 3)
    else if (data.cmd === 'solve_single_ik_result') {
      if (!data.success) {
        this.dom.singleIkFeedback.textContent = data.error;
        this.dom.singleIkFeedback.style.color = '#FF0055';
        this.log(`[Single IK] ${data.error}`, 'error');
        return;
      }
      this.currentJointAngles = data.joint_state;
      this._updateJointTables(data.joint_state);
      this._syncJoggerSliders(data.joint_state);
      this._renderJoggerSliders();
      for (const [rid, frames] of Object.entries(data.frames)) {
        if (this.robots[rid]) {
          const gState = this.activeGripperStates[rid] || 'open';
          this.robots[rid].updatePose(frames.frames_world, frames.tcp_R, gState);
        }
      }
      if (data.clearance) {
        this.laser.update(
          data.clearance.min_distance_mm,
          data.clearance.endpoint_a,
          data.clearance.endpoint_b
        );
      }
      if (data.frames[this.activeJogRobot]) {
        const tcp = data.frames[this.activeJogRobot].tcp_world;
        this.dom.jogTcpWorld.textContent = `X: ${tcp[0].toFixed(1)}, Y: ${tcp[1].toFixed(1)}, Z: ${tcp[2].toFixed(1)} mm`;
      }
      this.dom.singleIkFeedback.textContent = `✓ Solved! Target [${data.target_xyz.join(', ')}] mm | Error: ${data.error_mm} mm`;
      this.dom.singleIkFeedback.style.color = '#00FF9D';
      this.log(
        `[Single IK] ${data.robot_id.toUpperCase()} posed to [${data.target_xyz.join(', ')}] mm (error: ${data.error_mm} mm).`,
        'success'
      );
    }
  }

  _setupRobots(robotConfigs) {
    for (const [rid, cfg] of Object.entries(robotConfigs)) {
      this.robots[rid] = new RobotArmModel(rid, cfg, this.scene.scene);
      this.robots[rid].updatePose(cfg.frames_world, cfg.tcp_R, 'open');
    }
    this._updateJointTables({
      robot1: robotConfigs.robot1.home_deg,
      robot2: robotConfigs.robot2.home_deg,
      robot3: robotConfigs.robot3.home_deg,
    });
    this._renderDhTable();
  }

  _updateHandoverCard(handoverPoses) {
    const p12 = handoverPoses['robot3_robot1'] || handoverPoses['robot1_robot2'];
    if (p12) {
      this.dom.hoCoords.textContent = `[${p12.handover_position_world.join(', ')}] mm`;
      this.dom.hoMode.textContent = `${p12.approach_mode} (${p12.standoff_distance_mm}mm)`;
      this.dom.hoStandoff.textContent = `${p12.standoff_distance_mm} mm (Clr: ${p12.handover_clearance_mm}mm)`;
      this.scene.setHandoverPosition(p12.handover_position_world);
    }
  }

  _updateJointTables(jointState) {
    const buildRows = (tableElem, qArr) => {
      tableElem.innerHTML = '';
      qArr.forEach((val, i) => {
        const row = document.createElement('div');
        row.className = 'joint-row';
        row.innerHTML = `<span>Joint ${i + 1}</span><span class="joint-val">${val.toFixed(1)}°</span>`;
        tableElem.appendChild(row);
      });
    };

    if (jointState.robot1) buildRows(this.dom.jointR1Table, jointState.robot1);
    if (jointState.robot2) buildRows(this.dom.jointR2Table, jointState.robot2);
    if (jointState.robot3) buildRows(this.dom.jointR3Table, jointState.robot3);
  }

  _syncJoggerSliders(jointState) {
    if (!jointState) return;
    const rid = this.activeJogRobot;
    const angles = jointState[rid];
    if (!angles) return;
    angles.forEach((angle, idx) => {
      const slider = document.getElementById(`jog-slider-${rid}-${idx}`);
      const valLabel = document.getElementById(`jog-val-${rid}-${idx}`);
      if (slider && document.activeElement !== slider) {
        slider.value = Math.round(angle);
      }
      if (valLabel) {
        valLabel.textContent = `${angle.toFixed(1)}°`;
      }
    });
  }

  _renderJoggerSliders() {
    const rid = this.activeJogRobot;
    const cfg = this.robotConfigs[rid];
    if (!cfg) return;

    const currentQ = this.currentJointAngles[rid] || cfg.home_deg;
    this.dom.joggerSlidersContainer.innerHTML = '';

    const jointNames5 = ['J1: Base Yaw', 'J2: Shoulder Pitch', 'J3: Elbow Pitch', 'J4: Wrist Pitch', 'J5: Wrist Roll'];
    const jointNames6 = ['J1: Base Yaw', 'J2: Shoulder Pitch', 'J3: Elbow Pitch', 'J4: Wrist Pitch', 'J5: Wrist Roll', 'J6: Wrist Roll 2'];
    const names = cfg.dof === 6 ? jointNames6 : jointNames5;

    currentQ.forEach((angle, idx) => {
      const limits = cfg.joint_limits_deg[idx];
      const row = document.createElement('div');
      row.className = 'jogger-joint-row';

      row.innerHTML = `
        <div class="jogger-joint-header">
          <span class="jogger-joint-name">${names[idx] || 'Joint ' + (idx + 1)}</span>
          <span class="jogger-joint-val" id="jog-val-${rid}-${idx}">${angle.toFixed(1)}°</span>
        </div>
        <input type="range" class="jogger-slider" id="jog-slider-${rid}-${idx}"
               min="${limits[0]}" max="${limits[1]}" step="1" value="${Math.round(angle)}">
      `;

      const slider = row.querySelector('.jogger-slider');
      slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        row.querySelector(`#jog-val-${rid}-${idx}`).textContent = `${val.toFixed(1)}°`;
        this.send({
          cmd: 'manual_jog_joint',
          robot_id: rid,
          joint_idx: idx,
          angle_deg: val,
        });
      });

      this.dom.joggerSlidersContainer.appendChild(row);
    });

    // Update TCP display
    if (cfg.tcp_world) {
      this.dom.jogTcpWorld.textContent = `X: ${cfg.tcp_world[0].toFixed(1)}, Y: ${cfg.tcp_world[1].toFixed(1)}, Z: ${cfg.tcp_world[2].toFixed(1)} mm`;
    }
  }

  _renderDhTable() {
    if (!this.dom.dhParamsTbody) return;
    const cfg = this.robotConfigs[this.activeJogRobot];
    if (!cfg || !cfg.dh_table) return;
    this.dom.dhParamsTbody.innerHTML = '';
    cfg.dh_table.forEach((row) => {
      const tr = document.createElement('tr');
      tr.style.borderBottom = '1px solid rgba(255,255,255,0.08)';
      tr.innerHTML = `
        <td style="padding:4px 6px; font-weight:700; color:var(--robot1-cyan);">${row.joint}</td>
        <td style="padding:4px 6px;">${row.theta_offset_deg}°</td>
        <td style="padding:4px 6px;">${row.d_mm}</td>
        <td style="padding:4px 6px;">${row.a_mm}</td>
        <td style="padding:4px 6px;">${row.alpha_deg}°</td>
        <td style="padding:4px 6px; color:var(--text-muted);">${row.limit_min_deg}° / ${row.limit_max_deg}°</td>
      `;
      this.dom.dhParamsTbody.appendChild(tr);
    });
  }

  _getCurrentTargetKey() {
    const px = parseFloat(this.dom.pickX.value) || 0;
    const py = parseFloat(this.dom.pickY.value) || 0;
    const pz = parseFloat(this.dom.pickZ.value) || 0;
    const pyaw = parseFloat(this.dom.pickYaw.value) || 0;
    const plx = parseFloat(this.dom.placeX.value) || 0;
    const ply = parseFloat(this.dom.placeY.value) || 0;
    const plz = parseFloat(this.dom.placeZ.value) || 0;
    const plyaw = parseFloat(this.dom.placeYaw.value) || 0;
    return `${px},${py},${pz},${pyaw}|${plx},${ply},${plz},${plyaw}`;
  }

  executeOrSimulateTask(isRealRobot = false) {
    const curKey = this._getCurrentTargetKey();
    if (this.currentPlanSummary && this.currentPlanKey === curKey) {
      if (isRealRobot) {
        this.dom.safetyClearanceVal.textContent = `${this.currentPlanSummary.min_overall_clearance_mm} mm`;
        this.dom.safetyModal.classList.add('active');
      } else {
        // Show Solution Summary Confirmation Dialog (matching Stage 3 Test 2)
        this._showSolutionSummaryDialog(this.currentPlanSummary);
      }
      return;
    }
    // Targets changed or not yet planned: request plan with auto-action
    this.requestPlan({ autoSimulate: !isRealRobot, autoExecute: isRealRobot });
  }

  requestPlan(opts = {}) {
    const pickXyz = [
      parseFloat(this.dom.pickX.value),
      parseFloat(this.dom.pickY.value),
      parseFloat(this.dom.pickZ.value),
    ];
    const pickYaw = parseFloat(this.dom.pickYaw.value) || 0;

    const placeXyz = [
      parseFloat(this.dom.placeX.value),
      parseFloat(this.dom.placeY.value),
      parseFloat(this.dom.placeZ.value),
    ];
    const placeYaw = parseFloat(this.dom.placeYaw.value) || 0;

    this.pendingPlan = {
      cmd: 'plan_trajectory',
      pick_xyz: pickXyz,
      pick_yaw: pickYaw,
      place_xyz: placeXyz,
      place_yaw: placeYaw,
      autoSimulate: !!opts.autoSimulate,
      autoExecute: !!opts.autoExecute,
    };

    // Show loading banner & disable buttons
    this.dom.planLoadingBanner.style.display = 'flex';
    this.dom.planLoadingTitle.textContent = 'Computing Kinematics...';
    this.dom.planLoadingSubtitle.textContent = 'Solving reachability & adaptive tilts across robots...';
    if (this.dom.btnPlan) this.dom.btnPlan.disabled = true;
    if (this.dom.btnSimulateTask) this.dom.btnSimulateTask.disabled = true;
    if (this.dom.btnExecuteTask) this.dom.btnExecuteTask.disabled = true;
    if (this.dom.btnPlan) {
      this.dom.btnPlan.innerHTML = '<span class="spinner-neon" style="width:14px;height:14px;border-width:2px;display:inline-block;margin-right:6px;"></span> Solving Kinematics...';
    }

    this.log(`Requesting plan for Pick [${pickXyz}] -> Place [${placeXyz}]...`, 'info');
    this.send(this.pendingPlan);
  }

  _showRobotChooserModal(data) {
    this.dom.chooserPickContainer.innerHTML = '';
    this.dom.chooserPlaceContainer.innerHTML = '';

    data.pick_candidates.forEach((rid, idx) => {
      const card = document.createElement('div');
      card.className = `robot-choice-card pick-choice-card ${idx === 0 ? 'selected' : ''}`;
      card.setAttribute('data-robot', rid);
      const det = data.pick_details[rid];
      card.innerHTML = `<div style="font-weight:700; color:#00F0FF;">${rid.toUpperCase()}</div><div style="font-size:11px; color:#94A3B8;">${det.solver} (tilt ${det.tilt || 0}°)</div>`;
      card.addEventListener('click', () => {
        document.querySelectorAll('.pick-choice-card').forEach((c) => c.classList.remove('selected'));
        card.classList.add('selected');
      });
      this.dom.chooserPickContainer.appendChild(card);
    });

    data.place_candidates.forEach((rid, idx) => {
      const card = document.createElement('div');
      card.className = `robot-choice-card place-choice-card ${idx === 0 ? 'selected' : ''}`;
      card.setAttribute('data-robot', rid);
      const det = data.place_details[rid];
      card.innerHTML = `<div style="font-weight:700; color:#BD00FF;">${rid.toUpperCase()}</div><div style="font-size:11px; color:#94A3B8;">${det.solver} (tilt ${det.tilt || 0}°)</div>`;
      card.addEventListener('click', () => {
        document.querySelectorAll('.place-choice-card').forEach((c) => c.classList.remove('selected'));
        card.classList.add('selected');
      });
      this.dom.chooserPlaceContainer.appendChild(card);
    });

    this.dom.chooserModal.classList.add('active');
  }

  _showSolutionSummaryDialog(ps) {
    this.dom.summaryRoute.textContent = `${ps.pick_robot.toUpperCase()} ➔ ${ps.place_robot.toUpperCase()}`;
    this.dom.summaryPickSolver.textContent = `${ps.pick_robot.toUpperCase()} (${ps.pick_solver}, tilt ${ps.pick_tilt}°)`;
    this.dom.summaryPlaceSolver.textContent = `${ps.place_robot.toUpperCase()} (${ps.place_solver}, tilt ${ps.place_tilt}°)`;
    this.dom.summaryHoMode.textContent = ps.needs_handover
      ? `TOP-OF-WORKSPACE (${ps.approach_mode})`
      : 'DIRECT PICK & PLACE (NO HANDOVER)';
    this.dom.summaryHoCoords.textContent = ps.handover_xyz
      ? `[${ps.handover_xyz.join(', ')}] mm`
      : 'N/A';
    this.dom.summaryStandoff.textContent = ps.standoff_mm ? `${ps.standoff_mm} mm` : 'N/A';
    this.dom.summaryClearance.textContent = `${ps.min_overall_clearance_mm} mm (${ps.all_phases_safe ? '> 45.0 mm SAFE' : 'WARNING: NEAR LIMIT'})`;
    this.dom.summaryClearance.style.color = ps.all_phases_safe ? '#00FF9D' : '#FFB800';

    this.dom.solutionModal.classList.add('active');
  }

  // ─────────────────────────────────────────────────────────────
  // Playback & Simulation Engine (60 FPS)
  // ─────────────────────────────────────────────────────────────
  togglePlayback() {
    if (this.isPlaying) {
      this.pausePlayback();
    } else {
      const curKey = this._getCurrentTargetKey();
      if (!this.currentPlanSummary || this.currentPlanKey !== curKey) {
        // Targets changed, automatically plan and simulate!
        this.executeOrSimulateTask(false);
      } else {
        this.startPlayback();
      }
    }
  }

  startPlayback() {
    if (this.trajectorySteps.length === 0) return;
    if (this.currentStepIdx >= this.trajectorySteps.length - 1) {
      this.currentStepIdx = 0;
    }
    this.isPlaying = true;
    this.dom.btnPlayPause.innerHTML = '<i class="fa fa-pause"></i> Pause';
    this.dom.btnPlayPause.classList.remove('btn-success');
    this.dom.btnPlayPause.classList.add('btn-primary');

    this.log('▶ [3D Simulation] Starting 9-Phase Stage 3 Trajectory Preview...', 'info');

    const stepIntervalMs = Math.round(50 / this.playbackSpeed);

    if (this.playbackTimer) clearInterval(this.playbackTimer);
    this.playbackTimer = setInterval(() => {
      this.currentStepIdx++;
      if (this.currentStepIdx >= this.trajectorySteps.length) {
        this.currentStepIdx = this.trajectorySteps.length - 1;
        this.pausePlayback();
        this.log('✓ [3D Simulation] Complete! All trajectory phases safely completed.', 'success');
        this.dom.phaseBadge.textContent = '✓ Simulation Complete';
      }
      this.dom.timelineSlider.value = this.currentStepIdx;
      this.applyStep(this.currentStepIdx);
    }, stepIntervalMs);
  }

  pausePlayback() {
    this.isPlaying = false;
    if (this.playbackTimer) {
      clearInterval(this.playbackTimer);
      this.playbackTimer = null;
    }
    this.dom.btnPlayPause.innerHTML = '<i class="fa fa-play"></i> Simulate';
    this.dom.btnPlayPause.classList.remove('btn-primary');
    this.dom.btnPlayPause.classList.add('btn-success');
  }

  resetPlayback() {
    this.pausePlayback();
    this.currentStepIdx = 0;
    this._lastLoggedPhase = null;
    this.dom.timelineSlider.value = 0;
    this.applyStep(0);
  }

  applyStep(stepIdx) {
    if (!this.trajectorySteps || stepIdx < 0 || stepIdx >= this.trajectorySteps.length) return;
    const step = this.trajectorySteps[stepIdx];

    // Log phase transitions to System Log Console (matching stage3_test2_gui)
    if (step.phase_title && step.phase_title !== this._lastLoggedPhase) {
      this._lastLoggedPhase = step.phase_title;
      this.log(`>>> ${step.phase_title}`, 'info');
    }

    // Update joint angles table
    this._updateJointTables(step.joint_state);
    this._syncJoggerSliders(step.joint_state);

    // Update 3D robot arm poses
    for (const [rid, frames] of Object.entries(step.frames)) {
      if (this.robots[rid]) {
        const gState = step.grippers ? step.grippers[rid] : 'open';
        this.robots[rid].updatePose(frames.frames_world, frames.tcp_R, gState);
      }
    }

    // Update dynamic clearance laser
    if (step.clearance) {
      this.laser.update(
        step.clearance.min_distance_mm,
        step.clearance.endpoint_a,
        step.clearance.endpoint_b
      );
    }

    // Update workpiece attached/moving position
    if (step.workpiece) {
      if (step.workpiece.attached_to && step.frames[step.workpiece.attached_to]) {
        const tcpWorld = step.frames[step.workpiece.attached_to].tcp_world;
        this.scene.setWorkpiecePosition(tcpWorld, [0, 0, 0]);
      } else {
        this.scene.setWorkpiecePosition(step.workpiece.position, step.workpiece.rotation);
      }
    }

    // Update scrubber UI
    this.dom.timelineSlider.value = stepIdx;
    this.dom.stepCounter.textContent = `Step ${stepIdx + 1} / ${this.trajectorySteps.length}`;
    this.dom.phaseBadge.textContent = step.phase_title || `Phase ${step.phase_number}`;
  }

  log(msg, level = 'info') {
    const entry = document.createElement('div');
    entry.className = `log-entry log-${level}`;
    const timeStr = new Date().toTimeString().split(' ')[0];
    entry.innerHTML = `<span class="log-time">[${timeStr}]</span> ${msg}`;
    this.dom.terminalLog.appendChild(entry);
    this.dom.terminalLog.scrollTop = this.dom.terminalLog.scrollHeight;
  }
}

// Bootstrap Application on DOM Loaded
window.addEventListener('DOMContentLoaded', () => {
  window.app = new WebTwinApp();
});
