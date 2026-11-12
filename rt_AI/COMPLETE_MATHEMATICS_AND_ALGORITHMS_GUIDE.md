# Mathematical Formulations, Kinematic Equations, and Numerical Methods
## Complete Theoretical & Algorithmic Specification of the rt_AI Multi-Robot System

This document provides the definitive mathematical derivation and algorithmic specification of every method used across the **rt_AI** robotics project. It is specifically structured for academic examination, technical defense, and engineering reference.

---

# Table of Contents
1. [Kinematic Foundations & Modified Denavit-Hartenberg (MDH) Modeling](#1-kinematic-foundations--modified-denavit-hartenberg-mdh-modeling)
   - 1.1 [Modified DH (Craig) Transformation Formulation](#11-modified-dh-craig-transformation-formulation)
   - 1.2 [Active Joint to MDH Parameter Mapping](#12-active-joint-to-mdh-parameter-mapping)
   - 1.3 [Forward Kinematics (FK) Cumulative Matrix Chaining](#13-forward-kinematics-fk-cumulative-matrix-chaining)
   - 1.4 [Analytic Geometric Jacobian Derivation](#14-analytic-geometric-jacobian-derivation)
   - 1.5 [Robotic DH Parameters & World Placement Reference](#15-robotic-dh-parameters--world-placement-reference)
2. [Individual GUI Kinematics & Motion Control](#2-individual-gui-kinematics--motion-control)
   - 2.1 [Position-Only Numerical Inverse Kinematics (DLS / Levenberg-Marquardt)](#21-position-only-numerical-inverse-kinematics-dls--levenberg-marquardt)
   - 2.2 [Nullspace Posture Projection & Joint Limit Avoidance](#22-nullspace-posture-projection--joint-limit-avoidance)
   - 2.3 [Multi-Start Deterministic Seeding & Objective Scoring](#23-multi-start-deterministic-seeding--objective-scoring)
   - 2.4 [Pick-and-Place Cartesian Straight-Line Descent with Orientation Lock](#24-pick-and-place-cartesian-straight-line-descent-with-orientation-lock)
   - 2.5 [Monte Carlo Workspace Boundary Point Estimation](#25-monte-carlo-workspace-boundary-point-estimation)
3. [Stage 3 Test 2: Automated Kinematic Handover via TOPWS (Top of Workspace)](#3-stage-3-test-2-automated-kinematic-handover-via-topws-top-of-workspace)
   - 3.1 [Halton Low-Discrepancy Sampling & 3D Voxelization](#31-halton-low-discrepancy-sampling--3d-voxelization)
   - 3.2 [Candidate Selection via Conservative Reach Envelope & Strict Z-Descending Ordering](#32-candidate-selection-via-conservative-reach-envelope--strict-z-descending-ordering)
   - 3.3 [Coupled Handover Orientation Solver (Analytical Gradients & L-BFGS-B)](#33-coupled-handover-orientation-solver-analytical-gradients--l-bfgs-b)
   - 3.4 [Discrete 1D Optimization of Wrist Roll for Orthogonal Finger Mating](#34-discrete-1d-optimization-of-wrist-roll-for-orthogonal-finger-mating)
   - 3.5 [Cartesian Approach Corridors & Standoff Vector Formulation](#35-cartesian-approach-corridors--standoff-vector-formulation)
   - 3.6 [Dual-Synchronized vs Asymmetric Corridor Scheduling](#36-dual-synchronized-vs-asymmetric-corridor-scheduling)
4. [Multi-Robot Pick-and-Place & Synchronized Trajectory Planning](#4-multi-robot-pick-and-place--synchronized-trajectory-planning)
   - 4.1 [Stage 2.5 Object-Pose IK with Adaptive Tilt Synthesis](#41-stage-25-object-pose-ik-with-adaptive-tilt-synthesis)
   - 4.2 [5-DOF Task-Priority Nullspace Inverse Kinematics](#42-5-dof-task-priority-nullspace-inverse-kinematics)
   - 4.3 [6-DOF Balanced Units Pose Inverse Kinematics](#43-6-dof-balanced-units-pose-inverse-kinematics)
   - 4.4 [Lexicographic Feasibility Filter & Normalized Cost Function](#44-lexicographic-feasibility-filter--normalized-cost-function)
   - 4.5 [Minimum Jerk 7th-Order Septic Trajectory Formulation](#45-minimum-jerk-7th-order-septic-trajectory-formulation)
   - 4.6 [9-Phase Synchronized Multi-Robot State Machine](#46-9-phase-synchronized-multi-robot-state-machine)
5. [Whole-Body Link Clearance & Collision Avoidance](#5-whole-body-link-clearance--collision-avoidance)
   - 5.1 [Exact 3D Line-Segment to Line-Segment Distance (Dan Sunday / Lumelsky Algorithm)](#51-exact-3d-line-segment-to-line-segment-distance-dan-sunday--lumelsky-algorithm)
   - 5.2 [All-Pairs Whole-Body Link Clearance Evaluation](#52-all-pairs-whole-body-link-clearance-evaluation)
   - 5.3 [Tooltip / TCP Exclusion Rule for Contact Handover](#53-tooltip--tcp-exclusion-rule-for-contact-handover)
   - 5.4 [Collision-Aware Motion Scheduling: Concurrent vs Serialized Delay](#54-collision-aware-motion-scheduling-concurrent-vs-serialized-delay)
6. [Web Digital Twin Mathematical Architecture](#6-web-digital-twin-mathematical-architecture)
   - 6.1 [Coordinate Frame Transformations: Base $\to$ World $\to$ Three.js](#61-coordinate-frame-transformations-base-to-world-to-threejs)
   - 6.2 [Procedural Mesh Transformations: Link Cylinders & Flange Quaternions](#62-procedural-mesh-transformations-link-cylinders--flange-quaternions)
   - 6.3 [Real-Time Dynamic Laser Clearance Gauge Formulation](#63-real-time-dynamic-laser-clearance-gauge-formulation)

---

# 1. Kinematic Foundations & Modified Denavit-Hartenberg (MDH) Modeling

### 1.1 Modified DH (Craig) Transformation Formulation

The project adopts the **Modified Denavit-Hartenberg (MDH)** convention (Craig convention), where the coordinate frame $i$ has its origin located on axis $i$, with link parameters defined relative to link $i-1$:
- $\alpha_{i-1}$: Link twist — angle from $\hat{Z}_{i-1}$ to $\hat{Z}_i$ measured about $\hat{X}_{i-1}$.
- $a_{i-1}$: Link length — offset distance along $\hat{X}_{i-1}$ from $\hat{Z}_{i-1}$ to $\hat{Z}_i$.
- $d_i$: Link offset — distance along $\hat{Z}_i$ from $\hat{X}_{i-1}$ to $\hat{X}_i$.
- $\theta_i$: Joint angle — angle from $\hat{X}_{i-1}$ to $\hat{X}_i$ measured about $\hat{Z}_i$.

The homogeneous transformation matrix $^{i-1}T_i \in \text{SE}(3)$ from frame $i-1$ to frame $i$ is evaluated by the exact sequence:
$$^{i-1}T_i = \text{Rot}(\hat{X}, \alpha_{i-1}) \cdot \text{Trans}(\hat{X}, a_{i-1}) \cdot \text{Rot}(\hat{Z}, \theta_i) \cdot \text{Trans}(\hat{Z}, d_i)$$

Expanding into matrix form:
$$^{i-1}T_i = \begin{bmatrix}
\cos\theta_i & -\sin\theta_i & 0 & a_{i-1} \\
\sin\theta_i \cos\alpha_{i-1} & \cos\theta_i \cos\alpha_{i-1} & -\sin\alpha_{i-1} & -d_i \sin\alpha_{i-1} \\
\sin\theta_i \sin\alpha_{i-1} & \cos\theta_i \sin\alpha_{i-1} & \cos\alpha_{i-1} & d_i \cos\alpha_{i-1} \\
0 & 0 & 0 & 1
\end{bmatrix}$$

---

### 1.2 Active Joint to MDH Parameter Mapping

Physical actuator control angles $\mathbf{q} = [q_1, q_2, \dots, q_n]^T$ map to frame variables $\theta_i$ through configurable offsets and directional multipliers:
$$\theta_i = \theta_{\text{offset}, i} + \mu_i \cdot q_{\text{index}(i)}$$
Where:
- $\theta_{\text{offset}, i} \in \mathbb{R}$ is the fixed geometric zero-offset (in radians).
- $\mu_i \in \{+1.0, -1.0, 0.0\}$ is the joint direction multiplier.
- $\text{index}(i) \in \{0, 1, \dots, n-1, -1\}$ denotes the active joint index; if $\text{index}(i) = -1$, the frame is a fixed intermediate link offset ($\mu_i = 0$).

---

### 1.3 Forward Kinematics (FK) Cumulative Matrix Chaining

For a serial manipulator with $M$ frames, the cumulative transformation from base frame $\mathcal{F}_0$ to flange frame $\mathcal{F}_M$ is:
$$^0T_M(\mathbf{q}) = \prod_{i=1}^M \, ^{i-1}T_i(\theta_i(q)) = \, ^0T_1 \cdot \, ^1T_2 \cdots \, ^{M-1}T_M = \begin{bmatrix}
R_M(\mathbf{q}) & \mathbf{p}_M(\mathbf{q}) \\
\mathbf{0}_{1 \times 3} & 1
\end{bmatrix}$$

The Tool Center Point (TCP) is situated at a rigid physical translation $\mathbf{t}_{\text{offset}} = [dx, dy, dz]^T$ relative to the final flange:
$$^0T_{\text{tcp}}(\mathbf{q}) = \, ^0T_M(\mathbf{q}) \cdot \begin{bmatrix}
I_{3 \times 3} & \mathbf{t}_{\text{offset}} \\
\mathbf{0}_{1 \times 3} & 1
\end{bmatrix} = \begin{bmatrix}
R_{\text{tcp}}(\mathbf{q}) & \mathbf{p}_{\text{tcp}}(\mathbf{q}) \\
\mathbf{0}_{1 \times 3} & 1
\end{bmatrix}$$

Where:
- $\mathbf{p}_{\text{tcp}}(\mathbf{q}) = \mathbf{p}_M(\mathbf{q}) + R_M(\mathbf{q}) \mathbf{t}_{\text{offset}} \in \mathbb{R}^3$ is the TCP position in base coordinates.
- $R_{\text{tcp}}(\mathbf{q}) = [\mathbf{n}, \mathbf{s}, \mathbf{a}] \in \text{SO}(3)$ represents the end-effector rotation matrix:
  - $\mathbf{n} = R_{:, 0}$: Gripper local $\hat{X}$ axis (finger alignment direction).
  - $\mathbf{s} = R_{:, 1}$: Gripper local $\hat{Y}$ axis.
  - $\mathbf{a} = R_{:, 2}$: Gripper local $\hat{Z}$ axis (approach direction).

---

### 1.4 Analytic Geometric Jacobian Derivation

The $6 \times n$ Geometric Jacobian matrix $J(\mathbf{q})$ maps joint velocities $\dot{\mathbf{q}}$ to TCP translational and angular velocities:
$$\begin{bmatrix} \mathbf{v}_{\text{tcp}} \\ \boldsymbol{\omega}_{\text{tcp}} \end{bmatrix} = J(\mathbf{q}) \dot{\mathbf{q}} = \begin{bmatrix} J_v(\mathbf{q}) \\ J_\omega(\mathbf{q}) \end{bmatrix} \dot{\mathbf{q}}$$

For revolute joints, each column $j \in \{0, \dots, n-1\}$ is evaluated directly from the frame poses:
$$J(:, j) = \begin{bmatrix}
J_v(:, j) \\
J_\omega(:, j)
\end{bmatrix} = \sum_{i \in \mathcal{F}_j} \mu_i \begin{bmatrix}
\hat{\mathbf{z}}_i \times (\mathbf{p}_{\text{tcp}} - \mathbf{p}_i) \\
\hat{\mathbf{z}}_i
\end{bmatrix}$$
Where:
- $\mathcal{F}_j = \{i \mid \text{index}(i) = j\}$ is the set of MDH frames driven by joint $j$.
- $\hat{\mathbf{z}}_i = \, ^0T_i[:3, 2]$ is the unit actuation vector of frame $i$ in base coordinates.
- $\mathbf{p}_i = \, ^0T_i[:3, 3]$ is the origin of frame $i$ in base coordinates.

---

### 1.5 Robotic DH Parameters & World Placement Reference

#### Robot 1 (5-DOF, Articulated Serial Manipulator):
- $\boldsymbol{\alpha} = [0^\circ, 90^\circ, 0^\circ, 0^\circ, -90^\circ, 0^\circ, 0^\circ]$
- $\mathbf{a} = [0, 0, 105, 150, 0, 0, 0]\text{ mm}$
- $\mathbf{d} = [95, 0, 0, 0, 0, 80, 110]\text{ mm}$
- $\boldsymbol{\theta}_{\text{offset}} = [0^\circ, 90^\circ, 0^\circ, -90^\circ, 0^\circ, -90^\circ, -90^\circ]$
- World Base Placement: $\mathbf{t}_{w1} = [-250.0, 0.0, 0.0]^T\text{ mm}$, $\text{RPY} = [0^\circ, 0^\circ, 0^\circ]$

#### Robot 2 (5-DOF, High-Reach Serial Manipulator):
- $\boldsymbol{\alpha} = [0^\circ, 0^\circ, 90^\circ, 0^\circ, -90^\circ, 90^\circ, 90^\circ, 90^\circ]$
- $\mathbf{a} = [0, 20.6, 0, 130, 0, 0, 36.7, 0]\text{ mm}$
- $\mathbf{d} = [80, 45, 0, 0, 90, 0, 0, 130]\text{ mm}$
- $\boldsymbol{\theta}_{\text{offset}} = [0^\circ, 0^\circ, 90^\circ, -90^\circ, -90^\circ, 90^\circ, 90^\circ, 180^\circ]$
- World Base Placement: $\mathbf{t}_{w2} = [0.0, 250.0, 0.0]^T\text{ mm}$, $\text{RPY} = [0^\circ, 0^\circ, 180^\circ]$

#### Robot 3 (6-DOF, Spherical Wrist Manipulator):
- $\boldsymbol{\alpha} = [0^\circ, 90^\circ, 0^\circ, 0^\circ, 90^\circ, -90^\circ, 90^\circ, 0^\circ]$
- $\mathbf{a} = [0, 0, 60, 115, 60, 75, 70, 0]\text{ mm}$
- $\mathbf{d} = [65, 0, 0, 0, 0, 0, 0, 120]\text{ mm}$
- $\boldsymbol{\theta}_{\text{offset}} = [0^\circ, 90^\circ, 0^\circ, 0^\circ, 0^\circ, 0^\circ, 0^\circ, 0^\circ]$
- World Base Placement: $\mathbf{t}_{w3} = [250.0, 0.0, 0.0]^T\text{ mm}$, $\text{RPY} = [0^\circ, 0^\circ, 90^\circ]$

---

# 2. Individual GUI Kinematics & Motion Control

### 2.1 Position-Only Numerical Inverse Kinematics (DLS / Levenberg-Marquardt)

In the standalone GUI workstation (`robot_gui.py`, `robot_gui_test.py`), Cartesian positioning targets $\mathbf{p}_{\text{target}} \in \mathbb{R}^3$ are solved using regularized Levenberg-Marquardt / Damped Least Squares (DLS).

#### Error Vector:
$$\mathbf{e}_p(\mathbf{q}) = \mathbf{p}_{\text{target}} - \mathbf{p}_{\text{tcp}}(\mathbf{q})$$

#### Damped Pseudoinverse Formulation:
At configuration $\mathbf{q}_k$, the position Jacobian $J_v \in \mathbb{R}^{3 \times n}$ has Moore-Penrose damped inverse:
$$J_v^\dagger = J_v^T (J_v J_v^T + \lambda^2 I_3)^{-1}$$
Where $\lambda = 0.5$ is the damping constant preventing instability near singular configurations.

---

### 2.2 Nullspace Posture Projection & Joint Limit Avoidance

For redundant manipulators ($n > 3$), secondary joint centering objectives are projected into the nullspace of the primary task so that joint limits are avoided without perturbing the end-effector:

#### Secondary Posture Cost Function:
$$H(\mathbf{q}) = \sum_{j=1}^n \left( \frac{q_j - q_{\text{mid}, j}}{\frac{1}{2}(q_{\max, j} - q_{\min, j})} \right)^2$$
Where $q_{\text{mid}, j} = \frac{1}{2}(q_{\min, j} + q_{\max, j})$.

#### Nullspace Projection Operator:
$$N = I_n - J_v^\dagger J_v \in \mathbb{R}^{n \times n}$$

#### Total Iterative Joint Update:
$$\Delta \mathbf{q} = J_v^\dagger \mathbf{e}_p - k_{\text{posture}} N \nabla H(\mathbf{q})$$
$$\mathbf{q}_{k+1} = \text{clip}\left(\mathbf{q}_k + \alpha \Delta \mathbf{q}, \mathbf{q}_{\min}, \mathbf{q}_{\max}\right)$$
Where $\alpha = 0.7$ is the gradient step size, $k_{\text{posture}} = 0.01$, and iteration terminates when $\|\mathbf{e}_p\| < 0.05\text{ mm}$ or iterations reach 500.

---

### 2.3 Multi-Start Deterministic Seeding & Objective Scoring

To escape local minima and joint limit clamping, the solver runs $M = 32$ parallel starts:
$$\mathbf{q}_{\text{seed}}^{(0)} = \mathbf{0}, \quad \mathbf{q}_{\text{seed}}^{(1)} = \mathbf{q}_{\text{current}}, \quad \mathbf{q}_{\text{seed}}^{(k)} = \mathbf{q}_{\min} + \mathbf{r}_k \odot (\mathbf{q}_{\max} - \mathbf{q}_{\min}), \quad \mathbf{r}_k \sim \mathcal{U}(0, 1)^n$$

All converged solutions satisfying $\|\mathbf{e}_p\| \le 0.10\text{ mm}$ are scored via:
$$\text{Score}(\mathbf{q}) = 1000 \cdot \|\mathbf{e}_p\| + 1.0 \cdot H(\mathbf{q}) + 0.1 \cdot \|\mathbf{q} - \mathbf{q}_{\text{current}}\|$$
The configuration minimizing $\text{Score}(\mathbf{q})$ is selected.

---

### 2.4 Pick-and-Place Cartesian Straight-Line Descent with Orientation Lock

During individual GUI pick-and-place execution, the vertical approach and descent are constrained to a straight Cartesian line in operational space rather than a curved joint-space arc.

#### Cartesian Interpolation Path:
Between the elevated approach point $\mathbf{p}_{\text{start}} = [x, y, z + h_{\text{approach}}]^T$ ($h_{\text{approach}} = 50\text{ mm}$) and the grasp point $\mathbf{p}_{\text{target}} = [x, y, z]^T$, $M = 20$ linearly interpolated waypoints are constructed:
$$\mathbf{p}_{\text{interp}}(k) = \mathbf{p}_{\text{start}} + \frac{k}{M} (\mathbf{p}_{\text{target}} - \mathbf{p}_{\text{start}}), \quad k \in \{1, \dots, M\}$$

#### Orientation Locking & Jacobian Closed-Loop Tracking:
At each waypoint, the initial orientation $R_{\text{start}}$ is held fixed:
$$\mathbf{e} = \begin{bmatrix} \mathbf{p}_{\text{interp}}(k) - \mathbf{p}_{\text{tcp}}(\mathbf{q}) \\ \mathbf{e}_R(R_{\text{tcp}}(\mathbf{q}), R_{\text{start}}) \end{bmatrix} \in \mathbb{R}^6$$

Where the orientation error $\mathbf{e}_R$ is evaluated via quaternion conjugate multiplication:
$$q_e = q_{\text{start}} \otimes q_{\text{current}}^{-1} = [w_e, \mathbf{v}_e]^T$$
$$\mathbf{e}_R = 2 \cdot \text{atan2}(\|\mathbf{v}_e\|, |w_e|) \cdot \text{sgn}(w_e) \frac{\mathbf{v}_e}{\|\mathbf{v}_e\|}$$

#### Adaptive Damped Least Squares:
For 5-DOF manipulators, a diagonal task-weighting matrix $W = \text{diag}(1, 1, 1, 0.01, 0.01, 0.01)$ prioritizes position:
$$(J^T W J + \lambda^2 I_n) \Delta \mathbf{q} = J^T W \mathbf{e}$$
Where damping adapts to the minimum singular value $\sigma_{\min}$ of $J$:
$$\lambda = \begin{cases} 0.01 & \text{if } \sigma_{\min} \ge 0.01 \\ \min\left(0.20, 0.01 + 0.1 \left(1 - \frac{\sigma_{\min}}{0.01}\right)^2\right) & \text{if } \sigma_{\min} < 0.01 \end{cases}$$
$$\mathbf{q} \leftarrow \text{clip}(\mathbf{q} + 0.6 \Delta \mathbf{q}, \mathbf{q}_{\min}, \mathbf{q}_{\max})$$

---

### 2.5 Monte Carlo Workspace Boundary Point Estimation

In the GUI visualizer, reachable point cloud boundaries are computed by uniform random sampling over the joint limits:
$$\mathbf{q}_k = \mathbf{q}_{\min} + \mathbf{u}_k \odot (\mathbf{q}_{\max} - \mathbf{q}_{\min}), \quad \mathbf{u}_k \sim \mathcal{U}[0, 1]^n, \quad k \in \{1, \dots, 3000\}$$
$$\mathbf{p}_k = \text{FK}(\mathbf{q}_k)$$
The resulting point cloud is rendered directly in the embedded 3D canvas to visually bound reachable Cartesian space.

---

# 3. Stage 3 Test 2: Automated Kinematic Handover via TOPWS (Top of Workspace)

### 3.1 Halton Low-Discrepancy Sampling & 3D Voxelization

To discover viable handover candidate points, the configuration space of each robot is sampled with $N = 5000$ points generated by a **Halton quasi-random sequence**. 

For joint $k \in \{0, \dots, n-1\}$, the base is the $k$-th prime number $p_k \in \{2, 3, 5, 7, 11, 13\}$. The radical inverse function $\phi_{p}(i)$ expands index $i = \sum_{j=0}^M b_j p^j$:
$$\phi_p(i) = \sum_{j=0}^M b_j p^{-(j+1)}$$
$$q_k(i) = q_{\min, k} + \phi_{p_k}(i) \cdot (q_{\max, k} - q_{\min, k})$$

Each sample is projected to the world frame:
$$\mathbf{p}_{\text{world}}(i) = R_{w, \text{base}} \mathbf{p}_{\text{tcp}}(\mathbf{q}(i)) + \mathbf{t}_{w, \text{base}}$$

Continuous space is partitioned into a discrete 3D voxel grid with voxel resolution $\Delta v = 15\text{ mm}$:
$$\mathbf{i}_{\text{voxel}} = \left\lfloor \frac{\mathbf{p}_{\text{world}} - \mathbf{p}_{\text{origin}}}{\Delta v} \right\rfloor \in \mathbb{Z}^3$$

The common reachable workspace between Robot $A$ and Robot $B$ is the set intersection of populated voxels:
$$\mathcal{V}_{AB} = \mathcal{V}_A \cap \mathcal{V}_B$$
Voxel centers are given by:
$$\mathbf{p}_{\text{center}} = \mathbf{p}_{\text{origin}} + \left(\mathbf{i}_{\text{voxel}} + \frac{1}{2}\right) \Delta v$$

---

### 3.2 Candidate Selection via Conservative Reach Envelope & Strict Z-Descending Ordering

In **Stage 3 Test 2**, candidate points are filtered and ordered strictly by the **TOP_OF_WORKSPACE (TOPWS)** strategy:

1. **Conservative Reach Envelope Filter (90% Boundary)**:
   Points close to the kinematic singularity horizon ($\sum |a_i| + \sum |d_i|$) suffer severe manipulability degradation. Candidates must satisfy:
   $$\|\mathbf{p} - \mathbf{t}_{wA}\| \le 0.90 \cdot R_{\text{reach}, A}$$
   $$\|\mathbf{p} - \mathbf{t}_{wB}\| \le 0.90 \cdot R_{\text{reach}, B}$$
   Where $R_{\text{reach}} = \sum_{i} (|a_i| + |d_i|)$.

2. **Strict Z-Descending Sorting (High Altitude First)**:
   Rather than centroid-based or midpoint sorting, candidate points are strictly sorted descending by vertical height $Z$:
   $$\mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_K \quad \text{such that} \quad Z(\mathbf{p}_1) \ge Z(\mathbf{p}_2) \ge \dots \ge Z(\mathbf{p}_K)$$
   **Why this works**: Selecting candidates at the top of the workspace maximizes clearance from robot base pedestals, tables, and fixtures, guaranteeing that link clearance ($> 150\text{ mm}$) remains far above collision thresholds.

---

### 3.3 Coupled Handover Orientation Solver (Analytical Gradients & L-BFGS-B)

The coupled handover problem requires finding simultaneous joint vectors $\mathbf{q} = [\mathbf{q}_A^T, \mathbf{q}_B^T]^T \in \mathbb{R}^{n_A + n_B}$ such that:
1. Both TCP positions coincide at candidate point $\mathbf{p}_H \in \mathbb{R}^3$: $\mathbf{p}_A(\mathbf{q}_A) = \mathbf{p}_B(\mathbf{q}_B) = \mathbf{p}_H$.
2. The approach vectors $\hat{\mathbf{z}}_A, \hat{\mathbf{z}}_B \in \mathbb{R}^3$ are exactly **anti-parallel**: $\hat{\mathbf{z}}_A \cdot \hat{\mathbf{z}}_B = -1$.
3. The gripper finger alignment vectors $\hat{\mathbf{x}}_A, \hat{\mathbf{x}}_B \in \mathbb{R}^3$ are **mutually orthogonal**: $\hat{\mathbf{x}}_A \cdot \hat{\mathbf{x}}_B = 0$ (preventing finger clash during object transfer).

#### Coupled Objective Loss Function:
$$L(\mathbf{q}_A, \mathbf{q}_B) = 1000 \cdot \left(\|\mathbf{p}_A - \mathbf{p}_H\|^2 + \|\mathbf{p}_B - \mathbf{p}_H\|^2\right) + 150 \cdot (\hat{\mathbf{z}}_A \cdot \hat{\mathbf{z}}_B + 1)^2 + 80 \cdot (\hat{\mathbf{x}}_A \cdot \hat{\mathbf{x}}_B)^2$$

#### Exact Analytical Gradient Derivation:
Using geometric Jacobians transformed into the common world frame:
$$J_{vA} = R_{wA} J_{vA, \text{base}}, \quad J_{\omega A} = R_{wA} J_{\omega A, \text{base}}$$
$$J_{vB} = R_{wB} J_{vB, \text{base}}, \quad J_{\omega B} = R_{wB} J_{\omega B, \text{base}}$$

Recall that for any unit vector $\hat{\mathbf{u}}$ attached to the frame, its derivative with respect to joint angles is given by cross-product operator: $\frac{\partial \hat{\mathbf{u}}}{\partial \mathbf{q}} = -[\hat{\mathbf{u}}]_\times J_\omega$. Therefore:
$$\nabla_{\mathbf{q}_A} (\hat{\mathbf{z}}_A \cdot \hat{\mathbf{z}}_B) = (\hat{\mathbf{z}}_A \times \hat{\mathbf{z}}_B)^T J_{\omega A}$$
$$\nabla_{\mathbf{q}_B} (\hat{\mathbf{z}}_A \cdot \hat{\mathbf{z}}_B) = -(\hat{\mathbf{z}}_A \times \hat{\mathbf{z}}_B)^T J_{\omega B}$$
$$\nabla_{\mathbf{q}_A} (\hat{\mathbf{x}}_A \cdot \hat{\mathbf{x}}_B) = (\hat{\mathbf{x}}_A \times \hat{\mathbf{x}}_B)^T J_{\omega A}$$
$$\nabla_{\mathbf{q}_B} (\hat{\mathbf{x}}_A \cdot \hat{\mathbf{x}}_B) = -(\hat{\mathbf{x}}_A \times \hat{\mathbf{x}}_B)^T J_{\omega B}$$

Hence, the exact analytical gradients $\nabla_{\mathbf{q}_A} L$ and $\nabla_{\mathbf{q}_B} L$ are:
$$\nabla_{\mathbf{q}_A} L = 2000 (\mathbf{p}_A - \mathbf{p}_H)^T J_{vA} + 300 (\hat{\mathbf{z}}_A \cdot \hat{\mathbf{z}}_B + 1)(\hat{\mathbf{z}}_A \times \hat{\mathbf{z}}_B)^T J_{\omega A} + 160 (\hat{\mathbf{x}}_A \cdot \hat{\mathbf{x}}_B)(\hat{\mathbf{x}}_A \times \hat{\mathbf{x}}_B)^T J_{\omega A}$$
$$\nabla_{\mathbf{q}_B} L = 2000 (\mathbf{p}_B - \mathbf{p}_H)^T J_{vB} - 300 (\hat{\mathbf{z}}_A \cdot \hat{\mathbf{z}}_B + 1)(\hat{\mathbf{z}}_A \times \hat{\mathbf{z}}_B)^T J_{\omega B} - 160 (\hat{\mathbf{x}}_A \cdot \hat{\mathbf{x}}_B)(\hat{\mathbf{x}}_A \times \hat{\mathbf{x}}_B)^T J_{\omega B}$$

The solution is solved via **L-BFGS-B (Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bounds)** under box constraints $\mathbf{q} \in [\mathbf{q}_{\min}, \mathbf{q}_{\max}]$ within 70 iterations.

---

### 3.4 Discrete 1D Optimization of Wrist Roll for Orthogonal Finger Mating

When pairing 5-DOF arms with asymmetric wrist configurations (e.g., Robot 1 or Robot 3 paired with Robot 2), the continuous optimizer may find a near-orthogonal solution. To guarantee strict orthogonality ($\hat{\mathbf{x}}_A \cdot \hat{\mathbf{x}}_B \le 0.25$):

The wrist roll joint index $j_{\text{wrist}}$ ($j=4$ for Robot 1, $j=5$ for Robot 3) is isolated, and a 1D line-search over $K = 37$ discrete samples spanning the joint limits $[q_{\min}, q_{\max}]$ is performed:
$$q_{\text{wrist}}^* = \arg\min_{q \in \{q_{\min} + k \frac{\Delta q}{36}\}} \left| \hat{\mathbf{x}}_w(q) \cdot \hat{\mathbf{x}}_{\text{other}} \right|$$
This decouples the roll angle without affecting TCP position coincidence, producing perfectly orthogonal finger mating.

---

### 3.5 Cartesian Approach Corridors & Standoff Vector Formulation

Directly interpolating between an arm's pick pose and the final handover pose risks premature collisions. A linear approach corridor is enforced along each robot's actual TCP $\hat{\mathbf{z}}$ approach axis.

#### Standoff Distance Vector:
With $d_{\text{standoff}} = 50.0\text{ mm}$, the pre-handover Cartesian position is:
$$\mathbf{p}_{\text{pre}, A} = \mathbf{p}_H - d_{\text{standoff}} \cdot \hat{\mathbf{z}}_A$$
$$\mathbf{p}_{\text{pre}, B} = \mathbf{p}_H - d_{\text{standoff}} \cdot \hat{\mathbf{z}}_B$$

#### Corridor Parameterization:
The linear corridor trajectory is discretized into $S = 15$ steps for parameter $s \in [0, 1]$:
$$\mathbf{p}(s) = \mathbf{p}_{\text{pre}} + s \cdot d_{\text{standoff}} \cdot \hat{\mathbf{z}}, \quad R(s) = R_H$$

Continuous waypoints $\mathbf{q}_k$ are solved backward from $\mathbf{q}_H$ using posture-continuity regularization:
$$\mathbf{q}_k = \arg\min_{\mathbf{q}} \left( \|\mathbf{p}_{\text{tcp}}(\mathbf{q}) - \mathbf{p}(s_k)\|^2 + w_{\text{ori}} (1 - \hat{\mathbf{z}}(\mathbf{q}) \cdot \hat{\mathbf{z}}_H)^2 + 0.05 \|\mathbf{q} - \mathbf{q}_{k+1}\|^2 \right)$$

---

### 3.6 Dual-Synchronized vs Asymmetric Corridor Scheduling

Depending on corridor solvability:
1. **`DUAL_SYNCHRONIZED`**: Both robots advance simultaneously along their corridors at matched velocities:
   $$\mathbf{q}_A(t) = \mathbf{q}_{\text{pre}, A} + \tau(t) (\mathbf{q}_{H, A} - \mathbf{q}_{\text{pre}, A}), \quad \mathbf{q}_B(t) = \mathbf{q}_{\text{pre}, B} + \tau(t) (\mathbf{q}_{H, B} - \mathbf{q}_{\text{pre}, B})$$
2. **`ASYMMETRIC`**: If one arm operates near roll limits, the constrained arm stations safely at pre-handover/handover first, while the partner arm advances, avoiding joint limit clamping.

---

# 4. Multi-Robot Pick-and-Place & Synchronized Trajectory Planning

### 4.1 Stage 2.5 Object-Pose IK with Adaptive Tilt Synthesis

Given an object pose $\mathbf{p}_{\text{obj}} = [x, y, z]^T$ and object yaw $\psi \in [0, 2\pi)$, candidate gripper orientations are synthesized across 15 discrete tilt angles $\beta$:
$$\beta \in [0^\circ, 10^\circ, -10^\circ, 20^\circ, -20^\circ, 30^\circ, -30^\circ, 45^\circ, -45^\circ, 60^\circ, -60^\circ, 75^\circ, -75^\circ, 90^\circ, -90^\circ]$$

For finger axis $\hat{\mathbf{x}}$ and approach axis $\hat{\mathbf{z}}$:
$$R_{\text{desired}}(\psi, \beta) = R_z(\psi) \cdot R_x(\pi + \beta) = \begin{bmatrix}
\cos\psi & -\sin\psi & 0 \\
\sin\psi & \cos\psi & 0 \\
0 & 0 & 1
\end{bmatrix} \begin{bmatrix}
1 & 0 & 0 \\
0 & -\cos\beta & \sin\beta \\
0 & -\sin\beta & -\cos\beta
\end{bmatrix}$$
At zero tilt ($\beta = 0$), the approach axis points downward ($\hat{\mathbf{z}} = [0, 0, -1]^T$), and fingers align with the object yaw vector $[\cos\psi, \sin\psi, 0]^T$.

---

### 4.2 5-DOF Task-Priority Nullspace Inverse Kinematics

Because 5-DOF manipulators cannot independently track full 6D poses, a **task-priority nullspace formulation** is executed:
- **Primary Task (Hard Constraint)**: 3D TCP Position $\mathbf{e}_p = \mathbf{p}_{\text{target}} - \mathbf{p}_{\text{tcp}} \in \mathbb{R}^3$.
- **Secondary Task (Projected Constraint)**: Finger-axis yaw alignment $\mathbf{e}_{R, \perp}$ projected into the position nullspace.

#### Formulation:
$$J_v^\dagger = J_v^T (J_v J_v^T + \lambda_p^2 I_3)^{-1}$$
$$\Delta \mathbf{q}_p = J_v^\dagger \mathbf{e}_p$$
$$N_p = I_n - J_v^\dagger J_v \in \mathbb{R}^{n \times n}$$

Let $\hat{\mathbf{f}} = R_{\text{tcp}}[:, 0]$ be the current finger axis. Orientation error is split:
$$\mathbf{e}_{R, \parallel} = (\mathbf{e}_R \cdot \hat{\mathbf{f}}) \hat{\mathbf{f}}, \quad \mathbf{e}_{R, \perp} = \mathbf{e}_R - \mathbf{e}_{R, \parallel}$$
$$J_{N\omega} = J_\omega N_p, \quad J_{N\omega}^\dagger = J_{N\omega}^T (J_{N\omega} J_{N\omega}^T + \lambda_\omega^2 I_3)^{-1}$$
$$\Delta \mathbf{q}_\omega = J_{N\omega}^\dagger (\mathbf{e}_{R, \perp} + w_{\text{tilt}} \mathbf{e}_{R, \parallel})$$

If position error $\|\mathbf{e}_p\| > 0.8 \epsilon_p$, pure position is commanded: $\Delta \mathbf{q} = 0.8 \Delta \mathbf{q}_p$.
Once within tolerance, secondary orientation is admitted:
$$\Delta \mathbf{q} = 0.8 \Delta \mathbf{q}_p + 0.5 (N_p \Delta \mathbf{q}_\omega)$$

---

### 4.3 6-DOF Balanced Units Pose Inverse Kinematics

For 6-DOF arms (Robot 3), full 6D pose tracking is performed. To eliminate dimensional inconsistency between position (millimeters) and orientation (radians), a characteristic length $L_c = 100.0\text{ mm}$ balances units:
$$J_{\text{scaled}} = \begin{bmatrix} \frac{1}{L_c} J_v \\ J_\omega \end{bmatrix} \in \mathbb{R}^{6 \times n}, \quad \mathbf{e}_{\text{scaled}} = \begin{bmatrix} \frac{1}{L_c} \mathbf{e}_p \\ \mathbf{e}_R \end{bmatrix} \in \mathbb{R}^6$$

Using adaptive Levenberg-Marquardt damping:
$$(J_{\text{scaled}}^T J_{\text{scaled}} + \lambda^2 I_n) \Delta \mathbf{q} = J_{\text{scaled}}^T \mathbf{e}_{\text{scaled}}$$
$$\mathbf{q} \leftarrow \text{clip}(\mathbf{q} + 0.6 \Delta \mathbf{q}, \mathbf{q}_{\min}, \mathbf{q}_{\max})$$

---

### 4.4 Lexicographic Feasibility Filter & Normalized Cost Function

Candidate solutions are validated against strict feasibility gates:
1. Position error: $\|\mathbf{e}_p\| \le 1.0\text{ mm}$
2. Finger alignment: $\theta_{\text{align}} \le 5.0^\circ$
3. Orientation error: $\theta_{\text{ori}} \le 15.0^\circ$
4. Joint limit margin: $\min_j \min(q_j - q_{\min, j}, q_{\max, j} - q_j) \ge 0.1^\circ$
5. Manipulability: $\sqrt{\det(J J^T)} \ge 10^{-4}$

Surviving candidates are ranked by a normalized multi-objective scalar cost:
$$C = 0.40 \cdot \frac{|\beta|}{90^\circ} + 0.25 \cdot \frac{\|\mathbf{q} - \mathbf{q}_{\text{current}}\|}{\pi\sqrt{n}} + 0.15 \cdot \left(1 - \frac{m_q}{45^\circ}\right) + 0.10 \cdot \left(1 - \frac{\mu}{\mu_{\max}}\right) + 0.10 \cdot \frac{\theta_{\text{align}}}{5^\circ}$$

---

### 4.5 Minimum Jerk 7th-Order Septic Trajectory Formulation

Joint trajectories use a 7th-order polynomial $s_7(\tau)$ for normalized time $\tau = \frac{t}{T} \in [0, 1]$:
$$s(\tau) = 35 \tau^4 - 84 \tau^5 + 70 \tau^6 - 20 \tau^7$$
$$\dot{s}(\tau) = 140 \tau^3 - 420 \tau^4 + 420 \tau^5 - 140 \tau^6$$
$$\ddot{s}(\tau) = 420 \tau^2 - 1680 \tau^3 + 2100 \tau^4 - 840 \tau^5$$
$$\dddot{s}(\tau) = 840 \tau - 5040 \tau^2 + 8400 \tau^3 - 4200 \tau^4$$

#### Boundary Conditions:
$$s(0) = 0, \quad s(1) = 1$$
$$\dot{s}(0) = \dot{s}(1) = 0 \quad (\text{zero initial \& final velocity})$$
$$\ddot{s}(0) = \ddot{s}(1) = 0 \quad (\text{zero initial \& final acceleration})$$
$$\dddot{s}(0) = \dddot{s}(1) = 0 \quad (\text{zero initial \& final jerk})$$

Joint angles, velocities, accelerations, and jerks are evaluated as:
$$\mathbf{q}(t) = \mathbf{q}_0 + (\mathbf{q}_1 - \mathbf{q}_0) s(\tau)$$
$$\dot{\mathbf{q}}(t) = \frac{\mathbf{q}_1 - \mathbf{q}_0}{T} \dot{s}(\tau)$$
$$\ddot{\mathbf{q}}(t) = \frac{\mathbf{q}_1 - \mathbf{q}_0}{T^2} \ddot{s}(\tau)$$
$$\dddot{\mathbf{q}}(t) = \frac{\mathbf{q}_1 - \mathbf{q}_0}{T^3} \dddot{s}(\tau)$$
This formulation guarantees continuous $C^3$ smoothness and prevents mechanical shudder.

---

### 4.6 9-Phase Synchronized Multi-Robot State Machine

The complete handover task is partitioned into 9 collision-checked phases:

```
[Phase 0: Pre-Motion]  ── Open Grippers at Home
         │
[Phase 1: Approach]    ── Move to Pre-Pick (Z+50mm) ──> Cartesian Descent to Pick
         │
[Phase 2: Grasp]       ── Dwell (0.5s) ──> Close Gripper ──> Settle (3.0s)
         │
[Phase 3: Transfer]    ── Lift to Pre-Pick ──> Travel to Pre-Handover (Standoff)
         │
[Phase 4: Advance]     ── Synchronous Descent along Corridor to Handover Mating Point
         │
[Phase 5: Exchange]    ── Dwell ──> Close Receiver ──> Dwell ──> Open Picker
         │
[Phase 6: Retract]     ── Retract along Corridor back to Pre-Handover
         │
[Phase 7: Transfer]    ── Travel to Pre-Place (Z+50mm)
         │
[Phase 8: Release]     ── Cartesian Descent to Place ──> Dwell ──> Open Gripper ──> Lift
         │
[Phase 9: Home]        ── Synchronous Collision-Free Return to Home Pose
```

---

# 5. Whole-Body Link Clearance & Collision Avoidance

### 5.1 Exact 3D Line-Segment to Line-Segment Distance (Dan Sunday / Lumelsky Algorithm)

Each robot link is represented as a 3D line segment. Let Segment 1 run from $\mathbf{p}_1$ to $\mathbf{p}_2$, and Segment 2 run from $\mathbf{p}_3$ to $\mathbf{p}_4$:
$$\mathbf{u} = \mathbf{p}_2 - \mathbf{p}_1, \quad \mathbf{v} = \mathbf{p}_4 - \mathbf{p}_3, \quad \mathbf{w}_0 = \mathbf{p}_1 - \mathbf{p}_3$$

The line points are parameterized by $s, t \in [0, 1]$:
$$P(s) = \mathbf{p}_1 + s \mathbf{u}, \quad Q(t) = \mathbf{p}_3 + t \mathbf{v}$$

Vector difference: $\mathbf{w}(s, t) = P(s) - Q(t) = \mathbf{w}_0 + s \mathbf{u} - t \mathbf{v}$. The squared distance $\|\mathbf{w}(s, t)\|^2$ is minimized when $\mathbf{w}$ is orthogonal to both direction vectors:
$$\mathbf{u} \cdot \mathbf{w} = 0 \implies (\mathbf{u} \cdot \mathbf{u}) s - (\mathbf{u} \cdot \mathbf{v}) t = -\mathbf{u} \cdot \mathbf{w}_0$$
$$\mathbf{v} \cdot \mathbf{w} = 0 \implies (\mathbf{u} \cdot \mathbf{v}) s - (\mathbf{v} \cdot \mathbf{v}) t = -\mathbf{v} \cdot \mathbf{w}_0$$

Defining inner products:
$$a = \mathbf{u} \cdot \mathbf{u}, \quad b = \mathbf{u} \cdot \mathbf{v}, \quad c = \mathbf{v} \cdot \mathbf{v}, \quad d = \mathbf{u} \cdot \mathbf{w}_0, \quad e = \mathbf{v} \cdot \mathbf{w}_0$$
The determinant is:
$$D = ac - b^2 = \|\mathbf{u} \times \mathbf{v}\|^2$$

#### Evaluation & Boundary Clamping:
If $D < 10^{-8}$ (segments are parallel):
$$s_c = 0.0, \quad t_c = \text{clip}\left(\frac{e}{c}, 0.0, 1.0\right) \quad (\text{if } c > 10^{-8})$$
Else:
$$s_c = \text{clip}\left(\frac{be - cd}{D}, 0.0, 1.0\right), \quad t_c = \text{clip}\left(\frac{ae - bd}{D}, 0.0, 1.0\right)$$

Candidate closest points are:
$$P_1^* = \mathbf{p}_1 + s_c \mathbf{u}, \quad P_2^* = \mathbf{p}_3 + t_c \mathbf{v}$$
Boundary projection tests check point-to-segment distances $d(P_1^*, \text{Seg}_2)$ and $d(P_2^*, \text{Seg}_1)$, and the true minimum Euclidean distance is:
$$d_{\min} = \min\left(\|P_1^* - P_2^*\|, d(P_1^*, \text{Seg}_2), d(P_2^*, \text{Seg}_1)\right)$$

---

### 5.2 All-Pairs Whole-Body Link Clearance Evaluation

For Robot 1 with $N_1$ frames and Robot 2 with $N_2$ frames, the set of line segments in world coordinates is:
$$\mathcal{S}_1 = \{[\mathbf{p}_{w, i}^{(1)}, \mathbf{p}_{w, i+1}^{(1)}] \mid i = 0, \dots, N_1 - 2\}$$
$$\mathcal{S}_2 = \{[\mathbf{p}_{w, j}^{(2)}, \mathbf{p}_{w, j+1}^{(2)}] \mid j = 0, \dots, N_2 - 2\}$$

The instantaneous clearance $c(\mathbf{q}_1, \mathbf{q}_2)$ is evaluated across all pairs:
$$c(\mathbf{q}_1, \mathbf{q}_2) = \min_{S_a \in \mathcal{S}_1, S_b \in \mathcal{S}_2} \text{dist\_segment\_to\_segment}(S_a, S_b)$$
Safety requires:
$$c(\mathbf{q}_1(t), \mathbf{q}_2(t)) \ge d_{\text{safe}} = 50.0\text{ mm} \quad \forall t$$

---

### 5.3 Tooltip / TCP Exclusion Rule for Contact Handover

During Phase 4 (Advance to Handover) and Phase 5 (Grasp Exchange), the two grippers must intentionally meet ($\|\mathbf{p}_{\text{tcp}, 1} - \mathbf{p}_{\text{tcp}, 2}\| \to 0\text{ mm}$). 

If gripper tip segments were evaluated in body clearance checks, normal handover contact would be falsely rejected as a collision.

#### Mathematical Exclusion:
$$\mathcal{S}_{\text{body}}(\mathbf{q}) = \begin{cases}
\{[\mathbf{p}_0, \mathbf{p}_1], [\mathbf{p}_1, \mathbf{p}_2], \dots, [\mathbf{p}_{N-2}, \mathbf{p}_{N-1}]\} & \text{if } \text{exclude\_tcp} = \text{False} \\
\{[\mathbf{p}_0, \mathbf{p}_1], [\mathbf{p}_1, \mathbf{p}_2], \dots, [\mathbf{p}_{N-3}, \mathbf{p}_{N-2}]\} & \text{if } \text{exclude\_tcp} = \text{True}
\end{cases}$$
When $\text{exclude\_tcp} = \text{True}$, the final tool flange segment is excluded, while every body link (base, shoulder, upper arm, elbow, forearm, wrist) continues to be strictly monitored.

---

### 5.4 Collision-Aware Motion Scheduling: Concurrent vs Serialized Delay

When two robots must transition between poses:
1. **Concurrent Evaluation**: Both robots move simultaneously. If $c(t) \ge d_{\text{safe}}$ for all steps $t$, concurrent motion is approved.
2. **Dynamic Serialization**: If concurrent motion produces $c(t) < d_{\text{safe}}$:
   - **Option A (Pick Arm Moves First)**: Robot 1 moves to its goal while Robot 2 holds at its start pose. Once Robot 1 completes, Robot 2 executes.
   - **Option B (Place Arm Moves First)**: Robot 2 moves to its goal while Robot 1 holds at its start pose.
   The option yielding the largest clearance $\max(\min c_A, \min c_B)$ is selected automatically.

---

# 6. Web Digital Twin Mathematical Architecture

### 6.1 Coordinate Frame Transformations: Base $\to$ World $\to$ Three.js

The Web Digital Twin maintains a unified right-handed Cartesian coordinate system:
- $\hat{X}$: East / Right
- $\hat{Y}$: North / Forward
- $\hat{Z}$: Vertical Height (Up)

Frame transforms from robot base $\mathcal{F}_{\text{base}}$ to common world coordinates $\mathcal{F}_{\text{world}}$:
$$\mathbf{p}_{\text{world}} = R_{w, \text{base}} \mathbf{p}_{\text{base}} + \mathbf{t}_{w, \text{base}}$$
$$R_{\text{world}} = R_{w, \text{base}} R_{\text{base}}$$

Where $R_{w, \text{base}} = \text{Rot}_z(\psi) \text{Rot}_y(\theta) \text{Rot}_x(\phi)$ is defined by the robot base RPY configuration angles.

---

### 6.2 Procedural Mesh Transformations: Link Cylinders & Flange Quaternions

In the 3D WebGL scene (`robot_models.js`), links between consecutive frames $\mathbf{p}_i$ and $\mathbf{p}_{i+1}$ are procedurally generated using standard cylinders whose canonical geometry is aligned along the local $\hat{Y}$ axis:
- **Midpoint Position**: $\mathbf{p}_{\text{mid}} = \frac{1}{2}(\mathbf{p}_i + \mathbf{p}_{i+1})$
- **Length Scaling**: $h = \|\mathbf{p}_{i+1} - \mathbf{p}_i\|$ along $\hat{Y}$
- **Orientation Quaternion**: The rotation mapping canonical axis $\hat{\mathbf{y}} = [0, 1, 0]^T$ to link direction $\hat{\mathbf{d}} = \frac{\mathbf{p}_{i+1} - \mathbf{p}_i}{h}$ is:
  $$\mathbf{q}_{\text{mesh}} = \left[ \sqrt{2(1 + \hat{\mathbf{y}} \cdot \hat{\mathbf{d}})}, \; \frac{\hat{\mathbf{y}} \times \hat{\mathbf{d}}}{\sqrt{2(1 + \hat{\mathbf{y}} \cdot \hat{\mathbf{d}})}} \right]^T$$

For the end-effector gripper, the $3 \times 3$ orientation matrix $R_{\text{world}}$ maps directly to the Three.js $4 \times 4$ transformation matrix:
$$M_{\text{three}} = \begin{bmatrix}
R_{00} & R_{01} & R_{02} & p_x \\
R_{10} & R_{11} & R_{12} & p_y \\
R_{20} & R_{21} & R_{22} & p_z \\
0 & 0 & 0 & 1
\end{bmatrix}$$

---

### 6.3 Real-Time Dynamic Laser Clearance Gauge Formulation

The Web Twin streams telemetry at 60 Hz to evaluate the real-time distance between active robot links via `closest_segment_points`.

The exact closest points $P_1^*, P_2^* \in \mathbb{R}^3$ are computed and rendered as a 3D dynamic laser line:
$$d^*(t) = \|P_1^*(t) - P_2^*(t)\|$$

Dynamic HUD shader coloration reflects physical proximity:
$$\text{Color}(d^*) = \begin{cases}
\text{#00F0FF (Luminous Cyan)} & \text{if } d^* \ge 50.0\text{ mm (Nominal / Safe)} \\
\text{#FFB800 (Neon Amber)} & \text{if } 30.0\text{ mm} \le d^* < 50.0\text{ mm (Caution)} \\
\text{#FF0055 (Luminous Red)} & \text{if } d^* < 30.0\text{ mm (Critical Hazard)}
\end{cases}$$

This mathematical visualization allows operators to observe minimum clearances and potential bottlenecks in real-time.
