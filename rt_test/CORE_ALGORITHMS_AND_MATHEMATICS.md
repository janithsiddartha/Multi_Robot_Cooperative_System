# Core Algorithms and Mathematical Formulations

This document provides a comprehensive technical breakdown of every algorithm, kinematic formulation, mathematical derivation, input/output contract, and coordination mechanism across **Stage 1**, **Stage 1.5**, **Stage 2**, and **Stage 2.5**.

---

## Table of Contents

1. [Mathematical Foundations & Kinematic Modeling](#1-mathematical-foundations--kinematic-modeling)
   - [Modified Denavit-Hartenberg (DH) Convention](#11-modified-denavit-hartenberg-dh-convention)
   - [Forward Kinematics (FK)](#12-forward-kinematics-fk)
   - [Geometric Jacobian Matrix](#13-geometric-jacobian-matrix)
   - [World-to-Base Coordinate Transformations](#14-world-to-base-coordinate-transformations)
2. [Stage 1: Position-Only Numerical Inverse Kinematics](#2-stage-1-position-only-numerical-inverse-kinematics)
   - [Problem Definition, Inputs & Outputs](#21-problem-definition-inputs--outputs)
   - [Damped Least Squares (DLS) / Levenberg-Marquardt](#22-damped-least-squares-dls--levenberg-marquardt)
   - [Multi-Start Deterministic Seeding](#23-multi-start-deterministic-seeding)
   - [Workspace Intersection & Interactive Chooser](#24-workspace-intersection--interactive-chooser)
3. [Stage 1.5: Whole-Body Link Clearance & Collision Avoidance](#3-stage-15-whole-body-link-clearance--collision-avoidance)
   - [Problem Definition, Inputs & Outputs](#31-problem-definition-inputs--outputs)
   - [3D Line-Segment to Line-Segment Distance Algorithm](#32-3d-line-segment-to-line-segment-distance-algorithm)
   - [Continuous Trajectory Discretization & All-Pairs Link Checking](#33-continuous-trajectory-discretization--all-pairs-link-checking)
   - [Staggered Temporal Delays](#34-staggered-temporal-delays)
4. [Stage 2: Full 6D Pose Inverse Kinematics](#4-stage-2-full-6d-pose-inverse-kinematics)
   - [Problem Definition, Inputs & Outputs](#41-problem-definition-inputs--outputs)
   - [3D Orientation Error Formulation in $\text{SO}(3)$](#42-3d-orientation-error-formulation-in-textso3)
   - [Dimensional Scaling & Balanced Units Jacobian](#43-dimensional-scaling--balanced-units-jacobian)
   - [Singularity Handling & Adaptive Damping](#44-singularity-handling--adaptive-damping)
   - [Opposing Handover Orientation Synthesis](#45-opposing-handover-orientation-synthesis)
5. [Stage 2.5: Object-Pose IK, Adaptive Tilt, & Synchronized Handover](#5-stage-25-object-pose-ik-adaptive-tilt--synchronized-handover)
   - [Problem Definition, Inputs & Outputs](#51-problem-definition-inputs--outputs)
   - [Configurable Gripper Finger-Axis & Adaptive Tilt Synthesis](#52-configurable-gripper-finger-axis--adaptive-tilt-synthesis)
   - [5-DOF Task-Priority Nullspace Inverse Kinematics](#53-5-dof-task-priority-nullspace-inverse-kinematics)
   - [6-DOF Balanced Pose IK](#54-6-dof-balanced-pose-ik)
   - [Lexicographic Feasibility Filter](#55-lexicographic-feasibility-filter)
   - [Normalized Multi-Objective Cost Ranking](#56-normalized-multi-objective-cost-ranking)
   - [Independent Pick/Place Solver Fallback Logic](#57-independent-pickplace-solver-fallback-logic)
   - [Synchronized 9-Step Trajectory State Machine](#58-synchronized-9-step-trajectory-state-machine)
   - [Zero-Recalculation Simulation Plan Caching](#59-zero-recalculation-simulation-plan-caching)

---

## 1. Mathematical Foundations & Kinematic Modeling

### 1.1 Modified Denavit-Hartenberg (DH) Convention

The kinematic structure of each serial manipulator is parameterized by four standard Modified Denavit-Hartenberg parameters for each link $i \in \{1, \dots, n\}$:

- $\alpha_{i-1}$: Link twist (angle about $X_{i-1}$ from $Z_{i-1}$ to $Z_i$)
- $a_{i-1}$: Link length (distance along $X_{i-1}$ from $Z_{i-1}$ to $Z_i$)
- $d_i$: Link offset (distance along $Z_i$ from $X_{i-1}$ to $X_i$)
- $\theta_i$: Joint angle (angle about $Z_i$ from $X_{i-1}$ to $X_i$)

The homogeneous transformation matrix $^{i-1}T_i \in \text{SE}(3)$ from frame $i-1$ to frame $i$ is:

$$^{i-1}T_i = \begin{bmatrix}
\cos\theta_i & -\sin\theta_i & 0 & a_{i-1} \\
\sin\theta_i \cos\alpha_{i-1} & \cos\theta_i \cos\alpha_{i-1} & -\sin\alpha_{i-1} & -d_i \sin\alpha_{i-1} \\
\sin\theta_i \sin\alpha_{i-1} & \cos\theta_i \sin\alpha_{i-1} & \cos\alpha_{i-1} & d_i \cos\alpha_{i-1} \\
0 & 0 & 0 & 1
\end{bmatrix}$$

In our implementation, active joint variables $q_j$ map to frame angles $\theta_i$ via configured offsets and signs:
$$\theta_i = \theta_{\text{offset}, i} + \mu_i \cdot q_{\text{index}(i)}$$
where $\mu_i \in \{+1, -1\}$ is the multiplier and $\theta_{\text{offset}, i}$ is the zero-offset angle.

---

### 1.2 Forward Kinematics (FK)

Given joint vector $\mathbf{q} = [q_1, q_2, \dots, q_n]^T \in \mathbb{R}^n$, the total transformation from the robot base frame $\mathcal{F}_0$ to the end-flange frame $\mathcal{F}_n$ is the cumulative matrix product:

$$^0T_n(\mathbf{q}) = \prod_{i=1}^n \, ^{i-1}T_i(q_i) = \, ^0T_1(q_1) \cdot \, ^1T_2(q_2) \cdots \, ^{n-1}T_n(q_n)$$

The Tool Center Point (TCP) is defined by a rigid translation offset $\mathbf{t}_{\text{offset}} = [dx, dy, dz]^T$:

$$^0T_{\text{tcp}}(\mathbf{q}) = \, ^0T_n(\mathbf{q}) \cdot \begin{bmatrix}
I_{3 \times 3} & \mathbf{t}_{\text{offset}} \\
\mathbf{0}_{1 \times 3} & 1
\end{bmatrix} = \begin{bmatrix}
R_{\text{tcp}}(\mathbf{q}) & \mathbf{p}_{\text{tcp}}(\mathbf{q}) \\
\mathbf{0}_{1 \times 3} & 1
\end{bmatrix}$$

Where:
- $\mathbf{p}_{\text{tcp}}(\mathbf{q}) \in \mathbb{R}^3$ is the Cartesian position of the gripper tip in the robot's base frame.
- $R_{\text{tcp}}(\mathbf{q}) = [\mathbf{n}, \mathbf{s}, \mathbf{a}] \in \text{SO}(3)$ is the $3 \times 3$ orthonormal rotation matrix defining the gripper orientation:
  - Column 0 ($\mathbf{n} = R_{:,0}$): Local $X$ axis.
  - Column 1 ($\mathbf{s} = R_{:,1}$): Local $Y$ axis.
  - Column 2 ($\mathbf{a} = R_{:,2}$): Local $Z$ axis (approach vector).

---

### 1.3 Geometric Jacobian Matrix

The Geometric Jacobian $J(\mathbf{q}) \in \mathbb{R}^{6 \times n}$ maps joint velocities $\mathbf{\dot{q}}$ to the end-effector linear velocity $\mathbf{v}$ and angular velocity $\boldsymbol{\omega}$:

$$\begin{bmatrix} \mathbf{v} \\ \boldsymbol{\omega} \end{bmatrix} = J(\mathbf{q}) \, \mathbf{\dot{q}} = \begin{bmatrix} J_v(\mathbf{q}) \\ J_\omega(\mathbf{q}) \end{bmatrix} \mathbf{\dot{q}}$$

For revolute joints, each column $i \in \{1, \dots, n\}$ of the Jacobian is computed directly from forward kinematics:

$$J_{v, i}(\mathbf{q}) = \mathbf{z}_{i-1} \times (\mathbf{p}_{\text{tcp}} - \mathbf{p}_{i-1})$$
$$J_{\omega, i}(\mathbf{q}) = \mathbf{z}_{i-1}$$

Where:
- $\mathbf{z}_{i-1} \in \mathbb{R}^3$ is the unit vector along the rotation axis of joint $i$ expressed in the base frame (the 3rd column of $^0T_{i-1}$).
- $\mathbf{p}_{i-1} \in \mathbb{R}^3$ is the origin position of frame $i-1$ expressed in the base frame.
- $\mathbf{p}_{\text{tcp}} \in \mathbb{R}^3$ is the TCP position.

---

### 1.4 World-to-Base Coordinate Transformations

Each robot $i$ is mounted at a distinct position $\mathbf{t}_{W, B_i} \in \mathbb{R}^3$ and orientation $R_{W, B_i} \in \text{SO}(3)$ relative to the global World frame $\mathcal{F}_W$:

$$^W T_{B_i} = \begin{bmatrix} R_{W, B_i} & \mathbf{t}_{W, B_i} \\ \mathbf{0}_{1 \times 3} & 1 \end{bmatrix}$$

For any target position $\mathbf{p}_W \in \mathbb{R}^3$ and direction vector $\mathbf{v}_W \in \mathbb{R}^3$ specified in World coordinates:

$$\mathbf{p}_{B_i} = R_{W, B_i}^T (\mathbf{p}_W - \mathbf{t}_{W, B_i})$$
$$\mathbf{v}_{B_i} = R_{W, B_i}^T \mathbf{v}_W$$

For orientation matrices:
$$R_{B_i} = R_{W, B_i}^T R_W$$

---

## 2. Stage 1: Position-Only Numerical Inverse Kinematics

### 2.1 Problem Definition, Inputs & Outputs

- **Purpose**: Compute joint angles $\mathbf{q} \in \mathbb{R}^n$ such that the robot TCP reaches a 3D point $\mathbf{p}^* \in \mathbb{R}^3$ without orientation constraints.
- **Inputs**:
  - Target position $\mathbf{p}^* \in \mathbb{R}^3$ (mm).
  - Kinematic configuration (DH parameters, joint limits $[\mathbf{q}_{\min}, \mathbf{q}_{\max}]$).
  - Initial joint seed $\mathbf{q}_0 \in \mathbb{R}^n$.
  - Position tolerance $\epsilon_p$ (default: $1.0\text{ mm}$).
- **Outputs**:
  - Solution joint vector $\mathbf{q}^* \in \mathbb{R}^n$.
  - Cartesian position error $e_p = \|\mathbf{p}^* - \mathbf{p}(\mathbf{q}^*)\|$ (mm).
  - Success flag (`True` / `False`).

---

### 2.2 Damped Least Squares (DLS) / Levenberg-Marquardt

At iteration $k$, the position error vector is:
$$\mathbf{e}_p^{(k)} = \mathbf{p}^* - \mathbf{p}_{\text{tcp}}(\mathbf{q}^{(k)}) \in \mathbb{R}^3$$

The linear velocity sub-Jacobian is $J_v(\mathbf{q}^{(k)}) \in \mathbb{R}^{3 \times n}$. To avoid numerical instability near kinematic singularities, the joint update is computed via Damped Least Squares:

$$\Delta \mathbf{q}^{(k)} = J_v^T \left( J_v J_v^T + \lambda^2 I_{3 \times 3} \right)^{-1} \mathbf{e}_p^{(k)}$$

#### Adaptive Damping Update (Levenberg-Marquardt Strategy):
The damping parameter $\lambda^2$ adjusts dynamically based on the error reduction ratio:
$$\rho = \frac{\|\mathbf{e}_p^{(k)}\|^2 - \|\mathbf{e}_p^{(k+1)}\|^2}{\|\mathbf{e}_p^{(k)}\|^2 - \|\mathbf{e}_p^{(k)} - J_v \Delta \mathbf{q}^{(k)}\|^2}$$

- If $\rho > 0.75$: Reduce damping ($\lambda \leftarrow \max(\lambda / 2, \lambda_{\min})$).
- If $\rho < 0.25$: Increase damping ($\lambda \leftarrow \min(2\lambda, \lambda_{\max})$).
- If error increases: Reject step and increase damping.

#### Step Clamping & Joint Limits:
To guarantee convergence stability and stay within mechanical joint ranges:
$$\Delta \mathbf{q}_{\text{clamped}} = \Delta \mathbf{q} \cdot \min\left(1.0, \frac{\Delta q_{\text{max\_step}}}{\|\Delta \mathbf{q}\|}\right)$$
$$\mathbf{q}^{(k+1)} = \text{clip}\left(\mathbf{q}^{(k)} + \alpha \Delta \mathbf{q}_{\text{clamped}},\, \mathbf{q}_{\min},\, \mathbf{q}_{\max}\right)$$

Iterations stop when $\|\mathbf{e}_p\| \le \epsilon_p$ or when $\Delta \mathbf{q} < 10^{-6}\text{ rad}$.

---

### 2.3 Multi-Start Deterministic Seeding

Because iterative gradient methods can get trapped in local minima near joint limits, the solver evaluates deterministic seed points in order:
1. Current robot joint angles $\mathbf{q}_{\text{current}}$.
2. Robot home position $\mathbf{q}_{\text{home}}$.
3. Joint mid-range: $\mathbf{q}_{\text{mid}} = \frac{1}{2}(\mathbf{q}_{\min} + \mathbf{q}_{\max})$.
4. Quarter bounds: $\mathbf{q}_{\min} + 0.25(\mathbf{q}_{\max} - \mathbf{q}_{\min})$ and $\mathbf{q}_{\max} - 0.25(\mathbf{q}_{\max} - \mathbf{q}_{\min})$.
5. Deterministic pseudo-random seeds generated with a fixed random seed.

---

### 2.4 Workspace Intersection & Interactive Chooser

Let $\mathcal{W}_i \subset \mathbb{R}^3$ be the reachable workspace volume of robot $i$. For any Cartesian target $\mathbf{p}^*$:

$$\text{Reachable}(\mathbf{p}^*) = \left\{ i \in \{\text{robot1}, \text{robot2}, \text{robot3}\} \;\middle|\; \exists \mathbf{q} \in [\mathbf{q}_{\min, i}, \mathbf{q}_{\max, i}],\; \|\mathbf{p}^* - \mathbf{p}_i(\mathbf{q})\| \le \epsilon_p \right\}$$

- If $|\text{Reachable}(\mathbf{p}^*)| = 0$: Raise `RuntimeError("Target is out of reachable workspace")`.
- If $|\text{Reachable}(\mathbf{p}^*)| = 1$: Auto-assign the sole capable robot.
- If $|\text{Reachable}(\mathbf{p}^*)| > 1$: The target lies in an intersection zone. The interactive modal dialog (`select_robot`) triggers on the GUI main thread, prompting the operator to choose between the reachable robot candidates.

---

## 3. Stage 1.5: Whole-Body Link Clearance & Collision Avoidance

### 3.1 Problem Definition, Inputs & Outputs

- **Purpose**: Prevent physical collisions between two cooperating robots during simultaneous trajectory execution by computing minimum Euclidean distances between all robot links.
- **Inputs**:
  - Trajectory waypoints $\mathbf{q}_A(t), \mathbf{q}_B(t)$ for robots $A$ and $B$ across normalized time $t \in [0, 1]$.
  - Safety clearance threshold $d_{\text{safe}}$ (default: $40.0\text{ mm}$).
- **Outputs**:
  - Minimum whole-body clearance distance $d_{\min} = \min_{t} d(A(t), B(t))$.
  - Boolean flag `is_collision_free` ($d_{\min} \ge d_{\text{safe}}$).
  - Trajectory timestamp $t_{\text{critical}}$ and link indices of closest approach.

---

### 3.2 3D Line-Segment to Line-Segment Distance Algorithm

Every robotic link $k$ between joint frame $k-1$ and $k$ is geometrically modeled as a 3D cylindrical capsule centered around line segment $S_k$:
$$S_k(u) = \mathbf{p}_{k-1} + u (\mathbf{p}_k - \mathbf{p}_{k-1}) = \mathbf{p}_{k-1} + u \mathbf{u}_k, \quad u \in [0, 1]$$

Given two line segments $S_A(s) = \mathbf{P}_0 + s \mathbf{u}$ ($s \in [0, 1]$) and $S_B(t) = \mathbf{Q}_0 + t \mathbf{v}$ ($t \in [0, 1]$):

```
         P0 •─────────────────────────────• P1   (Link A)
                 \     | d_min
                  \    |
                   \   |
             Q0 •──────────────────• Q1          (Link B)
```

The displacement vector between arbitrary points on the two segments is:
$$\mathbf{w}(s, t) = S_A(s) - S_B(t) = (\mathbf{P}_0 - \mathbf{Q}_0) + s \mathbf{u} - t \mathbf{v} = \mathbf{w}_0 + s \mathbf{u} - t \mathbf{v}$$

The squared distance $D^2(s, t) = \|\mathbf{w}(s, t)\|^2$ is a quadratic function:
$$D^2(s, t) = a s^2 - 2b st + c t^2 + 2d s - 2e t + f$$

Where:
$$a = \mathbf{u} \cdot \mathbf{u}, \quad b = \mathbf{u} \cdot \mathbf{v}, \quad c = \mathbf{v} \cdot \mathbf{v}, \quad d = \mathbf{u} \cdot \mathbf{w}_0, \quad e = \mathbf{v} \cdot \mathbf{w}_0, \quad f = \mathbf{w}_0 \cdot \mathbf{w}_0$$

The unconstrained minimizer of the quadratic form satisfies $\nabla D^2(s, t) = \mathbf{0}$:
$$\begin{bmatrix} a & -b \\ -b & c \end{bmatrix} \begin{bmatrix} s_c \\ t_c \end{bmatrix} = \begin{bmatrix} -d \\ e \end{bmatrix}$$

The determinant is $\Delta = ac - b^2 = \|\mathbf{u} \times \mathbf{v}\|^2$:
1. **Non-Parallel Segments ($\Delta > 10^{-7}$)**:
   $$s_c = \frac{be - cd}{\Delta}, \quad t_c = \frac{ae - bd}{\Delta}$$
2. **Parallel Segments ($\Delta \le 10^{-7}$)**:
   Fix $s_c = 0.0$ and compute $t_c = \frac{e}{c}$.

#### Clamping to Unit Square $[0, 1] \times [0, 1]$:
If $s_c$ or $t_c$ lie outside $[0, 1]$, the minimum lies on the boundary of the unit square:
- Clamp $s_c \leftarrow \text{clip}(s_c, 0, 1)$, then recompute $t_c = \text{clip}\left(\frac{b s_c + e}{c}, 0, 1\right)$.
- Clamp $t_c \leftarrow \text{clip}(t_c, 0, 1)$, then recompute $s_c = \text{clip}\left(\frac{b t_c - d}{a}, 0, 1\right)$.

The minimum distance between the two segments is:
$$\text{dist}(S_A, S_B) = \|\mathbf{w}(s_c, t_c)\| = \|(\mathbf{P}_0 + s_c \mathbf{u}) - (\mathbf{Q}_0 + t_c \mathbf{v})\|$$

---

### 3.3 Continuous Trajectory Discretization & All-Pairs Link Checking

For trajectories $\mathbf{q}_A(t)$ and $\mathbf{q}_B(t)$ discretized into $M$ time steps ($t_1, t_2, \dots, t_M$):
At each step $m \in \{1, \dots, M\}$:
1. Compute FK frame origins for Robot $A$: $\mathcal{P}_A(t_m) = \{\mathbf{p}_{A, 0}, \mathbf{p}_{A, 1}, \dots, \mathbf{p}_{A, n_A}\}$.
2. Compute FK frame origins for Robot $B$: $\mathcal{P}_B(t_m) = \{\mathbf{p}_{B, 0}, \mathbf{p}_{B, 1}, \dots, \mathbf{p}_{B, n_B}\}$.
3. Compute all-pairs segment distances:
   $$d_{\min}(t_m) = \min_{\substack{i \in \{1, \dots, n_A\} \\ j \in \{1, \dots, n_B\}}} \text{dist}\left(S_{A, i}(t_m),\, S_{B, j}(t_m)\right)$$

The overall trajectory clearance is:
$$d_{\text{overall}} = \min_{m \in \{1, \dots, M\}} d_{\min}(t_m)$$
If $d_{\text{overall}} < d_{\text{safe}}$, the trajectory is flagged as unsafe and rejected.

---

### 3.4 Staggered Temporal Delays

When both robots must traverse shared airspace (such as Robot A moving to Pre-Handover while Robot B moves to Pick), simultaneous execution can cause link convergence. A stagger ratio $\delta_{\text{stagger}} \in [0, 0.5]$ introduces a temporal phase delay:

$$\mathbf{q}_A(t) = \text{Interp}\left(\mathbf{q}_{A, \text{start}}, \mathbf{q}_{A, \text{end}},\, \min\left(1.0,\, \frac{t}{1 - \delta_{\text{stagger}}}\right)\right)$$
$$\mathbf{q}_B(t) = \text{Interp}\left(\mathbf{q}_{B, \text{start}}, \mathbf{q}_{B, \text{end}},\, \max\left(0.0,\, \frac{t - \delta_{\text{stagger}}}{1 - \delta_{\text{stagger}}}\right)\right)$$

Robot A completes its motion early, holding position while Robot B finishes, maximizing clearance in intersection zones.

---

## 4. Stage 2: Full 6D Pose Inverse Kinematics

### 4.1 Problem Definition, Inputs & Outputs

- **Purpose**: Find joint configurations satisfying both Cartesian position $\mathbf{p}^* \in \mathbb{R}^3$ and 3D orientation $R^* \in \text{SO}(3)$ (6 degrees of freedom).
- **Inputs**:
  - Target position $\mathbf{p}^*$ (mm).
  - Target rotation matrix $R^* = [\mathbf{n}^*, \mathbf{s}^*, \mathbf{a}^*] \in \text{SO}(3)$.
  - Position tolerance $\epsilon_p$ (default: $1.0\text{ mm}$), orientation tolerance $\epsilon_o$ (default: $15.0^\circ$).
- **Outputs**:
  - Joint angles $\mathbf{q}^*$.
  - Position error $e_p$ (mm) and orientation angle error $e_o$ (degrees).

---

### 4.2 3D Orientation Error Formulation in $\text{SO}(3)$

Given achieved rotation $R = [\mathbf{n}, \mathbf{s}, \mathbf{a}]$ and desired rotation $R^* = [\mathbf{n}^*, \mathbf{s}^*, \mathbf{a}^*]$, the relative rotation error matrix is:
$$R_e = R^T R^* \in \text{SO}(3)$$

The rotation error angle $\theta_e \in [0, \pi]$ is:
$$\theta_e = \arccos\left(\text{clip}\left(\frac{\text{tr}(R_e) - 1}{2},\, -1.0,\, 1.0\right)\right)$$

The 3D angular error vector $\mathbf{e}_r \in \mathbb{R}^3$ (expressed in the robot base frame) is derived from the cross-product of column vectors:
$$\mathbf{e}_r = \frac{1}{2} \left( \mathbf{n} \times \mathbf{n}^* + \mathbf{s} \times \mathbf{s}^* + \mathbf{a} \times \mathbf{a}^* \right)$$

For small angles, $\mathbf{e}_r \approx \theta_e \hat{\mathbf{k}}$, representing the instantaneous rotational velocity vector required to rotate $R$ onto $R^*$.

---

### 4.3 Dimensional Scaling & Balanced Units Jacobian

Linear position is measured in millimeters ($\sim 10^2$), whereas angular orientation is measured in radians ($\sim 10^0$). Standard Levenberg-Marquardt on the raw Jacobian $J = [J_v; J_\omega]$ heavily biases joint updates toward position, neglecting orientation convergence.

We apply a dimensional weighting factor $w_p = \frac{1}{L_{\text{char}}}$ (where $L_{\text{char}} = 100.0\text{ mm}$ is the characteristic arm length):

$$J_{\text{scaled}} = \begin{bmatrix} w_p J_v \\ J_\omega \end{bmatrix} \in \mathbb{R}^{6 \times n}, \quad \mathbf{e}_{\text{scaled}} = \begin{bmatrix} w_p \mathbf{e}_p \\ \mathbf{e}_r \end{bmatrix} \in \mathbb{R}^6$$

The normal equations become:
$$\left( J_{\text{scaled}}^T J_{\text{scaled}} + \lambda^2 I_{n \times n} \right) \Delta \mathbf{q} = J_{\text{scaled}}^T \mathbf{e}_{\text{scaled}}$$

Both position and orientation errors contribute equally to the gradient descent.

---

### 4.4 Singularity Handling & Adaptive Damping

Near kinematic singularities (e.g. wrist alignment or arm full extension), $J_{\text{scaled}}$ loses rank. Let the singular value decomposition of $J$ be:
$$J = U \Sigma V^T, \quad \Sigma = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_{\min})$$

The smallest singular value $\sigma_{\min} = \sqrt{\lambda_{\min}(J J^T)}$ measures distance to singularity. Damping is computed adaptively:

$$\lambda(\sigma_{\min}) = \begin{cases}
\lambda_0 & \text{if } \sigma_{\min} \ge \epsilon_\sigma \\
\sqrt{\lambda_0^2 + \lambda_{\max}^2 \left(1 - \left(\frac{\sigma_{\min}}{\epsilon_\sigma}\right)^2\right)} & \text{if } \sigma_{\min} < \epsilon_\sigma
\end{cases}$$

This automatically provides zero damping in open workspace regions (preserving gradient descent speed) and injects smooth damping near singular configurations (preventing joint velocity explosions).

---

### 4.5 Opposing Handover Orientation Synthesis

During robot-to-robot handover, the two grippers must face each other directly:
- **Approach Alignment**: $\mathbf{a}_B = -\mathbf{a}_A$ (opposing approach vectors along the line connecting the two robots).
- **Finger Alignment**: $\mathbf{n}_B = \mathbf{n}_A$ (parallel finger planes) or rotated by $90^\circ$ depending on gripper geometry.
- **Normal Axis**: $\mathbf{s}_B = \mathbf{a}_B \times \mathbf{n}_B$.

---

## 5. Stage 2.5: Object-Pose IK, Adaptive Tilt, & Synchronized Handover

### 5.1 Problem Definition, Inputs & Outputs

- **Purpose**: Solve pick-and-place where the user supplies an object's position $(x, y, z)$ and optional table yaw angle $\psi$. Maximize reachable workspace by searching adaptive approach tilt angles $\tau$ while strictly maintaining finger-pad alignment.
- **Inputs**:
  - Object Cartesian position $(x, y, z)$ in mm.
  - Object yaw angle $\psi \in [0, 360)^\circ$ (normalized).
  - Robot configuration ($n$ DOFs, joint limits, gripper finger axis).
  - Tolerances: $\epsilon_p = 1.0\text{ mm}$, $\epsilon_{\text{align}} = 5.0^\circ$, $\epsilon_{\text{margin}} = 0.1^\circ$, $\sigma_{\text{threshold}} = 10^{-4}$.
- **Outputs**:
  - `ObjectPickIKResult`: Success flag, joint vector $\mathbf{q}^*$, achieved approach tilt $\tau^*$, finger alignment error $e_{\text{align}}$, manipulability $\sigma_{\min}$, multi-objective cost $C$.

---

### 5.2 Configurable Gripper Finger-Axis & Adaptive Tilt Synthesis

The gripper finger axis can be configured as local $X$ or local $Y$ depending on mechanical mounting:

```
        finger_axis = 'x'                         finger_axis = 'y'
        
        ▲ Y_tcp                                   ▲ Y_tcp (Fingers)
        │                                         │
 ┌──────┴──────┐                           ┌──────┴──────┐
 │   Gripper   │                           │   Gripper   │
 │ ◄─────────► │ Fingers                   │             │
 └──────┬──────┘                           └──────┬──────┘
        │                                         │
        ▼ X_tcp (Fingers)                         ▼ X_tcp
```

Let $\psi$ be the object yaw angle on the table. The target orientation matrix $R_{\text{target}}(\psi, \tau)$ is synthesized as follows:

#### Case A: `finger_axis: "x"` (Default)
- At zero tilt ($\tau = 0$): Gripper fingers align with $[\cos\psi, \sin\psi, 0]^T$; approach vector points downward $[0, 0, -1]^T$.
- Rotation Formula:
  $$R_{\text{target}}(\psi, \tau) = R_z(\psi) \cdot R_x(\pi + \tau)$$
  $$R_{\text{target}} = \begin{bmatrix}
  \cos\psi & \sin\psi\cos\tau & \sin\psi\sin\tau \\
  \sin\psi & -\cos\psi\cos\tau & -\cos\psi\sin\tau \\
  0 & \sin\tau & -\cos\tau
  \end{bmatrix}$$
  Notice that column 0 ($X_{\text{tcp}}$) is strictly $[\cos\psi, \sin\psi, 0]^T$ for all tilt angles $\tau$.

#### Case B: `finger_axis: "y"`
- At zero tilt ($\tau = 0$): Gripper fingers align with local $Y = [\cos\psi, \sin\psi, 0]^T$; approach vector points downward $[0, 0, -1]^T$.
- Base Right-Handed Orthonormal Frame:
  $$R_{\text{base}}(\psi) = R_z(\psi) \cdot \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & -1 \end{bmatrix} = \begin{bmatrix} -\sin\psi & \cos\psi & 0 \\ \cos\psi & \sin\psi & 0 \\ 0 & 0 & -1 \end{bmatrix}$$
- Rotation Formula (tilting about local $Y$):
  $$R_{\text{target}}(\psi, \tau) = R_{\text{base}}(\psi) \cdot R_y(\tau)$$
  Where $R_y(\tau) = \begin{bmatrix} \cos\tau & 0 & \sin\tau \\ 0 & 1 & 0 \\ -\sin\tau & 0 & \cos\tau \end{bmatrix}$.
  Notice that column 1 ($Y_{\text{tcp}}$) is strictly $[\cos\psi, \sin\psi, 0]^T$ for all tilt angles $\tau$.

#### Candidate Tilt Search Order:
Candidate tilt angles are searched deterministically in order of increasing tilt to prioritize clean downward picking:
$$\mathcal{T} = [0.0^\circ,\, 10.0^\circ,\, -10.0^\circ,\, 20.0^\circ,\, -20.0^\circ,\, 30.0^\circ,\, -30.0^\circ,\, 45.0^\circ,\, -45.0^\circ,\, 60.0^\circ,\, -60.0^\circ,\, 75.0^\circ,\, -75.0^\circ,\, 90.0^\circ,\, -90.0^\circ]$$

---

### 5.3 5-DOF Task-Priority Nullspace Inverse Kinematics

A 5-DOF articulated robot has 5 joint variables. Controlling 3D Cartesian position requires 3 DOFs:
$$\text{dim}(\mathbf{q}) = 5, \quad \text{dim}(\mathbf{p}) = 3 \implies \text{dim}(\text{Nullspace}) = 5 - 3 = 2$$

Only 2 DOFs remain in the nullspace. It is mathematically impossible for a 5-DOF arm to track an arbitrary 3D rotation (which requires 3 DOFs).

We formulate a **Task-Priority Nullspace Projection** scheme:
- **Primary Task (Hard)**: 3D Cartesian position $\mathbf{p}^*$ must converge within $1.0\text{ mm}$.
- **Secondary Task (Soft / Nullspace)**: Align the gripper finger axis with the object yaw vector without disturbing the primary position. Approach tilt is permitted to float.

```
                  ┌────────────────────────────────────────┐
                  │       Hard Primary Task: e_p           │
                  │   Position Error: p* - p_current       │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                        dq_p = J_v^† e_p  (Position step)
                                      │
                                      ▼
                       N = I - J_v^† J_v  (Nullspace Projector)
                                      │
                  ┌───────────────────▼────────────────────┐
                  │     Secondary Task in Nullspace        │
                  │   Decompose e_r into e_perp & e_par    │
                  │   dq_w = (J_w N)^† (e_perp + w_tilt*e) │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                     dq = dq_p + N dq_w  (Combined Step)
```

#### Step 1: Primary Task Position Solve
Position Jacobian: $J_v \in \mathbb{R}^{3 \times 5}$.
Damped Moore-Penrose pseudoinverse:
$$J_v^\dagger = J_v^T \left( J_v J_v^T + \lambda_p^2 I_{3 \times 3} \right)^{-1}$$
Primary joint displacement:
$$\Delta \mathbf{q}_p = J_v^\dagger \mathbf{e}_p$$

#### Step 2: Nullspace Projector Construction
The orthogonal projection operator onto the nullspace of the position Jacobian is:
$$N = I_{5 \times 5} - J_v^\dagger J_v \in \mathbb{R}^{5 \times 5}$$
By definition, for any arbitrary vector $\mathbf{z} \in \mathbb{R}^5$, $J_v (N \mathbf{z}) = \mathbf{0}$. Motion in the nullspace causes **zero** Cartesian position deviation.

#### Step 3: Secondary Task Orientation Decomposition
Let $\mathbf{u}_{\text{finger}}$ be the active gripper finger unit vector in base frame coordinates:
$$\mathbf{u}_{\text{finger}} = \begin{cases} R_{:, 0} & \text{if } \text{finger\_axis} = \text{'x'} \\ R_{:, 1} & \text{if } \text{finger\_axis} = \text{'y'} \end{cases}$$

The total 3D rotation error $\mathbf{e}_r = \frac{1}{2}(\mathbf{n}\times\mathbf{n}^* + \mathbf{s}\times\mathbf{s}^* + \mathbf{a}\times\mathbf{a}^*)$ is decomposed into components parallel and perpendicular to the finger axis:

$$\mathbf{e}_{r, \parallel} = (\mathbf{e}_r \cdot \mathbf{u}_{\text{finger}}) \mathbf{u}_{\text{finger}} \quad (\text{Rotation around finger axis = approach tilt})$$
$$\mathbf{e}_{r, \perp} = \mathbf{e}_r - \mathbf{e}_{r, \parallel} \quad (\text{Rotation orthogonal to finger axis = finger misalignment})$$

#### Step 4: Nullspace Projection of Orientation Task
The angular Jacobian projected into the position nullspace is:
$$J_{N, \omega} = J_\omega N \in \mathbb{R}^{3 \times 5}$$

Its damped pseudoinverse is:
$$J_{N, \omega}^\dagger = J_{N, \omega}^T \left( J_{N, \omega} J_{N, \omega}^T + \lambda_\omega^2 I_{3 \times 3} \right)^{-1}$$

Secondary joint velocity:
$$\Delta \mathbf{q}_\omega = J_{N, \omega}^\dagger \left( \mathbf{e}_{r, \perp} + w_{\text{tilt}} \mathbf{e}_{r, \parallel} \right)$$
Where $w_{\text{tilt}} = 0.05$ weakly encourages approach tilt while prioritizing zero finger misalignment $\mathbf{e}_{r, \perp}$.

#### Step 5: Hierarchical Step Combination
$$\Delta \mathbf{q} = \begin{cases}
\alpha_p \Delta \mathbf{q}_p & \text{if } \|\mathbf{e}_p\| > 0.8 \epsilon_p \\
\alpha_p \Delta \mathbf{q}_p + \alpha_\omega N \Delta \mathbf{q}_\omega & \text{if } \|\mathbf{e}_p\| \le 0.8 \epsilon_p
\end{cases}$$
Where $\alpha_p = 0.8, \alpha_\omega = 0.5$.

---

### 5.4 6-DOF Balanced Pose IK

For 6-DOF arms (Robot 3), $\text{dim}(\mathbf{q}) = 6$. The robot possesses enough degrees of freedom to control both 3D position and full 3D orientation simultaneously:

$$\begin{bmatrix} w_p J_v \\ J_\omega \end{bmatrix} \Delta \mathbf{q} = \begin{bmatrix} w_p \mathbf{e}_p \\ \mathbf{e}_r \end{bmatrix}$$
$$\Delta \mathbf{q} = \left( J_{\text{scaled}}^T J_{\text{scaled}} + \lambda^2(\sigma_{\min}) I_{6 \times 6} \right)^{-1} J_{\text{scaled}}^T \mathbf{e}_{\text{scaled}}$$

The solver checks both position error $e_p \le \epsilon_p$ ($1.0\text{ mm}$) and orientation error $e_o \le \epsilon_o$ ($15.0^\circ$).

---

### 5.5 Lexicographic Feasibility Filter

Candidate joint solutions $\mathbf{q}$ from all seed iterations are filtered against four strict physical constraints:

1. **Cartesian Position Error**:
   $$e_p = \|\mathbf{p}^* - \mathbf{p}_{\text{tcp}}(\mathbf{q})\| \le \epsilon_p \quad (1.0\text{ mm})$$
2. **Finger Alignment / Orientation Error**:
   - For 5-DOF: $e_{\text{align}} \le \epsilon_{\text{align}}$ ($5.0^\circ$).
   - For 6-DOF: $e_o \le \epsilon_o$ ($15.0^\circ$).
3. **Hardware Joint Limit Margins**:
   $$\Delta q_{\text{margin}} = \min_{i \in \{1, \dots, n\}} \min\left( q_i - q_{\min, i},\; q_{\max, i} - q_i \right) \ge \epsilon_{\text{margin}} \quad (0.1^\circ)$$
4. **Kinematic Manipulability**:
   $$\sigma_{\min}(J(\mathbf{q})) \ge \sigma_{\text{threshold}} \quad (10^{-4})$$

Solutions violating any of these four criteria are strictly discarded (`is_valid = False`).

---

### 5.6 Normalized Multi-Objective Cost Ranking

All valid candidates are ranked using a normalized scalar objective cost function $C(\mathbf{q}) \in \mathbb{R}^+$:

$$C(\mathbf{q}) = w_\tau s_\tau + w_p s_p + w_{\text{align}} s_{\text{align}} + w_m s_m + w_\sigma s_\sigma + w_{\Delta q} s_{\Delta q}$$

Where each sub-score $s_i \in [0, 1]$ is dimensionless and normalized:

- **Tilt Penalty**: $s_\tau = \frac{|\tau_{\text{achieved}}|}{90.0^\circ}$ (Penalizes steep approach tilts; prefers downward $0^\circ$).
- **Position Precision**: $s_p = \min\left(\frac{e_p}{\epsilon_p},\, 5.0\right)$.
- **Alignment Precision**: $s_{\text{align}} = \min\left(\frac{e_{\text{align}}}{\epsilon_{\text{align}}},\, 5.0\right)$.
- **Joint Limit Margin**: $s_m = \frac{1}{1.0 + \Delta q_{\text{margin}}}$ (Prefers center of joint range).
- **Singularity Avoidance**: $s_\sigma = \frac{1}{1.0 + 100 \sigma_{\min}}$ (Prefers high manipulability).
- **Minimal Joint Motion**: $s_{\Delta q} = \frac{\|\mathbf{q} - \mathbf{q}_{\text{current}}\|}{\pi \sqrt{n}}$ (Prefers configurations nearest to current robot state).

Weights configured in `handover.yaml`:
$$w_\tau = 30.0, \quad w_p = 20.0, \quad w_{\text{align}} = 20.0, \quad w_m = 10.0, \quad w_\sigma = 10.0, \quad w_{\Delta q} = 35.0$$

The valid candidate with minimum cost $C^* = \min_k C(\mathbf{q}_k)$ is selected for execution.

---

### 5.7 Independent Pick/Place Solver Fallback Logic

Stage 2.5 treats Pick and Place independently:

$$\text{SolveTarget}(\mathbf{p}, \psi) = \begin{cases}
\text{ObjectPickIK}(\mathbf{p}, \psi) & \text{if at least one robot succeeds} \\
\text{Stage1\_PositionIK}(\mathbf{p}) & \text{otherwise (Fallback Mode)}
\end{cases}$$

```
                ┌──────────────────────────────────────┐
                │          Input Target XYZ + Yaw      │
                └──────────────────┬───────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────────┐
                │  Attempt Object-Pose IK on All Robots│
                │     (Adaptive Tilts + Finger Yaw)    │
                └──────────────────┬───────────────────┘
                                   │
                        ┌──────────┴──────────┐
                  Solutions Found?       No Solution
                        │                     │
                        ▼                     ▼
             ┌─────────────────────┐┌────────────────────────┐
             │ Object-Pose IK Mode ││ Fallback Mode:         │
             │ (is_fallback = False││ Stage 1.5 Position IK  │
             │  Badge: GREEN)      ││ (is_fallback = True    │
             └─────────────────────┘│  Badge: ORANGE)        │
                                    └────────────────────────┘
```

This guarantees that extreme coordinates outside the orientation reach of all robots can still be picked or placed using standard position IK without breaking the handover pipeline.

---

### 5.8 Synchronized 9-Step Trajectory State Machine

When Pick and Place are assigned to two different robots ($R_{\text{pick}} \neq R_{\text{place}}$), Stage 2.5 executes a synchronized 9-phase dual-robot state machine:

```
Time ──►
Phase 1: R_pick ──► PICK TARGET
         R_place ──► PRE-HANDOVER (Stationed)
Phase 2: R_pick closes gripper (Dwell 2.0s)
Phase 3: R_pick ──► PRE-HANDOVER
         R_place waiting at PRE-HANDOVER
Phase 4: BOTH ADVANCE: Pre-Handover ──► Handover (Simultaneous at identical rate)
Phase 5: TRANSFER: R_place closes gripper; R_pick opens gripper (Dwell 1.0s)
Phase 6: BOTH RETRACT: Handover ──► Pre-Handover (Simultaneous at identical rate)
Phase 7: R_place ──► PLACE TARGET
         R_pick waiting at PRE-HANDOVER
Phase 8: R_place opens gripper (Dwell 2.0s)
Phase 9: BOTH RETURN: Return to configured Home positions
```

#### Synchronized Advance / Retract Velocity Matching:
During Phases 4 and 6, both robots must move simultaneously at identical rates:
$$\mathbf{q}_{\text{pick}}(s) = \mathbf{q}_{\text{pick, pre}} + s (\mathbf{q}_{\text{pick, handover}} - \mathbf{q}_{\text{pick, pre}})$$
$$\mathbf{q}_{\text{place}}(s) = \mathbf{q}_{\text{place, pre}} + s (\mathbf{q}_{\text{place, handover}} - \mathbf{q}_{\text{place, pre}})$$
For $s \in [0, 1]$ parameterized in $K = 30$ discrete trajectory steps.

---

### 5.9 Zero-Recalculation Simulation Plan Caching

Computing full 6D collision avoidance, multi-tilt object IK, and trajectory discretization takes 15–25 seconds. To provide instantaneous real hardware execution:

1. During `SIMULATE`:
   Upon plan completion, the full trajectory result is hashed by target key:
   $$\mathcal{K} = \left(\mathbf{p}_{\text{pick}},\, \psi_{\text{pick}},\, \mathbf{p}_{\text{place}},\, \psi_{\text{place}}\right)$$
   Stored in memory:
   $$\text{cache}[\mathcal{K}] = \{\text{plan},\, \text{trajectory\_result},\, \text{min\_clearance}\}$$
2. During `EXECUTE`:
   The system queries $\mathcal{K}_{\text{current}} == \mathcal{K}_{\text{cached}}$:
   - If match: **Bypasses all IK and collision checking**, directly reuses the pre-verified trajectory, and executes immediately.
   - If mismatch: Generates a new plan automatically.
