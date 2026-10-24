/**
 * rt_AI 3D WEB DIGITAL TWIN — ROBOT MODELS
 * Procedural PBR 3D Articulated Robot Arm & Gripper Builders in Three.js
 */

class RobotArmModel {
  /**
   * @param {string} id - 'robot1', 'robot2', or 'robot3'
   * @param {Object} config - Robot kinematics and visual configuration
   * @param {THREE.Scene} scene
   */
  constructor(id, config, scene) {
    this.id = id;
    this.config = config;
    this.scene = scene;
    this.dof = config.dof;
    this.basePos = new THREE.Vector3(...config.base_xyz);
    this.fingerAxis = config.finger_axis || 'x';

    // Accent theme by robot
    if (id === 'robot1') {
      this.accentHex = 0x00F0FF; // Luminous Neon Cyan
      this.linkHex = 0xD1DCE8;   // Gleaming Polished Aluminum
    } else if (id === 'robot2') {
      this.accentHex = 0xBD00FF; // Luminous Neon Violet
      this.linkHex = 0xDBD0E8;   // Gleaming Polished Titanium
    } else {
      this.accentHex = 0xFFB800; // Luminous Neon Gold / Amber
      this.linkHex = 0xE8DFCE;   // Gleaming Polished Platinum
    }

    this.group = new THREE.Group();
    this.group.name = `robot_group_${id}`;

    // Link meshes & joint hubs
    this.linkMeshes = [];
    this.jointHubs = [];
    this.gripperGroup = null;
    this.fingerLeft = null;
    this.fingerRight = null;
    this.tcpAxesHelper = null;

    this._buildMaterials();
    this._buildBasePedestal();
    this._buildSkeleton();

    this.scene.add(this.group);
  }

  _buildMaterials() {
    this.matLink = new THREE.MeshStandardMaterial({
      color: this.linkHex,
      metalness: 0.75,
      roughness: 0.2,
    });

    this.matAccent = new THREE.MeshStandardMaterial({
      color: this.accentHex,
      emissive: this.accentHex,
      emissiveIntensity: 0.6,
      metalness: 0.4,
      roughness: 0.15,
    });

    this.matJoint = new THREE.MeshStandardMaterial({
      color: 0x1E293B,
      metalness: 0.9,
      roughness: 0.2,
    });

    this.matGripper = new THREE.MeshStandardMaterial({
      color: 0x0F172A,
      metalness: 0.95,
      roughness: 0.15,
    });
  }

  _buildBasePedestal() {
    const baseGroup = new THREE.Group();
    baseGroup.position.copy(this.basePos);

    // Bottom mounting plate
    const plateGeo = new THREE.CylinderGeometry(70, 75, 12, 32);
    const plateMesh = new THREE.Mesh(plateGeo, this.matJoint);
    plateMesh.position.y = 6;
    plateMesh.receiveShadow = true;
    baseGroup.add(plateMesh);

    // Glowing ID ring
    const ringGeo = new THREE.TorusGeometry(72, 2.5, 16, 48);
    const ringMesh = new THREE.Mesh(ringGeo, this.matAccent);
    ringMesh.rotation.x = Math.PI / 2;
    ringMesh.position.y = 12;
    baseGroup.add(ringMesh);

    // Turntable column
    const colGeo = new THREE.CylinderGeometry(55, 65, 30, 32);
    const colMesh = new THREE.Mesh(colGeo, this.matLink);
    colMesh.position.y = 25;
    colMesh.castShadow = true;
    baseGroup.add(colMesh);

    this.group.add(baseGroup);
  }

  _buildSkeleton() {
    const numFrames = this.config.frames_world.length;

    // Build joint pivot hubs
    for (let i = 0; i < numFrames; i++) {
      const hubGeo = new THREE.SphereGeometry(14, 24, 24);
      const hubMesh = new THREE.Mesh(hubGeo, this.matJoint);
      hubMesh.castShadow = true;

      // Add small accent band
      const bandGeo = new THREE.TorusGeometry(14.5, 1.2, 12, 24);
      const bandMesh = new THREE.Mesh(bandGeo, this.matAccent);
      hubMesh.add(bandMesh);

      this.jointHubs.push(hubMesh);
      this.group.add(hubMesh);
    }

    // Build link cylinders connecting consecutive frames
    for (let i = 0; i < numFrames - 1; i++) {
      // Cylindrical link segment
      const linkGeo = new THREE.CylinderGeometry(10, 10, 1, 20);
      const linkMesh = new THREE.Mesh(linkGeo, this.matLink);
      linkMesh.castShadow = true;
      this.linkMeshes.push(linkMesh);
      this.group.add(linkMesh);
    }

    // Build parallel gripper at end-effector
    this._buildGripper();
  }

  _buildGripper() {
    this.gripperGroup = new THREE.Group();
    this.gripperGroup.name = `gripper_${this.id}`;

    // Gripper palm base
    const palmGeo = new THREE.BoxGeometry(36, 16, 24);
    const palmMesh = new THREE.Mesh(palmGeo, this.matGripper);
    palmMesh.castShadow = true;
    this.gripperGroup.add(palmMesh);

    // Finger jaws (left and right)
    const fingerGeo = new THREE.BoxGeometry(6, 32, 12);

    this.fingerLeft = new THREE.Mesh(fingerGeo, this.matAccent);
    this.fingerLeft.position.set(-14, 20, 0);
    this.fingerLeft.castShadow = true;
    this.gripperGroup.add(this.fingerLeft);

    this.fingerRight = new THREE.Mesh(fingerGeo, this.matAccent);
    this.fingerRight.position.set(14, 20, 0);
    this.fingerRight.castShadow = true;
    this.gripperGroup.add(this.fingerRight);

    // TCP RGB Coordinate Axes Helper
    this.tcpAxesHelper = new THREE.AxesHelper(35);
    this.tcpAxesHelper.position.set(0, 36, 0);
    this.gripperGroup.add(this.tcpAxesHelper);

    this.group.add(this.gripperGroup);
  }

  /**
   * Update all 3D mesh positions from server world frame coordinates.
   * @param {Array<Array<number>>} framesWorld - 3D points [[X,Y,Z], ...]
   * @param {Array<Array<number>>} tcpR - 3x3 orientation matrix in world frame
   * @param {string|number} gripperState - 'open' | 'close' | numeric angle
   */
  updatePose(framesWorld, tcpR = null, gripperState = 'open') {
    if (!framesWorld || framesWorld.length === 0) return;

    // In Three.js, Y is UP (Z in robotics is ceiling/height).
    // Mapping: Three.X = World.X, Three.Y = World.Z, Three.Z = -World.Y
    const toThreeVec = (pt) => new THREE.Vector3(pt[0], pt[2], -pt[1]);

    const numPoints = framesWorld.length;

    // Update joint hubs
    for (let i = 0; i < numPoints && i < this.jointHubs.length; i++) {
      const v = toThreeVec(framesWorld[i]);
      this.jointHubs[i].position.copy(v);
    }

    // Update link cylinders
    for (let i = 0; i < numPoints - 1 && i < this.linkMeshes.length; i++) {
      const p1 = toThreeVec(framesWorld[i]);
      const p2 = toThreeVec(framesWorld[i + 1]);

      const link = this.linkMeshes[i];
      const dist = p1.distanceTo(p2);

      if (dist < 1.0) {
        link.visible = false;
        continue;
      }

      link.visible = true;
      // Re-scale cylinder height along local Y
      link.scale.set(1, dist, 1);
      // Midpoint position
      link.position.copy(p1).add(p2).multiplyScalar(0.5);

      // Orientation from (0, 1, 0) to (p2 - p1)
      const dir = new THREE.Vector3().subVectors(p2, p1).normalize();
      link.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    }

    // Update gripper pose at TCP / final flange frame
    if (this.gripperGroup && numPoints >= 2) {
      const tcpPos = toThreeVec(framesWorld[numPoints - 1]);
      const prevPos = toThreeVec(framesWorld[numPoints - 2]);
      this.gripperGroup.position.copy(tcpPos);

      // Orientation from TCP rotation matrix or approach direction
      if (tcpR && tcpR.length === 3) {
        // Construct 3x3 matrix in Three.js coordinate system
        const m = new THREE.Matrix4();
        // Robotics World to Three.js mapping:
        // [X_three, Y_three, Z_three] = [X_rob, Z_rob, -Y_rob]
        const r0 = tcpR[0], r1 = tcpR[1], r2 = tcpR[2];
        m.set(
          r0[0],  r0[2], -r0[1], 0,
          r2[0],  r2[2], -r2[1], 0,
         -r1[0], -r1[2],  r1[1], 0,
          0,      0,      0,     1
        );
        this.gripperGroup.quaternion.setFromRotationMatrix(m);
      } else {
        const approachDir = new THREE.Vector3().subVectors(tcpPos, prevPos).normalize();
        this.gripperGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), approachDir);
      }

      // Animate gripper finger jaws open / close
      const isClosed = gripperState === 'close' || (typeof gripperState === 'number' && gripperState > 10);
      const targetOffset = isClosed ? 5 : 15;
      this.fingerLeft.position.x = -targetOffset;
      this.fingerRight.position.x = targetOffset;
    }
  }
}
