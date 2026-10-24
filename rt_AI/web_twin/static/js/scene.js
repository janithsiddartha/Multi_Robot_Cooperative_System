/**
 * rt_AI 3D WEB DIGITAL TWIN — THREE.JS SCENE & 3D INTERACTION
 * Scene setup, Studio Lights, Interactive Transform Gizmos, Camera Presets & Workpiece
 */

class WebTwinScene {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.robots = {};
    this.laser = null;
    this.workpiece = null;
    this.activeGizmo = null;
    this.onTargetMovedCallback = null;

    this._initScene();
    this._initLights();
    this._initWorkcellFloor();
    this._initTargetsAndGizmos();
    this._initWorkpiece();

    window.addEventListener('resize', () => this.onWindowResize());
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  _initScene() {
    this.scene = new THREE.Scene();
    // High-visibility illuminated engineering studio background
    this.scene.background = new THREE.Color(0x182234);
    // Subtle, gentle atmospheric depth (no murky darkness)
    this.scene.fog = new THREE.FogExp2(0x182234, 0.00015);

    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;

    this.camera = new THREE.PerspectiveCamera(45, w / h, 10, 5000);
    // Initial isometric view (Robotics Z=up is Three.js Y=up)
    this.camera.position.set(0, 700, 750);

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxPolarAngle = Math.PI / 2 + 0.05; // Don't flip below floor
    this.controls.minDistance = 100;
    this.controls.maxDistance = 2500;
    this.controls.target.set(0, 150, 0);
  }

  _initLights() {
    // 1. Brilliant uniform ambient light
    const ambLight = new THREE.AmbientLight(0xFFFFFF, 2.0);
    this.scene.add(ambLight);

    // 2. Sky-to-Ground natural hemisphere fill
    const hemiLight = new THREE.HemisphereLight(0xF8FAFC, 0x475569, 1.8);
    this.scene.add(hemiLight);

    // 3. Overhead Direct Downward Workspace Floodlight (at Y=850 directly over the 3 robots)
    const overheadFlood = new THREE.DirectionalLight(0xFFFFFF, 2.2);
    overheadFlood.position.set(0, 850, 0);
    overheadFlood.target.position.set(0, 0, 0);
    this.scene.add(overheadFlood);
    this.scene.add(overheadFlood.target);

    // 4. Main studio key light (Overhead Angle)
    const dirLight = new THREE.DirectionalLight(0xFFFFFF, 2.6);
    dirLight.position.set(500, 1000, 600);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 100;
    dirLight.shadow.camera.far = 2500;
    dirLight.shadow.camera.left = -800;
    dirLight.shadow.camera.right = 800;
    dirLight.shadow.camera.top = 800;
    dirLight.shadow.camera.bottom = -800;
    dirLight.shadow.bias = -0.0005;
    this.scene.add(dirLight);

    // 5. Fill Light 1 (Back-Left Fill)
    const fillLight1 = new THREE.DirectionalLight(0xF1F5F9, 1.8);
    fillLight1.position.set(-700, 700, -500);
    this.scene.add(fillLight1);

    // 6. Fill Light 2 (Back-Right Fill)
    const fillLight2 = new THREE.DirectionalLight(0xF1F5F9, 1.6);
    fillLight2.position.set(700, 700, -400);
    this.scene.add(fillLight2);

    // 7. Fill Light 3 (Front Low Fill for underside arm clarity)
    const fillLight3 = new THREE.DirectionalLight(0xE2E8F0, 1.5);
    fillLight3.position.set(0, 350, 800);
    this.scene.add(fillLight3);

    // 8. Accent Point Lights at Robot Bases (amplified radius and power)
    const r1Point = new THREE.PointLight(0x00F0FF, 3.0, 750);
    r1Point.position.set(-250, 70, 0);
    this.scene.add(r1Point);

    const r2Point = new THREE.PointLight(0xBD00FF, 3.0, 750);
    r2Point.position.set(250, 70, 0);
    this.scene.add(r2Point);

    const r3Point = new THREE.PointLight(0xFFB800, 3.0, 750);
    r3Point.position.set(0, 70, -250);
    this.scene.add(r3Point);
  }

  _initWorkcellFloor() {
    // 1. Expansive Cleanroom Laboratory Floor Disc (radius 2200mm, prevents black void)
    const floorGeo = new THREE.CylinderGeometry(2200, 2200, 10, 64);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x111827,
      metalness: 0.2,
      roughness: 0.8,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.position.y = -12.5;
    floor.receiveShadow = true;
    this.scene.add(floor);

    // Laboratory Outer Grid
    const outerGrid = new THREE.GridHelper(3000, 60, 0x38BDF8, 0x334155);
    outerGrid.position.y = -7.0;
    this.scene.add(outerGrid);

    // 2. Shared Workcell Table Mesh (Brushed Titanium Finish)
    const tableGeo = new THREE.CylinderGeometry(600, 600, 15, 64);
    const tableMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      metalness: 0.55,
      roughness: 0.22,
    });
    const table = new THREE.Mesh(tableGeo, tableMat);
    table.position.y = -7.5;
    table.receiveShadow = true;
    this.scene.add(table);

    // Luminous Cyan Table Rim Accent Ring
    const rimGeo = new THREE.TorusGeometry(600, 3.5, 16, 100);
    const rimMat = new THREE.MeshStandardMaterial({
      color: 0x00F0FF,
      emissive: 0x00F0FF,
      emissiveIntensity: 0.75,
    });
    const rimMesh = new THREE.Mesh(rimGeo, rimMat);
    rimMesh.rotation.x = Math.PI / 2;
    this.scene.add(rimMesh);

    // High-Contrast Metric Grid on Table Surface (Cyan Center & Bright Slate Lines)
    const grid = new THREE.GridHelper(1000, 40, 0x00F0FF, 0x64748B);
    grid.position.y = 0.5;
    this.scene.add(grid);

    // 3. Overhead Circular LED Light Ring (Suspended at Y=750mm above workspace)
    const ledRingGeo = new THREE.TorusGeometry(420, 5, 16, 80);
    const ledRingMat = new THREE.MeshStandardMaterial({
      color: 0xFFFFFF,
      emissive: 0xFFFFFF,
      emissiveIntensity: 1.5,
      metalness: 0.1,
      roughness: 0.1,
    });
    const ledRing = new THREE.Mesh(ledRingGeo, ledRingMat);
    ledRing.position.set(0, 750, 0);
    ledRing.rotation.x = Math.PI / 2;
    this.scene.add(ledRing);

    // Handover Zone Floating Ring (Z = 362.5 mm -> Three.js Y = 362.5)
    const hoRingGeo = new THREE.TorusGeometry(80, 2.5, 16, 64);
    const hoRingMat = new THREE.MeshBasicMaterial({
      color: 0x00FF9D,
      transparent: true,
      opacity: 0.65,
    });
    this.hoRing = new THREE.Mesh(hoRingGeo, hoRingMat);
    this.hoRing.rotation.x = Math.PI / 2;
    this.hoRing.position.set(37.5, 362.5, 37.5);
    this.scene.add(this.hoRing);
  }

  _initTargetsAndGizmos() {
    // ─────────────────────────────────────────────────────────────
    // 1. Pick Target Marker & Transform Gizmo (Cyan)
    // ─────────────────────────────────────────────────────────────
    const pickGeo = new THREE.OctahedronGeometry(18, 0);
    const pickMat = new THREE.MeshStandardMaterial({
      color: 0x00F0FF,
      emissive: 0x00F0FF,
      emissiveIntensity: 0.5,
      metalness: 0.2,
      roughness: 0.2,
      transparent: true,
      opacity: 0.85,
    });
    this.pickMarker = new THREE.Mesh(pickGeo, pickMat);
    this.pickMarker.name = 'pick_target_marker';
    // Three.js [X, Z, -Y] = Robotics [-200, 150, 150] -> [-200, 150, -150]
    this.pickMarker.position.set(-200, 150, -150);
    this.scene.add(this.pickMarker);

    this.pickGizmo = new THREE.TransformControls(this.camera, this.renderer.domElement);
    this.pickGizmo.size = 0.75;
    this.pickGizmo.attach(this.pickMarker);
    this.scene.add(this.pickGizmo);

    // ─────────────────────────────────────────────────────────────
    // 2. Place Target Marker & Transform Gizmo (Violet)
    // ─────────────────────────────────────────────────────────────
    const placeGeo = new THREE.OctahedronGeometry(18, 0);
    const placeMat = new THREE.MeshStandardMaterial({
      color: 0xBD00FF,
      emissive: 0xBD00FF,
      emissiveIntensity: 0.5,
      metalness: 0.2,
      roughness: 0.2,
      transparent: true,
      opacity: 0.85,
    });
    this.placeMarker = new THREE.Mesh(placeGeo, placeMat);
    this.placeMarker.name = 'place_target_marker';
    // Three.js [X, Z, -Y] = Robotics [150, 100, 180] -> [150, 180, -100]
    this.placeMarker.position.set(150, 180, -100);
    this.scene.add(this.placeMarker);

    this.placeGizmo = new THREE.TransformControls(this.camera, this.renderer.domElement);
    this.placeGizmo.size = 0.75;
    this.placeGizmo.attach(this.placeMarker);
    this.scene.add(this.placeGizmo);

    // Disable OrbitControls while dragging gizmos
    const setupGizmoEvents = (gizmo, type) => {
      gizmo.addEventListener('dragging-changed', (event) => {
        this.controls.enabled = !event.value;
      });
      gizmo.addEventListener('change', () => {
        if (this.onTargetMovedCallback) {
          const robCoord = this.toRoboticsCoord(gizmo.object.position);
          const yawDeg = THREE.MathUtils.radToDeg(gizmo.object.rotation.y);
          this.onTargetMovedCallback(type, robCoord, yawDeg);
        }
      });
    };

    setupGizmoEvents(this.pickGizmo, 'pick');
    setupGizmoEvents(this.placeGizmo, 'place');
  }

  _initWorkpiece() {
    // 3D Workpiece Cylinder with Chamfered Brass Accent Ring
    this.workpiece = new THREE.Group();
    this.workpiece.name = 'physical_workpiece';

    const cylGeo = new THREE.CylinderGeometry(14, 14, 38, 32);
    const cylMat = new THREE.MeshStandardMaterial({
      color: 0xE2E8F0,
      metalness: 0.9,
      roughness: 0.2,
    });
    const cylMesh = new THREE.Mesh(cylGeo, cylMat);
    cylMesh.castShadow = true;
    this.workpiece.add(cylMesh);

    // Center Accent Ring
    const bandGeo = new THREE.CylinderGeometry(14.8, 14.8, 8, 32);
    const bandMat = new THREE.MeshStandardMaterial({
      color: 0xFFB800,
      metalness: 0.8,
      roughness: 0.25,
    });
    const bandMesh = new THREE.Mesh(bandGeo, bandMat);
    this.workpiece.add(bandMesh);

    // Initial position on table at Pick target
    const initRob = [-200, 150, 19];
    this.workpiece.position.set(initRob[0], initRob[2], -initRob[1]);
    this.scene.add(this.workpiece);
  }

  /**
   * Set Gizmo transformation mode: 'translate' or 'rotate'
   */
  setGizmoMode(mode) {
    if (this.pickGizmo) this.pickGizmo.setMode(mode);
    if (this.placeGizmo) this.placeGizmo.setMode(mode);
  }

  /**
   * Robotics [X, Y, Z] -> Three.js [X, Z, -Y]
   */
  toThreeCoord(rob) {
    return new THREE.Vector3(rob[0], rob[2], -rob[1]);
  }

  /**
   * Three.js [X, Y, Z] -> Robotics [X, -Z, Y]
   */
  toRoboticsCoord(threePos) {
    return [
      Math.round(threePos.x * 10) / 10,
      Math.round(-threePos.z * 10) / 10,
      Math.round(threePos.y * 10) / 10,
    ];
  }

  /**
   * Update Pick Target position & yaw from UI inputs
   */
  setPickTarget(xyz, yawDeg = 0) {
    if (this.pickMarker) {
      this.pickMarker.position.copy(this.toThreeCoord(xyz));
      this.pickMarker.rotation.y = THREE.MathUtils.degToRad(-yawDeg);
    }
  }

  /**
   * Update Place Target position & yaw from UI inputs
   */
  setPlaceTarget(xyz, yawDeg = 0) {
    if (this.placeMarker) {
      this.placeMarker.position.copy(this.toThreeCoord(xyz));
      this.placeMarker.rotation.y = THREE.MathUtils.degToRad(-yawDeg);
    }
  }

  /**
   * Update Handover ring position in 3D
   */
  setHandoverPosition(xyz) {
    if (this.hoRing) {
      this.hoRing.position.copy(this.toThreeCoord(xyz));
    }
  }

  /**
   * Update Workpiece physical 3D location
   */
  updateWorkpiece(wpData) {
    if (!this.workpiece || !wpData) return;
    const pos = this.toThreeCoord(wpData.position);
    this.workpiece.position.copy(pos);
    if (wpData.rotation) {
      this.workpiece.rotation.y = THREE.MathUtils.degToRad(-wpData.rotation[2]);
    }
  }

  /**
   * Switch Camera to Preset Angle
   * @param {string} presetName - 'isometric' | 'top' | 'front' | 'handover' | 'r1' | 'r2' | 'r3'
   */
  setCameraPreset(presetName) {
    switch (presetName) {
      case 'isometric':
        this.camera.position.set(0, 700, 750);
        this.controls.target.set(0, 150, 0);
        break;
      case 'top':
        this.camera.position.set(0, 1100, 1);
        this.controls.target.set(0, 0, 0);
        break;
      case 'front':
        this.camera.position.set(0, 300, 950);
        this.controls.target.set(0, 180, 0);
        break;
      case 'handover':
        this.camera.position.set(0, 520, 320);
        this.controls.target.set(37.5, 362.5, 37.5);
        break;
      case 'r1':
        this.camera.position.set(-550, 450, 250);
        this.controls.target.set(-250, 200, 0);
        break;
      case 'r2':
        this.camera.position.set(550, 450, 250);
        this.controls.target.set(250, 200, 0);
        break;
      case 'r3':
        this.camera.position.set(0, 450, -550);
        this.controls.target.set(0, 200, -250);
        break;
    }
    this.controls.update();
  }

  onWindowResize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  animate() {
    requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
