/**
 * rt_AI 3D WEB DIGITAL TWIN — LASER & CLEARANCE GAUGE
 * Dynamic Glowing 3D Laser Beam & Real-Time Telemetry HUD Arc Gauge
 */

class DynamicClearanceLaser {
  /**
   * @param {THREE.Scene} scene
   */
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.group.name = 'clearance_laser_group';

    // Materials
    this.matCore = new THREE.MeshBasicMaterial({
      color: 0x00FF9D,
      transparent: true,
      opacity: 0.95,
    });

    this.matGlow = new THREE.MeshBasicMaterial({
      color: 0x00FF9D,
      transparent: true,
      opacity: 0.35,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
    });

    this.matBeacon = new THREE.MeshStandardMaterial({
      color: 0x00FF9D,
      emissive: 0x00FF9D,
      emissiveIntensity: 1.0,
      metalness: 0.1,
      roughness: 0.1,
    });

    // 3D Laser Cylinder Core
    const coreGeo = new THREE.CylinderGeometry(1.5, 1.5, 1, 16);
    this.coreMesh = new THREE.Mesh(coreGeo, this.matCore);
    this.group.add(this.coreMesh);

    // 3D Outer Glow Tube
    const glowGeo = new THREE.CylinderGeometry(4.0, 4.0, 1, 16);
    this.glowMesh = new THREE.Mesh(glowGeo, this.matGlow);
    this.group.add(this.glowMesh);

    // Endpoint Beacon Spheres
    const beaconGeo = new THREE.SphereGeometry(6, 16, 16);
    this.beaconA = new THREE.Mesh(beaconGeo, this.matBeacon);
    this.beaconB = new THREE.Mesh(beaconGeo, this.matBeacon);
    this.group.add(this.beaconA);
    this.group.add(this.beaconB);

    this.scene.add(this.group);
    this.currentDist = 500;
  }

  /**
   * Update laser endpoints and color based on clearance distance.
   * @param {number} distMm - Minimum distance in millimeters
   * @param {Array<number>} ptA - [X, Y, Z] on Robot A
   * @param {Array<number>} ptB - [X, Y, Z] on Robot B
   * @param {number} safeThreshold - Default 45 mm
   */
  update(distMm, ptA, ptB, safeThreshold = 45.0) {
    if (!ptA || !ptB) {
      this.group.visible = false;
      return;
    }

    this.group.visible = true;
    this.currentDist = distMm;

    // Coordinate conversion to Three.js: [X, Z, -Y]
    const pA = new THREE.Vector3(ptA[0], ptA[2], -ptA[1]);
    const pB = new THREE.Vector3(ptB[0], ptB[2], -ptB[1]);

    this.beaconA.position.copy(pA);
    this.beaconB.position.copy(pB);

    const length = pA.distanceTo(pB);
    if (length < 0.5) {
      this.coreMesh.visible = false;
      this.glowMesh.visible = false;
      return;
    }

    this.coreMesh.visible = true;
    this.glowMesh.visible = true;

    // Scale laser cylinders
    this.coreMesh.scale.set(1, length, 1);
    this.glowMesh.scale.set(1, length, 1);

    // Position at midpoint
    const mid = new THREE.Vector3().addVectors(pA, pB).multiplyScalar(0.5);
    this.coreMesh.position.copy(mid);
    this.glowMesh.position.copy(mid);

    // Orientation
    const dir = new THREE.Vector3().subVectors(pB, pA).normalize();
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    this.coreMesh.quaternion.copy(q);
    this.glowMesh.quaternion.copy(q);

    // Determine laser color & pulsing glow
    let colorHex = 0x00FF9D; // Green
    if (distMm < safeThreshold) {
      colorHex = 0xFF0055; // Red Alert
    } else if (distMm < safeThreshold + 20) {
      colorHex = 0xFFB800; // Amber Warning
    }

    this.matCore.color.setHex(colorHex);
    this.matGlow.color.setHex(colorHex);
    this.matBeacon.color.setHex(colorHex);
    this.matBeacon.emissive.setHex(colorHex);

    // Update HUD Widget if present
    this._updateHudGauge(distMm, safeThreshold);
  }

  _updateHudGauge(distMm, safeThreshold) {
    const valElem = document.getElementById('gauge-clearance-val');
    const fillElem = document.getElementById('gauge-bar-fill');
    const badgeElem = document.getElementById('gauge-status-badge');

    if (!valElem || !fillElem) return;

    valElem.textContent = distMm > 999 ? '> 999' : distMm.toFixed(1);

    // Bar fill percentage: 0mm to 200mm range
    const pct = Math.min(100, Math.max(0, (distMm / 200.0) * 100));
    fillElem.style.width = `${pct}%`;

    if (distMm >= safeThreshold + 20) {
      valElem.style.color = '#00FF9D';
      fillElem.style.background = '#00FF9D';
      fillElem.style.boxShadow = '0 0 10px #00FF9D';
      if (badgeElem) {
        badgeElem.textContent = 'SAFE';
        badgeElem.style.color = '#00FF9D';
        badgeElem.style.borderColor = 'rgba(0, 255, 157, 0.4)';
      }
    } else if (distMm >= safeThreshold) {
      valElem.style.color = '#FFB800';
      fillElem.style.background = '#FFB800';
      fillElem.style.boxShadow = '0 0 10px #FFB800';
      if (badgeElem) {
        badgeElem.textContent = 'APPROACHING';
        badgeElem.style.color = '#FFB800';
        badgeElem.style.borderColor = 'rgba(255, 184, 0, 0.4)';
      }
    } else {
      valElem.style.color = '#FF0055';
      fillElem.style.background = '#FF0055';
      fillElem.style.boxShadow = '0 0 14px #FF0055';
      if (badgeElem) {
        badgeElem.textContent = 'CRITICAL PROXIMITY';
        badgeElem.style.color = '#FF0055';
        badgeElem.style.borderColor = 'rgba(255, 0, 85, 0.4)';
      }
    }
  }
}
