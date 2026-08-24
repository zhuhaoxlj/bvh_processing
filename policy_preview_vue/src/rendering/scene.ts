import * as THREE from "three";
import type { G1Model } from "../simulation/mujoco";

const GEOM_PLANE = 0;
const GEOM_SPHERE = 2;
const GEOM_CAPSULE = 3;
const GEOM_ELLIPSOID = 4;
const GEOM_CYLINDER = 5;
const GEOM_BOX = 6;
const GEOM_MESH = 7;

// Soft pixel budget (~1080p @ 1.5×). Larger CSS windows / 4K DPR would otherwise
// multiply fragment cost and drag the fixed 50 Hz policy loop into slow motion.
const MAX_DEVICE_PIXELS = 1920 * 1080 * 1.5;
const MAX_PIXEL_RATIO = 2;
const SHADOW_MAP_HIGH = 2048;
const SHADOW_MAP_LOW = 1024;
const SHADOW_HIGH_BUDGET = 1600 * 900;

export class G1Scene {
  readonly scene = new THREE.Scene();
  readonly renderer: THREE.WebGLRenderer;
  readonly camera: THREE.PerspectiveCamera;
  private readonly root = new THREE.Group();
  private readonly geomMeshes: Array<THREE.Mesh | null> = [];
  private readonly geometryCache = new Map<string, THREE.BufferGeometry>();
  private readonly orbit = { azimuth: 0.55, elevation: 0.25, distance: 2.9 };
  private readonly target = new THREE.Vector3(0, 0, 0.85);
  private readonly keyLight: THREE.DirectionalLight;
  private dragStart: { x: number; y: number } | null = null;
  private readonly resizeObserver: ResizeObserver | null;
  private lastShadowSize = SHADOW_MAP_HIGH;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    this.camera.up.set(0, 0, 1);
    this.scene.background = new THREE.Color("#101713");
    this.scene.add(this.root);
    this.scene.add(new THREE.HemisphereLight(0xaed5c1, 0x151b18, 1.6));
    this.keyLight = new THREE.DirectionalLight(0xf4ffe8, 3.2);
    this.keyLight.position.set(2, -3, 5);
    this.keyLight.castShadow = true;
    this.keyLight.shadow.mapSize.set(SHADOW_MAP_HIGH, SHADOW_MAP_HIGH);
    this.scene.add(this.keyLight);
    this.scene.add(this.grid());
    this.attachControls();
    this.resize();
    this.resizeObserver = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => this.resize());
    this.resizeObserver?.observe(this.canvas);
    window.addEventListener("resize", () => this.resize());
  }

  load(sim: G1Model): void {
    this.clear();
    const model = sim.model as any;
    for (let id = 0; id < model.ngeom; id += 1) {
      const group = model.geom_group[id] as number;
      if (group !== 2 && id !== 0) continue;
      const geometry = this.makeGeometry(sim, id);
      if (!geometry) continue;
      const rgba = model.geom_rgba as Float32Array;
      const colorIndex = id * 4;
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(rgba[colorIndex] ?? 0.7, rgba[colorIndex + 1] ?? 0.7, rgba[colorIndex + 2] ?? 0.7),
        roughness: 0.48,
        metalness: 0.2,
        transparent: (rgba[colorIndex + 3] ?? 1) < 1,
        opacity: rgba[colorIndex + 3] ?? 1,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      this.geomMeshes[id] = mesh;
      this.root.add(mesh);
    }
  }

  sync(sim: G1Model): void {
    const data = sim.data as any;
    const xpos = data.geom_xpos as Float64Array;
    const xmat = data.geom_xmat as Float64Array;
    for (let id = 0; id < this.geomMeshes.length; id += 1) {
      const mesh = this.geomMeshes[id];
      if (!mesh) continue;
      const position = id * 3;
      const rotation = id * 9;
      mesh.matrix.set(
        xmat[rotation] ?? 1, xmat[rotation + 1] ?? 0, xmat[rotation + 2] ?? 0, xpos[position] ?? 0,
        xmat[rotation + 3] ?? 0, xmat[rotation + 4] ?? 1, xmat[rotation + 5] ?? 0, xpos[position + 1] ?? 0,
        xmat[rotation + 6] ?? 0, xmat[rotation + 7] ?? 0, xmat[rotation + 8] ?? 1, xpos[position + 2] ?? 0,
        0, 0, 0, 1,
      );
      mesh.matrixWorldNeedsUpdate = true;
    }
    this.target.set(data.xpos[sim.pelvisId * 3] ?? 0, data.xpos[sim.pelvisId * 3 + 1] ?? 0, 0.85);
    this.updateCamera();
  }

  render(): void {
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.clear();
    this.resizeObserver?.disconnect();
    this.renderer.dispose();
  }

  private clear(): void {
    for (const mesh of this.geomMeshes) {
      if (!mesh) continue;
      mesh.geometry.dispose();
      if (Array.isArray(mesh.material)) mesh.material.forEach((material) => material.dispose());
      else mesh.material.dispose();
      this.root.remove(mesh);
    }
    this.geomMeshes.length = 0;
    for (const geometry of this.geometryCache.values()) geometry.dispose();
    this.geometryCache.clear();
  }

  private makeGeometry(sim: G1Model, geomId: number): THREE.BufferGeometry | null {
    const model = sim.model as any;
    const type = model.geom_type[geomId] as number;
    const size = (model.geom_size as Float64Array).subarray(geomId * 3, geomId * 3 + 3);
    // MuJoCo and this scene are both Z-up. THREE.PlaneGeometry already lies
    // in the XY plane, so rotating it here would turn the floor into a wall.
    if (type === GEOM_PLANE) return new THREE.PlaneGeometry(12, 12);
    if (type === GEOM_SPHERE) return new THREE.SphereGeometry(size[0] ?? 0.05, 18, 12);
    if (type === GEOM_CAPSULE) return new THREE.CapsuleGeometry(size[0] ?? 0.03, (size[2] ?? 0.08) * 2, 8, 16).rotateX(Math.PI / 2);
    if (type === GEOM_CYLINDER) return new THREE.CylinderGeometry(size[0] ?? 0.03, size[0] ?? 0.03, (size[2] ?? 0.04) * 2, 18).rotateX(Math.PI / 2);
    if (type === GEOM_ELLIPSOID) return new THREE.SphereGeometry(1, 18, 12).scale(size[0] ?? 1, size[1] ?? 1, size[2] ?? 1);
    if (type === GEOM_BOX) return new THREE.BoxGeometry((size[0] ?? 0.1) * 2, (size[1] ?? 0.1) * 2, (size[2] ?? 0.1) * 2);
    if (type !== GEOM_MESH) return null;
    const meshId = model.geom_dataid[geomId] as number;
    if (meshId < 0) return null;
    const key = `${meshId}`;
    const cached = this.geometryCache.get(key);
    if (cached) return cached;
    const vertAdr = model.mesh_vertadr[meshId] as number;
    const vertNum = model.mesh_vertnum[meshId] as number;
    const faceAdr = model.mesh_faceadr[meshId] as number;
    const faceNum = model.mesh_facenum[meshId] as number;
    const vertices = (model.mesh_vert as Float32Array).subarray(vertAdr * 3, (vertAdr + vertNum) * 3);
    const faces = (model.mesh_face as Int32Array).subarray(faceAdr * 3, (faceAdr + faceNum) * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(vertices.slice(), 3));
    geometry.setIndex(Array.from(faces));
    geometry.computeVertexNormals();
    this.geometryCache.set(key, geometry);
    return geometry;
  }

  private grid(): THREE.GridHelper {
    const grid = new THREE.GridHelper(12, 24, 0x405849, 0x26372d);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.002;
    return grid;
  }

  private resize(): void {
    const width = Math.max(1, this.canvas.clientWidth || 800);
    const height = Math.max(1, this.canvas.clientHeight || 600);
    const rawDpr = window.devicePixelRatio || 1;
    // Prefer sharpness up to the pixel budget, then drop DPR instead of letting
    // a maximized 4K window render 8–16M fragments every frame.
    const budgetedDpr = Math.sqrt(MAX_DEVICE_PIXELS / (width * height));
    const pixelRatio = Math.min(rawDpr, MAX_PIXEL_RATIO, Math.max(1, budgetedDpr));
    this.renderer.setPixelRatio(pixelRatio);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();

    const drawPixels = width * height * pixelRatio * pixelRatio;
    const shadowSize = drawPixels > SHADOW_HIGH_BUDGET ? SHADOW_MAP_LOW : SHADOW_MAP_HIGH;
    if (shadowSize !== this.lastShadowSize) {
      this.lastShadowSize = shadowSize;
      this.keyLight.shadow.mapSize.set(shadowSize, shadowSize);
      // Force Three.js to rebuild the shadow map at the new resolution.
      this.keyLight.shadow.map?.dispose();
      this.keyLight.shadow.map = null;
    }
  }

  private updateCamera(): void {
    const { azimuth, elevation, distance } = this.orbit;
    this.camera.position.set(
      this.target.x + distance * Math.cos(elevation) * Math.cos(azimuth),
      this.target.y + distance * Math.cos(elevation) * Math.sin(azimuth),
      this.target.z + distance * Math.sin(elevation),
    );
    this.camera.lookAt(this.target);
  }

  private attachControls(): void {
    this.canvas.addEventListener("pointerdown", (event) => {
      this.dragStart = { x: event.clientX, y: event.clientY };
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (!this.dragStart) return;
      this.orbit.azimuth -= (event.clientX - this.dragStart.x) * 0.008;
      this.orbit.elevation = Math.max(-0.7, Math.min(1.0, this.orbit.elevation + (event.clientY - this.dragStart.y) * 0.006));
      this.dragStart = { x: event.clientX, y: event.clientY };
    });
    this.canvas.addEventListener("pointerup", () => { this.dragStart = null; });
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.orbit.distance = Math.max(1.2, Math.min(8, this.orbit.distance * (1 + event.deltaY * 0.001)));
    }, { passive: false });
  }
}
