import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const loader = new STLLoader();

export class G1Preview {
  constructor(canvas, hooks = {}) {
    this.canvas = canvas;
    this.hooks = hooks;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#0b1111");
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 10000);
    this.camera.up.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.rotateSpeed = 1.25;
    this.controls.zoomSpeed = 1.15;
    this.controls.panSpeed = 0.9;
    this.controls.enableRotate = true;
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;
    this.controls.minDistance = 10;
    this.controls.maxDistance = 5000;
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.4));
    const key = new THREE.DirectionalLight(0xffffff, 1.8);
    key.position.set(120, -160, 220);
    this.scene.add(key);
    this.grid = new THREE.GridHelper(640, 32, 0x2a3325, 0x1b2118);
    this.grid.rotation.x = Math.PI / 2;
    const gridMaterials = Array.isArray(this.grid.material) ? this.grid.material : [this.grid.material];
    gridMaterials.forEach((material) => {
      material.transparent = true;
      material.opacity = 0.38;
      material.depthWrite = false;
    });
    this.scene.add(this.grid);
    this.state = { motion: null, bvh: null, frame: 0, playing: false, robotRoot: null, bodies: [], meshes: [], comparison: null };
    this.resize = this.resize.bind(this);
    this.tick = this.tick.bind(this);
    addEventListener("resize", this.resize);
    // A WebGL canvas starts with a small default drawing buffer. Resize it
    // before the first frame so CSS scaling cannot blur the entire preview.
    this.resize();
    requestAnimationFrame(this.tick);
  }

  async setMotion(motion, bvh = null) {
    this.clearObjects();
    this.state.motion = motion;
    this.state.bvh = bvh;
    const robotBounds = boundsFromFrames(motion.frames);
    const bvhBounds = bvh ? boundsFromBvh(bvh) : null;
    const separation = bvhBounds ? Math.max(140, (widthOf(robotBounds) + widthOf(bvhBounds)) / 2 + 60) : 0;
    this.state.robotOffset = separation / 2;
    this.state.bvhOffset = -separation / 2;
    this.createComparison(bvh);
    const meshReady = await this.createRobotMeshes(motion);
    if (!meshReady) this.createRobotPoints(motion);
    const merged = bvhBounds ? mergeBounds(offsetBounds(robotBounds, this.state.robotOffset), offsetBounds(bvhBounds, this.state.bvhOffset)) : offsetBounds(robotBounds, this.state.robotOffset);
    this.updateStage(merged);
    this.fitCamera(merged);
    this.setFrame(0);
    return meshReady;
  }

  async createRobotMeshes(motion) {
    if (!motion.mesh_geoms?.length || !motion.body_frames?.length || !motion.body_quats?.length) return false;
    const root = new THREE.Group();
    root.scale.setScalar(100);
    root.position.x = this.state.robotOffset;
    this.state.robotRoot = root;
    this.state.bodies = (motion.body_names || []).map((name) => {
      const body = new THREE.Group();
      body.name = name;
      root.add(body);
      return body;
    });
    try {
      const geometries = await Promise.all(motion.mesh_geoms.map((geom) => loadStl(geom.mesh_url)));
      motion.mesh_geoms.forEach((geom, index) => {
        const body = this.state.bodies[geom.body];
        if (!body || !geometries[index]) return;
        const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(...(geom.rgba || [0.65, 0.75, 0.72])), roughness: 0.5, metalness: 0.14, side: THREE.DoubleSide });
        const mesh = new THREE.Mesh(geometries[index], material);
        const geomQuat = new THREE.Quaternion().fromArray(geom.quat || [0, 0, 0, 1]);
        const meshQuat = new THREE.Quaternion().fromArray(geom.mesh_quat || [0, 0, 0, 1]).invert();
        const correction = new THREE.Vector3().fromArray(geom.mesh_pos || [0, 0, 0]).negate().applyQuaternion(meshQuat).applyQuaternion(geomQuat);
        mesh.position.fromArray(geom.pos || [0, 0, 0]).add(correction);
        mesh.quaternion.copy(geomQuat).multiply(meshQuat);
        body.add(mesh);
        this.state.meshes.push(mesh);
      });
      this.scene.add(root);
      return this.state.meshes.length > 0;
    } catch (cause) {
      console.warn("STL mesh preview unavailable; using points", cause);
      return false;
    }
  }

  createRobotPoints(motion) {
    const points = motion.frames[0]?.length || 0;
    const positions = new Float32Array(points * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const object = new THREE.Points(geometry, new THREE.PointsMaterial({ color: 0x70fff4, size: 9, sizeAttenuation: false }));
    object.position.x = this.state.robotOffset;
    this.state.robotPoints = object;
    this.scene.add(object);
  }

  createComparison(bvh) {
    if (!bvh) return;
    const links = bvh.joints.map((joint, index) => joint.parent >= 0 ? [joint.parent, index] : null).filter(Boolean);
    const lineGeometry = new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(links.length * 6), 3));
    const jointGeometry = new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(bvh.joints.length * 3), 3));
    this.state.comparison = { bvh, links, groundHeight: footHeight(bvh), lineGeometry, jointGeometry, lines: new THREE.LineSegments(lineGeometry, new THREE.LineBasicMaterial({ color: 0xffc56b })), joints: new THREE.Points(jointGeometry, new THREE.PointsMaterial({ color: 0xffd59a, size: 4, sizeAttenuation: false })) };
    this.state.comparison.lines.position.x = this.state.bvhOffset;
    this.state.comparison.joints.position.x = this.state.bvhOffset;
    this.scene.add(this.state.comparison.lines, this.state.comparison.joints);
  }

  setFrame(frame) {
    const count = this.state.motion?.frames?.length || 0;
    if (!count) return;
    this.state.frame = Math.max(0, Math.min(count - 1, Math.round(frame)));
    this.updateFrame();
    this.hooks.onFrame?.(this.state.frame);
  }

  setPlaying(value) {
    this.state.playing = Boolean(value && this.state.motion);
    this.state.lastTime = performance.now();
    this.hooks.onPlaying?.(this.state.playing);
  }

  updateFrame() {
    const index = this.state.frame;
    const motion = this.state.motion;
    const positions = motion?.frames?.[index];
    if (this.state.robotRoot && motion?.body_frames?.[index]) {
      motion.body_frames[index].forEach((pos, bodyIndex) => {
        const body = this.state.bodies[bodyIndex];
        const quat = motion.body_quats?.[index]?.[bodyIndex];
        if (body && pos) { body.position.fromArray(pos); if (quat) body.quaternion.fromArray(quat); }
      });
    } else if (this.state.robotPoints && positions) {
      const attr = this.state.robotPoints.geometry.attributes.position;
      positions.forEach((point, pointIndex) => attr.array.set(point, pointIndex * 3));
      attr.needsUpdate = true;
    }
    const comparison = this.state.comparison;
    if (comparison) {
      const pose = computePose(comparison.bvh, Math.min(comparison.bvh.frames.length - 1, index + Number(motion.comparison?.source_frame_offset || 0))).map((point) => [point[0], -point[2], point[1] - comparison.groundHeight]);
      const line = comparison.lineGeometry.attributes.position.array;
      comparison.links.forEach(([parent, child], linkIndex) => line.set([...pose[parent], ...pose[child]], linkIndex * 6));
      comparison.lineGeometry.attributes.position.needsUpdate = true;
      const joints = comparison.jointGeometry.attributes.position.array;
      pose.forEach((point, pointIndex) => joints.set(point, pointIndex * 3));
      comparison.jointGeometry.attributes.position.needsUpdate = true;
    }
  }

  fitCamera(bounds) {
    const center = new THREE.Vector3((bounds.minX + bounds.maxX) / 2, (bounds.minY + bounds.maxY) / 2, (bounds.minZ + bounds.maxZ) / 2);
    const radius = Math.max(100, Math.hypot(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, bounds.maxZ - bounds.minZ) * 0.58);
    this.controls.target.copy(center);
    this.camera.position.set(center.x + radius * 1.28, center.y - radius * 1.7, center.z + radius * 0.68);
    this.camera.near = Math.max(0.1, radius / 120);
    this.camera.far = Math.max(1000, radius * 18);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    if (this._width === width && this._height === height) return;
    this._width = width;
    this._height = height;
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  updateStage(bounds) {
    const center = new THREE.Vector3((bounds.minX + bounds.maxX) / 2, (bounds.minY + bounds.maxY) / 2, (bounds.minZ + bounds.maxZ) / 2);
    const radius = Math.max(20, Math.hypot(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, bounds.maxZ - bounds.minZ) * 0.55);
    this.grid.scale.setScalar(Math.max(320, radius * 3.6) / 640);
    this.grid.rotation.x = Math.PI / 2;
    this.grid.position.set(center.x, center.y, 0);
  }

  tick(now) {
    this.resize();
    if (this.state.playing && this.state.motion) {
      const step = 1000 / Number(this.state.motion.fps || 30);
      if (now - this.state.lastTime >= step) {
        this.state.frame = (this.state.frame + Math.floor((now - this.state.lastTime) / step)) % this.state.motion.frames.length;
        this.state.lastTime = now;
        this.updateFrame();
        this.hooks.onFrame?.(this.state.frame);
      }
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(this.tick);
  }

  clearObjects() {
    [this.state.robotRoot, this.state.robotPoints, this.state.comparison?.lines, this.state.comparison?.joints].filter(Boolean).forEach((object) => this.scene.remove(object));
    this.state.meshes.forEach((mesh) => { mesh.geometry.dispose(); mesh.material.dispose(); });
    this.state = { ...this.state, robotRoot: null, robotPoints: null, bodies: [], meshes: [], comparison: null };
  }

  dispose() { removeEventListener("resize", this.resize); this.renderer.dispose(); this.controls.dispose(); }
}

function loadStl(url) { return new Promise((resolve, reject) => loader.load(url, resolve, undefined, reject)); }

export function parseBvh(text) {
  const tokens = text.replace(/[{}]/g, " $& ").trim().split(/\s+/);
  const cursor = { index: 0, channelCursor: 0 };
  const joints = [];
  const expect = (value) => { const actual = tokens[cursor.index++]; if (actual !== value) throw new Error(`BVH: expected ${value}, got ${actual || "EOF"}`); };
  function parseJoint(parent, isEndSite = false) {
    const type = tokens[cursor.index++];
    let name;
    if (isEndSite) { expect("Site"); name = `${joints[parent].name}_end`; }
    else { if (type !== "ROOT" && type !== "JOINT") throw new Error(`BVH: expected joint, got ${type}`); name = tokens[cursor.index++]; }
    const joint = { name, parent, children: [], offset: [0, 0, 0], channels: [], channelOffset: cursor.channelCursor, isEndSite };
    const index = joints.push(joint) - 1;
    if (parent >= 0) joints[parent].children.push(index);
    expect("{");
    while (cursor.index < tokens.length) {
      const token = tokens[cursor.index++];
      if (token === "}") break;
      if (token === "OFFSET") joint.offset = [Number(tokens[cursor.index++]), Number(tokens[cursor.index++]), Number(tokens[cursor.index++])];
      else if (token === "CHANNELS") { const count = Number(tokens[cursor.index++]); joint.channelOffset = cursor.channelCursor; joint.channels = tokens.slice(cursor.index, cursor.index + count); cursor.index += count; cursor.channelCursor += count; }
      else if (token === "JOINT") { cursor.index--; parseJoint(index); }
      else if (token === "End") { cursor.index--; parseJoint(index, true); }
      else throw new Error(`BVH: unexpected token ${token}`);
    }
    return index;
  }
  expect("HIERARCHY");
  const rootIndex = parseJoint(-1);
  expect("MOTION"); expect("Frames:");
  const frameCount = Number(tokens[cursor.index++]);
  expect("Frame"); expect("Time:");
  const frameTime = Number(tokens[cursor.index++]);
  const frames = new Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    frames[frame] = new Array(cursor.channelCursor);
    for (let channel = 0; channel < cursor.channelCursor; channel += 1) frames[frame][channel] = Number(tokens[cursor.index++]) || 0;
  }
  if (!frameCount || !Number.isFinite(frameTime) || !cursor.channelCursor) throw new Error("BVH 没有有效的帧数据");
  return { joints, rootIndex, frames, frameTime, channelCount: cursor.channelCursor };
}

function computePose(bvh, frameIndex) {
  const values = bvh.frames[Math.max(0, Math.min(bvh.frames.length - 1, frameIndex))];
  const positions = new Array(bvh.joints.length);
  const rotations = new Array(bvh.joints.length);
  const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const multiply = (a, b) => [a[0] * b[0] + a[1] * b[3] + a[2] * b[6], a[0] * b[1] + a[1] * b[4] + a[2] * b[7], a[0] * b[2] + a[1] * b[5] + a[2] * b[8], a[3] * b[0] + a[4] * b[3] + a[5] * b[6], a[3] * b[1] + a[4] * b[4] + a[5] * b[7], a[3] * b[2] + a[4] * b[5] + a[5] * b[8], a[6] * b[0] + a[7] * b[3] + a[8] * b[6], a[6] * b[1] + a[7] * b[4] + a[8] * b[7], a[6] * b[2] + a[7] * b[5] + a[8] * b[8]];
  const rotate = (axis, degrees) => { const r = degrees * Math.PI / 180, c = Math.cos(r), s = Math.sin(r); if (axis === "X") return [1, 0, 0, 0, c, -s, 0, s, c]; if (axis === "Y") return [c, 0, s, 0, 1, 0, -s, 0, c]; return [c, -s, 0, s, c, 0, 0, 0, 1]; };
  const transform = (m, v) => [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]];
  bvh.joints.forEach((joint, index) => {
    const localPos = [...joint.offset]; let localRot = identity;
    joint.channels.forEach((channel, channelIndex) => { const value = values[joint.channelOffset + channelIndex] || 0; if (channel === "Xposition") localPos[0] += value; else if (channel === "Yposition") localPos[1] += value; else if (channel === "Zposition") localPos[2] += value; else if (channel.endsWith("rotation")) localRot = multiply(localRot, rotate(channel[0], value)); });
    if (joint.parent < 0) { positions[index] = localPos; rotations[index] = localRot; } else { const parentPos = positions[joint.parent]; const parentRot = rotations[joint.parent]; const offset = transform(parentRot, localPos); positions[index] = [parentPos[0] + offset[0], parentPos[1] + offset[1], parentPos[2] + offset[2]]; rotations[index] = multiply(parentRot, localRot); }
  });
  return positions;
}

function footHeight(bvh) { const indices = bvh.joints.map((joint, index) => /foot|toe/i.test(joint.name) && !joint.isEndSite ? index : null).filter((index) => index !== null); const values = []; for (let i = 0; i < bvh.frames.length; i += Math.max(1, Math.floor(bvh.frames.length / 80))) indices.forEach((index) => values.push(computePose(bvh, i)[index][1])); values.sort((a, b) => a - b); return values[Math.floor(values.length * 0.1)] || 0; }
function boundsFromFrames(frames) { return bounds(frames.flatMap((frame) => frame.filter(Boolean))); }
function boundsFromBvh(bvh) { return bounds(computePose(bvh, 0).map((point) => [point[0], -point[2], point[1] - footHeight(bvh)])); }
function bounds(points) { const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]), zs = points.map((p) => p[2]); return { minX: Math.min(...xs, -100), maxX: Math.max(...xs, 100), minY: Math.min(...ys, 0), maxY: Math.max(...ys, 200), minZ: Math.min(...zs, -100), maxZ: Math.max(...zs, 100) }; }
function widthOf(value) { return value.maxX - value.minX; }
function offsetBounds(value, x) { return { ...value, minX: value.minX + x, maxX: value.maxX + x }; }
function mergeBounds(a, b) { return { minX: Math.min(a.minX, b.minX), maxX: Math.max(a.maxX, b.maxX), minY: Math.min(a.minY, b.minY), maxY: Math.max(a.maxY, b.maxY), minZ: Math.min(a.minZ, b.minZ), maxZ: Math.max(a.maxZ, b.maxZ) }; }
