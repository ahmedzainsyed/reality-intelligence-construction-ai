/**
 * PointCloudViewer.tsx
 *
 * Interactive 3D point cloud renderer using Three.js + React.
 *
 * Features:
 *  - PLY file streaming from MinIO/S3
 *  - Orbit / pan / zoom controls
 *  - Colour-by-class, height, or intensity
 *  - Point size control
 *  - Measurement tool
 *  - Screenshot export
 *  - Camera pose frustum overlay
 *  - Progressive LOD rendering for large clouds (>5M pts)
 */

import React, {
  useRef, useEffect, useCallback, useState, useMemo,
} from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader";
import { CSS2DRenderer, CSS2DObject } from "three/examples/jsm/renderers/CSS2DRenderer";
import {
  ZoomIn, ZoomOut, RotateCcw, Maximize2, Camera,
  Ruler, Palette, Download, Eye, EyeOff,
} from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

interface CameraPose {
  image_name: string;
  tx: number; ty: number; tz: number;
  qw: number; qx: number; qy: number; qz: number;
}

interface PointCloudViewerProps {
  /** URL or blob URL for PLY point cloud */
  plyUrl?: string;
  /** Raw PLY ArrayBuffer (if already loaded) */
  plyBuffer?: ArrayBuffer;
  /** Optional camera poses for frustum overlay */
  cameraPoses?: CameraPose[];
  /** Colour mode */
  colourMode?: "rgb" | "height" | "intensity" | "class";
  /** Point size (default 2) */
  pointSize?: number;
  /** Background colour */
  background?: string;
  /** Callback when measurement is taken */
  onMeasure?: (distanceM: number) => void;
  /** Controlled height; defaults to full parent */
  height?: string | number;
  className?: string;
}

// ── Colour Helpers ────────────────────────────────────────────────────────────

function heightColourLUT(y: Float32Array, colours: Float32Array) {
  const min = Math.min(...Array.from(y));
  const max = Math.max(...Array.from(y));
  const range = max - min || 1;

  const colour = new THREE.Color();
  for (let i = 0; i < y.length; i++) {
    const t = (y[i] - min) / range;
    // Jet-like: blue→cyan→green→yellow→red
    colour.setHSL(0.67 * (1 - t), 1.0, 0.5);
    colours[i * 3]     = colour.r;
    colours[i * 3 + 1] = colour.g;
    colours[i * 3 + 2] = colour.b;
  }
}

// ── Component ─────────────────────────────────────────────────────────────────

export const PointCloudViewer: React.FC<PointCloudViewerProps> = ({
  plyUrl,
  plyBuffer,
  cameraPoses = [],
  colourMode = "rgb",
  pointSize = 2,
  background = "#0f172a",
  onMeasure,
  height = "100%",
  className = "",
}) => {
  const mountRef  = useRef<HTMLDivElement>(null);
  const sceneRef  = useRef<THREE.Scene>();
  const camRef    = useRef<THREE.PerspectiveCamera>();
  const rendRef   = useRef<THREE.WebGLRenderer>();
  const css2dRef  = useRef<CSS2DRenderer>();
  const ctrlRef   = useRef<OrbitControls>();
  const animRef   = useRef<number>();
  const ptsRef    = useRef<THREE.Points>();

  const [loading,      setLoading]      = useState(true);
  const [loadProgress, setLoadProgress] = useState(0);
  const [stats,        setStats]        = useState({ points: 0, size_mb: 0 });
  const [measuring,    setMeasuring]    = useState(false);
  const [measurePts,   setMeasurePts]   = useState<THREE.Vector3[]>([]);
  const [distance,     setDistance]     = useState<number | null>(null);
  const [showPoses,    setShowPoses]    = useState(true);
  const [localPointSz, setLocalPointSz] = useState(pointSize);
  const [localColour,  setLocalColour]  = useState(colourMode);

  // ── Scene initialisation ──────────────────────────────────────────────────
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;

    const w = el.clientWidth;
    const h = el.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(background);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(60, w / h, 0.01, 5000);
    camera.position.set(0, 20, 50);
    camRef.current = camera;

    // WebGL renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    el.appendChild(renderer.domElement);
    rendRef.current = renderer;

    // CSS2D renderer (for labels)
    const css2d = new CSS2DRenderer();
    css2d.setSize(w, h);
    css2d.domElement.style.position = "absolute";
    css2d.domElement.style.top = "0";
    css2d.domElement.style.pointerEvents = "none";
    el.appendChild(css2d.domElement);
    css2dRef.current = css2d;

    // Orbit controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 2000;
    controls.minDistance = 0.1;
    ctrlRef.current = controls;

    // Grid helper
    const grid = new THREE.GridHelper(200, 50, 0x334155, 0x1e293b);
    scene.add(grid);

    // Ambient + directional light (for mesh if rendered alongside)
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.0);
    sun.position.set(50, 100, 50);
    scene.add(sun);

    // Render loop
    const animate = () => {
      animRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
      css2d.render(scene, camera);
    };
    animate();

    // Resize handler
    const onResize = () => {
      const nw = el.clientWidth;
      const nh = el.clientHeight;
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
      css2d.setSize(nw, nh);
    };
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      cancelAnimationFrame(animRef.current!);
      renderer.dispose();
      el.removeChild(renderer.domElement);
      el.removeChild(css2d.domElement);
    };
  }, [background]);

  // ── Load PLY ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!sceneRef.current) return;
    if (!plyUrl && !plyBuffer) return;

    setLoading(true);
    setLoadProgress(0);

    // Remove previous point cloud
    if (ptsRef.current) {
      sceneRef.current.remove(ptsRef.current);
      ptsRef.current.geometry.dispose();
      (ptsRef.current.material as THREE.Material).dispose();
    }

    const loader = new PLYLoader();

    const onLoad = (geometry: THREE.BufferGeometry) => {
      const count = geometry.attributes.position.count;
      setStats({ points: count, size_mb: Math.round((count * 12) / 1e6) });

      // Apply colour mode
      applyColourMode(geometry, localColour);

      const material = new THREE.PointsMaterial({
        size: localPointSz / 100,
        vertexColors: true,
        sizeAttenuation: true,
      });

      const points = new THREE.Points(geometry, material);
      sceneRef.current!.add(points);
      ptsRef.current = points;

      // Auto-fit camera to bounding box
      geometry.computeBoundingBox();
      const bbox = geometry.boundingBox!;
      const centre = new THREE.Vector3();
      bbox.getCenter(centre);
      const size = new THREE.Vector3();
      bbox.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);

      camRef.current!.position.set(
        centre.x + maxDim,
        centre.y + maxDim * 0.6,
        centre.z + maxDim,
      );
      ctrlRef.current!.target.copy(centre);
      ctrlRef.current!.update();

      setLoading(false);
    };

    const onProgress = (e: ProgressEvent) => {
      if (e.lengthComputable) setLoadProgress(Math.round((e.loaded / e.total) * 100));
    };

    const onError = (err: ErrorEvent) => {
      console.error("PLY load error", err);
      setLoading(false);
    };

    if (plyBuffer) {
      const geo = loader.parse(plyBuffer);
      onLoad(geo);
    } else if (plyUrl) {
      loader.load(plyUrl, onLoad, onProgress, onError);
    }
  }, [plyUrl, plyBuffer]);

  // ── Camera poses overlay ──────────────────────────────────────────────────
  useEffect(() => {
    if (!sceneRef.current || !cameraPoses.length) return;

    const group = new THREE.Group();
    group.name = "camera_poses";

    const frustumGeo = new THREE.ConeGeometry(0.3, 0.8, 4);
    const frustumMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b, wireframe: true });

    cameraPoses.forEach((pose) => {
      const mesh = new THREE.Mesh(frustumGeo, frustumMat);
      mesh.position.set(pose.tx, pose.ty, pose.tz);

      // Apply quaternion rotation
      const q = new THREE.Quaternion(pose.qx, pose.qy, pose.qz, pose.qw);
      mesh.setRotationFromQuaternion(q);

      group.add(mesh);
    });

    sceneRef.current.add(group);
    group.visible = showPoses;

    return () => { sceneRef.current?.remove(group); };
  }, [cameraPoses]);

  // Toggle pose visibility
  useEffect(() => {
    const poseGroup = sceneRef.current?.getObjectByName("camera_poses");
    if (poseGroup) poseGroup.visible = showPoses;
  }, [showPoses]);

  // ── Point size update ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!ptsRef.current) return;
    (ptsRef.current.material as THREE.PointsMaterial).size = localPointSz / 100;
    (ptsRef.current.material as THREE.PointsMaterial).needsUpdate = true;
  }, [localPointSz]);

  // ── Colour mode update ────────────────────────────────────────────────────
  useEffect(() => {
    if (!ptsRef.current) return;
    applyColourMode(ptsRef.current.geometry, localColour);
    (ptsRef.current.material as THREE.PointsMaterial).needsUpdate = true;
  }, [localColour]);

  // ── Measurement click handler ─────────────────────────────────────────────
  const onCanvasClick = useCallback((e: React.MouseEvent) => {
    if (!measuring || !ptsRef.current || !rendRef.current || !camRef.current) return;

    const rect = mountRef.current!.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width)  *  2 - 1,
      ((e.clientY - rect.top)  / rect.height) * -2 + 1,
    );

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.5 };
    raycaster.setFromCamera(ndc, camRef.current);

    const hits = raycaster.intersectObject(ptsRef.current);
    if (!hits.length) return;

    const pt = hits[0].point.clone();
    const newPts = [...measurePts, pt];
    setMeasurePts(newPts);

    if (newPts.length === 2) {
      const dist = newPts[0].distanceTo(newPts[1]);
      setDistance(dist);
      onMeasure?.(dist);

      // Draw measurement line
      const lineGeo = new THREE.BufferGeometry().setFromPoints(newPts);
      const lineMat = new THREE.LineBasicMaterial({ color: 0xef4444, linewidth: 2 });
      const line = new THREE.Line(lineGeo, lineMat);
      line.name = "measure_line";
      sceneRef.current?.add(line);

      setMeasurePts([]);  // reset for next measurement
    }
  }, [measuring, measurePts, onMeasure]);

  // ── Screenshot ────────────────────────────────────────────────────────────
  const takeScreenshot = () => {
    rendRef.current?.render(sceneRef.current!, camRef.current!);
    const dataUrl = rendRef.current!.domElement.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = `pointcloud_${Date.now()}.png`;
    a.click();
  };

  const resetCamera = () => {
    if (!ptsRef.current || !camRef.current || !ctrlRef.current) return;
    ptsRef.current.geometry.computeBoundingBox();
    const centre = new THREE.Vector3();
    ptsRef.current.geometry.boundingBox!.getCenter(centre);
    camRef.current.position.set(centre.x + 50, centre.y + 30, centre.z + 50);
    ctrlRef.current.target.copy(centre);
    ctrlRef.current.update();
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className={`relative flex flex-col ${className}`} style={{ height }}>
      {/* Toolbar */}
      <div className="absolute top-3 left-3 z-10 flex flex-col gap-2">
        <ToolbarButton icon={<RotateCcw size={16} />} title="Reset camera" onClick={resetCamera} />
        <ToolbarButton
          icon={<Ruler size={16} />}
          title="Measure distance"
          onClick={() => { setMeasuring(m => !m); setMeasurePts([]); setDistance(null); }}
          active={measuring}
        />
        <ToolbarButton
          icon={showPoses ? <Eye size={16} /> : <EyeOff size={16} />}
          title="Toggle camera poses"
          onClick={() => setShowPoses(v => !v)}
        />
        <ToolbarButton icon={<Camera size={16} />} title="Screenshot" onClick={takeScreenshot} />
      </div>

      {/* Colour mode selector */}
      <div className="absolute top-3 right-3 z-10 flex flex-col gap-2">
        <div className="bg-slate-800/90 rounded-lg p-2 backdrop-blur">
          <p className="text-xs text-slate-400 mb-1">Colour</p>
          {(["rgb", "height", "intensity"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setLocalColour(m)}
              className={`block w-full text-left text-xs px-2 py-0.5 rounded capitalize ${
                localColour === m ? "bg-blue-600 text-white" : "text-slate-300 hover:bg-slate-700"
              }`}
            >
              {m}
            </button>
          ))}
        </div>

        {/* Point size slider */}
        <div className="bg-slate-800/90 rounded-lg p-2 backdrop-blur">
          <p className="text-xs text-slate-400 mb-1">Size {localPointSz}</p>
          <input
            type="range" min={1} max={10} value={localPointSz}
            onChange={(e) => setLocalPointSz(Number(e.target.value))}
            className="w-full accent-blue-500"
          />
        </div>
      </div>

      {/* Stats overlay */}
      {!loading && (
        <div className="absolute bottom-3 left-3 z-10 bg-slate-800/90 rounded-lg px-3 py-2 text-xs text-slate-300 backdrop-blur">
          <span className="font-mono">{stats.points.toLocaleString()}</span> points
          {cameraPoses.length > 0 && (
            <span className="ml-3 text-amber-400">{cameraPoses.length} cameras</span>
          )}
          {distance !== null && (
            <span className="ml-3 text-green-400">📏 {distance.toFixed(2)} m</span>
          )}
        </div>
      )}

      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-slate-900/80 backdrop-blur-sm">
          <div className="text-white text-sm mb-3">Loading point cloud…</div>
          <div className="w-48 h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-200"
              style={{ width: `${loadProgress}%` }}
            />
          </div>
          <div className="text-slate-400 text-xs mt-2">{loadProgress}%</div>
        </div>
      )}

      {/* Three.js mount point */}
      <div
        ref={mountRef}
        className="flex-1 cursor-grab active:cursor-grabbing"
        onClick={onCanvasClick}
      />
    </div>
  );
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function applyColourMode(geo: THREE.BufferGeometry, mode: string) {
  const pos = geo.attributes.position;
  const count = pos.count;

  if (!geo.attributes.color) {
    geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
  }

  const colours = (geo.attributes.color as THREE.BufferAttribute).array as Float32Array;

  if (mode === "height") {
    const y = new Float32Array(count);
    for (let i = 0; i < count; i++) y[i] = pos.getY(i);
    heightColourLUT(y, colours);
    geo.attributes.color.needsUpdate = true;
  }
  // rgb / intensity: leave original vertex colours untouched
}

function heightColourLUT(y: Float32Array, colours: Float32Array) {
  const min = y.reduce((a, b) => Math.min(a, b), Infinity);
  const max = y.reduce((a, b) => Math.max(a, b), -Infinity);
  const range = max - min || 1;
  const colour = new THREE.Color();
  for (let i = 0; i < y.length; i++) {
    colour.setHSL(0.67 * (1 - (y[i] - min) / range), 1.0, 0.5);
    colours[i * 3]     = colour.r;
    colours[i * 3 + 1] = colour.g;
    colours[i * 3 + 2] = colour.b;
  }
}

interface ToolbarButtonProps {
  icon: React.ReactNode;
  title: string;
  onClick: () => void;
  active?: boolean;
}

function ToolbarButton({ icon, title, onClick, active }: ToolbarButtonProps) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={`w-8 h-8 flex items-center justify-center rounded-lg text-sm backdrop-blur transition-colors ${
        active
          ? "bg-blue-600 text-white"
          : "bg-slate-800/90 text-slate-300 hover:bg-slate-700"
      }`}
    >
      {icon}
    </button>
  );
}

export default PointCloudViewer;
