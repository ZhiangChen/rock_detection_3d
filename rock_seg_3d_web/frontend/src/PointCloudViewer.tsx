import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import type { PointCloudView, ViewerPayload } from "./api";

type Bounds = PointCloudView["bounds"];

type ScreenPoint = {
  x: number;
  y: number;
};

type ManualRemovalState = {
  active: boolean;
  drawing: boolean;
  polygon: ScreenPoint[];
  selectedIndices: number[];
  onAddVertex: (point: ScreenPoint) => void;
  onSelectionChange: (indices: number[]) => void;
};

type PointCloudViewerProps = {
  view: ViewerPayload | null;
  onPickPoint: (index: number) => void;
  onUnpickPoint: (index: number) => void;
  pickedIndices: number[];
  pickedColor?: number;
  pointSize: number;
  normalDisplayScale: number;
  highlightIndices?: number[];
  highlightColor?: [number, number, number];
  heatmapValues?: Array<number | null>;
  heatmapRange?: { min: number; max: number } | null;
  manualRemoval?: ManualRemovalState;
};

function boxFromBounds(bounds?: Bounds) {
  if (!bounds) {
    return null;
  }
  return new THREE.Box3(
    new THREE.Vector3(bounds.min[0], bounds.min[1], bounds.min[2]),
    new THREE.Vector3(bounds.max[0], bounds.max[1], bounds.max[2])
  );
}

function boundsKey(bounds?: Bounds) {
  if (!bounds) {
    return "";
  }
  return [...bounds.min, ...bounds.max].map((value) => Number(value).toPrecision(12)).join(",");
}

function frameBoundsForView(view: ViewerPayload) {
  return view.scene_bounds ?? view.bounds;
}

function activeBoundsForView(view: ViewerPayload) {
  return view.bounds ?? view.scene_bounds;
}

function fitCameraToBox(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  box: THREE.Box3
) {
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);

  const maxDim = Math.max(size.x, size.y, size.z, 1);
  const distance = maxDim / (2 * Math.tan((camera.fov * Math.PI) / 360));
  camera.position.set(center.x + distance, center.y - distance, center.z + distance * 0.75);
  camera.near = Math.max(distance / 100, 0.001);
  camera.far = distance * 100;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function defaultRotationQuaternion() {
  const matrix = new THREE.Matrix4()
    .makeRotationX(0.72)
    .multiply(new THREE.Matrix4().makeRotationY(0.62));
  return new THREE.Quaternion().setFromRotationMatrix(matrix);
}

function boxCenter(bounds?: Bounds) {
  const box = boxFromBounds(bounds);
  const center = new THREE.Vector3();
  if (box && !box.isEmpty()) {
    box.getCenter(center);
  }
  return center;
}

function makePointGeometry(
  view: PointCloudView,
  highlightIndices: number[] = [],
  highlightColor: [number, number, number] = [1.0, 0.84, 0.0],
  heatmapValues: Array<number | null> = [],
  heatmapRange: { min: number; max: number } | null = null
) {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(view.points.length * 3);
  const colors = new Float32Array(view.colors.length * 3);
  const hasNormals = Array.isArray(view.normals) && view.normals.length === view.points.length;
  const normals = new Float32Array(view.points.length * 3);
  const highlighted = new Set(highlightIndices.map(Number));
  const heatmapColor = (value: number | null | undefined): [number, number, number] | null => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || !heatmapRange || heatmapRange.max <= heatmapRange.min) {
      return null;
    }
    const stops: Array<[number, [number, number, number]]> = [
      [0.0, [0.17, 0.29, 0.85]],
      [0.25, [0.12, 0.61, 0.83]],
      [0.5, [0.19, 0.72, 0.44]],
      [0.75, [1.0, 0.83, 0.23]],
      [1.0, [0.84, 0.21, 0.16]]
    ];
    const t = THREE.MathUtils.clamp((numeric - heatmapRange.min) / (heatmapRange.max - heatmapRange.min), 0, 1);
    for (let stopIndex = 1; stopIndex < stops.length; stopIndex += 1) {
      if (t <= stops[stopIndex][0]) {
        const [leftT, leftColor] = stops[stopIndex - 1];
        const [rightT, rightColor] = stops[stopIndex];
        const localT = (t - leftT) / Math.max(rightT - leftT, 1e-9);
        return [
          leftColor[0] + (rightColor[0] - leftColor[0]) * localT,
          leftColor[1] + (rightColor[1] - leftColor[1]) * localT,
          leftColor[2] + (rightColor[2] - leftColor[2]) * localT
        ];
      }
    }
    return stops[stops.length - 1][1];
  };

  for (let i = 0; i < view.points.length; i += 1) {
    const point = view.points[i];
    const sourceIndex = view.indices[i];
    const color = highlighted.has(sourceIndex)
      ? highlightColor
      : heatmapColor(heatmapValues[sourceIndex]) ?? (view.colors[i] ?? [0.5, 0.5, 0.5]);
    const normal = hasNormals ? view.normals?.[i] ?? [0, 0, 1] : [0, 0, 1];
    positions[i * 3] = point[0];
    positions[i * 3 + 1] = point[1];
    positions[i * 3 + 2] = point[2];
    colors[i * 3] = color[0];
    colors[i * 3 + 1] = color[1];
    colors[i * 3 + 2] = color[2];
    const length = Math.hypot(normal[0], normal[1], normal[2]) || 1;
    normals[i * 3] = normal[0] / length;
    normals[i * 3 + 1] = normal[1] / length;
    normals[i * 3 + 2] = normal[2] / length;
  }

  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  if (hasNormals) {
    geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  }
  geometry.computeBoundingBox();
  return geometry;
}

function makePointMaterial(pointSize: number, hasNormals: boolean, viewportHeight: number) {
  if (!hasNormals) {
    return new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: true,
      sizeAttenuation: true
    });
  }

  return new THREE.ShaderMaterial({
    uniforms: {
      uPointSize: { value: pointSize },
      uViewportScale: { value: Math.max(1, viewportHeight) * 0.5 },
      uLightDirection: { value: new THREE.Vector3(0.28, -0.38, 0.88).normalize() }
    },
    vertexShader: `
      attribute vec3 color;
      varying vec3 vColor;
      varying vec3 vNormal;
      uniform float uPointSize;
      uniform float uViewportScale;
      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        gl_PointSize = clamp(uPointSize * uViewportScale / max(0.001, -mvPosition.z), 1.0, 96.0);
        vColor = color;
        vNormal = normalize(normalMatrix * normal);
      }
    `,
    fragmentShader: `
      precision mediump float;
      varying vec3 vColor;
      varying vec3 vNormal;
      uniform vec3 uLightDirection;
      void main() {
        vec2 uv = gl_PointCoord - vec2(0.5);
        float d = dot(uv, uv);
        if (d > 0.25) discard;
        vec3 normal = normalize(vNormal);
        float diffuse = abs(dot(normal, normalize(uLightDirection)));
        float specular = pow(diffuse, 28.0) * 0.24;
        vec3 color = vColor * (0.38 + diffuse * 0.68) + vec3(specular);
        gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
      }
    `
  });
}

function addMarkers(parent: THREE.Object3D, view: PointCloudView) {
  const group = new THREE.Group();
  group.name = "markers";

  view.markers.forEach((marker) => {
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.035, 16, 12),
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(marker.color[0], marker.color[1], marker.color[2]),
        roughness: 0.55,
        metalness: 0.05
      })
    );
    sphere.position.set(marker.point[0], marker.point[1], marker.point[2]);
    group.add(sphere);
  });

  parent.add(group);
}

function addPickedMarkers(parent: THREE.Object3D, view: PointCloudView, pickedIndices: number[], color = 0xffe14a) {
  if (!pickedIndices.length) {
    return;
  }
  const indexToPoint = new Map<number, [number, number, number]>();
  view.indices.forEach((sourceIndex, renderIndex) => {
    indexToPoint.set(sourceIndex, view.points[renderIndex]);
  });

  const group = new THREE.Group();
  group.name = "pending-selection-markers";
  pickedIndices.forEach((sourceIndex) => {
    const point = indexToPoint.get(sourceIndex);
    if (!point) {
      return;
    }
    const halo = new THREE.Mesh(
      new THREE.SphereGeometry(0.06, 18, 14),
      new THREE.MeshBasicMaterial({ color: 0x101010 })
    );
    halo.position.set(point[0], point[1], point[2]);
    const core = new THREE.Mesh(
      new THREE.SphereGeometry(0.04, 18, 14),
      new THREE.MeshBasicMaterial({ color })
    );
    core.position.copy(halo.position);
    group.add(halo);
    group.add(core);
  });
  parent.add(group);
}

function pointInPolygon(x: number, y: number, polygon: ScreenPoint[]) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    if ((yi > y) === (yj > y)) {
      continue;
    }
    const edgeX = ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (x < edgeX) {
      inside = !inside;
    }
  }
  return inside;
}

function addNormalSegments(parent: THREE.Object3D, view: PointCloudView, normalDisplayScale: number) {
  const segments = view.normal_segments ?? [];
  if (!segments.length) {
    return;
  }
  const positions = new Float32Array(segments.length * 6);
  segments.forEach((segment, index) => {
    const [start, end] = segment;
    const scaledEnd: [number, number, number] = [
      start[0] + (end[0] - start[0]) * normalDisplayScale,
      start[1] + (end[1] - start[1]) * normalDisplayScale,
      start[2] + (end[2] - start[2]) * normalDisplayScale
    ];
    positions[index * 6] = start[0];
    positions[index * 6 + 1] = start[1];
    positions[index * 6 + 2] = start[2];
    positions[index * 6 + 3] = scaledEnd[0];
    positions[index * 6 + 4] = scaledEnd[1];
    positions[index * 6 + 5] = scaledEnd[2];
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  parent.add(
    new THREE.LineSegments(
      geometry,
      new THREE.LineBasicMaterial({ color: 0x00f6ff, transparent: true, opacity: 0.95, depthTest: false })
    )
  );
}

function makeTriangleGeometry(
  vertices?: [number, number, number][],
  triangles?: [number, number, number][]
) {
  if (!vertices?.length || !triangles?.length) {
    return null;
  }
  const positions = new Float32Array(vertices.length * 3);
  vertices.forEach((vertex, index) => {
    positions[index * 3] = vertex[0];
    positions[index * 3 + 1] = vertex[1];
    positions[index * 3 + 2] = vertex[2];
  });
  const indices = new Uint32Array(triangles.length * 3);
  triangles.forEach((triangle, index) => {
    indices[index * 3] = triangle[0];
    indices[index * 3 + 1] = triangle[1];
    indices[index * 3 + 2] = triangle[2];
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  return geometry;
}

function makeMeshGeometry(view: ViewerPayload) {
  if (view.kind !== "mesh") {
    return null;
  }
  return makeTriangleGeometry(view.vertices, view.triangles);
}

function makeMeshMaterial(color: [number, number, number] = [0.5, 0.5, 0.5]) {
  return new THREE.MeshStandardMaterial({
    color: new THREE.Color(color[0], color[1], color[2]),
    roughness: 0.58,
    metalness: 0.04,
    side: THREE.DoubleSide
  });
}

function makeWireframeMaterial() {
  return new THREE.LineBasicMaterial({
    color: 0x2f3437,
    transparent: true,
    opacity: 0.32
  });
}

function analysisOverlayScale(view: ViewerPayload) {
  const bounds = frameBoundsForView(view);
  if (!bounds) {
    return 0.04;
  }
  const dx = bounds.max[0] - bounds.min[0];
  const dy = bounds.max[1] - bounds.min[1];
  const dz = bounds.max[2] - bounds.min[2];
  const diagonal = Math.max(Math.hypot(dx, dy, dz), 1);
  return Math.max(0.025, diagonal * 0.018);
}

function addAnalysisOverlays(parent: THREE.Object3D, view: ViewerPayload) {
  if (!view.analysis_segments?.length && !view.analysis_markers?.length) {
    return;
  }
  const markerRadius = analysisOverlayScale(view);
  (view.analysis_segments ?? []).forEach((segment) => {
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(segment.start[0], segment.start[1], segment.start[2]),
      new THREE.Vector3(segment.end[0], segment.end[1], segment.end[2])
    ]);
    const color = segment.color ?? [1, 0.82, 0];
    parent.add(
      new THREE.Line(
        geometry,
        new THREE.LineBasicMaterial({
          color: new THREE.Color(color[0], color[1], color[2]),
          depthTest: false,
          transparent: true,
          opacity: 0.96
        })
      )
    );
  });
  (view.analysis_markers ?? []).forEach((marker) => {
    const halo = new THREE.Mesh(
      new THREE.SphereGeometry(markerRadius * 1.45, 20, 14),
      new THREE.MeshBasicMaterial({ color: 0x101010, depthTest: false })
    );
    halo.position.set(marker.point[0], marker.point[1], marker.point[2]);
    const core = new THREE.Mesh(
      new THREE.SphereGeometry(markerRadius, 20, 14),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(marker.color[0], marker.color[1], marker.color[2]),
        depthTest: false
      })
    );
    core.position.copy(halo.position);
    parent.add(halo);
    parent.add(core);
  });
}

export function PointCloudViewer({
  view,
  onPickPoint,
  onUnpickPoint,
  pickedIndices,
  pickedColor,
  pointSize,
  normalDisplayScale,
  highlightIndices = [],
  highlightColor = [1.0, 0.84, 0.0],
  heatmapValues = [],
  heatmapRange = null,
  manualRemoval
}: PointCloudViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const onPickRef = useRef(onPickPoint);
  const onUnpickRef = useRef(onUnpickPoint);
  const pickedRef = useRef(pickedIndices);
  const rotationRef = useRef<THREE.Quaternion>(defaultRotationQuaternion());
  const cameraStateRef = useRef<{
    position: THREE.Vector3;
    target: THREE.Vector3;
    frameKey: string;
  } | null>(null);

  useEffect(() => {
    onPickRef.current = onPickPoint;
    onUnpickRef.current = onUnpickPoint;
    pickedRef.current = pickedIndices;
  }, [onPickPoint, onUnpickPoint, pickedIndices]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !view) {
      return undefined;
    }
    const activeContainer = container;
    const activeView = view;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf6f7f4);

    const camera = new THREE.PerspectiveCamera(55, 1, 0.001, 10000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(activeContainer.clientWidth, activeContainer.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const overlay = document.createElement("canvas");
    overlay.className = "manual-removal-overlay-react";
    activeContainer.replaceChildren(renderer.domElement, overlay);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = true;
    controls.enableRotate = false;
    controls.minDistance = 0;
    controls.maxDistance = Infinity;
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN
    };

    const frameBounds = frameBoundsForView(activeView);
    const frameKey = boundsKey(frameBounds);
    const activePivot = boxCenter(activeBoundsForView(activeView) ?? frameBounds);
    const contentGroup = new THREE.Group();
    const localGroup = new THREE.Group();
    contentGroup.position.copy(activePivot);
    contentGroup.quaternion.copy(rotationRef.current);
    localGroup.position.copy(activePivot).multiplyScalar(-1);
    contentGroup.add(localGroup);
    scene.add(contentGroup);

    function applyCameraFrame(box: THREE.Box3) {
      const previous = cameraStateRef.current;
      if (previous && previous.frameKey === frameKey) {
        camera.position.copy(previous.position);
        controls.target.copy(previous.target);
        controls.update();
        return;
      }
      fitCameraToBox(camera, controls, box);
    }

    const ambient = new THREE.HemisphereLight(0xffffff, 0xb8b8aa, 2.0);
    scene.add(ambient);
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(2, -3, 4);
    scene.add(key);

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.05 };
    const pointer = new THREE.Vector2();
    let pointsObject: THREE.Points | null = null;
    let pointMaterial: THREE.Material | null = null;
    let pointerStart: { x: number; y: number; button: number; mode: "shift" | "manual" | "rotate" } | null = null;
    let trackballLast: THREE.Vector3 | null = null;

    if (activeView.kind === "pointCloud") {
      const geometry = makePointGeometry(activeView, highlightIndices, highlightColor, heatmapValues, heatmapRange);
      const material = makePointMaterial(pointSize, Boolean(geometry.getAttribute("normal")), renderer.domElement.height);
      pointMaterial = material;
      pointsObject = new THREE.Points(geometry, material);
      localGroup.add(pointsObject);
      addMarkers(localGroup, activeView);

      const box = boxFromBounds(frameBounds) ?? geometry.boundingBox?.clone() ?? new THREE.Box3();
      if (box.isEmpty()) {
        box.setFromArray(Array.from(geometry.getAttribute("position").array));
      }
      addPickedMarkers(localGroup, activeView, pickedIndices, pickedColor);
      if (manualRemoval?.active) {
        addPickedMarkers(localGroup, activeView, manualRemoval.selectedIndices, 0xffd43b);
      }
      addNormalSegments(localGroup, activeView, normalDisplayScale);
      addAnalysisOverlays(localGroup, activeView);
      applyCameraFrame(box);
    } else if (activeView.kind === "combinedMesh") {
      const loader = new PLYLoader();
      activeView.components.forEach((component) => {
        if (component.kind === "pointCloud") {
          const componentView: PointCloudView = {
            kind: "pointCloud",
            points: component.points,
            colors: component.colors,
            normals: component.normals,
            indices: component.indices,
            markers: [],
            bounds: activeView.bounds,
            scene_bounds: activeView.scene_bounds,
            total_points: component.point_count,
            rendered_points: component.point_count
          };
          const geometry = makePointGeometry(componentView);
          localGroup.add(
            new THREE.Points(
              geometry,
              makePointMaterial(pointSize, Boolean(geometry.getAttribute("normal")), renderer.domElement.height)
            )
          );
          return;
        }
        const addMeshGeometry = (geometry: THREE.BufferGeometry) => {
          geometry.computeVertexNormals();
          const mesh = new THREE.Mesh(
            geometry,
            makeMeshMaterial(component.color)
          );
          localGroup.add(mesh);
          if (component.show_wireframe) {
            localGroup.add(
              new THREE.LineSegments(
                new THREE.WireframeGeometry(geometry),
                makeWireframeMaterial()
              )
            );
          }
        };
        const payloadGeometry = makeTriangleGeometry(component.vertices, component.triangles);
        if (payloadGeometry) {
          addMeshGeometry(payloadGeometry);
        } else if (component.url) {
          loader.load(
            `${component.url}?t=${Date.now()}`,
            addMeshGeometry,
            undefined,
            (error) => {
              console.error("Failed to load combined mesh component", error);
            }
          );
        }
      });
      const box = boxFromBounds(frameBounds) ?? new THREE.Box3();
      applyCameraFrame(box);
    } else {
      const payloadGeometry = makeMeshGeometry(activeView);
      if (payloadGeometry) {
        const mesh = new THREE.Mesh(
          payloadGeometry,
          makeMeshMaterial()
        );
        localGroup.add(mesh);
        if (activeView.show_wireframe) {
          localGroup.add(
            new THREE.LineSegments(
              new THREE.WireframeGeometry(payloadGeometry),
              makeWireframeMaterial()
            )
          );
        }
        addAnalysisOverlays(localGroup, activeView);
        const box = boxFromBounds(frameBounds) ?? payloadGeometry.boundingBox?.clone() ?? new THREE.Box3().setFromObject(mesh);
        applyCameraFrame(box);
      } else {
        const loader = new PLYLoader();
        loader.load(
          `${activeView.url}?t=${Date.now()}`,
          (geometry) => {
            geometry.computeVertexNormals();
            const mesh = new THREE.Mesh(
              geometry,
              makeMeshMaterial()
            );
            localGroup.add(mesh);
            if (activeView.show_wireframe) {
              const wireframe = new THREE.LineSegments(
                new THREE.WireframeGeometry(geometry),
                makeWireframeMaterial()
              );
              localGroup.add(wireframe);
            }
            addAnalysisOverlays(localGroup, activeView);
            const box = boxFromBounds(frameBounds) ?? new THREE.Box3().setFromObject(mesh);
            applyCameraFrame(box);
          },
          undefined,
          (error) => {
            console.error("Failed to load mesh", error);
          }
        );
      }
    }

    function resize() {
      const width = Math.max(1, activeContainer.clientWidth);
      const height = Math.max(1, activeContainer.clientHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      if (pointMaterial instanceof THREE.ShaderMaterial) {
        pointMaterial.uniforms.uViewportScale.value = Math.max(1, renderer.domElement.height) * 0.5;
      }
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      overlay.width = Math.max(1, Math.floor(width * dpr));
      overlay.height = Math.max(1, Math.floor(height * dpr));
      overlay.style.width = `${width}px`;
      overlay.style.height = `${height}px`;
      drawManualOverlay();
    }

    function drawManualOverlay() {
      const width = Math.max(1, activeContainer.clientWidth);
      const height = Math.max(1, activeContainer.clientHeight);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const ctx = overlay.getContext("2d");
      if (!ctx) {
        return;
      }
      if (overlay.width !== Math.floor(width * dpr) || overlay.height !== Math.floor(height * dpr)) {
        overlay.width = Math.max(1, Math.floor(width * dpr));
        overlay.height = Math.max(1, Math.floor(height * dpr));
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      if (!manualRemoval?.active || !manualRemoval.polygon.length) {
        return;
      }
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#ffdc3d";
      ctx.fillStyle = "rgba(255, 220, 61, 0.16)";
      ctx.beginPath();
      ctx.moveTo(manualRemoval.polygon[0].x, manualRemoval.polygon[0].y);
      manualRemoval.polygon.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
      if (manualRemoval.polygon.length >= 3) {
        ctx.closePath();
        ctx.fill();
      }
      ctx.stroke();
      ctx.fillStyle = "#101820";
      manualRemoval.polygon.forEach((point) => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
    }

    function trackballCenter() {
      const bounds = renderer.domElement.getBoundingClientRect();
      const projected = activePivot.clone().project(camera);
      if (!Number.isFinite(projected.x) || !Number.isFinite(projected.y)) {
        return new THREE.Vector2(bounds.width / 2, bounds.height / 2);
      }
      return new THREE.Vector2(
        THREE.MathUtils.clamp((projected.x * 0.5 + 0.5) * bounds.width, bounds.width * 0.25, bounds.width * 0.75),
        THREE.MathUtils.clamp((-projected.y * 0.5 + 0.5) * bounds.height, bounds.height * 0.25, bounds.height * 0.75)
      );
    }

    function virtualTrackballVector(event: PointerEvent) {
      const bounds = renderer.domElement.getBoundingClientRect();
      const center = trackballCenter();
      const radius = Math.max(Math.min(bounds.width, bounds.height) * 0.48, 1);
      const x = (event.clientX - bounds.left - center.x) / radius;
      const y = (center.y - (event.clientY - bounds.top)) / radius;
      const distanceSq = x * x + y * y;
      const vector = distanceSq <= 1
        ? new THREE.Vector3(x, y, Math.sqrt(1 - distanceSq))
        : new THREE.Vector3(x, y, 0);
      return vector.normalize();
    }

    function applyTrackballRotation(event: PointerEvent) {
      const current = virtualTrackballVector(event);
      if (trackballLast) {
        const deltaCamera = new THREE.Quaternion().setFromUnitVectors(trackballLast, current);
        const cameraSpace = camera.quaternion.clone();
        const deltaWorld = cameraSpace.clone().multiply(deltaCamera).multiply(cameraSpace.clone().invert());
        rotationRef.current.premultiply(deltaWorld).normalize();
        contentGroup.quaternion.copy(rotationRef.current);
      }
      trackballLast = current;
    }

    function pickAtPointer(event: PointerEvent) {
      if (!pointsObject || activeView.kind !== "pointCloud") {
        return;
      }
      const bounds = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObject(pointsObject);
      const first = hits[0];
      if (first?.index === undefined) {
        return;
      }
      const sourceIndex = activeView.indices[first.index];
      if (sourceIndex !== undefined) {
        onPickRef.current(sourceIndex);
      }
    }

    function selectedPointAtPointer(event: PointerEvent) {
      if (!pointsObject || activeView.kind !== "pointCloud" || !pickedRef.current.length) {
        return null;
      }
      const selected = new Set(pickedRef.current);
      const bounds = renderer.domElement.getBoundingClientRect();
      const x = event.clientX - bounds.left;
      const y = event.clientY - bounds.top;
      const positions = pointsObject.geometry.getAttribute("position") as THREE.BufferAttribute;
      const projected = new THREE.Vector3();
      const pickRadius = 18;
      const pickRadiusSq = pickRadius * pickRadius;
      let bestIndex: number | null = null;
      let bestDepth = Infinity;
      let bestDist = Infinity;

      for (let renderIndex = 0; renderIndex < activeView.indices.length; renderIndex += 1) {
        const sourceIndex = activeView.indices[renderIndex];
        if (!selected.has(sourceIndex)) {
          continue;
        }
        projected.fromBufferAttribute(positions, renderIndex);
        pointsObject.localToWorld(projected);
        projected.project(camera);
        if (
          projected.z < -1 ||
          projected.z > 1 ||
          projected.x < -1 ||
          projected.x > 1 ||
          projected.y < -1 ||
          projected.y > 1
        ) {
          continue;
        }
        const screenX = (projected.x * 0.5 + 0.5) * bounds.width;
        const screenY = (-projected.y * 0.5 + 0.5) * bounds.height;
        const dx = screenX - x;
        const dy = screenY - y;
        const dist = dx * dx + dy * dy;
        if (dist > pickRadiusSq) {
          continue;
        }
        if (dist < bestDist - 0.001 || (Math.abs(dist - bestDist) <= 0.001 && projected.z < bestDepth)) {
          bestDist = dist;
          bestDepth = projected.z;
          bestIndex = sourceIndex;
        }
      }

      return bestIndex;
    }

    function updateManualRemovalSelection(polygon: ScreenPoint[]) {
      if (!manualRemoval?.active || !pointsObject || activeView.kind !== "pointCloud" || polygon.length < 3) {
        if (manualRemoval?.selectedIndices.length) {
          manualRemoval.onSelectionChange([]);
        }
        return;
      }
      const bounds = renderer.domElement.getBoundingClientRect();
      const positions = pointsObject.geometry.getAttribute("position") as THREE.BufferAttribute;
      const projected = new THREE.Vector3();
      const selected = new Set<number>();
      const totalPointCount = Number.isFinite(activeView.total_points) ? Number(activeView.total_points) : Infinity;
      for (let renderIndex = 0; renderIndex < activeView.indices.length; renderIndex += 1) {
        projected.fromBufferAttribute(positions, renderIndex);
        pointsObject.localToWorld(projected);
        projected.project(camera);
        if (
          projected.z < -1 ||
          projected.z > 1 ||
          projected.x < -1 ||
          projected.x > 1 ||
          projected.y < -1 ||
          projected.y > 1
        ) {
          continue;
        }
        const sourceIndex = activeView.indices[renderIndex];
        if (sourceIndex === undefined || sourceIndex < 0 || sourceIndex >= totalPointCount) {
          continue;
        }
        const screenX = (projected.x * 0.5 + 0.5) * bounds.width;
        const screenY = (-projected.y * 0.5 + 0.5) * bounds.height;
        if (pointInPolygon(screenX, screenY, polygon)) {
          selected.add(sourceIndex);
        }
      }
      const next = Array.from(selected).sort((a, b) => a - b);
      const previous = manualRemoval.selectedIndices;
      if (next.length === previous.length && next.every((value, index) => value === previous[index])) {
        return;
      }
      manualRemoval.onSelectionChange(next);
    }

    function stopShiftPickEvent(event: PointerEvent) {
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
    }

    function handlePointerDown(event: PointerEvent) {
      if (manualRemoval?.active && manualRemoval.drawing && event.button === 0) {
        stopShiftPickEvent(event);
        controls.enabled = false;
        pointerStart = { x: event.clientX, y: event.clientY, button: event.button, mode: "manual" };
        renderer.domElement.setPointerCapture(event.pointerId);
        return;
      }
      if (!event.shiftKey || (event.button !== 0 && event.button !== 2)) {
        if (event.button === 0) {
          stopShiftPickEvent(event);
          controls.enabled = false;
          pointerStart = { x: event.clientX, y: event.clientY, button: event.button, mode: "rotate" };
          trackballLast = virtualTrackballVector(event);
          renderer.domElement.setPointerCapture(event.pointerId);
          return;
        }
        pointerStart = null;
        return;
      }
      stopShiftPickEvent(event);
      pointerStart = { x: event.clientX, y: event.clientY, button: event.button, mode: "shift" };
      renderer.domElement.setPointerCapture(event.pointerId);
    }

    function handlePointerMove(event: PointerEvent) {
      if (!pointerStart || pointerStart.mode !== "rotate") {
        return;
      }
      stopShiftPickEvent(event);
      applyTrackballRotation(event);
    }

    function handlePointerUp(event: PointerEvent) {
      if (!pointerStart) {
        return;
      }
      stopShiftPickEvent(event);
      const start = pointerStart;
      pointerStart = null;
      if (renderer.domElement.hasPointerCapture(event.pointerId)) {
        renderer.domElement.releasePointerCapture(event.pointerId);
      }
      if (start.mode === "rotate") {
        controls.enabled = true;
        trackballLast = null;
        return;
      }
      const dx = event.clientX - start.x;
      const dy = event.clientY - start.y;
      if (dx * dx + dy * dy > 25) {
        controls.enabled = true;
        return;
      }
      if (start.mode === "manual") {
        controls.enabled = true;
        const bounds = renderer.domElement.getBoundingClientRect();
        const nextPolygon = [
          ...(manualRemoval?.polygon ?? []),
          {
            x: event.clientX - bounds.left,
            y: event.clientY - bounds.top
          }
        ];
        manualRemoval?.onAddVertex(nextPolygon[nextPolygon.length - 1]);
        updateManualRemovalSelection(nextPolygon);
      } else if (start.button === 0) {
        pickAtPointer(event);
      } else if (start.button === 2) {
        const sourceIndex = selectedPointAtPointer(event);
        if (sourceIndex !== null) {
          onUnpickRef.current(sourceIndex);
        }
      }
    }

    function handlePointerCancel(event: PointerEvent) {
      if (renderer.domElement.hasPointerCapture(event.pointerId)) {
        renderer.domElement.releasePointerCapture(event.pointerId);
      }
      controls.enabled = true;
      pointerStart = null;
      trackballLast = null;
    }

    function handleContextMenu(event: MouseEvent) {
      event.preventDefault();
    }

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(activeContainer);
    drawManualOverlay();
    if (manualRemoval?.active) {
      updateManualRemovalSelection(manualRemoval.polygon);
    }
    renderer.domElement.addEventListener("contextmenu", handleContextMenu);
    renderer.domElement.addEventListener("pointerdown", handlePointerDown, true);
    renderer.domElement.addEventListener("pointermove", handlePointerMove, true);
    renderer.domElement.addEventListener("pointerup", handlePointerUp, true);
    renderer.domElement.addEventListener("pointercancel", handlePointerCancel, true);

    let frame = 0;
    function animate() {
      frame = window.requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    resize();
    animate();

    return () => {
      window.cancelAnimationFrame(frame);
      cameraStateRef.current = {
        position: camera.position.clone(),
        target: controls.target.clone(),
        frameKey
      };
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("contextmenu", handleContextMenu);
      renderer.domElement.removeEventListener("pointerdown", handlePointerDown, true);
      renderer.domElement.removeEventListener("pointermove", handlePointerMove, true);
      renderer.domElement.removeEventListener("pointerup", handlePointerUp, true);
      renderer.domElement.removeEventListener("pointercancel", handlePointerCancel, true);
      controls.dispose();
      renderer.dispose();
      scene.traverse((object) => {
        const mesh = object as THREE.Mesh;
        mesh.geometry?.dispose?.();
        const material = mesh.material;
        if (Array.isArray(material)) {
          material.forEach((item) => item.dispose());
        } else {
          material?.dispose?.();
        }
      });
    };
  }, [
    view,
    pointSize,
    pickedIndices,
    pickedColor,
    highlightIndices,
    highlightColor,
    heatmapValues,
    heatmapRange,
    normalDisplayScale,
    manualRemoval?.active,
    manualRemoval?.drawing,
    manualRemoval?.polygon,
    manualRemoval?.selectedIndices
  ]);

  return <div ref={containerRef} className="viewer-surface" />;
}
