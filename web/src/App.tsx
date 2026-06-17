import {
  type ButtonHTMLAttributes,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import {
  clearInterfaceDraft as clearInterfaceDraftApi,
  createSession,
  createInterfaceDraftFromSource,
  commitInterfaceDraft,
  downloadUrl,
  exportProject,
  getInterfaceDraft,
  getJob,
  getSession,
  getViewer,
  importProject,
  runJob,
  undoInterfaceDraft,
  uploadPointCloud,
  type DenoiseParams,
  type InterfaceDraft,
  type JobResponse,
  type MeshTarget,
  type ProjectUiState,
  type SegmentParams,
  type SessionSummary,
  type ViewerPayload
} from "./api";
import { PointCloudViewer } from "./PointCloudViewer";

type PickMode = "rock" | "pedestal" | "interface";
type ViewName = "raw" | "seeds" | "interface" | "voxel_segmented" | "segmented" | "mesh_prepared" | "mesh" | "combined_mesh" | "analysis";

type InterfacePartDraft = {
  selected_indices: number[];
  is_lateral: boolean;
};

type ScreenPoint = {
  x: number;
  y: number;
};

type HagVegetationParams = {
  grid_size: number;
  height_threshold: number;
  ground_percentile: number;
  min_points_per_cell: number;
};

type RoughnessParams = {
  radius: number;
  threshold: number;
};

type RoughnessStats = {
  min_roughness?: number;
  max_roughness?: number;
  mean_roughness?: number;
  valid_roughness_count?: number;
  voxel_size?: number;
  voxel_point_count?: number;
};

const PICKED_MARKER_COLORS: Record<PickMode, number> = {
  rock: 0xff0d05,
  pedestal: 0x003dff,
  interface: 0x00ff00
};

type InterfaceMetadataPart = {
  num_points?: number;
  point_indices?: number[];
  dense_points?: unknown[];
  selected_indices?: number[];
};

type InterfaceMetadata = {
  parts?: InterfaceMetadataPart[];
  close_loop?: boolean;
};

function interfaceDraftSegmentInfo(draft: InterfaceDraft | null) {
  if (!draft) {
    return {
      pointsText: "No segments"
    };
  }
  const metadata = (draft.metadata && typeof draft.metadata === "object" ? draft.metadata : {}) as InterfaceMetadata;
  const metadataParts = Array.isArray(metadata.parts) ? metadata.parts : [];
  const fallbackParts = Array.isArray(draft.parts) ? draft.parts : [];
  const parts: InterfaceMetadataPart[] = metadataParts.length
    ? metadataParts
    : fallbackParts.map((part) => ({ selected_indices: part.selected_indices }));
  const pointCounts = parts.map((part) => {
    if (Number.isFinite(part.num_points)) {
      return Number(part.num_points);
    }
    if (Array.isArray(part.point_indices)) {
      return part.point_indices.length;
    }
    if (Array.isArray(part.dense_points)) {
      return part.dense_points.length;
    }
    if (Array.isArray(part.selected_indices)) {
      return part.selected_indices.length;
    }
    return 0;
  });
  const pointsText = pointCounts.length
    ? pointCounts.map((count, index) => `S${index + 1}: ${count.toLocaleString()} pts`).join(" | ")
    : "No segments";
  return {
    pointsText: `Points per segment: ${pointsText}`
  };
}

const defaultSegmentParams: SegmentParams = {
  smoothness_threshold: 0.9,
  curvature_threshold: 0.1,
  basal_proximity_threshold: 0.05,
  voxel_size: 0.02,
  neighbor_count: 50,
  distance_threshold: 0.05,
  label_propagation_distance: 0.05
};

const defaultDenoiseParams: DenoiseParams = {
  method: "sor",
  sor_neighbors: 10,
  sor_std_ratio: 2.0,
  dbscan_eps: 0.02,
  dbscan_min_points: 20
};

const defaultHagVegetationParams: HagVegetationParams = {
  grid_size: 0.05,
  height_threshold: 0.08,
  ground_percentile: 10,
  min_points_per_cell: 3
};

const defaultRoughnessParams: RoughnessParams = {
  radius: 0.05,
  threshold: 0.01
};

const helpText = {
  pointSize:
    "Changes only the rendered point size. Increase it for sparse or distant clouds and decrease it when dense clouds look blotchy; segmentation and exports are unchanged.",
  lateral:
    "Use this when selected interface points trace contact with an adjacent rock or side support instead of the basal pedestal. Basal and lateral parts are handled separately; basal labels feed the contact-geometry analysis.",
  closeLoop:
    "Keep this on when your interface points outline a closed contact boundary. Turn it off for an open contact edge or when staging separate basal and lateral interface parts.",
  smoothness:
    "Normal-alignment threshold for accepting a neighboring point. Lower values are more permissive and can overspill across the interface; higher values are stricter and can leave rough rock unlabeled, which later propagation may fill incorrectly.",
  curvature:
    "Local curvature threshold for accepting a neighboring point. Larger values are more permissive and can overspill across sharp interface changes; smaller values are stricter and can fragment rough surfaces or leave gaps near the contact.",
  proximity:
    "Used by Run ICRG with a saved manual interface. Points within this radius of the interface are excluded from growth. Increase if labels cross the contact; decrease if too much rock near the interface stays unlabeled.",
  voxel:
    "Default 0.02 m in the manuscript for faster preprocessing. Smaller values preserve contact detail but run slower; larger values smooth noisy data but can erase small interface features.",
  neighbors:
    "Maximum neighbors for normal estimation used by the smoothness and curvature tests. Increase for sparse or noisy scans to stabilize normals; decrease to preserve sharper contact detail.",
  distance:
    "Maximum radius for region-growing neighbor search. Default is 0.05 m. Increase when the cloud is sparse or growth stalls across small gaps. Decrease to avoid jumps across narrow contacts or nearby supports.",
  labelPropagation:
    "Radius for distance-weighted completion after voxel region growing. Default is 0.05 m. Increase to fill larger unlabeled gaps; decrease to reduce leakage across contacts.",
  normalMethod:
    "PyMeshLab is preferred in the supplement because it estimates globally consistent normals for Poisson reconstruction. Use Open3D if PyMeshLab fails or the prepared mesh looks unstable.",
  normalK:
    "Default 200 in the supplement for PCA normal estimation. Higher values smooth normals and help noisy surfaces; lower values preserve detail but can create unstable or flipped normals.",
  depth:
    "Controls Poisson reconstruction resolution. Higher depth captures more detail but takes longer and may amplify noise; lower depth is faster and smoother for sparse data."
};

const viewHelp: Record<ViewName, string> = {
  raw: "Show the uploaded point cloud with its original colors before seed or interface edits.",
  seeds: "Show the seed-selection state, including saved rock and support seed markers.",
  interface: "Show the previewed or saved interface constraints near the contact.",
  voxel_segmented: "Show the immediate region-growing or ICRG labels on the voxelized point cloud before dense label propagation.",
  segmented: "Show the latest rock/support labels after running region growing or ICRG.",
  mesh_prepared: "Show the prepared point set used for normal estimation and mesh reconstruction.",
  mesh: "Show mesh status after reconstruction; download the PLY from the Downloads panel.",
  combined_mesh: "Load rock and pedestal together. Existing meshes are used first; missing meshes fall back to segmented point clouds.",
  analysis: "Show the reconstructed mesh with center of mass, Z axis, alpha point, and analysis metrics."
};

const viewLabels: Record<ViewName, string> = {
  raw: "Raw",
  seeds: "Seeds",
  interface: "Interface",
  voxel_segmented: "RG Result",
  segmented: "Segmented",
  mesh_prepared: "Mesh Prep",
  mesh: "Mesh",
  combined_mesh: "Rock + Pedestal",
  analysis: "Analysis"
};

const buttonHelp = {
  importProject: "Load a saved .rd3dproj archive and restore its point cloud, parameters, workflow state, and available outputs.",
  saveProject: "Overwrite the imported or previously chosen .rd3dproj file when browser file access is available. Otherwise, choose a save target first.",
  saveAsProject: "Choose a .rd3dproj file target for this project. Future Save Project clicks overwrite that chosen file when supported.",
  pickRock: "Shift + left click adds rock seed points. Shift + right click near a selected point removes it.",
  pickPedestal: "Shift + left click adds support or pedestal seed points. Shift + right click near a selected point removes it.",
  pickInterface: "Shift + left click adds interface contact points. Shift + right click near a selected point removes it.",
  autoSeeds: "Choose default rock and support seeds from the current point cloud geometry.",
  clearMode: "Clear only the current pick mode: rock seeds, pedestal seeds, or current interface points.",
  stagePart: "Store the current interface picks as one contact segment so you can pick another segment.",
  interpolateInterface: "Preview the dense interface path before saving it for segmentation.",
  saveInterface: "Finalize the interpolated interface constraints for region growing.",
  undoDraft: "Undo the most recent draft edit.",
  clearDraft: "Discard the editable draft without changing the saved manual interface.",
  saveDraftManual: "Commit this refined draft as the manual interface used by Run ICRG.",
  clearParts: "Remove staged interface parts and current interface picks.",
  runSegment: "Segment from seeds without using interface constraints.",
  runICRG: "Run interface-constrained region growing using the saved manual interface.",
  prepareMesh: "Open the existing prepared target if available; otherwise prepare it from the current label-propagated segmentation.",
  resetMeshPreparation: "Preview a fresh prepared target from label propagation. It is not saved until denoise or manual removal commits it.",
  removeNoise: "Run the selected denoise method: SOR, DBSCAN, or SOR followed by DBSCAN.",
  undoNoise: "Restore the prepared mesh point cloud to the state before the last denoise step.",
  hagVegetation: "For prepared pedestal points, estimate local ground and select vegetation points above the height threshold.",
  hagApply: "Run Height Above Ground selection and preview vegetation candidates.",
  hagConfirm: "Remove the currently selected vegetation candidates from the prepared pedestal point cloud.",
  hagClear: "Clear the Height Above Ground preview selection without changing the prepared pedestal point cloud.",
  roughnessRemoval: "For prepared pedestal points, compute local best-fit-plane roughness and select points above the threshold.",
  roughnessCalculate: "Calculate roughness for the current prepared pedestal points using the radius.",
  roughnessApply: "Use the threshold to select points from the current roughness heatmap without recalculating.",
  roughnessConfirm: "Remove the currently selected rough pedestal points from the prepared pedestal point cloud.",
  roughnessClear: "Clear the Roughness preview selection without changing the prepared pedestal point cloud.",
  manualRemoval: "Draw a screen-space polygon in Mesh Prep view and remove all prepared points projected inside it, including points hidden behind the front surface.",
  drawPolygon: "Add polygon vertices with left clicks in the viewer. The preview updates after three vertices.",
  undoVertex: "Remove the most recent polygon vertex and update the preview.",
  clearManualRemoval: "Clear the polygon and selected preview points without changing the prepared mesh.",
  removeSelected: "Remove selected prepared rock or interpolated bottom-face points and add this edit to the denoise undo history.",
  closeManualRemoval: "Close the manual-removal tool without changing the prepared mesh.",
  analyze: "Compute geometric metrics and make the analysis CSV available for download.",
  computeNormals: "Estimate and orient normals with PyMeshLab or Open3D before Poisson mesh reconstruction.",
  reconstruct: "Run Poisson for rock or local-plane filled-hole surface reconstruction for pedestal, based on the selected reconstruction target.",
  loadRockPedestal: "Load rock and pedestal together. Existing meshes are used first; missing meshes fall back to their segmented point clouds."
};

function wait(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function extractSummary(job: JobResponse): SessionSummary | null {
  const result = job.result as { summary?: SessionSummary } | SessionSummary | undefined;
  if (!result) {
    return null;
  }
  if ("summary" in result && result.summary) {
    return result.summary;
  }
  if ("session_id" in result) {
    return result as SessionSummary;
  }
  return null;
}

function addIndex(list: number[], index: number) {
  return list.includes(index) ? list : [...list, index];
}

function removeIndex(list: number[], index: number) {
  return list.filter((item) => item !== index);
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function projectFilenameFromName(name: string | null | undefined) {
  const fallback = "rock_detection_project";
  const raw = (name || fallback).trim();
  const withoutExtension = raw.replace(/\.(las|laz|rd3dproj|zip)$/i, "");
  const safeStem = withoutExtension.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^[._-]+|[._-]+$/g, "") || fallback;
  return `${safeStem}.rd3dproj`;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = projectFilenameFromName(filename);
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

type ProjectSaveFileHandle = {
  name?: string;
  getFile?: () => Promise<File>;
  queryPermission?: (options: { mode: "readwrite" }) => Promise<PermissionState>;
  requestPermission?: (options: { mode: "readwrite" }) => Promise<PermissionState>;
  createWritable: () => Promise<{
    write: (data: Blob) => Promise<void> | void;
    close: () => Promise<void> | void;
  }>;
};

type ProjectOpenSource = {
  file: File;
  handle: ProjectSaveFileHandle;
};

type ProjectSaveTarget = {
  filename: string;
  handle: ProjectSaveFileHandle | null;
};

async function chooseProjectOpenSource(): Promise<ProjectOpenSource | null> {
  const pickerWindow = window as Window & {
    showOpenFilePicker?: (options: {
      multiple: boolean;
      types: Array<{
        description: string;
        accept: Record<string, string[]>;
      }>;
    }) => Promise<ProjectSaveFileHandle[]>;
  };

  if (typeof pickerWindow.showOpenFilePicker !== "function") {
    return null;
  }

  let handles: ProjectSaveFileHandle[];
  try {
    handles = await pickerWindow.showOpenFilePicker({
      multiple: false,
      types: [
        {
          description: "Rock Detection 3D Project",
          accept: { "application/zip": [".rd3dproj", ".zip"] }
        }
      ]
    });
  } catch (caught) {
    if (caught instanceof DOMException && caught.name === "AbortError") {
      return null;
    }
    throw caught;
  }

  const handle = handles[0];
  if (!handle?.getFile) {
    return null;
  }
  return {
    file: await handle.getFile(),
    handle
  };
}

async function chooseProjectSaveTarget(defaultFilename: string): Promise<ProjectSaveTarget | null> {
  const suggestedName = projectFilenameFromName(defaultFilename);
  const pickerWindow = window as Window & {
    showSaveFilePicker?: (options: {
      suggestedName: string;
      types: Array<{
        description: string;
        accept: Record<string, string[]>;
      }>;
    }) => Promise<ProjectSaveFileHandle>;
  };

  if (typeof pickerWindow.showSaveFilePicker === "function") {
    let handle: ProjectSaveFileHandle;
    try {
      handle = await pickerWindow.showSaveFilePicker({
        suggestedName,
        types: [
          {
            description: "Rock Detection 3D Project",
            accept: { "application/zip": [".rd3dproj"] }
          }
        ]
      });
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") {
        return null;
      }
      throw caught;
    }
    return {
      filename: projectFilenameFromName(handle.name || suggestedName),
      handle
    };
  }

  const entered = window.prompt("Project file name", suggestedName);
  if (entered === null) {
    return null;
  }
  return {
    filename: projectFilenameFromName(entered),
    handle: null
  };
}

async function writeBlobToSaveHandle(handle: ProjectSaveFileHandle, blob: Blob) {
  const writable = await handle.createWritable();
  await writable.write(blob);
  await writable.close();
}

async function ensureProjectWritePermission(handle: ProjectSaveFileHandle | null) {
  if (!handle) {
    return false;
  }
  const permissionOptions = { mode: "readwrite" as const };
  if (handle.queryPermission) {
    const current = await handle.queryPermission(permissionOptions);
    if (current === "granted") {
      return true;
    }
  }
  if (handle.requestPermission) {
    const requested = await handle.requestPermission(permissionOptions);
    return requested === "granted";
  }
  return true;
}

function meshTargetAvailableInSummary(summary: SessionSummary, target: MeshTarget = "rock") {
  const targetState = summary.mesh_prepared_targets?.[target];
  if (targetState) {
    return Boolean(targetState.available ?? targetState.prepared ?? targetState.preview);
  }
  return target === "rock" && Boolean(summary.status.mesh_prepared);
}

function viewIsAvailable(summary: SessionSummary, viewName: ViewName, meshTarget: MeshTarget = "rock") {
  if (viewName === "raw") {
    return summary.status.point_cloud_loaded;
  }
  if (viewName === "seeds") {
    return summary.status.point_cloud_loaded;
  }
  if (viewName === "interface") {
    return summary.status.point_cloud_loaded && (summary.status.interface_ready || summary.status.manual_interface_ready || summary.status.auto_interface_ready);
  }
  if (viewName === "segmented") {
    return summary.status.segmentation_ready;
  }
  if (viewName === "voxel_segmented") {
    return Boolean(summary.status.voxel_segmentation_ready);
  }
  if (viewName === "mesh_prepared") {
    return meshTargetAvailableInSummary(summary, meshTarget);
  }
  if (viewName === "analysis") {
    return summary.status.analysis_completed;
  }
  if (viewName === "mesh") {
    const targetState = summary.mesh_reconstruction_targets?.[meshTarget];
    if (targetState) {
      return Boolean(targetState.completed);
    }
    return meshTarget === "rock" && summary.status.mesh_completed;
  }
  if (viewName === "combined_mesh") {
    return Boolean(summary.combined_reconstruction?.available);
  }
  return summary.status.mesh_completed;
}

function bestAvailableView(summary: SessionSummary, preferred?: string, meshTarget: MeshTarget = "rock"): ViewName {
  const candidateViews: ViewName[] = ["raw", "seeds", "interface", "voxel_segmented", "segmented", "mesh_prepared", "mesh", "analysis"];
  if (preferred && candidateViews.includes(preferred as ViewName) && viewIsAvailable(summary, preferred as ViewName, meshTarget)) {
    return preferred as ViewName;
  }
  for (const candidate of [...candidateViews].reverse()) {
    if (viewIsAvailable(summary, candidate, meshTarget)) {
      return candidate;
    }
  }
  return "raw";
}

function formatAnalysisPanelValue(value: unknown) {
  if (Array.isArray(value)) {
    return value.map((item) => Number(item).toFixed(4)).join(", ");
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return value.toFixed(3);
  }
  if (value === null || value === undefined || value === "") {
    return "--";
  }
  return String(value);
}

function StatusRow({ done, label }: { done: boolean; label: string }) {
  return (
    <div className={`status-row ${done ? "done" : ""}`}>
      <span>{label}</span>
    </div>
  );
}

function InfoTip({ title, children }: { title: string; children: ReactNode }) {
  return (
    <span
      className="info-anchor"
      role="button"
      tabIndex={0}
      aria-label={`${title} help`}
      onClick={(event) => {
        event.preventDefault();
        event.stopPropagation();
      }}
    >
      i
      <span className="info-popover" role="tooltip">
        <strong>{title}</strong>
        <span>{children}</span>
      </span>
    </span>
  );
}

type ActionButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  help: string;
  disabledHelp?: string;
};

function ActionButton({ help, disabledHelp, className = "", children, ...props }: ActionButtonProps) {
  const tooltip = props.disabled && disabledHelp ? disabledHelp : help;
  const isWide = className.split(/\s+/).includes("wide");
  return (
    <span className={`button-help-wrap ${isWide ? "wide-wrap" : ""}`}>
      <button {...props} className={className}>
        {children}
      </button>
      <span className="button-popover" role="tooltip">
        {tooltip}
      </span>
    </span>
  );
}

function NumericField({
  label,
  help,
  value,
  min,
  max,
  step,
  disabled = false,
  onChange
}: {
  label: string;
  help?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field">
      <span className="field-label">
        <span>{label}</span>
        {help ? <InfoTip title={label}>{help}</InfoTip> : null}
      </span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function SliderField({
  label,
  help,
  value,
  min,
  max,
  step,
  onChange
}: {
  label: string;
  help?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field">
      <span className="field-label">
        <span>
          {label} <output>{value.toFixed(2)}x</output>
        </span>
        {help ? <InfoTip title={label}>{help}</InfoTip> : null}
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

export default function App() {
  const [session, setSession] = useState<SessionSummary | null>(null);
  const [projectFilename, setProjectFilename] = useState("rock_detection_project.rd3dproj");
  const [, setProjectHasSaveTarget] = useState(false);
  const [projectSaveHandle, setProjectSaveHandle] = useState<ProjectSaveFileHandle | null>(null);
  const [view, setView] = useState<ViewerPayload | null>(null);
  const [activeView, setActiveView] = useState<ViewName>("raw");
  const [busyLabel, setBusyLabel] = useState<string | null>("Starting");
  const [error, setError] = useState<string | null>(null);
  const [pickMode, setPickMode] = useState<PickMode>("rock");
  const [rockSeeds, setRockSeeds] = useState<number[]>([]);
  const [pedestalSeeds, setPedestalSeeds] = useState<number[]>([]);
  const [interfacePoints, setInterfacePoints] = useState<number[]>([]);
  const [interfaceParts, setInterfaceParts] = useState<InterfacePartDraft[]>([]);
  const [currentPartLateral, setCurrentPartLateral] = useState(false);
  const [closeLoop, setCloseLoop] = useState(true);
  const [normalMethod, setNormalMethod] = useState<"pymeshlab" | "open3d">("pymeshlab");
  const [normalK, setNormalK] = useState(200);
  const [normalDisplayScale, setNormalDisplayScale] = useState(1);
  const [meshDepth, setMeshDepth] = useState(8);
  const [pointSize, setPointSize] = useState(0.025);
  const [segmentParams, setSegmentParams] = useState<SegmentParams>(defaultSegmentParams);
  const [denoiseParams, setDenoiseParams] = useState<DenoiseParams>(defaultDenoiseParams);
  const [hagVegetationParams, setHagVegetationParams] = useState<HagVegetationParams>(defaultHagVegetationParams);
  const [roughnessParams, setRoughnessParams] = useState<RoughnessParams>(defaultRoughnessParams);
  const [activeMeshTarget, setActiveMeshTarget] = useState<MeshTarget>("rock");
  const [hoverTipsEnabled, setHoverTipsEnabled] = useState(true);
  const [interfaceWindowOpen, setInterfaceWindowOpen] = useState(false);
  const [interfaceWindowPosition, setInterfaceWindowPosition] = useState<{ left: number; top: number } | null>(null);
  const interfaceWindowDragRef = useRef<{ offsetX: number; offsetY: number } | null>(null);
  const [interfaceEditorOpen, setInterfaceEditorOpen] = useState(false);
  const [interfaceDraft, setInterfaceDraft] = useState<InterfaceDraft | null>(null);
  const interfaceDraftSegments = useMemo(() => interfaceDraftSegmentInfo(interfaceDraft), [interfaceDraft]);
  const [manualRemovalOpen, setManualRemovalOpen] = useState(false);
  const [manualRemovalDrawing, setManualRemovalDrawing] = useState(false);
  const [manualRemovalPolygon, setManualRemovalPolygon] = useState<ScreenPoint[]>([]);
  const [manualRemovalSelected, setManualRemovalSelected] = useState<number[]>([]);
  const [vegetationSelected, setVegetationSelected] = useState<number[]>([]);
  const [roughnessSelected, setRoughnessSelected] = useState<number[]>([]);
  const [roughnessValues, setRoughnessValues] = useState<Array<number | null>>([]);
  const [roughnessStats, setRoughnessStats] = useState<RoughnessStats | null>(null);
  const [manualRemovalWindowPosition, setManualRemovalWindowPosition] = useState<{ left: number; top: number } | null>(null);
  const manualRemovalWindowDragRef = useRef<{ offsetX: number; offsetY: number } | null>(null);
  const [vegetationWindowOpen, setVegetationWindowOpen] = useState(false);
  const [roughnessWindowOpen, setRoughnessWindowOpen] = useState(false);
  const seedAutosaveSignatureRef = useRef("");

  useEffect(() => {
    let mounted = true;
    createSession()
      .then((created) => {
        if (!mounted) {
          return;
        }
        setSession(created);
        setBusyLabel(null);
      })
      .catch((caught: Error) => {
        if (!mounted) {
          return;
        }
        setError(caught.message);
        setBusyLabel(null);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const selectedForMode = useMemo(() => {
    if (pickMode === "rock") {
      return rockSeeds;
    }
    if (pickMode === "pedestal") {
      return pedestalSeeds;
    }
    return interfacePoints;
  }, [interfacePoints, pedestalSeeds, pickMode, rockSeeds]);

  const viewerMeta = useMemo(() => {
    if (!view) {
      return [];
    }
    if (view.analysis_summary) {
      return [
        view.kind === "mesh" && view.vertices?.length ? `${view.vertices.length.toLocaleString()} vertices` : null,
        view.kind === "mesh" && view.triangles?.length ? `${view.triangles.length.toLocaleString()} faces` : null,
        view.kind === "pointCloud" ? `${view.rendered_points.toLocaleString()} shown` : null,
        view.analysis_segments?.length ? `${view.analysis_segments.length.toLocaleString()} axes/lines` : null,
        view.analysis_markers?.length ? `${view.analysis_markers.length.toLocaleString()} markers` : null
      ].filter(Boolean);
    }
    if (view.kind === "mesh") {
      const target = view.mesh_target ?? activeMeshTarget;
      const method = view.mesh_method ?? (target === "pedestal" ? "local_plane_filled_holes" : "poisson");
      return [
        `${target} ${String(method).toUpperCase()}`,
        view.vertices?.length ? `${view.vertices.length.toLocaleString()} vertices` : null,
        view.triangles?.length ? `${view.triangles.length.toLocaleString()} faces` : null,
        Number.isFinite(view.triangle_count) ? `${Number(view.triangle_count).toLocaleString()} triangles` : null
      ].filter(Boolean);
    }
    if (view.kind === "combinedMesh") {
      return [
        `${view.total_points.toLocaleString()} total`,
        ...view.components.map((component) => `${component.target} ${component.source}`)
      ];
    }
    const parts = [
      `${view.rendered_points.toLocaleString()} shown`,
      view.normal_segments ? `${view.normal_segments.length.toLocaleString()} normal arrows` : null,
      view.normal_diagnostics ? `${view.normal_diagnostics.nonzero_normal_count.toLocaleString()} valid normals` : null
    ].filter(Boolean);
    if (activeView === "mesh_prepared") {
      const target = view.mesh_target ?? activeMeshTarget;
      if (Number.isFinite(view.object_point_count)) {
        parts.push(`${target} ${view.object_point_count?.toLocaleString()}`);
      }
      if (Number.isFinite(view.interface_fill_point_count) && Number(view.interface_fill_point_count) > 0) {
        parts.push(`interface fill ${view.interface_fill_point_count?.toLocaleString()}`);
      }
    }
    return parts;
  }, [activeMeshTarget, activeView, view]);

  const analysisSummary = view?.analysis_summary;
  const roughnessHeatmapRange = roughnessWindowOpen && roughnessStats
    ? {
      min: Number(roughnessStats.min_roughness ?? 0),
      max: Number(roughnessStats.max_roughness ?? 0)
    }
    : null;
  const meshTargetAvailable = useCallback(
    (target: MeshTarget = activeMeshTarget, summary: SessionSummary | null = session) => {
      const targetState = summary?.mesh_prepared_targets?.[target];
      if (targetState) {
        return Boolean(targetState.available ?? targetState.prepared ?? targetState.preview);
      }
      return target === "rock" && Boolean(summary?.status.mesh_prepared);
    },
    [activeMeshTarget, session]
  );
  const meshTargetSaved = useCallback(
    (target: MeshTarget = activeMeshTarget, summary: SessionSummary | null = session) => {
      const targetState = summary?.mesh_prepared_targets?.[target];
      if (targetState) {
        return Boolean(targetState.prepared);
      }
      return target === "rock" && Boolean(summary?.status.mesh_prepared);
    },
    [activeMeshTarget, session]
  );
  const meshTargetPrepared = meshTargetAvailable;
  const activeMeshTargetPrepared = meshTargetAvailable(activeMeshTarget);
  const activeMeshTargetSaved = meshTargetSaved(activeMeshTarget);
  const meshReconstructionCompleted = useCallback(
    (target: MeshTarget = activeMeshTarget, summary: SessionSummary | null = session) => {
      const targetState = summary?.mesh_reconstruction_targets?.[target];
      if (targetState) {
        return Boolean(targetState.completed);
      }
      return target === "rock" && Boolean(summary?.status.mesh_completed);
    },
    [activeMeshTarget, session]
  );

  const refreshSession = useCallback(
    async (preferredSession?: SessionSummary, options: { syncSeeds?: boolean } = {}) => {
      const next = preferredSession ?? (session ? await getSession(session.session_id) : null);
      if (!next) {
        return null;
      }
      setSession(next);
      if (options.syncSeeds) {
        setRockSeeds(next.seeds.rock ?? []);
        setPedestalSeeds(next.seeds.pedestal ?? []);
      }
      return next;
    },
    [session]
  );

  const refreshView = useCallback(
    async (viewName: ViewName, summary?: SessionSummary | null, options: { meshTarget?: MeshTarget } = {}) => {
      const targetSession = summary ?? session;
      if (!targetSession) {
        return;
      }
      const payload = await getViewer(
        targetSession.session_id,
        viewName,
        viewName === "mesh_prepared" || viewName === "mesh"
          ? { meshTarget: options.meshTarget ?? activeMeshTarget }
          : undefined
      );
      setView(payload);
      setActiveView(viewName);
    },
    [activeMeshTarget, session]
  );

  function buildProjectUiState(filename = projectFilename): ProjectUiState {
    return {
      project_filename: projectFilenameFromName(filename),
      active_view: activeView,
      pick_mode: pickMode,
      active_mesh_target: activeMeshTarget,
      point_size: pointSize,
      segment_params: segmentParams,
      denoise_params: denoiseParams,
      normal_method: normalMethod,
      normal_k: normalK,
      normal_display_scale: normalDisplayScale,
      mesh_depth: meshDepth,
      hover_tips_enabled: hoverTipsEnabled,
      interface_points: interfacePoints,
      interface_parts: interfaceParts,
      current_part_lateral: currentPartLateral,
      close_loop: closeLoop
    };
  }

  function restoreProjectUiState(uiState: ProjectUiState | undefined, summary: SessionSummary): MeshTarget {
    const restoredFilename = projectFilenameFromName(uiState?.project_filename || summary.current_file || "rock_detection_project");
    setProjectFilename(restoredFilename);
    setRockSeeds(summary.seeds.rock ?? []);
    setPedestalSeeds(summary.seeds.pedestal ?? []);
    seedAutosaveSignatureRef.current = JSON.stringify({
      rock_seed_indices: summary.seeds.rock ?? [],
      pedestal_seed_indices: summary.seeds.pedestal ?? []
    });
    if (uiState?.segment_params) {
      setSegmentParams({ ...defaultSegmentParams, ...uiState.segment_params });
    }
    if (uiState?.denoise_params) {
      setDenoiseParams({ ...defaultDenoiseParams, ...uiState.denoise_params });
    }
    if (uiState?.normal_method === "open3d" || uiState?.normal_method === "pymeshlab") {
      setNormalMethod(uiState.normal_method);
    }
    if (typeof uiState?.normal_k === "number" && Number.isFinite(uiState.normal_k)) {
      setNormalK(uiState.normal_k);
    }
    if (typeof uiState?.normal_display_scale === "number" && Number.isFinite(uiState.normal_display_scale)) {
      setNormalDisplayScale(uiState.normal_display_scale);
    }
    if (typeof uiState?.mesh_depth === "number" && Number.isFinite(uiState.mesh_depth)) {
      setMeshDepth(uiState.mesh_depth);
    }
    if (typeof uiState?.point_size === "number" && Number.isFinite(uiState.point_size)) {
      setPointSize(uiState.point_size);
    }
    if (typeof uiState?.hover_tips_enabled === "boolean") {
      setHoverTipsEnabled(uiState.hover_tips_enabled);
    }
    if (uiState?.pick_mode === "rock" || uiState?.pick_mode === "pedestal" || uiState?.pick_mode === "interface") {
      setPickMode(uiState.pick_mode);
    }
    let nextMeshTarget: MeshTarget = uiState?.active_mesh_target === "pedestal" ? "pedestal" : "rock";
    const targetState = summary.mesh_prepared_targets?.[nextMeshTarget];
    if (targetState && !(targetState.available ?? targetState.prepared ?? targetState.preview)) {
      const rockState = summary.mesh_prepared_targets?.rock;
      const pedestalState = summary.mesh_prepared_targets?.pedestal;
      if (rockState?.available ?? rockState?.prepared ?? rockState?.preview) {
        nextMeshTarget = "rock";
      } else if (pedestalState?.available ?? pedestalState?.prepared ?? pedestalState?.preview) {
        nextMeshTarget = "pedestal";
      }
    }
    setActiveMeshTarget(nextMeshTarget);
    setInterfacePoints(Array.isArray(uiState?.interface_points) ? uiState.interface_points : []);
    setInterfaceParts(Array.isArray(uiState?.interface_parts) ? uiState.interface_parts : []);
    setCurrentPartLateral(Boolean(uiState?.current_part_lateral));
    setCloseLoop(typeof uiState?.close_loop === "boolean" ? uiState.close_loop : true);
    setInterfaceDraft(null);
    setInterfaceEditorOpen(false);
    clearManualRemoval();
    return nextMeshTarget;
  }

  async function pollJob(jobId: string) {
    let job = await getJob(jobId);
    while (job.status === "queued" || job.status === "running") {
      await wait(650);
      job = await getJob(jobId);
    }
    if (job.status === "failed") {
      throw new Error(job.error ?? "Job failed");
    }
    return job;
  }

  async function runWorkflowAction(
    label: string,
    endpoint: string,
    body?: unknown,
    nextView?: ViewName | ((result: unknown) => ViewName | undefined),
    options: { syncSeeds?: boolean } = {}
  ) {
    if (!session) {
      return;
    }
    setBusyLabel(label);
    setError(null);
    try {
      const submitted = await runJob(endpoint, body);
      const job = await pollJob(submitted.job_id);
      const summary = await refreshSession(extractSummary(job) ?? undefined, { syncSeeds: Boolean(options.syncSeeds) });
      const resolvedNextView = typeof nextView === "function" ? nextView(job.result) : nextView;
      if (resolvedNextView && summary) {
        await refreshView(resolvedNextView, summary);
      }
      return job;
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      return null;
    } finally {
      setBusyLabel(null);
    }
  }

  async function saveSeedsNow() {
    if (!session?.status.point_cloud_loaded) {
      return true;
    }
    const payload = {
      rock_seed_indices: rockSeeds,
      pedestal_seed_indices: pedestalSeeds
    };
    const signature = JSON.stringify(payload);
    if (signature === seedAutosaveSignatureRef.current) {
      return true;
    }
    try {
      const submitted = await runJob(`/api/sessions/${session.session_id}/seeds/manual`, payload);
      const job = await pollJob(submitted.job_id);
      seedAutosaveSignatureRef.current = signature;
      await refreshSession(extractSummary(job) ?? undefined);
      return true;
    } catch (caught) {
      setError(caught instanceof Error ? `Could not save seeds automatically: ${caught.message}` : String(caught));
      return false;
    }
  }

  async function handleProjectExport(
    filename: string,
    options: { saveHandle?: ProjectSaveFileHandle | null; establishSaveTarget?: boolean } = {}
  ) {
    if (!session?.status.point_cloud_loaded) {
      return;
    }
    setBusyLabel("Saving project");
    setError(null);
    try {
      if (!(await saveSeedsNow())) {
        return;
      }
      const safeFilename = projectFilenameFromName(filename);
      const exported = await exportProject(session.session_id, {
        filename: safeFilename,
        ui_state: buildProjectUiState(safeFilename)
      });
      if (options.saveHandle) {
        await writeBlobToSaveHandle(options.saveHandle, exported.blob);
        setProjectSaveHandle(options.saveHandle);
        setProjectFilename(projectFilenameFromName(options.saveHandle.name || exported.filename || safeFilename));
        setProjectHasSaveTarget(true);
      } else {
        downloadBlob(exported.blob, exported.filename || safeFilename);
        setProjectSaveHandle(null);
        setProjectFilename(projectFilenameFromName(exported.filename || safeFilename));
        setProjectHasSaveTarget(false);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusyLabel(null);
    }
  }

  async function handleProjectSave() {
    if (!projectSaveHandle) {
      await handleProjectSaveAs();
      return;
    }
    const permitted = await ensureProjectWritePermission(projectSaveHandle);
    if (!permitted) {
      setError("Write permission was not granted. Use Save As to choose a writable project file.");
      return;
    }
    await handleProjectExport(projectFilename, { saveHandle: projectSaveHandle });
  }

  async function handleProjectSaveAs() {
    const target = await chooseProjectSaveTarget(projectFilename);
    if (!target) {
      return;
    }
    if (target.handle) {
      const permitted = await ensureProjectWritePermission(target.handle);
      if (!permitted) {
        setError("Write permission was not granted for that project file.");
        return;
      }
    }
    await handleProjectExport(target.filename, {
      saveHandle: target.handle,
      establishSaveTarget: true
    });
  }

  async function handleProjectImport(file: File | null, options: { saveHandle?: ProjectSaveFileHandle | null } = {}) {
    if (!file || !session) {
      return;
    }
    setBusyLabel("Importing project");
    setError(null);
    try {
      const imported = await importProject(session.session_id, file);
      await refreshSession(imported.summary);
      const restoredMeshTarget = restoreProjectUiState(imported.ui_state, imported.summary);
      setProjectFilename(projectFilenameFromName(imported.project_filename || file.name));
      setProjectSaveHandle(options.saveHandle || null);
      setProjectHasSaveTarget(Boolean(options.saveHandle));
      setVegetationWindowOpen(false);
      setVegetationSelected([]);
      setRoughnessWindowOpen(false);
      setRoughnessSelected([]);
      setRoughnessValues([]);
      setRoughnessStats(null);
      clearManualRemoval();
      const nextView = bestAvailableView(imported.summary, imported.ui_state?.active_view, restoredMeshTarget);
      await refreshView(nextView, imported.summary, { meshTarget: restoredMeshTarget });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusyLabel(null);
    }
  }

  useEffect(() => {
    if (!session?.status.point_cloud_loaded) {
      return undefined;
    }
    const payload = {
      rock_seed_indices: rockSeeds,
      pedestal_seed_indices: pedestalSeeds
    };
    const signature = JSON.stringify(payload);
    if (signature === seedAutosaveSignatureRef.current) {
      return undefined;
    }
    let cancelled = false;
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          const submitted = await runJob(`/api/sessions/${session.session_id}/seeds/manual`, payload);
          const job = await pollJob(submitted.job_id);
          seedAutosaveSignatureRef.current = signature;
          if (!cancelled) {
            await refreshSession(extractSummary(job) ?? undefined);
          }
        } catch (caught) {
          if (!cancelled) {
            setError(caught instanceof Error ? `Could not save seeds automatically: ${caught.message}` : String(caught));
          }
        }
      })();
    }, 300);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [pedestalSeeds, refreshSession, rockSeeds, session?.session_id, session?.status.point_cloud_loaded]);

  async function handleUpload(file: File | null) {
    if (!file || !session) {
      return;
    }
    setBusyLabel("Uploading");
    setError(null);
    try {
      const summary = await uploadPointCloud(session.session_id, file);
      await refreshSession(summary);
      setRockSeeds([]);
      setPedestalSeeds([]);
      setInterfaceParts([]);
      setInterfacePoints([]);
      setInterfaceDraft(null);
      setInterfaceEditorOpen(false);
      setCurrentPartLateral(false);
      setProjectFilename(projectFilenameFromName(file.name));
      setProjectHasSaveTarget(false);
      setProjectSaveHandle(null);
      setActiveMeshTarget("rock");
      setVegetationWindowOpen(false);
      setVegetationSelected([]);
      setRoughnessWindowOpen(false);
      setRoughnessSelected([]);
      setRoughnessValues([]);
      setRoughnessStats(null);
      clearManualRemoval();
      await refreshView("raw", summary);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusyLabel(null);
    }
  }

  function handlePickPoint(index: number) {
    if (activeView === "analysis") {
      return;
    }
    if (pickMode === "rock") {
      setRockSeeds((items) => addIndex(items, index));
    } else if (pickMode === "pedestal") {
      setPedestalSeeds((items) => addIndex(items, index));
    } else {
      setInterfacePoints((items) => addIndex(items, index));
    }
  }

  function handleUnpickPoint(index: number) {
    if (activeView === "analysis") {
      return;
    }
    if (pickMode === "rock") {
      setRockSeeds((items) => removeIndex(items, index));
    } else if (pickMode === "pedestal") {
      setPedestalSeeds((items) => removeIndex(items, index));
    } else {
      setInterfacePoints((items) => removeIndex(items, index));
    }
  }

  async function autoSeeds() {
    if (!session) {
      return;
    }
    setPickMode("rock");
    await runWorkflowAction(
      "Auto seeds",
      `/api/sessions/${session.session_id}/seeds/auto`,
      undefined,
      "seeds",
      { syncSeeds: true }
    );
  }

  function clearCurrentPickMode() {
    if (pickMode === "rock") {
      setRockSeeds([]);
    } else if (pickMode === "pedestal") {
      setPedestalSeeds([]);
    } else {
      setInterfacePoints([]);
    }
  }

  function openInterfaceTools() {
    setPickMode("interface");
    setInterfaceWindowOpen(true);
  }

  function beginInterfaceWindowDrag(event: ReactPointerEvent<HTMLDivElement>) {
    if (event.button !== 0) {
      return;
    }
    if ((event.target as HTMLElement).closest("button")) {
      return;
    }
    const panel = event.currentTarget.closest(".floating-window") as HTMLElement | null;
    if (!panel) {
      return;
    }
    const rect = panel.getBoundingClientRect();
    interfaceWindowDragRef.current = {
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  function dragInterfaceWindow(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = interfaceWindowDragRef.current;
    if (!drag) {
      return;
    }
    const panel = event.currentTarget.closest(".floating-window") as HTMLElement | null;
    const rect = panel?.getBoundingClientRect();
    const width = rect?.width ?? 340;
    const height = rect?.height ?? 260;
    const margin = 8;
    setInterfaceWindowPosition({
      left: clamp(event.clientX - drag.offsetX, margin, Math.max(margin, window.innerWidth - width - margin)),
      top: clamp(event.clientY - drag.offsetY, margin, Math.max(margin, window.innerHeight - height - margin))
    });
  }

  function endInterfaceWindowDrag(event: ReactPointerEvent<HTMLDivElement>) {
    interfaceWindowDragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function stageInterfacePart() {
    if (interfacePoints.length < 2) {
      setError("Interface parts need at least two points.");
      return;
    }
    setInterfaceParts((parts) => [
      ...parts,
      { selected_indices: [...interfacePoints], is_lateral: currentPartLateral }
    ]);
    setInterfacePoints([]);
    setCurrentPartLateral(false);
    setError(null);
  }

  function buildInterfacePayload() {
    const parts = [...interfaceParts];
    if (interfacePoints.length >= 2) {
      parts.push({ selected_indices: [...interfacePoints], is_lateral: currentPartLateral });
    }
    if (!parts.length) {
      setError("Select at least two interface points.");
      return null;
    }
    return { parts, close_loop: closeLoop };
  }

  async function interpolateInterface() {
    if (!session) {
      return;
    }
    const payload = buildInterfacePayload();
    if (!payload) {
      return;
    }
    await runWorkflowAction(
      "Interpolating interface path",
      `/api/sessions/${session.session_id}/interface/interpolate`,
      payload,
      "interface"
    );
  }

  async function saveInterface() {
    if (!session) {
      return;
    }
    const payload = buildInterfacePayload();
    if (!payload) {
      return;
    }
    await runWorkflowAction(
      "Saving interface",
      `/api/sessions/${session.session_id}/interface`,
      payload,
      "interface"
    );
  }

  async function clearInterfaceParts() {
    setInterfaceParts([]);
    setInterfacePoints([]);
    if (!session?.status.point_cloud_loaded) {
      return;
    }
    await runWorkflowAction(
      "Clearing interface preview",
      `/api/sessions/${session.session_id}/interface/preview/clear`,
      undefined,
      activeView === "interface" ? "interface" : undefined
    );
  }

  async function openInterfaceEditorForSource(source: "auto" | "manual") {
    if (!session) {
      return;
    }
    setBusyLabel("Creating interface draft");
    setError(null);
    try {
      const submitted = await createInterfaceDraftFromSource(session.session_id, source);
      const job = await pollJob(submitted.job_id);
      const summary = await refreshSession(extractSummary(job) ?? undefined);
      const result = job.result as { draft?: InterfaceDraft } | undefined;
      if (result?.draft) {
        setInterfaceDraft(result.draft);
      } else {
        const draftResponse = await getInterfaceDraft(session.session_id);
        setInterfaceDraft(draftResponse.draft);
      }
      setInterfaceEditorOpen(true);
      if (summary) {
        await refreshView("interface", summary);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusyLabel(null);
    }
  }

  async function undoInterfaceDraftEdit() {
    if (!session) {
      return;
    }
    setBusyLabel("Undoing interface edit");
    setError(null);
    try {
      const submitted = await undoInterfaceDraft(session.session_id);
      const job = await pollJob(submitted.job_id);
      const summary = await refreshSession(extractSummary(job) ?? undefined);
      const result = job.result as { draft?: InterfaceDraft } | undefined;
      setInterfaceDraft(result?.draft ?? (await getInterfaceDraft(session.session_id)).draft);
      if (summary) {
        await refreshView("interface", summary);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusyLabel(null);
    }
  }

  async function discardInterfaceDraft() {
    if (!session || !interfaceDraft) {
      return;
    }
    setBusyLabel("Clearing interface draft");
    setError(null);
    try {
      const submitted = await clearInterfaceDraftApi(session.session_id);
      const job = await pollJob(submitted.job_id);
      const summary = await refreshSession(extractSummary(job) ?? undefined);
      setInterfaceDraft(null);
      setInterfaceEditorOpen(false);
      if (summary) {
        await refreshView("interface", summary);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusyLabel(null);
    }
  }

  async function saveDraftAsManualInterface() {
    if (!session || !interfaceDraft) {
      return;
    }
    setBusyLabel("Saving draft as manual interface");
    setError(null);
    try {
      const submitted = await commitInterfaceDraft(session.session_id);
      const job = await pollJob(submitted.job_id);
      const summary = await refreshSession(extractSummary(job) ?? undefined);
      setInterfaceDraft(null);
      setInterfaceEditorOpen(false);
      if (summary) {
        await refreshView("interface", summary);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusyLabel(null);
    }
  }

  function clearManualRemoval() {
    setManualRemovalDrawing(false);
    setManualRemovalPolygon([]);
    setManualRemovalSelected([]);
  }

  function clearPreparedSelectionPreviews() {
    clearManualRemoval();
    setVegetationSelected([]);
    setRoughnessSelected([]);
    setRoughnessValues([]);
    setRoughnessStats(null);
  }

  function closeVegetationTool() {
    setVegetationWindowOpen(false);
    setVegetationSelected([]);
    clearManualRemoval();
  }

  function closeRoughnessTool() {
    setRoughnessWindowOpen(false);
    setRoughnessSelected([]);
    setRoughnessValues([]);
    setRoughnessStats(null);
    clearManualRemoval();
  }

  async function openVegetationTool() {
    if (activeMeshTarget !== "pedestal") {
      setError("Height Above Ground vegetation removal is only available for prepared pedestal mesh.");
      return;
    }
    if (!activeMeshTargetPrepared) {
      setError("Prepare the pedestal mesh before running Height Above Ground selection.");
      return;
    }
    setManualRemovalOpen(false);
    setRoughnessWindowOpen(false);
    setRoughnessSelected([]);
    setRoughnessValues([]);
    setRoughnessStats(null);
    clearManualRemoval();
    setVegetationWindowOpen(true);
    if (activeView !== "mesh_prepared") {
      await refreshView("mesh_prepared");
    }
  }

  async function applyHagVegetationSelection() {
    if (!session) {
      return;
    }
    const job = await runWorkflowAction(
      "Selecting vegetation",
      `/api/sessions/${session.session_id}/mesh/vegetation/hag/select`,
      { ...hagVegetationParams, target: "pedestal" },
      "mesh_prepared"
    );
    const result = job?.result as { selected_indices?: number[] } | undefined;
    const selected = Array.isArray(result?.selected_indices) ? result.selected_indices.map(Number).filter(Number.isFinite) : [];
    setManualRemovalDrawing(false);
    setManualRemovalPolygon([]);
    setManualRemovalSelected([]);
    setVegetationSelected(selected);
    setVegetationWindowOpen(true);
  }

  async function confirmHagVegetationRemoval() {
    if (!session || !vegetationSelected.length) {
      setError("Apply Height Above Ground first to select vegetation candidates.");
      return;
    }
    const job = await runWorkflowAction(
      "Removing vegetation",
      `/api/sessions/${session.session_id}/mesh/noise/manual-remove`,
      { selected_indices: vegetationSelected, target: "pedestal" },
      "mesh_prepared"
    );
    if (job) {
      setVegetationSelected([]);
      clearManualRemoval();
      setVegetationWindowOpen(true);
    }
  }

  async function openRoughnessTool() {
    if (activeMeshTarget !== "pedestal") {
      setError("Roughness removal is only available for prepared pedestal mesh.");
      return;
    }
    if (!activeMeshTargetPrepared) {
      setError("Prepare the pedestal mesh before running roughness selection.");
      return;
    }
    setManualRemovalOpen(false);
    setVegetationWindowOpen(false);
    setVegetationSelected([]);
    clearManualRemoval();
    setRoughnessWindowOpen(true);
    if (activeView !== "mesh_prepared") {
      await refreshView("mesh_prepared");
    }
  }

  async function calculateRoughnessHeatmap() {
    if (!session) {
      return;
    }
    const job = await runWorkflowAction(
      "Calculating roughness",
      `/api/sessions/${session.session_id}/mesh/roughness/calculate`,
      { radius: roughnessParams.radius, target: "pedestal" },
      "mesh_prepared"
    );
    const result = job?.result as ({ roughness_values?: Array<number | null> } & RoughnessStats) | undefined;
    const values = Array.isArray(result?.roughness_values)
      ? result.roughness_values.map((value) => {
        const numeric = Number(value);
        return Number.isFinite(numeric) ? numeric : null;
      })
      : [];
    setManualRemovalDrawing(false);
    setManualRemovalPolygon([]);
    setManualRemovalSelected([]);
    setRoughnessSelected([]);
    setRoughnessValues(values);
    setRoughnessStats({
      min_roughness: result?.min_roughness,
      max_roughness: result?.max_roughness,
      mean_roughness: result?.mean_roughness,
      valid_roughness_count: result?.valid_roughness_count,
      voxel_size: result?.voxel_size,
      voxel_point_count: result?.voxel_point_count
    });
    setRoughnessWindowOpen(true);
  }

  function applyRoughnessThreshold() {
    if (!roughnessValues.length) {
      setError("Calculate roughness before applying the threshold.");
      return;
    }
    const selected: number[] = [];
    roughnessValues.forEach((value, index) => {
      const numeric = Number(value);
      if (Number.isFinite(numeric) && numeric > roughnessParams.threshold) {
        selected.push(index);
      }
    });
    setRoughnessSelected(selected);
  }

  async function confirmRoughnessRemoval() {
    if (!session || !roughnessSelected.length) {
      setError("Apply Roughness first to select points above the threshold.");
      return;
    }
    const job = await runWorkflowAction(
      "Removing rough points",
      `/api/sessions/${session.session_id}/mesh/noise/manual-remove`,
      { selected_indices: roughnessSelected, target: "pedestal" },
      "mesh_prepared"
    );
    if (job) {
      setRoughnessSelected([]);
      setRoughnessValues([]);
      setRoughnessStats(null);
      clearManualRemoval();
      setRoughnessWindowOpen(true);
    }
  }

  async function openManualRemovalTools() {
    if (!activeMeshTargetPrepared) {
      setError(`Prepare the ${activeMeshTarget} mesh before manual removal.`);
      return;
    }
    setVegetationWindowOpen(false);
    setVegetationSelected([]);
    setRoughnessWindowOpen(false);
    setRoughnessSelected([]);
    setRoughnessValues([]);
    setRoughnessStats(null);
    setManualRemovalOpen(true);
    if (activeView !== "mesh_prepared") {
      await refreshView("mesh_prepared");
    }
  }

  async function applyManualRemoval() {
    if (!session || !manualRemovalSelected.length) {
      setError("Draw a polygon that selects at least one visible prepared point.");
      return;
    }
    const job = await runWorkflowAction(
      "Manual removal",
      `/api/sessions/${session.session_id}/mesh/noise/manual-remove`,
      { selected_indices: manualRemovalSelected, target: activeMeshTarget },
      "mesh_prepared"
    );
    if (job) {
      clearManualRemoval();
      setManualRemovalOpen(true);
    }
  }

  async function selectMeshTarget(target: MeshTarget) {
    setActiveMeshTarget(target);
    if (target !== "pedestal") {
      setVegetationWindowOpen(false);
      setVegetationSelected([]);
      setRoughnessWindowOpen(false);
      setRoughnessSelected([]);
      setRoughnessValues([]);
      setRoughnessStats(null);
    }
    clearManualRemoval();
    if (activeView === "mesh_prepared" && session) {
      const nextView = meshTargetPrepared(target) ? "mesh_prepared" : bestAvailableView(session, undefined, target);
      await refreshView(nextView, session, { meshTarget: target });
    } else if (activeView === "mesh" && session) {
      const nextView = meshReconstructionCompleted(target) ? "mesh" : bestAvailableView(session, undefined, target);
      await refreshView(nextView, session, { meshTarget: target });
    }
  }

  function beginManualRemovalWindowDrag(event: ReactPointerEvent<HTMLDivElement>) {
    if (event.button !== 0) {
      return;
    }
    if ((event.target as HTMLElement).closest("button")) {
      return;
    }
    const panel = event.currentTarget.closest(".floating-window") as HTMLElement | null;
    if (!panel) {
      return;
    }
    const rect = panel.getBoundingClientRect();
    manualRemovalWindowDragRef.current = {
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  function dragManualRemovalWindow(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = manualRemovalWindowDragRef.current;
    if (!drag) {
      return;
    }
    const panel = event.currentTarget.closest(".floating-window") as HTMLElement | null;
    const rect = panel?.getBoundingClientRect();
    const width = rect?.width ?? 340;
    const height = rect?.height ?? 260;
    const margin = 8;
    setManualRemovalWindowPosition({
      left: clamp(event.clientX - drag.offsetX, margin, Math.max(margin, window.innerWidth - width - margin)),
      top: clamp(event.clientY - drag.offsetY, margin, Math.max(margin, window.innerHeight - height - margin))
    });
  }

  function endManualRemovalWindowDrag(event: ReactPointerEvent<HTMLDivElement>) {
    manualRemovalWindowDragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  const canDownload = session?.outputs ?? { segmented: null, mesh: null, pedestal_mesh: null, analysis: null };
  const activeMeshDownloadKind = activeMeshTarget === "pedestal" ? "pedestal_mesh" : "mesh";
  const activeMeshDownloadPath = activeMeshTarget === "pedestal" ? canDownload.pedestal_mesh : canDownload.mesh;
  const interfaceWindowStyle: CSSProperties | undefined = interfaceWindowPosition
    ? { left: interfaceWindowPosition.left, top: interfaceWindowPosition.top, right: "auto" }
    : undefined;
  const manualRemovalWindowStyle: CSSProperties | undefined = manualRemovalWindowPosition
    ? { left: manualRemovalWindowPosition.left, top: manualRemovalWindowPosition.top, right: "auto" }
    : undefined;

  return (
    <div className={`app-shell ${hoverTipsEnabled ? "" : "tips-off"}`}>
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark" aria-hidden="true"></div>
          <div>
            <h1>Rock Detection 3D</h1>
            <p>{session?.current_file ?? "No point cloud loaded"}</p>
          </div>
        </div>

        <div className="project-actions">
          <label className="project-file-button">
            <span>Import Project</span>
            <input
              type="file"
              accept=".rd3dproj,.zip"
              onClick={(event) => {
                const pickerWindow = window as Window & { showOpenFilePicker?: unknown };
                if (typeof pickerWindow.showOpenFilePicker !== "function") {
                  return;
                }
                event.preventDefault();
                event.stopPropagation();
                void (async () => {
                  try {
                    const source = await chooseProjectOpenSource();
                    if (source) {
                      await handleProjectImport(source.file, { saveHandle: source.handle });
                    }
                  } catch (caught) {
                    setError(caught instanceof Error ? caught.message : String(caught));
                  }
                })();
                event.currentTarget.value = "";
              }}
              onChange={(event) => {
                void handleProjectImport(event.target.files?.[0] ?? null);
                event.currentTarget.value = "";
              }}
            />
            <span className="button-popover" role="tooltip">
              {buttonHelp.importProject}
            </span>
          </label>
          <ActionButton
            disabled={!session?.status.point_cloud_loaded}
            help={buttonHelp.saveProject}
            disabledHelp="Load or import a point cloud first."
            onClick={handleProjectSave}
          >
            Save Project
          </ActionButton>
          <ActionButton
            disabled={!session?.status.point_cloud_loaded}
            help={buttonHelp.saveAsProject}
            disabledHelp="Load or import a point cloud first."
            onClick={handleProjectSaveAs}
          >
            Save As
          </ActionButton>
        </div>

        <label className="upload-button">
          <span>Upload LAS/LAZ</span>
          <input
            type="file"
            accept=".las,.laz"
            onChange={(event) => handleUpload(event.target.files?.[0] ?? null)}
          />
        </label>

        <section className="panel">
          <h2>Workflow</h2>
          <StatusRow done={Boolean(session?.status.point_cloud_loaded)} label="Point cloud" />
          <StatusRow done={Boolean(session?.status.seeds_ready)} label="Seeds" />
          <StatusRow done={Boolean(session?.status.interface_ready)} label="Interface" />
          <StatusRow done={Boolean(session?.status.segmentation_ready)} label="Segmentation" />
          <StatusRow done={Boolean(session?.status.mesh_prepared)} label="Mesh prep" />
          <StatusRow done={meshReconstructionCompleted("rock") || meshReconstructionCompleted("pedestal")} label="Mesh" />
          <StatusRow done={Boolean(session?.status.analysis_completed)} label="Analysis" />
        </section>

        <section className="panel compact">
          <h2>Views</h2>
          <div className="view-grid">
            {(["raw", "seeds", "interface", "voxel_segmented", "segmented", "mesh_prepared", "mesh", "analysis"] as ViewName[]).map((name) => (
              <ActionButton
                key={name}
                className={activeView === name ? "active" : ""}
                disabled={!session || !viewIsAvailable(session, name, activeMeshTarget)}
                help={viewHelp[name]}
                disabledHelp={name === "analysis" ? "Run analysis first." : "Complete the required workflow step first."}
                onClick={() => refreshView(name)}
              >
                {viewLabels[name]}
              </ActionButton>
            ))}
          </div>
          <button
            className={`tip-toggle ${hoverTipsEnabled ? "" : "off"}`}
            type="button"
            aria-pressed={hoverTipsEnabled}
            onClick={() => setHoverTipsEnabled((enabled) => !enabled)}
          >
            {hoverTipsEnabled ? "Hover Tips On" : "Hover Tips Off"}
          </button>
        </section>

        <section className="panel compact">
          <h2>Downloads</h2>
          <a className={!canDownload.segmented ? "disabled-link" : ""} href={canDownload.segmented && session ? downloadUrl(session.session_id, "segmented") : undefined}>
            Segmented LAS
          </a>
          <a className={!activeMeshDownloadPath ? "disabled-link" : ""} href={activeMeshDownloadPath && session ? downloadUrl(session.session_id, activeMeshDownloadKind) : undefined}>
            Mesh PLY
          </a>
          <a className={!canDownload.analysis ? "disabled-link" : ""} href={canDownload.analysis && session ? downloadUrl(session.session_id, "analysis") : undefined}>
            Analysis CSV
          </a>
        </section>
      </aside>

      <main className="workspace">
        <div className="viewer-header">
          <div>
            <span className="eyebrow">Viewport</span>
            <strong>{activeView.replace("_", " ")}</strong>
          </div>
          <div className="meta-row">
            <span>{session?.point_count.toLocaleString() ?? 0} pts</span>
            <span>EPSG {session?.epsg_code ?? "--"}</span>
            {viewerMeta.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        </div>

        <PointCloudViewer
          view={view}
          onPickPoint={handlePickPoint}
          onUnpickPoint={handleUnpickPoint}
          pickedIndices={selectedForMode}
          pickedColor={PICKED_MARKER_COLORS[pickMode]}
          pointSize={pointSize}
          normalDisplayScale={normalDisplayScale}
          highlightIndices={
            roughnessWindowOpen && activeMeshTarget === "pedestal"
              ? roughnessSelected
              : vegetationWindowOpen && activeMeshTarget === "pedestal"
                ? vegetationSelected
                : []
          }
          highlightColor={roughnessWindowOpen ? [0.95, 0.18, 0.85] : [1.0, 0.84, 0.0]}
          heatmapValues={roughnessWindowOpen && activeMeshTarget === "pedestal" ? roughnessValues : []}
          heatmapRange={roughnessHeatmapRange}
          manualRemoval={{
            active: manualRemovalOpen && activeView === "mesh_prepared",
            drawing: manualRemovalDrawing,
            polygon: manualRemovalPolygon,
            selectedIndices: manualRemovalSelected,
            onAddVertex: (point) => setManualRemovalPolygon((polygon) => [...polygon, point]),
            onSelectionChange: setManualRemovalSelected
          }}
        />

        {roughnessWindowOpen && roughnessStats && roughnessValues.length > 0 && (
          <div className="roughness-colorbar">
            <div className="roughness-colorbar-title">Roughness</div>
            <div className="roughness-colorbar-gradient" />
            <div className="roughness-colorbar-labels">
              <span>{Number(roughnessStats.min_roughness ?? 0).toFixed(4)} m</span>
              <span>{Number(roughnessStats.max_roughness ?? 0).toFixed(4)} m</span>
            </div>
          </div>
        )}

        {activeView === "analysis" && analysisSummary && (
          <div className="analysis-window">
            <div className="analysis-window-title">{analysisSummary.title ?? "Analysis"}</div>
            <div className="analysis-window-list">
              {(analysisSummary.metrics ?? []).map((item) => (
                <div className="analysis-window-row" key={item.label}>
                  <span>{item.label}</span>
                  <strong>{formatAnalysisPanelValue(item.value)}</strong>
                </div>
              ))}
            </div>
            <div className="analysis-window-list analysis-window-vectors">
              {(analysisSummary.vectors ?? []).map((item) => (
                <div className="analysis-window-row" key={item.label}>
                  <span>{item.label}</span>
                  <strong>{formatAnalysisPanelValue(item.value)}</strong>
                </div>
              ))}
            </div>
          </div>
        )}

        {(busyLabel || error) && (
          <div className={`toast ${error ? "error" : ""}`}>
            <span>{error ?? busyLabel}</span>
          </div>
        )}
      </main>

      <aside className="controls">
        <section className="panel">
          <h2>1. Seeds & Picking</h2>
          <label className="field">
            <span className="field-label">
              <span>Point size</span>
              <InfoTip title="Point size">{helpText.pointSize}</InfoTip>
            </span>
            <input
              type="range"
              min={0.005}
              max={0.12}
              step={0.005}
              value={pointSize}
              onChange={(event) => setPointSize(Number(event.target.value))}
            />
          </label>
          <div className="segmented">
            <ActionButton className={pickMode === "rock" ? "active" : ""} help={buttonHelp.pickRock} onClick={() => setPickMode("rock")}>
              Rock
            </ActionButton>
            <ActionButton className={pickMode === "pedestal" ? "active" : ""} help={buttonHelp.pickPedestal} onClick={() => setPickMode("pedestal")}>
              Pedestal
            </ActionButton>
            <ActionButton className={pickMode === "interface" ? "active" : ""} help={buttonHelp.pickInterface} onClick={openInterfaceTools}>
              Interface
            </ActionButton>
          </div>
          <div className="selection-readout">
            <span>Rock {rockSeeds.length}</span>
            <span>Pedestal {pedestalSeeds.length}</span>
            <span>Interface {interfacePoints.length}</span>
          </div>
          <div className="pick-prompt">
            <span>Shift + Left Click to add a seed.</span>
            <span>Shift + Right Click to unselect the nearest selected point.</span>
          </div>
          <div className="button-row">
            <ActionButton
              disabled={!session?.status.point_cloud_loaded}
              help={buttonHelp.autoSeeds}
              disabledHelp="Upload a point cloud first."
              onClick={autoSeeds}
            >
              Auto
            </ActionButton>
            <ActionButton disabled={!session?.status.point_cloud_loaded} help={buttonHelp.clearMode} disabledHelp="Upload a point cloud first." onClick={clearCurrentPickMode}>
              Clear Mode
            </ActionButton>
          </div>
        </section>

        <section className="panel">
          <h2>2. Region Growing</h2>
          <NumericField
            label="Smoothness"
            help={helpText.smoothness}
            value={segmentParams.smoothness_threshold}
            min={0}
            max={1}
            step={0.01}
            onChange={(value) => setSegmentParams((params) => ({ ...params, smoothness_threshold: value }))}
          />
          <NumericField
            label="Curvature"
            help={helpText.curvature}
            value={segmentParams.curvature_threshold}
            min={0}
            max={1}
            step={0.01}
            onChange={(value) => setSegmentParams((params) => ({ ...params, curvature_threshold: value }))}
          />
          <NumericField
            label="Interface exclusion radius"
            help={helpText.proximity}
            value={segmentParams.basal_proximity_threshold}
            min={0}
            max={1}
            step={0.01}
            onChange={(value) => setSegmentParams((params) => ({ ...params, basal_proximity_threshold: value }))}
          />
          <NumericField
            label="Voxel"
            help={helpText.voxel}
            value={segmentParams.voxel_size}
            min={0.001}
            max={1}
            step={0.001}
            onChange={(value) => setSegmentParams((params) => ({ ...params, voxel_size: value }))}
          />
          <NumericField
            label="Neighborhood radius"
            help={helpText.distance}
            value={segmentParams.distance_threshold}
            min={0.001}
            max={1}
            step={0.001}
            onChange={(value) => setSegmentParams((params) => ({ ...params, distance_threshold: value }))}
          />
          <NumericField
            label="Normal neighbors"
            help={helpText.neighbors}
            value={segmentParams.neighbor_count}
            min={3}
            max={500}
            step={1}
            onChange={(value) => setSegmentParams((params) => ({ ...params, neighbor_count: value }))}
          />
          <ActionButton
            className="wide"
            disabled={!session?.status.seeds_ready}
            help={buttonHelp.runSegment}
            disabledHelp="Pick at least one rock seed and one pedestal seed first."
            onClick={async () => {
              if (!(await saveSeedsNow())) {
                return;
              }
              await runWorkflowAction(
                "Segmenting",
                `/api/sessions/${session?.session_id}/segment/region-growing`,
                segmentParams,
                "voxel_segmented"
              );
            }}
          >
            Run Region Growing
          </ActionButton>
          <ActionButton
            className="wide"
            disabled={!session?.status.seeds_ready || !session?.status.manual_interface_ready}
            help={buttonHelp.runICRG}
            disabledHelp={
              !session?.status.seeds_ready
                ? "Pick at least one rock seed and one pedestal seed first."
                : "Save a manual interface first."
            }
            onClick={async () => {
              if (!(await saveSeedsNow())) {
                return;
              }
              await runWorkflowAction(
                "Running ICRG",
                `/api/sessions/${session?.session_id}/segment/icrg/region-growing`,
                segmentParams,
                "voxel_segmented"
              );
            }}
          >
            Run ICRG
          </ActionButton>
        </section>

        <section className="panel">
          <h2>3. Label Propagating</h2>
          <NumericField
            label="Label propagation"
            help={helpText.labelPropagation}
            value={segmentParams.label_propagation_distance}
            min={0.001}
            max={1}
            step={0.001}
            onChange={(value) => setSegmentParams((params) => ({ ...params, label_propagation_distance: value }))}
          />
          <ActionButton
            className="wide"
            disabled={!session?.status.voxel_segmentation_ready}
            help="Complete dense distance-weighted label propagation and open the Segmented view."
            disabledHelp="Run region growing first."
            onClick={() => runWorkflowAction(
              "Running label propagation",
              `/api/sessions/${session?.session_id}/segment/label-propagation`,
              { label_propagation_distance: segmentParams.label_propagation_distance },
              "segmented"
            )}
          >
            Run Label Propagation
          </ActionButton>
        </section>

        <section className="panel">
          <h2>4. Mesh Preparation</h2>
          <div className="segmented mesh-target-control" aria-label="Mesh preparation target">
            <button className={activeMeshTarget === "rock" ? "active" : ""} type="button" onClick={() => void selectMeshTarget("rock")}>
              Prepare Rock Mesh
            </button>
            <button
              className={activeMeshTarget === "pedestal" ? "active" : ""}
              type="button"
              title="Use only pedestal/support points. Interface points are not included."
              onClick={() => void selectMeshTarget("pedestal")}
            >
              Prepare Pedestal Mesh
            </button>
          </div>
          <div className="button-row mesh-prep-actions">
            <ActionButton
              disabled={!session?.status.segmentation_ready}
              help={buttonHelp.prepareMesh}
              disabledHelp="Run segmentation first."
              onClick={async () => {
                const job = await runWorkflowAction(
                  `Preparing ${activeMeshTarget} mesh`,
                  `/api/sessions/${session?.session_id}/mesh/prepare`,
                  { target: activeMeshTarget },
                  "mesh_prepared"
                );
                if (job) {
                  clearPreparedSelectionPreviews();
                }
              }}
            >
              Run Preparation
            </ActionButton>
            <ActionButton
              disabled={!session?.status.segmentation_ready}
              help={buttonHelp.resetMeshPreparation}
              disabledHelp="Run segmentation first."
              onClick={async () => {
                const job = await runWorkflowAction(
                  `Resetting ${activeMeshTarget} preparation`,
                  `/api/sessions/${session?.session_id}/mesh/prepare`,
                  { target: activeMeshTarget, reset: true },
                  "mesh_prepared"
                );
                if (job) {
                  clearPreparedSelectionPreviews();
                }
              }}
            >
              Reset
            </ActionButton>
          </div>
          <label className="field">
            <span className="field-label">
              <span>Denoise method</span>
              <InfoTip title="Denoise method">
                SOR removes statistical outliers. DBSCAN removes small isolated clusters. Use both when floating noise remains after SOR.
              </InfoTip>
            </span>
            <select
              value={denoiseParams.method}
              onChange={(event) => setDenoiseParams((params) => ({ ...params, method: event.target.value as DenoiseParams["method"] }))}
            >
              <option value="sor">SOR filter</option>
              <option value="dbscan">DBSCAN</option>
              <option value="sor_dbscan">SOR + DBSCAN</option>
            </select>
          </label>
          <div className="inline-fields">
            <NumericField
              label="SOR k"
              help="Number of neighbors for statistical outlier removal. Increase for sparse or noisy scans; decrease for small, detailed scans."
              value={denoiseParams.sor_neighbors}
              min={3}
              max={500}
              step={1}
              onChange={(value) => setDenoiseParams((params) => ({ ...params, sor_neighbors: value }))}
            />
            <NumericField
              label="SOR std"
              help="Lower values remove more points; higher values preserve more points. The refactored desktop default is 2.0."
              value={denoiseParams.sor_std_ratio}
              min={0.1}
              max={10}
              step={0.1}
              onChange={(value) => setDenoiseParams((params) => ({ ...params, sor_std_ratio: value }))}
            />
          </div>
          <div className="inline-fields">
            <NumericField
              label="DBSCAN eps (m)"
              help="Maximum distance for DBSCAN cluster membership. The refactored desktop default is 0.02 m."
              value={denoiseParams.dbscan_eps}
              min={0.001}
              max={0.5}
              step={0.001}
              onChange={(value) => setDenoiseParams((params) => ({ ...params, dbscan_eps: value }))}
            />
            <NumericField
              label="DBSCAN min"
              help="Minimum cluster size for DBSCAN core neighborhoods. The refactored desktop default is 20 points."
              value={denoiseParams.dbscan_min_points}
              min={1}
              max={1000}
              step={1}
              onChange={(value) => setDenoiseParams((params) => ({ ...params, dbscan_min_points: value }))}
            />
          </div>
          <div className="button-row">
            <ActionButton
              disabled={!activeMeshTargetPrepared}
              help={buttonHelp.removeNoise}
              disabledHelp={`Prepare the ${activeMeshTarget} mesh point set first.`}
              onClick={async () => {
                const job = await runWorkflowAction(
                  `Denoising ${activeMeshTarget}`,
                  `/api/sessions/${session?.session_id}/mesh/noise/remove`,
                  { ...denoiseParams, target: activeMeshTarget },
                  "mesh_prepared"
                );
                if (job) {
                  clearPreparedSelectionPreviews();
                }
              }}
            >
              Denoise
            </ActionButton>
            <ActionButton
              disabled={!activeMeshTargetSaved}
              help={buttonHelp.undoNoise}
              disabledHelp={`Prepare the ${activeMeshTarget} mesh point set first.`}
              onClick={async () => {
                const job = await runWorkflowAction(
                  `Undo ${activeMeshTarget} noise`,
                  `/api/sessions/${session?.session_id}/mesh/noise/undo`,
                  { target: activeMeshTarget },
                  "mesh_prepared"
                );
                if (job) {
                  clearPreparedSelectionPreviews();
                }
              }}
            >
              Undo Denoise
            </ActionButton>
          </div>
          <ActionButton
            className="wide"
            disabled={!activeMeshTargetPrepared || activeMeshTarget !== "pedestal"}
            help={buttonHelp.hagVegetation}
            disabledHelp={activeMeshTarget === "rock" ? "Height Above Ground removal is only available for prepared pedestal mesh." : "Prepare the pedestal mesh point set first."}
            onClick={openVegetationTool}
          >
            Height Above Ground
          </ActionButton>
          <ActionButton
            className="wide"
            disabled={!activeMeshTargetPrepared || activeMeshTarget !== "pedestal"}
            help={buttonHelp.roughnessRemoval}
            disabledHelp={activeMeshTarget === "rock" ? "Roughness removal is only available for prepared pedestal mesh." : "Prepare the pedestal mesh point set first."}
            onClick={openRoughnessTool}
          >
            Roughness
          </ActionButton>
          <ActionButton
            className="wide"
            disabled={!activeMeshTargetPrepared}
            help={buttonHelp.manualRemoval}
            disabledHelp={`Prepare the ${activeMeshTarget} mesh point set first.`}
            onClick={openManualRemovalTools}
          >
            Manual Removal
          </ActionButton>
        </section>

        <section className="panel">
          <h2>5. Normals</h2>
          <div className="segmented mesh-target-control" aria-label="Normals target">
            <button className={activeMeshTarget === "rock" ? "active" : ""} type="button" onClick={() => void selectMeshTarget("rock")}>
              Rock Normals
            </button>
            <button className={activeMeshTarget === "pedestal" ? "active" : ""} type="button" onClick={() => void selectMeshTarget("pedestal")}>
              Pedestal Normals
            </button>
          </div>
          <label className="field">
            <span className="field-label">
              <span>Normal method</span>
              <InfoTip title="Normal method">{helpText.normalMethod}</InfoTip>
            </span>
            <select value={normalMethod} onChange={(event) => setNormalMethod(event.target.value as "pymeshlab" | "open3d")}>
              <option value="pymeshlab">PyMeshLab</option>
              <option value="open3d">Open3D</option>
            </select>
          </label>
          <NumericField label="Normal k" help={helpText.normalK} value={normalK} min={3} max={1000} step={1} onChange={setNormalK} />
          <SliderField
            label="Normal length"
            help="Changes only how long the normal vectors appear in Mesh Prep view. It does not change reconstruction or saved outputs."
            value={normalDisplayScale}
            min={0.1}
            max={10}
            step={0.1}
            onChange={setNormalDisplayScale}
          />
          <ActionButton
            className="wide"
            disabled={!activeMeshTargetPrepared}
            help={buttonHelp.computeNormals}
            disabledHelp={`Prepare the ${activeMeshTarget} mesh point set first.`}
            onClick={() =>
              runWorkflowAction(
                "Computing normals",
                `/api/sessions/${session?.session_id}/mesh/normals`,
                { method: normalMethod, k: normalK, target: activeMeshTarget },
                "mesh_prepared"
              )
            }
          >
            Compute Normals
          </ActionButton>
        </section>

        <section className="panel">
          <h2>6. Reconstruction</h2>
          <div className="segmented mesh-target-control" aria-label="Reconstruction target">
            <button className={activeMeshTarget === "rock" ? "active" : ""} type="button" onClick={() => void selectMeshTarget("rock")}>
              Rock
            </button>
            <button className={activeMeshTarget === "pedestal" ? "active" : ""} type="button" onClick={() => void selectMeshTarget("pedestal")}>
              Pedestal
            </button>
          </div>
          <NumericField
            label="Depth"
            help={activeMeshTarget === "pedestal" ? "Poisson depth is only used for rock reconstruction. Pedestal uses local-plane filled-hole surface reconstruction." : helpText.depth}
            value={meshDepth}
            min={5}
            max={12}
            step={1}
            disabled={activeMeshTarget === "pedestal"}
            onChange={setMeshDepth}
          />
          <ActionButton
            className="wide"
            disabled={!activeMeshTargetSaved}
            help={buttonHelp.reconstruct}
            disabledHelp={`Prepare the ${activeMeshTarget} mesh point set first.`}
            onClick={() =>
              runWorkflowAction(
                activeMeshTarget === "pedestal" ? "Reconstructing pedestal surface" : "Reconstructing",
                `/api/sessions/${session?.session_id}/mesh/reconstruct`,
                { depth: meshDepth, target: activeMeshTarget },
                "mesh"
              )
            }
          >
            {activeMeshTarget === "pedestal" ? "Reconstruct Surface" : "Reconstruct"}
          </ActionButton>
          <ActionButton
            className="wide"
            disabled={!session?.combined_reconstruction?.available}
            help={buttonHelp.loadRockPedestal}
            disabledHelp="Requires rock and pedestal sources. Each side needs either a reconstructed mesh or segmented point-cloud points."
            onClick={() => refreshView("combined_mesh")}
          >
            Load Rock + Pedestal
          </ActionButton>
        </section>

        <section className="panel">
          <h2>7. Analysis</h2>
          <ActionButton
            className="wide"
            disabled={!session?.status.mesh_completed || activeMeshTarget !== "rock"}
            help={buttonHelp.analyze}
            disabledHelp={activeMeshTarget === "pedestal" ? "Analysis is rock-only in this version." : "Reconstruct the mesh first."}
            onClick={() => runWorkflowAction("Analyzing", `/api/sessions/${session?.session_id}/analysis`, undefined, "analysis")}
          >
            Analyze
          </ActionButton>
        </section>
      </aside>

      {interfaceWindowOpen && (
        <div className="floating-window interface-window" role="dialog" aria-labelledby="interfaceWindowTitle" style={interfaceWindowStyle}>
          <div
            className="floating-window-header"
            onPointerDown={beginInterfaceWindowDrag}
            onPointerMove={dragInterfaceWindow}
            onPointerUp={endInterfaceWindowDrag}
            onPointerCancel={endInterfaceWindowDrag}
          >
            <h2 id="interfaceWindowTitle">Interface</h2>
            <button className="floating-window-close" type="button" aria-label="Close interface window" onClick={() => setInterfaceWindowOpen(false)}>
              x
            </button>
          </div>
          <div className="floating-window-body interface-window-body">
            <section className="interface-panel-section">
              <div className="section-title-row">
                <h3>Manual Interface</h3>
                <div className="selection-readout inline-readout">
                  <span>Parts {interfaceParts.length}</span>
                  <ActionButton className="link-button" help={buttonHelp.clearParts} onClick={clearInterfaceParts}>
                    Clear
                  </ActionButton>
                </div>
              </div>
              <div className="interface-options-grid">
                <label className="check-field">
                  <input
                    type="checkbox"
                    checked={currentPartLateral}
                    onChange={(event) => setCurrentPartLateral(event.target.checked)}
                  />
                  <span className="field-label">
                    <span>Lateral part</span>
                    <InfoTip title="Lateral part">{helpText.lateral}</InfoTip>
                  </span>
                </label>
                <label className="check-field">
                  <input type="checkbox" checked={closeLoop} onChange={(event) => setCloseLoop(event.target.checked)} />
                  <span className="field-label">
                    <span>Close loop</span>
                    <InfoTip title="Close loop">{helpText.closeLoop}</InfoTip>
                  </span>
                </label>
              </div>
              <div className="button-row">
                <ActionButton disabled={interfacePoints.length < 2} help={buttonHelp.stagePart} disabledHelp="Pick at least two interface points first." onClick={stageInterfacePart}>
                  Stage Part
                </ActionButton>
                <ActionButton disabled={!session?.status.point_cloud_loaded} help={buttonHelp.interpolateInterface} disabledHelp="Upload a point cloud first." onClick={interpolateInterface}>
                  Interpolate Path
                </ActionButton>
              </div>
              <ActionButton className="wide" disabled={!session?.status.point_cloud_loaded} help={buttonHelp.saveInterface} disabledHelp="Upload a point cloud first." onClick={saveInterface}>
                Save Interface
              </ActionButton>
            </section>

            <section className="interface-panel-section hybrid-edit-section">
              <div className="section-title-row">
                <h3>Hybrid Interface Edit</h3>
              </div>
              <p className="tool-note">Start from an existing interface, edit it as a temporary draft, then save it as the manual interface for ICRG.</p>
              {!interfaceEditorOpen && (
                <div className="source-choice">
                  <div className="step-label">Start draft from</div>
                  <div className="button-row">
                    <ActionButton
                      disabled={!session?.status.auto_interface_ready}
                      disabledHelp="Run regular region growing first."
                      help="Create an editable draft from the latest automatic interface."
                      onClick={() => openInterfaceEditorForSource("auto")}
                    >
                      Start From Auto
                    </ActionButton>
                    <ActionButton
                      disabled={!session?.status.manual_interface_ready}
                      disabledHelp="Save a manual interface first."
                      help="Create an editable draft from the saved manual interface."
                      onClick={() => openInterfaceEditorForSource("manual")}
                    >
                      Start From Manual
                    </ActionButton>
                  </div>
                </div>
              )}
              {interfaceEditorOpen && (
                <div className="interface-editor-panel">
                  <div className="editor-step">
                    <div className="step-label">Segment points</div>
                    <div className="selection-readout">
                      <span>{interfaceDraftSegments.pointsText}</span>
                    </div>
                  </div>
                  <div className="button-row">
                    <ActionButton disabled={!interfaceDraft} help={buttonHelp.undoDraft} disabledHelp="Create a draft first." onClick={undoInterfaceDraftEdit}>
                      Undo Edit
                    </ActionButton>
                    <ActionButton disabled={!interfaceDraft} help={buttonHelp.saveDraftManual} disabledHelp="Create a draft first." onClick={saveDraftAsManualInterface}>
                      Save as Manual
                    </ActionButton>
                  </div>
                  <ActionButton className="wide" disabled={!interfaceDraft} help={buttonHelp.clearDraft} disabledHelp="Create a draft first." onClick={discardInterfaceDraft}>
                    Quit Editing
                  </ActionButton>
                </div>
              )}
            </section>
          </div>
        </div>
      )}

      {vegetationWindowOpen && (
        <div className="floating-window vegetation-window" role="dialog" aria-labelledby="vegetationWindowTitle">
          <div className="floating-window-header">
            <h2 id="vegetationWindowTitle">Height Above Ground</h2>
            <button className="floating-window-close" type="button" aria-label="Close vegetation selection window" onClick={closeVegetationTool}>
              x
            </button>
          </div>
          <div className="floating-window-body">
            <p className="tool-note">Apply previews likely vegetation in yellow. Adjust parameters and apply again until the selection looks right, then confirm removal.</p>
            <div className="inline-fields">
              <NumericField
                label="Grid size"
                help="XY cell size used to estimate local ground elevation. Increase for sparse scans; decrease for denser or uneven ground."
                value={hagVegetationParams.grid_size}
                min={0.001}
                max={5}
                step={0.001}
                onChange={(value) => setHagVegetationParams((params) => ({ ...params, grid_size: value }))}
              />
              <NumericField
                label="Height threshold"
                help="Points this far above the local ground estimate are selected as vegetation candidates. Increase to select only taller vegetation."
                value={hagVegetationParams.height_threshold}
                min={0}
                max={5}
                step={0.001}
                onChange={(value) => setHagVegetationParams((params) => ({ ...params, height_threshold: value }))}
              />
            </div>
            <div className="inline-fields">
              <NumericField
                label="Ground percentile"
                help="Low Z percentile in each grid cell used as the local ground estimate. Lower values follow the lowest points more aggressively."
                value={hagVegetationParams.ground_percentile}
                min={0}
                max={100}
                step={1}
                onChange={(value) => setHagVegetationParams((params) => ({ ...params, ground_percentile: value }))}
              />
              <NumericField
                label="Min points"
                help="A cell needs at least this many points to estimate ground directly. Sparse cells borrow the nearest valid ground cell."
                value={hagVegetationParams.min_points_per_cell}
                min={1}
                max={1000}
                step={1}
                onChange={(value) => setHagVegetationParams((params) => ({ ...params, min_points_per_cell: value }))}
              />
            </div>
            <div className="button-row">
              <ActionButton help={buttonHelp.hagApply} onClick={applyHagVegetationSelection}>
                Apply
              </ActionButton>
              <ActionButton disabled={!vegetationSelected.length} help={buttonHelp.hagConfirm} disabledHelp="Apply Height Above Ground first." onClick={confirmHagVegetationRemoval}>
                Confirm Removal
              </ActionButton>
            </div>
            <div className="button-row">
              <ActionButton disabled={!vegetationSelected.length} help={buttonHelp.hagClear} disabledHelp="No preview selection is active." onClick={() => setVegetationSelected([])}>
                Clear
              </ActionButton>
              <ActionButton help={buttonHelp.closeManualRemoval} onClick={closeVegetationTool}>
                Close
              </ActionButton>
            </div>
            <div className="selection-readout">
              <span>{vegetationSelected.length.toLocaleString()} vegetation candidates selected</span>
            </div>
          </div>
        </div>
      )}

      {roughnessWindowOpen && (
        <div className="floating-window roughness-window" role="dialog" aria-labelledby="roughnessWindowTitle">
          <div className="floating-window-header">
            <h2 id="roughnessWindowTitle">Roughness</h2>
            <button className="floating-window-close" type="button" aria-label="Close roughness selection window" onClick={closeRoughnessTool}>
              x
            </button>
          </div>
          <div className="floating-window-body">
            <p className="tool-note">Calculate builds a local plane roughness heatmap from the radius. Apply uses the threshold on the current heatmap without recalculating.</p>
            <div className="inline-fields">
              <div className="field-with-action">
                <NumericField
                  label="Radius"
                  help="Spherical neighborhood radius used to fit a local least-squares plane around each pedestal point. Larger radii smooth over broader ground trends; smaller radii react to local texture and noise."
                  value={roughnessParams.radius}
                  min={0.001}
                  max={5}
                  step={0.001}
                  onChange={(value) => setRoughnessParams((params) => ({ ...params, radius: value }))}
                />
                <ActionButton help={buttonHelp.roughnessCalculate} onClick={calculateRoughnessHeatmap}>
                  Calculate
                </ActionButton>
              </div>
              <NumericField
                label="Threshold"
                help="Prepared pedestal points with point-to-local-plane roughness above this value are selected for removal. Increase to select only rougher points."
                value={roughnessParams.threshold}
                min={0}
                max={5}
                step={0.001}
                onChange={(value) => setRoughnessParams((params) => ({ ...params, threshold: value }))}
              />
            </div>
            <div className="button-row">
              <ActionButton disabled={!roughnessValues.length} disabledHelp="Calculate roughness first." help={buttonHelp.roughnessApply} onClick={applyRoughnessThreshold}>
                Apply
              </ActionButton>
              <ActionButton disabled={!roughnessSelected.length} help={buttonHelp.roughnessConfirm} disabledHelp="Apply Roughness first." onClick={confirmRoughnessRemoval}>
                Confirm Removal
              </ActionButton>
            </div>
            <div className="button-row">
              <ActionButton
                disabled={!roughnessSelected.length}
                help={buttonHelp.roughnessClear}
                disabledHelp="No preview selection is active."
                onClick={() => {
                  setRoughnessSelected([]);
                }}
              >
                Clear
              </ActionButton>
              <ActionButton help={buttonHelp.closeManualRemoval} onClick={closeRoughnessTool}>
                Close
              </ActionButton>
            </div>
            <div className="selection-readout">
              <span>{roughnessSelected.length.toLocaleString()} roughness candidates selected</span>
            </div>
            <div className="selection-readout">
              <span>
                {roughnessStats
                  ? `Heatmap ready: ${Number(roughnessStats.voxel_point_count ?? 0).toLocaleString()} voxel points at ${Number(roughnessStats.voxel_size ?? 0).toFixed(4)} m, max ${Number(roughnessStats.max_roughness ?? 0).toFixed(4)} m, mean ${Number(roughnessStats.mean_roughness ?? 0).toFixed(4)} m`
                  : "Click Calculate to build the roughness heatmap"}
              </span>
            </div>
          </div>
        </div>
      )}

      {manualRemovalOpen && (
        <div className="floating-window manual-removal-window" role="dialog" aria-labelledby="manualRemovalWindowTitle" style={manualRemovalWindowStyle}>
          <div
            className="floating-window-header"
            onPointerDown={beginManualRemovalWindowDrag}
            onPointerMove={dragManualRemovalWindow}
            onPointerUp={endManualRemovalWindowDrag}
            onPointerCancel={endManualRemovalWindowDrag}
          >
            <h2 id="manualRemovalWindowTitle">Manual Removal</h2>
            <button className="floating-window-close" type="button" aria-label="Close manual removal window" onClick={() => setManualRemovalOpen(false)}>
              x
            </button>
          </div>
          <div className="floating-window-body">
            <p className="tool-note">Draw around noise in Mesh Prep. All prepared points projected inside the polygon are removed, including points hidden behind the front surface.</p>
            <div className="button-row">
              <ActionButton className={manualRemovalDrawing ? "active" : ""} help={buttonHelp.drawPolygon} onClick={() => setManualRemovalDrawing((drawing) => !drawing)}>
                {manualRemovalDrawing ? "Drawing Polygon" : "Draw Polygon"}
              </ActionButton>
              <ActionButton
                disabled={!manualRemovalPolygon.length}
                help={buttonHelp.undoVertex}
                disabledHelp="Add at least one polygon vertex first."
                onClick={() => setManualRemovalPolygon((polygon) => polygon.slice(0, -1))}
              >
                Undo Vertex
              </ActionButton>
            </div>
            <div className="button-row">
              <ActionButton
                disabled={!manualRemovalPolygon.length && !manualRemovalSelected.length}
                help={buttonHelp.clearManualRemoval}
                disabledHelp="No polygon or preview selection is active."
                onClick={clearManualRemoval}
              >
                Clear
              </ActionButton>
              <ActionButton
                disabled={!manualRemovalSelected.length}
                help={buttonHelp.removeSelected}
                disabledHelp="Draw a polygon that selects visible prepared points first."
                onClick={applyManualRemoval}
              >
                Remove Selected
              </ActionButton>
            </div>
            <div className="selection-readout">
              <span>{manualRemovalSelected.length.toLocaleString()} selected</span>
              <span>{manualRemovalPolygon.length} vertices</span>
            </div>
            <ActionButton className="wide" help={buttonHelp.closeManualRemoval} onClick={() => setManualRemovalOpen(false)}>
              Close
            </ActionButton>
          </div>
        </div>
      )}
    </div>
  );
}
