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
  Activity,
  Check,
  Circle,
  Download,
  FileUp,
  Info,
  Layers3,
  Loader2,
  Pickaxe,
  Play,
  RotateCcw,
  Save,
  Sparkles,
  Triangle
} from "lucide-react";
import {
  createSession,
  downloadUrl,
  getJob,
  getSession,
  getViewer,
  runJob,
  uploadPointCloud,
  type DenoiseParams,
  type JobResponse,
  type SegmentParams,
  type SessionSummary,
  type ViewerPayload
} from "./api";
import { PointCloudViewer } from "./PointCloudViewer";

type PickMode = "rock" | "pedestal" | "interface";
type ViewName = "raw" | "seeds" | "interface" | "segmented" | "mesh_prepared" | "mesh";

type InterfacePartDraft = {
  selected_indices: number[];
  is_lateral: boolean;
};

type ScreenPoint = {
  x: number;
  y: number;
};

const defaultSegmentParams: SegmentParams = {
  smoothness_threshold: 0.9,
  curvature_threshold: 0.1,
  basal_proximity_threshold: 0.05,
  voxel_size: 0.02,
  neighbor_count: 50,
  distance_threshold: 0.05
};

const defaultDenoiseParams: DenoiseParams = {
  method: "sor",
  sor_neighbors: 10,
  sor_std_ratio: 2.0,
  dbscan_eps: 0.02,
  dbscan_min_points: 20
};

const helpText = {
  pointSize:
    "Changes only the rendered point size. Increase it for sparse or distant clouds and decrease it when dense clouds look blotchy; segmentation and exports are unchanged.",
  lateral:
    "Use this when selected interface points trace contact with an adjacent rock or side support instead of the basal pedestal. Basal and lateral parts are handled separately; basal labels feed the contact-geometry analysis.",
  closeLoop:
    "Keep this on when your interface points outline a closed contact boundary. Turn it off for an open contact edge or when staging separate basal and lateral interface parts.",
  smoothness:
    "Default 0.9. Higher values require more parallel normals and can reduce leakage across the rock-support contact. Lower values help rough or noisy rock surfaces grow, but can spill into support.",
  curvature:
    "Default 0.1. Lower values are stricter and favor smooth local continuity. Raise it when rough rock surfaces are being missed; lower it if growth crosses sharp contact changes.",
  proximity:
    "The manuscript default is 0.02 m; this UI starts at 0.05 m. Increase if labels cross the contact. Decrease if too much rock near the interface stays unlabeled.",
  voxel:
    "Default 0.02 m in the manuscript for faster preprocessing. Smaller values preserve contact detail but run slower; larger values smooth noisy data but can erase small interface features.",
  neighbors:
    "Controls how many nearby points support local geometry tests. Increase for sparse or noisy scans to stabilize normals; decrease to keep small contact details from being blurred.",
  distance:
    "Default radius is 0.05 m. Increase when the cloud is sparse or growth stalls across small gaps. Decrease to avoid jumps across narrow contacts or nearby supports.",
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
  segmented: "Show the latest rock/support labels after running interface-constrained region growing.",
  mesh_prepared: "Show the prepared point set used for normal estimation and mesh reconstruction.",
  mesh: "Show mesh status after reconstruction; download the PLY from the Downloads panel."
};

const buttonHelp = {
  pickRock: "Shift + left click adds rock seed points. Shift + right click near a selected point removes it.",
  pickPedestal: "Shift + left click adds support or pedestal seed points. Shift + right click near a selected point removes it.",
  pickInterface: "Shift + left click adds interface contact points. Shift + right click near a selected point removes it.",
  autoSeeds: "Choose default rock and support seeds from the current point cloud geometry.",
  clearMode: "Clear only the current pick mode: rock seeds, pedestal seeds, or current interface points.",
  stagePart: "Store the current interface picks as one contact segment so you can pick another segment.",
  interpolateInterface: "Preview the dense interface path before saving it for segmentation.",
  saveInterface: "Finalize the interpolated interface constraints for region growing.",
  clearParts: "Remove staged interface parts and current interface picks.",
  runSegment: "Segment rock and support using saved seeds, interface constraints, and the current parameter values.",
  prepareMesh: "Create the prepared rock point set used for normals, reconstruction, and analysis.",
  removeNoise: "Run the selected denoise method: SOR, DBSCAN, or SOR followed by DBSCAN.",
  undoNoise: "Restore the prepared mesh point cloud to the state before the last denoise step.",
  manualRemoval: "Draw a screen-space polygon in Mesh Prep view and remove selected prepared rock or interpolated bottom-face points.",
  drawPolygon: "Add polygon vertices with left clicks in the viewer. The preview updates after three vertices.",
  undoVertex: "Remove the most recent polygon vertex and update the preview.",
  clearManualRemoval: "Clear the polygon and selected preview points without changing the prepared mesh.",
  removeSelected: "Remove selected prepared rock or interpolated bottom-face points and add this edit to the denoise undo history.",
  closeManualRemoval: "Close the manual-removal tool without changing the prepared mesh.",
  analyze: "Compute geometric metrics and make the analysis CSV available for download.",
  computeNormals: "Estimate and orient normals with PyMeshLab or Open3D before Poisson mesh reconstruction.",
  reconstruct: "Run Poisson reconstruction and create the downloadable mesh PLY."
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

function StatusRow({ done, label }: { done: boolean; label: string }) {
  return (
    <div className={`status-row ${done ? "done" : ""}`}>
      {done ? <Check size={16} /> : <Circle size={16} />}
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
      <Info size={13} />
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
        <span>{label}</span>
        {help ? <InfoTip title={label}>{help}</InfoTip> : null}
      </span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
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
  const [hoverTipsEnabled, setHoverTipsEnabled] = useState(true);
  const [interfaceWindowOpen, setInterfaceWindowOpen] = useState(false);
  const [interfaceWindowPosition, setInterfaceWindowPosition] = useState<{ left: number; top: number } | null>(null);
  const interfaceWindowDragRef = useRef<{ offsetX: number; offsetY: number } | null>(null);
  const [manualRemovalOpen, setManualRemovalOpen] = useState(false);
  const [manualRemovalDrawing, setManualRemovalDrawing] = useState(false);
  const [manualRemovalPolygon, setManualRemovalPolygon] = useState<ScreenPoint[]>([]);
  const [manualRemovalSelected, setManualRemovalSelected] = useState<number[]>([]);
  const [manualRemovalWindowPosition, setManualRemovalWindowPosition] = useState<{ left: number; top: number } | null>(null);
  const manualRemovalWindowDragRef = useRef<{ offsetX: number; offsetY: number } | null>(null);
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
    if (view.kind === "mesh") {
      return [
        view.vertices?.length ? `${view.vertices.length.toLocaleString()} vertices` : null,
        view.triangles?.length ? `${view.triangles.length.toLocaleString()} faces` : null
      ].filter(Boolean);
    }
    return [
      `${view.rendered_points.toLocaleString()} shown`,
      view.normal_segments ? `${view.normal_segments.length.toLocaleString()} normal arrows` : null,
      view.normal_diagnostics ? `${view.normal_diagnostics.nonzero_normal_count.toLocaleString()} valid normals` : null
    ].filter(Boolean);
  }, [view]);

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
    async (viewName: ViewName, summary?: SessionSummary | null) => {
      const targetSession = summary ?? session;
      if (!targetSession) {
        return;
      }
      const payload = await getViewer(targetSession.session_id, viewName);
      setView(payload);
      setActiveView(viewName);
    },
    [session]
  );

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
      setCurrentPartLateral(false);
      clearManualRemoval();
      await refreshView("raw", summary);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusyLabel(null);
    }
  }

  function handlePickPoint(index: number) {
    if (pickMode === "rock") {
      setRockSeeds((items) => addIndex(items, index));
    } else if (pickMode === "pedestal") {
      setPedestalSeeds((items) => addIndex(items, index));
    } else {
      setInterfacePoints((items) => addIndex(items, index));
    }
  }

  function handleUnpickPoint(index: number) {
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

  function clearManualRemoval() {
    setManualRemovalDrawing(false);
    setManualRemovalPolygon([]);
    setManualRemovalSelected([]);
  }

  async function openManualRemovalTools() {
    if (!session?.status.mesh_prepared) {
      setError("Prepare the mesh before manual removal.");
      return;
    }
    setManualRemovalOpen(true);
    if (activeView !== "mesh_prepared") {
      await refreshView("mesh_prepared");
    }
  }

  async function applyManualRemoval() {
    if (!session || !manualRemovalSelected.length) {
      setError("Draw a polygon that selects at least one visible rock point.");
      return;
    }
    const job = await runWorkflowAction(
      "Manual removal",
      `/api/sessions/${session.session_id}/mesh/noise/manual-remove`,
      { selected_indices: manualRemovalSelected },
      "mesh_prepared"
    );
    if (job) {
      clearManualRemoval();
      setManualRemovalOpen(true);
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

  const canDownload = session?.outputs ?? { segmented: null, mesh: null, analysis: null };
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
          <Triangle size={26} />
          <div>
            <h1>Rock Detection 3D</h1>
            <p>{session?.current_file ?? "No point cloud loaded"}</p>
          </div>
        </div>

        <label className="upload-button">
          <FileUp size={18} />
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
          <StatusRow done={Boolean(session?.status.mesh_completed)} label="Mesh" />
          <StatusRow done={Boolean(session?.status.analysis_completed)} label="Analysis" />
        </section>

        <section className="panel compact">
          <h2>Views</h2>
          <div className="view-grid">
            {(["raw", "seeds", "interface", "segmented", "mesh_prepared", "mesh"] as ViewName[]).map((name) => (
              <ActionButton
                key={name}
                className={activeView === name ? "active" : ""}
                disabled={!session?.status.point_cloud_loaded}
                help={viewHelp[name]}
                disabledHelp="Upload a point cloud first."
                onClick={() => refreshView(name)}
              >
                {name.replace("_", " ")}
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
            <Download size={16} /> Segmented LAS
          </a>
          <a className={!canDownload.mesh ? "disabled-link" : ""} href={canDownload.mesh && session ? downloadUrl(session.session_id, "mesh") : undefined}>
            <Download size={16} /> Mesh PLY
          </a>
          <a className={!canDownload.analysis ? "disabled-link" : ""} href={canDownload.analysis && session ? downloadUrl(session.session_id, "analysis") : undefined}>
            <Download size={16} /> Analysis CSV
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
          pointSize={pointSize}
          normalDisplayScale={normalDisplayScale}
          manualRemoval={{
            active: manualRemovalOpen && activeView === "mesh_prepared",
            drawing: manualRemovalDrawing,
            polygon: manualRemovalPolygon,
            selectedIndices: manualRemovalSelected,
            onAddVertex: (point) => setManualRemovalPolygon((polygon) => [...polygon, point]),
            onSelectionChange: setManualRemovalSelected
          }}
        />

        {(busyLabel || error) && (
          <div className={`toast ${error ? "error" : ""}`}>
            {busyLabel && <Loader2 className="spin" size={16} />}
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
              <Sparkles size={16} /> Auto
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
            label="Basal proximity"
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
            label="Neighbors"
            help={helpText.neighbors}
            value={segmentParams.neighbor_count}
            min={3}
            max={500}
            step={1}
            onChange={(value) => setSegmentParams((params) => ({ ...params, neighbor_count: value }))}
          />
          <NumericField
            label="Distance"
            help={helpText.distance}
            value={segmentParams.distance_threshold}
            min={0.001}
            max={1}
            step={0.001}
            onChange={(value) => setSegmentParams((params) => ({ ...params, distance_threshold: value }))}
          />
          <ActionButton
            className="wide"
            disabled={!session?.status.seeds_ready}
            help={buttonHelp.runSegment}
            disabledHelp="Pick at least one rock seed and one pedestal seed first."
            onClick={() =>
              runWorkflowAction(
                "Segmenting",
                `/api/sessions/${session?.session_id}/segment`,
                segmentParams,
                (result) => {
                  const payload = result as { auto_interface_generated?: boolean } | undefined;
                  return payload?.auto_interface_generated ? "interface" : "segmented";
                }
              )
            }
          >
            <Play size={16} /> Run Region Growing
          </ActionButton>
        </section>

        <section className="panel">
          <h2>3. Mesh Preparation</h2>
          <ActionButton
            className="wide"
            disabled={!session?.status.segmentation_ready}
            help={buttonHelp.prepareMesh}
            disabledHelp="Run segmentation first."
            onClick={async () => {
              const job = await runWorkflowAction("Preparing mesh", `/api/sessions/${session?.session_id}/mesh/prepare`, undefined, "mesh_prepared");
              if (job) {
                clearManualRemoval();
              }
            }}
          >
            <Pickaxe size={16} /> Prepare Mesh
          </ActionButton>
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
              disabled={!session?.status.mesh_prepared}
              help={buttonHelp.removeNoise}
              disabledHelp="Prepare the mesh point set first."
              onClick={async () => {
                const job = await runWorkflowAction("Denoising", `/api/sessions/${session?.session_id}/mesh/noise/remove`, denoiseParams, "mesh_prepared");
                if (job) {
                  clearManualRemoval();
                }
              }}
            >
              <Activity size={16} /> Denoise
            </ActionButton>
            <ActionButton
              disabled={!session?.status.mesh_prepared}
              help={buttonHelp.undoNoise}
              disabledHelp="Prepare the mesh point set first."
              onClick={async () => {
                const job = await runWorkflowAction("Undo noise", `/api/sessions/${session?.session_id}/mesh/noise/undo`, undefined, "mesh_prepared");
                if (job) {
                  clearManualRemoval();
                }
              }}
            >
              <RotateCcw size={16} /> Undo Denoise
            </ActionButton>
          </div>
          <ActionButton
            className="wide"
            disabled={!session?.status.mesh_prepared}
            help={buttonHelp.manualRemoval}
            disabledHelp="Prepare the mesh point set first."
            onClick={openManualRemovalTools}
          >
            Manual Removal
          </ActionButton>
        </section>

        <section className="panel">
          <h2>4. Normals</h2>
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
            disabled={!session?.status.mesh_prepared}
            help={buttonHelp.computeNormals}
            disabledHelp="Prepare the mesh point set first."
            onClick={() =>
              runWorkflowAction(
                "Computing normals",
                `/api/sessions/${session?.session_id}/mesh/normals`,
                { method: normalMethod, k: normalK },
                "mesh_prepared"
              )
            }
          >
            <Activity size={16} /> Compute Normals
          </ActionButton>
        </section>

        <section className="panel">
          <h2>5. Reconstruction</h2>
          <NumericField label="Depth" help={helpText.depth} value={meshDepth} min={5} max={12} step={1} onChange={setMeshDepth} />
          <ActionButton
            className="wide"
            disabled={!session?.status.mesh_prepared}
            help={buttonHelp.reconstruct}
            disabledHelp="Prepare the mesh point set first."
            onClick={() =>
              runWorkflowAction(
                "Reconstructing",
                `/api/sessions/${session?.session_id}/mesh/reconstruct`,
                { depth: meshDepth },
                "mesh"
              )
            }
          >
            <Layers3 size={16} /> Reconstruct
          </ActionButton>
        </section>

        <section className="panel">
          <h2>6. Analysis</h2>
          <ActionButton
            className="wide"
            disabled={!session?.status.mesh_completed}
            help={buttonHelp.analyze}
            disabledHelp="Reconstruct the mesh first."
            onClick={() => runWorkflowAction("Analyzing", `/api/sessions/${session?.session_id}/analysis`)}
          >
            <Triangle size={16} /> Analyze
          </ActionButton>
        </section>
      </aside>

      {interfaceWindowOpen && (
        <div className="floating-window" role="dialog" aria-labelledby="interfaceWindowTitle" style={interfaceWindowStyle}>
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
          <div className="floating-window-body">
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
            <div className="button-row">
              <ActionButton disabled={interfacePoints.length < 2} help={buttonHelp.stagePart} disabledHelp="Pick at least two interface points first." onClick={stageInterfacePart}>
                <Layers3 size={16} /> Stage Part
              </ActionButton>
              <ActionButton disabled={!session?.status.point_cloud_loaded} help={buttonHelp.interpolateInterface} disabledHelp="Upload a point cloud first." onClick={interpolateInterface}>
                <Sparkles size={16} /> Interpolate Path
              </ActionButton>
            </div>
            <ActionButton className="wide" disabled={!session?.status.point_cloud_loaded} help={buttonHelp.saveInterface} disabledHelp="Upload a point cloud first." onClick={saveInterface}>
              <Save size={16} /> Save Interface
            </ActionButton>
            <div className="selection-readout">
              <span>Parts {interfaceParts.length}</span>
              <ActionButton className="link-button" help={buttonHelp.clearParts} onClick={clearInterfaceParts}>
                Clear
              </ActionButton>
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
            <p className="tool-note">Draw around visible noise in Mesh Prep. Front rendered rock and interpolated bottom-face points can be removed.</p>
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
