export type WorkflowStatus = {
  point_cloud_loaded: boolean;
  seeds_ready: boolean;
  manual_interface_ready: boolean;
  auto_interface_ready: boolean;
  interface_draft_ready: boolean;
  interface_ready: boolean;
  segmentation_ready: boolean;
  voxel_segmentation_ready?: boolean;
  mesh_prepared: boolean;
  mesh_completed: boolean;
  analysis_completed: boolean;
  last_segmentation_mode?: "rg" | "icrg" | null;
};

export type MeshTarget = "rock" | "pedestal";

export type MeshTargetState = {
  prepared: boolean;
  preview?: boolean;
  available?: boolean;
  normal_display_ready: boolean;
  object_point_count: number;
  interface_fill_point_count: number;
};

export type MeshReconstructionTargetState = {
  completed: boolean;
  path?: string | null;
  method?: "poisson" | "local_plane_filled_holes" | string;
  vertex_count?: number;
  triangle_count?: number;
};

export type CombinedReconstructionTargetState = {
  available: boolean;
  source: "mesh" | "segmentation" | "none";
  point_count?: number;
  vertex_count?: number;
  triangle_count?: number;
  bounds?: PointBounds | null;
};

export type PointBounds = {
  min: [number, number, number];
  max: [number, number, number];
};

export type SessionSummary = {
  session_id: string;
  status: WorkflowStatus;
  current_file: string | null;
  epsg_code: number | null;
  point_count: number;
  interface_source?: "manual" | "auto" | null;
  manual_interface_ready?: boolean;
  auto_interface_ready?: boolean;
  interface_draft_ready?: boolean;
  last_segmentation_mode?: "rg" | "icrg" | null;
  interface_draft?: InterfaceDraftSummary | null;
  seeds: {
    rock: number[];
    pedestal: number[];
  };
  outputs: {
    segmented: string | null;
    mesh: string | null;
    pedestal_mesh?: string | null;
    analysis: string | null;
  };
  mesh_prepared_targets?: Partial<Record<MeshTarget, MeshTargetState>>;
  mesh_reconstruction_targets?: Partial<Record<MeshTarget, MeshReconstructionTargetState>>;
  combined_reconstruction?: {
    available: boolean;
    components?: Partial<Record<MeshTarget, CombinedReconstructionTargetState>>;
  };
  normals_display_ready_by_target?: Partial<Record<MeshTarget, boolean>>;
};

export type PointMarker = {
  index: number;
  point: [number, number, number];
  color: [number, number, number];
  label: string;
};

export type PointCloudView = {
  kind: "pointCloud";
  points: [number, number, number][];
  colors: [number, number, number][];
  normals?: [number, number, number][];
  indices: number[];
  bounds: PointBounds;
  scene_bounds?: PointBounds;
  markers: PointMarker[];
  normal_segments?: [[number, number, number], [number, number, number]][];
  analysis_markers?: PointMarker[];
  analysis_segments?: {
    start: [number, number, number];
    end: [number, number, number];
    color: [number, number, number];
    label: string;
  }[];
  analysis_summary?: {
    title?: string;
    metrics?: { label: string; value: string }[];
    vectors?: { label: string; value: [number, number, number] | null }[];
    csv_path?: string | null;
  };
  normal_diagnostics?: {
    point_count: number;
    normal_shape: number[] | null;
    finite_normal_count: number;
    nonzero_normal_count: number;
    segment_count: number;
    stride: number;
    scale: number;
    min_norm: number;
    mean_norm: number;
    max_norm: number;
    status: string;
  };
  rock_point_count?: number;
  bottom_point_count?: number;
  pedestal_point_count?: number;
  object_point_count?: number;
  interface_fill_point_count?: number;
  mesh_target?: MeshTarget;
  reset_preview?: boolean;
  prepared_saved?: boolean;
  total_points: number;
  rendered_points: number;
};

export type MeshView = {
  kind: "mesh";
  url: string;
  show_wireframe: boolean;
  mesh_target?: MeshTarget;
  mesh_method?: "poisson" | "local_plane_filled_holes" | string;
  mesh_path?: string | null;
  vertex_count?: number;
  triangle_count?: number;
  vertices?: [number, number, number][];
  triangles?: [number, number, number][];
  analysis_markers?: PointCloudView["analysis_markers"];
  analysis_segments?: PointCloudView["analysis_segments"];
  analysis_summary?: PointCloudView["analysis_summary"];
  bounds?: PointBounds;
  scene_bounds?: PointBounds;
};

export type CombinedMeshComponent =
  | {
    kind: "mesh";
    target: MeshTarget;
    source: "mesh";
    url: string;
    mesh_path?: string | null;
    vertex_count?: number;
    triangle_count?: number;
    vertices?: [number, number, number][];
    triangles?: [number, number, number][];
    color?: [number, number, number];
    wire_color?: [number, number, number];
    show_wireframe?: boolean;
  }
  | {
    kind: "pointCloud";
    target: MeshTarget;
    source: "segmentation";
    points: [number, number, number][];
    colors: [number, number, number][];
    normals?: [number, number, number][];
    indices: number[];
    point_count: number;
  };

export type CombinedMeshView = {
  kind: "combinedMesh";
  components: CombinedMeshComponent[];
  total_points: number;
  rendered_points: number;
  bounds: PointBounds;
  scene_bounds?: PointBounds;
  analysis_markers?: PointCloudView["analysis_markers"];
  analysis_segments?: PointCloudView["analysis_segments"];
  analysis_summary?: PointCloudView["analysis_summary"];
  combined_reconstruction?: SessionSummary["combined_reconstruction"];
};

export type ViewerPayload = PointCloudView | MeshView | CombinedMeshView;

export type JobResponse = {
  job_id: string;
  session_id?: string;
  status: "queued" | "running" | "completed" | "failed";
  action?: string;
  result?: unknown;
  error?: string;
};

export type SegmentParams = {
  smoothness_threshold: number;
  curvature_threshold: number;
  basal_proximity_threshold: number;
  voxel_size: number;
  neighbor_count: number;
  distance_threshold: number;
  label_propagation_distance: number;
};

export type DenoiseParams = {
  method: "sor" | "dbscan" | "sor_dbscan";
  sor_neighbors: number;
  sor_std_ratio: number;
  dbscan_eps: number;
  dbscan_min_points: number;
};

export type ProjectUiState = {
  project_filename?: string | null;
  active_view?: string;
  pick_mode?: string;
  active_mesh_target?: MeshTarget;
  point_size?: number;
  segment_params?: SegmentParams;
  denoise_params?: DenoiseParams;
  normal_method?: "pymeshlab" | "open3d";
  normal_k?: number;
  normal_display_scale?: number;
  mesh_depth?: number;
  hover_tips_enabled?: boolean;
  interface_points?: number[];
  interface_parts?: InterfacePartRequest[];
  current_part_lateral?: boolean;
  close_loop?: boolean;
};

export type ProjectExportRequest = {
  filename?: string | null;
  ui_state: ProjectUiState;
};

export type ProjectExportResponse = {
  blob: Blob;
  filename: string;
};

export type ProjectImportResponse = {
  summary: SessionSummary;
  ui_state?: ProjectUiState;
  project_filename?: string | null;
};

export type InterfacePartRequest = {
  selected_indices: number[];
  is_lateral: boolean;
};

export type InterfaceRequest = {
  parts: InterfacePartRequest[];
  close_loop: boolean;
};

export type InterfaceDraftSummary = {
  part_count: number;
  anchor_count: number;
  include_count: number;
  exclude_count: number;
  effective_count: number;
  can_undo: boolean;
  close_loop: boolean;
};

export type InterfaceDraft = {
  source?: "auto" | "manual" | null;
  parts: InterfacePartRequest[];
  close_loop: boolean;
  include_indices: number[];
  exclude_indices: number[];
  effective_indices: number[];
  metadata?: unknown;
  summary?: InterfaceDraftSummary | null;
};

export type InterfaceDraftResponse = {
  draft: InterfaceDraft | null;
  summary?: SessionSummary;
  basal_point_count?: number;
  auto_point_count?: number;
  anchor_count?: number;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = await response.json();
      message = payload.detail ?? message;
    } catch {
      // Keep status text.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

function filenameFromContentDisposition(value: string | null, fallback: string) {
  if (!value) {
    return fallback;
  }
  const utf8Match = value.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) {
    return decodeURIComponent(utf8Match[1].replace(/"/g, ""));
  }
  const asciiMatch = value.match(/filename="?([^";]+)"?/i);
  return asciiMatch?.[1] ?? fallback;
}

export async function createSession(): Promise<SessionSummary> {
  return request<SessionSummary>("/api/sessions", { method: "POST" });
}

export async function getSession(sessionId: string): Promise<SessionSummary> {
  return request<SessionSummary>(`/api/sessions/${sessionId}`);
}

export async function uploadPointCloud(sessionId: string, file: File): Promise<SessionSummary> {
  const data = new FormData();
  data.append("file", file);
  return request<SessionSummary>(`/api/sessions/${sessionId}/point-cloud`, {
    method: "POST",
    body: data
  });
}

export async function getViewer(
  sessionId: string,
  viewName: string,
  options: { meshTarget?: MeshTarget } = {}
): Promise<ViewerPayload> {
  const params = new URLSearchParams({ t: String(Date.now()) });
  if (options.meshTarget) {
    params.set("mesh_target", options.meshTarget);
  }
  return request<ViewerPayload>(`/api/sessions/${sessionId}/viewer/${viewName}?${params.toString()}`);
}

export async function getJob(jobId: string): Promise<JobResponse> {
  return request<JobResponse>(`/api/jobs/${jobId}`);
}

export async function runJob(endpoint: string, body?: unknown): Promise<JobResponse> {
  return request<JobResponse>(endpoint, {
    method: "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body)
  });
}

export async function interpolateInterfacePath(sessionId: string, body: InterfaceRequest): Promise<JobResponse> {
  return runJob(`/api/sessions/${sessionId}/interface/interpolate`, body);
}

export async function saveInterfaceConstraints(sessionId: string, body: InterfaceRequest): Promise<JobResponse> {
  return runJob(`/api/sessions/${sessionId}/interface`, body);
}

export async function clearInterfacePreview(sessionId: string): Promise<JobResponse> {
  return runJob(`/api/sessions/${sessionId}/interface/preview/clear`);
}

export async function createInterfaceDraftFromAuto(sessionId: string): Promise<JobResponse> {
  return runJob(`/api/sessions/${sessionId}/interface/draft/from-auto`);
}

export async function createInterfaceDraftFromSource(sessionId: string, source: "auto" | "manual"): Promise<JobResponse> {
  return runJob(`/api/sessions/${sessionId}/interface/draft/from-source`, { source });
}

export async function getInterfaceDraft(sessionId: string): Promise<InterfaceDraftResponse> {
  return request<InterfaceDraftResponse>(`/api/sessions/${sessionId}/interface/draft?t=${Date.now()}`);
}

export async function updateInterfaceDraftAnchors(sessionId: string, body: InterfaceRequest): Promise<JobResponse> {
  return runJob(`/api/sessions/${sessionId}/interface/draft/anchors`, body);
}

export async function brushInterfaceDraft(
  sessionId: string,
  mode: "add" | "remove",
  selectedIndices: number[],
  strokeIndices: number[] = []
): Promise<JobResponse> {
  return runJob(`/api/sessions/${sessionId}/interface/draft/brush`, {
    mode,
    selected_indices: selectedIndices,
    stroke_indices: strokeIndices
  });
}

export async function undoInterfaceDraft(sessionId: string): Promise<JobResponse> {
  return runJob(`/api/sessions/${sessionId}/interface/draft/undo`);
}

export async function clearInterfaceDraft(sessionId: string): Promise<JobResponse> {
  return runJob(`/api/sessions/${sessionId}/interface/draft/clear`);
}

export async function commitInterfaceDraft(sessionId: string): Promise<JobResponse> {
  return runJob(`/api/sessions/${sessionId}/interface/draft/commit`);
}

export async function manualRemovePreparedPoints(
  sessionId: string,
  selectedIndices: number[],
  target: MeshTarget = "rock"
): Promise<JobResponse> {
  return runJob(`/api/sessions/${sessionId}/mesh/noise/manual-remove`, { selected_indices: selectedIndices, target });
}

export function downloadUrl(sessionId: string, kind: string): string {
  return `${API_BASE}/api/sessions/${sessionId}/downloads/${kind}`;
}

export async function exportProject(sessionId: string, body: ProjectExportRequest): Promise<ProjectExportResponse> {
  const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/project/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = await response.json();
      message = payload.detail ?? message;
    } catch {
      // Keep status text.
    }
    throw new Error(message);
  }
  return {
    blob: await response.blob(),
    filename: filenameFromContentDisposition(response.headers.get("Content-Disposition"), body.filename || "rock_detection_project.rd3dproj")
  };
}

export async function importProject(sessionId: string, file: File): Promise<ProjectImportResponse> {
  const data = new FormData();
  data.append("file", file);
  return request<ProjectImportResponse>(`/api/sessions/${sessionId}/project/import`, {
    method: "POST",
    body: data
  });
}
