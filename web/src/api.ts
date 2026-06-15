export type WorkflowStatus = {
  point_cloud_loaded: boolean;
  seeds_ready: boolean;
  manual_interface_ready: boolean;
  auto_interface_ready: boolean;
  interface_draft_ready: boolean;
  interface_ready: boolean;
  segmentation_ready: boolean;
  mesh_prepared: boolean;
  mesh_completed: boolean;
  analysis_completed: boolean;
  last_segmentation_mode?: "rg" | "icrg" | null;
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
    analysis: string | null;
  };
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
  bounds: {
    min: [number, number, number];
    max: [number, number, number];
  };
  scene_bounds?: {
    min: [number, number, number];
    max: [number, number, number];
  };
  markers: PointMarker[];
  normal_segments?: [[number, number, number], [number, number, number]][];
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
  total_points: number;
  rendered_points: number;
};

export type MeshView = {
  kind: "mesh";
  url: string;
  show_wireframe: boolean;
  vertices?: [number, number, number][];
  triangles?: [number, number, number][];
  bounds?: {
    min: [number, number, number];
    max: [number, number, number];
  };
  scene_bounds?: {
    min: [number, number, number];
    max: [number, number, number];
  };
};

export type ViewerPayload = PointCloudView | MeshView;

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
};

export type DenoiseParams = {
  method: "sor" | "dbscan" | "sor_dbscan";
  sor_neighbors: number;
  sor_std_ratio: number;
  dbscan_eps: number;
  dbscan_min_points: number;
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

export async function getViewer(sessionId: string, viewName: string): Promise<ViewerPayload> {
  return request<ViewerPayload>(`/api/sessions/${sessionId}/viewer/${viewName}?t=${Date.now()}`);
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

export async function manualRemovePreparedPoints(sessionId: string, selectedIndices: number[]): Promise<JobResponse> {
  return runJob(`/api/sessions/${sessionId}/mesh/noise/manual-remove`, { selected_indices: selectedIndices });
}

export function downloadUrl(sessionId: string, kind: string): string {
  return `${API_BASE}/api/sessions/${sessionId}/downloads/${kind}`;
}
