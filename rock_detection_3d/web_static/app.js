const REQUIRED_RUNTIME_BUILD = "20260613-trackball1";

const state = {
  session: null,
  view: null,
  runtime: null,
  activeView: "raw",
  pickMode: "rock",
  rockSeeds: [],
  pedestalSeeds: [],
  interfacePoints: [],
  interfaceParts: [],
  rotationX: -0.72,
  rotationY: 0.62,
  rotationMatrix: null,
  trackballVector: null,
  zoom: 1,
  panX: 0,
  panY: 0,
  dragging: false,
  dragMode: null,
  dragStart: null,
  lastPointer: null,
  gl: null,
  program: null,
  pickProgram: null,
  lineProgram: null,
  positionBuffer: null,
  colorBuffer: null,
  pointNormalBuffer: null,
  pickColorBuffer: null,
  normalPositionBuffer: null,
  normalColorBuffer: null,
  markerPositionBuffer: null,
  markerColorBuffer: null,
  markerHaloPositionBuffer: null,
  markerHaloColorBuffer: null,
  meshPositionBuffer: null,
  meshColorBuffer: null,
  meshLinePositionBuffer: null,
  meshLineColorBuffer: null,
  centeredPositions: null,
  pointNormalCount: 0,
  sourceIndices: [],
  markerPositions: null,
  markerColors: null,
  markerHaloPositions: null,
  markerHaloColors: null,
  pointCount: 0,
  normalLineCount: 0,
  markerCount: 0,
  markerHaloCount: 0,
  meshVertexCount: 0,
  meshLineVertexCount: 0,
  frameCenter: [0, 0, 0],
  frameRadius: 1,
  orbitOffset: [0, 0, 0],
  mvpMatrix: null,
  pickFramebuffer: null,
  pickTexture: null,
  pickDepthBuffer: null,
  pickWidth: 0,
  pickHeight: 0,
  renderKey: null,
  tooltipTarget: null,
  hoverTipsEnabled: true,
  interfaceWindowDragging: false,
  interfaceWindowDragOffset: null,
  manualRemovalWindowDragging: false,
  manualRemovalWindowDragOffset: null,
  manualRemovalWindowOpen: false,
  manualRemovalDrawMode: false,
  manualRemovalPolygon: [],
  manualRemovalSelected: [],
  seedSaveTimer: null,
  seedSaveInFlight: false,
  seedSaveQueued: false,
  seedSavePromise: null,
  seedSaveSignature: ""
};

const el = {};

function $(id) {
  return document.getElementById(id);
}

function initElements() {
  [
    "currentFile", "fileInput", "statusList", "activeViewLabel", "viewMeta", "viewer", "toast",
    "infoTooltip",
    "toggleTips",
    "rockCount", "pedestalCount", "interfaceCount", "partsCount", "pointSize", "pickRock", "pickPedestal",
    "pickInterface", "autoSeeds", "clearCurrentPick",
    "interfaceWindow", "interfaceWindowHandle", "closeInterfaceWindow", "partLateral", "closeLoop", "stagePart",
    "interpolateInterface", "saveInterface", "clearParts", "smoothness", "curvature", "proximity", "voxel",
    "neighbors", "distance", "runSegment", "prepareMesh", "removeNoise", "undoNoise",
    "manualRemoval", "manualRemovalOverlay", "manualRemovalWindow", "manualRemovalWindowHandle",
    "closeManualRemovalWindow", "manualRemovalDraw", "manualRemovalUndoVertex", "manualRemovalClear",
    "manualRemovalApply", "manualRemovalClose", "manualRemovalCount",
    "denoiseMethod", "sorNeighbors", "sorStdRatio", "dbscanEps", "dbscanMinPoints",
    "normalMethod", "normalK", "normalScale", "normalScaleValue", "computeNormals", "meshDepth", "reconstruct", "analyze",
    "downloadSegmented", "downloadMesh", "downloadAnalysis"
  ].forEach((id) => {
    el[id] = $(id);
  });
}

async function api(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch {
      // Keep response status text.
    }
    throw new Error(message);
  }
  return response.json();
}

function showToast(message, isError = false) {
  el.toast.textContent = message;
  el.toast.classList.toggle("error", isError);
  el.toast.classList.remove("hidden");
}

function hideToast() {
  el.toast.classList.add("hidden");
}

async function checkRuntime() {
  try {
    const diagnostics = await api("/api/diagnostics/runtime");
    state.runtime = diagnostics;
    console.info("[Rock3D] backend runtime", diagnostics);
    if (diagnostics.build !== REQUIRED_RUNTIME_BUILD) {
      showToast(`Backend is stale (${diagnostics.build || "unknown"}). Restart FastAPI before testing normals.`, true);
    }
  } catch (error) {
    console.warn("[Rock3D] backend runtime diagnostics unavailable", error);
    showToast("Backend is stale or missing /api/diagnostics/runtime. Restart FastAPI before testing normals.", true);
  }
}

function positionInfoTooltip(x, y) {
  const tooltip = el.infoTooltip;
  if (!tooltip) {
    return;
  }
  const margin = 12;
  const offset = 14;
  const rect = tooltip.getBoundingClientRect();
  let left = x + offset;
  let top = y + offset;
  if (left + rect.width + margin > window.innerWidth) {
    left = Math.max(margin, x - rect.width - offset);
  }
  if (top + rect.height + margin > window.innerHeight) {
    top = Math.max(margin, y - rect.height - offset);
  }
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

function showInfoTooltip(target, event) {
  const tooltip = el.infoTooltip;
  const isButtonTip = target.classList.contains("button-tooltip-wrap");
  if (!tooltip || (isButtonTip && !state.hoverTipsEnabled)) {
    return;
  }
  const titleText = target.dataset.infoTitle || target.dataset.helpTitle || "Help";
  const bodyText = target.dataset.infoBody || target.dataset.helpBody || "";
  if (!bodyText) {
    return;
  }
  const title = document.createElement("strong");
  const body = document.createElement("p");
  title.textContent = titleText;
  body.textContent = bodyText;
  tooltip.replaceChildren(title, body);
  tooltip.classList.remove("hidden");
  const rect = target.getBoundingClientRect();
  const x = event?.clientX ?? rect.left + rect.width / 2;
  const y = event?.clientY ?? rect.bottom;
  positionInfoTooltip(x, y);
}

function hideInfoTooltip() {
  if (el.infoTooltip) {
    el.infoTooltip.classList.add("hidden");
  }
  state.tooltipTarget = null;
}

function updateHoverTipsToggle() {
  if (!el.toggleTips) {
    return;
  }
  el.toggleTips.textContent = state.hoverTipsEnabled ? "Hover Tips On" : "Hover Tips Off";
  el.toggleTips.setAttribute("aria-pressed", String(state.hoverTipsEnabled));
  el.toggleTips.classList.toggle("off", !state.hoverTipsEnabled);
}

function toggleHoverTips() {
  state.hoverTipsEnabled = !state.hoverTipsEnabled;
  if (!state.hoverTipsEnabled && state.tooltipTarget?.classList.contains("button-tooltip-wrap")) {
    hideInfoTooltip();
  }
  updateHoverTipsToggle();
}

function wrapHelpButtons() {
  document.querySelectorAll("button[data-help-title]").forEach((button) => {
    if (button.parentElement?.classList.contains("button-tooltip-wrap")) {
      return;
    }
    const wrapper = document.createElement("span");
    wrapper.className = button.classList.contains("wide")
      ? "button-tooltip-wrap wide-wrap"
      : "button-tooltip-wrap";
    wrapper.dataset.helpTitle = button.dataset.helpTitle || "";
    wrapper.dataset.helpBody = button.dataset.helpBody || "";
    button.before(wrapper);
    wrapper.appendChild(button);
  });
}

function bindTooltipTarget(target) {
  target.addEventListener("pointerenter", (event) => {
    state.tooltipTarget = target;
    showInfoTooltip(target, event);
  });
  target.addEventListener("pointermove", (event) => {
    if (state.tooltipTarget === target) {
      positionInfoTooltip(event.clientX, event.clientY);
    }
  });
  target.addEventListener("pointerleave", hideInfoTooltip);
  target.addEventListener("focusin", (event) => {
    state.tooltipTarget = target;
    showInfoTooltip(target, event);
  });
  target.addEventListener("focusout", hideInfoTooltip);
  target.addEventListener("click", (event) => {
    if (target.classList.contains("info-button")) {
      event.preventDefault();
      event.stopPropagation();
      showInfoTooltip(target, event);
    }
  });
  target.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      hideInfoTooltip();
      if (typeof target.blur === "function") {
        target.blur();
      }
    }
  });
}

function setBusy(message) {
  if (message) {
    showToast(message);
  } else {
    hideToast();
  }
}

function setError(error) {
  showToast(error.message || String(error), true);
}

function updateStatus() {
  if (!state.session) {
    return;
  }
  const status = state.session.status;
  const rows = [
    ["point_cloud_loaded", "Point cloud"],
    ["seeds_ready", "Seeds"],
    ["interface_ready", "Interface"],
    ["segmentation_ready", "Segmentation"],
    ["mesh_prepared", "Mesh prep"],
    ["mesh_completed", "Mesh"],
    ["analysis_completed", "Analysis"]
  ];
  el.statusList.innerHTML = rows
    .map(([key, label]) => `<div class="status-row ${status[key] ? "done" : ""}">${label}</div>`)
    .join("");

  el.currentFile.textContent = state.session.current_file || "No point cloud loaded";
  updateViewMeta();
  el.rockCount.textContent = `Rock ${state.rockSeeds.length}`;
  el.pedestalCount.textContent = `Pedestal ${state.pedestalSeeds.length}`;
  el.interfaceCount.textContent = `Interface ${state.interfacePoints.length}`;
  el.partsCount.textContent = `Parts ${state.interfaceParts.length}`;

  setDownload(el.downloadSegmented, "segmented", state.session.outputs.segmented);
  setDownload(el.downloadMesh, "mesh", state.session.outputs.mesh);
  setDownload(el.downloadAnalysis, "analysis", state.session.outputs.analysis);
  if (el.manualRemoval) {
    el.manualRemoval.disabled = !status.mesh_prepared;
  }
  updateManualRemovalUI({ redraw: false });
}

function setDownload(anchor, kind, available) {
  if (!state.session || !available) {
    anchor.removeAttribute("href");
    anchor.classList.add("disabled");
    return;
  }
  anchor.href = `/api/sessions/${state.session.session_id}/downloads/${kind}`;
  anchor.classList.remove("disabled");
}

async function createSession() {
  state.session = await api("/api/sessions", { method: "POST" });
  updateStatus();
}

async function refreshSession(summary, options = {}) {
  if (summary) {
    state.session = summary;
  } else if (state.session) {
    state.session = await api(`/api/sessions/${state.session.session_id}`);
  }
  if (state.session && options.syncSeeds) {
    state.rockSeeds = state.session.seeds.rock || [];
    state.pedestalSeeds = state.session.seeds.pedestal || [];
  }
  updateStatus();
}

function resultSummary(job) {
  const result = job.result;
  if (!result) {
    return null;
  }
  if (result.summary) {
    return result.summary;
  }
  if (result.session_id) {
    return result;
  }
  return null;
}

async function pollJob(jobId) {
  let job = await api(`/api/jobs/${jobId}`);
  while (job.status === "queued" || job.status === "running") {
    await new Promise((resolve) => setTimeout(resolve, 650));
    job = await api(`/api/jobs/${jobId}`);
  }
  if (job.status === "failed") {
    throw new Error(job.error || "Job failed");
  }
  return job;
}

function jobCompletionMessage(label, result) {
  if (!result || typeof result !== "object") {
    return "";
  }
  if (!label.toLowerCase().includes("normal")) {
    return "";
  }
  if (Number.isFinite(result.normal_segment_count)) {
    const diagnostics = result.normal_diagnostics || {};
    const validCount = Number.isFinite(diagnostics.nonzero_normal_count)
      ? `, ${diagnostics.nonzero_normal_count.toLocaleString()} valid normals`
      : "";
    return `${label} complete: ${result.normal_segment_count.toLocaleString()} normal arrows${validCount}.`;
  }
  return "";
}

async function runAction(label, path, body, nextView, actionOptions = {}) {
  if (!state.session) {
    return;
  }
  try {
    setBusy(label);
    const requestOptions = body === undefined
      ? { method: "POST" }
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const submitted = await api(path, requestOptions);
    const job = await pollJob(submitted.job_id);
    await refreshSession(resultSummary(job), { syncSeeds: Boolean(actionOptions.syncSeeds) });
    const resolvedNextView = typeof nextView === "function" ? nextView(job.result) : nextView;
    if (resolvedNextView) {
      await loadView(resolvedNextView);
    }
    setBusy(null);
    const message = jobCompletionMessage(label, job.result);
    if (message) {
      showToast(message);
    }
    return job;
  } catch (error) {
    setError(error);
    return null;
  }
}

async function uploadFile(file) {
  if (!file || !state.session) {
    return;
  }
  try {
    setBusy("Uploading point cloud");
    const data = new FormData();
    data.append("file", file);
    state.session = await api(`/api/sessions/${state.session.session_id}/point-cloud`, {
      method: "POST",
      body: data
    });
    state.rockSeeds = [];
    state.pedestalSeeds = [];
    state.interfacePoints = [];
    state.interfaceParts = [];
    clearManualRemovalSelection({ redraw: false });
    state.zoom = 1;
    state.panX = 0;
    state.panY = 0;
    resetRotationMatrix();
    state.renderKey = null;
    state.seedSaveSignature = "";
    updateStatus();
    await loadView("raw");
    setBusy(null);
  } catch (error) {
    setError(error);
  }
}

async function loadView(viewName) {
  if (!state.session) {
    return;
  }
  try {
    const payload = await api(`/api/sessions/${state.session.session_id}/viewer/${viewName}?t=${Date.now()}`);
    if (payload.kind === "mesh") {
      await hydrateMeshView(payload);
    }
    state.view = payload;
    state.activeView = viewName;
    el.activeViewLabel.textContent = viewName.replace("_", " ");
    updateViewMeta();
    if (viewName !== "mesh_prepared" && state.manualRemovalWindowOpen) {
      state.manualRemovalDrawMode = false;
      state.manualRemovalSelected = [];
      state.manualRemovalPolygon = [];
    }
    updateManualRemovalUI({ redraw: false });
    document.querySelectorAll("[data-view]").forEach((button) => {
      button.classList.toggle("active", button.dataset.view === viewName);
    });
    draw();
  } catch (error) {
    setError(error);
  }
}

function updateViewMeta() {
  if (!state.session) {
    return;
  }
  const pointCount = state.view?.total_points ?? state.session.point_count ?? 0;
  const parts = [`${pointCount.toLocaleString()} pts`, `EPSG ${state.session.epsg_code || "--"}`];
  if (state.view?.kind === "pointCloud" && state.view.normal_segments) {
    const diagnostics = state.view.normal_diagnostics || {};
    parts.push(`${(state.view.normal_segments || []).length.toLocaleString()} normal arrows`);
    if (Number.isFinite(diagnostics.nonzero_normal_count)) {
      parts.push(`${diagnostics.nonzero_normal_count.toLocaleString()} valid normals`);
    }
    if (el.normalScale) {
      parts.push(`scale ${Number(el.normalScale.value || 1).toFixed(2)}x`);
    }
  }
  el.viewMeta.textContent = parts.join(" · ");
}

async function hydrateMeshView(payload) {
  if (payload.vertices?.length && payload.triangles?.length) {
    return;
  }
  if (!payload.url) {
    throw new Error("Mesh view is missing its PLY download URL.");
  }
  const url = `${payload.url}${payload.url.includes("?") ? "&" : "?"}t=${Date.now()}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load mesh PLY: ${response.statusText}`);
  }
  Object.assign(payload, parsePlyMesh(await response.arrayBuffer()));
}

function findPlyHeaderEnd(bytes) {
  const marker = new TextEncoder().encode("end_header");
  for (let i = 0; i <= bytes.length - marker.length; i += 1) {
    let matches = true;
    for (let j = 0; j < marker.length; j += 1) {
      if (bytes[i + j] !== marker[j]) {
        matches = false;
        break;
      }
    }
    if (!matches) {
      continue;
    }
    let end = i + marker.length;
    while (end < bytes.length && bytes[end] !== 10) {
      end += 1;
    }
    return Math.min(end + 1, bytes.length);
  }
  throw new Error("Invalid PLY: missing end_header.");
}

function parsePlyHeader(headerText) {
  const lines = headerText.split(/\r?\n/);
  const elements = [];
  let format = "ascii";
  let current = null;
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line === "ply" || line.startsWith("comment ")) {
      continue;
    }
    const tokens = line.split(/\s+/);
    if (tokens[0] === "format") {
      format = tokens[1];
    } else if (tokens[0] === "element") {
      current = { name: tokens[1], count: Number(tokens[2] || 0), properties: [] };
      elements.push(current);
    } else if (tokens[0] === "property" && current) {
      if (tokens[1] === "list") {
        current.properties.push({
          kind: "list",
          countType: tokens[2],
          itemType: tokens[3],
          name: tokens[4]
        });
      } else {
        current.properties.push({ kind: "scalar", type: tokens[1], name: tokens[2] });
      }
    }
  }
  return { format, elements };
}

const PLY_TYPE_SIZES = {
  char: 1,
  int8: 1,
  uchar: 1,
  uint8: 1,
  short: 2,
  int16: 2,
  ushort: 2,
  uint16: 2,
  int: 4,
  int32: 4,
  uint: 4,
  uint32: 4,
  float: 4,
  float32: 4,
  double: 8,
  float64: 8
};

function readPlyScalar(dataView, offset, type, littleEndian) {
  switch (type) {
    case "char":
    case "int8":
      return [dataView.getInt8(offset), offset + 1];
    case "uchar":
    case "uint8":
      return [dataView.getUint8(offset), offset + 1];
    case "short":
    case "int16":
      return [dataView.getInt16(offset, littleEndian), offset + 2];
    case "ushort":
    case "uint16":
      return [dataView.getUint16(offset, littleEndian), offset + 2];
    case "int":
    case "int32":
      return [dataView.getInt32(offset, littleEndian), offset + 4];
    case "uint":
    case "uint32":
      return [dataView.getUint32(offset, littleEndian), offset + 4];
    case "float":
    case "float32":
      return [dataView.getFloat32(offset, littleEndian), offset + 4];
    case "double":
    case "float64":
      return [dataView.getFloat64(offset, littleEndian), offset + 8];
    default:
      throw new Error(`Unsupported PLY property type: ${type}`);
  }
}

function pushTriangulatedFace(triangles, indices) {
  if (!indices || indices.length < 3) {
    return;
  }
  for (let i = 1; i < indices.length - 1; i += 1) {
    triangles.push([indices[0], indices[i], indices[i + 1]]);
  }
}

function parseAsciiPly(bodyText, elements) {
  const lines = bodyText.split(/\r?\n/);
  let lineIndex = 0;
  const vertices = [];
  const triangles = [];
  for (const element of elements) {
    for (let row = 0; row < element.count; row += 1) {
      while (lineIndex < lines.length && !lines[lineIndex].trim()) {
        lineIndex += 1;
      }
      const values = (lines[lineIndex] || "").trim().split(/\s+/);
      lineIndex += 1;
      if (element.name === "vertex") {
        const point = [0, 0, 0];
        for (let propIndex = 0; propIndex < element.properties.length; propIndex += 1) {
          const prop = element.properties[propIndex];
          if (prop.kind === "scalar" && prop.name === "x") point[0] = Number(values[propIndex]);
          if (prop.kind === "scalar" && prop.name === "y") point[1] = Number(values[propIndex]);
          if (prop.kind === "scalar" && prop.name === "z") point[2] = Number(values[propIndex]);
        }
        vertices.push(point);
      } else if (element.name === "face") {
        const count = Number(values[0]);
        const indices = values.slice(1, 1 + count).map((value) => Number(value));
        pushTriangulatedFace(triangles, indices);
      }
    }
  }
  return { vertices, triangles };
}

function parseBinaryPly(buffer, headerEnd, elements, littleEndian) {
  const dataView = new DataView(buffer);
  let offset = headerEnd;
  const vertices = [];
  const triangles = [];
  for (const element of elements) {
    for (let row = 0; row < element.count; row += 1) {
      const vertexValues = {};
      let faceIndices = null;
      for (const prop of element.properties) {
        if (prop.kind === "list") {
          let count;
          [count, offset] = readPlyScalar(dataView, offset, prop.countType, littleEndian);
          const values = [];
          for (let item = 0; item < count; item += 1) {
            let value;
            [value, offset] = readPlyScalar(dataView, offset, prop.itemType, littleEndian);
            values.push(value);
          }
          if (prop.name === "vertex_indices" || prop.name === "vertex_index") {
            faceIndices = values;
          }
        } else {
          let value;
          [value, offset] = readPlyScalar(dataView, offset, prop.type, littleEndian);
          if (element.name === "vertex") {
            vertexValues[prop.name] = value;
          }
        }
      }
      if (element.name === "vertex") {
        vertices.push([
          Number(vertexValues.x || 0),
          Number(vertexValues.y || 0),
          Number(vertexValues.z || 0)
        ]);
      } else if (element.name === "face") {
        pushTriangulatedFace(triangles, faceIndices);
      }
    }
  }
  return { vertices, triangles };
}

function parsePlyMesh(buffer) {
  const bytes = new Uint8Array(buffer);
  const headerEnd = findPlyHeaderEnd(bytes);
  const headerText = new TextDecoder("ascii").decode(bytes.slice(0, headerEnd));
  const { format, elements } = parsePlyHeader(headerText);
  let mesh;
  if (format === "ascii") {
    mesh = parseAsciiPly(new TextDecoder("utf-8").decode(bytes.slice(headerEnd)), elements);
  } else if (format === "binary_little_endian") {
    mesh = parseBinaryPly(buffer, headerEnd, elements, true);
  } else if (format === "binary_big_endian") {
    mesh = parseBinaryPly(buffer, headerEnd, elements, false);
  } else {
    throw new Error(`Unsupported PLY format: ${format}`);
  }
  if (!mesh.vertices.length || !mesh.triangles.length) {
    throw new Error("The mesh PLY did not contain drawable triangles.");
  }
  return { ...mesh, bounds: viewBounds(mesh.vertices) };
}

function currentSelection() {
  if (state.pickMode === "rock") {
    return state.rockSeeds;
  }
  if (state.pickMode === "pedestal") {
    return state.pedestalSeeds;
  }
  return state.interfacePoints;
}

function addIndex(list, index) {
  if (!list.includes(index)) {
    list.push(index);
    return true;
  }
  return false;
}

function seedSavePayload() {
  return {
    rock_seed_indices: [...state.rockSeeds],
    pedestal_seed_indices: [...state.pedestalSeeds]
  };
}

function seedSaveSignature(payload = seedSavePayload()) {
  return JSON.stringify(payload);
}

async function runSeedAutosave() {
  if (!state.session?.status?.point_cloud_loaded) {
    return null;
  }
  if (state.seedSaveInFlight) {
    state.seedSaveQueued = true;
    return state.seedSavePromise;
  }
  state.seedSaveInFlight = true;
  state.seedSavePromise = (async () => {
    try {
      do {
        state.seedSaveQueued = false;
        const payload = seedSavePayload();
        const signature = seedSaveSignature(payload);
        if (signature === state.seedSaveSignature) {
          continue;
        }
        const submitted = await api(`/api/sessions/${state.session.session_id}/seeds/manual`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const job = await pollJob(submitted.job_id);
        state.seedSaveSignature = signature;
        await refreshSession(resultSummary(job), { syncSeeds: false });
      } while (state.seedSaveQueued);
    } catch (error) {
      showToast(`Could not save seeds automatically: ${error.message}`, true);
    } finally {
      state.seedSaveInFlight = false;
      state.seedSavePromise = null;
    }
  })();
  await state.seedSavePromise;
  return state.seedSavePromise;
}

function scheduleSeedAutosave() {
  if (!state.session?.status?.point_cloud_loaded) {
    return;
  }
  if (state.seedSaveTimer) {
    window.clearTimeout(state.seedSaveTimer);
  }
  state.seedSaveTimer = window.setTimeout(() => {
    state.seedSaveTimer = null;
    void runSeedAutosave();
  }, 300);
}

async function flushSeedAutosave() {
  if (state.seedSaveTimer) {
    window.clearTimeout(state.seedSaveTimer);
    state.seedSaveTimer = null;
  }
  await runSeedAutosave();
}

function setPickMode(mode) {
  state.pickMode = mode;
  el.pickRock.classList.toggle("active", mode === "rock");
  el.pickPedestal.classList.toggle("active", mode === "pedestal");
  el.pickInterface.classList.toggle("active", mode === "interface");
  if (mode === "interface") {
    showInterfaceWindow();
  }
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function showInterfaceWindow() {
  if (!el.interfaceWindow) {
    return;
  }
  el.interfaceWindow.classList.remove("hidden");
  el.interfaceWindow.setAttribute("aria-hidden", "false");
}

function hideInterfaceWindow() {
  if (!el.interfaceWindow) {
    return;
  }
  el.interfaceWindow.classList.add("hidden");
  el.interfaceWindow.setAttribute("aria-hidden", "true");
}

function moveInterfaceWindow(clientX, clientY) {
  if (!state.interfaceWindowDragOffset || !el.interfaceWindow) {
    return;
  }
  const rect = el.interfaceWindow.getBoundingClientRect();
  const margin = 8;
  const maxLeft = Math.max(margin, window.innerWidth - rect.width - margin);
  const maxTop = Math.max(margin, window.innerHeight - rect.height - margin);
  const left = clamp(clientX - state.interfaceWindowDragOffset[0], margin, maxLeft);
  const top = clamp(clientY - state.interfaceWindowDragOffset[1], margin, maxTop);
  el.interfaceWindow.style.left = `${left}px`;
  el.interfaceWindow.style.top = `${top}px`;
  el.interfaceWindow.style.right = "auto";
}

function startInterfaceWindowDrag(event) {
  if (!el.interfaceWindow || event.button !== 0) {
    return;
  }
  if (event.target?.closest?.("button")) {
    return;
  }
  const rect = el.interfaceWindow.getBoundingClientRect();
  state.interfaceWindowDragging = true;
  state.interfaceWindowDragOffset = [event.clientX - rect.left, event.clientY - rect.top];
  el.interfaceWindowHandle.setPointerCapture(event.pointerId);
  event.preventDefault();
}

function dragInterfaceWindow(event) {
  if (!state.interfaceWindowDragging) {
    return;
  }
  moveInterfaceWindow(event.clientX, event.clientY);
}

function stopInterfaceWindowDrag(event) {
  if (!state.interfaceWindowDragging) {
    return;
  }
  state.interfaceWindowDragging = false;
  state.interfaceWindowDragOffset = null;
  if (el.interfaceWindowHandle?.hasPointerCapture?.(event.pointerId)) {
    el.interfaceWindowHandle.releasePointerCapture(event.pointerId);
  }
}

function showManualRemovalWindow() {
  if (!el.manualRemovalWindow) {
    return;
  }
  state.manualRemovalWindowOpen = true;
  el.manualRemovalWindow.classList.remove("hidden");
  el.manualRemovalWindow.setAttribute("aria-hidden", "false");
  updateManualRemovalUI();
}

function hideManualRemovalWindow() {
  if (!el.manualRemovalWindow) {
    return;
  }
  state.manualRemovalWindowOpen = false;
  state.manualRemovalDrawMode = false;
  el.manualRemovalWindow.classList.add("hidden");
  el.manualRemovalWindow.setAttribute("aria-hidden", "true");
  updateManualRemovalUI();
}

async function openManualRemovalWindow() {
  if (!state.session?.status?.mesh_prepared) {
    showToast("Prepare the mesh before manual removal.", true);
    return;
  }
  showManualRemovalWindow();
  if (state.activeView !== "mesh_prepared") {
    await loadView("mesh_prepared");
  }
}

function moveManualRemovalWindow(clientX, clientY) {
  if (!state.manualRemovalWindowDragOffset || !el.manualRemovalWindow) {
    return;
  }
  const rect = el.manualRemovalWindow.getBoundingClientRect();
  const margin = 8;
  const maxLeft = Math.max(margin, window.innerWidth - rect.width - margin);
  const maxTop = Math.max(margin, window.innerHeight - rect.height - margin);
  const left = clamp(clientX - state.manualRemovalWindowDragOffset[0], margin, maxLeft);
  const top = clamp(clientY - state.manualRemovalWindowDragOffset[1], margin, maxTop);
  el.manualRemovalWindow.style.left = `${left}px`;
  el.manualRemovalWindow.style.top = `${top}px`;
  el.manualRemovalWindow.style.right = "auto";
}

function startManualRemovalWindowDrag(event) {
  if (!el.manualRemovalWindow || event.button !== 0) {
    return;
  }
  if (event.target?.closest?.("button")) {
    return;
  }
  const rect = el.manualRemovalWindow.getBoundingClientRect();
  state.manualRemovalWindowDragging = true;
  state.manualRemovalWindowDragOffset = [event.clientX - rect.left, event.clientY - rect.top];
  el.manualRemovalWindowHandle.setPointerCapture(event.pointerId);
  event.preventDefault();
}

function dragManualRemovalWindow(event) {
  if (!state.manualRemovalWindowDragging) {
    return;
  }
  moveManualRemovalWindow(event.clientX, event.clientY);
}

function stopManualRemovalWindowDrag(event) {
  if (!state.manualRemovalWindowDragging) {
    return;
  }
  state.manualRemovalWindowDragging = false;
  state.manualRemovalWindowDragOffset = null;
  if (el.manualRemovalWindowHandle?.hasPointerCapture?.(event.pointerId)) {
    el.manualRemovalWindowHandle.releasePointerCapture(event.pointerId);
  }
}

function viewerPointFromEvent(event) {
  const rect = el.viewer.getBoundingClientRect();
  return {
    x: clamp(event.clientX - rect.left, 0, rect.width),
    y: clamp(event.clientY - rect.top, 0, rect.height)
  };
}

function resizeManualRemovalOverlay() {
  if (!el.manualRemovalOverlay) {
    return null;
  }
  const rect = el.viewer.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (el.manualRemovalOverlay.width !== width || el.manualRemovalOverlay.height !== height) {
    el.manualRemovalOverlay.width = width;
    el.manualRemovalOverlay.height = height;
  }
  el.manualRemovalOverlay.style.width = `${rect.width}px`;
  el.manualRemovalOverlay.style.height = `${rect.height}px`;
  return { rect, dpr, width, height };
}

function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const crosses = (yi > y) !== (yj > y);
    if (!crosses) {
      continue;
    }
    const edgeX = ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (x < edgeX) {
      inside = !inside;
    }
  }
  return inside;
}

function drawManualRemovalOverlay() {
  if (!el.manualRemovalOverlay) {
    return;
  }
  const geometry = resizeManualRemovalOverlay();
  if (!geometry) {
    return;
  }
  const ctx = el.manualRemovalOverlay.getContext("2d");
  ctx.setTransform(geometry.dpr, 0, 0, geometry.dpr, 0, 0);
  ctx.clearRect(0, 0, geometry.rect.width, geometry.rect.height);
  const shouldShow = state.manualRemovalWindowOpen && state.activeView === "mesh_prepared";
  el.manualRemovalOverlay.classList.toggle("hidden", !shouldShow);
  document.body.classList.toggle("manual-removal-drawing", shouldShow && state.manualRemovalDrawMode);
  if (!shouldShow) {
    return;
  }

  const polygon = state.manualRemovalPolygon;
  if (!polygon.length) {
    return;
  }
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#ffdc3d";
  ctx.fillStyle = "rgba(255, 220, 61, 0.16)";
  ctx.beginPath();
  ctx.moveTo(polygon[0].x, polygon[0].y);
  for (let i = 1; i < polygon.length; i += 1) {
    ctx.lineTo(polygon[i].x, polygon[i].y);
  }
  if (polygon.length >= 3) {
    ctx.closePath();
    ctx.fill();
  }
  ctx.stroke();

  ctx.fillStyle = "#101820";
  ctx.strokeStyle = "#ffdc3d";
  for (const vertex of polygon) {
    ctx.beginPath();
    ctx.arc(vertex.x, vertex.y, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

function collectManualRemovalSelection() {
  state.manualRemovalSelected = [];
  if (
    !state.manualRemovalWindowOpen ||
    state.activeView !== "mesh_prepared" ||
    !state.view ||
    state.view.kind !== "pointCloud" ||
    state.manualRemovalPolygon.length < 3
  ) {
    updateManualRemovalUI();
    return;
  }
  if (state.renderKey !== state.view) {
    uploadPointCloudToGPU();
  }
  const rect = el.viewer.getBoundingClientRect();
  const size = canvasSize();
  if (!renderPickBuffer(size)) {
    updateManualRemovalUI();
    return;
  }

  const gl = state.gl;
  const pixels = new Uint8Array(size.width * size.height * 4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, state.pickFramebuffer);
  gl.readPixels(0, 0, size.width, size.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  const dprX = size.width / Math.max(rect.width, 1);
  const dprY = size.height / Math.max(rect.height, 1);
  const selected = new Set();
  const totalPointCount = Number.isFinite(state.view.total_points)
    ? Number(state.view.total_points)
    : Infinity;
  for (let row = 0; row < size.height; row += 1) {
    for (let col = 0; col < size.width; col += 1) {
      const offset = (row * size.width + col) * 4;
      const encoded = (pixels[offset] << 16) | (pixels[offset + 1] << 8) | pixels[offset + 2];
      if (encoded === 0) {
        continue;
      }
      const cssX = (col + 0.5) / dprX;
      const cssY = rect.height - (row + 0.5) / dprY;
      if (!pointInPolygon(cssX, cssY, state.manualRemovalPolygon)) {
        continue;
      }
      const sourceIndex = state.sourceIndices[encoded - 1];
      if (sourceIndex !== undefined && sourceIndex >= 0 && sourceIndex < totalPointCount) {
        selected.add(sourceIndex);
      }
    }
  }
  state.manualRemovalSelected = Array.from(selected).sort((a, b) => a - b);
  updateManualRemovalUI();
}

function clearManualRemovalSelection(options = {}) {
  state.manualRemovalDrawMode = false;
  state.manualRemovalPolygon = [];
  state.manualRemovalSelected = [];
  updateManualRemovalUI({ redraw: options.redraw !== false });
}

function updateManualRemovalUI(options = {}) {
  const redraw = options.redraw !== false;
  const available = Boolean(state.session?.status?.mesh_prepared);
  if (el.manualRemoval) {
    el.manualRemoval.disabled = !available;
  }
  if (el.manualRemovalDraw) {
    el.manualRemovalDraw.disabled = !available || state.activeView !== "mesh_prepared";
    el.manualRemovalDraw.classList.toggle("active", state.manualRemovalDrawMode);
    el.manualRemovalDraw.textContent = state.manualRemovalDrawMode ? "Drawing Polygon" : "Draw Polygon";
  }
  if (el.manualRemovalUndoVertex) {
    el.manualRemovalUndoVertex.disabled = !state.manualRemovalPolygon.length;
  }
  if (el.manualRemovalClear) {
    el.manualRemovalClear.disabled = !state.manualRemovalPolygon.length && !state.manualRemovalSelected.length;
  }
  if (el.manualRemovalApply) {
    el.manualRemovalApply.disabled = !available || !state.manualRemovalSelected.length;
  }
  if (el.manualRemovalCount) {
    const vertexText = `${state.manualRemovalPolygon.length} vertices`;
    const selectedText = `${state.manualRemovalSelected.length.toLocaleString()} selected`;
    el.manualRemovalCount.textContent = `${selectedText} - ${vertexText}`;
  }
  drawManualRemovalOverlay();
  if (redraw && state.view?.kind === "pointCloud") {
    uploadMarkersToGPU();
    draw();
  }
}

function toggleManualRemovalDraw() {
  if (state.activeView !== "mesh_prepared") {
    showToast("Manual removal draws in Mesh Prep view.", true);
    return;
  }
  state.manualRemovalDrawMode = !state.manualRemovalDrawMode;
  updateManualRemovalUI();
}

function undoManualRemovalVertex() {
  state.manualRemovalPolygon.pop();
  if (state.manualRemovalPolygon.length >= 3) {
    collectManualRemovalSelection();
  } else {
    state.manualRemovalSelected = [];
    updateManualRemovalUI();
  }
}

async function applyManualRemoval() {
  if (!state.manualRemovalSelected.length) {
    showToast("Draw a polygon that selects at least one visible rock point.", true);
    return;
  }
  const job = await runAction(
    "Manual removal",
    `/api/sessions/${state.session.session_id}/mesh/noise/manual-remove`,
    { selected_indices: state.manualRemovalSelected },
    "mesh_prepared"
  );
  if (!job) {
    return;
  }
  state.manualRemovalPolygon = [];
  state.manualRemovalSelected = [];
  state.manualRemovalDrawMode = false;
  showManualRemovalWindow();
  updateManualRemovalUI();
}

function pickPoint(index) {
  let changed = false;
  if (state.pickMode === "rock") {
    changed = addIndex(state.rockSeeds, index);
  } else if (state.pickMode === "pedestal") {
    changed = addIndex(state.pedestalSeeds, index);
  } else {
    changed = addIndex(state.interfacePoints, index);
  }
  if (changed && (state.pickMode === "rock" || state.pickMode === "pedestal")) {
    scheduleSeedAutosave();
  }
  updateStatus();
  draw();
}

function removePoint(index) {
  const selection = currentSelection();
  const position = selection.indexOf(index);
  if (position < 0) {
    return false;
  }
  selection.splice(position, 1);
  if (state.pickMode === "rock" || state.pickMode === "pedestal") {
    scheduleSeedAutosave();
  }
  updateStatus();
  draw();
  return true;
}

async function autoSeeds() {
  setPickMode("rock");
  await flushSeedAutosave();
  const job = await runAction(
    "Computing auto seeds",
    `/api/sessions/${state.session.session_id}/seeds/auto`,
    undefined,
    "seeds",
    { syncSeeds: true }
  );
  if (job) {
    state.seedSaveSignature = seedSaveSignature();
  }
}

function clearCurrentSelectionMode() {
  if (state.pickMode === "rock") {
    state.rockSeeds = [];
    scheduleSeedAutosave();
  } else if (state.pickMode === "pedestal") {
    state.pedestalSeeds = [];
    scheduleSeedAutosave();
  } else {
    state.interfacePoints = [];
  }
  updateStatus();
  draw();
}

function stagePart() {
  if (state.interfacePoints.length < 2) {
    showToast("Interface parts need at least two selected points.", true);
    return;
  }
  state.interfaceParts.push({
    selected_indices: [...state.interfacePoints],
    is_lateral: el.partLateral.checked
  });
  state.interfacePoints = [];
  el.partLateral.checked = false;
  updateStatus();
  draw();
}

function buildInterfacePayload() {
  const parts = [...state.interfaceParts];
  if (state.interfacePoints.length >= 2) {
    parts.push({
      selected_indices: [...state.interfacePoints],
      is_lateral: el.partLateral.checked
    });
  }
  if (!parts.length) {
    showToast("Select at least two interface points.", true);
    return null;
  }
  return { parts, close_loop: el.closeLoop.checked };
}

async function interpolateInterface() {
  const payload = buildInterfacePayload();
  if (!payload) {
    return;
  }
  await runAction(
    "Interpolating interface path",
    `/api/sessions/${state.session.session_id}/interface/interpolate`,
    payload,
    "interface"
  );
}

async function saveInterface() {
  const payload = buildInterfacePayload();
  if (!payload) {
    return;
  }
  await runAction(
    "Saving interface",
    `/api/sessions/${state.session.session_id}/interface`,
    payload,
    "interface"
  );
}

async function clearInterfaceParts() {
  state.interfaceParts = [];
  state.interfacePoints = [];
  updateStatus();
  draw();
  if (!state.session || !state.session.status?.point_cloud_loaded) {
    return;
  }
  try {
    await runAction(
      "Clearing interface preview",
      `/api/sessions/${state.session.session_id}/interface/preview/clear`,
      undefined,
      state.activeView === "interface" ? "interface" : undefined
    );
  } catch (error) {
    showToast(error.message, true);
  }
}

function segmentParams() {
  return {
    smoothness_threshold: Number(el.smoothness.value),
    curvature_threshold: Number(el.curvature.value),
    basal_proximity_threshold: Number(el.proximity.value),
    voxel_size: Number(el.voxel.value),
    neighbor_count: Number(el.neighbors.value),
    distance_threshold: Number(el.distance.value)
  };
}

function denoiseParams() {
  return {
    method: el.denoiseMethod.value,
    sor_neighbors: Number(el.sorNeighbors.value),
    sor_std_ratio: Number(el.sorStdRatio.value),
    dbscan_eps: Number(el.dbscanEps.value),
    dbscan_min_points: Number(el.dbscanMinPoints.value)
  };
}

function mat4Identity() {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

  let b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  out[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

  b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
  out[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

  b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
  out[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

  b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
  out[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
  return out;
}

function mat4Perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}

function mat4TranslateZ(z) {
  const out = mat4Identity();
  out[14] = z;
  return out;
}

function mat4Translate(x, y, z) {
  const out = mat4Identity();
  out[12] = x;
  out[13] = y;
  out[14] = z;
  return out;
}

function mat4RotateX(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
}

function mat4RotateY(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]);
}

function mat4RotateAxis(axis, angle) {
  const [x, y, z] = normalize(axis);
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const t = 1 - c;
  return new Float32Array([
    t * x * x + c,
    t * x * y + s * z,
    t * x * z - s * y,
    0,
    t * x * y - s * z,
    t * y * y + c,
    t * y * z + s * x,
    0,
    t * x * z + s * y,
    t * y * z - s * x,
    t * z * z + c,
    0,
    0,
    0,
    0,
    1
  ]);
}

function transformPoint(matrix, x, y, z) {
  const clipX = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
  const clipY = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
  const clipZ = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
  const clipW = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
  return [clipX, clipY, clipZ, clipW];
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(message || "Shader compile failed");
  }
  return shader;
}

function createProgram(gl) {
  const vertex = compileShader(gl, gl.VERTEX_SHADER, `
    attribute vec3 a_position;
    attribute vec3 a_color;
    attribute vec3 a_normal;
    uniform mat4 u_matrix;
    uniform mat3 u_normalMatrix;
    uniform float u_pointSize;
    uniform bool u_hasNormals;
    varying vec3 v_color;
    varying vec3 v_normal;
    varying float v_hasNormals;
    void main() {
      gl_Position = u_matrix * vec4(a_position, 1.0);
      float depthScale = clamp(2.2 / max(0.2, gl_Position.w), 0.65, 2.4);
      gl_PointSize = u_pointSize * depthScale;
      v_color = a_color;
      v_normal = normalize(u_normalMatrix * a_normal);
      v_hasNormals = u_hasNormals ? 1.0 : 0.0;
    }
  `);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, `
    precision mediump float;
    uniform vec3 u_lightDirection;
    varying vec3 v_color;
    varying vec3 v_normal;
    varying float v_hasNormals;
    void main() {
      vec2 uv = gl_PointCoord - vec2(0.5);
      float d = dot(uv, uv);
      if (d > 0.25) discard;
      vec3 color = v_color;
      if (v_hasNormals > 0.5) {
        vec3 normal = normalize(v_normal);
        vec3 light = normalize(u_lightDirection);
        float diffuse = abs(dot(normal, light));
        float specular = pow(diffuse, 28.0) * 0.24;
        color = color * (0.38 + diffuse * 0.68) + vec3(specular);
      }
      gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
    }
  `);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(message || "Shader link failed");
  }
  return program;
}

function createPickProgram(gl) {
  const vertex = compileShader(gl, gl.VERTEX_SHADER, `
    attribute vec3 a_position;
    attribute vec3 a_pickColor;
    uniform mat4 u_matrix;
    uniform float u_pointSize;
    varying vec3 v_pickColor;
    void main() {
      gl_Position = u_matrix * vec4(a_position, 1.0);
      float depthScale = clamp(2.2 / max(0.2, gl_Position.w), 0.65, 2.4);
      gl_PointSize = u_pointSize * depthScale;
      v_pickColor = a_pickColor;
    }
  `);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying vec3 v_pickColor;
    void main() {
      vec2 uv = gl_PointCoord - vec2(0.5);
      if (dot(uv, uv) > 0.25) discard;
      gl_FragColor = vec4(v_pickColor, 1.0);
    }
  `);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(message || "Pick shader link failed");
  }
  return program;
}

function createLineProgram(gl) {
  const vertex = compileShader(gl, gl.VERTEX_SHADER, `
    attribute vec3 a_position;
    attribute vec3 a_color;
    uniform mat4 u_matrix;
    varying vec3 v_color;
    void main() {
      gl_Position = u_matrix * vec4(a_position, 1.0);
      v_color = a_color;
    }
  `);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying vec3 v_color;
    void main() {
      gl_FragColor = vec4(v_color, 1.0);
    }
  `);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(message || "Line shader link failed");
  }
  return program;
}

function ensureGL() {
  if (state.gl) {
    return state.gl;
  }
  const gl = el.viewer.getContext("webgl", { antialias: true, alpha: false, depth: true });
  if (!gl) {
    return null;
  }
  state.gl = gl;
  state.program = createProgram(gl);
  state.pickProgram = createPickProgram(gl);
  state.lineProgram = createLineProgram(gl);
  state.positionBuffer = gl.createBuffer();
  state.colorBuffer = gl.createBuffer();
  state.pointNormalBuffer = gl.createBuffer();
  state.pickColorBuffer = gl.createBuffer();
  state.normalPositionBuffer = gl.createBuffer();
  state.normalColorBuffer = gl.createBuffer();
  state.markerPositionBuffer = gl.createBuffer();
  state.markerColorBuffer = gl.createBuffer();
  state.markerHaloPositionBuffer = gl.createBuffer();
  state.markerHaloColorBuffer = gl.createBuffer();
  state.meshPositionBuffer = gl.createBuffer();
  state.meshColorBuffer = gl.createBuffer();
  state.meshLinePositionBuffer = gl.createBuffer();
  state.meshLineColorBuffer = gl.createBuffer();
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0.965, 0.975, 0.957, 1);
  return gl;
}

function canvasSize() {
  const rect = el.viewer.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (el.viewer.width !== width || el.viewer.height !== height) {
    el.viewer.width = width;
    el.viewer.height = height;
  }
  return { cssWidth: rect.width, cssHeight: rect.height, width, height };
}

function ensurePickFramebuffer(width, height) {
  const gl = state.gl;
  if (!gl) {
    return false;
  }
  if (!state.pickFramebuffer) {
    state.pickFramebuffer = gl.createFramebuffer();
    state.pickTexture = gl.createTexture();
    state.pickDepthBuffer = gl.createRenderbuffer();
  }
  if (state.pickWidth === width && state.pickHeight === height) {
    return true;
  }

  state.pickWidth = width;
  state.pickHeight = height;
  gl.bindFramebuffer(gl.FRAMEBUFFER, state.pickFramebuffer);

  gl.bindTexture(gl.TEXTURE_2D, state.pickTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, state.pickTexture, 0);

  gl.bindRenderbuffer(gl.RENDERBUFFER, state.pickDepthBuffer);
  gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, width, height);
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, state.pickDepthBuffer);

  const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return ok;
}

function drawText(text) {
  try {
    const gl = ensureGL();
    if (gl) {
      const size = canvasSize();
      gl.viewport(0, 0, size.width, size.height);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      el.viewMeta.textContent = text;
      return;
    }
  } catch (error) {
    console.error(error);
  }
  if (state.gl) {
    const gl = state.gl;
    const size = canvasSize();
    gl.viewport(0, 0, size.width, size.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    el.viewMeta.textContent = text;
    return;
  }
  const ctx = el.viewer.getContext("2d");
  const rect = el.viewer.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  el.viewer.width = Math.max(1, Math.floor(rect.width * dpr));
  el.viewer.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#f6f7f4";
  ctx.fillRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#59635d";
  ctx.font = "15px Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(text, rect.width / 2, rect.height / 2);
}

function viewBounds(points) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const point of points) {
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], point[axis]);
      max[axis] = Math.max(max[axis], point[axis]);
    }
  }
  return { min, max };
}

function boundsCenter(bounds) {
  const min = bounds.min;
  const max = bounds.max;
  return [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2
  ];
}

function boundsRadius(bounds) {
  const min = bounds.min;
  const max = bounds.max;
  const dx = max[0] - min[0];
  const dy = max[1] - min[1];
  const dz = max[2] - min[2];
  return Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz) / 2, 0.001);
}

function sameVector(a, b, epsilon = 1e-9) {
  return (
    Math.abs(a[0] - b[0]) <= epsilon &&
    Math.abs(a[1] - b[1]) <= epsilon &&
    Math.abs(a[2] - b[2]) <= epsilon
  );
}

function rotateFrameVector(vector) {
  const rotation = currentRotationMatrix();
  return [
    rotation[0] * vector[0] + rotation[4] * vector[1] + rotation[8] * vector[2],
    rotation[1] * vector[0] + rotation[5] * vector[1] + rotation[9] * vector[2],
    rotation[2] * vector[0] + rotation[6] * vector[1] + rotation[10] * vector[2]
  ];
}

function currentCameraDistance(radius = state.frameRadius || 1) {
  return Math.max(radius * 2.8 / Math.max(state.zoom, 1e-9), radius * 1e-9);
}

function compensatePivotChange(oldPivot, nextPivot, radius) {
  const delta = [
    nextPivot[0] - oldPivot[0],
    nextPivot[1] - oldPivot[1],
    nextPivot[2] - oldPivot[2]
  ];
  if (sameVector(delta, [0, 0, 0])) {
    return;
  }
  const rotatedDelta = rotateFrameVector(delta);
  const viewShift = [
    delta[0] - rotatedDelta[0],
    delta[1] - rotatedDelta[1],
    delta[2] - rotatedDelta[2]
  ];
  state.panX -= viewShift[0];
  state.panY -= viewShift[1];

  const oldDistance = currentCameraDistance(radius);
  const nextDistance = oldDistance + viewShift[2];
  const minDistance = radius * 1e-9;
  if (Number.isFinite(nextDistance) && nextDistance > minDistance) {
    state.zoom = Math.max(1e-9, (radius * 2.8) / nextDistance);
  }
}

function setViewFrame(frameBounds, orbitBounds = frameBounds, options = {}) {
  const frameCenter = boundsCenter(frameBounds);
  const orbitCenter = boundsCenter(orbitBounds);
  const frameRadius = boundsRadius(frameBounds);
  const nextOrbitOffset = [
    orbitCenter[0] - frameCenter[0],
    orbitCenter[1] - frameCenter[1],
    orbitCenter[2] - frameCenter[2]
  ];
  const canPreserveRegistration =
    options.preserveRegistration !== false &&
    state.renderKey &&
    sameVector(state.frameCenter, frameCenter) &&
    Math.abs((state.frameRadius || 1) - frameRadius) <= Math.max(frameRadius, 1) * 1e-9;
  if (canPreserveRegistration) {
    compensatePivotChange(state.orbitOffset || [0, 0, 0], nextOrbitOffset, frameRadius);
  }
  state.frameCenter = frameCenter;
  state.frameRadius = frameRadius;
  state.orbitOffset = nextOrbitOffset;
}

function framePoint(point) {
  return [
    point[0] - state.frameCenter[0],
    point[1] - state.frameCenter[1],
    point[2] - state.frameCenter[2]
  ];
}

function activeOrbitCenter() {
  return [
    state.frameCenter[0] + state.orbitOffset[0],
    state.frameCenter[1] + state.orbitOffset[1],
    state.frameCenter[2] + state.orbitOffset[2]
  ];
}

function normalDisplayScale() {
  const value = Number(el.normalScale?.value || 1);
  return Math.min(10, Math.max(0.1, Number.isFinite(value) ? value : 1));
}

function updateNormalScaleValue() {
  if (el.normalScaleValue) {
    el.normalScaleValue.textContent = `${normalDisplayScale().toFixed(2)}x`;
  }
}

function normalDebugMode() {
  return new URLSearchParams(window.location.search).get("debugNormals") || "";
}

function syntheticNormalSegments() {
  const length = Math.max(state.frameRadius * 0.35, 0.01);
  const center = activeOrbitCenter();
  return [
    [center, [center[0] + length, center[1], center[2]]],
    [center, [center[0], center[1] + length, center[2]]],
    [center, [center[0], center[1], center[2] + length]]
  ];
}

function displayNormalSegments(rawSegments) {
  const mode = normalDebugMode().toLowerCase();
  if (mode === "synthetic") {
    return syntheticNormalSegments();
  }
  if (mode === "append") {
    return [...rawSegments, ...syntheticNormalSegments()];
  }
  return rawSegments;
}

function uploadNormalSegmentsToGPU() {
  const gl = state.gl;
  if (
    !gl ||
    !state.normalPositionBuffer ||
    !state.normalColorBuffer ||
    !state.view ||
    state.view.kind !== "pointCloud"
  ) {
    return;
  }
  const rawSegments = state.view.normal_segments || [];
  const segments = displayNormalSegments(rawSegments);
  const displayScale = normalDisplayScale();
  state.normalLineCount = segments.length * 2;
  if (!segments.length) {
    gl.bindBuffer(gl.ARRAY_BUFFER, state.normalPositionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, state.normalColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(), gl.STATIC_DRAW);
    return;
  }
  const positions = new Float32Array(segments.length * 6);
  const colors = new Float32Array(segments.length * 6);
  for (let i = 0; i < segments.length; i += 1) {
    const start = segments[i][0];
    const end = segments[i][1];
    const scaledEnd = [
      start[0] + (end[0] - start[0]) * displayScale,
      start[1] + (end[1] - start[1]) * displayScale,
      start[2] + (end[2] - start[2]) * displayScale
    ];
    const framedStart = framePoint(start);
    const framedEnd = framePoint(scaledEnd);
    positions[i * 6] = framedStart[0];
    positions[i * 6 + 1] = framedStart[1];
    positions[i * 6 + 2] = framedStart[2];
    positions[i * 6 + 3] = framedEnd[0];
    positions[i * 6 + 4] = framedEnd[1];
    positions[i * 6 + 5] = framedEnd[2];
    colors.set([0.0, 0.98, 1.0, 0.0, 0.98, 1.0], i * 6);
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, state.normalPositionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.normalColorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
  console.info("[Rock3D] normal overlay", {
    rawSegments: rawSegments.length,
    renderedSegments: segments.length,
    displayScale,
    firstSegment: segments[0] || null,
    diagnostics: state.view.normal_diagnostics || null,
    debugMode: normalDebugMode() || null
  });
}

function uploadPointCloudToGPU() {
  if (!state.view || state.view.kind !== "pointCloud") {
    return;
  }
  const gl = ensureGL();
  if (!gl) {
    drawText("WebGL is not available in this browser.");
    return;
  }
  const points = state.view.points || [];
  const colors = state.view.colors || [];
  const normals = state.view.normals || [];
  const hasNormals = normals.length === points.length;
  const indices = state.view.indices || [];
  const fallbackBounds = state.view.bounds || viewBounds(points);
  setViewFrame(state.view.scene_bounds || fallbackBounds, fallbackBounds);
  state.pointCount = points.length;
  state.pointNormalCount = hasNormals ? points.length : 0;
  state.meshVertexCount = 0;
  state.meshLineVertexCount = 0;
  state.sourceIndices = indices;
  state.centeredPositions = new Float32Array(points.length * 3);
  const packedColors = new Float32Array(points.length * 3);
  const packedNormals = new Float32Array(points.length * 3);
  const pickColors = new Float32Array(points.length * 3);
  for (let i = 0; i < points.length; i += 1) {
    const point = points[i];
    const color = colors[i] || [0.5, 0.5, 0.5];
    const normal = hasNormals ? normalize(normals[i] || [0, 0, 1]) : [0, 0, 1];
    const encoded = i + 1;
    const framedPoint = framePoint(point);
    state.centeredPositions[i * 3] = framedPoint[0];
    state.centeredPositions[i * 3 + 1] = framedPoint[1];
    state.centeredPositions[i * 3 + 2] = framedPoint[2];
    packedColors[i * 3] = color[0];
    packedColors[i * 3 + 1] = color[1];
    packedColors[i * 3 + 2] = color[2];
    packedNormals[i * 3] = normal[0];
    packedNormals[i * 3 + 1] = normal[1];
    packedNormals[i * 3 + 2] = normal[2];
    pickColors[i * 3] = ((encoded >> 16) & 255) / 255;
    pickColors[i * 3 + 1] = ((encoded >> 8) & 255) / 255;
    pickColors[i * 3 + 2] = (encoded & 255) / 255;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, state.positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, state.centeredPositions, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, packedColors, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.pointNormalBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, packedNormals, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.pickColorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, pickColors, gl.STATIC_DRAW);
  state.renderKey = state.view;
  uploadNormalSegmentsToGPU();
  uploadMarkersToGPU();
}

function markerSourcePoints() {
  if (!state.view || state.view.kind !== "pointCloud") {
    return [];
  }
  const points = [];
  const indexToPoint = new Map();
  for (let i = 0; i < state.view.indices.length; i += 1) {
    indexToPoint.set(state.view.indices[i], state.view.points[i]);
  }

  function pushSelection(indices, color) {
    for (const selected of indices) {
      const point = indexToPoint.get(selected);
      if (point) {
        points.push({ point, color, haloColor: [0.02, 0.02, 0.02] });
      }
    }
  }

  pushSelection(state.rockSeeds, [1, 0.05, 0.02]);
  pushSelection(state.pedestalSeeds, [0.0, 0.24, 1.0]);
  pushSelection(state.interfacePoints, [0.0, 0.9, 0.45]);
  for (const part of state.interfaceParts) {
    pushSelection(part.selected_indices || [], part.is_lateral ? [1.0, 0.86, 0.0] : [0.0, 0.9, 0.45]);
  }
  if (state.activeView === "mesh_prepared" && state.manualRemovalSelected.length) {
    pushSelection(state.manualRemovalSelected, [1.0, 0.84, 0.0]);
  }

  for (const marker of state.view.markers || []) {
    points.push({ point: marker.point, color: marker.color || [1, 0, 0], haloColor: [1, 1, 1] });
  }
  return points;
}

function uploadMarkersToGPU() {
  const gl = state.gl;
  if (!gl || !state.view || state.view.kind !== "pointCloud") {
    return;
  }
  const markers = markerSourcePoints();
  state.markerCount = markers.length;
  state.markerHaloCount = markers.length;
  state.markerPositions = new Float32Array(markers.length * 3);
  state.markerColors = new Float32Array(markers.length * 3);
  state.markerHaloPositions = new Float32Array(markers.length * 3);
  state.markerHaloColors = new Float32Array(markers.length * 3);
  for (let i = 0; i < markers.length; i += 1) {
    const point = markers[i].point;
    const color = markers[i].color;
    const haloColor = markers[i].haloColor || [0.02, 0.02, 0.02];
    const framedPoint = framePoint(point);
    state.markerPositions[i * 3] = framedPoint[0];
    state.markerPositions[i * 3 + 1] = framedPoint[1];
    state.markerPositions[i * 3 + 2] = framedPoint[2];
    state.markerColors[i * 3] = color[0];
    state.markerColors[i * 3 + 1] = color[1];
    state.markerColors[i * 3 + 2] = color[2];
    state.markerHaloPositions[i * 3] = state.markerPositions[i * 3];
    state.markerHaloPositions[i * 3 + 1] = state.markerPositions[i * 3 + 1];
    state.markerHaloPositions[i * 3 + 2] = state.markerPositions[i * 3 + 2];
    state.markerHaloColors[i * 3] = haloColor[0];
    state.markerHaloColors[i * 3 + 1] = haloColor[1];
    state.markerHaloColors[i * 3 + 2] = haloColor[2];
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, state.markerHaloPositionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, state.markerHaloPositions, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.markerHaloColorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, state.markerHaloColors, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.markerPositionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, state.markerPositions, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.markerColorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, state.markerColors, gl.DYNAMIC_DRAW);
}

function computeMatrices(size) {
  const radius = state.frameRadius || 1;
  const distance = currentCameraDistance(radius);
  const near = Math.max(Math.min(distance * 0.05, radius * 0.001), radius * 1e-9, 1e-9);
  const far = Math.max(distance + radius * 8, radius * 0.01);
  const projection = mat4Perspective(Math.PI / 4, size.width / size.height, near, far);
  const pivot = state.orbitOffset || [0, 0, 0];
  const rotation = currentRotationMatrix();
  const pivotedRotation = mat4Multiply(
    mat4Translate(pivot[0], pivot[1], pivot[2]),
    mat4Multiply(rotation, mat4Translate(-pivot[0], -pivot[1], -pivot[2]))
  );
  const modelView = mat4Multiply(
    mat4Translate(state.panX, state.panY, -distance),
    pivotedRotation
  );
  return mat4Multiply(projection, modelView);
}

function panScale(size) {
  const radius = state.frameRadius || 1;
  const distance = currentCameraDistance(radius);
  const visibleHeight = 2 * distance * Math.tan(Math.PI / 8);
  return visibleHeight / Math.max(size.cssHeight || size.height || 1, 1);
}

function bindAttribute(gl, program, buffer, name, size) {
  const location = gl.getAttribLocation(program, name);
  if (location < 0) {
    return;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
}

function bindOptionalNormalAttribute(gl, program, normalBuffer) {
  const location = gl.getAttribLocation(program, "a_normal");
  if (location < 0) {
    return;
  }
  if (!normalBuffer) {
    gl.disableVertexAttribArray(location);
    gl.vertexAttrib3f(location, 0, 0, 1);
    return;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, 3, gl.FLOAT, false, 0, 0);
}

function computeNormalMatrix() {
  const rotation = currentRotationMatrix();
  return new Float32Array([
    rotation[0], rotation[1], rotation[2],
    rotation[4], rotation[5], rotation[6],
    rotation[8], rotation[9], rotation[10]
  ]);
}

function drawPointBatch(gl, program, positionBuffer, colorBuffer, colorAttributeName, count, pointSize, normalBuffer = null, shaded = false) {
  if (!count) {
    return;
  }
  bindAttribute(gl, program, positionBuffer, "a_position", 3);
  bindAttribute(gl, program, colorBuffer, colorAttributeName, 3);
  bindOptionalNormalAttribute(gl, program, shaded ? normalBuffer : null);
  const pointSizeLocation = gl.getUniformLocation(program, "u_pointSize");
  if (pointSizeLocation) {
    gl.uniform1f(pointSizeLocation, pointSize);
  }
  const hasNormalsLocation = gl.getUniformLocation(program, "u_hasNormals");
  if (hasNormalsLocation) {
    gl.uniform1i(hasNormalsLocation, shaded && normalBuffer ? 1 : 0);
  }
  const normalMatrixLocation = gl.getUniformLocation(program, "u_normalMatrix");
  if (normalMatrixLocation) {
    gl.uniformMatrix3fv(normalMatrixLocation, false, computeNormalMatrix());
  }
  const lightLocation = gl.getUniformLocation(program, "u_lightDirection");
  if (lightLocation) {
    gl.uniform3f(lightLocation, 0.28, -0.38, 0.88);
  }
  gl.drawArrays(gl.POINTS, 0, count);
}

function drawSolidBatch(gl, positionBuffer, colorBuffer, count, mode) {
  if (!count || !state.lineProgram) {
    return;
  }
  gl.useProgram(state.lineProgram);
  gl.uniformMatrix4fv(gl.getUniformLocation(state.lineProgram, "u_matrix"), false, state.mvpMatrix);
  bindAttribute(gl, state.lineProgram, positionBuffer, "a_position", 3);
  bindAttribute(gl, state.lineProgram, colorBuffer, "a_color", 3);
  gl.drawArrays(mode, 0, count);
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalize(vector) {
  const length = Math.hypot(vector[0], vector[1], vector[2]) || 1;
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizeRotationMatrix(matrix) {
  const xAxis = normalize([matrix[0], matrix[1], matrix[2]]);
  const rawY = [matrix[4], matrix[5], matrix[6]];
  const yWithoutX = [
    rawY[0] - xAxis[0] * dot(rawY, xAxis),
    rawY[1] - xAxis[1] * dot(rawY, xAxis),
    rawY[2] - xAxis[2] * dot(rawY, xAxis)
  ];
  const yAxis = normalize(yWithoutX);
  const zAxis = normalize(cross(xAxis, yAxis));
  return new Float32Array([
    xAxis[0], xAxis[1], xAxis[2], 0,
    yAxis[0], yAxis[1], yAxis[2], 0,
    zAxis[0], zAxis[1], zAxis[2], 0,
    0, 0, 0, 1
  ]);
}

function resetRotationMatrix() {
  state.rotationMatrix = null;
  state.trackballVector = null;
  currentRotationMatrix();
}

function currentRotationMatrix() {
  if (!state.rotationMatrix) {
    state.rotationMatrix = mat4Multiply(mat4RotateX(state.rotationX), mat4RotateY(state.rotationY));
  }
  return state.rotationMatrix;
}

function trackballCenter(size) {
  const rect = el.viewer.getBoundingClientRect();
  const fallback = {
    x: rect.width / 2,
    y: rect.height / 2
  };
  const pivot = state.orbitOffset || [0, 0, 0];
  let clip;
  try {
    clip = transformPoint(computeMatrices(size), pivot[0], pivot[1], pivot[2]);
  } catch {
    return fallback;
  }
  if (!clip || !Number.isFinite(clip[3]) || Math.abs(clip[3]) < 1e-9) {
    return fallback;
  }
  const ndcX = clip[0] / clip[3];
  const ndcY = clip[1] / clip[3];
  const projected = {
    x: (ndcX * 0.5 + 0.5) * rect.width,
    y: (-ndcY * 0.5 + 0.5) * rect.height
  };
  if (!Number.isFinite(projected.x) || !Number.isFinite(projected.y)) {
    return fallback;
  }
  return {
    x: clamp(projected.x, rect.width * 0.25, rect.width * 0.75),
    y: clamp(projected.y, rect.height * 0.25, rect.height * 0.75)
  };
}

function virtualTrackballVector(clientX, clientY) {
  const rect = el.viewer.getBoundingClientRect();
  const size = canvasSize();
  const center = trackballCenter(size);
  const radius = Math.max(Math.min(rect.width, rect.height) * 0.48, 1);
  const x = (clientX - rect.left - center.x) / radius;
  const y = (center.y - (clientY - rect.top)) / radius;
  const distanceSq = x * x + y * y;
  if (distanceSq <= 1) {
    return normalize([x, y, Math.sqrt(1 - distanceSq)]);
  }
  return normalize([x, y, 0]);
}

function rotationBetweenVectors(previous, current) {
  const axis = cross(previous, current);
  const axisLength = Math.hypot(axis[0], axis[1], axis[2]);
  const amount = clamp(dot(previous, current), -1, 1);
  if (axisLength < 1e-9) {
    return null;
  }
  return mat4RotateAxis(axis, Math.atan2(axisLength, amount));
}

function applyTrackballRotation(clientX, clientY) {
  const current = virtualTrackballVector(clientX, clientY);
  if (state.trackballVector) {
    const delta = rotationBetweenVectors(state.trackballVector, current);
    if (delta) {
      state.rotationMatrix = normalizeRotationMatrix(mat4Multiply(delta, currentRotationMatrix()));
    }
  }
  state.trackballVector = current;
}

function uploadMeshToGPU() {
  if (!state.view || state.view.kind !== "mesh") {
    return;
  }
  const gl = ensureGL();
  if (!gl) {
    drawText("WebGL is not available in this browser.");
    return;
  }
  const vertices = state.view.vertices || [];
  const triangles = state.view.triangles || [];
  if (!vertices.length || !triangles.length) {
    drawText("Mesh is ready. Use the Mesh PLY download link.");
    return;
  }

  const fallbackBounds = state.view.bounds || viewBounds(vertices);
  setViewFrame(state.view.scene_bounds || fallbackBounds, fallbackBounds);
  state.pointCount = 0;
  state.pointNormalCount = 0;
  state.normalLineCount = 0;
  state.markerCount = 0;
  state.markerHaloCount = 0;
  state.centeredPositions = null;
  state.sourceIndices = [];

  const trianglePositions = new Float32Array(triangles.length * 9);
  const triangleColors = new Float32Array(triangles.length * 9);
  const wirePositions = new Float32Array(triangles.length * 18);
  const wireColors = new Float32Array(triangles.length * 18);
  const light = normalize([0.25, -0.45, 0.86]);
  const base = [0.74, 0.36, 0.3];
  const wire = [0.13, 0.17, 0.17];

  for (let triIndex = 0; triIndex < triangles.length; triIndex += 1) {
    const tri = triangles[triIndex];
    const a = vertices[tri[0]];
    const b = vertices[tri[1]];
    const c = vertices[tri[2]];
    if (!a || !b || !c) {
      continue;
    }
    const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const normal = normalize(cross(ab, ac));
    const shade = 0.42 + 0.58 * Math.max(0, normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2]);
    const color = base.map((value) => Math.min(1, value * shade + 0.08));
    const points = [a, b, c];
    for (let corner = 0; corner < 3; corner += 1) {
      const point = points[corner];
      const offset = triIndex * 9 + corner * 3;
      const framedPoint = framePoint(point);
      trianglePositions[offset] = framedPoint[0];
      trianglePositions[offset + 1] = framedPoint[1];
      trianglePositions[offset + 2] = framedPoint[2];
      triangleColors[offset] = color[0];
      triangleColors[offset + 1] = color[1];
      triangleColors[offset + 2] = color[2];
    }
    const edges = [a, b, b, c, c, a];
    for (let edgeIndex = 0; edgeIndex < edges.length; edgeIndex += 1) {
      const point = edges[edgeIndex];
      const offset = triIndex * 18 + edgeIndex * 3;
      const framedPoint = framePoint(point);
      wirePositions[offset] = framedPoint[0];
      wirePositions[offset + 1] = framedPoint[1];
      wirePositions[offset + 2] = framedPoint[2];
      wireColors[offset] = wire[0];
      wireColors[offset + 1] = wire[1];
      wireColors[offset + 2] = wire[2];
    }
  }

  state.meshVertexCount = triangles.length * 3;
  state.meshLineVertexCount = triangles.length * 6;
  gl.bindBuffer(gl.ARRAY_BUFFER, state.meshPositionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, trianglePositions, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.meshColorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, triangleColors, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.meshLinePositionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, wirePositions, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.meshLineColorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, wireColors, gl.STATIC_DRAW);
  state.renderKey = state.view;
}

function draw() {
  if (!state.view) {
    drawText("Upload a LAS or LAZ file to begin.");
    drawManualRemovalOverlay();
    return;
  }
  if (state.view.kind === "mesh") {
    if (state.renderKey !== state.view) {
      uploadMeshToGPU();
    }
    const gl = state.gl;
    if (!gl || !state.meshVertexCount) {
      return;
    }
    const size = canvasSize();
    gl.viewport(0, 0, size.width, size.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    state.mvpMatrix = computeMatrices(size);
    drawSolidBatch(gl, state.meshPositionBuffer, state.meshColorBuffer, state.meshVertexCount, gl.TRIANGLES);
    if (state.view.show_wireframe) {
      drawSolidBatch(gl, state.meshLinePositionBuffer, state.meshLineColorBuffer, state.meshLineVertexCount, gl.LINES);
    }
    drawManualRemovalOverlay();
    return;
  }
  if (!state.view.points || !state.view.points.length) {
    drawText("No points available for this view.");
    drawManualRemovalOverlay();
    return;
  }

  if (state.renderKey !== state.view) {
    uploadPointCloudToGPU();
  } else {
    uploadMarkersToGPU();
  }
  const gl = state.gl;
  if (!gl) {
    return;
  }
  const size = canvasSize();
  gl.viewport(0, 0, size.width, size.height);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(state.program);
  state.mvpMatrix = computeMatrices(size);
  gl.uniformMatrix4fv(gl.getUniformLocation(state.program, "u_matrix"), false, state.mvpMatrix);

  const pointSize = Number(el.pointSize.value || 2.5);
  drawPointBatch(
    gl,
    state.program,
    state.positionBuffer,
    state.colorBuffer,
    "a_color",
    state.pointCount,
    pointSize,
    state.pointNormalCount === state.pointCount ? state.pointNormalBuffer : null,
    state.pointNormalCount === state.pointCount
  );
  gl.disable(gl.DEPTH_TEST);
  gl.lineWidth(2);
  drawSolidBatch(gl, state.normalPositionBuffer, state.normalColorBuffer, state.normalLineCount, gl.LINES);
  gl.enable(gl.DEPTH_TEST);
  gl.useProgram(state.program);
  gl.uniformMatrix4fv(gl.getUniformLocation(state.program, "u_matrix"), false, state.mvpMatrix);
  drawPointBatch(gl, state.program, state.markerHaloPositionBuffer, state.markerHaloColorBuffer, "a_color", state.markerHaloCount, Math.max(pointSize + 16, 18), null, false);
  drawPointBatch(gl, state.program, state.markerPositionBuffer, state.markerColorBuffer, "a_color", state.markerCount, Math.max(pointSize + 9, 12), null, false);
  drawManualRemovalOverlay();
}

function renderPickBuffer(size) {
  const gl = state.gl;
  if (!gl || !state.pickProgram || !state.centeredPositions || !ensurePickFramebuffer(size.width, size.height)) {
    return false;
  }
  state.mvpMatrix = computeMatrices(size);
  gl.bindFramebuffer(gl.FRAMEBUFFER, state.pickFramebuffer);
  gl.viewport(0, 0, size.width, size.height);
  gl.disable(gl.BLEND);
  gl.enable(gl.DEPTH_TEST);
  gl.depthMask(true);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(state.pickProgram);
  gl.uniformMatrix4fv(gl.getUniformLocation(state.pickProgram, "u_matrix"), false, state.mvpMatrix);
  const pointSize = Number(el.pointSize.value || 3.5);
  drawPointBatch(gl, state.pickProgram, state.positionBuffer, state.pickColorBuffer, "a_pickColor", state.pointCount, pointSize);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.clearColor(0.965, 0.975, 0.957, 1);
  gl.enable(gl.BLEND);
  return true;
}

function pickFromProjection(event) {
  if (!state.view || state.view.kind !== "pointCloud" || !state.centeredPositions || !state.mvpMatrix) {
    return;
  }
  const rect = el.viewer.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  let bestIndex = null;
  let bestDepth = Infinity;
  let bestDist = Infinity;
  const positions = state.centeredPositions;
  const pickRadius = Math.max(10, Number(el.pointSize.value || 3.5) + 8);
  const pickRadiusSq = pickRadius * pickRadius;
  for (let i = 0; i < state.pointCount; i += 1) {
    const clip = transformPoint(state.mvpMatrix, positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    if (clip[3] <= 0) {
      continue;
    }
    const ndcX = clip[0] / clip[3];
    const ndcY = clip[1] / clip[3];
    const ndcZ = clip[2] / clip[3];
    if (ndcX < -1 || ndcX > 1 || ndcY < -1 || ndcY > 1 || ndcZ < -1 || ndcZ > 1) {
      continue;
    }
    const sx = (ndcX * 0.5 + 0.5) * rect.width;
    const sy = (-ndcY * 0.5 + 0.5) * rect.height;
    const dx = sx - x;
    const dy = sy - y;
    const dist = dx * dx + dy * dy;
    if (dist > pickRadiusSq) {
      continue;
    }
    const depth = ndcZ;
    if (depth < bestDepth - 0.002 || (Math.abs(depth - bestDepth) <= 0.002 && dist < bestDist)) {
      bestDepth = depth;
      bestDist = dist;
      bestIndex = state.sourceIndices[i];
    }
  }
  if (bestIndex !== null) {
    return bestIndex;
  }
  return null;
}

function selectedPointFromProjection(event) {
  const selection = currentSelection();
  if (
    !selection.length ||
    !state.view ||
    state.view.kind !== "pointCloud" ||
    !state.centeredPositions ||
    !state.mvpMatrix
  ) {
    return null;
  }
  const selected = new Set(selection);
  const rect = el.viewer.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const positions = state.centeredPositions;
  const pickRadius = Math.max(18, Number(el.pointSize.value || 3.5) + 14);
  const pickRadiusSq = pickRadius * pickRadius;
  let bestIndex = null;
  let bestDepth = Infinity;
  let bestDist = Infinity;

  for (let i = 0; i < state.pointCount; i += 1) {
    const sourceIndex = state.sourceIndices[i];
    if (!selected.has(sourceIndex)) {
      continue;
    }
    const clip = transformPoint(state.mvpMatrix, positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    if (clip[3] <= 0) {
      continue;
    }
    const ndcX = clip[0] / clip[3];
    const ndcY = clip[1] / clip[3];
    const ndcZ = clip[2] / clip[3];
    if (ndcX < -1 || ndcX > 1 || ndcY < -1 || ndcY > 1 || ndcZ < -1 || ndcZ > 1) {
      continue;
    }
    const sx = (ndcX * 0.5 + 0.5) * rect.width;
    const sy = (-ndcY * 0.5 + 0.5) * rect.height;
    const dx = sx - x;
    const dy = sy - y;
    const dist = dx * dx + dy * dy;
    if (dist > pickRadiusSq) {
      continue;
    }
    if (dist < bestDist - 0.001 || (Math.abs(dist - bestDist) <= 0.001 && ndcZ < bestDepth)) {
      bestDepth = ndcZ;
      bestDist = dist;
      bestIndex = sourceIndex;
    }
  }
  return bestIndex;
}

function unselectFromCanvas(event) {
  if (state.renderKey !== state.view) {
    uploadPointCloudToGPU();
  }
  if (!state.mvpMatrix) {
    state.mvpMatrix = computeMatrices(canvasSize());
  }
  const sourceIndex = selectedPointFromProjection(event);
  if (sourceIndex !== null) {
    removePoint(sourceIndex);
  }
}

function pickFromCanvas(event) {
  if (!state.view || state.view.kind !== "pointCloud" || !state.centeredPositions) {
    return;
  }
  if (state.renderKey !== state.view) {
    uploadPointCloudToGPU();
  }
  const rect = el.viewer.getBoundingClientRect();
  const size = canvasSize();
  if (!renderPickBuffer(size)) {
    const fallbackIndex = pickFromProjection(event);
    if (fallbackIndex !== null) {
      pickPoint(fallbackIndex);
    }
    draw();
    return;
  }

  const gl = state.gl;
  const dprX = size.width / Math.max(rect.width, 1);
  const dprY = size.height / Math.max(rect.height, 1);
  const targetX = Math.round((event.clientX - rect.left) * dprX);
  const targetY = Math.round((rect.bottom - event.clientY) * dprY);
  const readRadius = Math.ceil(Math.max(5, Number(el.pointSize.value || 3.5) * Math.max(dprX, dprY) + 4));
  const startX = Math.max(0, targetX - readRadius);
  const startY = Math.max(0, targetY - readRadius);
  const endX = Math.min(size.width - 1, targetX + readRadius);
  const endY = Math.min(size.height - 1, targetY + readRadius);
  const width = Math.max(1, endX - startX + 1);
  const height = Math.max(1, endY - startY + 1);
  const pixels = new Uint8Array(width * height * 4);

  gl.bindFramebuffer(gl.FRAMEBUFFER, state.pickFramebuffer);
  gl.readPixels(startX, startY, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  let bestRenderIndex = null;
  let bestDist = Infinity;
  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      const offset = (row * width + col) * 4;
      const encoded = (pixels[offset] << 16) | (pixels[offset + 1] << 8) | pixels[offset + 2];
      if (encoded === 0) {
        continue;
      }
      const pixelX = startX + col;
      const pixelY = startY + row;
      const dx = pixelX - targetX;
      const dy = pixelY - targetY;
      const dist = dx * dx + dy * dy;
      if (dist < bestDist) {
        bestDist = dist;
        bestRenderIndex = encoded - 1;
      }
    }
  }

  if (bestRenderIndex === null) {
    const fallbackIndex = pickFromProjection(event);
    if (fallbackIndex !== null) {
      pickPoint(fallbackIndex);
      return;
    }
    draw();
    return;
  }

  const sourceIndex = state.sourceIndices[bestRenderIndex];
  if (sourceIndex !== undefined) {
    pickPoint(sourceIndex);
  } else {
    draw();
  }
}

function bindEvents() {
  wrapHelpButtons();
  document.querySelectorAll(".info-button, .button-tooltip-wrap").forEach(bindTooltipTarget);
  window.addEventListener("scroll", hideInfoTooltip, true);

  el.toggleTips.addEventListener("click", toggleHoverTips);
  updateHoverTipsToggle();
  el.fileInput.addEventListener("change", (event) => uploadFile(event.target.files[0]));
  el.pointSize.addEventListener("input", draw);
  el.normalScale.addEventListener("input", () => {
    updateNormalScaleValue();
    if (state.activeView === "mesh_prepared" && state.view?.kind === "pointCloud") {
      state.renderKey = null;
      updateViewMeta();
    }
    draw();
  });
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => loadView(button.dataset.view));
  });

  el.pickRock.addEventListener("click", () => setPickMode("rock"));
  el.pickPedestal.addEventListener("click", () => setPickMode("pedestal"));
  el.pickInterface.addEventListener("click", () => setPickMode("interface"));
  el.closeInterfaceWindow.addEventListener("click", hideInterfaceWindow);
  el.interfaceWindowHandle.addEventListener("pointerdown", startInterfaceWindowDrag);
  el.interfaceWindowHandle.addEventListener("pointermove", dragInterfaceWindow);
  el.interfaceWindowHandle.addEventListener("pointerup", stopInterfaceWindowDrag);
  el.interfaceWindowHandle.addEventListener("pointercancel", stopInterfaceWindowDrag);
  el.manualRemoval.addEventListener("click", openManualRemovalWindow);
  el.closeManualRemovalWindow.addEventListener("click", hideManualRemovalWindow);
  el.manualRemovalWindowHandle.addEventListener("pointerdown", startManualRemovalWindowDrag);
  el.manualRemovalWindowHandle.addEventListener("pointermove", dragManualRemovalWindow);
  el.manualRemovalWindowHandle.addEventListener("pointerup", stopManualRemovalWindowDrag);
  el.manualRemovalWindowHandle.addEventListener("pointercancel", stopManualRemovalWindowDrag);
  el.manualRemovalDraw.addEventListener("click", toggleManualRemovalDraw);
  el.manualRemovalUndoVertex.addEventListener("click", undoManualRemovalVertex);
  el.manualRemovalClear.addEventListener("click", clearManualRemovalSelection);
  el.manualRemovalApply.addEventListener("click", applyManualRemoval);
  el.manualRemovalClose.addEventListener("click", hideManualRemovalWindow);
  el.autoSeeds.addEventListener("click", autoSeeds);
  el.clearCurrentPick.addEventListener("click", clearCurrentSelectionMode);
  el.stagePart.addEventListener("click", stagePart);
  el.interpolateInterface.addEventListener("click", interpolateInterface);
  el.saveInterface.addEventListener("click", saveInterface);
  el.clearParts.addEventListener("click", clearInterfaceParts);
  el.runSegment.addEventListener("click", async () => {
    await flushSeedAutosave();
    await runAction(
      "Running region growing",
      `/api/sessions/${state.session.session_id}/segment`,
      segmentParams(),
      (result) => result?.auto_interface_generated ? "interface" : "segmented"
    );
  });
  el.prepareMesh.addEventListener("click", async () => {
    const job = await runAction("Preparing mesh", `/api/sessions/${state.session.session_id}/mesh/prepare`, undefined, "mesh_prepared");
    if (job) {
      clearManualRemovalSelection();
    }
  });
  el.removeNoise.addEventListener("click", async () => {
    const job = await runAction("Denoising", `/api/sessions/${state.session.session_id}/mesh/noise/remove`, denoiseParams(), "mesh_prepared");
    if (job) {
      clearManualRemovalSelection();
    }
  });
  el.undoNoise.addEventListener("click", async () => {
    const job = await runAction("Undoing noise", `/api/sessions/${state.session.session_id}/mesh/noise/undo`, undefined, "mesh_prepared");
    if (job) {
      clearManualRemovalSelection();
    }
  });
  el.computeNormals.addEventListener("click", () => runAction("Computing normals", `/api/sessions/${state.session.session_id}/mesh/normals`, {
    method: el.normalMethod.value,
    k: Number(el.normalK.value)
  }, "mesh_prepared"));
  el.reconstruct.addEventListener("click", () => runAction("Reconstructing mesh", `/api/sessions/${state.session.session_id}/mesh/reconstruct`, {
    depth: Number(el.meshDepth.value)
  }, "mesh"));
  el.analyze.addEventListener("click", () => runAction("Computing analysis", `/api/sessions/${state.session.session_id}/analysis`));

  el.viewer.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });
  el.viewer.addEventListener("pointerdown", (event) => {
    if (![0, 1, 2].includes(event.button)) {
      return;
    }
    event.preventDefault();
    state.dragging = true;
    if (state.manualRemovalDrawMode && state.activeView === "mesh_prepared" && event.button === 0) {
      state.dragMode = "manual_removal_vertex";
    } else if (event.shiftKey && event.button === 0) {
      state.dragMode = "select";
    } else if (event.shiftKey && event.button === 2) {
      state.dragMode = "unselect";
    } else {
      state.dragMode = event.button === 1 || event.button === 2 ? "pan" : "rotate";
    }
    state.dragStart = [event.clientX, event.clientY];
    state.lastPointer = [event.clientX, event.clientY];
    state.trackballVector = state.dragMode === "rotate"
      ? virtualTrackballVector(event.clientX, event.clientY)
      : null;
    el.viewer.setPointerCapture(event.pointerId);
  });
  el.viewer.addEventListener("pointermove", (event) => {
    if (!state.dragging || !state.lastPointer) {
      return;
    }
    const dx = event.clientX - state.lastPointer[0];
    const dy = event.clientY - state.lastPointer[1];
    if (state.dragMode === "pan") {
      const scale = panScale(canvasSize());
      state.panX += dx * scale;
      state.panY -= dy * scale;
    } else if (state.dragMode === "rotate") {
      applyTrackballRotation(event.clientX, event.clientY);
    }
    state.lastPointer = [event.clientX, event.clientY];
    draw();
  });
  el.viewer.addEventListener("pointerup", (event) => {
    const start = state.dragStart;
    const mode = state.dragMode;
    state.dragging = false;
    state.dragMode = null;
    state.lastPointer = null;
    state.trackballVector = null;
    if (start) {
      const dx = event.clientX - start[0];
      const dy = event.clientY - start[1];
      if (mode === "manual_removal_vertex" && (dx * dx + dy * dy) < 25) {
        state.manualRemovalPolygon.push(viewerPointFromEvent(event));
        if (state.manualRemovalPolygon.length >= 3) {
          collectManualRemovalSelection();
        } else {
          state.manualRemovalSelected = [];
          updateManualRemovalUI();
        }
      } else if (mode === "select" && (dx * dx + dy * dy) < 25) {
        pickFromCanvas(event);
      } else if (mode === "unselect" && (dx * dx + dy * dy) < 25) {
        unselectFromCanvas(event);
      }
    }
    state.dragStart = null;
  });
  el.viewer.addEventListener("pointercancel", () => {
    state.dragging = false;
    state.dragMode = null;
    state.dragStart = null;
    state.lastPointer = null;
    state.trackballVector = null;
  });
  el.viewer.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.zoom *= Math.exp(-event.deltaY * 0.0015);
    state.zoom = Math.max(1e-9, state.zoom);
    draw();
  }, { passive: false });
  window.addEventListener("resize", draw);
}

async function main() {
  initElements();
  updateNormalScaleValue();
  bindEvents();
  draw();
  try {
    setBusy("Starting session");
    await createSession();
    setBusy(null);
    await checkRuntime();
  } catch (error) {
    setError(error);
  }
}

main();
