"""FastAPI entry point for the browser-based 3D rock segmentation tool."""

from __future__ import annotations

import logging
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from web_workflow import WebWorkflowSession, copy_upload_to_session, temp_traceback


RUNS_DIR = REPO_ROOT / "web_runs"
WEB_DIST_DIR = REPO_ROOT / "web" / "dist"
WEB_STATIC_DIR = MODULE_DIR / "web_static"
ALLOWED_POINT_CLOUD_SUFFIXES = {".las", ".laz"}
APP_BUILD = "20260615-interface-width1"


class ManualSeedsRequest(BaseModel):
    rock_seed_indices: list[int] = Field(default_factory=list)
    pedestal_seed_indices: list[int] = Field(default_factory=list)


class InterfacePart(BaseModel):
    selected_indices: list[int] = Field(default_factory=list)
    is_lateral: bool = False


class InterfaceRequest(BaseModel):
    parts: list[InterfacePart]
    close_loop: bool = True


class DraftBrushRequest(BaseModel):
    mode: Literal["add", "remove"]
    selected_indices: list[int] = Field(default_factory=list)
    stroke_indices: list[int] = Field(default_factory=list)
    target_part_index: int | None = None
    target_edge_index: int | None = None
    target_anchor_index: int | None = None
    target_source_index: int | None = None
    start_target_part_index: int | None = None
    start_target_edge_index: int | None = None
    start_target_anchor_index: int | None = None
    start_target_edge_t: float | None = None
    start_target_source_index: int | None = None
    end_target_part_index: int | None = None
    end_target_edge_index: int | None = None
    end_target_anchor_index: int | None = None
    end_target_edge_t: float | None = None
    end_target_source_index: int | None = None
    replace_direction: Literal["forward", "opposite"] | None = None


class InterfaceDraftSourceRequest(BaseModel):
    source: Literal["auto", "manual"]


class SegmentRequest(BaseModel):
    smoothness_threshold: float | None = None
    curvature_threshold: float | None = None
    basal_proximity_threshold: float | None = None
    voxel_size: float | None = None
    neighbor_count: int | None = None
    distance_threshold: float | None = None


class NormalsRequest(BaseModel):
    method: Literal["pymeshlab", "open3d"] = "pymeshlab"
    k: int = 200


class DenoiseRequest(BaseModel):
    method: Literal["sor", "dbscan", "sor_dbscan"] = "sor"
    sor_neighbors: int = 10
    sor_std_ratio: float = 2.0
    dbscan_eps: float = 0.02
    dbscan_min_points: int = 20


class ManualRemoveRequest(BaseModel):
    selected_indices: list[int] = Field(default_factory=list)


class ReconstructRequest(BaseModel):
    depth: int = 8


class RuntimeDiagnostics(BaseModel):
    build: str
    module_dir: str
    repo_root: str
    python_executable: str
    dbscan_eps: bool = True
    sor_dbscan: bool = True
    normal_segment_count: bool = True
    normal_diagnostics: bool = True


@dataclass
class SessionRecord:
    workflow: WebWorkflowSession
    lock: Lock


@dataclass
class JobRecord:
    id: str
    session_id: str
    status: Literal["queued", "running", "completed", "failed"]
    action: str
    result: Any = None
    error: str | None = None
    traceback: str | None = None


app = FastAPI(title="Rock Detection 3D Web Tool")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, SessionRecord] = {}
jobs: dict[str, JobRecord] = {}
registry_lock = Lock()
executor = ThreadPoolExecutor(max_workers=2)


@app.get("/api/diagnostics/runtime", response_model=RuntimeDiagnostics)
def runtime_diagnostics() -> RuntimeDiagnostics:
    return RuntimeDiagnostics(
        build=APP_BUILD,
        module_dir=str(MODULE_DIR),
        repo_root=str(REPO_ROOT),
        python_executable=sys.executable,
    )


def _job_payload(job: JobRecord) -> dict[str, Any]:
    payload = {
        "job_id": job.id,
        "session_id": job.session_id,
        "status": job.status,
        "action": job.action,
    }
    if job.result is not None:
        payload["result"] = job.result
    if job.error:
        payload["error"] = job.error
    return payload


def _get_session(session_id: str) -> SessionRecord:
    with registry_lock:
        record = sessions.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    return record


def _submit_job(
    session_id: str,
    action: str,
    callback: Callable[[WebWorkflowSession], Any],
) -> dict[str, str]:
    record = _get_session(session_id)
    job_id = uuid.uuid4().hex
    job = JobRecord(id=job_id, session_id=session_id, status="queued", action=action)
    with registry_lock:
        jobs[job_id] = job

    def runner() -> None:
        job.status = "running"
        try:
            with record.lock:
                job.result = callback(record.workflow)
            job.status = "completed"
        except Exception as exc:  # noqa: BLE001 - returned through job API
            logging.error("Web workflow job failed: %s", action, exc_info=True)
            job.error = str(exc)
            job.traceback = temp_traceback()
            job.status = "failed"

    executor.submit(runner)
    return {"job_id": job_id, "status": job.status}


@app.post("/api/sessions")
def create_session() -> dict[str, Any]:
    session_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / session_id
    workflow = WebWorkflowSession(session_id=session_id, run_dir=run_dir)
    with registry_lock:
        sessions[session_id] = SessionRecord(workflow=workflow, lock=Lock())
    return workflow.summary()


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    record = _get_session(session_id)
    with record.lock:
        return record.workflow.summary()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with registry_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return _job_payload(job)


@app.post("/api/sessions/{session_id}/point-cloud")
def upload_point_cloud(session_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    record = _get_session(session_id)
    filename = Path(file.filename or "point_cloud.las").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_POINT_CLOUD_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload a .las or .laz point cloud file.")

    target = record.workflow.upload_dir / filename
    try:
        copy_upload_to_session(file.file, target)
        with record.lock:
            return record.workflow.load_point_cloud(target)
    except Exception as exc:  # noqa: BLE001
        logging.error("Point cloud upload failed", exc_info=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/viewer/{view_name}")
def get_viewer(session_id: str, view_name: str) -> dict[str, Any]:
    record = _get_session(session_id)
    mesh_url = f"/api/sessions/{session_id}/downloads/mesh" if view_name == "mesh" else None
    try:
        with record.lock:
            return record.workflow.viewer_payload(view_name, mesh_url=mesh_url)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/diagnostics/normals")
def get_normal_diagnostics(session_id: str) -> dict[str, Any]:
    record = _get_session(session_id)
    try:
        with record.lock:
            return record.workflow.prepared_normal_diagnostics()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/seeds/auto")
def auto_seeds(session_id: str) -> dict[str, str]:
    return _submit_job(session_id, "auto_seeds", lambda workflow: workflow.auto_seeds())


@app.post("/api/sessions/{session_id}/seeds/manual")
def manual_seeds(session_id: str, request: ManualSeedsRequest) -> dict[str, str]:
    return _submit_job(
        session_id,
        "manual_seeds",
        lambda workflow: workflow.manual_seeds(request.rock_seed_indices, request.pedestal_seed_indices),
    )


@app.post("/api/sessions/{session_id}/interface")
def set_interface(session_id: str, request: InterfaceRequest) -> dict[str, str]:
    parts = [part.model_dump() for part in request.parts]
    return _submit_job(
        session_id,
        "set_interface",
        lambda workflow: workflow.set_interface(parts, request.close_loop),
    )


@app.post("/api/sessions/{session_id}/interface/interpolate")
def interpolate_interface(session_id: str, request: InterfaceRequest) -> dict[str, str]:
    parts = [part.model_dump() for part in request.parts]
    return _submit_job(
        session_id,
        "interpolate_interface",
        lambda workflow: workflow.interpolate_interface(parts, request.close_loop),
    )


@app.post("/api/sessions/{session_id}/interface/preview/clear")
def clear_interface_preview(session_id: str) -> dict[str, str]:
    return _submit_job(
        session_id,
        "clear_interface_preview",
        lambda workflow: workflow.clear_interface_preview(),
    )


@app.post("/api/sessions/{session_id}/interface/draft/from-auto")
def create_interface_draft_from_auto(session_id: str) -> dict[str, str]:
    return _submit_job(
        session_id,
        "create_interface_draft_from_auto",
        lambda workflow: workflow.create_interface_draft_from_auto(),
    )


@app.post("/api/sessions/{session_id}/interface/draft/from-source")
def create_interface_draft_from_source(session_id: str, request: InterfaceDraftSourceRequest) -> dict[str, str]:
    return _submit_job(
        session_id,
        "create_interface_draft_from_source",
        lambda workflow: workflow.create_interface_draft_from_source(request.source),
    )


@app.get("/api/sessions/{session_id}/interface/draft")
def get_interface_draft(session_id: str) -> dict[str, Any]:
    record = _get_session(session_id)
    try:
        with record.lock:
            return record.workflow.get_interface_draft()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/interface/draft/anchors")
def update_interface_draft_anchors(session_id: str, request: InterfaceRequest) -> dict[str, str]:
    parts = [part.model_dump() for part in request.parts]
    return _submit_job(
        session_id,
        "update_interface_draft_anchors",
        lambda workflow: workflow.update_interface_draft_anchors(parts, request.close_loop),
    )


@app.post("/api/sessions/{session_id}/interface/draft/brush")
def brush_interface_draft(session_id: str, request: DraftBrushRequest) -> dict[str, str]:
    return _submit_job(
        session_id,
        "brush_interface_draft",
        lambda workflow: workflow.brush_interface_draft(
            request.mode,
            request.selected_indices,
            request.stroke_indices,
            request.target_part_index,
            request.target_edge_index,
            request.target_anchor_index,
            request.target_source_index,
            request.start_target_part_index,
            request.start_target_edge_index,
            request.start_target_anchor_index,
            request.start_target_edge_t,
            request.start_target_source_index,
            request.end_target_part_index,
            request.end_target_edge_index,
            request.end_target_anchor_index,
            request.end_target_edge_t,
            request.end_target_source_index,
            request.replace_direction,
        ),
    )


@app.post("/api/sessions/{session_id}/interface/draft/undo")
def undo_interface_draft(session_id: str) -> dict[str, str]:
    return _submit_job(
        session_id,
        "undo_interface_draft",
        lambda workflow: workflow.undo_interface_draft(),
    )


@app.post("/api/sessions/{session_id}/interface/draft/clear")
def clear_interface_draft(session_id: str) -> dict[str, str]:
    return _submit_job(
        session_id,
        "clear_interface_draft",
        lambda workflow: workflow.clear_interface_draft(),
    )


@app.post("/api/sessions/{session_id}/interface/draft/commit")
def commit_interface_draft(session_id: str) -> dict[str, str]:
    return _submit_job(
        session_id,
        "commit_interface_draft",
        lambda workflow: workflow.commit_interface_draft(),
    )


@app.post("/api/sessions/{session_id}/segment")
def segment(session_id: str, request: SegmentRequest) -> dict[str, str]:
    params = {key: value for key, value in request.model_dump().items() if value is not None}
    return _submit_job(session_id, "segment", lambda workflow: workflow.segment(params))


@app.post("/api/sessions/{session_id}/segment/icrg")
def segment_icrg(session_id: str, request: SegmentRequest) -> dict[str, str]:
    params = {key: value for key, value in request.model_dump().items() if value is not None}
    return _submit_job(session_id, "segment_icrg", lambda workflow: workflow.segment_icrg(params))


@app.post("/api/sessions/{session_id}/mesh/prepare")
def prepare_mesh(session_id: str) -> dict[str, str]:
    return _submit_job(session_id, "prepare_mesh", lambda workflow: workflow.prepare_mesh())


@app.post("/api/sessions/{session_id}/mesh/normals")
def compute_normals(session_id: str, request: NormalsRequest) -> dict[str, str]:
    return _submit_job(
        session_id,
        "compute_normals",
        lambda workflow: workflow.compute_normals(method=request.method, k=request.k),
    )


@app.post("/api/sessions/{session_id}/mesh/noise/remove")
def remove_noise(session_id: str, request: DenoiseRequest = DenoiseRequest()) -> dict[str, str]:
    return _submit_job(
        session_id,
        "remove_noise",
        lambda workflow: workflow.remove_noise(request.model_dump()),
    )


@app.post("/api/sessions/{session_id}/mesh/noise/manual-remove")
def manual_remove_noise(session_id: str, request: ManualRemoveRequest) -> dict[str, str]:
    return _submit_job(
        session_id,
        "manual_remove_noise",
        lambda workflow: workflow.manual_remove_prepared_points(request.selected_indices),
    )


@app.post("/api/sessions/{session_id}/mesh/noise/undo")
def undo_noise(session_id: str) -> dict[str, str]:
    return _submit_job(session_id, "undo_noise", lambda workflow: workflow.undo_noise())


@app.post("/api/sessions/{session_id}/mesh/reconstruct")
def reconstruct_mesh(session_id: str, request: ReconstructRequest) -> dict[str, str]:
    return _submit_job(session_id, "reconstruct_mesh", lambda workflow: workflow.reconstruct_mesh(depth=request.depth))


@app.post("/api/sessions/{session_id}/analysis")
def analyze(session_id: str) -> dict[str, str]:
    return _submit_job(session_id, "analysis", lambda workflow: workflow.analyze())


@app.get("/api/sessions/{session_id}/downloads/{kind}")
def download(session_id: str, kind: str) -> FileResponse:
    record = _get_session(session_id)
    try:
        with record.lock:
            path = record.workflow.download_path(kind)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, filename=path.name)


if (WEB_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST_DIR / "assets"), name="assets")
if WEB_STATIC_DIR.exists():
    app.mount("/web-static", StaticFiles(directory=WEB_STATIC_DIR), name="web_static")


@app.get("/", response_model=None)
def root() -> FileResponse | JSONResponse:
    index = WEB_DIST_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    static_index = WEB_STATIC_DIR / "index.html"
    if static_index.exists():
        return FileResponse(static_index)
    return JSONResponse({
        "message": "Rock Detection 3D API is running. Build the React app in web/ to serve the browser UI.",
        "docs": "/docs",
    })


@app.get("/{full_path:path}", response_model=None)
def spa_fallback(full_path: str) -> FileResponse | JSONResponse:
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    index = WEB_DIST_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return root()
