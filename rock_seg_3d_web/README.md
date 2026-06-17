# Rock Seg 3D Web

`rock_seg_3d_web` is the self-contained browser web tool for 3D rock segmentation, mesh preparation, reconstruction, and analysis.

It contains:

- the FastAPI backend and workflow state,
- the served static browser UI,
- the React frontend source mirror,
- copied core algorithms used by the web tool.

The package does not depend on importing implementation modules from `rock_detection_3d`.

## Install Dependencies

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r rock_seg_3d_web\requirements.txt
```

## Launch

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn rock_seg_3d_web.web_app:app --host 127.0.0.1 --port 8010
```

Open:

```text
http://127.0.0.1:8010/
```

The old compatibility path still works:

```powershell
.\.venv\Scripts\python.exe -m uvicorn rock_detection_3d.web_app:app --host 127.0.0.1 --port 8010
```

## Runtime Output

By default, web sessions are written to:

```text
<current working directory>/web_runs/
```

Set `ROCK_SEG_3D_WEB_RUNS_DIR` to override that location.

## Frontend Development

The production FastAPI app serves `rock_seg_3d_web/rock_seg_3d_web/web_static` directly, so a frontend build is not required for normal use.

The React source mirror is in `rock_seg_3d_web/frontend`.

```powershell
cd rock_seg_3d_web\frontend
npm install
npm run dev
```

For a production-style React build:

```powershell
npm run build
```

After building, restart the FastAPI backend. If `rock_seg_3d_web/frontend/dist` exists, the backend mounts that React build's assets while the static UI remains the default first screen.
