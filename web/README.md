# Rock Detection 3D Web UI

## Backend

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn rock_detection_3d.web_app:app --host 127.0.0.1 --port 8000 --reload
```

## Frontend

From `web/`:

```powershell
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

For a production-style local build:

```powershell
npm run build
```

Then restart the backend and open `http://127.0.0.1:8000`.
