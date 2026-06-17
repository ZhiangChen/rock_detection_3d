# Rock Seg 3D Web Frontend

## Backend

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn rock_seg_3d_web.web_app:app --host 127.0.0.1 --port 8010 --reload
```

## Frontend

From `rock_seg_3d_web/frontend/`:

```powershell
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

For a production-style local build:

```powershell
npm run build
```

Then restart the backend and open `http://127.0.0.1:8010`.
