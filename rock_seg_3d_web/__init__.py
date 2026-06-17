"""Development import bridge for the nested rock_seg_3d_web package.

This lets `python -m uvicorn rock_seg_3d_web.web_app:app` work from the
repository root without requiring an editable install first.
"""

from pathlib import Path

_INNER_PACKAGE = Path(__file__).resolve().parent / "rock_seg_3d_web"
if _INNER_PACKAGE.exists():
    __path__.append(str(_INNER_PACKAGE))  # type: ignore[name-defined]
