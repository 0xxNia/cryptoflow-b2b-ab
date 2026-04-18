"""
Streamlit Community Cloud entry when "Main file path" is streamlit_app.py.

Loads cryptoflow/dashboard.py so __file__ resolves inside cryptoflow/ (paths + DB).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parent
_dashboard = _root / "cryptoflow" / "dashboard.py"

_spec = importlib.util.spec_from_file_location("_cryptoflow_dashboard_main", _dashboard)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load dashboard: {_dashboard}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
