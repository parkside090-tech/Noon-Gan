# streamlit_app.py — Streamlit Cloud's default entry point; simply runs the dashboard.
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("dashboard.py")), run_name="__main__")
