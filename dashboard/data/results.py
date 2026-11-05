"""Offline eval runs: the result JSON files written by tests.eval.run_eval, and the thresholds they are gated on."""
import json
from pathlib import Path
from typing import Any

import streamlit as st

from tests.eval.thresholds import RESULTS_DIR, THRESHOLDS_PATH


def result_files() -> list[Path]:
    return sorted(RESULTS_DIR.glob("*.json"), reverse=True)


@st.cache_data(ttl=60)
def load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_runs() -> list[dict[str, Any]]:
    """Every run, newest first."""
    return [load_result(f) for f in result_files()]


def load_thresholds() -> dict[str, dict[str, float]]:
    return json.loads(THRESHOLDS_PATH.read_text())
