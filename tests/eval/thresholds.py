"""Result paths and threshold checks. Dependency-free so the dashboard can import it without loading torch.

thresholds.json is the target every golden set aims for. floors.json holds, per golden file, gates at that set's
current level: they replace the target's, so a run fails only when it gets worse, and the target gap stays visible.
"""
import json
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("tests/eval/results")
THRESHOLDS_PATH = Path("tests/eval/thresholds.json")
FLOORS_PATH = Path("tests/eval/floors.json")

Bounds = dict[str, dict[str, float]]


def lookup(report: dict[str, Any], dotted: str) -> Any:
    node: Any = report
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def target() -> Bounds:
    return json.loads(THRESHOLDS_PATH.read_text())


def gates(report: dict[str, Any]) -> Bounds:
    """The target, with the floors of this run's golden set in place of its gates."""
    floors = json.loads(FLOORS_PATH.read_text()) if FLOORS_PATH.exists() else {}
    return {**target(), **floors.get(lookup(report, "meta.golden"), {})}


def check_thresholds(report: dict[str, Any], bounds: Bounds | None = None) -> list[str]:
    """Returns human-readable violations of `bounds` (default: this run's gates). Sections absent from this run
    (e.g. `llm` with --skip-llm) are not checked."""
    violations: list[str] = []
    for path, limit in (gates(report) if bounds is None else bounds).items():
        if path.split(".")[0] not in report:
            continue
        parent, _, key = path.rpartition(".")
        node = lookup(report, parent)
        if not isinstance(node, dict) or key not in node:
            # Renamed or removed metric: the gate would otherwise pass silently forever
            violations.append(f"{path} = missing")
            continue
        value = node[key]
        if value is None:  # present but nothing to score, e.g. no unanswerable rows in a --limit run
            continue
        if "min" in limit and value < limit["min"]:
            violations.append(f"{path} = {value} < min {limit['min']}")
        if "max" in limit and value > limit["max"]:
            violations.append(f"{path} = {value} > max {limit['max']}")
    return violations
