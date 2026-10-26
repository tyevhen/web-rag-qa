"""Result paths and threshold checks. Dependency-free so the dashboard can import it without loading torch."""
import json
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("tests/eval/results")
THRESHOLDS_PATH = Path("tests/eval/thresholds.json")


def lookup(report: dict[str, Any], dotted: str) -> Any:
    node: Any = report
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def check_thresholds(report: dict[str, Any]) -> list[str]:
    """Returns human-readable violations. Sections absent from this run (e.g. `llm` with --skip-llm) are not checked."""
    violations: list[str] = []
    for path, bounds in json.loads(THRESHOLDS_PATH.read_text()).items():
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
        if "min" in bounds and value < bounds["min"]:
            violations.append(f"{path} = {value} < min {bounds['min']}")
        if "max" in bounds and value > bounds["max"]:
            violations.append(f"{path} = {value} > max {bounds['max']}")
    return violations
