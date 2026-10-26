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
    """Returns human-readable violations. Metrics absent from this run (e.g. --skip-llm) are not checked."""
    violations: list[str] = []
    for path, bounds in json.loads(THRESHOLDS_PATH.read_text()).items():
        value = lookup(report, path)
        if value is None:
            continue
        if "min" in bounds and value < bounds["min"]:
            violations.append(f"{path} = {value} < min {bounds['min']}")
        if "max" in bounds and value > bounds["max"]:
            violations.append(f"{path} = {value} > max {bounds['max']}")
    return violations
