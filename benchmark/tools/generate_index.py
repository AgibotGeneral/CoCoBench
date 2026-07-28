#!/usr/bin/env python3
"""Regenerate task_config/index.json from task instance configs.

The benchmark's publishable unit is the task_config JSON, so this script treats
those files as the source of truth and rebuilds the EB-style index plus eval_sets
without hand editing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


BENCH = Path(__file__).resolve().parent.parent
TASK_ROOT = BENCH / "task_config"


COORDINATION_DIMS = {
    "D1": "Independent parallelism / task allocation",
    "D2": "Sequential dependency / temporal planning",
    "D3": "Resource exclusion / conflict resolution",
    "D4": "Relay handoff / producer-consumer",
}


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_task_configs(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("*/*/*.json")):
        if path.name == "family_goal_and_skill_pool.json":
            continue
        yield path


def _checks_hint(config: Dict[str, Any]) -> str:
    n = len(config.get("goal_predicates", []) or [])
    return f"{n}/{n}" if n else "0/0"


def _status_for(task_id: str, validation: Optional[Dict[str, Any]]) -> str:
    if not validation:
        return "UNVERIFIED"
    item = validation.get("results", {}).get(task_id)
    if not item:
        return "UNVERIFIED"
    return "PASS" if item.get("success") and item.get("legal_plan") else "FAIL"


def _checks_for(task_id: str, config: Dict[str, Any], validation: Optional[Dict[str, Any]]) -> str:
    if validation:
        item = validation.get("results", {}).get(task_id)
        if item and item.get("checks"):
            return item["checks"]
    return _checks_hint(config)


def _curated_sets(root: Path, known_ids: set[str]) -> Dict[str, List[str]]:
    curated: Dict[str, List[str]] = {}
    for path in sorted((root / "subsets").glob("*.json")):
        subset = _load(path)
        name = str(subset.get("name") or path.stem)
        task_ids = list(subset.get("task_ids") or [])
        unknown = sorted(set(task_ids) - known_ids)
        if unknown:
            raise ValueError(f"{path} references unknown task ids: {', '.join(unknown)}")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"{path} contains duplicate task ids")
        curated[name] = task_ids
    return curated


def build_index(root: Path, validation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    instances: List[Dict[str, Any]] = []
    by_dim: Dict[str, List[str]] = {}
    by_family: Dict[str, List[str]] = {}
    by_cell: Dict[str, List[str]] = {}

    for path in iter_task_configs(root):
        config = _load(path)
        task_id = config["task_id"]
        family = config.get("task_family")
        dim = config.get("coordination_dim")
        rel = path.relative_to(root).as_posix()
        instance = {
            "task_id": task_id,
            "family": family,
            "dim": dim,
            "scene": config.get("scene_id"),
            "seed": config.get("seed"),
            "agent_count": config.get("agent_count"),
            "config": f"task_config/{rel}",
            "oracle": f"{str(family).lower()}_{str(dim).lower()}",
            "eval_layer": config.get("eval_layer", "L0"),
            "status": _status_for(task_id, validation),
            "checks": _checks_for(task_id, config, validation),
            "difficulty": config.get("difficulty", {}),
            "goal_count": len(config.get("goal_predicates", []) or []),
            "summary": config.get("task_name", ""),
        }
        instances.append(instance)
        by_dim.setdefault(dim, []).append(task_id)
        by_family.setdefault(family, []).append(task_id)
        by_cell.setdefault(f"{family}_{dim}", []).append(task_id)

    passed = sum(1 for item in instances if item["status"] == "PASS")
    verified = sum(1 for item in instances if item["status"] != "UNVERIFIED")
    curated = _curated_sets(root, {item["task_id"] for item in instances})
    return {
        "benchmark": "multi-agent-embodied-planning (iTHOR / AI2-THOR 5.0)",
        "description": (
            "Diagnostic benchmark for VLM-based multi-agent high-level embodied planning. "
            "The instance index is generated from task_config by tools/generate_index.py."
        ),
        "reference": (
            "EmbodiedBench/EB-ALFRED groups eval_sets by capability, keeps instances "
            "machine-readable, and judges success from goal states. This benchmark uses "
            "coordination_dim (D1-D4) as the analogous grouping axis."
        ),
        "eval_layers": {
            "L0": "Open-loop oracle: assumes skill execution succeeds and evaluates high-level coordination logic (allocation, ordering, dependencies, handoff). Primary evaluation layer.",
            "L1": "Closed-loop failure injection (occupancy conflicts, affordance incompatibility, unmet preconditions): evaluates coordination robustness and replanning. Optional extension.",
        },
        "runner": "python eval/runner.py --task-config <path> --oracle-plan auto",
        "coordination_dims": COORDINATION_DIMS,
        "instances": instances,
        "eval_sets": {
            "by_dim": {k: sorted(v) for k, v in sorted(by_dim.items())},
            "by_family": {k: sorted(v) for k, v in sorted(by_family.items())},
            "by_cell": {k: sorted(v) for k, v in sorted(by_cell.items())},
            "curated": curated,
        },
        "status_summary": (
            f"{passed}/{len(instances)} instances PASS"
            if verified
            else f"{len(instances)} instances indexed; validation status not attached"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate benchmark/task_config/index.json.")
    parser.add_argument("--task-root", default=str(TASK_ROOT), help="task_config directory.")
    parser.add_argument("--validation-report", default="", help="Optional validate_instances.py JSON report.")
    parser.add_argument("--out", default="", help="Output path. Defaults to <task-root>/index.json.")
    parser.add_argument("--check", action="store_true", help="Do not write; fail if output would differ.")
    args = parser.parse_args()

    task_root = Path(args.task_root).resolve()
    validation = _load(Path(args.validation_report)) if args.validation_report else None
    index = build_index(task_root, validation)
    text = json.dumps(index, ensure_ascii=False, indent=2) + "\n"
    out = Path(args.out).resolve() if args.out else task_root / "index.json"

    if args.check:
        current = out.read_text(encoding="utf-8") if out.exists() else ""
        if current != text:
            print(f"index differs: {out}")
            return 1
        print(f"index up to date: {out}")
        return 0

    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(index['instances'])} instances)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
