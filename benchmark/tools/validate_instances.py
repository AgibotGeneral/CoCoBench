#!/usr/bin/env python3
"""Batch oracle validation for task_config instances.

Runs eval/runner.py for each selected config, reads the resulting trajectory, and
stores a compact machine-readable report for the dataset quality gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List


BENCH = Path(__file__).resolve().parent.parent
TASK_ROOT = BENCH / "task_config"
RUNNER = BENCH / "eval" / "runner.py"
REPORT_DIR = BENCH / "outputs" / "validation"

sys.path.insert(0, str(BENCH / "eval"))
from metrics import compute_metrics  # noqa: E402


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_task_configs(root: Path, only_cell: str = "", min_agents: int = 0, name_contains: str = "") -> Iterable[Path]:
    for path in sorted(root.glob("*/*/*.json")):
        if path.name == "family_goal_and_skill_pool.json":
            continue
        if name_contains and name_contains not in path.name:
            continue
        if only_cell or min_agents:
            config = _load(path)
            if only_cell:
                cell = f"{config.get('task_family')}_{config.get('coordination_dim')}"
                if cell != only_cell:
                    continue
            if min_agents and int(config.get("agent_count", 2) or 2) < min_agents:
                continue
        yield path


def _checks(final_eval: Dict[str, Any]) -> str:
    checks = final_eval.get("checks", []) or []
    return f"{sum(1 for item in checks if item.get('passed'))}/{len(checks)}"


def _env() -> Dict[str, str]:
    env = dict(os.environ)
    exe_dir = str(Path(sys.executable).resolve().parent)
    env["PATH"] = exe_dir + os.pathsep + env.get("PATH", "")
    return env


def run_one(config_path: Path, out_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    config = _load(config_path)
    task_id = config["task_id"]
    out_dir = out_root / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "runner.stdout.txt"
    stderr_path = out_dir / "runner.stderr.txt"
    started = time.time()

    cmd = [
        sys.executable,
        str(RUNNER),
        "--task-config",
        str(config_path),
        "--oracle-plan",
        "auto",
        "--max-steps",
        str(args.max_steps),
        "--output-dir",
        str(out_dir),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--gpu-device",
        str(args.gpu_device),
    ]
    if args.platform:
        cmd += ["--platform", args.platform]

    if args.dry_run:
        return {
            "task_id": task_id,
            "config": str(config_path.relative_to(BENCH)),
            "dry_run": True,
            "command": cmd,
        }

    proc = subprocess.run(
        cmd,
        cwd=str(BENCH),
        env=_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    trajectory_path = out_dir / "trajectory.json"
    item: Dict[str, Any] = {
        "task_id": task_id,
        "config": str(config_path.relative_to(BENCH)),
        "returncode": proc.returncode,
        "duration_sec": round(time.time() - started, 2),
        "output_dir": str(out_dir.relative_to(BENCH)),
        "stdout": str(stdout_path.relative_to(BENCH)),
        "stderr": str(stderr_path.relative_to(BENCH)),
    }
    if proc.returncode != 0:
        item.update({"success": False, "legal_plan": False, "error": "runner_failed"})
        return item
    if not trajectory_path.exists():
        item.update({"success": False, "legal_plan": False, "error": "missing_trajectory"})
        return item

    report = _load(trajectory_path)
    metrics = compute_metrics(report)
    final_eval = report.get("final_eval", {}) or {}
    item.update(
        {
            "success": bool(final_eval.get("success")),
            "checks": _checks(final_eval),
            "legal_plan": bool(metrics.get("legal_plan")),
            "subgoal_success_rate": metrics.get("subgoal_success_rate"),
            "construct_score": metrics.get("construct_score"),
            "n_action_steps": metrics.get("n_action_steps"),
            "makespan": metrics.get("makespan"),
            "load_imbalance": metrics.get("load_imbalance"),
            "coordination_overhead": metrics.get("coordination_overhead"),
            "failed_steps": metrics.get("failed_steps"),
            "dependency_violations": metrics.get("dependency_violations"),
            "occupancy_conflicts": metrics.get("occupancy_conflicts"),
            "affordance_failures": metrics.get("affordance_failures"),
            "illegal_skill": metrics.get("illegal_skill"),
            "safety_violations": metrics.get("safety_violations"),
        }
    )
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description="Run oracle validation over task_config instances.")
    parser.add_argument("--task-root", default=str(TASK_ROOT), help="task_config directory.")
    parser.add_argument("--out-root", default=str(REPORT_DIR), help="Directory for validation runs and report.")
    parser.add_argument("--only-cell", default="", help="Filter to one cell, e.g. K_D3.")
    parser.add_argument("--name-contains", default="", help="Only validate configs whose filename contains this substring (e.g. seed1).")
    parser.add_argument("--min-agents", type=int, default=0, help="Only validate configs with agent_count >= this (e.g. 3 for the scaling axis).")
    parser.add_argument("--limit", type=int, default=0, help="Validate at most N configs.")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--platform", default="CloudRendering", choices=("CloudRendering", "Linux64", ""))
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel runner subprocesses (each boots its own controller).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    task_root = Path(args.task_root).resolve()
    out_root = Path(args.out_root).resolve()
    configs = list(iter_task_configs(task_root, args.only_cell, args.min_agents, args.name_contains))
    if args.limit:
        configs = configs[: args.limit]

    out_root.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}

    def _safe_run(config_path: Path) -> Dict[str, Any]:
        config = _load(config_path)
        task_id = config["task_id"]
        try:
            return run_one(config_path, out_root, args)
        except subprocess.TimeoutExpired:
            return {
                "task_id": task_id,
                "config": str(config_path.relative_to(BENCH)),
                "success": False,
                "legal_plan": False,
                "error": "timeout",
            }

    if args.jobs > 1 and not args.dry_run:
        done = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {pool.submit(_safe_run, cp): cp for cp in configs}
            for fut in as_completed(futs):
                item = fut.result()
                results[item["task_id"]] = item
                done += 1
                ok = item.get("success") and item.get("legal_plan")
                print(f"[{done}/{len(configs)}] {item['task_id']} {'PASS' if ok else 'FAIL'}")
    else:
        for i, config_path in enumerate(configs, 1):
            config = _load(config_path)
            task_id = config["task_id"]
            print(f"[{i}/{len(configs)}] {task_id}")
            results[task_id] = _safe_run(config_path)

    passed = sum(1 for item in results.values() if item.get("success") and item.get("legal_plan"))
    dry_runs = sum(1 for item in results.values() if item.get("dry_run"))
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_root": str(task_root),
        "count": len(results),
        "passed": passed,
        "failed": 0 if args.dry_run else len(results) - passed,
        "dry_runs": dry_runs,
        "results": results,
    }
    report_path = out_root / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    if args.dry_run:
        print(f"DRY-RUN {dry_runs}/{len(results)}")
        return 0
    print(f"PASS {passed}/{len(results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
