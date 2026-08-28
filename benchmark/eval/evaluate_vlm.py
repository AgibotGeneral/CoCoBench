#!/usr/bin/env python3
"""Batch VLM evaluation over task_config instances.

This is the model-facing counterpart of ``validate_instances.py``. It selects
tasks from ``task_config/index.json``, runs ``eval/runner.py --vlm`` per task,
stores per-episode result JSON files, and writes aggregate summaries suitable
for model comparisons.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


BENCH = Path(__file__).resolve().parent.parent
TASK_ROOT = BENCH / "task_config"
INDEX_PATH = TASK_ROOT / "index.json"
RUNNER = BENCH / "eval" / "runner.py"
OUT_ROOT = BENCH / "outputs" / "eval_runs"
CONFIG_DIR = BENCH / "configs"

sys.path.insert(0, str(BENCH / "eval"))
from metrics import compute_metrics  # noqa: E402


SUMMARY_KEYS = [
    "task_success",
    "task_progress",
    "legal_plan",
    "construct_score",
    "num_steps",
    "planner_steps",
    "planner_output_error",
    "parse_errors",
    "backend_errors",
    "empty_plan",
    "num_invalid_actions",
    "num_invalid_action_ratio",
    "failed_steps",
    "dependency_violations",
    "occupancy_conflicts",
    "affordance_failures",
    "illegal_skill",
    "elapsed_seconds",
]


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = CONFIG_DIR / config_path
    if not config_path.exists() and config_path.suffix == "":
        config_path = config_path.with_suffix(".yaml")
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to read benchmark configs.") from exc
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _apply_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    config = _load_config(args.config)
    defaults = parser.parse_args([])

    model_profiles = config.pop("models", {}) or {}
    config_model_name = str(config.pop("model_name", "") or "")
    cli_model_name = str(getattr(args, "model_name", "") or "")
    selected_model_name = cli_model_name if cli_model_name != getattr(defaults, "model_name", "") else config_model_name

    if model_profiles:
        if not selected_model_name:
            raise ValueError("Config contains a 'models' section; set top-level model_name or pass --model-name.")
        profile = model_profiles.get(selected_model_name)
        if not isinstance(profile, dict):
            choices = ", ".join(sorted(str(key) for key in model_profiles))
            raise ValueError(f"Unknown model_name '{selected_model_name}'. Available models: {choices}")
        config = {**config, **profile}
        config.setdefault("vlm_model", selected_model_name)
        if hasattr(args, "model_name") and getattr(args, "model_name") == getattr(defaults, "model_name"):
            setattr(args, "model_name", selected_model_name)
    elif selected_model_name and "vlm_model" not in config:
        # Backward compatibility for old configs that used model_name as a model id.
        config["vlm_model"] = selected_model_name
        if hasattr(args, "model_name") and getattr(args, "model_name") == getattr(defaults, "model_name"):
            setattr(args, "model_name", selected_model_name)

    aliases = {
        "eval_sets": "eval_set",
        "model_type": "vlm_backend",
        "resolution": "width",
    }
    for src, dest in aliases.items():
        if src in config and dest not in config:
            config[dest] = config[src]
    if "resolution" in config and "height" not in config:
        config["height"] = config["resolution"]
    for key, value in config.items():
        if hasattr(args, key) and getattr(args, key) == getattr(defaults, key):
            setattr(args, key, value)
    return args


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip() or "default")


def _env(args: argparse.Namespace) -> Dict[str, str]:
    env = dict(os.environ)
    exe_dir = str(Path(sys.executable).resolve().parent)
    env["PATH"] = exe_dir + os.pathsep + env.get("PATH", "")
    if getattr(args, "openai_base_url", None):
        env["OPENAI_BASE_URL"] = args.openai_base_url
    if hasattr(args, "openai_max_tokens"):
        if args.openai_max_tokens is None:
            env.pop("OPENAI_MAX_TOKENS", None)
        else:
            env["OPENAI_MAX_TOKENS"] = str(args.openai_max_tokens)
    if getattr(args, "openai_extra_body", ""):
        env["OPENAI_EXTRA_BODY"] = args.openai_extra_body
    else:
        env.pop("OPENAI_EXTRA_BODY", None)
    return env


def _all_config_paths(task_root: Path) -> List[Path]:
    return [
        path for path in sorted(task_root.glob("*/*/*.json"))
        if path.name != "family_goal_and_skill_pool.json"
    ]


def _index_config_map(index: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {item["task_id"]: item for item in index.get("instances", [])}


def _ids_for_eval_set(index: Dict[str, Any], names: Sequence[str]) -> List[str]:
    if not names:
        return [item["task_id"] for item in index.get("instances", [])]
    buckets = index.get("eval_sets", {})
    maps = [buckets.get("by_cell", {}), buckets.get("by_dim", {}), buckets.get("by_family", {}), buckets.get("curated", {})]
    known_ids = {item["task_id"] for item in index.get("instances", [])}
    out: List[str] = []
    for name in names:
        matched: Optional[List[str]] = None
        for mapping in maps:
            if name in mapping:
                matched = list(mapping[name])
                break
        if matched is None and name in known_ids:
            matched = [name]
        if matched is None:
            raise ValueError(f"Unknown eval set or task id: {name}")
        out.extend(matched)
    seen = set()
    return [task_id for task_id in out if not (task_id in seen or seen.add(task_id))]


def select_configs(args: argparse.Namespace) -> List[Path]:
    task_root = Path(args.task_root).resolve()
    if args.index and Path(args.index).exists():
        index = _load(Path(args.index).resolve())
        config_by_id = _index_config_map(index)
        names = [x for part in args.eval_set for x in part.split(",") if x]
        ids = _ids_for_eval_set(index, names)
        paths = [BENCH / config_by_id[task_id]["config"] for task_id in ids]
    else:
        paths = _all_config_paths(task_root)

    if args.only_cell:
        paths = [p for p in paths if _cell(_load(p)) == args.only_cell]
    if args.only_family:
        paths = [p for p in paths if _load(p).get("task_family") == args.only_family]
    if args.only_dim:
        paths = [p for p in paths if _load(p).get("coordination_dim") == args.only_dim]
    if args.task_id:
        wanted = set(args.task_id)
        paths = [p for p in paths if _load(p).get("task_id") in wanted]
    if args.limit:
        paths = paths[: args.limit]
    return paths


def _cell(config: Dict[str, Any]) -> str:
    return f"{config.get('task_family')}_{config.get('coordination_dim')}"


def _checks(final_eval: Dict[str, Any]) -> str:
    checks = final_eval.get("checks", []) or []
    return f"{sum(1 for item in checks if item.get('passed'))}/{len(checks)}"


def _policy_decisions(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        step.get("policy_decision")
        for step in report.get("trajectory", [])
        if isinstance(step.get("policy_decision"), dict)
    ]


def _policy_stats(report: Dict[str, Any], success: bool) -> Dict[str, Any]:
    decisions = _policy_decisions(report)
    terminal = next((d for d in reversed(decisions) if d.get("terminal_reason")), {})
    parse_errors = sum(1 for d in decisions if d.get("status") in {"parse_error", "invalid_action_id"})
    backend_errors = sum(1 for d in decisions if d.get("status") == "backend_error")
    planner_steps = max([int(d.get("planner_steps") or 0) for d in decisions] or [0])
    terminal_reason = terminal.get("terminal_reason")
    return {
        "planner_steps": planner_steps,
        "parse_errors": parse_errors,
        "backend_errors": backend_errors,
        "planner_output_error": parse_errors + backend_errors,
        "empty_plan": int(terminal_reason == "done" and not success),
        "terminal_reason": terminal_reason,
    }


def _episode_result(config: Dict[str, Any], report: Dict[str, Any], duration: float, runner_item: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    final_eval = report.get("final_eval", {}) or {}
    metrics = compute_metrics(report)
    success = bool(final_eval.get("success"))
    policy = _policy_stats(report, success)
    failed_steps = int(metrics.get("failed_steps") or 0)
    num_steps = int(metrics.get("n_action_steps") or 0)
    # legal_plan is NA (None) for a 0-action episode; preserve None so it is excluded
    # from averages rather than collapsing to a misleading 0.0.
    legal_plan = metrics.get("legal_plan")
    dim = config.get("coordination_dim", "")
    # D2 measures shared-resource ordering: success requires legal_plan compliance.
    if dim == "D2" and success and not legal_plan:
        success = False
    out = {
        "task_id": config.get("task_id"),
        "family": config.get("task_family"),
        "dim": dim,
        "cell": _cell(config),
        "scene": config.get("scene_id"),
        "model": args.vlm_model,
        "backend": args.vlm_backend,
        "task_success": float(success),
        "task_progress": metrics.get("subgoal_success_rate"),
        "checks": _checks(final_eval),
        "legal_plan": (None if legal_plan is None else float(bool(legal_plan))),
        "construct_score": metrics.get("construct_score"),
        "num_steps": num_steps,
        "num_invalid_actions": failed_steps,
        "num_invalid_action_ratio": failed_steps / num_steps if num_steps else 0.0,
        "failed_steps": failed_steps,
        "dependency_violations": metrics.get("dependency_violations"),
        "occupancy_conflicts": metrics.get("occupancy_conflicts"),
        "affordance_failures": metrics.get("affordance_failures"),
        "illegal_skill": metrics.get("illegal_skill"),
        "elapsed_seconds": round(duration, 2),
        "output_dir": runner_item.get("output_dir"),
        "trajectory": runner_item.get("trajectory"),
        "stdout": runner_item.get("stdout"),
        "stderr": runner_item.get("stderr"),
    }
    out.update(policy)
    return out


def run_one(config_path: Path, out_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    config = _load(config_path)
    task_id = config["task_id"]
    out_dir = out_root / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "runner.stdout.txt"
    stderr_path = out_dir / "runner.stderr.txt"
    result_path = out_dir / "episode_result.json"

    # Resume: if this task already has a complete episode result, reuse it and
    # skip re-running. A result counts as complete when task_success is recorded
    # and (unless --resume-retry-errors) it did not end in a transient infra
    # failure (runner_failed / missing_trajectory / timeout / backend errors).
    if getattr(args, "resume", False) and not args.dry_run and result_path.exists():
        try:
            cached = _load(result_path)
        except Exception:
            cached = None
        if isinstance(cached, dict) and cached.get("task_success") is not None:
            transient = cached.get("error") in {"runner_failed", "missing_trajectory", "timeout"} \
                or (cached.get("backend_errors") or 0) > 0
            if not (getattr(args, "resume_retry_errors", False) and transient):
                cached["resumed"] = True
                return cached

    cmd = [
        sys.executable,
        str(RUNNER),
        "--task-config",
        str(config_path),
        "--vlm",
        "--vlm-backend",
        args.vlm_backend,
        "--output-dir",
        str(out_dir),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--gpu-device",
        str(args.gpu_device),
    ]
    if args.max_steps is not None:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.vlm_model:
        cmd += ["--vlm-model", args.vlm_model]
    if args.platform:
        cmd += ["--platform", args.platform]
    if args.policy:
        cmd += ["--policy", args.policy]
    if args.comm:
        cmd += ["--comm", args.comm]
    if getattr(args, "obs_mode", "image"):
        cmd += ["--obs-mode", args.obs_mode]
    if getattr(args, "save_images", False):
        cmd += ["--save-images"]

    if args.dry_run:
        return {"task_id": task_id, "config": str(config_path.relative_to(BENCH)), "dry_run": True, "command": cmd}

    started = time.time()
    runner_item: Dict[str, Any] = {
        "task_id": task_id,
        "family": config.get("task_family"),
        "dim": config.get("coordination_dim"),
        "cell": _cell(config),
        "scene": config.get("scene_id"),
        "model": args.vlm_model,
        "backend": args.vlm_backend,
        "policy": args.policy,
        "comm": args.comm if args.policy == "distributed" else None,
        "config": str(config_path.relative_to(BENCH)),
        "output_dir": str(out_dir.relative_to(BENCH)),
        "stdout": str(stdout_path.relative_to(BENCH)),
        "stderr": str(stderr_path.relative_to(BENCH)),
        "trajectory": str((out_dir / "trajectory.json").relative_to(BENCH)),
    }
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BENCH),
            env=_env(args),
            text=True,
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
        )
        duration = time.time() - started
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        runner_item["returncode"] = proc.returncode
        if proc.returncode != 0:
            item = {**runner_item, "task_success": 0.0, "legal_plan": 0.0, "error": "runner_failed", "elapsed_seconds": round(duration, 2)}
            result_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return item
        trajectory_path = out_dir / "trajectory.json"
        if not trajectory_path.exists():
            item = {**runner_item, "task_success": 0.0, "legal_plan": 0.0, "error": "missing_trajectory", "elapsed_seconds": round(duration, 2)}
            result_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return item
        report = _load(trajectory_path)
        item = _episode_result(config, report, duration, runner_item, args)
        result_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return item
    except subprocess.TimeoutExpired:
        item = {**runner_item, "task_success": 0.0, "legal_plan": 0.0, "error": "timeout", "elapsed_seconds": args.timeout}
        result_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return item


def _average(results: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for item in results:
        for key in SUMMARY_KEYS:
            value = item.get(key)
            if value is None or isinstance(value, str):
                continue
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)):
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sorted(sums)}


def _group_summary(results: Dict[str, Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in results.values():
        groups.setdefault(str(item.get(key)), []).append(item)
    return {name: _average(items) for name, items in sorted(groups.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch evaluate a VLM backend over multi-agent tasks.")
    parser.add_argument("--config", default="config.yaml", help="YAML config under benchmark/configs, or an absolute path.")
    parser.add_argument("--model-name", default="", help="Model profile name inside the YAML config's models section.")
    parser.add_argument("--task-root", default=str(TASK_ROOT))
    parser.add_argument("--index", default=str(INDEX_PATH), help="task_config/index.json; eval sets are read from here.")
    parser.add_argument("--eval-set", action="append", default=[], help="Eval set/task id from index: a dim (D3), family (K), cell (K_D3), curated subset (rep240), or a task_id. May be comma-separated.")
    parser.add_argument("--only-cell", default="")
    parser.add_argument("--only-family", default="")
    parser.add_argument("--only-dim", default="")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--vlm-backend", default="openai", choices=("anthropic", "openai", "mock", "random", "greedy"))
    parser.add_argument("--vlm-model", default="")
    parser.add_argument("--policy", default="centralized", choices=("centralized", "distributed"), help="VLM control architecture passed to runner: centralized planner or distributed per-agent (concurrent rounds).")
    parser.add_argument("--comm", default="none", choices=("none", "broadcast"), help="Distributed inter-agent communication (only used with --policy distributed).")
    parser.add_argument("--obs-mode", default="image", choices=("image", "blind"), help="Observation modality: image (egocentric views) or blind (no image; de-image ablation). Also settable per model profile in config.yaml.")
    parser.add_argument("--openai-base-url", default="")
    parser.add_argument("--openai-max-tokens", type=int, default=None, help="None/null means omit max_tokens from OpenAI-compatible requests.")
    parser.add_argument("--openai-extra-body", default="", help="JSON dict of extra request-body params for OpenAI-compatible requests, e.g. '{\"enable_thinking\": false}' to disable a reasoning model's chain-of-thought.")
    parser.add_argument("--resume", action="store_true", help="Skip tasks that already have a complete episode_result.json (reuse cached result). Makes a batch run resumable after interruption.")
    parser.add_argument("--resume-retry-errors", action="store_true", help="With --resume, still re-run tasks whose cached result ended in a transient infra failure (runner_failed / missing_trajectory / timeout / backend error).")
    parser.add_argument("--save-images", action="store_true", help="Accumulate every step's PNG frame on disk. Default: off — only a single rolling frame per step is kept (the VLM still sees the current frame), greatly reducing disk usage.")
    parser.add_argument("--exp-name", default="")
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel runner subprocesses (each boots its own controller).")
    parser.add_argument("--platform", default="CloudRendering", choices=("CloudRendering", "Linux64", ""))
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=None, help="None/null means no runner step limit.")
    parser.add_argument("--timeout", type=int, default=None, help="None/null means no subprocess timeout.")
    parser.add_argument("--dry-run", action="store_true")
    args = _apply_config(parser.parse_args(), parser)

    if args.eval_set is None:
        args.eval_set = []
    if args.task_id is None:
        args.task_id = []

    selected = select_configs(args)
    model_name = _safe_name(args.vlm_model or os.environ.get("OPENAI_MODEL") or os.environ.get("ANTHROPIC_MODEL") or "env-model")
    eval_name = _safe_name("_".join(args.eval_set) if args.eval_set else args.only_cell or args.only_family or args.only_dim or "all")
    exp_name = _safe_name(args.exp_name) if args.exp_name else f"{args.vlm_backend}_{model_name}"
    out_base = Path(args.out_root).resolve() if args.out_root else OUT_ROOT
    out_root = out_base / exp_name / eval_name
    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    if getattr(args, "jobs", 1) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        done = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {pool.submit(run_one, cp, out_root, args): cp for cp in selected}
            for fut in as_completed(futs):
                item = fut.result()
                tid = item.get("task_id") or _load(futs[fut])["task_id"]
                results[tid] = item
                done += 1
                tag = " (resumed)" if item.get("resumed") else ""
                print(f"[{done}/{len(selected)}] {tid}{tag}")
    else:
        for i, config_path in enumerate(selected, 1):
            task_id = _load(config_path)["task_id"]
            item = run_one(config_path, out_root, args)
            results[task_id] = item
            tag = " (resumed)" if item.get("resumed") else ""
            print(f"[{i}/{len(selected)}] {task_id}{tag}")

    if getattr(args, "resume", False):
        n_resumed = sum(1 for r in results.values() if r.get("resumed"))
        print(f"resume: reused {n_resumed} cached, ran {len(results) - n_resumed} fresh")

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": args.vlm_backend,
        "model_name": args.model_name,
        "model": args.vlm_model,
        "eval_set": args.eval_set,
        "count": len(results),
        "success_count": sum(1 for item in results.values() if item.get("task_success") == 1.0),
        "summary": _average(results.values()),
        "by_cell": _group_summary(results, "cell"),
        "by_dim": _group_summary(results, "dim"),
        "by_family": _group_summary(results, "family"),
        "results": results,
    }
    report_name = "dry_run_report.json" if args.dry_run else "eval_report.json"
    summary_name = "dry_run_summary.json" if args.dry_run else "summary.json"
    (out_root / report_name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_root / summary_name).write_text(json.dumps(report["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_root / report_name}")
    print(f"success {report['success_count']}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
