#!/usr/bin/env python3
"""Thin CLI for running a multi-agent AI2-THOR task config.

Wires the pieces of the benchmark skeleton together:

    task_config JSON --> MultiAgentThorEnv (env.py)
                     --> Policy            (policy.py: oracle / VLM)
                     --> run_episode       (harness.py)
                     --> goal-predicate eval (evaluator.py)

Modes:
  --oracle-plan auto|<family>_<dim>   run the built-in oracle reference policy
  --interactive                       terminal action-id loop (manual policy)

All heavy lifting lives in the modules above; this file only parses args,
selects a policy, and prints/saves results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# This eval layer lives in benchmark/eval/; the simulator core (env, action_space,
# taskutil, skill_executor, ...) is the parent benchmark/ dir. Put it on the path
# so the core modules import cleanly when running `python eval/runner.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from action_space import MultiAgentActionSpace
from env import MultiAgentThorEnv, RenderConfig
from harness import run_episode, run_episode_concurrent
from oracles import ORACLE_PLANS
from policy import make_oracle_policy, make_distributed_oracle_policy
from taskutil import agent_names, compact_entries


BENCH = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = BENCH / "task_config" / "A" / "D1" / "A_D1__FloorPlan1__seed0.json"
ORACLE_CHOICES = ("none", "auto") + tuple(sorted(key.lower() for key in ORACLE_PLANS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-agent task config in AI2-THOR.")
    parser.add_argument("--task-config", default=str(DEFAULT_CONFIG), help="Path to a task_config JSON file.")
    parser.add_argument("--x-display", default=os.environ.get("DISPLAY", ":99").lstrip(":"), help="X display id, e.g. 99.")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--quality", default="Low")
    parser.add_argument("--platform", default="CloudRendering", choices=("CloudRendering", "Linux64"), help="AI2-THOR 5.0 rendering platform. CloudRendering for headless GPU.")
    parser.add_argument("--gpu-device", type=int, default=0, help="GPU index for CloudRendering.")
    parser.add_argument("--output-dir", default="", help="Directory for trajectory files. Defaults to outputs/task_runs/<task_id>.")
    parser.add_argument("--full-actions", action="store_true", help="Print all action ids instead of config-allowed skills only.")
    parser.add_argument("--oracle-plan", choices=ORACLE_CHOICES, default="none", help="Run a built-in oracle reference policy. 'auto' picks by task_family/coordination_dim.")
    parser.add_argument("--max-steps", type=int, default=0, help="Episode step budget for policy-driven runs. <=0 means no step limit.")
    parser.add_argument("--interactive", action="store_true", help="Enter terminal action-id loop after initialization/oracle plan.")
    parser.add_argument("--vlm", action="store_true", help="Run a VLM-driven policy (centralized planner, or distributed per-agent with --policy distributed).")
    parser.add_argument("--vlm-backend", default="anthropic", choices=("anthropic", "openai", "mock", "random", "greedy"), help="VLM backend (mock = offline plumbing).")
    parser.add_argument("--vlm-model", default="", help="Model id for the VLM backend (default: backend env var).")
    parser.add_argument("--policy", default="centralized", choices=("centralized", "distributed"), help="VLM control architecture: one omniscient planner vs one decision maker per agent (concurrent rounds).")
    parser.add_argument("--comm", default="none", choices=("none", "broadcast"), help="Distributed inter-agent communication: none (silos, D0) or broadcast (shared message board, D1).")
    parser.add_argument("--obs-mode", default="image", choices=("image", "blind"), help="Observation modality for the VLM policy: image (egocentric views) or blind (no image; plan from goal+menu+action feedback — the de-image ablation).")
    parser.add_argument("--dist-oracle", action="store_true", help="Replay the family/dimension oracle through the concurrent loop.")
    parser.add_argument("--save-images", action="store_true", help="Accumulate every step's PNG frame on disk (step_NNN*.png). Default: off — only a single rolling frame per step is kept (overwritten each step); the VLM still sees the current frame. Saving every frame greatly increases disk usage.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_max_steps(cli_max_steps: int, config: Dict[str, Any]) -> Optional[int]:
    """Effective per-episode step budget for policy-driven runs.

    An explicit ``--max-steps > 0`` always wins. Otherwise fall back to the task
    config's declared ``difficulty.step_budget`` so an episode has a real
    termination guard. Without this, a policy that never emits DONE relies on an
    error (e.g. a transient empty/parse failure) to stop — turning an
    infrastructure blip into the de-facto episode terminator. Returns ``None``
    (unbounded) only when neither source provides a positive budget."""
    if cli_max_steps and cli_max_steps > 0:
        return cli_max_steps
    budget = (config.get("difficulty") or {}).get("step_budget")
    try:
        budget = int(budget)
    except (TypeError, ValueError):
        return None
    return budget if budget > 0 else None


def _print_metrics(trajectory_path: Path) -> None:
    """Print the coordination diagnostic for the just-finished episode."""
    try:
        from metrics import compute_metrics
        report = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
        m = compute_metrics(report)
    except Exception as exc:  # metrics are diagnostic, never fatal
        print(f"[warn] metrics unavailable: {exc!r}")
        return
    print("\n===== COORDINATION METRICS =====")
    for k in ["coordination_dim", "success", "subgoal_success_rate", "legal_plan", "construct_score",
              "construct_status", "construct_sample",
              "makespan", "load_imbalance", "coordination_overhead", "dependency_violations",
              "occupancy_conflicts", "affordance_failures", "illegal_skill", "safety_violations"]:
        print(f"  {k}: {m.get(k)}")


def print_action_prompt(space: MultiAgentActionSpace, config: Dict[str, Any], *, full: bool = False) -> None:
    rows = compact_entries(space, config, full)
    print("\n===== ACTION SPACE =====")
    print(f"task_id: {config.get('task_id')} | shown {len(rows)} / total {len(space.entries)}")
    print("Enter an action id to execute it; actions/full/state/eval/image/plan/q are available.")
    for agent in agent_names(config):
        print(f"\n[{agent}]")
        shown = [entry for entry in rows if entry.agent == agent]
        if not shown:
            print("  (no actions)")
        for entry in shown:
            print(f"  action id {entry.action_id}: {entry.action_name}")


def interactive_loop(env: MultiAgentThorEnv, config: Dict[str, Any], full_actions: bool) -> None:
    while True:
        user_input = input("\naction id > ").strip()
        if not user_input:
            continue
        if user_input in {"q", "quit", "exit"}:
            break
        if user_input == "actions":
            print_action_prompt(env.space, config, full=False)
            continue
        if user_input == "full":
            print_action_prompt(env.space, config, full=True)
            continue
        if user_input == "state":
            from evaluator import state_summary
            print(json.dumps(state_summary(env.controller.last_event, config), ensure_ascii=False, indent=2))
            continue
        if user_input == "eval":
            print(json.dumps(env.evaluate(), ensure_ascii=False, indent=2))
            continue
        if user_input == "image":
            print(f"observation: {env._last_image if env._last_image else '(none)'}")
            continue
        if user_input == "plan":
            policy = make_oracle_policy(config)
            if policy is None:
                print("[warn] no oracle plan for this config")
                continue
            run_episode(env, policy, env.observe(), max_steps=200)
            continue
        try:
            entry = env.space.resolve(int(user_input))
            result = env.step(entry.call)
            print(f"EXECUTE {entry.action_id}: {entry.action_name}")
            print(f"CALL: {entry.call}")
            print(f"SUCCESS: {result.success}")
            if result.errorMessage:
                print(f"ERROR: {result.errorMessage}")
            print(json.dumps(env.evaluate(), ensure_ascii=False, indent=2))
            if env._last_image:
                print(f"observation: {env._last_image}")
        except Exception as exc:
            print(f"[error] {exc}")


def main() -> int:
    args = parse_args()
    config = load_config(args.task_config)
    output_dir = Path(args.output_dir) if args.output_dir else None
    render = RenderConfig(x_display=args.x_display, width=args.width, height=args.height, quality=args.quality, platform=args.platform, gpu_device=args.gpu_device)
    env = MultiAgentThorEnv(config, render=render, output_dir=output_dir, save_images=args.save_images)
    try:
        obs = env.reset()
        print_action_prompt(env.space, config, full=args.full_actions)
        print(json.dumps(env.evaluate(), ensure_ascii=False, indent=2))
        if obs.image:
            print(f"observation: {obs.image}")

        if args.dist_oracle:
            key = None if args.oracle_plan in ("none", "auto") else args.oracle_plan
            policy = make_distributed_oracle_policy(config, key=key)
            if policy is None:
                print(f"[warn] no oracle plan for {config.get('task_id')}")
            else:
                max_steps = resolve_max_steps(args.max_steps, config)
                print(f"\n===== DISTRIBUTED ORACLE (concurrent loop, max_steps={max_steps}) =====")
                final_eval = run_episode_concurrent(env, policy, env.observe(), max_steps=max_steps)
                print("\n===== FINAL EVAL =====")
                print(json.dumps(final_eval, ensure_ascii=False, indent=2))
        elif args.vlm:
            max_steps = resolve_max_steps(args.max_steps, config)
            if args.policy == "distributed":
                from distributed_policy import make_distributed_policy
                policy = make_distributed_policy(backend=args.vlm_backend, model=args.vlm_model or None, comm=args.comm, verbose=True, obs_mode=args.obs_mode)
                print(f"\n===== DISTRIBUTED VLM RUN (backend={args.vlm_backend}, model={args.vlm_model or 'env-default'}, comm={args.comm}, obs={args.obs_mode}, max_steps={max_steps}) =====")
                final_eval = run_episode_concurrent(env, policy, env.observe(), max_steps=max_steps)
            else:
                from vlm_policy import make_vlm_policy
                policy = make_vlm_policy(backend=args.vlm_backend, model=args.vlm_model or None, verbose=True, obs_mode=args.obs_mode)
                print(f"\n===== VLM RUN (backend={args.vlm_backend}, model={args.vlm_model or 'env-default'}, obs={args.obs_mode}, max_steps={max_steps}) =====")
                final_eval = run_episode(env, policy, env.observe(), max_steps=max_steps)
            print("\n===== FINAL EVAL =====")
            print(json.dumps(final_eval, ensure_ascii=False, indent=2))
        elif args.oracle_plan != "none":
            key = None if args.oracle_plan == "auto" else args.oracle_plan
            policy = make_oracle_policy(config, key=key)
            if policy is None:
                print(f"[warn] no oracle plan for {args.oracle_plan} / {config.get('task_id')}")
            else:
                final_eval = run_episode(env, policy, env.observe(), max_steps=args.max_steps if args.max_steps > 0 else None)
                print("\n===== FINAL EVAL =====")
                print(json.dumps(final_eval, ensure_ascii=False, indent=2))

        if args.interactive or (args.oracle_plan == "none" and not args.vlm and not args.dist_oracle):
            interactive_loop(env, config, args.full_actions)

        final_eval = env.finalize()
        print(f"trajectory: {env.output_dir / 'trajectory.json'}")
        print(f"final_success: {final_eval.get('success')}")
        _print_metrics(env.output_dir / "trajectory.json")
    finally:
        env.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
