"""Policy-agnostic episode loop.

Drives any :class:`Policy` against a :class:`MultiAgentThorEnv`: the env is reset
by the caller (so observations can be inspected/printed first), then the policy
proposes one skill call per step until it returns ``None`` or the step budget is
hit. The same loop runs oracle, scripted, and VLM policies unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from env import MultiAgentThorEnv, Observation
from policy import Policy
from skill_executor import SkillExecutionResult


def _last_policy_decision(policy: Policy) -> Optional[Dict[str, Any]]:
    decision = getattr(policy, "last_decision", None)
    return dict(decision) if isinstance(decision, dict) else None


def run_episode(env: MultiAgentThorEnv, policy: Policy, obs: Observation, max_steps: Optional[int] = None, verbose: bool = True) -> Dict[str, Any]:
    """Run one episode. Assumes ``env`` is already reset and ``obs`` is its current
    observation. Returns the final evaluation dict."""
    policy.reset(env, env.config)
    last_result: Optional[SkillExecutionResult] = None
    stopped_by_policy = False
    while max_steps is None or env.step_index < max_steps:
        call = policy.act(obs, last_result)
        decision = _last_policy_decision(policy)
        if not call:
            stopped_by_policy = True
            if decision is not None:
                env.record_policy_stop(decision)
            break
        if verbose:
            print(f"CALL: {call}")
        last_result = env.step(call)
        if decision is not None:
            env.attach_policy_decision(decision)
        if verbose:
            print(f"  success={last_result.success} error={last_result.errorMessage}")
        obs = env.observe()
    if not stopped_by_policy and max_steps is not None and env.step_index >= max_steps:
        env.record_policy_stop({
            "policy": "harness",
            "status": "max_steps",
            "terminal_reason": "max_steps",
            "step_index": env.step_index,
        })
    return env.evaluate()


def run_episode_concurrent(env: MultiAgentThorEnv, policy: Any, obs: Observation, max_steps: Optional[int] = None, verbose: bool = True) -> Dict[str, Any]:
    """Run one episode under the **distributed / concurrent** execution model.

    Every round, all agents decide simultaneously from the *same* observation
    snapshot (``policy.propose(obs, last_results)`` returns one decision per agent),
    then their chosen calls are landed one after another (AI2-THOR has no true
    simultaneity). Deciding from a shared snapshot — before seeing each other's
    move — is what makes same-round contention real (two agents that both target
    the same object/pose: the first wins, the second's landing hits the engine's
    occupancy / "no valid positions" refusal), re-activating the engine-level D3
    signals that the turn-based :func:`run_episode` structurally suppresses.

    ``policy`` must expose ``reset(env, task)`` and
    ``propose(obs, last_results) -> List[{agent, call, message, decision}]``.
    ``max_steps`` bounds total landed actions (same budget unit as run_episode)."""
    policy.reset(env, env.config)
    last_results: Dict[str, Optional[SkillExecutionResult]] = {}
    stopped_by_policy = False
    while max_steps is None or env.step_index < max_steps:
        env.round_index += 1
        decisions = policy.propose(obs, last_results)
        if not any(d.get("call") for d in decisions):
            stopped_by_policy = True
            stop = decisions[-1].get("decision") if decisions else None
            env.record_policy_stop(stop or {
                "policy": "distributed", "status": "all_idle", "terminal_reason": "all_idle",
                "step_index": env.step_index,
            })
            break
        round_messages = []
        for d in decisions:
            call = d.get("call")
            if not call:
                continue
            if max_steps is not None and env.step_index >= max_steps:
                break
            if verbose:
                print(f"R{env.round_index} CALL: {call}")
            result = env.step(call, round_index=env.round_index, message=d.get("message") or None)
            last_results[d["agent"]] = result
            if d.get("decision") is not None:
                env.attach_policy_decision(d["decision"])
            if d.get("message"):
                round_messages.append({"round": env.round_index, "agent": d["agent"], "text": d["message"]})
            if verbose:
                print(f"  success={result.success} error={result.errorMessage}")
        env.post_messages(round_messages)
        obs = env.observe()
    if not stopped_by_policy and max_steps is not None and env.step_index >= max_steps:
        env.record_policy_stop({
            "policy": "harness",
            "status": "max_steps",
            "terminal_reason": "max_steps",
            "step_index": env.step_index,
        })
    return env.evaluate()
