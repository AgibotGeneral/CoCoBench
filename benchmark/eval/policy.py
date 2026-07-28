"""Decision-maker interface for the benchmark.

A ``Policy`` is the single seam between the harness and whoever decides actions:
an oracle reference plan, a scripted replay, or a VLM. The harness is
policy-agnostic — swapping in a VLM means implementing :class:`VLMPolicy`, not
touching the env or the loop.

Contract
--------
``reset(env, task)`` is called once per episode. ``act(obs, last_result)`` is
called each step and returns the next skill-call string (e.g.
``"Put(agent_1, Plate|...)"``), or ``None`` to end the episode. ``last_result``
is the :class:`SkillExecutionResult` of the previous action (``None`` on the
first step).

Centralized vs decentralized is a harness concern: a centralized policy sees all
agents' observations and emits one agent's action per step; a decentralized
setup runs one policy instance per agent with a communication channel. The
single-string return keeps both expressible (the call names its own agent).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from env import Observation
from oracles import ORACLE_PLANS, oracle_plan_for_config
from skill_executor import SkillExecutionResult


class Policy(ABC):
    def reset(self, env: Any, task: Dict[str, Any]) -> None:
        """Called once at episode start. ``env`` is provided for privileged
        (oracle) policies; fair policies should rely only on ``obs`` in act()."""

    @abstractmethod
    def act(self, obs: Observation, last_result: Optional[SkillExecutionResult]) -> Optional[str]:
        """Return the next skill-call string, or None when finished."""


class GeneratorPolicy(Policy):
    """Adapts an imperative oracle plan (a generator over skill-call strings) to
    the step-driven ``Policy`` interface. The plan ``yield``s the next call and is
    ``.send``-ed the previous call's execution result."""

    def __init__(self, plan_fn: Callable[[Any], Any], label: str = "oracle") -> None:
        self._plan_fn = plan_fn
        self.label = label
        self._gen = None
        self._started = False

    def reset(self, env: Any, task: Dict[str, Any]) -> None:
        self._gen = self._plan_fn(env)
        self._started = False

    def act(self, obs: Observation, last_result: Optional[SkillExecutionResult]) -> Optional[str]:
        if self._gen is None:
            raise RuntimeError("Policy not reset; call reset() before act().")
        try:
            if not self._started:
                self._started = True
                return next(self._gen)
            return self._gen.send(last_result)
        except StopIteration:
            return None


class VLMPolicy(Policy):
    """Seam for a VLM-driven planner (not yet implemented).

    A concrete implementation packages ``obs`` (per-agent egocentric images +
    text goal + action_menu + history) into a prompt, queries the model, and
    parses the chosen action back into a skill-call string from
    ``obs.action_menu``. It must consume only ``obs`` — never the privileged
    EnvView — so the evaluation stays fair.
    """

    def __init__(self, client: Any = None, **kwargs: Any) -> None:
        self.client = client
        self.kwargs = kwargs

    def act(self, obs: Observation, last_result: Optional[SkillExecutionResult]) -> Optional[str]:
        raise NotImplementedError(
            "VLMPolicy.act is a stub. Implement: build prompt from obs.per_agent + "
            "obs.action_menu + obs.goal_text, call self.client, parse to a menu 'call'."
        )


def make_oracle_policy(config: Dict[str, Any], key: Optional[str] = None) -> Optional[GeneratorPolicy]:
    """Build the oracle policy for a config (auto by family/dim) or an explicit
    ``<family>_<dim>`` key (case-insensitive)."""
    if key:
        plan = ORACLE_PLANS.get(key.upper())
        label = key.upper()
    else:
        plan = oracle_plan_for_config(config)
        label = f"{config.get('task_family')}_{config.get('coordination_dim')}"
    if plan is None:
        return None
    return GeneratorPolicy(plan, label=label)


def _agent_of(call: str) -> str:
    """Extract the acting agent from a skill-call string ``Skill(agent_x, ...)``."""
    try:
        return call.split("(", 1)[1].split(",", 1)[0].split(")", 1)[0].strip()
    except (IndexError, AttributeError):
        return "agent_1"


class DistributedOraclePolicy(Policy):
    """Replays a (centralized) oracle generator through the *concurrent* loop, one
    call per round attributed to the agent it names; the other agents idle that
    round. The executed action **sequence is identical** to the turn-based oracle,
    which checks that ``run_episode_concurrent`` and the round/message recording do
    not alter a known-good trajectory.
    """

    def __init__(self, plan_fn: Callable[[Any], Any], label: str = "dist-oracle") -> None:
        self._plan_fn = plan_fn
        self.label = label
        self._gen = None
        self._started = False
        self._emitted_agent: Optional[str] = None
        self.last_decision: Dict[str, Any] = {}

    def reset(self, env: Any, task: Dict[str, Any]) -> None:
        self._gen = self._plan_fn(env)
        self._started = False
        self._emitted_agent = None
        self.last_decision = {}

    def act(self, obs: Observation, last_result: Optional[SkillExecutionResult]) -> Optional[str]:
        raise NotImplementedError("DistributedOraclePolicy is driven by run_episode_concurrent.")

    def propose(self, obs: Observation, last_results: Dict[str, Optional[SkillExecutionResult]]) -> list:
        if self._gen is None:
            raise RuntimeError("Policy not reset; call reset() before propose().")
        try:
            if not self._started:
                self._started = True
                call = next(self._gen)
            else:
                # feed back the result of the call this oracle emitted last round
                prev = last_results.get(self._emitted_agent) if self._emitted_agent else None
                call = self._gen.send(prev)
        except StopIteration:
            return []  # plan exhausted -> all idle -> episode ends
        agent = _agent_of(call)
        self._emitted_agent = agent
        decision = {"policy": "distributed_oracle", "label": self.label, "agent": agent,
                    "step_index": obs.step_index, "status": "action", "parsed_call": call}
        self.last_decision = decision
        return [{"agent": agent, "call": call, "message": "", "decision": decision}]


def make_distributed_oracle_policy(config: Dict[str, Any], key: Optional[str] = None) -> Optional[DistributedOraclePolicy]:
    """Replay a family/dimension oracle through the concurrent loop."""
    if key:
        plan, label = ORACLE_PLANS.get(key.upper()), key.upper()
    else:
        plan = oracle_plan_for_config(config)
        label = f"{config.get('task_family')}_{config.get('coordination_dim')}"
    return DistributedOraclePolicy(plan, label=label) if plan is not None else None
