"""Distributed (decentralized) multi-agent VLM policy.

The contrast to :class:`vlm_policy.CentralizedVLMPolicy`: instead of one omniscient
planner that sees every agent's view and emits one ``(agent, skill)`` per step,
this runs **one VLM client per agent**, each seeing **only its own egocentric
view + its own filtered action menu + its own history** and (optionally) a shared
message board. All agents decide **simultaneously** from the same observation
snapshot; the concurrent harness loop (:func:`harness.run_episode_concurrent`)
then lands their chosen calls one after another (AI2-THOR has no true simultaneity)
— which is exactly what makes same-round contention real and re-activates the
engine-level coordination signals (occupancy filter, ``No valid positions``) that
turn-based execution suppresses.

World perception is held equal to the centralized condition (each agent's menu
names all task objects, same as the centralized full menu), so the only thing a
distributed agent lacks is the *global coordination state* — what its teammates
see / hold / intend — which it must reconstruct via communication. That keeps the
centralized↔distributed comparison a clean test of coordination-under-
decentralization rather than a confounded perception test.

Communication (``--comm``):
  * ``none``      — agents are information silos (D0).
  * ``broadcast`` — each agent may emit a ``MSG:`` line; the shared board is shown
                    to every agent next round (D1).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from env import Observation
from policy import Policy
from skill_executor import SkillExecutionResult
from taskutil import coordination_hint
from vlm_policy import make_vlm_client, parse_action, parse_message, _build_goal_progress


DIST_PROMPT_HEADER = """You are {me}, one of {n} robots ({agents}) jointly doing an AI2-THOR household task.
{view}
Your teammates are deciding their own next action at the same time — you will NOT see
their choice until next round, so coordinate by reasoning about what they are likely doing.

GOAL: {goal}

GOAL PROGRESS (live simulator state — trust this over your action history):
{goal_progress}
{hint}{comm}
Choose the single best NEXT action for YOURSELF to make progress while coordinating
(divide the work, respect ordering/dependencies, do NOT contend with a teammate for the
same object / spot / resource). After Find succeeds you are at the object — proceed to the
interaction (PickUp/Put/Toggle/...), do NOT repeat Find. Reply on these lines exactly:
REASON: <one short sentence>
ACTION: <the integer action id from YOUR menu, or DONE if the whole task is satisfied>{msg_hint}

YOUR ACTION MENU (action_id: skill):
{menu}

YOUR RECENT ACTIONS (most recent last):
{history}

YOUR STATE: holding={inv}"""

DIST_VIEW_IMAGE = "You control ONLY yourself and you see ONLY your own egocentric view (the image above)."
DIST_VIEW_BLIND = "You control ONLY yourself. No image is provided and you have NO direct perception: infer object states only from the GOAL, your available actions, your inventory, and your past actions' success/failure."


class DistributedVLMPolicy(Policy):
    """One VLM client per agent; ``propose`` returns one decision per agent per round."""

    def __init__(self, backend: str = "anthropic", model: Optional[str] = None,
                 comm: str = "none", verbose: bool = False, obs_mode: str = "image") -> None:
        self.backend = backend
        self.model = model
        self.comm = comm                       # "none" | "broadcast"
        self.obs_mode = obs_mode               # "image" | "blind"
        self.verbose = verbose
        self.clients: Dict[str, Any] = {}
        self._agents: List[str] = []
        self._hint: str = ""
        self._history: Dict[str, List[str]] = {}
        self._last_call: Dict[str, Optional[str]] = {}
        self.planner_steps = 0
        self.parse_errors = 0
        self.backend_errors = 0
        self.last_decision: Dict[str, Any] = {}

    # ---- lifecycle -------------------------------------------------------
    def reset(self, env: Any, task: Dict[str, Any]) -> None:
        self._agents = [a.get("id") for a in task.get("agents", [])] or [
            f"agent_{i+1}" for i in range(int(task.get("agent_count", 2) or 2))
        ]
        self._hint = coordination_hint(task)
        # One independent client per agent (each is its own decision maker).
        self.clients = {a: make_vlm_client(self.backend, self.model) for a in self._agents}
        self._history = {a: [] for a in self._agents}
        self._last_call = {a: None for a in self._agents}
        self.planner_steps = 0
        self.parse_errors = 0
        self.backend_errors = 0
        self.last_decision = {}

    # The concurrent harness drives this policy through ``propose``; ``act`` (the
    # single-call turn-based seam) does not apply to a decentralized setup.
    def act(self, obs: Observation, last_result: Optional[SkillExecutionResult]) -> Optional[str]:
        raise NotImplementedError(
            "DistributedVLMPolicy is driven by harness.run_episode_concurrent (one "
            "call per agent per round), not the turn-based run_episode."
        )

    # ---- per-round decision ---------------------------------------------
    def propose(self, obs: Observation, last_results: Dict[str, Optional[SkillExecutionResult]]) -> List[Dict[str, Any]]:
        """Query every agent's client concurrently for its next action this round.

        Returns one dict per agent: ``{agent, call, message, decision}``. ``call`` is
        ``None`` when the agent is DONE / has no actions / errored (it simply does not
        act this round). ``last_results[agent]`` is that agent's own previous result."""
        # fold each agent's previous outcome into its private history first
        for agent in self._agents:
            lr = last_results.get(agent)
            if self._last_call.get(agent) is not None and lr is not None:
                ok = "ok" if lr.success else f"FAIL({(lr.errorMessage or '')[:60]})"
                self._history[agent].append(f"{self._last_call[agent]} -> {ok}")

        agents = [a for a in self._agents]
        with ThreadPoolExecutor(max_workers=max(len(agents), 1)) as pool:
            decided = list(pool.map(lambda a: self._decide_one(a, obs), agents))

        # remember each agent's chosen call for next round's history line
        for d in decided:
            self._last_call[d["agent"]] = d["call"]
        self.last_decision = decided[-1]["decision"] if decided else {}
        return decided

    def _decide_one(self, agent: str, obs: Observation) -> Dict[str, Any]:
        menu = [e for e in obs.action_menu if e.get("agent") == agent]
        if not menu:
            return {"agent": agent, "call": None, "message": "",
                    "decision": self._decision(agent, obs, status="no_actions", terminal_reason="no_actions")}
        prompt = self._build_prompt(agent, obs, menu)
        image = None if self.obs_mode == "blind" else (obs.per_agent.get(agent) or {}).get("image")
        try:
            self.planner_steps += 1
            raw = self.clients[agent].choose(prompt, image, menu)
        except Exception as exc:  # one agent's transient failure idles it, not the episode
            self.backend_errors += 1
            if self.verbose:
                print(f"[dist:{agent}] backend error: {exc!r}")
            return {"agent": agent, "call": None, "message": "",
                    "decision": self._decision(agent, obs, status="backend_error", terminal_reason="backend_error", error=repr(exc))}
        parsed = parse_action(raw, menu)
        message = parse_message(raw) if self.comm == "broadcast" else ""
        if parsed["status"] in {"parse_error", "invalid_action_id"}:
            self.parse_errors += 1
        call = parsed.get("call")
        if self.verbose:
            print(f"[dist:{agent}] {(raw or '').strip()[:120]} -> {call}")
        return {
            "agent": agent,
            "call": call,
            "message": message,
            "decision": self._decision(
                agent, obs, status=parsed["status"],
                terminal_reason=parsed["status"] if call is None else None,
                raw_output=raw, parsed_action_id=parsed.get("action_id"),
                parsed_call=call, parse_mode=parsed.get("parse_mode"), error=parsed.get("error"),
                message=message,
            ),
        }

    # ---- prompt ----------------------------------------------------------
    def _build_prompt(self, agent: str, obs: Observation, menu: List[Dict[str, Any]]) -> str:
        menu_lines = "\n".join(f"  {e['action_id']}: {e['name']}" for e in menu) or "  (no actions available)"
        history = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(self._history[agent][-6:])) or "  (none yet)"
        inv = ", ".join((obs.per_agent.get(agent) or {}).get("inventory") or []) or "empty"
        comm_block, msg_hint = "", ""
        if self.comm == "broadcast":
            msgs = [m for m in (obs.messages or []) if m.get("agent") != agent][-6:]
            board = "\n".join(f"  {m.get('agent')}: {m.get('text')}" for m in msgs) or "  (no messages yet)"
            comm_block = f"\nTEAMMATE MESSAGES (most recent last):\n{board}\n"
            msg_hint = "\nMSG: <one short note telling teammates what you are doing / will do>"
        view = DIST_VIEW_BLIND if self.obs_mode == "blind" else DIST_VIEW_IMAGE
        return DIST_PROMPT_HEADER.format(
            me=agent,
            n=len(self._agents),
            agents=", ".join(self._agents),
            view=view,
            goal=obs.goal_text,
            goal_progress=_build_goal_progress(obs),
            hint=(self._hint + "\n") if self._hint else "",
            comm=comm_block,
            msg_hint=msg_hint,
            menu=menu_lines,
            history=history,
            inv=inv,
        )

    def _decision(self, agent: str, obs: Observation, *, status: str, terminal_reason: Optional[str] = None,
                  raw_output: Optional[str] = None, parsed_action_id: Optional[str] = None,
                  parsed_call: Optional[str] = None, parse_mode: Optional[str] = None,
                  error: Optional[str] = None, message: Optional[str] = None) -> Dict[str, Any]:
        decision: Dict[str, Any] = {
            "policy": "distributed_vlm",
            "comm": self.comm,
            "agent": agent,
            "backend": self.backend,
            "model": getattr(self.clients.get(agent), "model", None),
            "step_index": obs.step_index,
            "status": status,
            "terminal_reason": terminal_reason,
            "parsed_action_id": parsed_action_id,
            "parsed_call": parsed_call,
            "parse_mode": parse_mode,
            "planner_steps": self.planner_steps,
            "parse_errors": self.parse_errors,
            "backend_errors": self.backend_errors,
        }
        if raw_output is not None:
            decision["raw_output"] = raw_output
        if message:
            decision["message"] = message
        if error:
            decision["error"] = error
        return decision


def make_distributed_policy(backend: str = "anthropic", model: Optional[str] = None,
                            comm: str = "none", verbose: bool = False, obs_mode: str = "image") -> DistributedVLMPolicy:
    return DistributedVLMPolicy(backend=backend, model=model, comm=comm, verbose=verbose, obs_mode=obs_mode)
