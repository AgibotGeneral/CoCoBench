"""Gym-style environment wrapper for multi-agent AI2-THOR task configs.

This is the simulator-facing core of the benchmark skeleton. A ``Policy``
(oracle, scripted, or VLM) drives it through the harness loop:

    env = MultiAgentThorEnv(config, render)
    obs = env.reset()                 # start controller, apply init_actions
    result = env.step(call)           # execute one skill call, record a step
    obs = env.observe()               # rebuild per-agent observation
    final = env.finalize()            # write trajectory.json, return final eval

The env owns the controller lifecycle, the skill executor, and observation
packaging. Scene setup (``init_actions``) is delegated to :mod:`scene_setup`,
and step/trajectory recording + frame rendering to
:class:`recorder.TrajectoryRecorder`. It exposes a read-only ``EnvView`` for
oracle/privileged policies to query metadata (sliced pieces, knob ids, ...).
Fair (VLM) policies should consume only ``Observation``.

Controller lifecycle uses the AI2-THOR 5.0 API: a single ``Controller(...)``
constructor call (CloudRendering platform, ``agentCount`` passed inline) that
both launches the build and initializes the scene — no separate ``start`` /
``reset`` / ``Initialize`` sequence. Everything downstream (``controller.step``
with a dict action, per-agent ``event.events[i]`` fan-out, settable
``last_event``) is unchanged from 2.1.0 and verified against the live 5.0 build.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from action_space import MultiAgentActionSpace
from evaluator import evaluate_task
from recorder import TrajectoryRecorder
from scene_setup import apply_init_actions
from skill_executor import SkillExecutor, SkillExecutionResult
from taskutil import agent_names, compact_entries, predicate_to_nl
from thormeta import metadata_for_agent, object_by_id


@dataclass
class RenderConfig:
    x_display: str = "0"  # unused under CloudRendering; kept for CLI back-compat
    width: int = 400
    height: int = 400
    quality: str = "Low"
    platform: str = "CloudRendering"
    gpu_device: int = 0


@dataclass
class Observation:
    """What a decision maker sees at one step.

    This is the observation contract the VLM policy will consume. ``per_agent``
    holds egocentric image paths + inventory + last feedback; ``action_menu`` is
    the EB-ALFRED-style discrete action list filtered to allowed skills.
    """

    step_index: int
    goal_text: str
    per_agent: Dict[str, Dict[str, Any]]
    action_menu: List[Dict[str, Any]]
    eval: Dict[str, Any]
    image: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    object_mapping: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "goal_text": self.goal_text,
            "per_agent": self.per_agent,
            "action_menu": self.action_menu,
            "eval": self.eval,
            "image": self.image,
            "messages": self.messages or [],
            "object_mapping": self.object_mapping,
        }


class EnvView:
    """Read-only metadata queries over the live controller state.

    Privileged surface for oracle/scripted policies (they may know object ids and
    object types). VLM policies should not use this — they get :class:`Observation`.
    """

    def __init__(self, env: "MultiAgentThorEnv") -> None:
        self._env = env

    @property
    def last_event(self) -> Any:
        return getattr(self._env.controller, "last_event", None)

    def objects(self, agent_index: int = 0) -> List[Dict[str, Any]]:
        return metadata_for_agent(self.last_event, agent_index).get("objects", [])

    def object_by_id(self, object_id: str) -> Optional[Dict[str, Any]]:
        return object_by_id(self.last_event, object_id)

    def object_ids_of_type(self, object_type: str) -> List[str]:
        return [
            obj.get("objectId")
            for obj in self.objects()
            if obj.get("objectType") == object_type and obj.get("objectId")
        ]

    def sliced_pieces(self, source_id: str) -> List[str]:
        """All sliced-piece objectIds produced from ``source_id`` (appear after SliceObject)."""
        return [
            obj.get("objectId")
            for obj in self.objects()
            if (obj.get("objectType") or "").endswith("Sliced")
            and (obj.get("objectId") or "").startswith(source_id + "|")
        ]


class MultiAgentThorEnv:
    def __init__(self, config: Dict[str, Any], render: Optional[RenderConfig] = None, output_dir: Optional[Path] = None, controller: Any = None, save_images: bool = False) -> None:
        self.config = config
        self.render = render or RenderConfig()
        self.output_dir = Path(output_dir) if output_dir else (
            Path(__file__).resolve().parent / "outputs" / "task_runs" / str(config.get("task_id", "task"))
        )
        self.controller = controller  # may be injected (tests); otherwise built on reset
        self._owns_controller = controller is None
        self.executor: Optional[SkillExecutor] = None
        self.space: Optional[MultiAgentActionSpace] = None
        self.agent_names: Sequence[str] = agent_names(config)
        self.step_index: int = 0
        self.init_trace: List[Dict[str, Any]] = []
        self.recorder = TrajectoryRecorder(self.output_dir, self.agent_names, config, save_images=save_images)
        # Shared message board for the distributed policy (D1 comm condition). Each
        # entry is {round, agent, text}; the centralized policy never reads it.
        self._board: List[Dict[str, Any]] = []
        self.round_index: int = 0

    # ---- lifecycle -------------------------------------------------------
    def reset(self) -> Observation:
        if self.controller is None:
            self.controller = self._start_controller()
        # Reset to the scene before applying init_actions. AI2-THOR's
        # InitialRandomSpawn(randomSeed=...) is deterministic only when issued after a
        # controller.reset(scene) (the constructor's boot leaves the spawn RNG in a
        # non-reproducible state); the depth-axis seeded configs are generated via
        # reset+spawn, so eval must reset first to reproduce the recorded layout.
        init = self.config.get("init_state", {}).get("controller_init", {})
        scene = init.get("scene") or self.config.get("scene_id")
        if scene:
            self.controller.reset(scene=scene)
        self.executor = SkillExecutor(self.controller)
        self.init_trace = apply_init_actions(self.controller, self.config)
        self.step_index = 0
        self.round_index = 0
        self._board = []
        self.recorder.reset()
        self._record("init", native_trace=self.init_trace)
        self._rebuild_space()
        return self.observe()

    def _start_controller(self) -> Any:
        from ai2thor.controller import Controller  # imported lazily; needs the thor5 env

        init = self.config.get("init_state", {}).get("controller_init", {})
        scene = init.get("scene") or self.config.get("scene_id")
        agent_count = init.get("agentCount", self.config.get("agent_count", 1))
        kwargs: Dict[str, Any] = dict(
            agentMode="default",
            scene=scene,
            gridSize=init.get("gridSize", 0.25),
            visibilityDistance=init.get("visibilityDistance", 1.5),
            renderImage=True,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=max(self.render.width, 300),
            height=max(self.render.height, 300),
            agentCount=agent_count,
            quality=self.render.quality,
        )
        if self.render.platform == "CloudRendering":
            from ai2thor.platform import CloudRendering

            kwargs["platform"] = CloudRendering
            kwargs["gpu_device"] = self.render.gpu_device
        # AI2-THOR 5.0: the constructor launches the build and initializes the
        # scene in one call (the 2.1.0 start/reset/Initialize trio is gone).
        return Controller(**kwargs)

    def stop(self) -> None:
        if self.controller is not None and self._owns_controller:
            self.controller.stop()
            self.controller = None

    # ---- stepping --------------------------------------------------------
    def step(self, call: str, round_index: Optional[int] = None, message: Optional[str] = None) -> SkillExecutionResult:
        """Execute one high-level skill call (e.g. ``Put(agent_1, Plate|...)``) and record a step.

        ``round_index`` / ``message`` are recorded for the distributed concurrent
        loop (which agent acted in which decision round, and any message it broadcast)
        and are ``None`` for the centralized turn-based loop."""
        if self.executor is None:
            raise RuntimeError("Env not reset; call reset() before step().")
        result = self.executor.execute_call(call)
        self.step_index += 1
        self._record(call, result=result, round_index=round_index, message=message)
        self._rebuild_space()
        return result

    def post_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Append this round's broadcast messages to the shared board (D1 comm).

        ``observe()`` then surfaces the board so each distributed agent reads what
        its teammates said last round. A no-op for conditions that never call it."""
        self._board.extend(m for m in messages if (m or {}).get("text"))

    def advance_physics(self, n: int = 5, time_step: float = 1.0) -> None:
        """Advance the physics simulation without recording a trajectory step.

        Used by oracle cook plans to let state changes settle. Not an agent skill.
        """
        for _ in range(n):
            self.controller.step({"action": "AdvancePhysicsStep", "timeStep": time_step})

    # ---- observation / evaluation ---------------------------------------
    def view(self) -> EnvView:
        return EnvView(self)

    def evaluate(self) -> Dict[str, Any]:
        return evaluate_task(self.controller.last_event, self.config,
                             self.recorder.trajectory)

    def goal_text(self, obj_map: Optional[Dict[str, str]] = None) -> str:
        preds = self.config.get("goal_predicates", [])
        task_name = self.config.get("task_name", self.config.get("task_id", ""))
        title = task_name.replace("_", " ").capitalize()
        om = obj_map or {}
        items = "\n".join(
            f"  {i + 1}. {predicate_to_nl(p, om)}" for i, p in enumerate(preds)
        )
        return f"{title}:\n{items}"

    def observe(self) -> Observation:
        event = self.controller.last_event
        agent_images = self.recorder.last_agent_images
        per_agent: Dict[str, Dict[str, Any]] = {}
        for index, name in enumerate(self.agent_names):
            meta = metadata_for_agent(event, index)
            img = agent_images.get(name)
            per_agent[name] = {
                "inventory": [obj.get("objectId") for obj in meta.get("inventoryObjects", [])],
                "lastActionSuccess": meta.get("lastActionSuccess"),
                "errorMessage": meta.get("errorMessage") or "",
                "image": str(img) if img else None,
            }
        menu = [
            {"action_id": e.action_id, "agent": e.agent, "name": e.action_name, "call": e.call}
            for e in (
                compact_entries(self.space, self.config, objects=metadata_for_agent(event, 0).get("objects", []))
                if self.space else []
            )
        ]
        space_names = self.space.id_to_name if self.space else {}
        obj_map = {}
        for alias, oid in self.config.get("init_state", {}).get("objects", {}).items():
            obj_map[alias] = space_names.get(oid, oid.split("|")[0])
        return Observation(
            step_index=self.step_index,
            goal_text=self.goal_text(obj_map),
            per_agent=per_agent,
            action_menu=menu,
            eval=self.evaluate(),
            image=str(self.recorder.last_image) if self.recorder.last_image else None,
            messages=list(self._board),
            object_mapping=obj_map or None,
        )

    # ---- internals -------------------------------------------------------
    @property
    def _last_image(self) -> Optional[Path]:
        """Path of the most recent saved observation (kept for CLI compatibility)."""
        return self.recorder.last_image

    def _rebuild_space(self) -> None:
        self.space = MultiAgentActionSpace.from_controller(self.controller, agent_names=self.agent_names)

    def _record(self, label: str, result: Optional[SkillExecutionResult] = None, native_trace: Optional[List[Dict[str, Any]]] = None, policy_decision: Optional[Dict[str, Any]] = None, round_index: Optional[int] = None, message: Optional[str] = None) -> None:
        self.recorder.record(self.step_index, self.controller.last_event, label, result=result, native_trace=native_trace, policy_decision=policy_decision, round_index=round_index, message=message)

    def attach_policy_decision(self, policy_decision: Dict[str, Any]) -> None:
        self.recorder.attach_policy_decision(policy_decision)

    def record_policy_stop(self, policy_decision: Dict[str, Any]) -> None:
        self._record("policy_stop", policy_decision=policy_decision)

    def finalize(self) -> Dict[str, Any]:
        return self.recorder.finalize(self.init_trace)
