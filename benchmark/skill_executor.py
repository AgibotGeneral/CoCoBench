"""Execute benchmark-level skills with AI2-THOR native actions.

The collector only records high-level skills. This module is the execution layer:
it maps each agent-facing skill to one or more ``controller.step`` calls and
returns a compact native action trace.

It owns *what the skills are* (the ``_skill_*`` handlers and their affordance
preconditions); *how to navigate and step* the controller is delegated to
:class:`navigation.Navigator`. This mirrors the split used by related AI2-THOR
work (LLaMAR/MAP-THOR, SMART-LLM): object navigation in a wrapper, object
interactions as mostly-direct AI2-THOR actions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import thormeta
from navigation import Navigator


EXPLORE_DIRECTION_TO_AI2THOR_ACTION = {
    "forward": "MoveAhead",
    "back": "MoveBack",
    "left": "MoveLeft",
    "right": "MoveRight",
    "turn_left": "RotateLeft",
    "turn_right": "RotateRight",
    "look_up": "LookUp",
    "look_down": "LookDown",
}

EXPLORE_TRANSLATION_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
DEFAULT_EXPLORE_MOVE_MAGNITUDE = 0.25

SLICER_TYPES = {"Knife", "ButterKnife"}

CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$")


@dataclass
class SkillExecutionResult:
    skill: str
    agent: str
    args: Dict[str, Any]
    success: bool
    errorMessage: str = ""
    native_trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill": self.skill,
            "agent": self.agent,
            "args": self.args,
            "success": self.success,
            "errorMessage": self.errorMessage,
            "native_trace": self.native_trace,
        }


class SkillExecutor:
    """Dispatch high-level benchmark skills to AI2-THOR native actions.

    Parameters
    ----------
    controller:
        An ``ai2thor.controller.Controller``-like object.
    navigation_mode:
        Passed through to :class:`navigation.Navigator` (``"teleport"`` for
        oracle/debug execution; ``"primitive"`` is reserved).
    check_preconditions:
        When true, lightweight metadata affordance checks are run before direct
        object interactions. AI2-THOR remains the source of truth.
    """

    def __init__(self, controller: Any, navigation_mode: str = "teleport", check_preconditions: bool = True) -> None:
        if navigation_mode not in {"teleport", "primitive"}:
            raise ValueError(f"Unsupported navigation_mode: {navigation_mode}")
        self.controller = controller
        self.navigation_mode = navigation_mode
        self.check_preconditions = check_preconditions
        self.nav = Navigator(controller, navigation_mode=navigation_mode)

    def execute_call(self, call: str) -> SkillExecutionResult:
        """Execute a string call such as ``Put(agent_1, Fridge|0)``."""
        skill_name, args = self.parse_call(call)
        return self.dispatch(skill_name, *args)

    @staticmethod
    def parse_call(call: str) -> Tuple[str, List[str]]:
        match = CALL_RE.match(call)
        if not match:
            raise ValueError(f"Invalid skill call: {call}")
        skill_name = match.group(1)
        raw_args = match.group(2).strip()
        args = [] if not raw_args else [part.strip() for part in raw_args.split(",")]
        return skill_name, args

    def dispatch(self, skill_name: str, *positional_args: Any, **keyword_args: Any) -> SkillExecutionResult:
        handler = getattr(self, f"_skill_{skill_name}", None)
        if handler is None:
            return SkillExecutionResult(skill_name, str(keyword_args.get("agent", positional_args[0] if positional_args else "")), keyword_args, False, f"Unsupported skill: {skill_name}")
        return handler(*positional_args, **keyword_args)

    # ---- skill handlers --------------------------------------------------
    def _skill_Find(self, agent: Any, objectId: str) -> SkillExecutionResult:
        def body(trace: List[Dict[str, Any]]) -> Tuple[bool, str]:
            original_pose = self.nav.agent_pose(agent)
            ok, error = self.nav.find_object(agent, objectId, trace)
            if not ok and original_pose:
                self.nav.teleport_to_pose(agent, original_pose, trace)
            return ok, error
        return self._run("Find", agent, {"objectId": objectId}, body)

    def _skill_Explore(self, agent: Any, direction: str) -> SkillExecutionResult:
        native_action = EXPLORE_DIRECTION_TO_AI2THOR_ACTION.get(direction)
        if not native_action:
            return SkillExecutionResult("Explore", str(agent), {"direction": direction}, False, f"Unsupported direction: {direction}")
        action = {"action": native_action}
        if native_action in EXPLORE_TRANSLATION_ACTIONS:
            action["moveMagnitude"] = DEFAULT_EXPLORE_MOVE_MAGNITUDE
        return self._run("Explore", agent, {"direction": direction}, lambda trace: self.nav.step(agent, action, trace))

    def _skill_Wait(self, agent: Any) -> SkillExecutionResult:
        return self._run("Wait", agent, {}, lambda trace: self.nav.step(agent, {"action": "Pass"}, trace))

    def _skill_PickUp(self, agent: Any, objectId: str) -> SkillExecutionResult:
        # Mutual exclusion: another agent already holding this object means the
        # claim is contended. forceAction below would otherwise *steal* it from
        # their hand (the engine does not deny a contested PickUp), so reject here
        # — this is the D3 competitive-collection ground-truth ("loser fails").
        if self.check_preconditions and self.nav.held_by_other_agent(agent, objectId):
            return SkillExecutionResult(
                "PickUp", str(agent), {"objectId": objectId}, False,
                f"{objectId} is already held by another agent",
            )
        # AI2-THOR 5.0 runs a strict collide/clip check when teleporting the
        # object into the hand; forceAction skips it. Find already navigates the
        # agent to a visible interactable pose first, so the pickup is legitimate.
        return self._run_direct_object_skill("PickUp", agent, objectId, "pickupable", {"action": "PickupObject", "objectId": objectId, "forceAction": True})

    def _skill_Put(self, agent: Any, receptacleId: str) -> SkillExecutionResult:
        object_id = self.nav.held_object_id(agent)
        if not object_id:
            return SkillExecutionResult("Put", str(agent), {"receptacleId": receptacleId}, False, "Agent is not holding any object")

        def body(trace: List[Dict[str, Any]]) -> Tuple[bool, str]:
            if self.check_preconditions:
                ok, error = self.nav.check_object(object_id)
                if not ok:
                    return False, error
                ok, error = self.nav.check_object(receptacleId, "receptacle")
                if not ok:
                    return False, error
                robj = thormeta.object_by_id(self.controller.last_event, receptacleId)
                if robj and robj.get("openable") and not robj.get("isOpen"):
                    return False, f"{receptacleId} is closed; Open it first"
            # AI2-THOR 5.0: PutObject takes the receptacle as ``objectId`` and
            # places whatever the agent is holding (the 2.1.0 ``receptacleObjectId``
            # / explicit held ``objectId`` pair was removed). ``forceAction`` skips
            # 5.0's stricter free-position search, matching the EB-ALFRED convention.
            return self.nav.step(agent, {"action": "PutObject", "objectId": receptacleId, "forceAction": True, "placeStationary": True}, trace)
        return self._run("Put", agent, {"objectId": object_id, "receptacleId": receptacleId}, body)

    def _skill_Drop(self, agent: Any) -> SkillExecutionResult:
        # AI2-THOR 5.0 refuses to drop unless the held object is collision-clear;
        # forceAction makes Drop behave like the 2.1.0 release-from-hand.
        return self._run("Drop", agent, {}, lambda trace: self.nav.step(agent, {"action": "DropHandObject", "forceAction": True}, trace))

    def _skill_Open(self, agent: Any, objectId: str) -> SkillExecutionResult:
        return self._run_direct_object_skill("Open", agent, objectId, "openable", {"action": "OpenObject", "objectId": objectId, "forceAction": True})

    def _skill_Close(self, agent: Any, objectId: str) -> SkillExecutionResult:
        return self._run_direct_object_skill("Close", agent, objectId, "openable", {"action": "CloseObject", "objectId": objectId, "forceAction": True})

    def _skill_ToggleOn(self, agent: Any, objectId: str) -> SkillExecutionResult:
        return self._run_direct_object_skill("ToggleOn", agent, objectId, "toggleable", {"action": "ToggleObjectOn", "objectId": objectId})

    def _skill_ToggleOff(self, agent: Any, objectId: str) -> SkillExecutionResult:
        return self._run_direct_object_skill("ToggleOff", agent, objectId, "toggleable", {"action": "ToggleObjectOff", "objectId": objectId})

    def _skill_Slice(self, agent: Any, objectId: str) -> SkillExecutionResult:
        if self.check_preconditions:
            held_id = self.nav.held_object_id(agent)
            if not held_id:
                return SkillExecutionResult(
                    "Slice", str(agent), {"objectId": objectId}, False,
                    "Agent must hold a Knife or ButterKnife to slice",
                )
            held_obj = thormeta.object_by_id(self.controller.last_event, held_id)
            if not held_obj or held_obj.get("objectType") not in SLICER_TYPES:
                return SkillExecutionResult(
                    "Slice", str(agent), {"objectId": objectId}, False,
                    f"Agent is holding {held_id}, not a Knife/ButterKnife",
                )
        return self._run_direct_object_skill("Slice", agent, objectId, "sliceable", {"action": "SliceObject", "objectId": objectId})

    def _skill_CleanObject(self, agent: Any, objectId: str) -> SkillExecutionResult:
        return self._run_direct_object_skill("CleanObject", agent, objectId, "dirtyable", {"action": "CleanObject", "objectId": objectId})

    def _skill_FillObjectWithLiquid(self, agent: Any, objectId: str, liquid: str) -> SkillExecutionResult:
        return self._run_direct_object_skill("FillObjectWithLiquid", agent, objectId, None, {"action": "FillObjectWithLiquid", "objectId": objectId, "fillLiquid": liquid})

    def _skill_EmptyLiquidFromObject(self, agent: Any, objectId: str) -> SkillExecutionResult:
        return self._run_direct_object_skill("EmptyLiquidFromObject", agent, objectId, None, {"action": "EmptyLiquidFromObject", "objectId": objectId})

    def _skill_PushObject(self, agent: Any, objectId: str, moveMagnitude: float = 150.0) -> SkillExecutionResult:
        return self._run_direct_object_skill("PushObject", agent, objectId, None, {"action": "PushObject", "objectId": objectId, "moveMagnitude": moveMagnitude})

    def _skill_PullObject(self, agent: Any, objectId: str, moveMagnitude: float = 150.0) -> SkillExecutionResult:
        return self._run_direct_object_skill("PullObject", agent, objectId, None, {"action": "PullObject", "objectId": objectId, "moveMagnitude": moveMagnitude})

    def _skill_BreakObject(self, agent: Any, objectId: str) -> SkillExecutionResult:
        return self._run_direct_object_skill("BreakObject", agent, objectId, "breakable", {"action": "BreakObject", "objectId": objectId})

    # ---- dispatch plumbing ----------------------------------------------
    def _run(self, skill: str, agent: Any, args: Dict[str, Any], body: Callable[[List[Dict[str, Any]]], Tuple[bool, str]]) -> SkillExecutionResult:
        trace: List[Dict[str, Any]] = []
        try:
            success, error = body(trace)
        except Exception as exc:  # Keep executor failures serializable for runners.
            return SkillExecutionResult(skill, str(agent), args, False, repr(exc), trace)
        return SkillExecutionResult(skill, str(agent), args, success, error, trace)

    def _run_direct_object_skill(self, skill: str, agent: Any, objectId: str, required_affordance: Optional[str], action: Dict[str, Any]) -> SkillExecutionResult:
        def body(trace: List[Dict[str, Any]]) -> Tuple[bool, str]:
            if self.check_preconditions:
                ok, error = self.nav.check_object(objectId, required_affordance)
                if not ok:
                    return False, error
            return self.nav.step(agent, action, trace)
        return self._run(skill, agent, {"objectId": objectId}, body)
