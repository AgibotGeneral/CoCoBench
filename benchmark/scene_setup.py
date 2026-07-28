"""Apply a task config's ``init_state.init_actions`` to a live controller.

These run once at ``reset`` to put the scene into its declared starting state
(toggle a light off, empty a container, assert an object is present, or run a
raw native action). The semantics are documented in the repository README under
the runner init_actions section. Returns a per-action trace for the trajectory log.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import thormeta


_FORCE_ACTIONS = {"DirtyObject", "CleanObject", "FillObjectWithLiquid", "EmptyLiquidFromObject"}


def native_step(controller: Any, action: Dict[str, Any], agent_id: Optional[int] = None) -> Dict[str, Any]:
    """Run one native action and return a serializable trace entry."""
    native = dict(action)
    if agent_id is not None:
        native["agentId"] = agent_id
    try:
        event = controller.step(native)
        meta = thormeta.metadata_for_agent(event, agent_id or 0)
        return {"action": native, "lastActionSuccess": bool(meta.get("lastActionSuccess", True)), "errorMessage": meta.get("errorMessage") or ""}
    except Exception as exc:
        return {"action": native, "lastActionSuccess": False, "errorMessage": repr(exc)}


def apply_init_actions(controller: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    trace: List[Dict[str, Any]] = []
    for action in config.get("init_state", {}).get("init_actions", []):
        name = action.get("action")
        object_id = action.get("objectId")
        if name == "ensure_toggle_state":
            obj = thormeta.object_by_id(controller.last_event, object_id)
            desired = bool(action.get("isToggled"))
            if obj and bool(obj.get("isToggled")) != desired:
                native = {"action": "ToggleObjectOn" if desired else "ToggleObjectOff", "objectId": object_id, "forceAction": True}
                trace.append(native_step(controller, native, agent_id=0))
            else:
                trace.append({"action": action, "lastActionSuccess": True, "errorMessage": "already satisfied"})
        elif name == "assert_present":
            obj = thormeta.object_by_id(controller.last_event, object_id)
            present = obj is not None
            trace.append({"action": action, "lastActionSuccess": present, "errorMessage": "" if present else f"object not present: {object_id}"})
        elif name == "ensure_empty":
            obj = thormeta.object_by_id(controller.last_event, object_id)
            if obj and obj.get("isFilledWithLiquid"):
                trace.append(native_step(controller, {"action": "EmptyLiquidFromObject", "objectId": object_id}, agent_id=0))
            else:
                trace.append({"action": action, "lastActionSuccess": True, "errorMessage": "already empty"})
        else:
            native = dict(action)
            if name in _FORCE_ACTIONS:
                native.setdefault("forceAction", True)
            trace.append(native_step(controller, native, agent_id=None))
    return trace
