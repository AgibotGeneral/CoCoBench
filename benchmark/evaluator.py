"""Goal-predicate evaluation and state summaries for task configs.

Pure functions over an AI2-THOR event + task config. ``event.metadata['objects']``
is the single source of truth (read per-agent via ``thormeta``). The success
function is ``all(goal_predicates passed)``; coordination-quality metrics
(makespan, dependency violations, ...) are computed separately by the metrics
engine from the recorded trajectory, not here.
"""

from __future__ import annotations

from typing import Any, Dict, List

from taskutil import agent_names, aliases, resolve_object
from thormeta import metadata_for_agent, object_by_id


def predicate_status(event: Any, config: Dict[str, Any], predicate: Dict[str, Any],
                     trajectory: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    pred = predicate.get("predicate")
    if pred == "on_sliced_piece":
        # Slicing creates new pickupable pieces whose objectId is "<source>|<Type>Sliced_N";
        # the goal is satisfied when any such piece of the source sits on the receptacle.
        source_id = resolve_object(config, predicate.get("source", ""))
        receptacle_id = resolve_object(config, predicate.get("receptacle", ""))
        pieces = [
            obj for obj in metadata_for_agent(event).get("objects", [])
            if (obj.get("objectType") or "").endswith("Sliced")
            and (obj.get("objectId") or "").startswith(source_id + "|")
        ]
        matched = [obj for obj in pieces if receptacle_id in (obj.get("parentReceptacles") or [])]
        return {
            "predicate": predicate,
            "objectId": source_id,
            "passed": bool(matched),
            "detail": f"sliced_pieces={len(pieces)} on_{receptacle_id}={len(matched)}",
        }
    object_id = resolve_object(config, predicate.get("object", ""))
    obj = object_by_id(event, object_id)
    passed = False
    detail = ""
    if obj is None:
        detail = f"object not found: {object_id}"
    elif pred == "clean":
        passed = obj.get("isDirty") is False
        detail = f"isDirty={obj.get('isDirty')}"
    elif pred == "filled":
        liquid = predicate.get("liquid")
        actual_liquid = obj.get("fillLiquid")
        # AI2-THOR v2.1 often records isFilledWithLiquid=True while leaving fillLiquid as None.
        passed = bool(obj.get("isFilledWithLiquid")) and (not liquid or actual_liquid in {liquid, None, ""})
        detail = f"isFilledWithLiquid={obj.get('isFilledWithLiquid')} fillLiquid={actual_liquid}"
    elif pred == "empty":
        passed = not bool(obj.get("isFilledWithLiquid"))
        detail = f"isFilledWithLiquid={obj.get('isFilledWithLiquid')}"
    elif pred == "toggled":
        passed = bool(obj.get("isToggled")) == bool(predicate.get("value"))
        detail = f"isToggled={obj.get('isToggled')}"
    elif pred in ("open", "closed"):
        is_open = bool(obj.get("isOpen"))
        passed = is_open if pred == "open" else not is_open
        detail = f"isOpen={obj.get('isOpen')}"
    elif pred == "on":
        receptacle_id = resolve_object(config, predicate.get("receptacle", ""))
        parents = obj.get("parentReceptacles") or []
        passed = receptacle_id in parents
        detail = f"parentReceptacles={parents} expected={receptacle_id}"
    elif pred == "sliced":
        passed = bool(obj.get("isSliced"))
        detail = f"isSliced={obj.get('isSliced')}"
    elif pred == "sliced_by":
        required_agent = predicate.get("agent", "")
        is_sliced = bool(obj.get("isSliced"))
        matched = False
        for step in (trajectory or []):
            sr = step.get("skill_result") or {}
            if (sr.get("skill") == "Slice" and sr.get("success")
                    and sr.get("agent") == required_agent
                    and (sr.get("args") or {}).get("objectId") == object_id):
                matched = True
                break
        passed = is_sliced and matched
        detail = f"isSliced={is_sliced} required_agent={required_agent} matched={matched}"
    elif pred == "cooked":
        passed = bool(obj.get("isCooked"))
        detail = f"isCooked={obj.get('isCooked')}"
    else:
        detail = f"unsupported predicate: {pred}"
    return {"predicate": predicate, "objectId": object_id, "passed": passed, "detail": detail}


def evaluate_task(event: Any, config: Dict[str, Any], trajectory: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    checks = [predicate_status(event, config, pred, trajectory) for pred in config.get("goal_predicates", [])]
    return {"success": all(check["passed"] for check in checks), "checks": checks}


def state_summary(event: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"objects": {}, "agents": []}
    for alias, object_id in aliases(config).items():
        obj = object_by_id(event, object_id)
        if obj:
            out["objects"][alias] = {
                "objectId": object_id,
                "visible": obj.get("visible"),
                "isDirty": obj.get("isDirty"),
                "isFilledWithLiquid": obj.get("isFilledWithLiquid"),
                "fillLiquid": obj.get("fillLiquid"),
                "isToggled": obj.get("isToggled"),
                "parentReceptacles": obj.get("parentReceptacles"),
            }
    for index, name in enumerate(agent_names(config)):
        meta = metadata_for_agent(event, index)
        out["agents"].append({
            "agent": name,
            "inventory": [obj.get("objectId") for obj in meta.get("inventoryObjects", [])],
            "lastActionSuccess": meta.get("lastActionSuccess"),
            "errorMessage": meta.get("errorMessage") or "",
        })
    return out
