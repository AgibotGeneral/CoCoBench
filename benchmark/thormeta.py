"""AI2-THOR event/metadata access helpers (single-agent and multi-agent).

Centralizes the per-agent event indexing used across the benchmark so that the
env, evaluator, action space, and navigation all read agent-local metadata the
same way (object lookup, visibility, inventory, agent pose, peer positions).
``event.events[i]`` is the per-agent Event in a multi-agent (``agentCount>1``)
``MultiAgentEvent``; for single-agent runs ``event`` itself carries metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def event_for_agent(event: Any, agent_index: int) -> Any:
    events = getattr(event, "events", None)
    if events and 0 <= agent_index < len(events):
        return events[agent_index]
    return event


def metadata_for_agent(event: Any, agent_index: int = 0) -> Dict[str, Any]:
    return getattr(event_for_agent(event, agent_index), "metadata", {}) or {}


def frame_for_agent(event: Any, agent_index: int) -> Any:
    return getattr(event_for_agent(event, agent_index), "frame", None)


def objects_for_agent(event: Any, agent_index: Optional[int] = 0) -> List[Dict[str, Any]]:
    return metadata_for_agent(event, agent_index or 0).get("objects", [])


def object_by_id(event: Any, object_id: str, agent_index: int = 0) -> Optional[Dict[str, Any]]:
    for obj in objects_for_agent(event, agent_index):
        if obj.get("objectId") == object_id:
            return obj
    return None


def object_visible(event: Any, object_id: str, agent_index: Optional[int] = 0) -> bool:
    """Whether ``object_id`` is currently visible in the given agent's view."""
    for obj in objects_for_agent(event, agent_index):
        if obj.get("objectId") == object_id:
            return bool(obj.get("visible"))
    return False


def held_object_id(event: Any, agent_index: Optional[int] = 0) -> Optional[str]:
    """objectId of the first inventory object for the agent, or None."""
    inventory = metadata_for_agent(event, agent_index or 0).get("inventoryObjects", [])
    return inventory[0].get("objectId") if inventory else None


def object_holder(event: Any, object_id: str) -> Optional[int]:
    """Index of the agent whose inventory currently holds ``object_id``, or None.

    Used to enforce pickup mutual-exclusion: AI2-THOR's ``PickupObject`` with
    ``forceAction=True`` will happily teleport an object out of another agent's
    hand, so the engine never denies a contested PickUp on its own. This lets the
    executor reject it (the D3 competitive-collection ground-truth signal).
    """
    events = getattr(event, "events", None)
    if not events:
        inv = metadata_for_agent(event, 0).get("inventoryObjects", []) or []
        return 0 if any(o.get("objectId") == object_id for o in inv) else None
    for index, agent_event in enumerate(events):
        inv = (getattr(agent_event, "metadata", {}) or {}).get("inventoryObjects", []) or []
        if any(o.get("objectId") == object_id for o in inv):
            return index
    return None


def agent_pose(event: Any, agent_index: Optional[int] = 0) -> Dict[str, Any]:
    """TeleportFull-shaped pose dict for the agent, or {} if no position metadata."""
    agent_meta = metadata_for_agent(event, agent_index or 0).get("agent") or {}
    position = agent_meta.get("position") or {}
    if not position:
        return {}
    rotation = agent_meta.get("rotation") or {}
    return {
        "x": position.get("x"),
        "y": position.get("y", 0.9),
        "z": position.get("z"),
        "rotation": rotation.get("y", 0),
        "horizon": agent_meta.get("cameraHorizon", 0),
        "standing": agent_meta.get("isStanding", True),
    }


def other_agent_positions(event: Any, agent_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Positions of every agent except ``agent_id`` (None excludes none)."""
    events = getattr(event, "events", None)
    if not events:
        return []
    positions: List[Dict[str, Any]] = []
    for index, agent_event in enumerate(events):
        if agent_id is not None and index == agent_id:
            continue
        meta = getattr(agent_event, "metadata", {}) or {}
        position = (meta.get("agent") or {}).get("position")
        if position:
            positions.append(position)
    return positions
