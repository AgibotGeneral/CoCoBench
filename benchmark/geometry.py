"""Pure geometric / parsing helpers for navigation.

Stateless functions shared by the navigator: agent-id parsing and the
distance / heading / camera-horizon math used to rank and orient teleport
poses. No controller or metadata access here — callers pass plain dicts.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional


def agent_id_from(agent: Any) -> Optional[int]:
    """Map an agent handle (``"agent_1"``, an int, or a digit string) to a
    zero-based AI2-THOR ``agentId``; ``None`` if it cannot be parsed."""
    if agent is None:
        return None
    if isinstance(agent, int):
        return agent
    text = str(agent)
    match = re.fullmatch(r"agent_(\d+)", text)
    if match:
        value = int(match.group(1))
        return value - 1 if value >= 1 else 0
    if text.isdigit():
        return int(text)
    return None


def distance_sq(position: Dict[str, Any], pose: Dict[str, Any]) -> float:
    px = pose.get("x", pose.get("position", {}).get("x", 0.0))
    py = pose.get("y", pose.get("position", {}).get("y", 0.0))
    pz = pose.get("z", pose.get("position", {}).get("z", 0.0))
    return (
        (float(position.get("x", 0.0)) - float(px)) ** 2
        + (float(position.get("y", 0.0)) - float(py)) ** 2
        + (float(position.get("z", 0.0)) - float(pz)) ** 2
    )


def horizontal_distance_sq(position: Dict[str, Any], pose: Dict[str, Any]) -> float:
    px = float(position.get("x", 0.0))
    pz = float(position.get("z", 0.0))
    qx = float(pose.get("x", pose.get("position", {}).get("x", 0.0)))
    qz = float(pose.get("z", pose.get("position", {}).get("z", 0.0)))
    return (px - qx) ** 2 + (pz - qz) ** 2


def yaw_to_face(position: Dict[str, Any], target: Dict[str, Any]) -> float:
    px = float(position.get("x", position.get("position", {}).get("x", 0.0)))
    pz = float(position.get("z", position.get("position", {}).get("z", 0.0)))
    tx = float(target.get("x", target.get("position", {}).get("x", 0.0)))
    tz = float(target.get("z", target.get("position", {}).get("z", 0.0)))
    return math.degrees(math.atan2(tx - px, tz - pz)) % 360


def horizons_to_try(position: Dict[str, Any], target: Dict[str, Any]) -> List[float]:
    py = float(position.get("y", position.get("position", {}).get("y", 0.9)))
    ty = float(target.get("y", target.get("position", {}).get("y", py)))
    if ty < py + 0.3:
        return [30.0, 0.0, 60.0, -30.0]
    return [0.0, 30.0, -30.0, 60.0]
