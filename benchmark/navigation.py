"""Controller-bound navigation and stepping for the skill executor.

The :class:`Navigator` owns the AI2-THOR controller interaction that the
high-level skills delegate to: issuing native ``controller.step`` calls (with
per-agent ``agentId`` and the OpenObject/CloseObject ``last_event`` fix-up),
lightweight affordance checks, and ``Find``'s teleport-based object navigation
(``GetInteractablePoses`` first, then ``GetReachablePositions`` ranked by
distance and oriented to face the target). Pure math lives in :mod:`geometry`;
metadata reads go through :mod:`thormeta`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import geometry
import thormeta


MIN_AGENT_SEPARATION = 0.45


class Navigator:
    """Navigation / stepping layer bound to a single controller.

    Parameters
    ----------
    controller:
        An ``ai2thor.controller.Controller``-like object.
    navigation_mode:
        ``"teleport"`` tries ``GetInteractablePoses`` and falls back to
        ``GetReachablePositions`` + ``TeleportFull``. ``"primitive"`` is reserved
        for a future shortest-path controller that expands into Move/Rotate/Look.
    """

    def __init__(self, controller: Any, navigation_mode: str = "teleport") -> None:
        self.controller = controller
        self.navigation_mode = navigation_mode

    # ---- stepping --------------------------------------------------------
    def step(self, agent: Any, action: Dict[str, Any], trace: List[Dict[str, Any]]) -> Tuple[bool, str]:
        _event, success, error = self.step_event(agent, action, trace)
        return success, error

    def step_event(self, agent: Any, action: Dict[str, Any], trace: List[Dict[str, Any]]) -> Tuple[Any, bool, str]:
        native_action = dict(action)
        agent_id = geometry.agent_id_from(agent)
        if agent_id is not None:
            native_action["agentId"] = agent_id
        previous_event = getattr(self.controller, "last_event", None)
        if agent_id is not None and native_action.get("action") in {"OpenObject", "CloseObject"}:
            events = getattr(previous_event, "events", None)
            if events and 0 <= agent_id < len(events):
                self.controller.last_event = events[agent_id]
        try:
            event = self.controller.step(native_action)
        except Exception:
            if previous_event is not None:
                self.controller.last_event = previous_event
            raise
        metadata = thormeta.metadata_for_agent(event, agent_id or 0)
        success = bool(metadata.get("lastActionSuccess", True))
        error = metadata.get("errorMessage") or ""
        trace.append({"action": native_action, "lastActionSuccess": success, "errorMessage": error})
        return event, success, error

    # ---- metadata reads (agent-local) ------------------------------------
    def check_object(self, object_id: str, affordance: Optional[str] = None) -> Tuple[bool, str]:
        obj = thormeta.object_by_id(self.controller.last_event, object_id)
        if obj is None:
            return False, f"Object not found in metadata: {object_id}"
        if affordance and not obj.get(affordance):
            return False, f"Object {object_id} does not satisfy affordance: {affordance}"
        return True, ""

    def held_object_id(self, agent: Any) -> Optional[str]:
        return thormeta.held_object_id(self.controller.last_event, geometry.agent_id_from(agent))

    def held_by_other_agent(self, agent: Any, object_id: str) -> bool:
        """True if ``object_id`` is currently in *another* agent's inventory."""
        holder = thormeta.object_holder(self.controller.last_event, object_id)
        return holder is not None and holder != geometry.agent_id_from(agent)

    def agent_pose(self, agent: Any) -> Dict[str, Any]:
        return thormeta.agent_pose(self.controller.last_event, geometry.agent_id_from(agent))

    # ---- Find: teleport-based object navigation --------------------------
    def find_object(self, agent: Any, object_id: str, trace: List[Dict[str, Any]]) -> Tuple[bool, str]:
        if self.navigation_mode == "primitive":
            return False, "Primitive navigation is not implemented yet; use navigation_mode='teleport' for oracle/debug execution."
        ok, error = self.check_object(object_id, None)
        if not ok:
            return ok, error
        try:
            event, success, error = self.step_event(agent, {"action": "GetInteractablePoses", "objectId": object_id}, trace)
        except ValueError:
            event, success, error = None, False, ""
        if success:
            metadata = thormeta.metadata_for_agent(event, geometry.agent_id_from(agent) or 0)
            poses = metadata.get("actionReturn") or []
            if poses:
                last_error = ""
                for pose in self.rank_poses_for_agent(agent, poses):
                    if self.pose_too_close_to_other_agent(agent, pose):
                        continue
                    ok, last_error = self.teleport_to_pose(agent, pose, trace)
                    if ok and thormeta.object_visible(self.controller.last_event, object_id, geometry.agent_id_from(agent)):
                        return True, ""
                if last_error:
                    return False, last_error
        return self.find_reachable_object_pose(agent, object_id, trace)

    def find_reachable_object_pose(self, agent: Any, object_id: str, trace: List[Dict[str, Any]]) -> Tuple[bool, str]:
        obj = thormeta.object_by_id(self.controller.last_event, object_id)
        target = (obj or {}).get("position") or {}
        if not target:
            return False, f"Object has no position metadata: {object_id}"
        event, success, error = self.step_event(agent, {"action": "GetReachablePositions"}, trace)
        if not success:
            return False, error
        metadata = thormeta.metadata_for_agent(event, geometry.agent_id_from(agent) or 0)
        positions = metadata.get("actionReturn") or metadata.get("reachablePositions") or []
        if not positions:
            return False, f"No reachable positions found for {object_id}"
        candidates = sorted(positions, key=lambda pose: self.navigation_pose_score(agent, target, pose))[:16]
        last_error = ""
        for pose in candidates:
            if self.pose_too_close_to_other_agent(agent, pose):
                continue
            rotation = geometry.yaw_to_face(pose, target)
            for horizon in geometry.horizons_to_try(pose, target):
                ok, last_error = self.teleport_to_pose(agent, {**pose, "rotation": rotation, "horizon": horizon}, trace)
                if ok and thormeta.object_visible(self.controller.last_event, object_id, geometry.agent_id_from(agent)):
                    return True, ""
        return False, last_error or f"Could not navigate to a visible, unoccupied pose for {object_id}"

    def teleport_to_pose(self, agent: Any, pose: Dict[str, Any], trace: List[Dict[str, Any]]) -> Tuple[bool, str]:
        rotation = pose.get("rotation", 0)
        if isinstance(rotation, dict):
            rotation = rotation.get("y", 0)
        action = {
            "action": "TeleportFull",
            "x": pose.get("x", pose.get("position", {}).get("x")),
            "y": pose.get("y", pose.get("position", {}).get("y", 0.9)),
            "z": pose.get("z", pose.get("position", {}).get("z")),
            "rotation": rotation,
            "horizon": pose.get("horizon", 0),
            # AI2-THOR 5.0 TeleportFull requires ``standing``. Interactable-pose
            # results carry it, but the GetReachablePositions fallback returns bare
            # {x,y,z}; default to standing so the fallback path does not crash with
            # "TeleportFull is missing the following arguments: standing".
            "standing": pose.get("standing", True),
        }
        return self.step(agent, action, trace)

    # ---- pose ranking / multi-agent separation ---------------------------
    def rank_poses_for_agent(self, agent: Any, poses: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        agent_meta = thormeta.metadata_for_agent(self.controller.last_event, geometry.agent_id_from(agent) or 0).get("agent", {})
        agent_pos = agent_meta.get("position") or {}
        if not agent_pos:
            return [dict(pose) for pose in poses]
        return [dict(pose) for pose in sorted(poses, key=lambda pose: geometry.distance_sq(agent_pos, pose))]

    def navigation_pose_score(self, agent: Any, target: Dict[str, Any], pose: Dict[str, Any]) -> float:
        score = geometry.distance_sq(target, pose)
        if self.pose_too_close_to_other_agent(agent, pose):
            score += 1000.0
        return score

    def pose_too_close_to_other_agent(self, agent: Any, pose: Dict[str, Any]) -> bool:
        agent_id = geometry.agent_id_from(agent)
        px = pose.get("x", pose.get("position", {}).get("x"))
        pz = pose.get("z", pose.get("position", {}).get("z"))
        if px is None or pz is None:
            return False
        candidate = {"x": float(px), "z": float(pz)}
        for other_pos in thormeta.other_agent_positions(self.controller.last_event, agent_id):
            if geometry.horizontal_distance_sq(candidate, other_pos) < MIN_AGENT_SEPARATION ** 2:
                return True
        return False
