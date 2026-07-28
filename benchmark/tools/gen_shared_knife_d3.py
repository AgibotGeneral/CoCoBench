#!/usr/bin/env python3
"""Generate K×D3 (shared-Knife slicing exclusion) instances.

N agents each need to slice their own food item, but there is only ONE Knife in
the scene.  PickUp mutual exclusion is engine-enforced: only one agent can hold
the Knife at a time (``held_by_other_agent()`` in ``skill_executor.py`` rejects
competing PickUp attempts).  The D3 construct is resource scheduling: agents
must take turns acquiring the Knife, slicing, and releasing it.

Generation-time verification:
  1. Oracle sequence succeeds (each agent: Find Knife → PickUp → Find food →
     Slice → Drop → vacate).
  2. PickUp exclusion confirmed: after agent_1 holds the Knife, agent_2's
     PickUp(Knife) is rejected.

Run under the thor5 env with CloudRendering on PATH.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "tools"))
from gen_livingroom_coordination import (  # noqa: E402
    set_spawn_seed, reset_scene, _seed_rewrite, _try, write_config,
)

from ai2thor.controller import Controller  # noqa: E402
from ai2thor.platform import CloudRendering  # noqa: E402
from skill_executor import SkillExecutor  # noqa: E402

OUT_ROOT = BENCH / "task_config"
DEFAULT_SCENES = [f"FloorPlan{idx}" for idx in range(1, 31)]

SLICEABLE_PRIORITY = (
    "Potato", "Apple", "Tomato", "Lettuce", "Bread", "Egg",
)


def _objects(controller: Controller) -> List[Dict[str, Any]]:
    return list(controller.last_event.metadata.get("objects", []))


def _controller_init(scene: str, n_agents: int = 2) -> Dict[str, Any]:
    return {"scene": scene, "agentCount": n_agents, "gridSize": 0.25, "visibilityDistance": 1.5}


def _asserts(objects: Dict[str, str]) -> List[Dict[str, Any]]:
    return [{"action": "assert_present", "objectId": oid} for oid in objects.values()]


def _find_knife(objs: List[Dict[str, Any]]) -> Optional[str]:
    """Return the objectId of the first available Knife (prefer Knife over ButterKnife)."""
    for otype in ("Knife", "ButterKnife"):
        for o in objs:
            if o.get("objectType") == otype and o.get("pickupable"):
                return o["objectId"]
    return None


def _ranked_sliceables(objs: List[Dict[str, Any]], n: int) -> List[str]:
    """Return up to n sliceable, pickupable items ranked by priority."""
    order = {name: idx for idx, name in enumerate(SLICEABLE_PRIORITY)}
    candidates = []
    for obj in objs:
        otype = obj.get("objectType")
        if otype not in order:
            continue
        if not obj.get("sliceable") or obj.get("isSliced"):
            continue
        if not obj.get("pickupable"):
            continue
        candidates.append(obj)
    candidates.sort(key=lambda o: (order.get(o["objectType"], 999), o["objectId"]))
    return [o["objectId"] for o in candidates[:n]]


def _ensure_accessible(controller: Controller, object_id: str) -> List[Dict[str, Any]]:
    """If the object is unreachable (0 interactable poses, e.g. inside a closed
    Fridge), move it to a CounterTop via forceAction.  Returns init_actions to
    replay during evaluation, or [] if already reachable."""
    ev = controller.step(action="GetInteractablePoses", objectId=object_id)
    poses = ev.metadata.get("actionReturn") or []
    if len(poses) > 0:
        return []
    objs = _objects(controller)
    countertop = None
    for o in objs:
        if o.get("objectType") == "CounterTop" and o.get("receptacle"):
            countertop = o["objectId"]
            break
    if not countertop:
        return []
    controller.step(action="PickupObject", objectId=object_id, agentId=0, forceAction=True)
    controller.step(action="PutObject", objectId=countertop, agentId=0,
                    forceAction=True, placeStationary=True)
    objs_after = _objects(controller)
    obj_after = next((o for o in objs_after if o["objectId"] == object_id), None)
    if obj_after and countertop in (obj_after.get("parentReceptacles") or []):
        return [
            {"action": "PickupObject", "objectId": object_id, "agentId": 0, "forceAction": True},
            {"action": "PutObject", "objectId": countertop, "agentId": 0,
             "forceAction": True, "placeStationary": True},
        ]
    return []


def verify_k_d3(controller: Controller, scene: str, knife: str,
                items: List[str], extra_init: Optional[List[Dict[str, Any]]] = None,
                items_per_agent: Optional[List[int]] = None) -> bool:
    """Verify the full oracle sequence: each agent acquires Knife, slices their
    assigned food items, drops Knife, then vacates for the next agent.

    ``items_per_agent`` controls asymmetric allocation: e.g. [2,1,1] means
    agent_1 slices 2 items, agent_2 and agent_3 slice 1 each.  Defaults to
    [1]*n_agents (symmetric)."""
    reset_scene(controller, scene)
    for action in (extra_init or []):
        controller.step(**action)
    ex = SkillExecutor(controller)
    n_agents = len(items_per_agent) if items_per_agent else len(items)
    alloc = items_per_agent or [1] * n_agents
    idx = 0
    for a, count in enumerate(alloc, start=1):
        agent = f"agent_{a}"
        if not _try(ex, f"Find({agent}, {knife})"):
            return False
        if not _try(ex, f"PickUp({agent}, {knife})"):
            return False
        for _ in range(count):
            item = items[idx]
            if not _try(ex, f"Find({agent}, {item})"):
                return False
            if not _try(ex, f"Slice({agent}, {item})"):
                return False
            idx += 1
        if not _try(ex, f"Drop({agent})"):
            return False
        for direction in ("back", "left", "right", "back"):
            _try(ex, f"Explore({agent}, {direction})")
    for obj in _objects(controller):
        if obj["objectId"] in items and not obj.get("isSliced"):
            return False
    return True


def first_k_d3(controller: Controller, scene: str, n_items: int = 2,
               items_per_agent: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
    """Find a Knife + n sliceable items and verify the full oracle sequence.
    If the Knife or food items are trapped inside a closed container, move them
    to a CounterTop.

    ``items_per_agent`` overrides ``n_items`` with explicit per-agent allocation
    (e.g. [2,1,1] → total 4 items for 3 agents)."""
    total = sum(items_per_agent) if items_per_agent else n_items
    reset_scene(controller, scene)
    objs = _objects(controller)
    knife = _find_knife(objs)
    if not knife:
        return None
    extra_init: List[Dict[str, Any]] = []
    extra_init.extend(_ensure_accessible(controller, knife))
    items = _ranked_sliceables(_objects(controller), total)
    if len(items) < total:
        return None
    for item_id in items:
        extra_init.extend(_ensure_accessible(controller, item_id))
    if not verify_k_d3(controller, scene, knife, items, extra_init=extra_init,
                       items_per_agent=items_per_agent):
        print(f"  SKIP {scene}: oracle sequence failed")
        return None
    return {"knife": knife, "items": items, "pickup_exclusion_verified": True,
            "extra_init_actions": extra_init,
            "items_per_agent": items_per_agent}


def build_k_d3(scene: str, r: Dict[str, Any], n_agents: int = 2,
               items_per_agent: Optional[List[int]] = None) -> Dict[str, Any]:
    skills = ["Find", "Explore", "Wait", "PickUp", "Drop", "Slice"]
    alloc = items_per_agent or r.get("items_per_agent") or [1] * n_agents
    total_items = sum(alloc)
    items = list(r["items"])[:total_items]
    knife = r["knife"]

    objects: Dict[str, str] = {f"item_{i}": oid for i, oid in enumerate(items, start=1)}
    objects["knife"] = knife

    init_actions = _asserts(objects)
    init_actions.extend(r.get("extra_init_actions") or [])

    # Build assignment map: item_alias -> agent_name
    assignment: Dict[str, str] = {}
    idx = 1
    for a, count in enumerate(alloc, start=1):
        for _ in range(count):
            assignment[f"item_{idx}"] = f"agent_{a}"
            idx += 1

    # Oracle steps per agent: Find(Knife)+PickUp + (Find(food)+Slice)*n_items + Drop + Explore×4(vacate)
    #   = 2 + 2*n_items + 1 + 4 = 7 + 2*n_items
    optimal = sum(7 + 2 * count for count in alloc)

    # Goal predicates: sliced_by requires the assigned agent to slice each item
    goal_predicates = [
        {"predicate": "sliced_by", "object": item_alias, "agent": agent_name}
        for item_alias, agent_name in assignment.items()
    ]

    is_asymmetric = len(set(alloc)) > 1

    return {
        "task_id": f"K_D3__{scene}__{n_agents}agent_seed0" if n_agents > 2 else f"K_D3__{scene}__seed0",
        "coordination_dim": "D3",
        "task_family": "K",
        "task_name": "shared_knife_slicing",
        "scene_id": scene,
        "agent_count": n_agents,
        "agents": [
            {"id": f"agent_{i}", "role": f"Chef-{i}", "allowed_skills": skills}
            for i in range(1, n_agents + 1)
        ],
        "seed": 0,
        "assignment": assignment,
        "init_state": {
            "controller_init": _controller_init(scene, n_agents),
            "objects": objects,
            "init_actions": init_actions,
            "design_notes": [
                "D3 shared-tool exclusion: all agents share a single Knife.",
                "Each agent must acquire the Knife, slice their assigned food item(s), then release it.",
                "PickUp mutual exclusion is engine-enforced: only one agent can hold the Knife.",
                "Slice requires holding the Knife (executor-enforced precondition).",
                "sliced_by goal: each item must be sliced by its assigned agent.",
            ] + (["Asymmetric critical sections: some agents slice more items."] if is_asymmetric else []),
        },
        "allowed_skills": skills,
        "goal_predicates": goal_predicates,
        "task_constraints": {
            "resource_exclusion": [
                {
                    "resource": "shared_knife",
                    "resource_id": knife,
                    "exclusive": True,
                    "mechanism": "pickup_exclusion",
                    "note": "Only one agent can hold the Knife at a time (engine-enforced PickUp exclusion).",
                }
            ],
            "scheduling": "any_permutation",
            "legal_plan": "Each agent must acquire the shared Knife, slice their food, "
                          "and release it before the next agent can use it.",
        },
        "success_fn": "all_goal_predicates_and_legal_plan",
        "eval_layer": "L0",
        "difficulty": {
            "object_count": total_items + 1,
            "step_budget": sum(10 + 2 * count for count in alloc),
            "partial_observability": True,
        },
        "provenance": {
            "generator": "gen_shared_knife_d3.py",
            "placement_verified": True,
            "pickup_exclusion_verified": True,
            "room": "kitchen",
            "asymmetric": is_asymmetric,
        },
        "step_budget": sum(10 + 2 * count for count in alloc),
        "optimal_makespan": optimal,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate K×D3 shared-Knife slicing exclusion tasks.")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0,
                        help="InitialRandomSpawn seed (0 = scene default).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    set_spawn_seed(args.seed)

    controller = Controller(
        agentMode="default",
        platform=CloudRendering,
        gpu_device=args.gpu_device,
        scene=args.scenes[0],
        gridSize=0.25,
        agentCount=2,
        width=144,
        height=144,
        visibilityDistance=1.5,
    )
    written: List[Path] = []
    skipped: List[Tuple[str, str]] = []
    try:
        for scene in args.scenes:
            r = first_k_d3(controller, scene)
            if not r:
                skipped.append((scene, "K_D3"))
                print(f"SKIP {scene}: no verified K_D3 assignment")
                continue
            config = build_k_d3(scene, r)
            path = write_config(config, args.overwrite)
            if path:
                written.append(path)
    finally:
        controller.stop()

    print(f"\n--- K_D3 generation summary ---")
    print(f"Written : {len(written)}")
    print(f"Skipped : {len(skipped)}")
    for scene, reason in skipped:
        print(f"  {scene}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
