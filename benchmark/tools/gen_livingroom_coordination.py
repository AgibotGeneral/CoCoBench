#!/usr/bin/env python3
"""Generate living-room coordination instances with placement verification.

This expands the benchmark beyond kitchen-only semantics:

* G_D1: independent parallel sorting in living rooms.
* C_D4: relay transport through a living-room transfer surface.
* E_D3: competitive collection with two ownership baskets.

Each candidate is checked by driving the same SkillExecutor used by validation.
The script writes only instances whose representative oracle sequence succeeds.
Full oracle validation should still be run afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))

from ai2thor.controller import Controller  # noqa: E402
from ai2thor.platform import CloudRendering  # noqa: E402
from skill_executor import SkillExecutor  # noqa: E402


OUT_ROOT = BENCH / "task_config"
DEFAULT_SCENES = [f"FloorPlan{idx}" for idx in range(201, 231)]

# Depth axis (Option A): per-seed layout randomization via InitialRandomSpawn. When
# _SPAWN_SEED > 0, every controller.reset re-applies the SAME InitialRandomSpawn(seed)
# so the resolved (position-encoded) objectIds stay consistent across the many resets
# the verifiers perform; the spawn is recorded as an init_action so the runner
# reproduces the identical layout at eval time (probe-confirmed deterministic &
# cross-process reproducible in thor5 multi-agent).
_SPAWN_SEED = 0


def set_spawn_seed(seed: int) -> None:
    global _SPAWN_SEED
    _SPAWN_SEED = int(seed)


def _spawn_action() -> List[Dict[str, Any]]:
    if _SPAWN_SEED <= 0:
        return []
    return [{"action": "InitialRandomSpawn", "randomSeed": _SPAWN_SEED,
             "forceVisible": False, "numPlacementAttempts": 5, "placeStationary": True}]


def reset_scene(controller: "Controller", scene: str) -> None:
    """Reset the scene and (when seeded) re-apply the layout randomization so every
    verifier reset sees the SAME spawned layout the config will record."""
    controller.reset(scene=scene)
    if _SPAWN_SEED > 0:
        controller.step(action="InitialRandomSpawn", randomSeed=_SPAWN_SEED,
                        forceVisible=False, numPlacementAttempts=5, placeStationary=True)


def _seed_rewrite(config: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp the active spawn seed onto a built config: seed field, task_id suffix,
    and a leading InitialRandomSpawn init_action (before assert_present, so the
    asserts match the spawned layout)."""
    if _SPAWN_SEED <= 0:
        return config
    config["seed"] = _SPAWN_SEED
    config["task_id"] = config["task_id"].replace("seed0", f"seed{_SPAWN_SEED}")
    ia = config["init_state"].get("init_actions", [])
    config["init_state"]["init_actions"] = _spawn_action() + ia
    config.setdefault("provenance", {})["spawn_seed"] = _SPAWN_SEED
    return config


PICKUP_PRIORITY = (
    "KeyChain",
    "Watch",
    "CreditCard",
    "RemoteControl",
    "CellPhone",
    "Pen",
    "Pencil",
    "Book",
    "Newspaper",
    "Laptop",
)
OPEN_SURFACE_PRIORITY = (
    "Sofa",
    "CoffeeTable",
    "SideTable",
    "DiningTable",
    "Desk",
    "ArmChair",
    "Ottoman",
    "TVStand",
    "Dresser",
    "Shelf",
    # Bedroom-native surfaces appended last so living-room ranking is unchanged.
    "Bed",
    "Chair",
)


def _room(scene: str) -> str:
    """Infer iTHOR room type from the floorplan index (cosmetic labeling only)."""
    digits = "".join(ch for ch in scene if ch.isdigit())
    n = int(digits) if digits else 0
    if 1 <= n <= 30:
        return "kitchen"
    if 201 <= n <= 230:
        return "living_room"
    if 301 <= n <= 330:
        return "bedroom"
    if 401 <= n <= 430:
        return "bathroom"
    return "other"
E_BASKET2_PRIORITY = ("Box",) + OPEN_SURFACE_PRIORITY
OPENABLE_CONTAINER_PRIORITY = ("Box", "Safe", "Drawer", "Cabinet")
# Bedroom family H uses storage *furniture* (not Box) so its container-loading cells
# are visibly distinct from the C family's Box-based instances.
FURNITURE_CONTAINER_PRIORITY = ("Drawer", "Cabinet", "Dresser", "Safe")


def _objects(controller: Controller) -> List[Dict[str, Any]]:
    return list(controller.last_event.metadata.get("objects", []))


def _controller_init(scene: str) -> Dict[str, Any]:
    return {"scene": scene, "agentCount": 2, "gridSize": 0.25, "visibilityDistance": 1.5}


def _asserts(objects: Dict[str, str], keys: Iterable[str]) -> List[Dict[str, Any]]:
    return [{"action": "assert_present", "objectId": objects[key]} for key in keys]


def _ranked(objs: Iterable[Dict[str, Any]], type_order: Sequence[str], *, pickupable: Optional[bool] = None,
            receptacle: Optional[bool] = None, allow_openable: bool = True) -> List[Dict[str, Any]]:
    order = {name: idx for idx, name in enumerate(type_order)}
    out: List[Dict[str, Any]] = []
    for obj in objs:
        object_type = obj.get("objectType")
        if object_type not in order:
            continue
        if pickupable is not None and bool(obj.get("pickupable")) != pickupable:
            continue
        if receptacle is not None and bool(obj.get("receptacle")) != receptacle:
            continue
        if not allow_openable and bool(obj.get("openable")):
            continue
        out.append(obj)
    return sorted(out, key=lambda item: (order[item.get("objectType")], item.get("objectId", "")))


def _try(executor: SkillExecutor, call: str) -> bool:
    return executor.execute_call(call).success


def _open_if_needed(executor: SkillExecutor, agent: str, object_id: str, openable: bool) -> bool:
    if not openable:
        return True
    _try(executor, f"Find({agent}, {object_id})")
    return _try(executor, f"Open({agent}, {object_id})")


def _put_sequence(executor: SkillExecutor, agent: str, item: str, target: str, *, open_target: bool = False) -> bool:
    return (
        _try(executor, f"Find({agent}, {item})")
        and _try(executor, f"PickUp({agent}, {item})")
        and _open_if_needed(executor, agent, target, open_target)
        and _try(executor, f"Find({agent}, {target})")
        and _try(executor, f"Put({agent}, {target})")
    )


def _target_openable(objs: Sequence[Dict[str, Any]], object_id: str) -> bool:
    for obj in objs:
        if obj.get("objectId") == object_id:
            return bool(obj.get("openable"))
    return False


def _candidate_base(controller: Controller, scene: str) -> Dict[str, List[Dict[str, Any]]]:
    reset_scene(controller, scene)
    objs = _objects(controller)
    return {
        "items": _ranked(objs, PICKUP_PRIORITY, pickupable=True),
        "open_surfaces": _ranked(objs, OPEN_SURFACE_PRIORITY, receptacle=True, allow_openable=False),
        "e_baskets": _ranked(objs, E_BASKET2_PRIORITY, receptacle=True),
        "containers": [o for o in _ranked(objs, OPENABLE_CONTAINER_PRIORITY, receptacle=True) if o.get("openable")],
        "furniture_containers": [o for o in _ranked(objs, FURNITURE_CONTAINER_PRIORITY, receptacle=True) if o.get("openable")],
        "objects": objs,
    }


def verify_g_d1(controller: Controller, scene: str, item_1: str, item_2: str, target_1: str, target_2: str) -> bool:
    reset_scene(controller, scene)
    executor = SkillExecutor(controller)
    return (
        _put_sequence(executor, "agent_1", item_1, target_1)
        and _put_sequence(executor, "agent_2", item_2, target_2)
    )


def verify_c_d4(controller: Controller, scene: str, items: Sequence[str], transfer: str, target: str) -> bool:
    """In-process relay check: each item goes source -> transfer surface -> target,
    producer/consumer alternating and vacating (so the occupancy filter does not
    block the other agent). All items must complete for the instance to be kept."""
    reset_scene(controller, scene)
    executor = SkillExecutor(controller)
    for item in items:
        if not _put_sequence(executor, "agent_1", item, transfer):
            return False
        for direction in ("back", "left", "right", "back"):
            _try(executor, f"Explore(agent_1, {direction})")
        if not _put_sequence(executor, "agent_2", item, target):
            return False
        for direction in ("back", "left", "right", "back"):
            _try(executor, f"Explore(agent_2, {direction})")
    return True


def verify_e_d3(controller: Controller, scene: str, item_1: str, item_2: str, basket_1: str, basket_2: str,
                *, open_basket_2: bool) -> bool:
    reset_scene(controller, scene)
    executor = SkillExecutor(controller)
    return (
        _put_sequence(executor, "agent_1", item_1, basket_1)
        and _put_sequence(executor, "agent_2", item_2, basket_2, open_target=open_basket_2)
    )


def first_g_d1(controller: Controller, scene: str, base: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, str]]:
    items = base["items"][:8]
    targets = base["open_surfaces"][:8]
    for first_idx, first in enumerate(items):
        for second in items[first_idx + 1:]:
            for target_1 in targets:
                for target_2 in targets:
                    if target_1["objectId"] == target_2["objectId"]:
                        continue
                    if verify_g_d1(controller, scene, first["objectId"], second["objectId"], target_1["objectId"], target_2["objectId"]):
                        return {
                            "item_1": first["objectId"],
                            "item_2": second["objectId"],
                            "target_1": target_1["objectId"],
                            "target_2": target_2["objectId"],
                        }
    return None


def first_c_d4(controller: Controller, scene: str, base: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Find 3 personal items + a transfer surface + a target surface such that the
    full multi-object relay verifies. Items must not start on the transfer/target
    (else the relay would be trivial)."""
    items = base["items"][:10]
    surfaces = base["open_surfaces"][:8]
    for transfer in surfaces:
        for target in surfaces:
            if transfer["objectId"] == target["objectId"]:
                continue
            reserved = {transfer["objectId"], target["objectId"]}
            chosen: List[str] = []
            for item in items:
                if len(chosen) >= 3:
                    break
                parents = set(item.get("parentReceptacles") or [])
                if parents & reserved:
                    continue  # already on the transfer/target -> trivial relay
                if verify_c_d4(controller, scene, [item["objectId"]], transfer["objectId"], target["objectId"]):
                    chosen.append(item["objectId"])
            if len(chosen) == 3 and verify_c_d4(controller, scene, chosen, transfer["objectId"], target["objectId"]):
                return {"items": chosen, "transfer": transfer["objectId"], "target": target["objectId"]}
    return None


def first_e_d3(controller: Controller, scene: str, base: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    items = base["items"][:8]
    basket_1_candidates = base["open_surfaces"][:8]
    basket_2_candidates = base["e_baskets"][:10]
    for first_idx, first in enumerate(items):
        for second in items[first_idx + 1:]:
            for basket_1 in basket_1_candidates:
                for basket_2 in basket_2_candidates:
                    if basket_1["objectId"] == basket_2["objectId"]:
                        continue
                    open_basket_2 = _target_openable(base["objects"], basket_2["objectId"])
                    if verify_e_d3(
                        controller,
                        scene,
                        first["objectId"],
                        second["objectId"],
                        basket_1["objectId"],
                        basket_2["objectId"],
                        open_basket_2=open_basket_2,
                    ):
                        return {
                            "item_1": first["objectId"],
                            "item_2": second["objectId"],
                            "basket_1": basket_1["objectId"],
                            "basket_2": basket_2["objectId"],
                            "open_basket_2": open_basket_2,
                        }
    return None


def build_g_d1(scene: str, r: Dict[str, str]) -> Dict[str, Any]:
    skills = ["Find", "Explore", "PickUp", "Put"]
    objects = {key: r[key] for key in ("item_1", "item_2", "target_1", "target_2")}
    room = _room(scene)
    return {
        "task_id": f"G_D1__{scene}__seed0",
        "coordination_dim": "D1",
        "task_family": "G",
        "task_name": f"{room}_parallel_sorting",
        "scene_id": scene,
        "agent_count": 2,
        "agents": [
            {"id": "agent_1", "role": "Sorter-A", "allowed_skills": skills},
            {"id": "agent_2", "role": "Sorter-B", "allowed_skills": skills},
        ],
        "seed": 0,
        "init_state": {
            "controller_init": _controller_init(scene),
            "objects": objects,
            "init_actions": _asserts(objects, objects.keys()),
            "design_notes": [
                "D1 has no cross-agent ordering dependency: each agent moves one personal item to a distinct receptacle surface.",
                "Generated instances use only non-openable receptacles so the task stays focused on independent allocation rather than precondition handling.",
            ],
        },
        "allowed_skills": skills,
        "goal_predicates": [
            {"predicate": "on", "object": "item_1", "receptacle": "target_1"},
            {"predicate": "on", "object": "item_2", "receptacle": "target_2"},
        ],
        "task_constraints": {
            "independent_subtasks": [
                ["PickUp(item_1)", "Put(item_1, target_1)"],
                ["PickUp(item_2)", "Put(item_2, target_2)"],
            ],
            "legal_plan": "The two placement lines are independent and may be executed in either order or in parallel.",
        },
        "success_fn": "all_goal_predicates_and_legal_plan",
        "eval_layer": "L0",
        "difficulty": {"object_count": 2, "step_budget": 20, "partial_observability": True},
        "provenance": {"generator": "gen_livingroom_coordination.py", "placement_verified": True, "room": room},
    }


def build_c_d4(scene: str, r: Dict[str, Any]) -> Dict[str, Any]:
    source = ["Find", "Explore", "PickUp", "Put", "Drop"]
    target_skills = ["Find", "Explore", "PickUp", "Put"]
    items = list(r["items"])
    objects: Dict[str, str] = {f"item_{i}": oid for i, oid in enumerate(items, start=1)}
    objects["transfer"] = r["transfer"]
    objects["target"] = r["target"]
    return {
        "task_id": f"C_D4__{scene}__livingroom_seed0",
        "coordination_dim": "D4",
        "task_family": "C",
        "task_name": "living_room_multi_object_relay",
        "scene_id": scene,
        "agent_count": 2,
        "agents": [
            {"id": "agent_1", "role": "Agent-Source", "allowed_skills": source},
            {"id": "agent_2", "role": "Agent-Target", "allowed_skills": target_skills},
        ],
        "seed": 0,
        "init_state": {
            "controller_init": _controller_init(scene),
            "objects": objects,
            "init_actions": _asserts(objects, objects.keys()),
            "design_notes": [
                f"{len(items)} personal items are relayed source -> transfer surface (capacity 1) -> target surface.",
                "Multi-object + capacity-limited buffer creates producer-consumer queue pressure (overflow/starvation).",
                "Agent-Source may only Put on the transfer surface; Agent-Target may only PickUp on-transfer items and Put on the target.",
            ],
        },
        "allowed_skills": sorted(set(source + target_skills)),
        "goal_predicates": [{"predicate": "on", "object": f"item_{i}", "receptacle": "target"}
                            for i in range(1, len(items) + 1)],
        "task_constraints": {
            "precedence": [
                ["Source.Put(item, transfer)", "Target.PickUp(item from transfer)"],
                ["Target.PickUp(item)", "Target.Put(item, target)"],
            ],
            "role_constraints": ["Source handles source-to-transfer; Target handles transfer-to-target."],
            "legal_plan": "Each item must pass through the transfer surface before final placement.",
            "buffer_stations": [{"objectId": r["transfer"], "capacity": 1}],
            "producer_consumer": {
                "producer": "agent_1",
                "consumer": "agent_2",
                "buffer": r["transfer"],
                "targets": [r["target"]],
                "note": "Relay enforced at the action menu: producer Put->buffer only; consumer PickUp only on-buffer items, Put->target only. Forces the transfer point (D4).",
            },
        },
        "success_fn": "all_goal_predicates_and_legal_plan",
        "eval_layer": "L0",
        "difficulty": {"object_count": len(items), "step_budget": 60, "partial_observability": True},
        "provenance": {"generator": "gen_livingroom_coordination.py", "placement_verified": True},
    }


def build_e_d3(scene: str, r: Dict[str, Any]) -> Dict[str, Any]:
    skills = ["Find", "Explore", "PickUp", "Put", "Open", "Close"]
    objects = {key: r[key] for key in ("item_1", "item_2", "basket_1", "basket_2")}
    open_basket_2 = bool(r["open_basket_2"])
    precedence = [["Open(basket_2)", "Put(item_2, basket_2)"]] if open_basket_2 else []
    return {
        "task_id": f"E_D3__{scene}__seed0",
        "coordination_dim": "D3",
        "task_family": "E",
        "task_name": "competitive_collection_two_baskets",
        "scene_id": scene,
        "agent_count": 2,
        "agents": [
            {"id": "agent_1", "role": "Collector-A", "allowed_skills": skills},
            {"id": "agent_2", "role": "Collector-B", "allowed_skills": skills},
        ],
        "seed": 0,
        "init_state": {
            "controller_init": _controller_init(scene),
            "objects": objects,
            "init_actions": _asserts(objects, objects.keys()),
            "design_notes": [
                "Each collector owns a basket or surface; the L0 assignment gives one target object to each agent.",
                "The D3 pressure comes from exclusive access to scattered personal objects and local navigation around shared furniture.",
            ],
        },
        "allowed_skills": skills,
        "goal_predicates": [
            {"predicate": "on", "object": "item_1", "receptacle": "basket_1"},
            {"predicate": "on", "object": "item_2", "receptacle": "basket_2"},
        ],
        "task_constraints": {
            "resource_exclusion": [
                {"resource": "contested_object", "exclusive": True, "note": "Only one agent can hold a pickupable object at a time."}
            ],
            "precedence": precedence,
            "open_basket_2": open_basket_2,
            "scoring": "score(agent) = number of target objects in that agent's basket; higher score wins.",
            "legal_plan": "The plan must not use unauthorized skills; open basket_2 before placement when basket_2 is openable.",
        },
        "success_fn": "relative_score_and_legal_plan (L0 feasibility check uses all_goal_predicates)",
        "eval_layer": "L0",
        "difficulty": {"object_count": 2, "step_budget": 28, "partial_observability": True},
        "provenance": {"generator": "gen_livingroom_coordination.py", "placement_verified": True, "room": _room(scene)},
    }


def _force_close(controller: Controller, container: str) -> None:
    """Put the container into a closed start state (so Open is genuinely required)."""
    controller.step({"action": "CloseObject", "objectId": container, "forceAction": True})


def _all_in(controller: Controller, items: Sequence[str], container: str) -> bool:
    byid = {o["objectId"]: o for o in _objects(controller)}
    return all(container in ((byid.get(it) or {}).get("parentReceptacles") or []) for it in items)


def verify_c_d2(controller: Controller, scene: str, items: Sequence[str], container: str) -> bool:
    """Ordered load: agent_1 opens + loads item 0, agent_2 loads the rest then closes.
    Kept only if every item is inside AND the container ends closed."""
    reset_scene(controller, scene)
    _force_close(controller, container)
    ex = SkillExecutor(controller)
    if not (_try(ex, f"Find(agent_1, {container})") and _try(ex, f"Open(agent_1, {container})")):
        return False
    if not _put_sequence(ex, "agent_1", items[0], container):
        return False
    for direction in ("back", "left"):
        _try(ex, f"Explore(agent_1, {direction})")
    for item in items[1:]:
        if not _put_sequence(ex, "agent_2", item, container):
            return False
    _try(ex, f"Find(agent_2, {container})")
    if not _try(ex, f"Close(agent_2, {container})"):
        return False
    if not _all_in(controller, items, container):
        return False
    obj = next((o for o in _objects(controller) if o["objectId"] == container), None)
    return bool(obj) and not obj.get("isOpen")


def _container_items(base: Dict[str, List[Dict[str, Any]]], container_id: str, n: int = 2) -> List[str]:
    """Up to n pickupable items that are not already inside the container."""
    out: List[str] = []
    for it in base["items"][:10]:
        if container_id in set(it.get("parentReceptacles") or []):
            continue
        out.append(it["objectId"])
        if len(out) >= n:
            break
    return out


def first_c_d2(controller: Controller, scene: str, base: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    for container in base["containers"][:6]:
        cid = container["objectId"]
        items = _container_items(base, cid)
        if len(items) < 2:
            continue
        if verify_c_d2(controller, scene, items, cid):
            return {"container": cid, "items": items}
    return None


def build_c_d2(scene: str, r: Dict[str, Any]) -> Dict[str, Any]:
    skills = ["Find", "Explore", "Wait", "PickUp", "Put", "Drop", "Open", "Close"]
    items = list(r["items"])
    container = r["container"]
    objects: Dict[str, str] = {f"item_{i}": oid for i, oid in enumerate(items, start=1)}
    objects["container"] = container
    return {
        "task_id": f"C_D2__{scene}__livingroom_seed0",
        "coordination_dim": "D2",
        "task_family": "C",
        "task_name": "ordered_container_loading",
        "scene_id": scene,
        "agent_count": 2,
        "agents": [
            {"id": "agent_1", "role": "Agent-A", "allowed_skills": skills},
            {"id": "agent_2", "role": "Agent-B", "allowed_skills": skills},
        ],
        "seed": 0,
        "init_state": {
            "controller_init": _controller_init(scene),
            "objects": objects,
            "init_actions": _asserts(objects, objects.keys()) + [
                {"action": "CloseObject", "objectId": container, "forceAction": True}],
            "design_notes": [
                "Engine-enforced precedence: a closed container rejects Put, so Open must precede every Put; closing before all items are in would block the remaining Puts.",
                "agent_1 opens + loads the first item; agent_2 loads the rest then closes -> cross-agent ordering.",
            ],
        },
        "allowed_skills": skills,
        "goal_predicates": [{"predicate": "on", "object": f"item_{i}", "receptacle": "container"}
                            for i in range(1, len(items) + 1)]
        + [{"predicate": "closed", "object": "container"}],
        "task_constraints": {
            "precedence": [
                ["Open(container)", "Put(item, container)"],
                ["Put(item, container)", "Close(container)"],
            ],
            "role_constraints": ["Both agents share all skills; the ordering open -> load(all) -> close is the coordination signal."],
            "legal_plan": "Open before any placement; close only after all items are inside.",
        },
        "success_fn": "all_goal_predicates_and_legal_plan",
        "eval_layer": "L0",
        "difficulty": {"object_count": len(items), "step_budget": 50, "partial_observability": True},
        "provenance": {"generator": "gen_livingroom_coordination.py", "placement_verified": True},
    }


def verify_h_d1(controller: Controller, scene: str, item_1: str, item_2: str,
                drawer_1: str, drawer_2: str) -> bool:
    """Independent personal-drawer storage: each agent opens its OWN (closed) drawer
    and loads one item. Both items must end in their respective drawers."""
    reset_scene(controller, scene)
    _force_close(controller, drawer_1)
    _force_close(controller, drawer_2)
    ex = SkillExecutor(controller)
    if not (_try(ex, f"Find(agent_1, {drawer_1})") and _try(ex, f"Open(agent_1, {drawer_1})")):
        return False
    if not _put_sequence(ex, "agent_1", item_1, drawer_1):
        return False
    for direction in ("back", "left", "right", "back"):
        _try(ex, f"Explore(agent_1, {direction})")
    if not (_try(ex, f"Find(agent_2, {drawer_2})") and _try(ex, f"Open(agent_2, {drawer_2})")):
        return False
    if not _put_sequence(ex, "agent_2", item_2, drawer_2):
        return False
    return _all_in(controller, [item_1], drawer_1) and _all_in(controller, [item_2], drawer_2)


def first_h_d1(controller: Controller, scene: str, base: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    drawers = base["furniture_containers"][:6]
    if len(drawers) < 2:
        return None
    items = base["items"][:8]
    for i, d1 in enumerate(drawers):
        for d2 in drawers[i + 1:]:
            reserved = {d1["objectId"], d2["objectId"]}
            chosen = [it["objectId"] for it in items
                      if not (set(it.get("parentReceptacles") or []) & reserved)][:2]
            if len(chosen) < 2:
                continue
            if verify_h_d1(controller, scene, chosen[0], chosen[1], d1["objectId"], d2["objectId"]):
                return {"item_1": chosen[0], "item_2": chosen[1],
                        "drawer_1": d1["objectId"], "drawer_2": d2["objectId"]}
    return None


def build_h_d1(scene: str, r: Dict[str, Any]) -> Dict[str, Any]:
    skills = ["Find", "Explore", "PickUp", "Put", "Open", "Close"]
    objects = {key: r[key] for key in ("item_1", "item_2", "drawer_1", "drawer_2")}
    room = _room(scene)
    return {
        "task_id": f"H_D1__{scene}__seed0",
        "coordination_dim": "D1",
        "task_family": "H",
        "task_name": f"{room}_personal_drawer_storage",
        "scene_id": scene,
        "agent_count": 2,
        "agents": [
            {"id": "agent_1", "role": "Tidier-A", "allowed_skills": skills},
            {"id": "agent_2", "role": "Tidier-B", "allowed_skills": skills},
        ],
        "seed": 0,
        "init_state": {
            "controller_init": _controller_init(scene),
            "objects": objects,
            "init_actions": _asserts(objects, objects.keys()) + [
                {"action": "CloseObject", "objectId": r["drawer_1"], "forceAction": True},
                {"action": "CloseObject", "objectId": r["drawer_2"], "forceAction": True}],
            "design_notes": [
                "D1 independent parallelism: each agent owns one drawer and stores one personal item into it; the two lines share no object and impose no cross-agent ordering.",
                "The open-before-put is a LOCAL precondition inside each agent's own line, not a coordination dependency; action counts are symmetric (6 per agent), so the balanced schedule is the makespan optimum and a single-agent solution doubles it.",
            ],
        },
        "allowed_skills": skills,
        "goal_predicates": [
            {"predicate": "on", "object": "item_1", "receptacle": "drawer_1"},
            {"predicate": "on", "object": "item_2", "receptacle": "drawer_2"},
        ],
        "task_constraints": {
            "independent_subtasks": [
                ["Open(drawer_1)", "PickUp(item_1)", "Put(item_1, drawer_1)"],
                ["Open(drawer_2)", "PickUp(item_2)", "Put(item_2, drawer_2)"],
            ],
            "legal_plan": "Each agent opens its own drawer and stores its own item; the two lines are independent and may run in parallel.",
        },
        "success_fn": "all_goal_predicates_and_legal_plan",
        "eval_layer": "L0",
        "difficulty": {"object_count": 2, "step_budget": 24, "partial_observability": True},
        "provenance": {"generator": "gen_livingroom_coordination.py", "placement_verified": True, "room": room},
    }


def first_h_d2(controller: Controller, scene: str, base: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Ordered loading into a bedroom storage drawer (same engine-enforced precedence
    as C_D2, but a furniture container rather than a Box)."""
    for container in base["furniture_containers"][:6]:
        cid = container["objectId"]
        items = _container_items(base, cid)
        if len(items) < 2:
            continue
        if verify_c_d2(controller, scene, items, cid):
            return {"container": cid, "items": items}
    return None


def build_h_d2(scene: str, r: Dict[str, Any]) -> Dict[str, Any]:
    skills = ["Find", "Explore", "Wait", "PickUp", "Put", "Drop", "Open", "Close"]
    items = list(r["items"])
    container = r["container"]
    objects: Dict[str, str] = {f"item_{i}": oid for i, oid in enumerate(items, start=1)}
    objects["container"] = container
    room = _room(scene)
    return {
        "task_id": f"H_D2__{scene}__seed0",
        "coordination_dim": "D2",
        "task_family": "H",
        "task_name": f"{room}_ordered_drawer_loading",
        "scene_id": scene,
        "agent_count": 2,
        "agents": [
            {"id": "agent_1", "role": "Agent-A", "allowed_skills": skills},
            {"id": "agent_2", "role": "Agent-B", "allowed_skills": skills},
        ],
        "seed": 0,
        "init_state": {
            "controller_init": _controller_init(scene),
            "objects": objects,
            "init_actions": _asserts(objects, objects.keys()) + [
                {"action": "CloseObject", "objectId": container, "forceAction": True}],
            "design_notes": [
                "Engine-enforced precedence: a closed drawer rejects Put, so Open must precede every Put; closing before all items are in would block the remaining Puts.",
                "agent_1 opens + loads the first item; agent_2 loads the rest then closes -> cross-agent ordering. Bedroom storage furniture (Drawer/Cabinet) instead of the C family's Box.",
            ],
        },
        "allowed_skills": skills,
        "goal_predicates": [{"predicate": "on", "object": f"item_{i}", "receptacle": "container"}
                            for i in range(1, len(items) + 1)]
        + [{"predicate": "closed", "object": "container"}],
        "task_constraints": {
            "precedence": [
                ["Open(container)", "Put(item, container)"],
                ["Put(item, container)", "Close(container)"],
            ],
            "role_constraints": ["Both agents share all skills; the ordering open -> load(all) -> close is the coordination signal."],
            "legal_plan": "Open before any placement; close only after all items are inside.",
        },
        "success_fn": "all_goal_predicates_and_legal_plan",
        "eval_layer": "L0",
        "difficulty": {"object_count": len(items), "step_budget": 50, "partial_observability": True},
        "provenance": {"generator": "gen_livingroom_coordination.py", "placement_verified": True, "room": room},
    }


BUILDERS = {
    "G_D1": (first_g_d1, build_g_d1),
    "C_D2": (first_c_d2, build_c_d2),
    "C_D4": (first_c_d4, build_c_d4),
    "E_D3": (first_e_d3, build_e_d3),
    "H_D1": (first_h_d1, build_h_d1),
    "H_D2": (first_h_d2, build_h_d2),
}


def write_config(config: Dict[str, Any], overwrite: bool) -> Optional[Path]:
    config = _seed_rewrite(config)
    cell_dir = OUT_ROOT / config["task_family"] / config["coordination_dim"]
    path = cell_dir / f"{config['task_id']}.json"
    if path.exists() and not overwrite:
        print(f"KEEP {path.relative_to(BENCH)} (exists; pass --overwrite to replace)")
        return None
    cell_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {path.relative_to(BENCH)}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate living-room coordination task configs.")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--cells", nargs="+", default=sorted(BUILDERS), choices=sorted(BUILDERS))
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="InitialRandomSpawn seed for depth-axis layout randomization (0 = scene default).")
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
            base = _candidate_base(controller, scene)
            if len(base["items"]) < 2 or len(base["open_surfaces"]) < 2:
                skipped.append((scene, "missing items or surfaces"))
                print(f"SKIP {scene}: missing items or surfaces")
                continue
            for cell in args.cells:
                resolver, builder = BUILDERS[cell]
                resolved = resolver(controller, scene, base)
                if not resolved:
                    skipped.append((scene, cell))
                    print(f"SKIP {scene}: no verified {cell} assignment")
                    continue
                config = builder(scene, resolved)
                path = write_config(config, overwrite=args.overwrite)
                if path is not None:
                    written.append(path)
    finally:
        controller.stop()
    print(f"generated {len(written)} configs; skipped={len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
