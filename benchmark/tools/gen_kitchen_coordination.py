#!/usr/bin/env python3
"""Generate kitchen coordination instances for sparse benchmark cells.

This complements ``gen_f_d3.py``. It resolves object ids from live AI2-THOR
metadata and emits task configs for the existing oracle templates:

* A_D1: independent storage + light switch
* B_D2: ordered slicing/cooking/plating
* C_D4: producer-consumer relay through a transfer receptacle

The generator performs only metadata-level filtering. Oracle validation remains
the quality gate, so generated candidates should be followed by
``tools/validate_instances.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "tools"))
from gen_livingroom_coordination import set_spawn_seed, reset_scene, _seed_rewrite  # noqa: E402

from ai2thor.controller import Controller  # noqa: E402
from ai2thor.platform import CloudRendering  # noqa: E402


OUT_ROOT = BENCH / "task_config"
FOOD_STORAGE_ITEMS = ("potato", "lettuce")


def _objects(controller: Controller) -> List[Dict[str, Any]]:
    return list(controller.last_event.metadata.get("objects", []))


def _first(objs: Iterable[Dict[str, Any]], object_type: str, **flags: Any) -> Optional[str]:
    for obj in objs:
        if obj.get("objectType") != object_type:
            continue
        if all(obj.get(key) == value for key, value in flags.items()):
            return obj.get("objectId")
    return None


def _first_pot_on_burner(objs: Iterable[Dict[str, Any]]) -> Optional[str]:
    for obj in objs:
        if obj.get("objectType") != "Pot":
            continue
        parents = obj.get("parentReceptacles") or []
        if any(parent.startswith("StoveBurner|") for parent in parents):
            return obj.get("objectId")
    return None


def _burner_for_pot(objs: Iterable[Dict[str, Any]], pot_id: str) -> Optional[str]:
    obj = next((item for item in objs if item.get("objectId") == pot_id), None)
    if not obj:
        return None
    for parent in obj.get("parentReceptacles") or []:
        if parent.startswith("StoveBurner|"):
            return parent
    return None


def _require(scene: str, resolved: Dict[str, Optional[str]], keys: Iterable[str]) -> Optional[List[str]]:
    missing = [key for key in keys if not resolved.get(key)]
    if missing:
        print(f"SKIP {scene}: missing {missing}")
        return missing
    return None


def _controller_init(scene: str) -> Dict[str, Any]:
    return {"scene": scene, "agentCount": 2, "gridSize": 0.25, "visibilityDistance": 1.5}


def _asserts(objects: Dict[str, str], keys: Iterable[str]) -> List[Dict[str, Any]]:
    return [{"action": "assert_present", "objectId": objects[key]} for key in keys]


def build_a_d1(scene: str, r: Dict[str, str]) -> Dict[str, Any]:
    skills = ["Find", "Explore", "PickUp", "Put", "ToggleOn", "ToggleOff", "Open", "Close"]
    objects = {key: r[key] for key in ("potato", "lettuce", "fridge", "light")}
    init_actions = _asserts(objects, ("potato", "lettuce", "fridge"))
    init_actions.append({
        "action": "ensure_toggle_state",
        "objectId": objects["light"],
        "isToggled": True,
        "note": "The light starts on; the goal is to turn it off.",
    })
    return {
        "task_id": f"A_D1__{scene}__seed0",
        "coordination_dim": "D1",
        "task_family": "A",
        "task_name": "store_groceries_and_lights_off",
        "scene_id": scene,
        "agent_count": 2,
        "agents": [
            {"id": "agent_1", "role": "Agent-Tidy-A", "allowed_skills": skills},
            {"id": "agent_2", "role": "Agent-Tidy-B", "allowed_skills": skills},
        ],
        "seed": 0,
        "init_state": {
            "controller_init": _controller_init(scene),
            "objects": objects,
            "init_actions": init_actions,
            "design_notes": [
                "D1 has no cross-dependency: food storage and light switching can be assigned independently.",
                "The Fridge is used as the storage receptacle because it is volumetric and reliably accepts multiple food items in validated instances.",
            ],
        },
        "allowed_skills": skills,
        "goal_predicates": [
            {"predicate": "on", "object": "potato", "receptacle": "fridge"},
            {"predicate": "on", "object": "lettuce", "receptacle": "fridge"},
            {"predicate": "toggled", "object": "light", "value": False},
        ],
        "task_constraints": {
            "precedence": [["Open(fridge)", "Put(food, fridge)"]],
            "legal_plan": "The storage line requires Open before Put; the light-switch line is independent.",
        },
        "success_fn": "all_goal_predicates_and_legal_plan",
        "eval_layer": "L0",
        "difficulty": {"object_count": 2, "step_budget": 22, "partial_observability": True},
        "provenance": {"generator": "gen_kitchen_coordination.py"},
    }


def _cooking_objects(r: Dict[str, str]) -> Dict[str, str]:
    return {
        "tomato": r["tomato"],
        "lettuce": r["lettuce"],
        "potato": r["potato"],
        "plate": r["plate"],
        "bowl": r["bowl"],
        "pot": r["pot"],
        "stove_burner": r["stove_burner"],
        "stove_knob": r["stove_knob"],
    }


def build_b_d2(scene: str, r: Dict[str, str]) -> Dict[str, Any]:
    prep = ["Find", "Explore", "PickUp", "Put", "Slice"]
    cook = ["Find", "Explore", "Wait", "PickUp", "Put", "ToggleOn", "ToggleOff"]
    objects = _cooking_objects(r)
    init_actions = _asserts(objects, ("tomato", "lettuce", "potato", "plate", "bowl", "pot"))
    init_actions.append({"action": "ensure_toggle_state", "objectId": objects["stove_knob"], "isToggled": False})
    return {
        "task_id": f"B_D2__{scene}__seed0",
        "coordination_dim": "D2",
        "task_family": "B",
        "task_name": "prepare_salad_and_cooked_potato_plate",
        "scene_id": scene,
        "agent_count": 2,
        "agents": [
            {"id": "agent_1", "role": "Agent-Prep", "allowed_skills": prep},
            {"id": "agent_2", "role": "Agent-CookPlate", "allowed_skills": cook},
        ],
        "seed": 0,
        "init_state": {
            "controller_init": _controller_init(scene),
            "objects": objects,
            "init_actions": init_actions,
            "verified_facts": {
                "candidate_filter": "Generated only when the Pot is already on a StoveBurner in live metadata.",
                "serving_split": "Tomato slices use the Bowl and cooked potato uses the Plate to avoid single-receptacle capacity instability.",
            },
        },
        "allowed_skills": sorted(set(prep + cook)),
        "goal_predicates": [
            {"predicate": "sliced", "object": "tomato"},
            {"predicate": "sliced", "object": "lettuce"},
            {"predicate": "cooked", "object": "potato"},
            {"predicate": "on_sliced_piece", "source": "tomato", "receptacle": "bowl"},
            {"predicate": "on", "object": "potato", "receptacle": "plate"},
        ],
        "task_constraints": {
            "precedence": [
                ["Slice(tomato)", "Put(tomato_sliced, bowl)"],
                ["Put(potato, pot) + ToggleOn(stove_knob)", "cooked(potato)"],
                ["cooked(potato)", "Put(potato, plate)"],
            ],
            "role_constraints": ["agent_1 slices; agent_2 cooks and plates."],
            "legal_plan": "The plan must not use unauthorized skills; slicing and cooking must precede their dependent placements.",
        },
        "success_fn": "all_goal_predicates_and_legal_plan",
        "eval_layer": "L0",
        "difficulty": {"object_count": 6, "step_budget": 45, "partial_observability": True},
        "provenance": {"generator": "gen_kitchen_coordination.py"},
    }


def build_c_d4(scene: str, r: Dict[str, str]) -> Dict[str, Any]:
    source = ["Find", "Explore", "PickUp", "Put", "Drop"]
    target = ["Find", "Explore", "PickUp", "Put", "Open", "Close"]
    foods = [r[k] for k in ("food_1", "food_2", "food_3") if r.get(k)]
    objects: Dict[str, str] = {f"food_{i}": fid for i, fid in enumerate(foods, start=1)}
    objects["transfer_plate"] = r["plate"]
    objects["fridge"] = r["fridge"]
    # Pre-open the Fridge so the relay stays pure PickUp/Put (consumer needs no Open
    # skill); a loaded Fridge cannot reliably re-close in this build, so the goal does
    # not require closing it.
    init_actions = _asserts(objects, tuple(objects.keys()))
    init_actions.append({"action": "OpenObject", "objectId": r["fridge"], "forceAction": True,
                         "note": "Fridge starts open so the consumer can deposit relayed foods."})
    return {
        "task_id": f"C_D4__{scene}__seed0",
        "coordination_dim": "D4",
        "task_family": "C",
        "task_name": "multi_object_relay_via_transfer_point",
        "scene_id": scene,
        "agent_count": 2,
        "agents": [
            {"id": "agent_1", "role": "Agent-Source", "allowed_skills": source},
            {"id": "agent_2", "role": "Agent-Target", "allowed_skills": target},
        ],
        "seed": 0,
        "init_state": {
            "controller_init": _controller_init(scene),
            "objects": objects,
            "init_actions": init_actions,
            "design_notes": [
                f"{len(foods)} foods are relayed source -> transfer Plate (capacity 1) -> Fridge.",
                "Multi-object + capacity-limited buffer creates producer-consumer queue pressure:",
                "a producer that over-deposits overflows the Plate; a consumer that lags starves it.",
                "Agent-Source may only Put on the Plate; Agent-Target may only PickUp on-Plate foods and Put in the Fridge.",
            ],
        },
        "allowed_skills": sorted(set(source + target)),
        "goal_predicates": [{"predicate": "on", "object": f"food_{i}", "receptacle": "fridge"}
                            for i in range(1, len(foods) + 1)],
        "task_constraints": {
            "precedence": [
                ["Source.Put(food, transfer_plate)", "Target.PickUp(food from transfer_plate)"],
                ["Target.PickUp(food)", "Target.Put(food, fridge)"],
            ],
            "role_constraints": ["Source handles source-to-transfer; Target handles transfer-to-fridge."],
            "legal_plan": "Each food must pass through the transfer point before the fridge placement.",
            "buffer_stations": [{"objectId": r["plate"], "capacity": 1}],
            "producer_consumer": {
                "producer": "agent_1",
                "consumer": "agent_2",
                "buffer": r["plate"],
                "targets": [r["fridge"]],
                "note": "Relay enforced at the action menu: producer Put->buffer only; consumer PickUp only on-buffer objects, Put->fridge only. Forces the transfer point (D4).",
            },
        },
        "success_fn": "all_goal_predicates_and_legal_plan",
        "eval_layer": "L0",
        "difficulty": {"object_count": len(foods), "step_budget": 60, "partial_observability": True},
        "provenance": {"generator": "gen_kitchen_coordination.py"},
    }


BUILDERS = {
    "A_D1": build_a_d1,
    "B_D2": build_b_d2,
    "C_D4": build_c_d4,
}


def resolve_scene(controller: Controller, scene: str) -> Dict[str, Optional[str]]:
    reset_scene(controller, scene)
    objs = _objects(controller)
    pot = _first_pot_on_burner(objs) or _first(objs, "Pot", receptacle=True) or _first(objs, "Pot")
    fridge = _first(objs, "Fridge", receptacle=True)
    resolved: Dict[str, Optional[str]] = {
        "apple": _first(objs, "Apple", pickupable=True),
        "bread": _first(objs, "Bread", pickupable=True),
        "lettuce": _first(objs, "Lettuce", pickupable=True),
        "potato": _first(objs, "Potato", pickupable=True),
        "tomato": _first(objs, "Tomato", pickupable=True),
        "fridge": fridge,
        "light": _first(objs, "LightSwitch", toggleable=True),
        "plate": _first(objs, "Plate", receptacle=True),
        "bowl": _first(objs, "Bowl", receptacle=True),
        "pot": pot,
        "stove_burner": _burner_for_pot(objs, pot) if pot else _first(objs, "StoveBurner"),
        "stove_knob": _first(objs, "StoveKnob", toggleable=True),
    }
    # Relay (C_D4) needs ≥3 pickupable foods that start OUTSIDE the fridge (so the
    # producer can fetch them at a source and the relay into the fridge is non-trivial).
    # Prefer Plate-compatible compact foods (Potato/Tomato/Apple) over large/awkward
    # ones (Bread too big for a Plate; Egg has flaky interactable poses).
    food_types = ("Potato", "Tomato", "Apple", "Lettuce", "Bread", "Egg")
    by_type: Dict[str, str] = {}
    for obj in objs:
        if obj.get("objectType") in food_types and obj.get("pickupable"):
            parents = obj.get("parentReceptacles") or []
            if fridge and fridge in parents:
                continue  # already in the fridge -> not a relay source
            by_type.setdefault(obj["objectType"], obj["objectId"])
    source_foods = [by_type[t] for t in food_types if t in by_type]
    for i, oid in enumerate(source_foods[:4], start=1):
        resolved[f"food_{i}"] = oid
    return resolved


def build_cell(scene: str, cell: str, resolved: Dict[str, Optional[str]]) -> Optional[Dict[str, Any]]:
    requirements = {
        "A_D1": ("potato", "lettuce", "fridge", "light"),
        "B_D2": ("tomato", "lettuce", "potato", "plate", "bowl", "pot", "stove_burner", "stove_knob"),
        "C_D4": ("food_1", "food_2", "food_3", "plate", "fridge"),
    }[cell]
    if _require(scene, resolved, requirements):
        return None
    if cell == "B_D2":
        pot_id = resolved.get("pot") or ""
        burner_id = resolved.get("stove_burner") or ""
        if not burner_id or not burner_id.startswith("StoveBurner|"):
            print(f"SKIP {scene}: {cell} requires Pot on StoveBurner")
            return None
        if not pot_id:
            return None
    concrete = {key: value for key, value in resolved.items() if value}
    return BUILDERS[cell](scene, concrete)  # type: ignore[arg-type]


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
    parser = argparse.ArgumentParser(description="Generate kitchen coordination task configs.")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--cells", nargs="+", default=sorted(BUILDERS), choices=sorted(BUILDERS))
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="InitialRandomSpawn seed (0=scene default).")
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
    try:
        for scene in args.scenes:
            resolved = resolve_scene(controller, scene)
            for cell in args.cells:
                config = build_cell(scene, cell, resolved)
                if config is None:
                    continue
                path = write_config(config, overwrite=args.overwrite)
                if path is not None:
                    written.append(path)
    finally:
        controller.stop()
    print(f"generated {len(written)} configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
