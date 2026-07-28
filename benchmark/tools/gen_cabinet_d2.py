#!/usr/bin/env python3
"""Generate family-A D2 instances: kitchen Cabinet/Drawer ordered loading.

A third, non-C D2 carrier (alongside C=living-room Box and H=bedroom Drawer), in the
kitchen room. The construct is identical to C_D2 (engine-enforced open -> load(all) ->
close precedence), so this REUSES the family-agnostic oracle ``plan_c_d2`` (registered
as ``A_D2``) and the proven container-loading resolver/verifier from
gen_livingroom_coordination. Only the family label and kitchen room/container differ.

Purpose: dilute the C family's concentration while keeping D2 balanced and giving D2
a kitchen-native carrier. Run ``validate_instances.py --only-cell A_D2`` afterwards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "tools"))

from ai2thor.controller import Controller  # noqa: E402
from ai2thor.platform import CloudRendering  # noqa: E402

from gen_livingroom_coordination import (  # noqa: E402
    _objects,
    _ranked,
    _controller_init,
    _asserts,
    _room,
    first_c_d2,
    write_config,
    reset_scene,
    set_spawn_seed,
)

DEFAULT_SCENES = [f"FloorPlan{idx}" for idx in range(1, 31)]

# Kitchen-native pickupables (the living-room PICKUP_PRIORITY is absent in kitchens,
# which is why a generic base yields no items here) and openable storage containers.
KITCHEN_ITEM_PRIORITY = (
    "Apple", "Tomato", "Potato", "Lettuce", "Egg", "Bread", "Mug", "Cup",
    "Fork", "Knife", "Spoon", "ButterKnife", "Ladle", "Spatula", "DishSponge",
    "SaltShaker", "PepperShaker", "SoapBottle", "PaperTowelRoll", "Pan", "Pot",
)
KITCHEN_CONTAINER_PRIORITY = ("Cabinet", "Drawer", "Microwave", "Fridge")


def _kitchen_base(controller: Controller, scene: str) -> Dict[str, Any]:
    """Kitchen candidate base for ordered loading: food/utensil pickupables + openable
    storage containers (Cabinet/Drawer/Microwave/Fridge). Mirrors the keys first_c_d2
    reads (``items``/``containers``); verify_c_d2 drives the executor so any container
    that physically rejects the items is filtered out."""
    reset_scene(controller, scene)
    objs = _objects(controller)
    return {
        "items": _ranked(objs, KITCHEN_ITEM_PRIORITY, pickupable=True),
        "containers": [o for o in _ranked(objs, KITCHEN_CONTAINER_PRIORITY, receptacle=True) if o.get("openable")],
        "objects": objs,
    }



def build_a_d2(scene: str, r: Dict[str, Any]) -> Dict[str, Any]:
    """Same ordered-loading schema as C_D2 (open -> load(all) -> close), family A,
    kitchen Cabinet/Drawer container. Alias-driven: drives plan_c_d2 unchanged."""
    skills = ["Find", "Explore", "Wait", "PickUp", "Put", "Drop", "Open", "Close"]
    items = list(r["items"])
    container = r["container"]
    objects: Dict[str, str] = {f"item_{i}": oid for i, oid in enumerate(items, start=1)}
    objects["container"] = container
    room = _room(scene)
    return {
        "task_id": f"A_D2__{scene}__{room}_seed0",
        "coordination_dim": "D2",
        "task_family": "A",
        "task_name": f"{room}_ordered_cabinet_loading",
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
                "Engine-enforced precedence: a closed cabinet rejects Put, so Open must precede every Put; closing before all items are in would block the remaining Puts.",
                "agent_1 opens + loads the first item; agent_2 loads the rest then closes -> cross-agent ordering. Kitchen storage (Cabinet/Drawer) -- a third, non-C D2 carrier.",
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
        "provenance": {"generator": "gen_cabinet_d2.py", "placement_verified": True, "room": room},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate family-A kitchen-cabinet D2 configs (third non-C D2 carrier).")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
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
            base = _kitchen_base(controller, scene)
            if len(base["items"]) < 2 or not base["containers"]:
                skipped.append((scene, "no items/openable container"))
                print(f"SKIP {scene}: items={len(base['items'])} containers={len(base['containers'])}")
                continue
            resolved = first_c_d2(controller, scene, base)
            if not resolved:
                skipped.append((scene, "A_D2"))
                print(f"SKIP {scene}: no verified A_D2 ordered-loading assignment")
                continue
            path = write_config(build_a_d2(scene, resolved), overwrite=args.overwrite)
            if path is not None:
                written.append(path)
    finally:
        controller.stop()
    print(f"generated {len(written)} A_D2 configs; skipped={len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
