#!/usr/bin/env python3
"""Generate family-J relay-handoff instances.

Family J defines a cross-zone handoff relay in bedroom and living-room scenes:
clutter is relayed from a source through a capacity-limited buffer to a target.

The D4 construct is identical to C_D4 (producer -> single capacity-limited buffer ->
consumer), so this REUSES the family-agnostic oracle ``plan_c_d4`` (registered as
``J_D4``) and the menu gating ``taskutil.producer_consumer_allows`` unchanged. Only
the family label, room, and object mix differ.

Resolver/verifier are reused from gen_livingroom_coordination (the relay placement
check is family-independent); this module only adds the J-flavored config builder.
Each kept instance has its full relay sequence verified in-process; run
``validate_instances.py --only-cell J_D4`` afterwards for reference validation.
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

from ai2thor.controller import Controller  # noqa: E402
from ai2thor.platform import CloudRendering  # noqa: E402

# Reuse the family-agnostic relay resolver/verifier + scene helpers from the
# living-room generator (the relay placement check does not depend on the family).
from gen_livingroom_coordination import (  # noqa: E402
    OUT_ROOT,
    _candidate_base,
    _controller_init,
    _asserts,
    _room,
    first_c_d4,
    write_config,
    set_spawn_seed,
)

# Bedroom is the new room for D4 (kitchen+living already carry C_D4); living room is
# allowed as a fallback to top up yield while staying off the C family.
DEFAULT_SCENES = [f"FloorPlan{idx}" for idx in range(301, 331)]


def build_j_d4(scene: str, r: Dict[str, Any]) -> Dict[str, Any]:
    """Same relay schema as C_D4 (producer -> buffer -> consumer), family J, with a
    bedroom/living-room object+surface mix. ``producer_consumer`` drives both the
    reused oracle (plan_c_d4) and the menu gating; the buffer is the single D4
    chokepoint (capacity 1)."""
    source = ["Find", "Explore", "PickUp", "Put", "Drop"]
    target_skills = ["Find", "Explore", "PickUp", "Put"]
    items = list(r["items"])
    objects: Dict[str, str] = {f"item_{i}": oid for i, oid in enumerate(items, start=1)}
    objects["transfer"] = r["transfer"]
    objects["target"] = r["target"]
    room = _room(scene)
    return {
        "task_id": f"J_D4__{scene}__{room}_seed0",
        "coordination_dim": "D4",
        "task_family": "J",
        "task_name": f"{room}_cross_zone_handoff_relay",
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
                f"{len(items)} clutter items are relayed source -> buffer surface (capacity 1) -> storage target.",
                "Second D4 carrier (family J) in a new room type so the D4 construct is not measured on family C alone.",
                "Multi-object + capacity-limited buffer creates producer-consumer queue pressure (overflow/starvation).",
                "Agent-Source may only Put on the buffer; Agent-Target may only PickUp on-buffer items and Put on the target.",
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
            "role_constraints": ["Source handles source-to-buffer; Target handles buffer-to-target."],
            "legal_plan": "Each item must pass through the buffer surface before final placement.",
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
        "provenance": {"generator": "gen_relay_handoff.py", "placement_verified": True, "room": room},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate family-J relay-handoff (D4 second carrier) configs.")
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
            base = _candidate_base(controller, scene)
            if len(base["items"]) < 3 or len(base["open_surfaces"]) < 2:
                skipped.append((scene, "missing items or surfaces"))
                print(f"SKIP {scene}: missing items or surfaces")
                continue
            resolved = first_c_d4(controller, scene, base)
            if not resolved:
                skipped.append((scene, "J_D4"))
                print(f"SKIP {scene}: no verified J_D4 relay assignment")
                continue
            config = build_j_d4(scene, resolved)
            path = write_config(config, overwrite=args.overwrite)
            if path is not None:
                written.append(path)
    finally:
        controller.stop()
    print(f"generated {len(written)} J_D4 configs; skipped={len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
