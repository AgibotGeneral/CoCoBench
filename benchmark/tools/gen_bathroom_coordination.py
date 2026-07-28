#!/usr/bin/env python3
"""Generate bathroom coordination instances (I_D3: shared Sink/Faucet exclusion).

iTHOR bathrooms expose no fillable container and only one dirtyable pickup (Cloth),
so the kitchen wet-cleaning family (F) cannot be ported. The realizable D3 construct
here is contention over the single shared Sink/Faucet station: two agents each fetch
a scattered toiletry and deposit it at the one station, gated by the single Faucet.
The D3 signal is scored from contested claims, penalizing duplicate claims and
contention.

Each candidate is verified by driving the same SkillExecutor used by evaluation;
only instances whose representative oracle sequence succeeds are written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "tools"))
from gen_livingroom_coordination import set_spawn_seed, reset_scene, _seed_rewrite  # noqa: E402

from ai2thor.controller import Controller  # noqa: E402
from ai2thor.platform import CloudRendering  # noqa: E402
from skill_executor import SkillExecutor  # noqa: E402

OUT_ROOT = BENCH / "task_config"
DEFAULT_SCENES = [f"FloorPlan{idx}" for idx in range(401, 431)]

# Bathroom pickups, ordered by how cleanly they sit in a basin / on a counter.
BATH_ITEM_PRIORITY = (
    "SoapBar",
    "ScrubBrush",
    "Cloth",
    "DishSponge",
    "HandTowel",
    "Candle",
    "SprayBottle",
    "ToiletPaper",
    "SoapBottle",
)
# The shared deposit station, most "sink-like" first.
STATION_PRIORITY = ("SinkBasin", "CounterTop", "Bathtub", "Shelf", "SideTable")


def _objects(controller: Controller) -> List[Dict[str, Any]]:
    return list(controller.last_event.metadata.get("objects", []))


def _controller_init(scene: str) -> Dict[str, Any]:
    return {"scene": scene, "agentCount": 2, "gridSize": 0.25, "visibilityDistance": 1.5}


def _asserts(objects: Dict[str, str], keys: Iterable[str]) -> List[Dict[str, Any]]:
    return [{"action": "assert_present", "objectId": objects[key]} for key in keys]


def _ranked(objs: Iterable[Dict[str, Any]], type_order: Sequence[str], *, pickupable: Optional[bool] = None,
            receptacle: Optional[bool] = None) -> List[Dict[str, Any]]:
    order = {name: idx for idx, name in enumerate(type_order)}
    out: List[Dict[str, Any]] = []
    for obj in objs:
        if obj.get("objectType") not in order:
            continue
        if pickupable is not None and bool(obj.get("pickupable")) != pickupable:
            continue
        if receptacle is not None and bool(obj.get("receptacle")) != receptacle:
            continue
        out.append(obj)
    return sorted(out, key=lambda item: (order[item.get("objectType")], item.get("objectId", "")))


def _try(executor: SkillExecutor, call: str) -> bool:
    return executor.execute_call(call).success


def _put_sequence(executor: SkillExecutor, agent: str, item: str, target: str) -> bool:
    return (
        _try(executor, f"Find({agent}, {item})")
        and _try(executor, f"PickUp({agent}, {item})")
        and _try(executor, f"Find({agent}, {target})")
        and _try(executor, f"Put({agent}, {target})")
    )


def _all_in(controller: Controller, items: Sequence[str], station: str) -> bool:
    byid = {o["objectId"]: o for o in _objects(controller)}
    return all(station in ((byid.get(it) or {}).get("parentReceptacles") or []) for it in items)


def verify_i_d3(controller: Controller, scene: str, faucet: str, station: str,
                item_1: str, item_2: str) -> bool:
    reset_scene(controller, scene)
    ex = SkillExecutor(controller)
    # Open the shared station (toggle the single Faucet on), then both deposit lines.
    _try(ex, f"Find(agent_1, {faucet})")
    _try(ex, f"ToggleOn(agent_1, {faucet})")
    if not _put_sequence(ex, "agent_1", item_1, station):
        return False
    for direction in ("back", "left", "right", "back"):
        _try(ex, f"Explore(agent_1, {direction})")
    if not _put_sequence(ex, "agent_2", item_2, station):
        return False
    return _all_in(controller, [item_1, item_2], station)


def first_i_d3(controller: Controller, scene: str) -> Optional[Dict[str, str]]:
    reset_scene(controller, scene)
    objs = _objects(controller)
    faucets = [o for o in objs if o.get("objectType") == "Faucet"]
    if not faucets:
        return None
    faucet = faucets[0]["objectId"]
    items = _ranked(objs, BATH_ITEM_PRIORITY, pickupable=True)
    stations = _ranked(objs, STATION_PRIORITY, receptacle=True)
    if len(items) < 2 or not stations:
        return None
    for station in stations:
        sid = station["objectId"]
        chosen = [it["objectId"] for it in items
                  if sid not in set(it.get("parentReceptacles") or [])][:2]
        if len(chosen) < 2:
            continue
        if verify_i_d3(controller, scene, faucet, sid, chosen[0], chosen[1]):
            return {"faucet": faucet, "station": sid, "item_1": chosen[0], "item_2": chosen[1]}
    return None


def build_i_d3(scene: str, r: Dict[str, str]) -> Dict[str, Any]:
    skills = ["Find", "Explore", "Wait", "PickUp", "Put", "ToggleOn", "ToggleOff"]
    objects = {key: r[key] for key in ("faucet", "station", "item_1", "item_2")}
    return {
        "task_id": f"I_D3__{scene}__seed0",
        "coordination_dim": "D3",
        "task_family": "I",
        "task_name": "bathroom_shared_sink_collection",
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
            "init_actions": _asserts(objects, objects.keys()),
            "design_notes": [
                "D3 resource exclusion at the single shared Sink/Faucet station: both deposit lines must pass through the one station, only one agent at a time.",
                "iTHOR bathrooms have no fillable container and only one dirtyable pickup (Cloth), so the kitchen wet-cleaning mechanic (F) is not portable; the contention is the shared station itself.",
                "Scored as contested claims: duplicate claims and collisions are penalized.",
            ],
        },
        "allowed_skills": skills,
        "goal_predicates": [
            {"predicate": "on", "object": "item_1", "receptacle": "station"},
            {"predicate": "on", "object": "item_2", "receptacle": "station"},
        ],
        "task_constraints": {
            "resource_exclusion": [
                {"resource": "shared_sink_station", "exclusive": True,
                 "note": "Both deposit lines pass through the single Sink/Faucet station; only one agent can occupy it at a time."}
            ],
            "shared_station": "station",
            "legal_plan": "Each agent fetches one toiletry and deposits it at the shared station; the single Faucet gates the station.",
        },
        "success_fn": "relative_score_and_legal_plan (L0 feasibility check uses all_goal_predicates)",
        "eval_layer": "L0",
        "difficulty": {"object_count": 2, "step_budget": 30, "partial_observability": True},
        "provenance": {"generator": "gen_bathroom_coordination.py", "placement_verified": True, "room": "bathroom"},
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
    parser = argparse.ArgumentParser(description="Generate bathroom I_D3 task configs.")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
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
    skipped: List[Tuple[str, str]] = []
    try:
        for scene in args.scenes:
            resolved = first_i_d3(controller, scene)
            if not resolved:
                skipped.append((scene, "no verified I_D3 assignment"))
                print(f"SKIP {scene}: no verified I_D3 assignment")
                continue
            path = write_config(build_i_d3(scene, resolved), overwrite=args.overwrite)
            if path is not None:
                written.append(path)
    finally:
        controller.stop()
    print(f"generated {len(written)} configs; skipped={len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
