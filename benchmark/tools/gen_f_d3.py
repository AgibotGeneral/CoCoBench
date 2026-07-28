"""Self-adaptive F×D3 instance generator with generation-time placement verification.

Generates F×D3 instances on new kitchen scenes by resolving task objects from
live metadata (no hardcoded objectIds), then — the key yield lever — actually
DRIVING the skill executor during generation to confirm each placement the task
depends on. A dish is committed to the first candidate CounterTop the engine
accepts (PickUp dish -> Find counter -> Put succeeds); the kettle is kept only if
Put(kettle, stove) succeeds. This filters placement failures during generation.

  --verify-place    (default) test placements with the executor, pick working counters
  --no-verify-place use the emptiest-counter heuristic only

Run under the thor5 env with CloudRendering on PATH.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))            # core modules (skill_executor, ...)
sys.path.insert(0, str(BENCH / "tools"))
from gen_livingroom_coordination import set_spawn_seed, reset_scene, _seed_rewrite  # noqa: E402

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from skill_executor import SkillExecutor   # noqa: E402

OUT_DIR = BENCH / "task_config" / "F" / "D3"
DISH_TYPES = ("Mug", "Bowl", "Cup", "Plate")
N_CANDIDATE_COUNTERS = 4


def _objs(controller) -> List[Dict[str, Any]]:
    return controller.last_event.metadata["objects"]


def _by_type(objs, t):
    return [o for o in objs if o["objectType"] == t]


def _first(objs, t, **flags):
    for o in objs:
        if o["objectType"] == t and all(o.get(k) == v for k, v in flags.items()):
            return o["objectId"]
    return None


def ranked_countertops(objs, k=N_CANDIDATE_COUNTERS) -> List[str]:
    counters = _by_type(objs, "CounterTop")
    load = {c["objectId"]: 0 for c in counters}
    for o in objs:
        for p in (o.get("parentReceptacles") or []):
            if p in load:
                load[p] += 1
    return sorted(load, key=load.get)[:k]


def resolve_candidates(controller, scene: str) -> Optional[Dict[str, Any]]:
    reset_scene(controller, scene)
    objs = _objs(controller)
    faucet = _first(objs, "Faucet")
    sink = _first(objs, "Sink") or _first(objs, "SinkBasin")
    stove = _first(objs, "StoveBurner")
    counters = ranked_countertops(objs)
    dishes = [o["objectId"] for t in DISH_TYPES for o in _by_type(objs, t) if o.get("dirtyable")]
    kettle = _first(objs, "Kettle", canFillWithLiquid=True) or _first(objs, "Kettle")
    missing = [n for n, v in [("faucet", faucet), ("sink", sink), ("stove_burner", stove)] if not v]
    if missing or len(dishes) < 2 or len(counters) < 1:
        return {"scene": scene, "skip": missing + ([] if len(dishes) >= 2 else ["<2 dishes"]) +
                ([] if counters else ["no CounterTop"])}
    return {"scene": scene, "faucet": faucet, "sink": sink, "stove_burner": stove,
            "counters": counters, "dishes": dishes, "kettle": kettle}


def _try(ex: SkillExecutor, call: str) -> bool:
    return ex.execute_call(call).success


def verify_placements(controller, c: Dict[str, Any]) -> Dict[str, Any]:
    """Drive the executor to pick a working counter for each of 2 dishes (and
    confirm kettle->stove). Returns finalized assignment or {skip:...}."""
    reset_scene(controller, c["scene"])
    ex = SkillExecutor(controller)
    used: List[str] = []
    assigned: List[str] = []   # (dish, counter) pairs flattened -> dish list with counters
    dish_counter: Dict[str, str] = {}
    for dish in c["dishes"]:
        if len(dish_counter) == 2:
            break
        if not (_try(ex, f"Find(agent_1, {dish})") or True):  # Find best-effort
            pass
        _try(ex, f"Find(agent_1, {dish})")
        if not _try(ex, f"PickUp(agent_1, {dish})"):
            continue  # unreachable/unpickable dish, try next
        placed = None
        for counter in [c for c in c["counters"] if c not in used]:
            _try(ex, f"Find(agent_1, {counter})")
            if _try(ex, f"Put(agent_1, {counter})"):
                placed = counter
                break
        if placed:
            dish_counter[dish] = placed
            used.append(placed)
        else:
            _try(ex, "Drop(agent_1)")  # free the hand and try another dish
    if len(dish_counter) < 2:
        return {"scene": c["scene"], "skip": ["no 2 placeable dish/counter pairs"]}
    dishes = list(dish_counter.keys())
    out = {"scene": c["scene"], "faucet": c["faucet"], "sink": c["sink"], "stove_burner": c["stove_burner"],
           "mug": dishes[0], "bowl": dishes[1],
           "clean_zone": dish_counter[dishes[0]], "clean_zone_2": dish_counter[dishes[1]], "kettle": None}
    # verify kettle -> stove (fresh reset so dishes don't block)
    if c["kettle"]:
        reset_scene(controller, c["scene"])
        ex = SkillExecutor(controller)
        _try(ex, f"Find(agent_1, {c['kettle']})")
        if _try(ex, f"PickUp(agent_1, {c['kettle']})"):
            _try(ex, f"Find(agent_1, {c['stove_burner']})")
            if _try(ex, f"Put(agent_1, {c['stove_burner']})"):
                out["kettle"] = c["kettle"]
    return out


def heuristic_assignment(c: Dict[str, Any]) -> Dict[str, Any]:
    """No-verify baseline: emptiest two counters, no executor check (for E2)."""
    counters = c["counters"]
    return {"scene": c["scene"], "faucet": c["faucet"], "sink": c["sink"], "stove_burner": c["stove_burner"],
            "mug": c["dishes"][0], "bowl": c["dishes"][1],
            "clean_zone": counters[0], "clean_zone_2": counters[1] if len(counters) > 1 else counters[0],
            "kettle": c["kettle"]}


def build_config(r: Dict[str, Any]) -> Dict[str, Any]:
    scene = r["scene"]
    objects = {"mug": r["mug"], "bowl": r["bowl"], "faucet": r["faucet"], "sink": r["sink"],
               "clean_zone": r["clean_zone"], "clean_zone_2": r["clean_zone_2"], "stove_burner": r["stove_burner"]}
    goals = [{"predicate": "clean", "object": "mug"}, {"predicate": "clean", "object": "bowl"},
             {"predicate": "on", "object": "mug", "receptacle": "clean_zone"},
             {"predicate": "on", "object": "bowl", "receptacle": "clean_zone_2"},
             {"predicate": "toggled", "object": "faucet", "value": False}]
    init_actions = [{"action": "DirtyObject", "objectId": r["mug"]},
                    {"action": "DirtyObject", "objectId": r["bowl"]},
                    {"action": "ensure_toggle_state", "objectId": r["faucet"], "isToggled": False}]
    if r.get("kettle"):
        objects["kettle"] = r["kettle"]
        goals += [{"predicate": "filled", "object": "kettle", "liquid": "water"},
                  {"predicate": "on", "object": "kettle", "receptacle": "stove_burner"}]
        init_actions.append({"action": "ensure_empty", "objectId": r["kettle"]})
    skills = ["Find", "Explore", "Wait", "PickUp", "Put", "ToggleOn", "ToggleOff",
              "CleanObject", "FillObjectWithLiquid", "EmptyLiquidFromObject"]
    return {
        "task_id": f"F_D3__{scene}__seed0", "coordination_dim": "D3", "task_family": "F",
        "task_name": "shared_sink_clean_and_fill", "scene_id": scene, "agent_count": 2,
        "agents": [{"id": "agent_1", "role": "Agent-Cleaner-A", "allowed_skills": skills},
                   {"id": "agent_2", "role": "Agent-Cleaner-B", "allowed_skills": skills}],
        "seed": 0,
        "init_state": {"controller_init": {"scene": scene, "agentCount": 2, "gridSize": 0.25, "visibilityDistance": 1.5},
                       "objects": objects, "init_actions": init_actions},
        "allowed_skills": skills, "goal_predicates": goals,
        "task_constraints": {"exclusive_resources": [{"objectId": r["faucet"], "capacity": 1},
                                                      {"objectId": r["sink"], "capacity": 1}]},
        "success_fn": "all_goal_predicates_and_legal_plan", "eval_layer": "L0",
        "difficulty": {"object_count": len(objects), "step_budget": 30, "partial_observability": True},
        "provenance": {"generator": "gen_f_d3.py", "placement_verified": bool(r.get("_verified"))},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--gpu-device", type=int, default=0)
    ap.add_argument("--no-verify-place", action="store_true", help="E2 baseline: emptiest-counter heuristic, no executor check.")
    ap.add_argument("--out-suffix", default="", help="Append to task_id/filename to keep ablation variants separate.")
    ap.add_argument("--seed", type=int, default=0, help="InitialRandomSpawn seed (0=scene default).")
    args = ap.parse_args()
    set_spawn_seed(args.seed)
    verify = not args.no_verify_place
    c = Controller(agentMode="default", platform=CloudRendering, gpu_device=args.gpu_device,
                   scene=args.scenes[0], gridSize=0.25, agentCount=2, width=144, height=144, visibilityDistance=1.5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []
    for scene in args.scenes:
        cand = resolve_candidates(c, scene)
        if cand.get("skip"):
            skipped.append((scene, cand["skip"])); print(f"SKIP {scene}: {cand['skip']}"); continue
        r = verify_placements(c, cand) if verify else heuristic_assignment(cand)
        if r.get("skip"):
            skipped.append((scene, r["skip"])); print(f"SKIP {scene}: {r['skip']}"); continue
        r["_verified"] = verify
        cfg = build_config(r)
        cfg = _seed_rewrite(cfg)
        if args.out_suffix:
            cfg["task_id"] += args.out_suffix
        path = OUT_DIR / f"{cfg['task_id']}.json"
        json.dump(cfg, open(path, "w"), ensure_ascii=False, indent=2)
        written.append(scene)
        print(f"WROTE {path.name}  verify={verify} kettle={'Y' if r.get('kettle') else 'N'} "
              f"zones=({r['clean_zone'].split('|')[0]},{r['clean_zone_2'].split('|')[0]})")
    print(f"\nverify_place={verify} written={len(written)} skipped={len(skipped)}")
    print(f"written_scenes={written}")
    print(f"skipped={skipped}")
    c.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
