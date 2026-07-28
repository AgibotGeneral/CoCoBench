#!/usr/bin/env python3
"""Generate 3-4 agent scaling variants for the coordination benchmark.

This is the dedicated N-agent generator for the agent-count axis. It boots one
controller at ``--agents N`` and,
for each scalable cell, selects N (item, receptacle) lines, verifies the N-agent
oracle sequence IN-PROCESS by driving the same ``SkillExecutor`` used at
evaluation, and writes only instances that succeed. Full oracle M0 validation
(``tools/validate_instances.py``) should still be run afterwards.

The low-level placement helpers are imported from the 2-agent generators so
the object-ranking / executor-driving logic is shared, not duplicated.

Scalable cells and how each scales with agent_count N:
  * H_D1  D1  N agents, N personal drawers, N items (independent makespan).
  * G_D1  D1  N agents, N items, N distinct target surfaces (independent).
  * E_D3  D3  N agents, N items, N own baskets (contested-claim exclusion).
  * I_D3  D3  N agents share ONE Sink/Faucet station (resource exclusion ↑).
  * K_D3  D3  N agents share ONE Knife to slice N food items (PickUp exclusion ↑).
  * C_D2  D2  open → N agents load N items in order → close (precedence chain).
  * H_D2  D2  same, bedroom furniture container.
  * C_D4  D4  fan-in/out: producers → ONE buffer → consumers (relay chokepoint).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "tools"))

from ai2thor.controller import Controller  # noqa: E402
from ai2thor.platform import CloudRendering  # noqa: E402
from skill_executor import SkillExecutor  # noqa: E402

# Reuse the placement/ranking helpers from the 2-agent generators (single source).
import gen_livingroom_coordination as L  # noqa: E402
import gen_bathroom_coordination as Bm  # noqa: E402
import gen_shared_knife_d3 as Kd3  # noqa: E402

OUT_ROOT = BENCH / "task_config"


def _controller_init(scene: str, n: int) -> Dict[str, Any]:
    return {"scene": scene, "agentCount": n, "gridSize": 0.25, "visibilityDistance": 1.5}


def _asserts(objects: Dict[str, str]) -> List[Dict[str, Any]]:
    return [{"action": "assert_present", "objectId": oid} for oid in objects.values()]


def _agents(n: int, role: str, skills: Sequence[str]) -> List[Dict[str, Any]]:
    return [{"id": f"agent_{i}", "role": f"{role}-{i}", "allowed_skills": list(skills)}
            for i in range(1, n + 1)]


def _items_not_in(base: Dict[str, List[Dict[str, Any]]], reserved: set, n: int) -> List[str]:
    """Up to n pickupable items not already inside any reserved receptacle."""
    out: List[str] = []
    for it in base["items"][:12]:
        if set(it.get("parentReceptacles") or []) & reserved:
            continue
        out.append(it["objectId"])
        if len(out) >= n:
            break
    return out


def _distinct(objs: List[Dict[str, Any]], n: int) -> List[str]:
    seen, out = set(), []
    for o in objs:
        oid = o["objectId"]
        if oid in seen:
            continue
        seen.add(oid)
        out.append(oid)
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# D1 — independent parallelism
# --------------------------------------------------------------------------- #

def gen_h_d1(controller: Controller, scene: str, base: Dict[str, Any], n: int) -> Optional[Dict[str, Any]]:
    drawers = _distinct(base["furniture_containers"], n)
    if len(drawers) < n:
        return None
    items = _items_not_in(base, set(drawers), n)
    if len(items) < n:
        return None
    # Verify: each agent opens its OWN drawer and stores its OWN item.
    L.reset_scene(controller, scene)
    for d in drawers:
        L._force_close(controller, d)
    ex = SkillExecutor(controller)
    for i in range(n):
        agent = f"agent_{i + 1}"
        if not (L._try(ex, f"Find({agent}, {drawers[i]})") and L._try(ex, f"Open({agent}, {drawers[i]})")):
            return None
        if not L._put_sequence(ex, agent, items[i], drawers[i]):
            return None
        for direction in ("back", "left", "right", "back"):
            L._try(ex, f"Explore({agent}, {direction})")
    if not all(L._all_in(controller, [items[i]], drawers[i]) for i in range(n)):
        return None
    return {"items": items, "drawers": drawers}


def build_h_d1(scene: str, r: Dict[str, Any], n: int) -> Dict[str, Any]:
    skills = ["Find", "Explore", "PickUp", "Put", "Open", "Close"]
    objects: Dict[str, str] = {}
    for i, (it, dr) in enumerate(zip(r["items"], r["drawers"]), start=1):
        objects[f"item_{i}"] = it
        objects[f"drawer_{i}"] = dr
    room = L._room(scene)
    return {
        "task_id": f"H_D1__{scene}__{n}agent_seed0",
        "coordination_dim": "D1", "task_family": "H",
        "task_name": f"{room}_personal_drawer_storage",
        "scene_id": scene, "agent_count": n,
        "agents": _agents(n, "Tidier", skills), "seed": 0,
        "init_state": {
            "controller_init": _controller_init(scene, n),
            "objects": objects,
            "init_actions": _asserts(objects)
            + [{"action": "CloseObject", "objectId": dr, "forceAction": True} for dr in r["drawers"]],
            "design_notes": [
                f"D1 independent parallelism scaled to {n} agents: each agent owns one drawer and stores one item; lines share no object and impose no cross-agent ordering.",
                "Action counts are symmetric per agent, so the balanced schedule is the makespan optimum and a single-agent solution multiplies makespan by n.",
            ],
        },
        "allowed_skills": skills,
        "goal_predicates": [{"predicate": "on", "object": f"item_{i}", "receptacle": f"drawer_{i}"}
                            for i in range(1, n + 1)],
        "task_constraints": {
            "independent_subtasks": [[f"Open(drawer_{i})", f"PickUp(item_{i})", f"Put(item_{i}, drawer_{i})"]
                                     for i in range(1, n + 1)],
            "legal_plan": "Each agent opens its own drawer and stores its own item; all lines are independent and may run in parallel.",
        },
        "success_fn": "all_goal_predicates_and_legal_plan", "eval_layer": "L0",
        "difficulty": {"object_count": n, "step_budget": 12 * n, "partial_observability": True},
        "provenance": {"generator": "gen_multiagent_scaling.py", "placement_verified": True, "room": room, "agent_count": n},
    }


def gen_g_d1(controller: Controller, scene: str, base: Dict[str, Any], n: int) -> Optional[Dict[str, Any]]:
    targets = _distinct(base["open_surfaces"], n)
    if len(targets) < n:
        return None
    items = _items_not_in(base, set(targets), n)
    if len(items) < n:
        return None
    L.reset_scene(controller, scene)
    ex = SkillExecutor(controller)
    for i in range(n):
        agent = f"agent_{i + 1}"
        if not L._put_sequence(ex, agent, items[i], targets[i]):
            return None
    return {"items": items, "targets": targets}


def build_g_d1(scene: str, r: Dict[str, Any], n: int) -> Dict[str, Any]:
    skills = ["Find", "Explore", "PickUp", "Put"]
    objects: Dict[str, str] = {}
    for i, (it, tg) in enumerate(zip(r["items"], r["targets"]), start=1):
        objects[f"item_{i}"] = it
        objects[f"target_{i}"] = tg
    room = L._room(scene)
    return {
        "task_id": f"G_D1__{scene}__{n}agent_seed0",
        "coordination_dim": "D1", "task_family": "G",
        "task_name": f"{room}_parallel_sorting",
        "scene_id": scene, "agent_count": n,
        "agents": _agents(n, "Sorter", skills), "seed": 0,
        "init_state": {
            "controller_init": _controller_init(scene, n),
            "objects": objects, "init_actions": _asserts(objects),
            "design_notes": [
                f"D1 has no cross-agent ordering: each of {n} agents moves one item to a distinct receptacle surface.",
                "Only non-openable receptacles are used so the task stays focused on independent allocation.",
            ],
        },
        "allowed_skills": skills,
        "goal_predicates": [{"predicate": "on", "object": f"item_{i}", "receptacle": f"target_{i}"}
                            for i in range(1, n + 1)],
        "task_constraints": {
            "independent_subtasks": [[f"PickUp(item_{i})", f"Put(item_{i}, target_{i})"] for i in range(1, n + 1)],
            "legal_plan": "The placement lines are independent and may run in any order or in parallel.",
        },
        "success_fn": "all_goal_predicates_and_legal_plan", "eval_layer": "L0",
        "difficulty": {"object_count": n, "step_budget": 10 * n, "partial_observability": True},
        "provenance": {"generator": "gen_multiagent_scaling.py", "placement_verified": True, "room": room, "agent_count": n},
    }


# --------------------------------------------------------------------------- #
# D3 — resource exclusion
# --------------------------------------------------------------------------- #

def gen_e_d3(controller: Controller, scene: str, base: Dict[str, Any], n: int) -> Optional[Dict[str, Any]]:
    baskets = _distinct(base["open_surfaces"], n)  # non-openable surfaces -> no Open needed
    if len(baskets) < n:
        return None
    items = _items_not_in(base, set(baskets), n)
    if len(items) < n:
        return None
    L.reset_scene(controller, scene)
    ex = SkillExecutor(controller)
    for i in range(n):
        agent = f"agent_{i + 1}"
        if not L._put_sequence(ex, agent, items[i], baskets[i]):
            return None
    return {"items": items, "baskets": baskets}


def build_e_d3(scene: str, r: Dict[str, Any], n: int) -> Dict[str, Any]:
    skills = ["Find", "Explore", "PickUp", "Put", "Open", "Close"]
    objects: Dict[str, str] = {}
    for i, (it, bk) in enumerate(zip(r["items"], r["baskets"]), start=1):
        objects[f"item_{i}"] = it
        objects[f"basket_{i}"] = bk
    room = L._room(scene)
    return {
        "task_id": f"E_D3__{scene}__{n}agent_seed0",
        "coordination_dim": "D3", "task_family": "E",
        "task_name": "competitive_collection_baskets",
        "scene_id": scene, "agent_count": n,
        "agents": _agents(n, "Collector", skills), "seed": 0,
        "init_state": {
            "controller_init": _controller_init(scene, n),
            "objects": objects, "init_actions": _asserts(objects),
            "design_notes": [
                f"Each of {n} collectors owns a basket; the L0 assignment gives one target object to each agent.",
                "D3 pressure comes from exclusive access to scattered objects and contention around shared furniture; more agents raise contention.",
            ],
        },
        "allowed_skills": skills,
        "goal_predicates": [{"predicate": "on", "object": f"item_{i}", "receptacle": f"basket_{i}"}
                            for i in range(1, n + 1)],
        "task_constraints": {
            "resource_exclusion": [{"resource": "contested_object", "exclusive": True,
                                    "note": "Only one agent can hold a pickupable object at a time."}],
            "open_baskets": [],
            "scoring": "score(agent) = number of target objects in that agent's basket; higher score wins.",
            "legal_plan": "The plan must not use unauthorized skills; each agent claims its own item.",
        },
        "success_fn": "relative_score_and_legal_plan (L0 feasibility check uses all_goal_predicates)",
        "eval_layer": "L0",
        "difficulty": {"object_count": n, "step_budget": 12 * n, "partial_observability": True},
        "provenance": {"generator": "gen_multiagent_scaling.py", "placement_verified": True, "room": room, "agent_count": n},
    }


def gen_i_d3(controller: Controller, scene: str, base_unused: Any, n: int) -> Optional[Dict[str, Any]]:
    """Bathroom shared single Sink/Faucet: n agents each deposit one toiletry at the
    one shared station, gated by the one Faucet (resource exclusion scales up)."""
    L.reset_scene(controller, scene)
    objs = Bm._objects(controller)
    faucets = [o for o in objs if o.get("objectType") == "Faucet"]
    if not faucets:
        return None
    faucet = faucets[0]["objectId"]
    items_all = Bm._ranked(objs, Bm.BATH_ITEM_PRIORITY, pickupable=True)
    stations = Bm._ranked(objs, Bm.STATION_PRIORITY, receptacle=True)
    if len(items_all) < n or not stations:
        return None
    for station in stations:
        sid = station["objectId"]
        chosen = [it["objectId"] for it in items_all if sid not in set(it.get("parentReceptacles") or [])][:n]
        if len(chosen) < n:
            continue
        L.reset_scene(controller, scene)
        ex = SkillExecutor(controller)
        L._try(ex, f"Find(agent_1, {faucet})")
        L._try(ex, f"ToggleOn(agent_1, {faucet})")
        ok = True
        for i in range(n):
            agent = f"agent_{i + 1}"
            if not Bm._put_sequence(ex, agent, chosen[i], sid):
                ok = False
                break
            for direction in ("back", "left", "right", "back"):
                L._try(ex, f"Explore({agent}, {direction})")
        if ok and Bm._all_in(controller, chosen, sid):
            return {"faucet": faucet, "station": sid, "items": chosen}
    return None


def build_i_d3(scene: str, r: Dict[str, Any], n: int) -> Dict[str, Any]:
    skills = ["Find", "Explore", "Wait", "PickUp", "Put", "ToggleOn", "ToggleOff"]
    objects: Dict[str, str] = {"faucet": r["faucet"], "station": r["station"]}
    for i, it in enumerate(r["items"], start=1):
        objects[f"item_{i}"] = it
    return {
        "task_id": f"I_D3__{scene}__{n}agent_seed0",
        "coordination_dim": "D3", "task_family": "I",
        "task_name": "bathroom_shared_sink_collection",
        "scene_id": scene, "agent_count": n,
        "agents": _agents(n, "Tidier", skills), "seed": 0,
        "init_state": {
            "controller_init": _controller_init(scene, n),
            "objects": objects, "init_actions": _asserts(objects),
            "design_notes": [
                f"D3 resource exclusion at the single shared Sink/Faucet station, now contended by {n} agents: all deposit lines pass through the one station, only one agent at a time.",
                "Scored as contested claims: duplicate claims and collisions are penalized.",
            ],
        },
        "allowed_skills": skills,
        "goal_predicates": [{"predicate": "on", "object": f"item_{i}", "receptacle": "station"}
                            for i in range(1, n + 1)],
        "task_constraints": {
            "resource_exclusion": [{"resource": "shared_sink_station", "exclusive": True,
                                    "note": "All deposit lines pass through the single Sink/Faucet station; only one agent can occupy it at a time."}],
            "shared_station": "station",
            "legal_plan": "Each agent fetches one toiletry and deposits it at the shared station; the single Faucet gates the station.",
        },
        "success_fn": "relative_score_and_legal_plan (L0 feasibility check uses all_goal_predicates)",
        "eval_layer": "L0",
        "difficulty": {"object_count": n, "step_budget": 12 * n + 6, "partial_observability": True},
        "provenance": {"generator": "gen_multiagent_scaling.py", "placement_verified": True, "room": "bathroom", "agent_count": n},
    }


# --------------------------------------------------------------------------- #
# K_D3 — shared Knife slicing exclusion (kitchen)
# --------------------------------------------------------------------------- #

def gen_k_d3(controller: Controller, scene: str, base_unused: Any, n: int,
             items_per_agent: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
    """Kitchen shared Knife: n agents each slice their assigned food using the single Knife.
    PickUp exclusion enforces one-at-a-time access.  ``items_per_agent`` enables
    asymmetric critical sections (e.g. [2,1,1] for 3-agent)."""
    r = Kd3.first_k_d3(controller, scene, n_items=n,
                        items_per_agent=items_per_agent)
    return r


def build_k_d3(scene: str, r: Dict[str, Any], n: int,
               items_per_agent: Optional[List[int]] = None) -> Dict[str, Any]:
    return Kd3.build_k_d3(scene, r, n_agents=n, items_per_agent=items_per_agent)


# --------------------------------------------------------------------------- #
# D2 — sequential dependency (ordered container loading)
# --------------------------------------------------------------------------- #

def _verify_ordered(controller: Controller, scene: str, container: str, items: Sequence[str], n: int) -> bool:
    """agent_1 opens + loads item_1; agent_i loads item_i; last loader closes.
    Kept only if every item is inside AND the container ends closed."""
    L.reset_scene(controller, scene)
    L._force_close(controller, container)
    ex = SkillExecutor(controller)
    if not (L._try(ex, f"Find(agent_1, {container})") and L._try(ex, f"Open(agent_1, {container})")):
        return False
    last = "agent_1"
    for i in range(n):
        agent = f"agent_{i + 1}"
        if not L._put_sequence(ex, agent, items[i], container):
            return False
        for direction in ("back", "left"):
            L._try(ex, f"Explore({agent}, {direction})")
        last = agent
    L._try(ex, f"Find({last}, {container})")
    if not L._try(ex, f"Close({last}, {container})"):
        return False
    if not L._all_in(controller, items, container):
        return False
    obj = next((o for o in L._objects(controller) if o["objectId"] == container), None)
    return bool(obj) and not obj.get("isOpen")


def _gen_ordered(controller: Controller, scene: str, containers: List[Dict[str, Any]], base: Dict[str, Any], n: int) -> Optional[Dict[str, Any]]:
    for container in containers[:6]:
        cid = container["objectId"]
        items = _items_not_in(base, {cid}, n)
        if len(items) < n:
            continue
        if _verify_ordered(controller, scene, cid, items, n):
            return {"container": cid, "items": items}
    return None


def gen_c_d2(controller, scene, base, n):
    return _gen_ordered(controller, scene, base["containers"], base, n)


def gen_h_d2(controller, scene, base, n):
    return _gen_ordered(controller, scene, base["furniture_containers"], base, n)


def _build_ordered(scene: str, r: Dict[str, Any], n: int, family: str, suffix: str, task_name: str, note: str) -> Dict[str, Any]:
    skills = ["Find", "Explore", "Wait", "PickUp", "Put", "Drop", "Open", "Close"]
    items = list(r["items"])
    container = r["container"]
    objects: Dict[str, str] = {f"item_{i}": oid for i, oid in enumerate(items, start=1)}
    objects["container"] = container
    room = L._room(scene)
    return {
        "task_id": f"{family}_D2__{scene}__{suffix}{n}agent_seed0",
        "coordination_dim": "D2", "task_family": family,
        "task_name": task_name, "scene_id": scene, "agent_count": n,
        "agents": _agents(n, "Agent", skills), "seed": 0,
        "init_state": {
            "controller_init": _controller_init(scene, n),
            "objects": objects,
            "init_actions": _asserts(objects) + [{"action": "CloseObject", "objectId": container, "forceAction": True}],
            "design_notes": [
                "Engine-enforced precedence: a closed container rejects Put, so Open must precede every Put; closing before all items are in would block the remaining Puts.",
                f"agent_1 opens + loads the first item; the other agents load the rest; the last loader closes -> a cross-agent ordering chain that lengthens with {n} items. {note}",
            ],
        },
        "allowed_skills": skills,
        "goal_predicates": [{"predicate": "on", "object": f"item_{i}", "receptacle": "container"}
                            for i in range(1, n + 1)] + [{"predicate": "closed", "object": "container"}],
        "task_constraints": {
            "precedence": [["Open(container)", "Put(item, container)"], ["Put(item, container)", "Close(container)"]],
            "role_constraints": ["All agents share all skills; the ordering open -> load(all) -> close is the coordination signal."],
            "legal_plan": "Open before any placement; close only after all items are inside.",
        },
        "success_fn": "all_goal_predicates_and_legal_plan", "eval_layer": "L0",
        "difficulty": {"object_count": n, "step_budget": 16 * n, "partial_observability": True},
        "provenance": {"generator": "gen_multiagent_scaling.py", "placement_verified": True, "room": room, "agent_count": n},
    }


def build_c_d2(scene, r, n):
    return _build_ordered(scene, r, n, "C", "livingroom_", "ordered_container_loading", "Box-type container.")


def build_h_d2(scene, r, n):
    return _build_ordered(scene, r, n, "H", "", f"{L._room(scene)}_ordered_drawer_loading", "Bedroom storage furniture (Drawer/Cabinet) instead of a Box.")


# --------------------------------------------------------------------------- #
# D4 — relay (fan-in/fan-out through one buffer)
# --------------------------------------------------------------------------- #

def _split_relay(n: int) -> Tuple[List[str], List[str]]:
    """Split n agents into producers and consumers (≈half each, ≥1 of each)."""
    n_prod = max(1, n // 2)
    producers = [f"agent_{i}" for i in range(1, n_prod + 1)]
    consumers = [f"agent_{i}" for i in range(n_prod + 1, n + 1)]
    return producers, consumers


def gen_c_d4(controller: Controller, scene: str, base: Dict[str, Any], n: int) -> Optional[Dict[str, Any]]:
    """Fan-in/out relay: several producers feed ONE transfer buffer, several
    consumers drain it to the target. Verify with the same strict per-item
    produce→consume alternation the oracle uses (buffer holds ≤1)."""
    producers, consumers = _split_relay(n)
    n_items = n  # one relayed item per agent's worth of work
    surfaces = base["open_surfaces"][:8]
    for transfer in surfaces:
        for target in surfaces:
            if transfer["objectId"] == target["objectId"]:
                continue
            reserved = {transfer["objectId"], target["objectId"]}
            items = _items_not_in(base, reserved, n_items)
            if len(items) < n_items:
                continue
            L.reset_scene(controller, scene)
            ex = SkillExecutor(controller)
            ok = True
            for idx, item in enumerate(items):
                p = producers[idx % len(producers)]
                c = consumers[idx % len(consumers)]
                if not L._put_sequence(ex, p, item, transfer["objectId"]):
                    ok = False
                    break
                for direction in ("back", "left", "right", "back"):
                    L._try(ex, f"Explore({p}, {direction})")
                if not L._put_sequence(ex, c, item, target["objectId"]):
                    ok = False
                    break
                for direction in ("back", "left", "right", "back"):
                    L._try(ex, f"Explore({c}, {direction})")
            if ok:
                return {"items": items, "transfer": transfer["objectId"], "target": target["objectId"],
                        "producers": producers, "consumers": consumers}
    return None


def build_c_d4(scene: str, r: Dict[str, Any], n: int) -> Dict[str, Any]:
    source_skills = ["Find", "Explore", "PickUp", "Put", "Drop"]
    target_skills = ["Find", "Explore", "PickUp", "Put"]
    items = list(r["items"])
    objects: Dict[str, str] = {f"item_{i}": oid for i, oid in enumerate(items, start=1)}
    objects["transfer"] = r["transfer"]
    objects["target"] = r["target"]
    producers, consumers = r["producers"], r["consumers"]
    agents = ([{"id": a, "role": f"Source-{i}", "allowed_skills": source_skills} for i, a in enumerate(producers, 1)]
              + [{"id": a, "role": f"Target-{i}", "allowed_skills": target_skills} for i, a in enumerate(consumers, 1)])
    room = L._room(scene)
    return {
        "task_id": f"C_D4__{scene}__{room}_{n}agent_seed0",
        "coordination_dim": "D4", "task_family": "C",
        "task_name": "multi_object_relay_fan", "scene_id": scene, "agent_count": n,
        "agents": agents, "seed": 0,
        "init_state": {
            "controller_init": _controller_init(scene, n),
            "objects": objects, "init_actions": _asserts(objects),
            "design_notes": [
                f"{len(items)} items are relayed source -> ONE transfer surface (capacity 1) -> target by {len(producers)} producers feeding the buffer and {len(consumers)} consumers draining it.",
                "Fan-in/fan-out keeps the single buffer as the coordination chokepoint at any agent count; producers may only Put on the buffer, consumers may only PickUp on-buffer items and Put on the target.",
            ],
        },
        "allowed_skills": sorted(set(source_skills + target_skills)),
        "goal_predicates": [{"predicate": "on", "object": f"item_{i}", "receptacle": "target"}
                            for i in range(1, len(items) + 1)],
        "task_constraints": {
            "precedence": [["Source.Put(item, transfer)", "Target.PickUp(item from transfer)"],
                           ["Target.PickUp(item)", "Target.Put(item, target)"]],
            "role_constraints": ["Producers handle source-to-buffer; consumers handle buffer-to-target."],
            "legal_plan": "Each item must pass through the transfer surface before final placement.",
            "buffer_stations": [{"objectId": r["transfer"], "capacity": 1}],
            "producer_consumer": {
                "producers": producers, "consumers": consumers,
                "buffer": r["transfer"], "targets": [r["target"]],
                "note": "Relay enforced at the action menu: producers Put->buffer only; consumers PickUp only on-buffer items, Put->target only. The single buffer is the D4 chokepoint regardless of agent count.",
            },
        },
        "success_fn": "all_goal_predicates_and_legal_plan", "eval_layer": "L0",
        "difficulty": {"object_count": len(items), "step_budget": 22 * n, "partial_observability": True},
        "provenance": {"generator": "gen_multiagent_scaling.py", "placement_verified": True, "room": room, "agent_count": n},
    }


def build_j_d4(scene: str, r: Dict[str, Any], n: int) -> Dict[str, Any]:
    """Family-J relay (D4 second carrier) at agent_count n. Identical fan-in/fan-out
    relay schema as build_c_d4 (reuses the family-agnostic gen_c_d4 resolver), only
    the family label / room naming differ — gives D4 a second carrier at 3/4 agents."""
    cfg = build_c_d4(scene, r, n)
    room = L._room(scene)
    cfg["task_id"] = f"J_D4__{scene}__{room}_{n}agent_seed0"
    cfg["task_family"] = "J"
    cfg["task_name"] = f"{room}_cross_zone_handoff_relay_fan"
    cfg["agents"] = ([{"id": a, "role": f"Agent-Source-{i}", "allowed_skills": ["Find", "Explore", "PickUp", "Put", "Drop"]}
                      for i, a in enumerate(r["producers"], 1)]
                     + [{"id": a, "role": f"Agent-Target-{i}", "allowed_skills": ["Find", "Explore", "PickUp", "Put"]}
                        for i, a in enumerate(r["consumers"], 1)])
    cfg["init_state"]["design_notes"] = [
        f"{len(r['items'])} clutter items relayed source -> ONE buffer surface (capacity 1) -> storage target by "
        f"{len(r['producers'])} producers feeding the buffer and {len(r['consumers'])} consumers draining it.",
        "Second D4 carrier (family J): the single buffer is the coordination chokepoint at any agent count.",
    ]
    cfg["provenance"]["generator"] = "gen_multiagent_scaling.py(J)"
    return cfg


# --------------------------------------------------------------------------- #
# Registry + driver
# --------------------------------------------------------------------------- #

BUILDERS = {
    "H_D1": (gen_h_d1, build_h_d1, "living"),
    "G_D1": (gen_g_d1, build_g_d1, "living"),
    "E_D3": (gen_e_d3, build_e_d3, "living"),
    "I_D3": (gen_i_d3, build_i_d3, "bath"),
    "K_D3": (gen_k_d3, build_k_d3, "kitchen"),
    "C_D2": (gen_c_d2, build_c_d2, "living"),
    "H_D2": (gen_h_d2, build_h_d2, "living"),
    "C_D4": (gen_c_d4, build_c_d4, "living"),
    "J_D4": (gen_c_d4, build_j_d4, "living"),
}


def write_config(config: Dict[str, Any], overwrite: bool) -> Optional[Path]:
    config = L._seed_rewrite(config)
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
    parser = argparse.ArgumentParser(description="Generate 3-4 agent scaling task configs.")
    parser.add_argument("--agents", type=int, required=True, help="agent_count for the generated instances (3 or 4).")
    parser.add_argument("--scenes", nargs="+", required=True, help="FloorPlan ids to attempt.")
    parser.add_argument("--cells", nargs="+", default=sorted(BUILDERS), choices=sorted(BUILDERS))
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="InitialRandomSpawn seed for depth-axis layout randomization (0 = scene default).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--asymmetric", action="store_true",
                        help="K_D3 only: one agent slices 2 items (asymmetric critical section).")
    args = parser.parse_args()
    L.set_spawn_seed(args.seed)

    n = args.agents
    # Asymmetric allocation for K_D3: agent_1 gets 2 items, others get 1 each.
    k_d3_alloc: Optional[List[int]] = None
    if args.asymmetric and n >= 3:
        k_d3_alloc = [2] + [1] * (n - 1)

    controller = Controller(
        agentMode="default", platform=CloudRendering, gpu_device=args.gpu_device,
        scene=args.scenes[0], gridSize=0.25, agentCount=n, width=144, height=144,
        visibilityDistance=1.5,
    )
    written: List[Path] = []
    skipped: List[Tuple[str, str]] = []
    try:
        for scene in args.scenes:
            base = L._candidate_base(controller, scene)
            for cell in args.cells:
                resolver, builder, _ = BUILDERS[cell]
                try:
                    if cell == "K_D3" and k_d3_alloc:
                        resolved = resolver(controller, scene, base, n,
                                            items_per_agent=k_d3_alloc)
                    else:
                        resolved = resolver(controller, scene, base, n)
                except Exception as exc:
                    resolved = None
                    print(f"ERR  {scene} {cell}: {exc}")
                if not resolved:
                    skipped.append((scene, cell))
                    print(f"SKIP {scene}: no verified {cell}__{n}agent assignment")
                    continue
                if cell == "K_D3" and k_d3_alloc:
                    path = write_config(builder(scene, resolved, n,
                                                items_per_agent=k_d3_alloc),
                                        overwrite=args.overwrite)
                else:
                    path = write_config(builder(scene, resolved, n), overwrite=args.overwrite)
                if path is not None:
                    written.append(path)
    finally:
        controller.stop()
    print(f"generated {len(written)} configs ({n}-agent); skipped={len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
