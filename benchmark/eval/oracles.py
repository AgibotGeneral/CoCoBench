"""Oracle reference plans, expressed as step-driven generators.

Each plan is a generator that ``yield``s a high-level skill-call string and
receives the resulting :class:`SkillExecutionResult` back (via ``.send``), so the
imperative, feedback-dependent logic of the original scripted oracles is
preserved almost verbatim while fitting the policy-agnostic harness loop. A
:class:`GeneratorPolicy` drives them; the harness executes each yielded call and
records the step.

These are *privileged* reference policies: they know object ids/types and may
query :class:`EnvView`. They serve as the L0 feasibility proof and the
upper-bound baseline. A VLM policy implements the same ``Policy`` interface but
consumes only the :class:`Observation`.
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List

from skill_executor import SkillExecutionResult
from taskutil import aliases

# A plan yields a skill-call string and is sent back the execution result.
Plan = Generator[str, SkillExecutionResult, None]


def _agent_list(env: Any) -> list:
    """Ordered agent ids for this instance (agent_1..agent_N). Reads the declared
    ``agents`` list, falling back to ``agent_count``. N-agent oracles are written
    against this so they scale to any agent_count and stay 2-agent backward compatible."""
    agents = env.config.get("agents") or []
    if agents:
        return [a.get("id", f"agent_{i + 1}") for i, a in enumerate(agents)]
    n = int(env.config.get("agent_count", 2) or 2)
    return [f"agent_{i + 1}" for i in range(n)]


def _skills_map(env: Any) -> Dict[str, set]:
    """agent_id -> its allowed skill set (for resolving heterogeneous roles)."""
    out: Dict[str, set] = {}
    for a in env.config.get("agents") or []:
        out[a.get("id", "")] = set(a.get("allowed_skills") or [])
    return out


def _vacate(agent: str, directions=("back", "left", "right", "back")) -> Plan:
    """Step an agent away from a shared pose so the multi-agent teleport-occupancy
    filter does not block the next agent that needs the same station."""
    for direction in directions:
        yield f"Explore({agent}, {direction})"


def plan_f_d3(env: Any) -> Plan:
    """F×D3 (shared-sink resource exclusion): clean dishes + (optionally) fill the
    kettle, sharing the single Faucet/Sink station; end with the faucet off.

    Scene-adaptive: the dish/kettle/stove targets come from the config aliases, so
    the kettle+stove block runs only when the instance declares a ``kettle`` (some
    kitchens have none)."""
    obj = aliases(env.config)
    yield f"Find(agent_1, {obj['faucet']})"
    yield f"ToggleOn(agent_1, {obj['faucet']})"
    yield f"Find(agent_1, {obj['mug']})"
    yield f"CleanObject(agent_1, {obj['mug']})"
    yield f"PickUp(agent_1, {obj['mug']})"
    yield f"Find(agent_1, {obj['clean_zone']})"
    yield f"Put(agent_1, {obj['clean_zone']})"
    yield f"Find(agent_2, {obj['bowl']})"
    yield f"CleanObject(agent_2, {obj['bowl']})"
    yield f"PickUp(agent_2, {obj['bowl']})"
    yield f"Find(agent_2, {obj.get('clean_zone_2', obj['clean_zone'])})"
    yield f"Put(agent_2, {obj.get('clean_zone_2', obj['clean_zone'])})"
    if obj.get("kettle") and obj.get("stove_burner"):
        yield f"Find(agent_1, {obj['kettle']})"
        yield f"FillObjectWithLiquid(agent_1, {obj['kettle']}, water)"
        yield f"PickUp(agent_1, {obj['kettle']})"
        yield f"Find(agent_1, {obj['stove_burner']})"
        yield f"Put(agent_1, {obj['stove_burner']})"
    yield f"Find(agent_1, {obj['faucet']})"
    yield f"ToggleOff(agent_1, {obj['faucet']})"


def _plate_one_piece(env: Any, source_alias: str, receptacle_alias: str, agent: str = "agent_2") -> Plan:
    obj = aliases(env.config)
    for piece in env.view().sliced_pieces(obj[source_alias]):
        result = yield f"Find({agent}, {piece})"
        if not result.success:
            continue
        result = yield f"PickUp({agent}, {piece})"
        if not result.success:
            continue
        yield f"Find({agent}, {obj[receptacle_alias]})"
        yield f"Put({agent}, {obj[receptacle_alias]})"
        return
    print(f"[b_d2] no reachable sliced piece for {source_alias}")


def plan_b_d2(env: Any) -> Plan:
    """B×D2 (cooking / sequential dependency). Agent-Prep slices then vacates so
    Agent-CookPlate can reach the scattered pieces without a teleport-occupancy
    conflict; Agent-CookPlate cooks the potato in the Pot on a knob-lit burner and
    plates the salad pieces + cooked potato."""
    obj = aliases(env.config)

    # 1) Prep (agent_1): slice tomato and lettuce, then vacate the prep spot so
    #    agent_2 can reach the pieces (multi-agent teleport-occupancy filter).
    yield f"Find(agent_1, {obj['tomato']})"
    yield f"Slice(agent_1, {obj['tomato']})"
    yield f"Find(agent_1, {obj['lettuce']})"
    yield f"Slice(agent_1, {obj['lettuce']})"
    for direction in ("back", "back", "left", "right"):
        yield f"Explore(agent_1, {direction})"

    # 2) Cook (agent_2): put potato into the Pot on the burner, light the stove.
    yield f"Find(agent_2, {obj['potato']})"
    yield f"PickUp(agent_2, {obj['potato']})"
    yield f"Find(agent_2, {obj['pot']})"
    yield f"Put(agent_2, {obj['pot']})"
    for knob in env.view().object_ids_of_type("StoveKnob"):
        result = yield f"Find(agent_2, {knob})"
        if result.success:
            yield f"ToggleOn(agent_2, {knob})"
    env.advance_physics(5)

    # 3) Plate (agent_2): place a reachable tomato piece in the bowl and the
    #    cooked potato on the plate. Keeping them separate avoids a plate
    #    capacity/collision failure in AI2-THOR 5.0.
    yield from _plate_one_piece(env, "tomato", "bowl")
    yield f"Find(agent_2, {obj['potato']})"
    yield f"PickUp(agent_2, {obj['potato']})"
    yield f"Find(agent_2, {obj['plate']})"
    yield f"Put(agent_2, {obj['plate']})"


def plan_a_d1(env: Any) -> Plan:
    """A×D1 (tidy / independent-parallel): two **balanced** independent lines so the
    schedule is load-optimal (the D1 reference must minimise makespan, not pile the
    work on one agent). agent_1 opens the fridge and stores the potato, then vacates
    the fridge pose; agent_2 turns the light off and stores the lettuce. The fridge
    door is left open (closing a loaded fridge is physically blocked in this build)."""
    obj = aliases(env.config)

    # agent_1: open fridge -> store potato -> step away so agent_2 can reach the fridge
    # (multi-agent teleport-occupancy filter blocks a second agent at the same pose).
    yield f"Find(agent_1, {obj['fridge']})"
    yield f"Open(agent_1, {obj['fridge']})"
    yield f"Find(agent_1, {obj['potato']})"
    yield f"PickUp(agent_1, {obj['potato']})"
    yield f"Find(agent_1, {obj['fridge']})"
    yield f"Put(agent_1, {obj['fridge']})"
    yield "Explore(agent_1, back)"

    # agent_2: light off (independent) -> store lettuce into the now-open fridge.
    yield f"Find(agent_2, {obj['light']})"
    yield f"ToggleOff(agent_2, {obj['light']})"
    yield f"Find(agent_2, {obj['lettuce']})"
    yield f"PickUp(agent_2, {obj['lettuce']})"
    yield f"Find(agent_2, {obj['fridge']})"
    yield f"Put(agent_2, {obj['fridge']})"


def plan_g_d1(env: Any) -> Plan:
    """G×D1 (living-room parallel sorting): N independent deposit lines.

    This is a semantic expansion of D1 beyond kitchen storage. Each agent moves
    one small object to its own target receptacle/surface; no ordering relation
    exists between the lines. Generalizes to any agent_count: one
    (agent_i, item_i, target_i) line per declared item (2-agent backward compatible).
    """
    obj = aliases(env.config)
    i = 1
    while f"item_{i}" in obj and f"target_{i}" in obj:
        agent = f"agent_{i}"
        item = obj[f"item_{i}"]
        target = obj[f"target_{i}"]
        yield f"Find({agent}, {item})"
        yield f"PickUp({agent}, {item})"
        yield f"Find({agent}, {target})"
        yield f"Put({agent}, {target})"
        i += 1



def plan_h_d1(env: Any) -> Plan:
    """H×D1 (bedroom independent storage): N agents each open their OWN drawer and
    store one personal item into it. The lines are fully independent — there is no
    cross-agent ordering — and the per-agent open-before-put is a local precondition,
    not a coordination dependency. Action counts are symmetric across agents.
    Generalizes to any agent_count:
    one (agent_i, item_i, drawer_i) line per declared item."""
    obj = aliases(env.config)
    i = 1
    while f"item_{i}" in obj and f"drawer_{i}" in obj:
        agent = f"agent_{i}"
        drawer = obj[f"drawer_{i}"]
        item = obj[f"item_{i}"]
        yield f"Find({agent}, {drawer})"
        yield f"Open({agent}, {drawer})"
        yield f"Find({agent}, {item})"
        yield f"PickUp({agent}, {item})"
        yield f"Find({agent}, {drawer})"
        yield f"Put({agent}, {drawer})"
        i += 1


def _collect(agent: str, item: str, basket: str, obj: Dict[str, str]) -> Plan:
    yield f"Find({agent}, {obj[item]})"
    yield f"PickUp({agent}, {obj[item]})"
    yield f"Find({agent}, {obj[basket]})"
    yield f"Put({agent}, {obj[basket]})"


def plan_e_d3(env: Any) -> Plan:
    """E×D3 (competitive collection / resource exclusion): an L0 feasibility plan
    that deposits a fixed assignment into each agent's own basket. The
    competitive/exclusion dynamic (one object, one winner) is an L1 metric, not an
    L0 goal. Generalizes to any agent_count: one (agent_i, item_i, basket_i) line.

    Two schemas are supported: the legacy 2-agent one (keychain→basket_1,
    watch→basket_2, with ``open_basket_2`` governing basket_2's Open) and the
    N-agent one (item_i→basket_i, with ``task_constraints.open_baskets`` listing the
    indices whose basket must be opened first)."""
    obj = aliases(env.config)
    tc = env.config.get("task_constraints", {}) or {}
    pairs = []  # (item_objectId, basket_objectId, index)
    if "item_1" in obj:  # N-agent schema
        i = 1
        while f"item_{i}" in obj and f"basket_{i}" in obj:
            pairs.append((obj[f"item_{i}"], obj[f"basket_{i}"], i))
            i += 1
    else:  # legacy 2-agent schema
        if "keychain" in obj and "basket_1" in obj:
            pairs.append((obj["keychain"], obj["basket_1"], 1))
        if "watch" in obj and "basket_2" in obj:
            pairs.append((obj["watch"], obj["basket_2"], 2))
    open_baskets = set(tc.get("open_baskets") or [])
    if "open_baskets" not in tc and tc.get("open_basket_2", True) and any(idx == 2 for _, _, idx in pairs):
        open_baskets.add(2)  # legacy 2-agent default: open the (openable) basket_2
    for item, basket, idx in pairs:
        agent = f"agent_{idx}"
        if idx in open_baskets:
            yield f"Find({agent}, {basket})"
            yield f"Open({agent}, {basket})"
        yield f"Find({agent}, {item})"
        yield f"PickUp({agent}, {item})"
        yield f"Find({agent}, {basket})"
        yield f"Put({agent}, {basket})"




def plan_c_d4(env: Any) -> Plan:
    """C×D4 (relay transport / producer-consumer). Multi-object: each food goes
    source → transfer Plate (capacity-limited buffer) → target (open Fridge).
    Producer and consumer strictly alternate per item so the buffer holds ≤1 at a
    time (never overflows) and both roles contribute; each vacates after acting so
    the multi-agent teleport-occupancy filter does not block the other at the same
    pose. Backward compatible with the single-object schema (item/potato → pot).

    Generalizes to any agent_count via fan-in/fan-out: ``task_constraints.
    producer_consumer`` may declare ``producers``/``consumers`` lists (N agents).
    Foods are dealt round-robin across producers and consumers, but each food still
    strictly alternates produce→consume so the single shared buffer (the D4
    chokepoint) holds ≤1 item at a time regardless of how many agents share it."""
    obj = aliases(env.config)
    foods = _ordered_items(obj) or [obj[k] for k in sorted(obj) if k.startswith("food")]
    if not foods:
        foods = [obj.get("item") or obj.get("potato")]
    transfer = obj.get("transfer_plate") or obj.get("transfer")
    target = obj.get("fridge") or obj.get("target_pot") or obj.get("target")
    pc = (env.config.get("task_constraints", {}) or {}).get("producer_consumer", {}) or {}
    producers = list(pc.get("producers") or []) or ([pc["producer"]] if pc.get("producer") else ["agent_1"])
    consumers = list(pc.get("consumers") or []) or ([pc["consumer"]] if pc.get("consumer") else ["agent_2"])
    # If the target is the (openable) Fridge, a consumer opens it first (the init
    # OpenObject cannot run without an agent positioned at it). Left open: a loaded
    # fridge will not reliably re-close, and the goal does not require it.
    if obj.get("fridge"):
        yield f"Find({consumers[0]}, {target})"
        yield f"Open({consumers[0]}, {target})"
        yield from _vacate(consumers[0], ("back", "left"))
    for idx, food in enumerate(foods):
        producer = producers[idx % len(producers)]
        consumer = consumers[idx % len(consumers)]
        # Producer: source -> transfer point, then vacate.
        yield f"Find({producer}, {food})"
        yield f"PickUp({producer}, {food})"
        yield f"Find({producer}, {transfer})"
        yield f"Put({producer}, {transfer})"
        yield from _vacate(producer)
        # Consumer: transfer point -> target, then vacate for the next round.
        yield f"Find({consumer}, {food})"
        yield f"PickUp({consumer}, {food})"
        yield f"Find({consumer}, {target})"
        yield f"Put({consumer}, {target})"
        yield from _vacate(consumer)


def _ordered_items(obj: Dict[str, str]):
    """item_1, item_2, ... in declared order (shared by the container-loading cells)."""
    return [obj[k] for k in sorted(obj) if k.startswith("item_")]


def plan_c_d2(env: Any) -> Plan:
    """C×D2 (sequential dependency: ordered container loading). Homogeneous agents
    share an engine-enforced order: the container must be Opened before any Put (a
    closed container rejects placement) and Closed only after every item is inside
    (closing first would block the remaining Puts). agent_1 opens + loads the first
    item; the remaining agents load the rest round-robin; the last loader Closes —
    a cross-agent precedence chain that lengthens with the item count.

    Generalizes to any agent_count: with N agents and N items, agent_1 opens and
    loads item_1, agent_i loads item_i, and the last agent Closes (2-agent backward
    compatible). Each loader vacates so the shared container pose is never blocked."""
    obj = aliases(env.config)
    container = obj["container"]
    items = _ordered_items(obj)
    agents = _agent_list(env)
    n = len(agents)
    # agent_1: open (precedence root), then load the first item, then vacate.
    yield f"Find({agents[0]}, {container})"
    yield f"Open({agents[0]}, {container})"
    last_loader = agents[0]
    for idx, item in enumerate(items):
        agent = agents[idx % n]
        yield f"Find({agent}, {item})"
        yield f"PickUp({agent}, {item})"
        yield f"Find({agent}, {container})"
        yield f"Put({agent}, {container})"
        yield from _vacate(agent, ("back", "left"))
        last_loader = agent
    # Close last (precedence: all Puts before Close), by whichever agent loaded last.
    yield f"Find({last_loader}, {container})"
    yield f"Close({last_loader}, {container})"


def plan_i_d3(env: Any) -> Plan:
    """I×D3 (bathroom shared-station resource exclusion): two agents each retrieve a
    scattered toiletry and deposit it at the single shared Sink station, gated by the
    one Faucet. iTHOR bathrooms expose no fillable container and only one dirtyable
    pickup (Cloth), so the kitchen wet-cleaning mechanic (F) cannot be ported; the
    contention here is the single shared station/Faucet that both deposit lines must
    pass through. The D3 signal penalizes duplicate claims over the shared
    resource."""
    obj = aliases(env.config)
    faucet = obj["faucet"]
    station = obj["station"]
    yield f"Find(agent_1, {faucet})"
    yield f"ToggleOn(agent_1, {faucet})"
    for direction in ("back", "left"):
        yield f"Explore(agent_1, {direction})"
    i = 1
    while f"item_{i}" in obj:
        agent = f"agent_{i}"
        yield f"Find({agent}, {obj[f'item_{i}']})"
        yield f"PickUp({agent}, {obj[f'item_{i}']})"
        yield f"Find({agent}, {station})"
        yield f"Put({agent}, {station})"
        for direction in ("back", "left", "right", "back"):
            yield f"Explore({agent}, {direction})"
        i += 1
    yield f"Find(agent_1, {faucet})"
    yield f"ToggleOff(agent_1, {faucet})"


def plan_k_d3(env: Any) -> Plan:
    """K×D3 (shared-Knife slicing exclusion): N agents each slice their assigned
    food items using a single shared Knife.  PickUp mutual exclusion is
    engine-enforced (only one agent can hold the Knife), so agents must take
    turns: acquire Knife, slice food(s), drop Knife, vacate for the next agent.

    Supports asymmetric allocation via the ``assignment`` map in the config:
    when present, some agents may slice more than one item per turn."""
    obj = aliases(env.config)
    knife = obj["knife"]
    assignment = env.config.get("assignment")
    if assignment:
        from collections import defaultdict
        agent_items: Dict[str, List[str]] = defaultdict(list)
        for item_alias, agent_name in assignment.items():
            agent_items[agent_name].append(item_alias)
        for agent in sorted(agent_items.keys()):
            items = agent_items[agent]
            yield f"Find({agent}, {knife})"
            yield f"PickUp({agent}, {knife})"
            for item_alias in items:
                yield f"Find({agent}, {obj[item_alias]})"
                yield f"Slice({agent}, {obj[item_alias]})"
            yield f"Drop({agent})"
            yield from _vacate(agent)
    else:
        i = 1
        while f"item_{i}" in obj:
            agent = f"agent_{i}"
            yield f"Find({agent}, {knife})"
            yield f"PickUp({agent}, {knife})"
            yield f"Find({agent}, {obj[f'item_{i}']})"
            yield f"Slice({agent}, {obj[f'item_{i}']})"
            yield f"Drop({agent})"
            yield from _vacate(agent)
            i += 1


# Registry of built-in oracle plans, keyed by "<family>_<dim>".
ORACLE_PLANS = {
    "A_D1": plan_a_d1,
    # Kitchen-native family A also carries D2 via Cabinet ordered loading (a third,
    # non-C D2 carrier alongside C/H — alias-driven, so the kitchen Cabinet instances
    # drive plan_c_d2 unchanged; added to dilute C-family concentration).
    "A_D2": plan_c_d2,
    "B_D2": plan_b_d2,
    "C_D2": plan_c_d2,
    "C_D4": plan_c_d4,
    "E_D3": plan_e_d3,
    "F_D3": plan_f_d3,
    "G_D1": plan_g_d1,
    # Bedroom-native family H: D1 uses independent personal-drawer storage; D2
    # shares the alias-driven container-loading plan with C.
    "H_D1": plan_h_d1,
    "H_D2": plan_c_d2,
    # Bathroom-native family I: shared single Sink/Faucet station (resource exclusion).
    "I_D3": plan_i_d3,
    # Kitchen-native family K: shared Knife slicing (PickUp mutual exclusion).
    "K_D3": plan_k_d3,
    # Living/bedroom relay family J: cross-zone handoff (D4 second carrier alongside
    # C). Clutter relayed source -> capacity-limited buffer table -> storage target.
    # Alias-driven: plan_c_d4 is family-agnostic (reads producer_consumer + aliases;
    # skips the Fridge-open step when the target is not a Fridge).
    "J_D4": plan_c_d4,
}


def oracle_plan_for_config(config: Dict[str, Any]):
    key = f"{config.get('task_family')}_{config.get('coordination_dim')}"
    return ORACLE_PLANS.get(key)
