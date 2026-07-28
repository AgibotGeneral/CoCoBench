"""Task-config helpers shared by the env, harness, and runner.

These read the machine-readable ``task_config`` schema (see
``task_config/<family>/<dim>/<task_id>.json``): alias->objectId maps, per-agent
allowed skills, and the action-menu filtering used to present a compact action
space to a decision maker.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from action_space import ActionEntry, MultiAgentActionSpace


def predicate_to_nl(pred: Dict[str, Any], obj_map: Dict[str, str]) -> str:
    """Convert a goal-predicate dict to a natural-language instruction string.

    ``obj_map`` maps goal aliases (e.g. ``"item_1"``, ``"container"``) to
    human-readable object-type names (e.g. ``"KeyChain"``, ``"Drawer"``).  When
    an alias is absent from the map the raw alias is used as a fallback.
    """
    def resolve(alias: str) -> str:
        return obj_map.get(alias, alias).replace("_", " ")

    predicate = pred.get("predicate", "")
    obj = pred.get("object", "")

    if predicate == "on":
        return f"Put the {resolve(obj)} in the {resolve(pred.get('receptacle', ''))}"
    if predicate == "on_sliced_piece":
        return f"Place a slice of {resolve(pred.get('source', ''))} on the {resolve(pred.get('receptacle', ''))}"
    if predicate == "toggled":
        action = "Turn on" if pred.get("value") else "Turn off"
        return f"{action} the {resolve(obj)}"
    if predicate == "filled":
        return f"Fill the {resolve(obj)} with {pred.get('liquid', 'liquid')}"
    if predicate == "empty":
        return f"Empty the {resolve(obj)}"
    if predicate == "closed":
        return f"Close the {resolve(obj)}"
    if predicate == "open":
        return f"Open the {resolve(obj)}"
    if predicate == "clean":
        return f"Clean the {resolve(obj)}"
    if predicate == "sliced":
        return f"Slice the {resolve(obj)}"
    if predicate == "sliced_by":
        agent = pred.get("agent", "")
        return f"{agent} must slice the {resolve(obj)}"
    if predicate == "cooked":
        return f"Cook the {resolve(obj)}"
    return f"{predicate}({resolve(obj)})"


def aliases(config: Dict[str, Any]) -> Dict[str, str]:
    """alias -> real objectId map declared in init_state.objects."""
    return dict(config.get("init_state", {}).get("objects", {}))


def resolve_object(config: Dict[str, Any], name_or_id: str) -> str:
    """Resolve an alias to its objectId; pass through if already an id."""
    return aliases(config).get(name_or_id, name_or_id)


def agent_names(config: Dict[str, Any]) -> Tuple[str, ...]:
    agents = config.get("agents") or []
    if agents:
        return tuple(agent.get("id", f"agent_{idx + 1}") for idx, agent in enumerate(agents))
    return tuple(f"agent_{idx + 1}" for idx in range(int(config.get("agent_count", 1))))


def allowed_skills_for_agent(config: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Per-agent allowed skill sets; falls back to the task-level allowed_skills."""
    default = set(config.get("allowed_skills") or [])
    result: Dict[str, Set[str]] = {}
    for agent in config.get("agents") or []:
        result[agent.get("id", "")] = set(agent.get("allowed_skills") or default)
    return result


def entry_allowed(entry: ActionEntry, allowed: Dict[str, Set[str]]) -> bool:
    skills = allowed.get(entry.agent)
    return not skills or entry.skill in skills


def _parent_receptacles(objects: List[Dict[str, Any]], object_id: str) -> List[str]:
    for obj in objects or []:
        if obj.get("objectId") == object_id:
            return obj.get("parentReceptacles") or []
    return []


def _pc_roles(pc: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    """Resolve the relay's producer/consumer agent sets, accepting either the
    single-agent keys (``producer``/``consumer``, 2-agent D4) or the multi-agent
    fan-in/fan-out lists (``producers``/``consumers``, 3-4 agent D4)."""
    producers = set(pc.get("producers") or [])
    consumers = set(pc.get("consumers") or [])
    if pc.get("producer"):
        producers.add(pc["producer"])
    if pc.get("consumer"):
        consumers.add(pc["consumer"])
    return producers, consumers


def producer_consumer_allows(entry: ActionEntry, pc: Dict[str, Any], objects: List[Dict[str, Any]]) -> bool:
    """Gate one menu entry by a producer→buffer→consumer relay role (D4).

    Enforced at the menu so a fair policy *cannot* bypass the transfer point:
      * producer may ``Put`` only onto the buffer station (cannot reach targets),
      * consumer may ``Put`` only onto the target receptacle(s), and may ``PickUp``
        only objects that are *currently on the buffer* (so it cannot grab at the
        source — forcing it to wait for the producer to deposit).
      * neither role may ``Drop``: dropping on the floor is not part of a relay and
        would strand the transferred object.
    The oracle drives explicit calls and does not use this menu.

    Multi-agent fan-in/fan-out (3-4 agent D4): several producers may share the one
    buffer and several consumers may drain it; the per-role gating is identical, so
    the construct (the buffer is the single chokepoint) is preserved at any count.
    """
    buffer = pc.get("buffer")
    targets = set(pc.get("targets") or [])
    target = entry.args[0] if entry.args else None
    producers, consumers = _pc_roles(pc)
    if entry.agent in producers:
        if entry.skill == "Drop":
            return False
        if entry.skill == "Put":
            return target == buffer
        return True
    if entry.agent in consumers:
        if entry.skill == "Drop":
            return False
        if entry.skill == "Put":
            return target in targets
        if entry.skill == "PickUp":
            return bool(target) and buffer in _parent_receptacles(objects, target)
        return True
    return True


def coordination_hint(config: Dict[str, Any]) -> str:
    """Natural-language statement of machine-readable coordination constraints that
    a fair planner would otherwise never see (they live in ``task_constraints`` and
    are enforced only by silently filtering the menu).

    Currently surfaces the D4 producer→buffer→consumer **relay**: the goal predicate
    is a pure end-state (``on(food, fridge)``) with no hint that a transfer point is
    mandatory, so a planner naturally tries to carry food straight to the
    destination — an action the menu silently removes for the producer, which then
    degrades to dropping on the floor. Making the relay explicit removes that
    information gap (the menu still enforces it as a hard constraint)."""
    tc = config.get("task_constraints", {}) or {}

    # K_D3: shared-knife mutual exclusion -- minimal hint (only the physical
    # constraint; the model must discover the acquire-use-release protocol itself)
    rex = tc.get("resource_exclusion")
    if rex:
        res = rex[0] if isinstance(rex, list) else rex
        return (
            "RESOURCE CONSTRAINT: There is only one shared Knife in the scene. "
            "An agent must be holding the Knife to Slice. "
            "Only one agent can hold it at a time."
        )

    pc = tc.get("producer_consumer")
    if not pc:
        return ""
    rev = {v: k for k, v in aliases(config).items()}
    buf = pc.get("buffer")
    buf_name = rev.get(buf, buf)
    producers, consumers = _pc_roles(pc)
    producer = ", ".join(sorted(producers)) or "the producer"
    consumer = ", ".join(sorted(consumers)) or "the consumer"
    targets = pc.get("targets") or []
    tgt_names = ", ".join(rev.get(t, t) for t in targets) or "the destination"
    return (
        "RELAY CONSTRAINT (mandatory two-stage transport): every target object must "
        f"pass through the transfer point '{buf_name}'. Only {producer} can place objects "
        f"onto '{buf_name}', and {producer} CANNOT place into the final destination. Only "
        f"{consumer} can pick an object up from '{buf_name}' and place it onto {tgt_names}. "
        f"Never drop objects on the floor. So: {producer} repeatedly picks up a target object "
        f"and Puts it on '{buf_name}'; {consumer} waits until an object is on '{buf_name}', "
        f"picks it up, and Puts it on {tgt_names}."
    )


def compact_entries(
    space: MultiAgentActionSpace,
    config: Dict[str, Any],
    full: bool = False,
    objects: List[Dict[str, Any]] | None = None,
) -> List[ActionEntry]:
    """Filter the full action space down to config-allowed skills on the task's
    important objects (aliased ids), keeping argument-free moves always visible.
    Shorter menus reduce impossible actions for a human/VLM decision maker.

    When the config declares a ``task_constraints.producer_consumer`` relay and
    current ``objects`` metadata is supplied, also gate the menu by relay role so
    the transfer point cannot be bypassed (D4)."""
    allowed = allowed_skills_for_agent(config)
    entries = list(space.entries) if full else [entry for entry in space.entries if entry_allowed(entry, allowed)]
    pc = (config.get("task_constraints", {}) or {}).get("producer_consumer")
    if pc and objects is not None and not full:
        entries = [e for e in entries if producer_consumer_allows(e, pc, objects)]
    important_ids = set(aliases(config).values())
    if full or not important_ids:
        return entries
    compact: List[ActionEntry] = []
    for entry in entries:
        if not entry.args or entry.skill in {"Explore", "Wait", "Drop"}:
            compact.append(entry)
        elif entry.args[0] in important_ids:
            compact.append(entry)
    return compact
