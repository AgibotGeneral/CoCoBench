"""Action-id surface for the multi-agent AI2-THOR benchmark.

The executor accepts explicit skill calls such as ``PickUp(agent_1, Mug|...)``.
This module builds an EB-ALFRED-style discrete action list for a decision maker:
models or humans choose an integer action id, and the runner resolves it to a
skill call with concrete object ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import thormeta


CORE_DIRECTIONS = ("forward", "back", "left", "right", "turn_left", "turn_right", "look_up", "look_down")

# Affordance flags that make an object worth surfacing as an interaction target.
INTERACTABLE_AFFORDANCE_KEYS = (
    "pickupable",
    "receptacle",
    "openable",
    "toggleable",
    "sliceable",
    "dirtyable",
    "canFillWithLiquid",
    "breakable",
    "moveable",
)


@dataclass(frozen=True)
class ActionEntry:
    action_id: int
    action_name: str
    skill: str
    agent: str
    args: List[str]

    @property
    def call(self) -> str:
        return f"{self.skill}({', '.join([self.agent] + self.args)})"


@dataclass(frozen=True)
class ObjectAlias:
    alias: str
    object_id: str
    object_type: str
    affordances: Dict[str, bool]


class MultiAgentActionSpace:
    """Build scene-specific action ids from AI2-THOR metadata.

    The action list is intentionally scene-specific rather than global-static:
    for a first task-collection/debug loop, shorter action lists are easier for a
    human decision maker and reduce impossible actions. Object aliases are still
    stable within one reset because they are stored in ``name_to_id``.
    """

    def __init__(self, agent_names: Sequence[str] = ("agent_1", "agent_2")) -> None:
        self.agent_names = list(agent_names)
        self.entries: List[ActionEntry] = []
        self.name_to_id: Dict[str, str] = {}
        self.id_to_name: Dict[str, str] = {}

    @classmethod
    def from_event(cls, event: Any, agent_names: Sequence[str] = ("agent_1", "agent_2")) -> "MultiAgentActionSpace":
        space = cls(agent_names=agent_names)
        metadata = cls._metadata(event)
        aliases = space._build_object_aliases(metadata.get("objects", []))
        space.entries = space._build_entries(aliases)
        return space

    @classmethod
    def from_controller(cls, controller: Any, agent_names: Sequence[str] = ("agent_1", "agent_2")) -> "MultiAgentActionSpace":
        return cls.from_event(getattr(controller, "last_event", None), agent_names=agent_names)

    def resolve(self, action_id: int) -> ActionEntry:
        if action_id < 0 or action_id >= len(self.entries):
            raise IndexError(f"Invalid action id {action_id}; valid range is 0..{len(self.entries) - 1}")
        return self.entries[action_id]

    def _build_entries(self, aliases: Sequence[ObjectAlias]) -> List[ActionEntry]:
        entries: List[ActionEntry] = []

        def add(agent: str, name: str, skill: str, args: List[str]) -> None:
            entries.append(ActionEntry(len(entries), f"{agent}: {name}", skill, agent, args))

        navigable = [a for a in aliases if self._is_interactable(a)]
        pickupable = [a for a in aliases if a.affordances.get("pickupable")]
        receptacles = [a for a in aliases if a.affordances.get("receptacle")]
        openable = [a for a in aliases if a.affordances.get("openable")]
        toggleable = [a for a in aliases if a.affordances.get("toggleable")]
        sliceable = [a for a in aliases if a.affordances.get("sliceable")]
        dirtyable = [a for a in aliases if a.affordances.get("dirtyable")]
        fillable = [a for a in aliases if a.affordances.get("canFillWithLiquid")]
        breakable = [a for a in aliases if a.affordances.get("breakable")]
        physical = [a for a in aliases if a.affordances.get("pickupable") or a.affordances.get("moveable")]

        for agent in self.agent_names:
            for alias in navigable:
                add(agent, f"find a {alias.alias}", "Find", [alias.object_id])
            for direction in CORE_DIRECTIONS:
                add(agent, f"explore {direction}", "Explore", [direction])
            add(agent, "wait", "Wait", [])
            for alias in pickupable:
                add(agent, f"pick up the {alias.alias}", "PickUp", [alias.object_id])
            for alias in receptacles:
                add(agent, f"put object in hand on the {alias.alias}", "Put", [alias.object_id])
            add(agent, "drop object in hand", "Drop", [])
            for alias in openable:
                add(agent, f"open the {alias.alias}", "Open", [alias.object_id])
                add(agent, f"close the {alias.alias}", "Close", [alias.object_id])
            for alias in toggleable:
                add(agent, f"turn on the {alias.alias}", "ToggleOn", [alias.object_id])
                add(agent, f"turn off the {alias.alias}", "ToggleOff", [alias.object_id])
            for alias in sliceable:
                add(agent, f"slice the {alias.alias}", "Slice", [alias.object_id])
            for alias in dirtyable:
                add(agent, f"clean the {alias.alias}", "CleanObject", [alias.object_id])
            for alias in fillable:
                add(agent, f"fill the {alias.alias} with water", "FillObjectWithLiquid", [alias.object_id, "water"])
                add(agent, f"empty liquid from the {alias.alias}", "EmptyLiquidFromObject", [alias.object_id])
            for alias in physical:
                add(agent, f"push the {alias.alias}", "PushObject", [alias.object_id])
                add(agent, f"pull the {alias.alias}", "PullObject", [alias.object_id])
            for alias in breakable:
                add(agent, f"break the {alias.alias}", "BreakObject", [alias.object_id])
        return entries

    def _build_object_aliases(self, objects: Iterable[Dict[str, Any]]) -> List[ObjectAlias]:
        object_list = list(objects)
        sliced_source_ids = self._sliced_source_object_ids(object_list)
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for obj in object_list:
            object_id = obj.get("objectId")
            object_type = obj.get("objectType") or (object_id.split("|")[0] if object_id else None)
            if not object_id or not object_type:
                continue
            if object_id in sliced_source_ids:
                continue
            if not self._is_interactable_obj(obj):
                continue
            grouped.setdefault(object_type, []).append(obj)

        aliases: List[ObjectAlias] = []
        for object_type in sorted(grouped):
            objs = sorted(grouped[object_type], key=self._alias_sort_key)
            for index, obj in enumerate(objs):
                alias = object_type if index == 0 else f"{object_type}_{index + 1}"
                object_id = obj["objectId"]
                self.name_to_id[alias] = object_id
                self.id_to_name[object_id] = alias
                aliases.append(ObjectAlias(alias, object_id, object_type, self._affordances(obj)))
        return aliases


    @staticmethod
    def _alias_sort_key(obj: Dict[str, Any]) -> tuple[int, str]:
        return (0 if obj.get("visible") else 1, obj.get("objectId", ""))

    @staticmethod
    def _sliced_source_object_ids(objects: Sequence[Dict[str, Any]]) -> set[str]:
        """Return original object ids that have visible sliced-piece children.

        AI2-THOR keeps the original object in metadata after SliceObject, but it
        may become invisible while new pickupable AppleSliced/TomatoSliced/etc.
        objects appear with ids prefixed by the original id. Keeping both in the
        action space makes the decision maker see impossible PickUp actions for
        the invisible source object, so the source is hidden once its pieces are
        present.
        """
        object_ids = {obj.get("objectId") for obj in objects if obj.get("objectId")}
        sources: set[str] = set()
        for obj in objects:
            object_id = obj.get("objectId")
            object_type = obj.get("objectType") or ""
            if not object_id or not object_type.endswith("Sliced"):
                continue
            source_id = object_id.rsplit("|", 1)[0]
            if source_id in object_ids:
                sources.add(source_id)
        return sources

    @staticmethod
    def _metadata(event: Any) -> Dict[str, Any]:
        return thormeta.metadata_for_agent(event, 0)

    @staticmethod
    def _affordances(obj: Dict[str, Any]) -> Dict[str, bool]:
        return {key: bool(obj.get(key)) for key in INTERACTABLE_AFFORDANCE_KEYS}

    @staticmethod
    def _is_interactable(alias: ObjectAlias) -> bool:
        return any(alias.affordances.values())

    @staticmethod
    def _is_interactable_obj(obj: Dict[str, Any]) -> bool:
        return any(bool(obj.get(key)) for key in INTERACTABLE_AFFORDANCE_KEYS)
