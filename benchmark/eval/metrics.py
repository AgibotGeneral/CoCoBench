"""Coordination metrics over a recorded ``trajectory.json`` + task config.

Pure functions; the recorded trajectory (see :mod:`recorder`) is the single
source of truth — **no controller needed**. Consumes each step's
``skill_result`` (``{skill, agent, success, errorMessage}``) plus the config's
``agents[].allowed_skills`` and (optional, machine-readable)
``task_constraints``.

Two layers of output:

* **General coordination metrics**: subgoal success,
  legal-plan, makespan / load imbalance, coordination overhead, dependency /
  occupancy / affordance violation counts, safety violations.
* **Per-dimension construct score** ∈[0,1] (higher = better coordination) for
  the config's own ``coordination_dim``,
  computed when the needed signal is available, else ``None`` (NA).

Failure classification of a *failed* state-changing step:
  - ``occupancy_conflict``  — errorMessage names an unoccupied-pose failure (D3)
  - ``affordance_miss``     — errorMessage names a "No valid positions" Put failure (D4/affordance)
  - ``precondition_violation`` — any other interaction failure = acted before its
    precondition held (the execution-grounded D2 signal)

These are events that a correct reference plan should not trigger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- errorMessage signatures (substring match, case-insensitive) -------------
OCCUPANCY_MARKERS = ("unoccupied pose", "could not navigate to a visible")
AFFORDANCE_MARKERS = ("no valid positions",)
# Benign no-ops: the action failed only because its effect already held (the goal
# was already satisfied). These are redundant/wasted actions, NOT precondition or
# dependency violations, so they must not flip ``legal_plan`` or count as a
# dependency violation. (e.g. toggling off an already-off light on a successful run.)
NOOP_MARKERS = (
    "already off", "already on", "already open", "already close",
    "already satisfied", "already empty", "already filled", "already clean",
    "already sliced", "already cooked",
)
# Intrinsic action preconditions — universal to the action's own semantics, NOT a
# violation of the task's *declared* precedence. e.g. Put with an empty hand
# (PickUp-before-Put is intrinsic to Put), or PickUp with a full hand. These are
# plan-coherence errors that a single agent can make with no ordering constraint at
# all, so they must not count toward the D2 precedence construct.
INTRINSIC_PRECOND_MARKERS = (
    "is not holding any object",
    "hand has something in it already",
    "can't pick up anything",
    "is not pickupable",
)
# Multi-agent concurrency conflicts — a second agent claims an object that is already
# held by another.  This is a world-state constraint (one holder per object at a time),
# NOT a violation of the task's *declared* precedence ordering (Open→Put→Close).
# These collisions are tracked separately so they do not inflate the D2 penalty.
CONCURRENCY_MARKERS = (
    "is already held by another",
    "already picked up by",
)
INTERACTION_SKILLS = {
    "PickUp", "Put", "Drop", "Open", "Close", "ToggleOn", "ToggleOff",
    "Slice", "CleanObject", "FillObjectWithLiquid", "EmptyLiquidFromObject",
    "PushObject", "PullObject", "BreakObject",
}

def _has(marker_set, text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in marker_set)


def _allowed_skills_by_agent(config: Dict[str, Any]) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for agent in config.get("agents", []):
        out[agent.get("id")] = set(agent.get("allowed_skills", []))
    return out


def classify_step(skill_result: Dict[str, Any], allowed: Dict[str, set]) -> str:
    """Map one executed skill step to a coordination-event type."""
    skill = skill_result.get("skill", "")
    agent = skill_result.get("agent", "")
    success = bool(skill_result.get("success"))
    err = skill_result.get("errorMessage", "") or ""

    if agent in allowed and skill not in allowed[agent]:
        return "illegal_skill"
    if skill == "Wait":
        return "idle_wait"
    if success:
        return "progress"
    if _has(NOOP_MARKERS, err):
        return "redundant_noop"          # already-satisfied no-op: wasted, not illegal
    if _has(OCCUPANCY_MARKERS, err):
        return "occupancy_conflict"
    if _has(AFFORDANCE_MARKERS, err):
        return "affordance_miss"
    if _has(CONCURRENCY_MARKERS, err):
        return "concurrency_conflict"
    if skill in INTERACTION_SKILLS:
        return "precondition_violation"
    return "action_failure"


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _target_id(skill_result: Dict[str, Any]) -> Optional[str]:
    """The objectId a step acts on: receptacle for Put, else the object."""
    args = skill_result.get("args", {}) or {}
    return args.get("receptacleId") or args.get("objectId")


def compute_metrics(report: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute coordination metrics for one recorded trajectory report."""
    config = config or report.get("task_config", {})
    allowed = _allowed_skills_by_agent(config)
    dim = config.get("coordination_dim")
    constraints = config.get("task_constraints", {}) or {}

    traj_steps = [s for s in report.get("trajectory", []) if s.get("skill_result")]
    steps = [s["skill_result"] for s in traj_steps]   # aligned 1:1 with traj_steps
    events = [classify_step(sr, allowed) for sr in steps]

    # per-agent action accounting -> makespan / load imbalance
    per_agent: Dict[str, int] = {}
    for sr in steps:
        per_agent[sr.get("agent", "?")] = per_agent.get(sr.get("agent", "?"), 0) + 1
    counts = list(per_agent.values()) or [0]
    makespan = max(counts)                       # parallel-rounds model (turn-based sim)
    load_imbalance = max(counts) - min(counts)

    # event tallies
    def n(ev: str) -> int:
        return sum(1 for e in events if e == ev)

    illegal = n("illegal_skill")
    occupancy = n("occupancy_conflict")
    affordance_miss = n("affordance_miss")
    precond_viol = n("precondition_violation")
    concurrency = n("concurrency_conflict")
    idle_wait = n("idle_wait")
    other_fail = n("action_failure")
    failed_total = sum(1 for sr in steps if not sr.get("success"))

    # affordance accuracy = Put once-correct / (Put correct + Put compat-failures)
    put_attempts = [sr for sr in steps if sr.get("skill") == "Put"]
    put_ok = sum(1 for sr in put_attempts if sr.get("success"))
    put_compat_fail = sum(1 for sr in put_attempts if _has(AFFORDANCE_MARKERS, sr.get("errorMessage", "")))
    affordance_acc = (put_ok / (put_ok + put_compat_fail)) if (put_ok + put_compat_fail) else None

    # safety: Break on a declared fragile object
    fragile = set(constraints.get("fragile_objects", []) or [])
    safety_viol = sum(
        1 for sr in steps
        if sr.get("skill") == "BreakObject" and (sr.get("args", {}) or {}).get("objectId") in fragile
    )

    # subgoal success from final eval
    final_eval = report.get("final_eval", {}) or {}
    checks = final_eval.get("checks", []) or []
    subgoal_rate = (sum(1 for c in checks if c.get("passed")) / len(checks)) if checks else None

    coordination_overhead = idle_wait + failed_total  # idle + wasted actions
    # legal_plan is the absence of illegal-skill / precondition violations, but it
    # is only meaningful once the policy has actually acted. A 0-action episode
    # (immediate parse/backend error, empty plan) has no plan to judge -> NA, so it
    # does not score a vacuous "perfect legal plan".
    legal_plan = None if not steps else ((illegal == 0) and (precond_viol == 0))

    metrics: Dict[str, Any] = {
        "task_id": config.get("task_id"),
        "coordination_dim": dim,
        "success": bool(final_eval.get("success")),
        "subgoal_success_rate": subgoal_rate,
        "legal_plan": legal_plan,
        "n_action_steps": len(steps),
        "n_agents": len(allowed) or int(config.get("agent_count", 2) or 2),
        "makespan": makespan,
        "load_imbalance": load_imbalance,
        "per_agent_actions": per_agent,
        "coordination_overhead": coordination_overhead,
        "wait_steps": idle_wait,
        "failed_steps": failed_total,
        "illegal_skill": illegal,
        "dependency_violations": precond_viol,      # execution-grounded D2 signal
        "concurrency_conflicts": concurrency,        # inter-agent object-hold contention (not D2)
        "occupancy_conflicts": occupancy,           # D3 signal
        "affordance_failures": affordance_miss,
        "affordance_accuracy": affordance_acc,
        "safety_violations": safety_viol,
        "event_histogram": {ev: events.count(ev) for ev in sorted(set(events))},
    }
    score, status, sample = _construct_eval(dim, metrics, steps, constraints, traj_steps, config)
    metrics["construct_score"] = score
    metrics["construct_status"] = status        # scored | not_engaged | low_signal | na_*
    metrics["construct_sample"] = sample        # # of mechanism-engaging events the score rests on
    metrics["distributed"] = _distributed_metrics(traj_steps, steps, events)
    return metrics


# Minimum number of mechanism-engaging events required to report a construct score.
# A rate estimated from a single event (e.g. one clean buffer transfer in a relay
# the policy otherwise never completed) is not evidence of good coordination — it
# is insufficient sample. Below this floor the score is withheld (status set), so a
# task the policy barely attempted can never masquerade as perfect coordination.
CONSTRUCT_MIN_SAMPLE = 2


def _dim_sample(dim: str, m: Dict[str, Any], steps: List[Dict[str, Any]], constraints: Dict[str, Any]) -> int:
    """How many times this dimension's coordination mechanism was actually exercised
    — the sample the construct score is computed over (its formula's denominator /
    evidence count). 0 ⇒ the mechanism was never touched; small ⇒ thin evidence."""
    inter_ok = sum(1 for sr in steps if sr.get("success") and sr.get("skill") in INTERACTION_SKILLS)
    if dim == "D1":  # load balancing: evidence = actions actually scheduled
        return sum((m.get("per_agent_actions") or {}).values())
    if dim == "D2":  # sequencing: evidence = executed task progress
        return inter_ok
    if dim == "D3":
        lock_skills: set = set()
        for r in (constraints.get("resource_exclusion") or []):
            lock_skills |= set(r.get("skills_requiring_lock") or [])
        station = bool(constraints.get("exclusive_resources")) or bool(lock_skills)

        # Shared-tool path (K family): resource_exclusion declares the shared
        # tool's objectId.  Evidence = PickUp attempts targeting it (the engine
        # enforces one-holder-at-a-time via held_by_other_agent()).
        resource_ids: set = set()
        for r in (constraints.get("resource_exclusion") or []):
            rid = r.get("resource_id")
            if rid:
                resource_ids.add(rid)
        if resource_ids:
            return sum(1 for sr in steps
                       if sr.get("skill") == "PickUp" and _target_id(sr) in resource_ids)

        if station:  # shared-station (F): evidence = lock-requiring skill attempts
            if not lock_skills:
                lock_skills = {"CleanObject", "FillObjectWithLiquid", "EmptyLiquidFromObject"}
            return sum(1 for sr in steps if sr.get("skill") in lock_skills)
        # Competitive-object path (E and I families).
        #
        # E family: agents race to claim the same scarce basket items.
        # I family: agents deposit toiletries to a shared Sink/Bathtub station.
        #   Although I tasks declare ``resource_exclusion.resource = "shared_sink_station"``,
        #   they carry no ``skills_requiring_lock`` and no ``exclusive_resources`` key, so
        #   station_mode is False and both families land here.  This is intentional: the
        #   engine does NOT deny concurrent Put to Sink/Bathtub (the receptacle accepts
        #   multiple simultaneous deposits), so the station-based denied-claim formula
        #   would give trivially 1.0 for every I episode.  The only observable contention
        #   in I tasks is at the PickUp level (two agents trying to grab the same toiletry),
        #   which is exactly what this path measures.
        return sum(1 for sr in steps if sr.get("skill") == "PickUp" and _target_id(sr))
    if dim == "D4":  # producer-consumer: evidence = objects actually landed on the buffer
        buffers = {b.get("objectId") for b in (constraints.get("buffer_stations") or []) if b.get("objectId")}
        if buffers:
            return sum(1 for sr in steps if sr.get("skill") == "Put" and sr.get("success") and _target_id(sr) in buffers)
        return sum(1 for sr in steps if sr.get("skill") in {"Put", "PickUp"})
    return inter_ok


def _construct_eval(dim, m, steps, constraints, traj_steps, config=None):
    """Gate the construct score on mechanism engagement, then compute it.

    Returns ``(score, status, sample)``. ``score`` is ``None`` whenever the score is
    withheld; ``status`` records *why* so a withheld score is never confused with a
    perfect one:
      * ``na_no_actions``  — the episode took no action at all
      * ``not_engaged``    — the dimension's mechanism was never exercised (sample 0)
      * ``low_signal``     — exercised too few times to estimate a rate (sample < floor)
      * ``na``             — engaged enough, but the formula itself has no signal
                             (e.g. single-agent D1) → genuine NA
      * ``scored``         — a real, reportable construct score
    """
    if m["n_action_steps"] == 0:
        return (None, "na_no_actions", 0)
    sample = _dim_sample(dim, m, steps, constraints)
    if sample == 0:
        return (None, "not_engaged", 0)
    if sample < CONSTRUCT_MIN_SAMPLE:
        return (None, "low_signal", sample)
    score = _construct_score(dim, m, steps, constraints, traj_steps, config)
    if score is None:
        return (None, "na", sample)
    return (score, "scored", sample)


def _distributed_metrics(traj_steps: List[Dict[str, Any]], steps: List[Dict[str, Any]], events: List[str]) -> Optional[Dict[str, Any]]:
    """Communication / concurrency diagnostics, computed only for distributed runs.

    Returns ``None`` for the centralized turn-based loop (no ``round_index`` is
    recorded), so the field is present but unambiguously NA. For a distributed run
    (``run_episode_concurrent`` records ``round_index`` and any broadcast
    ``message`` per landed step):

    * ``decision_parallelism`` — mean landed actions per decision round (1.0 ⇒ no
      real concurrency; →n_agents ⇒ all agents act every round).
    * ``redundant_work_rate`` — fraction of actions that duplicate another agent's
      same-round target (wasted work from lacking a shared plan).
    * ``deadlock_rate`` — fraction of rounds with zero successful progress
      (mutual waiting / repeated contention).
    * ``comm_volume`` — message count + approx tokens + messages/round.
    * ``comm_efficiency`` — progress steps per message (None when no messages).
    """
    rounds_map: Dict[int, List[Dict[str, Any]]] = {}
    for s in traj_steps:
        ri = s.get("round_index")
        if ri is None:
            continue
        rounds_map.setdefault(ri, []).append(s["skill_result"])
    if not rounds_map:
        return None  # centralized / turn-based run -> distributed metrics N/A

    n_rounds = len(rounds_map)
    landed = sum(len(srs) for srs in rounds_map.values())
    redundant = 0
    for srs in rounds_map.values():
        tids = [_target_id(sr) for sr in srs
                if sr.get("skill") in INTERACTION_SKILLS and _target_id(sr)]
        redundant += len(tids) - len(set(tids))      # >1 agent targeting same object this round
    stalled = sum(1 for srs in rounds_map.values() if not any(sr.get("success") for sr in srs))

    msgs = [s.get("message") for s in traj_steps if s.get("message")]
    n_msgs = len(msgs)
    tokens = sum(len((m or "").split()) for m in msgs)
    progress = sum(1 for e in events if e == "progress")

    return {
        "n_rounds": n_rounds,
        "decision_parallelism": landed / n_rounds,
        "redundant_work_rate": redundant / max(landed, 1),
        "deadlock_rate": stalled / n_rounds,
        "comm_volume": {"messages": n_msgs, "tokens": tokens, "per_round": n_msgs / n_rounds},
        "comm_efficiency": (progress / n_msgs) if n_msgs else None,
    }


def _buffer_occupancy(step: Dict[str, Any], buffer_ids: set) -> int:
    """How many tracked objects sit on the buffer station(s) in this step's snapshot.

    Reconstructed from the recorded per-step ``state.objects[*].parentReceptacles``
    (recorder.py) — the buffer's instantaneous load, with no controller needed.
    """
    objs = (step.get("state", {}) or {}).get("objects", {}) or {}
    return sum(
        1 for o in objs.values()
        if o.get("parentReceptacles") and any(b in o["parentReceptacles"] for b in buffer_ids)
    )


def _construct_score(dim, m, steps, constraints, traj_steps=None, config=None) -> Optional[float]:
    """Per-dimension construct score ∈[0,1] for the config's own dimension.

    D3/D4 are scored from the **executed action sequence and recorded object state**
    (a logical resource timeline), not from engine error strings. Under turn-based,
    one-action-per-step execution cannot represent simultaneous physical contention.
    The timeline-based detectors below instead capture failed exclusive-resource
    claims and buffer over-production.

    D1 is likewise execution-grounded and needs no separately-run oracle baseline.

    ``None`` (NA) only when there is no schedule/signal to score (a single agent, a
    0-action episode, or a resource/buffer the policy never engaged) — never a
    vacuous 1.0.
    """
    traj_steps = traj_steps or []
    if m["n_action_steps"] == 0:
        return None  # nothing was attempted -> no coordination signal to score
    if dim == "D2":  # precedence: violation rate over container-chain ops,
        # gated by legal_plan compliance.
        precedence = constraints.get("precedence", []) or []
        if not precedence:
            return None
        container_ids = set()
        cfg = config or {}
        for alias, oid in (cfg.get("init_state", {}).get("objects", {}) or {}).items():
            if "container" in alias or "basket" in alias or "drawer" in alias:
                container_ids.add(oid)
        chain_skills = {"Open", "Put", "Close"}
        def _on_container(sr):
            return _target_id(sr) in container_ids if container_ids else True
        viol = 0
        correct = 0
        for sr in steps:
            skill = sr.get("skill", "")
            if skill not in chain_skills or not _on_container(sr):
                continue
            if sr.get("success"):
                correct += 1
            else:
                msg = sr.get("errorMessage", "") or ""
                if _has(NOOP_MARKERS, msg) or _has(OCCUPANCY_MARKERS, msg) \
                        or _has(INTRINSIC_PRECOND_MARKERS, msg) \
                        or _has(CONCURRENCY_MARKERS, msg):
                    continue
                viol += 1
        ops = viol + correct
        if ops == 0:
            return None
        efficiency = _clip01(1 - viol / ops)
        legal = (m.get("illegal_skill", 0) == 0) and (m.get("dependency_violations", 0) == 0)
        return efficiency if legal else efficiency * 0.5
    if dim == "D3":  # mutual exclusion: contention score × scheduling efficiency
        lock_skills: set = set()
        for r in (constraints.get("resource_exclusion") or []):
            lock_skills |= set(r.get("skills_requiring_lock") or [])
        station_mode = bool(constraints.get("exclusive_resources")) or bool(lock_skills)

        # Shared-tool path (K family): PickUp contention on the shared tool.
        # The engine denies PickUp when another agent holds the object; the error
        # message contains "held by another agent".
        resource_ids: set = set()
        for r in (constraints.get("resource_exclusion") or []):
            rid = r.get("resource_id")
            if rid:
                resource_ids.add(rid)
        if resource_ids:
            access = [sr for sr in steps
                      if sr.get("skill") == "PickUp" and _target_id(sr) in resource_ids]
            uses = len(access)
            if uses == 0:
                return None
            collisions = sum(
                1 for sr in access
                if not sr.get("success") and "held by another agent" in (sr.get("errorMessage") or "")
            )
            contention = _clip01(1 - collisions / uses)
        elif station_mode:
            if not lock_skills:
                lock_skills = {"CleanObject", "FillObjectWithLiquid", "EmptyLiquidFromObject"}
            uses_steps = [sr for sr in steps if sr.get("skill") in lock_skills]
            uses = len(uses_steps)
            if uses == 0:
                return None
            denied = sum(
                1 for sr in uses_steps
                if not sr.get("success") and not _has(NOOP_MARKERS, sr.get("errorMessage", ""))
            )
            contention = _clip01(1 - denied / uses)
        else:
            pickup_oids = [_target_id(sr) for sr in steps if sr.get("skill") == "PickUp"]
            pickup_oids = [o for o in pickup_oids if o]
            uses = len(pickup_oids)
            if uses == 0:
                return None
            denied = uses - len(set(pickup_oids))
            contention = _clip01(1 - denied / uses)
        cfg = config or {}
        optimal = cfg.get("optimal_makespan")
        if optimal and m["n_action_steps"] > 0:
            efficiency = _clip01(optimal / m["n_action_steps"])
        else:
            efficiency = 1.0
        return _clip01(contention * efficiency)
    if dim == "D4":  # producer-consumer: 1 - (overflow + idle) rate over buffer transfers
        buffers = {b.get("objectId") for b in (constraints.get("buffer_stations") or []) if b.get("objectId")}
        if not buffers:
            transfers = sum(1 for sr in steps if sr.get("skill") in {"Put", "PickUp"}) or 1
            return _clip01(1 - (m["affordance_failures"] + m["wait_steps"]) / transfers)
        capacity = max((b.get("capacity", 1) for b in (constraints.get("buffer_stations") or [])), default=1)
        # throughput = objects successfully landed on the buffer (the denominator)
        transfers = sum(1 for sr in steps
                        if sr.get("skill") == "Put" and sr.get("success") and _target_id(sr) in buffers)
        if transfers == 0:
            return None  # the buffer station was never used -> NA
        # over-production: each insertion that pushes the buffer past capacity once
        # (logical), plus any engine-refused Put onto a full buffer (defensive).
        overflow = 0
        prev_occ = 0
        for st in traj_steps:
            occ = _buffer_occupancy(st, buffers)
            if occ > capacity and occ > prev_occ:
                overflow += 1
            prev_occ = occ
        overflow += sum(1 for sr in steps
                        if sr.get("skill") == "Put" and not sr.get("success")
                        and _has(AFFORDANCE_MARKERS, sr.get("errorMessage", "")) and _target_id(sr) in buffers)
        # coordination-relevant idle: consumer Wait while the buffer is empty (stalled
        # because the producer has not delivered) — not arbitrary waiting.
        idle = sum(1 for st in traj_steps
                   if (st.get("skill_result", {}) or {}).get("skill") == "Wait"
                   and _buffer_occupancy(st, buffers) == 0)
        return _clip01(1 - (overflow + idle) / transfers)
    if dim == "D1":  # load balancing: optimal makespan / realized makespan.
        # The reference makespan is the balanced oracle's (declared per-instance as
        # ``task_constraints.optimal_makespan``); absent that, fall back to the
        # fractional lower bound ceil(total / n_agents), which is exact only when
        # the independent subgoals are evenly divisible. AQ < 1 ⇒ one agent
        # overloaded; a single-agent solution lands near 1/n_agents.
        counts = list((m.get("per_agent_actions") or {}).values())
        n_agents = max(m.get("n_agents", 2), 1)
        total = sum(counts)
        realized = max(counts) if counts else 0
        if realized == 0 or n_agents < 2:
            return None  # no schedule to score (single agent / no actions) -> NA
        opt = constraints.get("optimal_makespan")
        if not opt:
            opt = -(-total // n_agents)  # ceil(total / n_agents)
        return _clip01(opt / realized)
    return None


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute coordination metrics from trajectory.json.")
    ap.add_argument("trajectories", nargs="+", help="One or more trajectory.json paths.")
    ap.add_argument("--json", action="store_true", help="Emit raw JSON per trajectory.")
    args = ap.parse_args()

    rows = []
    for path in args.trajectories:
        report = _load(path)
        m = compute_metrics(report)
        rows.append(m)
        if args.json:
            print(json.dumps(m, ensure_ascii=False, indent=2))
    if not args.json:
        hdr = f"{'task_id':<28}{'dim':<5}{'succ':<6}{'sub%':<6}{'legal':<6}{'mk':<4}{'dep':<4}{'occ':<4}{'aff':<4}{'ill':<4}{'C-score'}"
        print(hdr)
        print("-" * len(hdr))
        for m in rows:
            sub = f"{m['subgoal_success_rate']:.2f}" if m['subgoal_success_rate'] is not None else "-"
            cs = f"{m['construct_score']:.2f}" if m['construct_score'] is not None else "NA"
            print(f"{str(m['task_id']):<28}{str(m['coordination_dim']):<5}"
                  f"{('Y' if m['success'] else 'N'):<6}{sub:<6}{('Y' if m['legal_plan'] else 'N'):<6}"
                  f"{m['makespan']:<4}{m['dependency_violations']:<4}{m['occupancy_conflicts']:<4}"
                  f"{m['affordance_failures']:<4}{m['illegal_skill']:<4}{cs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
