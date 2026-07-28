"""VLM-driven centralized planner policy + pluggable VLM backends.

Closes the evaluation loop: turns each :class:`env.Observation` (the merged
per-agent egocentric image + text goal + filtered action menu + last feedback)
into a prompt, queries a VLM, and parses the chosen ``action id`` back into a
skill-call string from ``obs.action_menu``. It consumes **only** ``obs`` (never
the privileged EnvView), so the evaluation is fair.

Centralized planner: one model sees all agents' views and emits one
``(agent, skill)`` action per step (the menu entry already names its agent), so
the multi-agent coordination — who acts, in what order, who waits — is exactly
what is being scored.

Backends (``--vlm-backend``):
  * ``anthropic`` — Anthropic Messages API via ``requests`` (vision), using
    ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN`` / a model id.
  * ``openai`` — OpenAI-compatible Chat Completions API via ``requests``
    (vision), using ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` / a model id.
  * ``mock`` — deterministic, no network; proves the full chain end-to-end.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, List, Optional

from env import Observation
from policy import Policy
from skill_executor import SkillExecutionResult
from taskutil import coordination_hint, predicate_to_nl


# --------------------------------------------------------------------------- #
# Backends                                                                     #
# --------------------------------------------------------------------------- #
class MockVLM:
    """Deterministic stand-in: cycles through menu action ids. Proves plumbing
    (obs -> prompt -> parse -> call -> step -> metrics) without any network."""

    name = "mock"

    def __init__(self, **_: Any) -> None:
        self._i = 0

    def choose(self, prompt: str, image_path: Optional[str], menu: List[Dict[str, Any]]) -> str:
        if not menu:
            return "DONE"
        entry = menu[self._i % len(menu)]
        self._i += 1
        return f"ACTION: {entry['action_id']}"


class RandomVLM:
    """Uniform-random legal action each step — the no-strategy floor baseline.

    Picks a random menu entry (seeded for reproducibility) and never emits DONE;
    the harness step budget bounds the episode."""

    name = "random"

    def __init__(self, seed: int = 0, **_: Any) -> None:
        import random as _random
        self._rng = _random.Random(seed)

    def choose(self, prompt: str, image_path: Optional[str], menu: List[Dict[str, Any]]) -> str:
        if not menu:
            return "DONE"
        return f"ACTION: {self._rng.choice(menu)['action_id']}"


# Myopic progress priority: try to make single-step progress, never reasoning
# about allocation / ordering / contention across agents. State-changing skills
# first, then navigation, then idle — the "greedy but uncoordinated" floor.
_GREEDY_PRIORITY = (
    "PickUp", "Put", "Open", "Close", "Slice", "ToggleOn", "ToggleOff",
    "CleanObject", "FillObjectWithLiquid", "EmptyLiquidFromObject",
    "PushObject", "PullObject", "Drop", "Find", "Explore", "Wait",
)


class GreedyVLM:
    """Myopic, uncoordinated baseline: always take the highest-priority available
    action (state-change > navigate > idle), first entry within a tier. Acts
    plausibly per-agent but never coordinates, so it engages the mechanisms
    (sample>=2 -> scored) yet should score low on the coordination constructs —
    a stronger floor than random. Step budget bounds the episode."""

    name = "greedy"

    def __init__(self, **_: Any) -> None:
        self._rank = {s: i for i, s in enumerate(_GREEDY_PRIORITY)}

    def _skill_of(self, entry: Dict[str, Any]) -> str:
        call = entry.get("call") or ""
        return call.split("(", 1)[0].strip()

    def choose(self, prompt: str, image_path: Optional[str], menu: List[Dict[str, Any]]) -> str:
        if not menu:
            return "DONE"
        best = min(menu, key=lambda e: self._rank.get(self._skill_of(e), len(_GREEDY_PRIORITY)))
        return f"ACTION: {best['action_id']}"


class AnthropicVLM:
    """Anthropic Messages API (vision) via raw requests — no SDK dependency."""

    name = "anthropic"

    def __init__(self, model: Optional[str] = None, max_tokens: int = 300, timeout: int = 180, retries: int = 3, **_: Any) -> None:
        self.base = os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
        self.token = os.environ["ANTHROPIC_AUTH_TOKEN"]
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or os.environ.get("ANTHROPIC_DEFAULT_OPUS_MODEL")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    def choose(self, prompt: str, image_path: Optional[str], menu: List[Dict[str, Any]]) -> str:
        import requests

        content: List[Dict[str, Any]] = []
        if image_path and os.path.exists(image_path):
            data = base64.b64encode(open(image_path, "rb").read()).decode()
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
        content.append({"type": "text", "text": prompt})
        payload = {"model": self.model, "max_tokens": self.max_tokens,
                   "messages": [{"role": "user", "content": content}]}
        headers = {"x-api-key": self.token, "authorization": f"Bearer {self.token}",
                   "anthropic-version": "2023-06-01", "content-type": "application/json"}
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                resp = requests.post(f"{self.base}/v1/messages", headers=headers, json=payload,
                                     timeout=self.timeout * (attempt + 1))
                _raise_for_status(resp)
                data = resp.json()
                blocks = data.get("content", [])
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                if not text.strip():
                    # Empty completion (truncation / transient blip) is not a model
                    # decision to stop — retry rather than let it terminate the episode.
                    raise RuntimeError(f"empty completion (stop_reason={data.get('stop_reason')})")
                return text
            except Exception as exc:  # transient timeout / 5xx / empty -> retry with longer timeout
                last_exc = exc
                if not _is_retryable(exc):  # deterministic 4xx -> fail fast
                    break
        raise last_exc  # exhausted retries


class OpenAICompatVLM:
    """OpenAI-compatible Chat Completions API (vision) via raw requests.

    The custom endpoint can be configured through ``OPENAI_BASE_URL``. It should
    point either to the API root (``https://host``) or to a versioned root
    (``https://host/v1``); the backend appends ``/chat/completions`` when needed.
    """

    name = "openai"

    def __init__(self, model: Optional[str] = None, max_tokens: Optional[int] = None, timeout: int = 180, retries: int = 3, **_: Any) -> None:
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.token = os.environ["OPENAI_API_KEY"]
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        env_max_tokens = os.environ.get("OPENAI_MAX_TOKENS")
        self.max_tokens = int(env_max_tokens) if env_max_tokens else max_tokens
        # Optional extra request-body params merged into every payload, as a JSON
        # dict in OPENAI_EXTRA_BODY. Used e.g. to disable a reasoning model's
        # chain-of-thought ({"enable_thinking": false}) so latency stays tractable.
        self.extra_body: Dict[str, Any] = {}
        raw_extra = os.environ.get("OPENAI_EXTRA_BODY")
        if raw_extra:
            try:
                parsed = json.loads(raw_extra)
                if isinstance(parsed, dict):
                    self.extra_body = parsed
            except Exception:
                pass
        self.timeout = timeout
        self.retries = retries
        # Some OpenAI-compatible endpoints reject `temperature`. Send it only for
        # model families known to accept it, with an environment override available.
        send_temp = os.environ.get("OPENAI_SEND_TEMPERATURE")
        model_l = (self.model or "").lower()
        if send_temp in {"0", "1"}:
            self.send_temperature = send_temp == "1"
        else:
            self.send_temperature = model_l.startswith(("gpt", "o1", "o3", "o4"))

    def _url(self) -> str:
        if self.base.endswith("/chat/completions"):
            return self.base
        if self.base.endswith("/v1"):
            return f"{self.base}/chat/completions"
        return f"{self.base}/v1/chat/completions"

    def choose(self, prompt: str, image_path: Optional[str], menu: List[Dict[str, Any]]) -> str:
        import requests

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_path and os.path.exists(image_path):
            data = base64.b64encode(open(image_path, "rb").read()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        }
        if self.send_temperature:
            payload["temperature"] = 0
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.extra_body:
            payload.update(self.extra_body)
        headers = {"authorization": f"Bearer {self.token}", "content-type": "application/json"}
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                resp = requests.post(self._url(), headers=headers, json=payload,
                                     timeout=self.timeout * (attempt + 1))
                _raise_for_status(resp)
                data = resp.json()
                text = _extract_openai_text(data)
                if not (text or "").strip():
                    # Empty completion (truncation / transient blip) is not a model
                    # decision to stop — retry rather than let it terminate the episode.
                    raise RuntimeError(f"empty completion (finish_reason={_finish_reason(data)})")
                return text
            except Exception as exc:  # transient timeout / 5xx / empty -> retry with longer timeout
                last_exc = exc
                if not _is_retryable(exc):  # deterministic 4xx -> fail fast
                    break
        raise last_exc


def _raise_for_status(resp: Any) -> None:
    """Raise with a short response body; never include request headers."""
    if resp.status_code < 400:
        return
    body = (resp.text or "").strip().replace("\n", " ")
    if len(body) > 500:
        body = body[:500] + "..."
    raise BackendHTTPError(resp.status_code, f"HTTP {resp.status_code} from {resp.url}: {body}")


class BackendHTTPError(RuntimeError):
    """HTTP error carrying the status code so the retry loop can tell a transient
    failure (timeout / 5xx / 429) from a deterministic client error (4xx) that
    will never succeed on retry — e.g. a model that rejects image content."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _is_retryable(exc: Exception) -> bool:
    """A 4xx (except 408/409/429) is a deterministic client error — do not retry;
    retrying just wastes time and obscures the real cause. Everything else
    (timeouts, 5xx, empty completions, network errors) is worth a retry."""
    if isinstance(exc, BackendHTTPError) and 400 <= exc.status_code < 500:
        return exc.status_code in (408, 409, 429)
    return True


def _extract_openai_text(data: Dict[str, Any]) -> str:
    """Accept common OpenAI-compatible response shapes."""
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
            )
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    return ""


def _finish_reason(data: Dict[str, Any]) -> Optional[str]:
    """Best-effort finish/stop reason for diagnosing empty completions."""
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        return choices[0].get("finish_reason") or choices[0].get("stop_reason")
    return data.get("stop_reason")


BACKENDS = {"mock": MockVLM, "anthropic": AnthropicVLM, "openai": OpenAICompatVLM,
            "random": RandomVLM, "greedy": GreedyVLM}


def make_vlm_client(backend: str, model: Optional[str] = None) -> Any:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown vlm backend: {backend} (choices: {list(BACKENDS)})")
    return BACKENDS[backend](model=model)


# --------------------------------------------------------------------------- #
# Policy                                                                        #
# --------------------------------------------------------------------------- #
PROMPT_HEADER = """You are the centralized planner coordinating {n} agents ({agents}) in an AI2-THOR household task.
{view}
GOAL: {goal}

GOAL PROGRESS (live simulator state — trust this over your action history):
{goal_progress}
{hint}
Choose the single best NEXT action for ONE agent to make progress while coordinating
(divide work, respect ordering/dependencies, avoid two agents contending for the same spot/resource).
After Find succeeds, the agent is at the object — proceed to the interaction (PickUp/Put/Toggle/...),
do NOT repeat Find. Reply on two lines exactly:
REASON: <one short sentence>
ACTION: <the integer action id from the menu, or DONE if every goal is satisfied>

ACTION MENU (action_id: skill [agent]):
{menu}

RECENT ACTIONS (most recent last):
{history}
{state}"""

VIEW_IMAGE = "The image shows each agent's egocentric view side by side, left to right in the order: {agents}."
VIEW_BLIND = "No image is provided. You have NO direct perception of the scene: infer object states only from the GOAL, the available actions, your inventory, and the success/failure of your past actions."

ACTION_RE = re.compile(r"ACTION:\s*(\d+|DONE)", re.IGNORECASE)


def _build_goal_progress(obs: "Observation") -> str:
    """Build a live goal-progress block from obs.eval and obs.object_mapping.

    Each objective is shown as ``[✓]`` or ``[✗]`` with its natural-language
    description so the model always knows the real simulator state — independent
    of how many history entries have scrolled out of the rolling window.
    """
    checks = (obs.eval or {}).get("checks", [])
    if not checks:
        return "  (no goal predicates)"
    obj_map = obs.object_mapping or {}
    lines = []
    done = sum(1 for c in checks if c.get("passed"))
    for check in checks:
        mark = "[✓]" if check.get("passed") else "[✗]"
        desc = predicate_to_nl(check.get("predicate", {}), obj_map)
        lines.append(f"  {mark} {desc}")
    lines.append(f"  {done}/{len(checks)} objectives complete.")
    return "\n".join(lines)


def parse_action(raw: str, menu: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse a model reply into a menu action.

    Shared by the centralized and distributed policies so both interpret the
    ``ACTION: <id|DONE>`` contract (with a bare-integer fallback) identically.
    Returns ``{status, action_id, call, parse_mode, error}``; ``call`` is ``None``
    for DONE / parse failure / an id not in the supplied ``menu``.
    """
    m = ACTION_RE.search(raw or "")
    if not m:
        # fallback: first bare integer in the reply
        m2 = re.search(r"\b(\d+)\b", raw or "")
        if not m2:
            return {"status": "parse_error", "error": "missing ACTION line or integer"}
        choice = m2.group(1)
        parse_mode = "bare_integer"
    else:
        choice = m.group(1)
        parse_mode = "action_line"
    if str(choice).upper() == "DONE":
        return {"status": "done", "action_id": "DONE", "call": None, "parse_mode": parse_mode}
    by_id = {str(e["action_id"]): e["call"] for e in menu}
    call = by_id.get(str(choice))
    if call is None:
        return {"status": "invalid_action_id", "action_id": str(choice), "call": None, "parse_mode": parse_mode, "error": "action id not in current menu"}
    return {"status": "action", "action_id": str(choice), "call": call, "parse_mode": parse_mode}


def parse_message(raw: str) -> str:
    """Extract a ``MSG: <text>`` broadcast line from a model reply (D1 comm), or ''."""
    m = re.search(r"MSG:\s*(.+)", raw or "")
    return m.group(1).strip()[:200] if m else ""



class CentralizedVLMPolicy(Policy):
    def __init__(self, client: Any, verbose: bool = False, obs_mode: str = "image") -> None:
        self.client = client
        self.verbose = verbose
        self.obs_mode = obs_mode  # "image" (egocentric views) | "blind" (no image; plan from goal/menu/feedback)
        self._agents: List[str] = []
        self._hint = ""
        self.last_decision: Dict[str, Any] = {}
        self.planner_steps = 0
        self.parse_errors = 0
        self.backend_errors = 0

    def reset(self, env: Any, task: Dict[str, Any]) -> None:
        self._agents = [a.get("id") for a in task.get("agents", [])] or list(
            task.get("agent_count", 2) * [""])
        self._hint = coordination_hint(task)
        self._history: List[str] = []   # recent "call -> ok/FAIL" lines
        self._last_call: Optional[str] = None
        self.last_decision = {}
        self.planner_steps = 0
        self.parse_errors = 0
        self.backend_errors = 0

    # ---- prompt building -------------------------------------------------
    def _build_prompt(self, obs: Observation, last_result: Optional[SkillExecutionResult]) -> str:
        menu_lines = "\n".join(
            f"  {e['action_id']}: {e['name']} [{e['agent']}]" for e in obs.action_menu
        ) or "  (no actions available)"
        state_lines = ["", "AGENT STATE:"]
        for name, st in obs.per_agent.items():
            inv = ", ".join(st.get("inventory") or []) or "empty"
            state_lines.append(f"  {name}: holding={inv}")
        agents_str = ", ".join(obs.per_agent.keys()) or ", ".join(self._agents)
        view = (VIEW_BLIND if self.obs_mode == "blind" else VIEW_IMAGE).format(agents=agents_str)
        history = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(self._history[-6:])) or "  (none yet)"
        return PROMPT_HEADER.format(
            n=len(obs.per_agent) or len(self._agents),
            agents=agents_str,
            view=view,
            goal=obs.goal_text,
            goal_progress=_build_goal_progress(obs),
            hint=(self._hint + "\n") if self._hint else "",
            menu=menu_lines,
            history=history,
            state="\n".join(state_lines),
        )

    # ---- decision --------------------------------------------------------
    def act(self, obs: Observation, last_result: Optional[SkillExecutionResult]) -> Optional[str]:
        # record the outcome of the previous action so the model has memory
        if self._last_call is not None and last_result is not None:
            ok = "ok" if last_result.success else f"FAIL({last_result.errorMessage[:60]})"
            self._history.append(f"{self._last_call} -> {ok}")
        if not obs.action_menu:
            self.last_decision = self._decision(obs, status="no_actions", terminal_reason="no_actions")
            return None
        prompt = self._build_prompt(obs, last_result)
        try:
            self.planner_steps += 1
            image = None if self.obs_mode == "blind" else obs.image
            raw = self.client.choose(prompt, image, obs.action_menu)
        except Exception as exc:  # network/parse errors end the episode gracefully
            self.backend_errors += 1
            self.last_decision = self._decision(obs, status="backend_error", terminal_reason="backend_error", error=repr(exc))
            if self.verbose:
                print(f"[vlm] backend error: {exc!r}")
            return None
        parsed = self._parse(raw, obs.action_menu)
        status = parsed["status"]
        if status in {"parse_error", "invalid_action_id"}:
            self.parse_errors += 1
        call = parsed.get("call")
        terminal_reason = status if call is None else None
        self.last_decision = self._decision(
            obs,
            status=status,
            terminal_reason=terminal_reason,
            raw_output=raw,
            parsed_action_id=parsed.get("action_id"),
            parsed_call=call,
            parse_mode=parsed.get("parse_mode"),
            error=parsed.get("error"),
        )
        self._last_call = call
        if self.verbose:
            print(f"[vlm] {(raw or '').strip()[:160]} -> {call}")
        return call

    @staticmethod
    def _parse(raw: str, menu: List[Dict[str, Any]]) -> Dict[str, Any]:
        return parse_action(raw, menu)

    def _decision(self, obs: Observation, *, status: str, terminal_reason: Optional[str] = None, raw_output: Optional[str] = None,
                  parsed_action_id: Optional[str] = None, parsed_call: Optional[str] = None,
                  parse_mode: Optional[str] = None, error: Optional[str] = None) -> Dict[str, Any]:
        decision: Dict[str, Any] = {
            "policy": "centralized_vlm",
            "backend": getattr(self.client, "name", type(self.client).__name__),
            "model": getattr(self.client, "model", None),
            "step_index": obs.step_index,
            "status": status,
            "terminal_reason": terminal_reason,
            "parsed_action_id": parsed_action_id,
            "parsed_call": parsed_call,
            "parse_mode": parse_mode,
            "planner_steps": self.planner_steps,
            "parse_errors": self.parse_errors,
            "backend_errors": self.backend_errors,
        }
        if raw_output is not None:
            decision["raw_output"] = raw_output
        if error:
            decision["error"] = error
        return decision


def make_vlm_policy(backend: str = "anthropic", model: Optional[str] = None, verbose: bool = False, obs_mode: str = "image") -> CentralizedVLMPolicy:
    return CentralizedVLMPolicy(make_vlm_client(backend, model), verbose=verbose, obs_mode=obs_mode)
