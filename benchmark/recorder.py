"""Trajectory recording and egocentric-frame rendering.

Owns the run's step log and image artifacts so the env can stay focused on the
controller/observation loop. ``record`` snapshots state + goal eval and saves a
side-by-side per-agent RGB frame each step; ``finalize`` writes
``trajectory.json`` and returns the final eval. Output goes to ``output_dir``
(default ``outputs/task_runs/<task_id>/``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from evaluator import evaluate_task, state_summary
from thormeta import frame_for_agent


class TrajectoryRecorder:
    def __init__(self, output_dir: Path, agent_names: Sequence[str], config: Dict[str, Any], save_images: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.agent_names = list(agent_names)
        self.config = config
        # Default off: frames are not accumulated per step — each step overwrites a
        # single rolling file per agent (+ a merged frame), so the VLM still reads
        # the current observation but disk usage stays ~constant. With save_images
        # on, every step's frame is kept (step_NNN*.png) for full-trajectory replay.
        # trajectory.json's per-step "image" then points at the rolling file (latest
        # frame only) when off.
        self.save_images = save_images
        self.trajectory: List[Dict[str, Any]] = []
        self.last_image: Optional[Path] = None
        # Per-agent egocentric frame paths from the most recent observation. The
        # merged side-by-side image (``last_image``) feeds the centralized planner;
        # these single-agent crops feed the distributed policy, where each agent
        # sees only its own view.
        self.last_agent_images: Dict[str, Optional[Path]] = {}

    def reset(self) -> None:
        self.trajectory = []

    def record(
        self,
        step_index: int,
        event: Any,
        label: str,
        *,
        result: Optional[Any] = None,
        native_trace: Optional[List[Dict[str, Any]]] = None,
        policy_decision: Optional[Dict[str, Any]] = None,
        round_index: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        image_path = self._save_observation(step_index, event)
        if image_path is not None:
            self.last_image = image_path
        item: Dict[str, Any] = {
            "step_index": step_index,
            "label": label,
            "image": str(image_path) if image_path else None,
            "state": state_summary(event, self.config),
        }
        if round_index is not None:
            item["round_index"] = round_index
        if message is not None:
            item["message"] = message
        if result is not None:
            item["skill_result"] = result.to_dict()
        # Evaluate goals with access to trajectory (including current step) so
        # predicates like sliced_by can check which agent performed an action.
        eval_traj = self.trajectory + [item] if item.get("skill_result") else self.trajectory
        item["eval"] = evaluate_task(event, self.config, eval_traj)
        if native_trace is not None:
            item["native_trace"] = native_trace
        if policy_decision is not None:
            item["policy_decision"] = policy_decision
        self.trajectory.append(item)

    def attach_policy_decision(self, policy_decision: Dict[str, Any]) -> None:
        if not self.trajectory:
            return
        self.trajectory[-1]["policy_decision"] = policy_decision

    def _save_observation(self, step_index: int, event: Any) -> Optional[Path]:
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:
            print(f"[warn] PIL unavailable; cannot save observation: {exc}")
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        images = []
        agent_paths: Dict[str, Optional[Path]] = {}
        for index, name in enumerate(self.agent_names):
            frame = frame_for_agent(event, index)
            if frame is None:
                agent_paths[name] = None
                continue
            image = Image.fromarray(frame)
            label_h = 28
            canvas = Image.new("RGB", (image.width, image.height + label_h), "white")
            canvas.paste(image, (0, label_h))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 7), name, fill=(0, 0, 0))
            images.append(canvas)
            # Save each agent's own egocentric frame separately for the distributed
            # policy (one VLM per agent, each seeing only its own view). With
            # save_images off, reuse a rolling per-agent filename (overwritten each
            # step) so disk stays ~constant.
            if self.save_images:
                agent_path = self.output_dir / f"step_{step_index:03d}_{name}.png"
            else:
                agent_path = self.output_dir / f"frame_{name}.png"
            canvas.save(agent_path)
            agent_paths[name] = agent_path
        self.last_agent_images = agent_paths
        if not images:
            return None
        merged = Image.new("RGB", (sum(img.width for img in images), max(img.height for img in images)), "white")
        x = 0
        for image in images:
            merged.paste(image, (x, 0))
            x += image.width
        if self.save_images:
            path = self.output_dir / f"step_{step_index:03d}.png"
        else:
            path = self.output_dir / "frame.png"
        merged.save(path)
        merged.save(self.output_dir / "latest.png")
        return path

    def finalize(self, init_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_eval = self.trajectory[-1].get("eval", {"success": False, "checks": []}) if self.trajectory else {"success": False, "checks": []}
        report = {
            "task_id": self.config.get("task_id"),
            "task_config": self.config,
            "init_trace": init_trace,
            "final_eval": final_eval,
            "trajectory": self.trajectory,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "trajectory.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return final_eval
