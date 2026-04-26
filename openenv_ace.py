"""OpenEnv-compatible adapter for the current EIE Economic Intelligence Engine environment.

This file is intentionally small: Hugging Face Spaces launches `app.py`, while
OpenEnv-style evaluators can import `ACEOpenMultiAgentEnv` from this module.
"""

from __future__ import annotations

import json
from typing import Any

from ace_world_env import ACEWorldEnv

try:
    from openenv import Environment  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class Environment:  # type: ignore[no-redef]
        """Minimal fallback base so local smoke tests do not require OpenEnv."""

        def reset(self) -> dict[str, Any]:
            raise NotImplementedError

        def step(self, actions: Any) -> tuple[dict[str, Any], list[float], bool, dict[str, Any]]:
            raise NotImplementedError

        def state(self) -> dict[str, Any]:
            raise NotImplementedError


class ACEOpenMultiAgentEnv(Environment):
    """Thin OpenEnv-style wrapper around the live multi-agent simulator."""

    def __init__(self, *, seed: int = 7, max_rounds: int = 10, event_text: str | None = None) -> None:
        self.seed = seed
        self.max_rounds = max_rounds
        self.event_text = event_text
        self._env = ACEWorldEnv(rng_seed=seed)
        if event_text:
            self._env.apply_event(event_text, provider="fallback")

    def reset(self) -> dict[str, Any]:
        self._env = ACEWorldEnv(rng_seed=self.seed)
        if self.event_text:
            self._env.apply_event(self.event_text, provider="fallback")
        return self.state()

    def step(self, actions: Any = None) -> tuple[dict[str, Any], list[float], bool, dict[str, Any]]:
        parsed_actions = self._parse_actions(actions)
        result = self._env.step(parsed_actions)
        rewards = [float(item["reward"]["total"]) for item in result["results"]]
        done = self._env.round_number >= self.max_rounds
        info = {
            "round": self._env.round_number,
            "ground_truth": result["ground_truth"],
            "history_entry": result["history_entry"],
        }
        return self.state(), rewards, done, info

    def state(self) -> dict[str, Any]:
        return self._env.state()

    def _parse_actions(self, actions: Any) -> list[dict[str, Any]] | None:
        if actions is None:
            return None
        if isinstance(actions, str):
            try:
                actions = json.loads(actions)
            except json.JSONDecodeError:
                return None
        if not isinstance(actions, list):
            return None
        parsed: list[dict[str, Any]] = []
        for action in actions:
            if isinstance(action, str):
                try:
                    action = json.loads(action)
                except json.JSONDecodeError:
                    action = {}
            parsed.append(action if isinstance(action, dict) else {})
        return parsed


class ACEOpenEnv(ACEOpenMultiAgentEnv):
    """Alias kept for OpenEnv registries that expect a single environment class."""


__all__ = ["ACEOpenEnv", "ACEOpenMultiAgentEnv", "Environment"]
