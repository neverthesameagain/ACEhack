"""Adaptive company agents for ACE++ Option B."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from ace_reward import ACTIONS


ROUND_TYPES = ["cooperative", "competitive", "resource"]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass
class AgentProfile:
    agent_id: int
    name: str
    company_type: str
    emoji: str
    primary_objective: str
    stake_oil: float
    stake_gold: float
    stake_food: float
    stake_cooperation: float
    risk_tolerance: float
    resources: float = 100.0
    self_memory: list[dict[str, Any]] = field(default_factory=list)
    opponent_memory: dict[int, dict[str, float]] = field(default_factory=dict)
    trust_scores: dict[int, float] = field(default_factory=dict)
    strategy_success: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=lambda: {rt: {} for rt in ROUND_TYPES}
    )

    def system_prompt(self, world_state_str: str) -> str:
        return f"""You are {self.name}, a {self.company_type} in ACE++.

Objective: {self.primary_objective}

Exposures:
- Oil: {self.stake_oil:+.1f}
- Gold: {self.stake_gold:+.1f}
- Food: {self.stake_food:+.1f}
- Cooperation preference: {self.stake_cooperation:+.1f}
- Risk tolerance: {self.risk_tolerance:.1f}
- Resources: {self.resources:.1f}

World state:
{world_state_str}

Memory:
{self.memory_summary()}

Choose based on your identity, incentives, trust, and learned habits.

Output ONLY valid JSON:
{{
  "predicted_round": "cooperative|competitive|resource",
  "action": "submit_bid|propose_alliance|accept_alliance|reject_alliance|betray|challenge|allocate_resources|execute_contract",
  "parameters": {{"amount": 50}},
  "reasoning": "one sentence"
}}"""

    def memory_summary(self) -> str:
        recent = self.self_memory[-3:]
        habit_lines = []
        for round_type, stats in self.strategy_success.items():
            if not stats:
                continue
            best_action = max(
                stats,
                key=lambda action: stats[action].get("successes", 0) / max(1, stats[action].get("attempts", 0)),
            )
            item = stats[best_action]
            rate = item.get("successes", 0) / max(1, item.get("attempts", 0))
            habit_lines.append(f"- In {round_type}, {best_action} has worked {rate:.0%} of attempts")

        opponent_lines = []
        for agent_id, model in sorted(self.opponent_memory.items()):
            opponent_lines.append(
                f"- Agent {agent_id}: aggression={model.get('aggressiveness_score', 0):.2f}, "
                f"cooperation={model.get('cooperation_score', 0):.2f}, betrayals={int(model.get('betrayal_count', 0))}, "
                f"trust={self.trust_scores.get(agent_id, 0.5):.2f}"
            )

        return "\n".join(
            [
                "Recent self outcomes:",
                json.dumps(recent, indent=2) if recent else "None yet",
                "Learned habits:",
                "\n".join(habit_lines) if habit_lines else "No strong habits yet",
                "Opponent model:",
                "\n".join(opponent_lines) if opponent_lines else "No opponent observations yet",
            ]
        )

    def choose_fallback_action(
        self,
        world_probs: dict[str, float],
        round_number: int,
        available_agents: list[int],
    ) -> dict[str, Any]:
        predicted_round = self._predict_round_from_identity(world_probs)
        candidates = self._candidate_actions(predicted_round, available_agents)
        scored = []
        for action, parameters in candidates:
            historical = self._historical_score(predicted_round, action)
            trust_alignment = self._trust_alignment(action, parameters)
            risk_score = self._risk_score(action)
            base = 0.35
            score = base + 0.3 * historical + 0.2 * trust_alignment + 0.2 * risk_score
            scored.append((score, action, parameters))
        scored.sort(reverse=True, key=lambda item: item[0])
        _, action, parameters = scored[0]
        return {
            "predicted_round": predicted_round,
            "action": action,
            "parameters": parameters,
            "reasoning": self._fallback_reasoning(predicted_round, action, round_number),
        }

    def update_after_round(
        self,
        *,
        round_number: int,
        action: str,
        predicted_round: str,
        actual_round: str,
        reward: float,
        success: bool,
        other_actions: dict[int, str],
    ) -> None:
        self.resources = max(0.0, self.resources + reward * 8.0)
        self.self_memory.append(
            {
                "round": round_number,
                "action": action,
                "predicted_round": predicted_round,
                "actual_round": actual_round,
                "reward": round(reward, 3),
                "success": bool(success),
            }
        )
        self.self_memory = self.self_memory[-12:]

        action_stats = self.strategy_success.setdefault(actual_round, {}).setdefault(
            action, {"attempts": 0, "successes": 0}
        )
        action_stats["attempts"] += 1
        if success:
            action_stats["successes"] += 1

        for other_id, other_action in other_actions.items():
            if other_id == self.agent_id:
                continue
            model = self.opponent_memory.setdefault(
                other_id,
                {"aggressiveness_score": 0.0, "cooperation_score": 0.0, "betrayal_count": 0.0},
            )
            if other_action in {"challenge", "betray", "submit_bid"}:
                model["aggressiveness_score"] = _clamp(model["aggressiveness_score"] + 0.12)
            if other_action in {"propose_alliance", "accept_alliance", "execute_contract"}:
                model["cooperation_score"] = _clamp(model["cooperation_score"] + 0.12)
                self.trust_scores[other_id] = _clamp(self.trust_scores.get(other_id, 0.5) + 0.06)
            if other_action == "betray":
                model["betrayal_count"] += 1
                self.trust_scores[other_id] = _clamp(self.trust_scores.get(other_id, 0.5) - 0.3)

    def _predict_round_from_identity(self, world_probs: dict[str, float]) -> str:
        adjusted = dict(world_probs)
        if self.stake_cooperation > 0.5:
            adjusted["cooperative"] += 0.1
        if self.risk_tolerance > 0.75 or self.stake_cooperation < -0.2:
            adjusted["competitive"] += 0.12
        if self.stake_food < -0.5 or self.stake_oil < -0.5:
            adjusted["resource"] += 0.1
        return max(adjusted, key=adjusted.get)

    def _candidate_actions(self, predicted_round: str, available_agents: list[int]) -> list[tuple[str, dict[str, Any]]]:
        partner = next((agent_id for agent_id in available_agents if agent_id != self.agent_id), 0)
        if predicted_round == "competitive":
            return [
                ("challenge", {"target_id": partner}),
                ("submit_bid", {"amount": 80}),
                ("betray", {"partner_id": partner}),
            ]
        if predicted_round == "cooperative":
            return [
                ("propose_alliance", {"target_id": partner}),
                ("execute_contract", {"team_id": partner}),
                ("submit_bid", {"amount": 25, "partner_id": partner}),
            ]
        return [
            ("allocate_resources", {"amount": 50}),
            ("execute_contract", {"team_id": partner}),
            ("submit_bid", {"amount": 50}),
        ]

    def _historical_score(self, predicted_round: str, action: str) -> float:
        stats = self.strategy_success.get(predicted_round, {}).get(action)
        if not stats:
            return 0.3
        return _clamp(stats.get("successes", 0) / max(1, stats.get("attempts", 0)))

    def _trust_alignment(self, action: str, parameters: dict[str, Any]) -> float:
        partner = parameters.get("target_id", parameters.get("partner_id", parameters.get("team_id")))
        trust = self.trust_scores.get(int(partner), 0.5) if partner is not None else 0.5
        if action in {"propose_alliance", "accept_alliance", "execute_contract"}:
            return trust
        if action in {"challenge", "betray"}:
            return 1.0 - trust
        return 0.5

    def _risk_score(self, action: str) -> float:
        if action in {"challenge", "betray", "submit_bid"}:
            return self.risk_tolerance
        if action in {"propose_alliance", "accept_alliance", "execute_contract"}:
            return _clamp(self.stake_cooperation)
        return 1.0 - abs(self.risk_tolerance - 0.5)

    def _fallback_reasoning(self, predicted_round: str, action: str, round_number: int) -> str:
        return (
            f"Round {round_number}: {self.name} expects {predicted_round} conditions and chooses "
            f"{action} based on exposures, trust, and learned habits."
        )


AGENT_PROFILES = [
    AgentProfile(
        agent_id=0,
        name="PetroCorp",
        company_type="Energy Company",
        emoji="Oil",
        primary_objective="Maximise revenue from oil and energy assets during price spikes",
        stake_oil=0.9,
        stake_gold=0.1,
        stake_food=-0.2,
        stake_cooperation=-0.3,
        risk_tolerance=0.8,
    ),
    AgentProfile(
        agent_id=1,
        name="GlobalFoods Inc",
        company_type="Food Importer & Distributor",
        emoji="Food",
        primary_objective="Secure supply chains and minimise commodity cost exposure",
        stake_oil=-0.6,
        stake_gold=0.0,
        stake_food=-0.8,
        stake_cooperation=0.7,
        risk_tolerance=0.3,
    ),
    AgentProfile(
        agent_id=2,
        name="Aurelius Capital",
        company_type="Hedge Fund",
        emoji="Fund",
        primary_objective="Generate alpha from volatility and market dislocations",
        stake_oil=0.4,
        stake_gold=0.7,
        stake_food=0.3,
        stake_cooperation=-0.5,
        risk_tolerance=0.95,
    ),
    AgentProfile(
        agent_id=3,
        name="CentralBank of ACE",
        company_type="Central Bank / Regulator",
        emoji="Bank",
        primary_objective="Maintain market stability, control inflation, and prevent systemic risk",
        stake_oil=-0.3,
        stake_gold=-0.2,
        stake_food=-0.5,
        stake_cooperation=0.9,
        risk_tolerance=0.1,
    ),
]


def fresh_agent_profiles() -> list[AgentProfile]:
    """Return independent mutable agent profiles for a new simulation."""
    import copy

    agents = copy.deepcopy(AGENT_PROFILES)
    ids = [agent.agent_id for agent in agents]
    for agent in agents:
        agent.trust_scores = {other_id: 0.5 for other_id in ids if other_id != agent.agent_id}
    return agents
