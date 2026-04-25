"""Structured economic world and multi-agent simulator for ACE++ Option B."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

from ace_agents import AgentProfile, fresh_agent_profiles
from ace_reward import compute_total_reward
from ace_text_inject import parse_event_payload


ROUND_TYPES = ["competitive", "cooperative", "resource"]

CLAMP_RANGES = {
    "oil_price": (0.1, 3.0),
    "gold_price": (0.1, 3.0),
    "food_index": (0.1, 3.0),
    "energy_cost": (0.1, 3.0),
    "interest_rate": (0.0, 0.25),
    "inflation": (0.0, 0.5),
    "gdp_growth": (-0.2, 0.2),
    "trade_tension": (0.0, 1.0),
    "market_volatility": (0.0, 1.0),
    "cooperation_index": (0.0, 1.0),
    "resource_scarcity": (0.0, 1.0),
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


@dataclass
class WorldState:
    oil_price: float = 1.0
    gold_price: float = 1.0
    food_index: float = 1.0
    energy_cost: float = 1.0
    interest_rate: float = 0.05
    inflation: float = 0.02
    gdp_growth: float = 0.03
    trade_tension: float = 0.2
    market_volatility: float = 0.3
    cooperation_index: float = 0.5
    resource_scarcity: float = 0.3
    event_log: list[str] = field(default_factory=list)
    causal_log: list[dict[str, Any]] = field(default_factory=list)

    def to_prompt_str(self) -> str:
        recent = "; ".join(self.event_log[-3:]) if self.event_log else "None"
        return (
            f"Oil: {self.oil_price:.2f}x | Gold: {self.gold_price:.2f}x | "
            f"Food: {self.food_index:.2f}x | Energy: {self.energy_cost:.2f}x\n"
            f"Interest rate: {self.interest_rate:.1%} | Inflation: {self.inflation:.1%} | "
            f"GDP growth: {self.gdp_growth:.1%} | Trade tension: {self.trade_tension:.2f}\n"
            f"Volatility: {self.market_volatility:.2f} | Cooperation index: {self.cooperation_index:.2f} | "
            f"Resource scarcity: {self.resource_scarcity:.2f}\n"
            f"Recent events: {recent}"
        )

    def derive_round_probabilities(self) -> dict[str, float]:
        positive_gdp = max(0.0, self.gdp_growth)
        competitive_score = (
            0.4 * self.trade_tension
            + 0.3 * self.market_volatility
            + 0.3 * max(0.0, self.oil_price - 1.0)
        )
        cooperative_score = (
            0.5 * self.cooperation_index
            + 0.3 * (1.0 - self.market_volatility)
            + 0.2 * positive_gdp * 5.0
        )
        resource_score = (
            0.5 * self.resource_scarcity
            + 0.3 * max(0.0, self.food_index - 1.0)
            + 0.2 * max(0.0, self.energy_cost - 1.0)
        )
        total = competitive_score + cooperative_score + resource_score + 1e-9
        return {
            "competitive": competitive_score / total,
            "cooperative": cooperative_score / total,
            "resource": resource_score / total,
        }

    def sample_round_type(self, rng: random.Random | None = None) -> str:
        rand = rng or random
        r = rand.random()
        cumulative = 0.0
        for round_type, probability in self.derive_round_probabilities().items():
            cumulative += probability
            if r <= cumulative:
                return round_type
        return "resource"

    def apply_deltas(self, deltas: dict[str, float]) -> None:
        for field_name, delta in deltas.items():
            if field_name not in CLAMP_RANGES:
                continue
            low, high = CLAMP_RANGES[field_name]
            setattr(self, field_name, clamp(getattr(self, field_name) + float(delta), low, high))

    def apply_event(self, event_text: str) -> dict[str, Any]:
        payload = parse_event_payload(event_text, self.to_prompt_str())
        deltas = payload["deltas"]
        self.apply_deltas(deltas)
        self.event_log.append(event_text)
        self.event_log = self.event_log[-5:]
        trace = {
            "event": event_text,
            "deltas": deltas,
            "reasoning": payload.get("causal_reasoning", ""),
            "confidence": payload.get("confidence", 0.0),
            "probabilities_after": self.derive_round_probabilities(),
        }
        self.causal_log.append(trace)
        self.causal_log = self.causal_log[-20:]
        return trace

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ACEWorldEnv:
    world: WorldState = field(default_factory=WorldState)
    agents: list[AgentProfile] = field(default_factory=fresh_agent_profiles)
    round_number: int = 0
    round_history: list[dict[str, Any]] = field(default_factory=list)
    alliances: set[tuple[int, int]] = field(default_factory=set)
    rng_seed: int = 7

    def __post_init__(self) -> None:
        self.rng = random.Random(self.rng_seed)

    def apply_event(self, event_text: str) -> dict[str, Any]:
        return self.world.apply_event(event_text)

    def step(self, actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        self.round_number += 1
        ground_truth = self.world.sample_round_type(self.rng)
        available_ids = [agent.agent_id for agent in self.agents]

        if actions is None:
            probs = self.world.derive_round_probabilities()
            actions = [
                agent.choose_fallback_action(probs, self.round_number, available_ids)
                for agent in self.agents
            ]

        while len(actions) < len(self.agents):
            agent = self.agents[len(actions)]
            actions.append(agent.choose_fallback_action(self.world.derive_round_probabilities(), self.round_number, available_ids))

        action_by_agent = {
            agent.agent_id: _safe_action(actions[idx])
            for idx, agent in enumerate(self.agents)
        }

        self._apply_social_side_effects(action_by_agent, ground_truth)

        results = []
        for agent in self.agents:
            parsed = action_by_agent[agent.agent_id]
            reward_info = compute_total_reward(
                completion_text=str(parsed),
                predicted_round=parsed["predicted_round"],
                action=parsed["action"],
                parameters=parsed["parameters"],
                ground_truth=ground_truth,
                valid_json=True,
                agent_profile=agent,
            )
            success = bool(reward_info["total"] >= 0.8)
            other_actions = {
                other_id: other_action["action"]
                for other_id, other_action in action_by_agent.items()
                if other_id != agent.agent_id
            }
            agent.update_after_round(
                round_number=self.round_number,
                action=parsed["action"],
                predicted_round=parsed["predicted_round"],
                actual_round=ground_truth,
                reward=reward_info["total"],
                success=success,
                other_actions=other_actions,
            )
            results.append(
                {
                    "agent": agent,
                    "action": parsed,
                    "reward": reward_info,
                    "correct": parsed["predicted_round"] == ground_truth,
                    "resources": agent.resources,
                }
            )

        history_entry = {
            "round": self.round_number,
            "ground_truth": ground_truth,
            "event": self.world.event_log[-1] if self.world.event_log else "none",
            "probabilities": self.world.derive_round_probabilities(),
            "alliances": sorted([list(pair) for pair in self.alliances]),
            "results": [
                {
                    "agent_id": item["agent"].agent_id,
                    "name": item["agent"].name,
                    "predicted": item["action"]["predicted_round"],
                    "action": item["action"]["action"],
                    "correct": item["correct"],
                    "reward": item["reward"]["total"],
                }
                for item in results
            ],
        }
        self.round_history.append(history_entry)
        self.round_history = self.round_history[-25:]
        return {"ground_truth": ground_truth, "results": results, "history_entry": history_entry}

    def reset(self) -> None:
        self.world = WorldState()
        self.agents = fresh_agent_profiles()
        self.round_number = 0
        self.round_history = []
        self.alliances = set()
        self.rng = random.Random(self.rng_seed)

    def state(self) -> dict[str, Any]:
        return {
            "world": self.world.snapshot(),
            "round_number": self.round_number,
            "round_probabilities": self.world.derive_round_probabilities(),
            "alliances": sorted([list(pair) for pair in self.alliances]),
            "agents": [
                {
                    "id": agent.agent_id,
                    "name": agent.name,
                    "company_type": agent.company_type,
                    "resources": agent.resources,
                    "trust_scores": agent.trust_scores,
                    "memory": agent.self_memory[-3:],
                }
                for agent in self.agents
            ],
            "round_history": self.round_history,
        }

    def _apply_social_side_effects(self, actions: dict[int, dict[str, Any]], ground_truth: str) -> None:
        proposals = set()
        for agent_id, parsed in actions.items():
            action = parsed["action"]
            params = parsed["parameters"]
            target = params.get("target_id", params.get("partner_id", params.get("team_id")))
            try:
                target_id = int(target)
            except (TypeError, ValueError):
                target_id = None
            if target_id is None or target_id == agent_id:
                continue
            pair = tuple(sorted((agent_id, target_id)))
            if action == "propose_alliance":
                proposals.add(pair)
            elif action == "accept_alliance" or (action == "execute_contract" and ground_truth == "cooperative"):
                self.alliances.add(pair)
            elif action == "betray" and pair in self.alliances:
                self.alliances.remove(pair)
                self._drop_trust(agent_id, target_id, 0.35)
            elif action == "challenge":
                self._drop_trust(agent_id, target_id, 0.08)
        self.alliances.update(proposals)

    def _drop_trust(self, a: int, b: int, amount: float) -> None:
        for agent in self.agents:
            if agent.agent_id == a and b in agent.trust_scores:
                agent.trust_scores[b] = clamp(agent.trust_scores[b] - amount, 0.0, 1.0)
            if agent.agent_id == b and a in agent.trust_scores:
                agent.trust_scores[a] = clamp(agent.trust_scores[a] - amount, 0.0, 1.0)


def _safe_action(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    predicted = raw.get("predicted_round")
    if predicted not in {"cooperative", "competitive", "resource"}:
        predicted = "resource"
    action = raw.get("action")
    if action not in {
        "submit_bid",
        "propose_alliance",
        "accept_alliance",
        "reject_alliance",
        "betray",
        "challenge",
        "allocate_resources",
        "execute_contract",
    }:
        action = "allocate_resources"
    parameters = raw.get("parameters") if isinstance(raw.get("parameters"), dict) else {}
    if action in {"submit_bid", "allocate_resources"}:
        try:
            parameters["amount"] = float(parameters.get("amount", 50))
        except (TypeError, ValueError):
            parameters["amount"] = 50.0
    return {
        "predicted_round": predicted,
        "action": action,
        "parameters": parameters,
        "reasoning": str(raw.get("reasoning", "Fallback or structured decision.")),
    }
