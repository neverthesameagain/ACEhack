"""Structured economic world and multi-agent simulator for EIE Economic Intelligence Engine."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

from ace_agents import AgentProfile, fresh_agent_profiles
from ace_reward import compute_total_reward
from ace_text_inject import parse_event_payload


ROUND_TYPES = ["competitive", "cooperative", "resource"]

CLAMP_RANGES = {
    "oil_price": (0.5, 2.5),
    "gold_price": (0.5, 2.5),
    "food_index": (0.5, 2.5),
    "energy_cost": (0.5, 2.5),
    "interest_rate": (0.0, 0.25),
    "inflation": (0.0, 0.5),
    "gdp_growth": (-0.2, 0.2),
    "trade_tension": (0.0, 1.0),
    "market_volatility": (0.0, 1.0),
    "cooperation_index": (0.0, 1.0),
    "resource_scarcity": (0.0, 1.0),
    "liquidity_index": (0.0, 1.0),
    "credit_spread": (0.0, 1.0),
    "geopolitical_risk": (0.0, 1.0),
    "supply_chain_stability": (0.0, 1.0),
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
    sector_health: dict[str, float] = field(
        default_factory=lambda: {
            "energy": 0.0,
            "agriculture": 0.0,
            "finance": 0.0,
            "manufacturing": 0.0,
            "technology": 0.0,
        }
    )
    liquidity_index: float = 0.7
    credit_spread: float = 0.25
    geopolitical_risk: float = 0.2
    supply_chain_stability: float = 0.75
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
            f"Liquidity: {self.liquidity_index:.2f} | Credit spread: {self.credit_spread:.2f} | "
            f"Geopolitical risk: {self.geopolitical_risk:.2f} | Supply chains: {self.supply_chain_stability:.2f}\n"
            f"Sector health: {self.sector_health}\n"
            f"Economic regime: {self.economic_regime()}\n"
            f"Recent events: {recent}"
        )

    def economic_regime(self) -> str:
        if self.market_volatility > 0.72 and self.liquidity_index < 0.45:
            return "crisis"
        if self.inflation > 0.08 and self.gdp_growth < 0.0:
            return "stagflation"
        if self.gdp_growth < -0.02:
            return "recession"
        if self.inflation > 0.06:
            return "inflationary"
        if self.gdp_growth > 0.04 and self.market_volatility < 0.35:
            return "growth"
        return "mixed"

    def derive_round_probabilities(self) -> dict[str, float]:
        positive_gdp = max(0.0, self.gdp_growth)
        competitive_score = (
            0.4 * self.trade_tension
            + 0.3 * self.market_volatility
            + 0.3 * max(0.0, self.oil_price - 1.0)
            + 0.2 * self.geopolitical_risk
            + 0.1 * self.credit_spread
        )
        cooperative_score = (
            0.5 * self.cooperation_index
            + 0.3 * (1.0 - self.market_volatility)
            + 0.2 * positive_gdp * 5.0
            + 0.1 * self.liquidity_index
        )
        resource_score = (
            0.5 * self.resource_scarcity
            + 0.3 * max(0.0, self.food_index - 1.0)
            + 0.2 * max(0.0, self.energy_cost - 1.0)
            + 0.25 * (1.0 - self.supply_chain_stability)
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
            if field_name == "sector_health" and isinstance(delta, dict):
                for sector, sector_delta in delta.items():
                    if sector in self.sector_health:
                        self.sector_health[sector] = clamp(
                            self.sector_health[sector] + float(sector_delta), -1.0, 1.0
                        )
                continue
            if field_name.startswith("sector_"):
                sector = field_name.removeprefix("sector_")
                if sector in self.sector_health:
                    self.sector_health[sector] = clamp(self.sector_health[sector] + float(delta), -1.0, 1.0)
                continue
            if field_name not in CLAMP_RANGES:
                continue
            low, high = CLAMP_RANGES[field_name]
            setattr(self, field_name, clamp(getattr(self, field_name) + float(delta), low, high))

    def apply_endogenous_dynamics(self, shock: bool = False) -> dict[str, float]:
        """Simple feedback loops and decay toward baseline."""
        deltas = {
            "inflation": 0.03 * max(0.0, self.energy_cost - 1.0) + 0.02 * max(0.0, self.food_index - 1.0),
            "market_volatility": 0.04 * self.trade_tension + 0.03 * self.resource_scarcity + 0.04 * self.geopolitical_risk,
            "gdp_growth": -0.015 * self.credit_spread - 0.01 * max(0.0, self.inflation - 0.05),
        }
        if self.inflation > 0.05:
            deltas["interest_rate"] = 0.005
            deltas["liquidity_index"] = -0.025
            deltas["credit_spread"] = 0.02
        if self.gdp_growth < 0.0:
            deltas["cooperation_index"] = 0.02
        if not shock:
            deltas.update(
                {
                    "oil_price": (1.0 - self.oil_price) * 0.06,
                    "gold_price": (1.0 - self.gold_price) * 0.05,
                    "food_index": (1.0 - self.food_index) * 0.04,
                    "energy_cost": (1.0 - self.energy_cost) * 0.05,
                    "trade_tension": (0.2 - self.trade_tension) * 0.03,
                    "market_volatility": deltas["market_volatility"] + (0.3 - self.market_volatility) * 0.04,
                    "resource_scarcity": (0.3 - self.resource_scarcity) * 0.03,
                    "supply_chain_stability": (0.75 - self.supply_chain_stability) * 0.04,
                }
            )
        self.apply_deltas(deltas)
        return deltas

    def noisy_observation(self, rng: random.Random, sigma: float = 0.045) -> dict[str, Any]:
        observed: dict[str, Any] = {}
        for field_name, (low, high) in CLAMP_RANGES.items():
            observed[field_name] = clamp(getattr(self, field_name) + rng.gauss(0.0, sigma), low, high)
        observed["sector_health"] = {
            sector: clamp(value + rng.gauss(0.0, sigma), -1.0, 1.0)
            for sector, value in self.sector_health.items()
        }
        observed["economic_regime"] = self.economic_regime()
        return observed

    def apply_event(self, event_text: str, provider: str | None = None, debug: bool = False) -> dict[str, Any]:
        payload = parse_event_payload(event_text, self.to_prompt_str(), provider=provider, debug=debug)
        deltas = payload["deltas"]
        self.apply_deltas(deltas)
        endogenous = self.apply_endogenous_dynamics(shock=True)
        self.event_log.append(event_text)
        self.event_log = self.event_log[-5:]
        trace = {
            "event": event_text,
            "deltas": deltas,
            "endogenous_deltas": endogenous,
            "reasoning": payload.get("reasoning", payload.get("causal_reasoning", "")),
            "causal_reasoning": payload.get("causal_reasoning", payload.get("reasoning", "")),
            "confidence": payload.get("confidence", 0.0),
            "event_type": payload.get("event_type", "demand shock"),
            "affected_sectors": payload.get("affected_sectors", []),
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
    previous_market: dict[str, float] = field(default_factory=dict)
    interaction_log: list[str] = field(default_factory=list)
    world_history: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.rng_seed)
        self.previous_market = self._market_snapshot()
        self._record_world_history()

    def apply_event(self, event_text: str, provider: str | None = None, debug: bool = False) -> dict[str, Any]:
        trace = self.world.apply_event(event_text, provider=provider, debug=debug)
        self._record_world_history()
        return trace

    def step(self, actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        self.round_number += 1
        self.world.apply_endogenous_dynamics(shock=False)
        ground_truth = self.world.sample_round_type(self.rng)
        available_ids = [agent.agent_id for agent in self.agents]
        observations = {
            agent.agent_id: self.world.noisy_observation(self.rng, sigma=0.035 + 0.03 * self.world.market_volatility)
            for agent in self.agents
        }

        if actions is None:
            probs = self.world.derive_round_probabilities()
            actions = [
                agent.choose_fallback_action(
                    probs,
                    self.round_number,
                    available_ids,
                    observations[agent.agent_id],
                    rng=self.rng,
                )
                for agent in self.agents
            ]

        while len(actions) < len(self.agents):
            agent = self.agents[len(actions)]
            actions.append(
                agent.choose_fallback_action(
                    self.world.derive_round_probabilities(),
                    self.round_number,
                    available_ids,
                    observations[agent.agent_id],
                    rng=self.rng,
                )
            )

        action_by_agent = {
            agent.agent_id: _safe_action(actions[idx])
            for idx, agent in enumerate(self.agents)
        }

        social_bonus = self._apply_social_side_effects(action_by_agent, ground_truth)

        results = []
        for agent in self.agents:
            agent.update_beliefs(observations[agent.agent_id], self.world.derive_round_probabilities())
            parsed = action_by_agent[agent.agent_id]
            market_return = self._market_return(agent)
            reward_info = compute_total_reward(
                completion_text=str(parsed),
                predicted_round=parsed["predicted_round"],
                action=parsed["action"],
                parameters=parsed["parameters"],
                ground_truth=ground_truth,
                valid_json=True,
                agent_profile=agent,
            )
            reward_info["market_return"] = market_return
            reward_info["market"] = market_return
            reward_info["total"] += market_return
            if agent.name.startswith("CentralBank"):
                reward_info["stability"] = self._stability_bonus()
                reward_info["total"] += reward_info["stability"]
            reward_info["social_bonus"] = social_bonus.get(agent.agent_id, 0.0)
            reward_info["social"] = reward_info["social_bonus"]
            reward_info["total"] += reward_info["social_bonus"]
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
                reward_components={
                    "total": reward_info["total"],
                    "inference": reward_info.get("inference", 0.0),
                    "action": reward_info.get("action", 0.0),
                    "social": reward_info.get("social", 0.0),
                    "market": reward_info.get("market", 0.0),
                    "behavior": reward_info.get("behavior", 0.0),
                },
            )
            results.append(
                {
                    "agent": agent,
                    "action": parsed,
                    "reward": reward_info,
                    "correct": parsed["predicted_round"] == ground_truth,
                    "resources": agent.resources,
                    "beliefs": dict(agent.beliefs),
                    "factors": parsed.get("factors", {}),
                    "observed_state": observations[agent.agent_id],
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
                    "beliefs": item["beliefs"],
                    "factors": item["factors"],
                }
                for item in results
            ],
            "interaction_log": self.interaction_log[-8:],
        }
        self.round_history.append(history_entry)
        self.round_history = self.round_history[-25:]
        self.previous_market = self._market_snapshot()
        self._record_world_history()
        return {"ground_truth": ground_truth, "results": results, "history_entry": history_entry}

    def reset(self) -> None:
        self.world = WorldState()
        self.agents = fresh_agent_profiles()
        self.round_number = 0
        self.round_history = []
        self.alliances = set()
        self.interaction_log = []
        self.rng = random.Random(self.rng_seed)
        self.previous_market = self._market_snapshot()
        self.world_history = []
        self._record_world_history()

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
                    "balance_sheet": agent.balance_sheet,
                    "portfolio": agent.portfolio,
                    "beliefs": agent.beliefs,
                    "trust_scores": agent.trust_scores,
                    "opponent_memory": agent.opponent_memory,
                    "q_values": agent.q_values,
                    "memory": agent.self_memory[-3:],
                }
                for agent in self.agents
            ],
            "round_history": self.round_history,
            "interaction_log": self.interaction_log[-20:],
            "world_history": self.world_history[-30:],
        }

    def _market_snapshot(self) -> dict[str, float]:
        return {
            "oil_price": self.world.oil_price,
            "food_index": self.world.food_index,
            "gold_price": self.world.gold_price,
        }

    def _record_world_history(self) -> None:
        self.world_history.append(
            {
                "round": float(self.round_number),
                "oil_price": self.world.oil_price,
                "market_volatility": self.world.market_volatility,
                "cooperation_index": self.world.cooperation_index,
                "resource_scarcity": self.world.resource_scarcity,
            }
        )
        self.world_history = self.world_history[-40:]

    def _market_return(self, agent: AgentProfile) -> float:
        oil_delta = self.world.oil_price - self.previous_market.get("oil_price", 1.0)
        food_delta = self.world.food_index - self.previous_market.get("food_index", 1.0)
        gold_delta = self.world.gold_price - self.previous_market.get("gold_price", 1.0)
        return 0.8 * (
            agent.portfolio.get("oil_exposure", agent.stake_oil) * oil_delta
            + agent.portfolio.get("food_exposure", agent.stake_food) * food_delta
            + agent.portfolio.get("gold_exposure", agent.stake_gold) * gold_delta
        )

    def _stability_bonus(self) -> float:
        inflation_penalty = max(0.0, self.world.inflation - 0.05)
        volatility_penalty = max(0.0, self.world.market_volatility - 0.45)
        return max(-0.4, 0.3 - 2.0 * inflation_penalty - 0.8 * volatility_penalty)

    def _apply_social_side_effects(self, actions: dict[int, dict[str, Any]], ground_truth: str) -> dict[int, float]:
        proposals = set()
        social_bonus = {agent.agent_id: 0.0 for agent in self.agents}
        names = {agent.agent_id: agent.name for agent in self.agents}
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
                target_action = actions.get(target_id, {}).get("action")
                if target_action in {"accept_alliance", "propose_alliance", "execute_contract", "allocate_resources"}:
                    self.alliances.add(pair)
                    self._raise_trust(agent_id, target_id, 0.1)
                    social_bonus[agent_id] += 0.35
                    social_bonus[target_id] += 0.35
                    self.interaction_log.append(
                        f"🤝 {names[agent_id]} forged an ALLIANCE with {names[target_id]} → both gain, trust rises."
                    )
                elif target_action == "betray":
                    self._drop_trust(agent_id, target_id, 0.5)
                    social_bonus[agent_id] -= 0.8
                    social_bonus[target_id] += 0.7
                    self.interaction_log.append(
                        f"💥 {names[target_id]} BETRAYED {names[agent_id]}'s alliance offer → short-term gain, trust collapses."
                    )
                else:
                    proposals.add(pair)
                    self.interaction_log.append(
                        f"🤝 {names[agent_id]} offered an alliance to {names[target_id]} → no commitment yet."
                    )
            elif action == "accept_alliance" or (action == "execute_contract" and ground_truth == "cooperative"):
                self.alliances.add(pair)
                self._raise_trust(agent_id, target_id, 0.1)
                social_bonus[agent_id] += 0.2
                social_bonus[target_id] += 0.2
                self.interaction_log.append(f"🤝 {names[agent_id]} stabilized cooperation with {names[target_id]} → trust rises.")
            elif action == "betray" and pair in self.alliances:
                self.alliances.remove(pair)
                self._drop_trust(agent_id, target_id, 0.5)
                social_bonus[agent_id] += 0.8 if ground_truth == "competitive" else 0.25
                social_bonus[target_id] -= 0.8
                self.interaction_log.append(
                    f"💥 {names[agent_id]} BETRAYED {names[target_id]} → short-term gain, trust collapses."
                )
            elif action == "challenge":
                self._drop_trust(agent_id, target_id, 0.1)
                winner = agent_id if self.rng.random() < 0.55 else target_id
                loser = target_id if winner == agent_id else agent_id
                social_bonus[winner] += 0.35
                social_bonus[loser] -= 0.25
                self.interaction_log.append(f"⚔️ {names[agent_id]} CHALLENGED {names[target_id]} → {names[winner]} won the clash.")
        self.alliances.update(proposals)
        self.interaction_log = self.interaction_log[-30:]
        return social_bonus

    def _drop_trust(self, a: int, b: int, amount: float) -> None:
        for agent in self.agents:
            if agent.agent_id == a and b in agent.trust_scores:
                agent.trust_scores[b] = clamp(agent.trust_scores[b] - amount, 0.0, 1.0)
            if agent.agent_id == b and a in agent.trust_scores:
                agent.trust_scores[a] = clamp(agent.trust_scores[a] - amount, 0.0, 1.0)

    def _raise_trust(self, a: int, b: int, amount: float) -> None:
        for agent in self.agents:
            if agent.agent_id == a and b in agent.trust_scores:
                agent.trust_scores[b] = clamp(agent.trust_scores[b] + amount, 0.0, 1.0)
            if agent.agent_id == b and a in agent.trust_scores:
                agent.trust_scores[a] = clamp(agent.trust_scores[a] + amount, 0.0, 1.0)


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
        "beliefs": raw.get("beliefs") if isinstance(raw.get("beliefs"), dict) else {},
        "factors": raw.get("factors") if isinstance(raw.get("factors"), dict) else {},
        "reasoning": str(raw.get("reasoning", "Fallback or structured decision.")),
    }
