"""Adaptive company agents for EIE Economic Intelligence Engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ace_reward import ACTIONS


ROUND_TYPES = ["cooperative", "competitive", "resource"]
LEARNING_RATE = 0.25
DEFAULT_EPSILON = 0.1
COOPERATIVE_ACTIONS = {"propose_alliance", "accept_alliance", "execute_contract"}
AGGRESSIVE_ACTIONS = {"challenge", "betray", "submit_bid"}
DEFENSIVE_ACTIONS = {"allocate_resources", "execute_contract"}


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
    balance_sheet: dict[str, float] = field(default_factory=lambda: {"cash": 70.0, "debt": 20.0, "assets": 50.0})
    portfolio: dict[str, float] = field(default_factory=dict)
    beliefs: dict[str, float] = field(default_factory=lambda: {rt: 1.0 / 3.0 for rt in ROUND_TYPES})
    self_memory: list[dict[str, Any]] = field(default_factory=list)
    opponent_memory: dict[int, dict[str, float]] = field(default_factory=dict)
    trust_scores: dict[int, float] = field(default_factory=dict)
    strategy_success: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=lambda: {rt: {} for rt in ROUND_TYPES}
    )
    q_values: dict[str, dict[str, float]] = field(
        default_factory=lambda: {action: {rt: 0.0 for rt in ROUND_TYPES} for action in ACTIONS}
    )

    def system_prompt(self, world_state_str: str) -> str:
        return f"""You are {self.name}, a {self.company_type} in EIE.

Objective: {self.primary_objective}

Exposures:
- Oil: {self.stake_oil:+.1f}
- Gold: {self.stake_gold:+.1f}
- Food: {self.stake_food:+.1f}
- Cooperation preference: {self.stake_cooperation:+.1f}
- Risk tolerance: {self.risk_tolerance:.1f}
- Resources: {self.resources:.1f}
- Beliefs: {self.beliefs}
- Balance sheet: {self.balance_sheet}
- Portfolio: {self.portfolio}

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
  "beliefs": {{"competitive": 0.4, "cooperative": 0.3, "resource": 0.3}},
  "factors": {{"oil_exposure": "positive|negative|neutral", "volatility": "high|moderate", "trust_target": 0.5, "past_success": "short memory note"}},
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
                f"- Agent {agent_id}: aggression={model.get('aggression', model.get('aggressiveness_score', 0)):.2f}, "
                f"cooperation={model.get('cooperation', model.get('cooperation_score', 0)):.2f}, "
                f"betrayal_rate={model.get('betrayal_rate', 0):.2f}, "
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
        observed_state: dict[str, Any] | None = None,
        rng: Any | None = None,
        epsilon: float = DEFAULT_EPSILON,
    ) -> dict[str, Any]:
        if observed_state is not None:
            self.update_beliefs(observed_state, world_probs)
        predicted_round = max(self.beliefs, key=self.beliefs.get)
        candidates = self._candidate_actions(predicted_round, available_agents)
        scored = []
        for action, parameters in candidates:
            expected_profit = self._expected_profit(action, parameters, predicted_round, observed_state or {})
            expected_risk = self._expected_risk(action, parameters, observed_state or {})
            trust_alignment = self._trust_alignment(action, parameters)
            historical = self._historical_score(predicted_round, action)
            personality = self._risk_score(action)
            identity = self._identity_bias(action, predicted_round)
            opponent_adjustment = self._opponent_adjustment(action, parameters)
            q_value = self._q_value(action, predicted_round)
            score = (
                expected_profit
                - expected_risk
                + 0.30 * q_value
                + 0.20 * trust_alignment
                + 0.20 * personality
                + 0.35 * identity
                + 0.20 * historical
                + opponent_adjustment
            )
            scored.append((score, action, dict(parameters)))
        scored.sort(reverse=True, key=lambda item: item[0])
        explored = False
        if rng is not None and round_number > 1 and rng.random() < epsilon:
            _, action, parameters = rng.choice(scored)
            explored = True
        else:
            _, action, parameters = scored[0]
        return {
            "predicted_round": predicted_round,
            "action": action,
            "parameters": parameters,
            "beliefs": dict(self.beliefs),
            "factors": self._decision_factors(action, parameters, observed_state or {}, predicted_round, explored),
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
        reward_components: dict[str, float] | None = None,
    ) -> None:
        self.resources = max(0.0, self.resources + reward * 8.0)
        self.balance_sheet["cash"] = max(0.0, self.balance_sheet.get("cash", 0.0) + reward * 5.0)
        self.balance_sheet["assets"] = max(0.0, self.balance_sheet.get("assets", 0.0) + reward * 2.0)
        self.self_memory.append(
            {
                "round": round_number,
                "action": action,
                "predicted_round": predicted_round,
                "actual_round": actual_round,
                "reward": round(reward, 3),
                "reward_components": reward_components or {},
                "success": bool(success),
                "opponents": dict(other_actions),
            }
        )
        self.self_memory = self.self_memory[-10:]

        action_stats = self.strategy_success.setdefault(actual_round, {}).setdefault(
            action, {"attempts": 0, "successes": 0}
        )
        action_stats["attempts"] += 1
        if success:
            action_stats["successes"] += 1
        self.q_values.setdefault(action, {rt: 0.0 for rt in ROUND_TYPES})
        old_q = self.q_values[action].get(actual_round, 0.0)
        self.q_values[action][actual_round] = old_q + LEARNING_RATE * (reward - old_q)

        for other_id, other_action in other_actions.items():
            if other_id == self.agent_id:
                continue
            model = self.opponent_memory.setdefault(
                other_id,
                {"aggression": 0.0, "cooperation": 0.0, "betrayal_rate": 0.0, "observations": 0.0},
            )
            model["observations"] = min(20.0, model.get("observations", 0.0) + 1.0)
            if other_action in {"challenge", "submit_bid"}:
                model["aggression"] = _clamp(model.get("aggression", 0.0) + 0.1)
            if other_action in COOPERATIVE_ACTIONS:
                model["cooperation"] = _clamp(model.get("cooperation", 0.0) + 0.1)
                self.trust_scores[other_id] = _clamp(self.trust_scores.get(other_id, 0.5) + 0.06)
            if other_action == "betray":
                model["aggression"] = _clamp(model.get("aggression", 0.0) + 0.1)
                model["betrayal_rate"] = _clamp(model.get("betrayal_rate", 0.0) + 0.2)
                self.trust_scores[other_id] = _clamp(self.trust_scores.get(other_id, 0.5) - 0.5)
            if other_action == "challenge":
                self.trust_scores[other_id] = _clamp(self.trust_scores.get(other_id, 0.5) - 0.1)

    def update_beliefs(self, observed_state: dict[str, Any], world_probs: dict[str, float]) -> None:
        oil = float(observed_state.get("oil_price", 1.0))
        volatility = float(observed_state.get("market_volatility", 0.3))
        tension = float(observed_state.get("trade_tension", 0.2))
        scarcity = float(observed_state.get("resource_scarcity", 0.3))
        food = float(observed_state.get("food_index", 1.0))
        energy = float(observed_state.get("energy_cost", 1.0))
        cooperation = float(observed_state.get("cooperation_index", 0.5))
        liquidity = float(observed_state.get("liquidity_index", 0.7))

        likelihood = {
            "competitive": 0.15 + 0.35 * tension + 0.3 * volatility + 0.2 * max(0.0, oil - 1.0),
            "cooperative": 0.15 + 0.45 * cooperation + 0.2 * (1.0 - volatility) + 0.1 * liquidity,
            "resource": 0.15 + 0.4 * scarcity + 0.2 * max(0.0, food - 1.0) + 0.2 * max(0.0, energy - 1.0),
        }
        posterior = {}
        for round_type in ROUND_TYPES:
            prior = 0.55 * self.beliefs.get(round_type, 1 / 3) + 0.45 * world_probs.get(round_type, 1 / 3)
            posterior[round_type] = max(0.001, prior * likelihood[round_type])
        total = sum(posterior.values())
        self.beliefs = {round_type: posterior[round_type] / total for round_type in ROUND_TYPES}

    def _candidate_actions(self, predicted_round: str, available_agents: list[int]) -> list[tuple[str, dict[str, Any]]]:
        partner = self._select_partner(available_agents, prefer_trust=True)
        risky_partner = self._select_partner(available_agents, prefer_trust=False)
        if "Food" in self.company_type:
            return [
                ("allocate_resources", {"amount": 60}),
                ("propose_alliance", {"target_id": partner}),
                ("execute_contract", {"team_id": partner}),
                ("submit_bid", {"amount": 30, "partner_id": risky_partner}),
            ]
        if "Central Bank" in self.company_type:
            return [
                ("execute_contract", {"team_id": partner}),
                ("allocate_resources", {"amount": 45}),
                ("propose_alliance", {"target_id": partner}),
                ("accept_alliance", {"target_id": partner}),
            ]
        if "Logistics" in self.company_type:
            return [
                ("allocate_resources", {"amount": 70}),
                ("execute_contract", {"team_id": partner}),
                ("propose_alliance", {"target_id": partner}),
                ("submit_bid", {"amount": 45, "partner_id": risky_partner}),
            ]
        if "Insurance" in self.company_type:
            return [
                ("allocate_resources", {"amount": 55}),
                ("execute_contract", {"team_id": partner}),
                ("submit_bid", {"amount": 35, "partner_id": risky_partner}),
                ("reject_alliance", {"target_id": risky_partner}),
            ]
        if "Technology" in self.company_type:
            return [
                ("submit_bid", {"amount": 65}),
                ("execute_contract", {"team_id": partner}),
                ("propose_alliance", {"target_id": partner}),
                ("challenge", {"target_id": risky_partner}),
            ]
        if "Hedge Fund" in self.company_type:
            return [
                ("submit_bid", {"amount": 85}),
                ("challenge", {"target_id": risky_partner}),
                ("betray", {"partner_id": risky_partner}),
                ("allocate_resources", {"amount": 40}),
            ]
        if "Energy" in self.company_type:
            return [
                ("challenge", {"target_id": risky_partner}),
                ("submit_bid", {"amount": 85}),
                ("betray", {"partner_id": risky_partner}),
                ("execute_contract", {"team_id": partner}),
            ]
        if predicted_round == "competitive":
            return [
                ("challenge", {"target_id": risky_partner}),
                ("submit_bid", {"amount": 80}),
                ("betray", {"partner_id": risky_partner}),
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
        q_value = self._q_value(action, predicted_round)
        if abs(q_value) > 1e-9:
            return _clamp((q_value + 2.0) / 4.0)
        stats = self.strategy_success.get(predicted_round, {}).get(action)
        if not stats:
            return 0.3
        return _clamp(stats.get("successes", 0) / max(1, stats.get("attempts", 0)))

    def _expected_profit(self, action: str, parameters: dict[str, Any], predicted_round: str, observed_state: dict[str, Any]) -> float:
        oil = float(observed_state.get("oil_price", 1.0))
        food = float(observed_state.get("food_index", 1.0))
        gold = float(observed_state.get("gold_price", 1.0))
        market_return = (
            self.portfolio.get("oil_exposure", self.stake_oil) * (oil - 1.0)
            + self.portfolio.get("food_exposure", self.stake_food) * (food - 1.0)
            + self.portfolio.get("gold_exposure", self.stake_gold) * (gold - 1.0)
        )
        action_bonus = {
            "challenge": 0.35 if predicted_round == "competitive" else -0.1,
            "betray": 0.3 if predicted_round == "competitive" else -0.2,
            "propose_alliance": 0.3 if predicted_round == "cooperative" else 0.0,
            "accept_alliance": 0.25 if predicted_round == "cooperative" else 0.0,
            "allocate_resources": 0.35 if predicted_round == "resource" else 0.05,
            "execute_contract": 0.18,
            "submit_bid": 0.25 if predicted_round == "competitive" else 0.08,
        }.get(action, 0.0)
        return market_return + action_bonus

    def _expected_risk(self, action: str, parameters: dict[str, Any], observed_state: dict[str, Any]) -> float:
        volatility = float(observed_state.get("market_volatility", 0.3))
        credit = float(observed_state.get("credit_spread", 0.25))
        action_risk = {
            "challenge": 0.55,
            "betray": 0.7,
            "submit_bid": float(parameters.get("amount", 50)) / 140.0,
            "propose_alliance": 0.15,
            "accept_alliance": 0.12,
            "reject_alliance": 0.1,
            "allocate_resources": 0.22,
            "execute_contract": 0.08,
        }.get(action, 0.2)
        return (volatility + 0.5 * credit) * action_risk * (1.15 - self.risk_tolerance)

    def _trust_alignment(self, action: str, parameters: dict[str, Any]) -> float:
        partner = parameters.get("target_id", parameters.get("partner_id", parameters.get("team_id")))
        trust = self.trust_scores.get(int(partner), 0.5) if partner is not None else 0.5
        if action in {"propose_alliance", "accept_alliance", "execute_contract"}:
            return trust
        if action in {"challenge", "betray"}:
            return 1.0 - trust
        return 0.5

    def _opponent_adjustment(self, action: str, parameters: dict[str, Any]) -> float:
        partner = parameters.get("target_id", parameters.get("partner_id", parameters.get("team_id")))
        if partner is None:
            return 0.0
        model = self.opponent_memory.get(int(partner), {})
        betrayal_rate = model.get("betrayal_rate", 0.0)
        aggression = model.get("aggression", model.get("aggressiveness_score", 0.0))
        if action in COOPERATIVE_ACTIONS and betrayal_rate > 0.5:
            return -0.45
        if action in COOPERATIVE_ACTIONS and aggression > 0.5:
            return -0.2
        if action in DEFENSIVE_ACTIONS and aggression > 0.5:
            return 0.2
        if action in {"challenge", "betray"} and betrayal_rate > 0.5:
            return 0.15
        return 0.0

    def _select_partner(self, available_agents: list[int], *, prefer_trust: bool) -> int:
        candidates = [agent_id for agent_id in available_agents if agent_id != self.agent_id]
        if not candidates:
            return self.agent_id
        if prefer_trust:
            return max(
                candidates,
                key=lambda agent_id: (
                    self.trust_scores.get(agent_id, 0.5)
                    - self.opponent_memory.get(agent_id, {}).get("betrayal_rate", 0.0)
                    - 0.3 * self.opponent_memory.get(agent_id, {}).get("aggression", 0.0)
                ),
            )
        return min(candidates, key=lambda agent_id: self.trust_scores.get(agent_id, 0.5))

    def _q_value(self, action: str, regime: str) -> float:
        # Supports both current Q[action][regime] and older Q[regime][action] snapshots.
        if action in self.q_values:
            return float(self.q_values.get(action, {}).get(regime, 0.0))
        return float(self.q_values.get(regime, {}).get(action, 0.0))

    def _risk_score(self, action: str) -> float:
        if action in {"challenge", "betray", "submit_bid"}:
            return self.risk_tolerance
        if action in {"propose_alliance", "accept_alliance", "execute_contract"}:
            return _clamp(self.stake_cooperation)
        return 1.0 - abs(self.risk_tolerance - 0.5)

    def _identity_bias(self, action: str, predicted_round: str) -> float:
        if "Energy" in self.company_type:
            if action in {"challenge", "submit_bid"} and predicted_round == "competitive":
                return 1.0
            if action == "execute_contract":
                return 0.25
        if "Food" in self.company_type:
            if action in {"allocate_resources", "propose_alliance", "execute_contract"}:
                return 1.0
            if action in {"challenge", "betray"}:
                return -0.6
        if "Hedge Fund" in self.company_type:
            if action in {"submit_bid", "challenge", "betray"}:
                return 1.0
            if action == "allocate_resources":
                return 0.25
        if "Central Bank" in self.company_type:
            if action in {"execute_contract", "propose_alliance", "allocate_resources"}:
                return 1.0
            if action in {"challenge", "betray", "submit_bid"}:
                return -0.8
        if "Logistics" in self.company_type:
            if action in {"allocate_resources", "execute_contract", "propose_alliance"}:
                return 0.9
            if action == "betray":
                return -0.5
        if "Insurance" in self.company_type:
            if action in {"allocate_resources", "execute_contract", "reject_alliance"}:
                return 0.9
            if action in {"challenge", "betray"}:
                return -0.7
        if "Technology" in self.company_type:
            if action in {"submit_bid", "execute_contract", "propose_alliance"}:
                return 0.85
            if action == "challenge" and predicted_round == "competitive":
                return 0.45
        return 0.0

    def _decision_factors(
        self,
        action: str,
        parameters: dict[str, Any],
        observed_state: dict[str, Any],
        predicted_round: str,
        explored: bool = False,
    ) -> dict[str, Any]:
        partner = parameters.get("target_id", parameters.get("partner_id", parameters.get("team_id")))
        trust = self.trust_scores.get(int(partner), 0.5) if partner is not None else None
        return {
            "oil_exposure": "positive" if self.stake_oil > 0 else "negative" if self.stake_oil < 0 else "neutral",
            "food_exposure": "positive" if self.stake_food > 0 else "negative" if self.stake_food < 0 else "neutral",
            "volatility": "high" if float(observed_state.get("market_volatility", 0.3)) > 0.55 else "moderate",
            "credit": "tight" if float(observed_state.get("credit_spread", 0.25)) > 0.45 else "normal",
            "trust_target": None if trust is None else round(trust, 2),
            "opponent_betrayal_rate": None
            if partner is None
            else round(self.opponent_memory.get(int(partner), {}).get("betrayal_rate", 0.0), 2),
            "opponent_aggression": None
            if partner is None
            else round(self.opponent_memory.get(int(partner), {}).get("aggression", 0.0), 2),
            "past_success": self._best_memory_snippet(predicted_round),
            "q_value": round(self._q_value(action, predicted_round), 3),
            "exploration": bool(explored),
            "personality": self.company_type,
            "chosen_bias": round(self._identity_bias(action, predicted_round), 2),
        }

    def _best_memory_snippet(self, predicted_round: str) -> str:
        stats = self.strategy_success.get(predicted_round, {})
        if not stats:
            return "no strong habit yet"
        best_action = max(
            stats,
            key=lambda action: stats[action].get("successes", 0) / max(1, stats[action].get("attempts", 0)),
        )
        item = stats[best_action]
        rate = item.get("successes", 0) / max(1, item.get("attempts", 0))
        return f"{best_action} worked {rate:.0%} in {predicted_round} rounds"

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
        emoji="🛢️",
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
        emoji="🌾",
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
        emoji="📈",
        primary_objective="Generate alpha from volatility and market dislocations",
        stake_oil=0.4,
        stake_gold=0.7,
        stake_food=0.3,
        stake_cooperation=-0.5,
        risk_tolerance=0.95,
    ),
    AgentProfile(
        agent_id=3,
        name="CentralBank of EIE",
        company_type="Central Bank / Regulator",
        emoji="🏦",
        primary_objective="Maintain market stability, control inflation, and prevent systemic risk",
        stake_oil=-0.3,
        stake_gold=-0.2,
        stake_food=-0.5,
        stake_cooperation=0.9,
        risk_tolerance=0.1,
    ),
    AgentProfile(
        agent_id=4,
        name="LogiChain Global",
        company_type="Logistics & Supply Chain Operator",
        emoji="🚢",
        primary_objective="Preserve shipping capacity and arbitrage supply-chain bottlenecks",
        stake_oil=-0.4,
        stake_gold=0.0,
        stake_food=-0.3,
        stake_cooperation=0.6,
        risk_tolerance=0.45,
    ),
    AgentProfile(
        agent_id=5,
        name="ShieldRe Insurance",
        company_type="Insurance & Risk Underwriter",
        emoji="🛡️",
        primary_objective="Minimise systemic losses by hedging volatility and avoiding unreliable partners",
        stake_oil=-0.2,
        stake_gold=0.4,
        stake_food=-0.1,
        stake_cooperation=0.4,
        risk_tolerance=0.25,
    ),
    AgentProfile(
        agent_id=6,
        name="NovaTech Systems",
        company_type="Technology Infrastructure Firm",
        emoji="💻",
        primary_objective="Capture growth contracts while maintaining compute and infrastructure resilience",
        stake_oil=-0.3,
        stake_gold=0.1,
        stake_food=0.0,
        stake_cooperation=0.2,
        risk_tolerance=0.65,
    ),
]


def fresh_agent_profiles() -> list[AgentProfile]:
    """Return independent mutable agent profiles for a new simulation."""
    import copy

    agents = copy.deepcopy(AGENT_PROFILES)
    ids = [agent.agent_id for agent in agents]
    for agent in agents:
        agent.trust_scores = {other_id: 0.5 for other_id in ids if other_id != agent.agent_id}
        agent.portfolio = {
            "oil_exposure": agent.stake_oil,
            "food_exposure": agent.stake_food,
            "gold_exposure": agent.stake_gold,
            "cash": 1.0 - min(0.8, abs(agent.stake_oil) * 0.2 + abs(agent.stake_food) * 0.2),
        }
    return agents
