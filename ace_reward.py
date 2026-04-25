"""Decoupled rewards for ACE++ Option B.

The reward is intentionally decomposed so a policy cannot improve by only
gaming bid size. Inference, action quality, formatting, personality alignment,
and learned behavior are logged separately.
"""

from __future__ import annotations

from typing import Any


ROUND_TYPES = ["cooperative", "competitive", "resource"]
ACTIONS = [
    "submit_bid",
    "propose_alliance",
    "accept_alliance",
    "reject_alliance",
    "betray",
    "challenge",
    "allocate_resources",
    "execute_contract",
]

INFERENCE_WEIGHT = 1.0
ACTION_WEIGHT = 0.5
FORMAT_WEIGHT = 0.25
PERSONALITY_WEIGHT = 0.2
BEHAVIOR_WEIGHT = 0.2


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def compute_inference_reward(predicted_round: str, ground_truth: str) -> float:
    """Pure hidden-state inference reward."""
    return 1.0 if predicted_round == ground_truth else 0.0


def compute_action_reward(action: str, parameters: dict[str, Any], ground_truth: str) -> float:
    """Action quality independent of the prediction string."""
    correct_actions = {
        "cooperative": {"submit_bid", "propose_alliance", "accept_alliance", "execute_contract"},
        "competitive": {"challenge", "betray", "submit_bid"},
        "resource": {"allocate_resources", "execute_contract", "submit_bid"},
    }
    family_reward = 0.5 if action in correct_actions.get(ground_truth, set()) else 0.0

    amount_reward = 0.0
    if action in {"submit_bid", "allocate_resources"}:
        try:
            amount = float(parameters.get("amount", 50.0))
        except (TypeError, ValueError):
            amount = 50.0
        normalized = _clamp(amount / 100.0)
        if ground_truth == "cooperative":
            amount_reward = 1.0 - normalized
        elif ground_truth == "competitive":
            amount_reward = normalized
        else:
            amount_reward = 1.0 - abs(normalized - 0.5) * 2.0

    return _clamp(family_reward + 0.4 * amount_reward)


def compute_format_reward(completion_text: str, valid_json: bool) -> float:
    if not valid_json:
        return -0.3
    if len(completion_text) <= 320:
        return 0.3
    if len(completion_text) <= 520:
        return 0.1
    return 0.0


def compute_personality_reward(action: str, ground_truth: str, agent_profile: Any | None = None) -> float:
    if agent_profile is None:
        return 0.0

    cooperation = float(getattr(agent_profile, "stake_cooperation", 0.0))
    risk = float(getattr(agent_profile, "risk_tolerance", 0.5))

    if action in {"propose_alliance", "accept_alliance", "execute_contract"} and cooperation > 0.4:
        return 1.0
    if action in {"challenge", "betray"} and (cooperation < -0.2 or risk > 0.75):
        return 1.0
    if action == "allocate_resources" and ground_truth == "resource":
        return 0.8
    if action == "submit_bid" and ground_truth == "competitive" and risk > 0.6:
        return 0.8
    if action == "submit_bid" and ground_truth == "cooperative" and cooperation > 0.0:
        return 0.6
    return 0.0


def compute_behavior_reward(
    action: str,
    ground_truth: str,
    agent_profile: Any | None = None,
) -> float:
    """Reward choosing actions that have worked for this agent before."""
    if agent_profile is None:
        return 0.0
    strategy_success = getattr(agent_profile, "strategy_success", {})
    round_stats = strategy_success.get(ground_truth, {})
    attempts = float(round_stats.get(action, {}).get("attempts", 0))
    successes = float(round_stats.get(action, {}).get("successes", 0))
    if attempts <= 0:
        return 0.0
    return _clamp(successes / attempts)


def compute_total_reward(
    completion_text: str,
    predicted_round: str,
    action: str,
    parameters: dict[str, Any],
    ground_truth: str,
    valid_json: bool,
    agent_profile: Any | None = None,
    inference_weight: float = INFERENCE_WEIGHT,
    action_weight: float = ACTION_WEIGHT,
    format_weight: float = FORMAT_WEIGHT,
    personality_weight: float = PERSONALITY_WEIGHT,
    behavior_weight: float = BEHAVIOR_WEIGHT,
) -> dict[str, Any]:
    r_inference = compute_inference_reward(predicted_round, ground_truth)
    r_action = compute_action_reward(action, parameters, ground_truth)
    r_format = compute_format_reward(completion_text, valid_json)
    r_personality = compute_personality_reward(action, ground_truth, agent_profile)
    r_behavior = compute_behavior_reward(action, ground_truth, agent_profile)

    total = (
        inference_weight * r_inference
        + action_weight * r_action
        + format_weight * r_format
        + personality_weight * r_personality
        + behavior_weight * r_behavior
    )

    return {
        "total": float(total),
        "inference": float(r_inference),
        "action": float(r_action),
        "format": float(r_format),
        "personality": float(r_personality),
        "behavior": float(r_behavior),
        "correct": r_inference == 1.0,
        "valid": bool(valid_json),
    }
