"""Natural language event injection for ACE++ Option B."""

from __future__ import annotations

import json
import os
import re
from typing import Any


INJECT_SYSTEM_PROMPT = """You are an economic analyst AI.
Given a real-world event, output ONLY valid JSON with:
- numeric deltas for affected world state fields
- confidence in [0,1]
- causal_reasoning in 1-2 concise lines

Allowed delta fields:
oil_price, gold_price, food_index, energy_cost, interest_rate, inflation,
gdp_growth, trade_tension, market_volatility, cooperation_index, resource_scarcity.

Rules:
1. Output ONLY JSON. No markdown.
2. Omit unchanged fields.
3. Use realistic magnitudes.
4. Include second-order effects.

Example:
{"oil_price": 0.35, "energy_cost": 0.25, "market_volatility": 0.2, "inflation": 0.02, "confidence": 0.84, "causal_reasoning": "Oil supply shock raises fuel and energy costs while increasing uncertainty."}
"""


DELTA_FIELDS = {
    "oil_price",
    "gold_price",
    "food_index",
    "energy_cost",
    "interest_rate",
    "inflation",
    "gdp_growth",
    "trade_tension",
    "market_volatility",
    "cooperation_index",
    "resource_scarcity",
}


def parse_event_payload(
    event_text: str,
    world_state_str: str = "",
    model: str = "claude-sonnet-4-20250514",
) -> dict[str, Any]:
    """Return {deltas, confidence, causal_reasoning}; never raises."""
    if not event_text.strip():
        return {"deltas": {}, "confidence": 0.0, "causal_reasoning": "No event provided."}

    raw = ""
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model=model,
                max_tokens=320,
                system=INJECT_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"Current world state:\n{world_state_str}\n\nEvent: {event_text}",
                    }
                ],
            )
            raw = response.content[0].text.strip()
        except Exception:
            raw = ""

    if raw:
        parsed = _parse_json_payload(raw)
        if parsed is not None:
            return parsed

    return _fallback_event_payload(event_text)


def parse_event_to_deltas(event_text: str, world_state_str: str = "") -> dict[str, float]:
    return parse_event_payload(event_text, world_state_str)["deltas"]


def describe_impact(deltas: dict[str, float], event_text: str, causal_reasoning: str = "") -> str:
    if not deltas:
        return f"No major economic impact detected for: '{event_text}'."

    labels = {
        "oil_price": "Oil prices",
        "gold_price": "Gold prices",
        "food_index": "Food costs",
        "energy_cost": "Energy costs",
        "interest_rate": "Interest rates",
        "inflation": "Inflation",
        "gdp_growth": "GDP growth",
        "trade_tension": "Trade tension",
        "market_volatility": "Market volatility",
        "cooperation_index": "Cooperation willingness",
        "resource_scarcity": "Resource scarcity",
    }
    lines = [f"Economic impact of: '{event_text}'"]
    if causal_reasoning:
        lines.append(f"Cause: {causal_reasoning}")
    for field, delta in deltas.items():
        if field not in labels:
            continue
        arrow = "up" if delta > 0 else "down"
        lines.append(f"- {labels[field]}: {arrow} {abs(delta):.3f}")
    return "\n".join(lines)


def _parse_json_payload(raw: str) -> dict[str, Any] | None:
    clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    try:
        obj = json.loads(clean)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    deltas = {}
    for key, value in obj.items():
        if key not in DELTA_FIELDS:
            continue
        try:
            deltas[key] = float(value)
        except (TypeError, ValueError):
            continue
    try:
        confidence = float(obj.get("confidence", 0.65))
    except (TypeError, ValueError):
        confidence = 0.65
    reasoning = str(obj.get("causal_reasoning", "Structured economic impact inferred."))
    return {
        "deltas": deltas,
        "confidence": max(0.0, min(1.0, confidence)),
        "causal_reasoning": reasoning[:500],
    }


def _fallback_event_payload(event_text: str) -> dict[str, Any]:
    text = event_text.lower()
    deltas: dict[str, float] = {}
    reasons: list[str] = []

    if any(word in text for word in ["oil", "opec", "energy", "middle east"]):
        deltas.update(
            {
                "oil_price": 0.45,
                "energy_cost": 0.32,
                "market_volatility": 0.22,
                "inflation": 0.025,
                "trade_tension": 0.12,
                "cooperation_index": -0.08,
            }
        )
        reasons.append("Energy supply risk raises oil prices, costs, and uncertainty.")
    if any(word in text for word in ["drought", "food", "crop", "grain", "famine"]):
        deltas.update({"food_index": 0.35, "resource_scarcity": 0.28, "market_volatility": 0.12})
        reasons.append("Food supply disruption increases scarcity and commodity prices.")
    if any(word in text for word in ["peace", "ceasefire", "agreement", "cooperation", "climate pact"]):
        deltas.update({"cooperation_index": 0.28, "trade_tension": -0.2, "market_volatility": -0.12})
        reasons.append("Diplomatic cooperation lowers tension and improves coordination incentives.")
    if any(word in text for word in ["rate", "central bank", "basis points", "inflation hike"]):
        deltas.update({"interest_rate": 0.0075, "market_volatility": 0.12, "gdp_growth": -0.015, "gold_price": -0.04})
        reasons.append("Tighter monetary policy slows growth and increases market uncertainty.")
    if any(word in text for word in ["crash", "recession", "bank failure", "panic"]):
        deltas.update({"market_volatility": 0.35, "gdp_growth": -0.06, "gold_price": 0.2, "cooperation_index": -0.06})
        reasons.append("Financial stress increases volatility and defensive positioning.")
    if any(word in text for word in ["trade war", "tariff", "sanction", "embargo"]):
        deltas.update({"trade_tension": 0.35, "market_volatility": 0.2, "cooperation_index": -0.25, "resource_scarcity": 0.15})
        reasons.append("Trade conflict reduces cooperation and raises supply friction.")

    if not deltas:
        deltas = {"market_volatility": 0.04}
        reasons.append("Event has uncertain but limited market impact.")

    return {
        "deltas": deltas,
        "confidence": 0.58,
        "causal_reasoning": " ".join(reasons),
    }
