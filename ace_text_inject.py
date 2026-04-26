"""Natural language event injection for EIE Economic Intelligence Engine."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_local_env(path: str = ".env") -> None:
    """Load local env vars for scripts/tests without overriding real env vars."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not value:
            # Never bind an empty .env value — that can mask Space secrets or block alias fill-in.
            continue
        os.environ.setdefault(key, value)


load_local_env()

# Default for Hugging Face Inference (serverless). Override with HF_INFERENCE_MODEL in env / Space.
DEFAULT_HF_INFERENCE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Groq: HF Spaces and local setups sometimes use a different env name — normalize to GROQ_API_KEY.
GROQ_KEY_ALIASES: tuple[str, ...] = (
    "GROQ_API_KEY",
    "GROQ_KEY",
    "GROQ_SECRET",
    "GROQ_TOKEN",
    "GROQ",
)


def get_groq_api_key() -> str | None:
    """Return a usable Groq API key, checking common variable names and trimming whitespace."""
    for name in GROQ_KEY_ALIASES:
        val = os.getenv(name)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def ensure_groq_key_in_environ() -> None:
    """If only an alias (e.g. GROQ) is set, copy into GROQ_API_KEY for SDKs that expect that name."""
    if os.environ.get("GROQ_API_KEY", "").strip():
        return
    key = get_groq_api_key()
    if key:
        os.environ["GROQ_API_KEY"] = key


ensure_groq_key_in_environ()


def get_hf_user_token() -> str | None:
    """Token for Hugging Face Hub / Inference. Spaces often expose HF_TOKEN; accept common aliases."""
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_API_TOKEN", "HUGGINGFACE_TOKEN"):
        val = os.getenv(key)
        if val and str(val).strip():
            return str(val).strip()
    return None


INJECT_SYSTEM_PROMPT = """You are an economic analyst AI.
Given a real-world event, output ONLY valid JSON with this schema:
{
  "event_type": "supply shock|demand shock|geopolitical|policy|systemic crisis|cooperation / agreement",
  "deltas": {
    "oil_price": 0.0,
    "gold_price": 0.0,
    "food_index": 0.0,
    "energy_cost": 0.0,
    "interest_rate": 0.0,
    "inflation": 0.0,
    "gdp_growth": 0.0,
    "trade_tension": 0.0,
    "market_volatility": 0.0,
    "cooperation_index": 0.0,
    "resource_scarcity": 0.0,
    "liquidity_index": 0.0,
    "credit_spread": 0.0,
    "geopolitical_risk": 0.0,
    "supply_chain_stability": 0.0
  },
  "confidence": 0.0,
  "reasoning": "1-2 concise causal lines",
  "affected_sectors": ["energy"]
}

Allowed delta fields:
oil_price, gold_price, food_index, energy_cost, interest_rate, inflation,
gdp_growth, trade_tension, market_volatility, cooperation_index, resource_scarcity,
liquidity_index, credit_spread, geopolitical_risk, supply_chain_stability,
sector_energy, sector_agriculture, sector_finance, sector_manufacturing, sector_technology.

Rules:
1. Output ONLY JSON. No markdown.
2. Include every delta key, using 0.0 for neutral.
3. Use realistic magnitudes and never exceed absolute delta 0.4.
4. Include second-order effects: oil -> energy -> inflation; volatility -> lower cooperation.
5. Use low, stable magnitudes: slight=0.05, moderate=0.1, major=0.2, severe=0.3-0.4.

Example:
{"event_type": "supply shock", "deltas": {"oil_price": 0.2, "gold_price": 0.0, "food_index": 0.0, "energy_cost": 0.15, "interest_rate": 0.0, "inflation": 0.05, "gdp_growth": 0.0, "trade_tension": 0.0, "market_volatility": 0.1, "cooperation_index": -0.04, "resource_scarcity": 0.0, "liquidity_index": 0.0, "credit_spread": 0.0, "geopolitical_risk": 0.1, "supply_chain_stability": 0.0}, "confidence": 0.84, "reasoning": "Oil supply shock raises energy costs and inflation while increasing uncertainty.", "affected_sectors": ["energy", "manufacturing"]}
"""


DELTA_FIELDS = (
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
    "liquidity_index",
    "credit_spread",
    "geopolitical_risk",
    "supply_chain_stability",
)

SECTOR_DELTA_FIELDS = (
    "sector_energy",
    "sector_agriculture",
    "sector_finance",
    "sector_manufacturing",
    "sector_technology",
)

VALID_SECTORS = {"energy", "agriculture", "finance", "manufacturing", "technology"}
VALID_EVENT_TYPES = {
    "supply shock",
    "demand shock",
    "geopolitical",
    "policy",
    "systemic crisis",
    "cooperation / agreement",
}

DELTA_LIMIT = 0.4
EVENT_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}

HARD_BOUNDS = {
    "oil_price": (0.5, 2.5),
    "gold_price": (0.5, 2.5),
    "food_index": (0.5, 2.5),
    "energy_cost": (0.5, 2.5),
    "market_volatility": (0.0, 1.0),
    "cooperation_index": (0.0, 1.0),
    "resource_scarcity": (0.0, 1.0),
    "liquidity_index": (0.0, 1.0),
    "credit_spread": (0.0, 1.0),
    "geopolitical_risk": (0.0, 1.0),
    "supply_chain_stability": (0.0, 1.0),
    "inflation": (0.0, 0.5),
    "interest_rate": (0.0, 0.25),
    "gdp_growth": (-0.2, 0.2),
    "trade_tension": (0.0, 1.0),
}


def parse_event_payload(
    event_text: str,
    world_state_str: str = "",
    model: str = "claude-sonnet-4-20250514",
    provider: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Return {deltas, confidence, causal_reasoning}; never raises."""
    normalized_event = normalize_event_text(event_text)
    if not normalized_event:
        return _empty_payload("No event provided.", confidence=0.0)

    raw = ""
    provider = (provider or os.getenv("LLM_PROVIDER", "fallback")).lower().strip()
    if provider == "groq":
        _cache_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    elif provider == "huggingface":
        _cache_model = os.getenv("HF_INFERENCE_MODEL", DEFAULT_HF_INFERENCE_MODEL)
    elif provider == "anthropic":
        _cache_model = os.getenv("ANTHROPIC_MODEL", model)
    else:
        _cache_model = str(model)
    cache_key = (provider, _cache_model, normalized_event)
    if not debug and cache_key in EVENT_CACHE:
        return deepcopy(EVENT_CACHE[cache_key])

    if provider == "groq" and get_groq_api_key():
        try:
            raw = call_groq_chat_completion(
                [
                    {"role": "system", "content": INJECT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Current world state:\n{world_state_str}\n\nEvent: {event_text}",
                    },
                ],
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                temperature=0.15,
                max_tokens=360,
            )
        except Exception:
            raw = ""
    elif provider == "huggingface" and get_hf_user_token():
        try:
            hf_model = os.getenv("HF_INFERENCE_MODEL", DEFAULT_HF_INFERENCE_MODEL)
            raw = call_huggingface_chat_completion(
                [
                    {"role": "system", "content": INJECT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Current world state:\n{world_state_str}\n\nEvent: {event_text}",
                    },
                ],
                model=hf_model,
                temperature=0.15,
                max_tokens=360,
            )
        except Exception:
            raw = ""
    elif provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", model),
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

    if debug and raw:
        print("\n[LLM RAW:event_injection]")
        print(raw)
        print("[/LLM RAW:event_injection]\n")

    if raw:
        parsed = _parse_json_payload(raw)
        if parsed is not None:
            EVENT_CACHE[cache_key] = parsed
            return deepcopy(parsed)

    fallback = _fallback_event_payload(normalized_event)
    EVENT_CACHE[cache_key] = fallback
    return deepcopy(fallback)


def parse_event_to_deltas(event_text: str, world_state_str: str = "") -> dict[str, float]:
    return parse_event_payload(event_text, world_state_str)["deltas"]


def normalize_event_text(event_text: str) -> str:
    return re.sub(r"\s+", " ", event_text.strip().lower())


def call_groq_chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call Groq chat completions with SDK if available, otherwise plain HTTPS.

    On network/API errors, returns an empty string so callers can fall back to
    structured policies instead of crashing the UI.
    """
    ensure_groq_key_in_environ()
    api_key = get_groq_api_key()
    if not api_key:
        return ""

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except ModuleNotFoundError:
        pass
    except Exception:
        pass

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"] or "")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, KeyError, json.JSONDecodeError, IndexError):
        return ""


def call_huggingface_chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Chat completion via Hugging Face Inference (same token as HF Spaces: HF_TOKEN)."""
    token = get_hf_user_token()
    if not token:
        raise RuntimeError("No Hugging Face token (set HF_TOKEN or HUGGINGFACE_HUB_TOKEN).")
    try:
        from huggingface_hub import InferenceClient
    except ModuleNotFoundError as err:
        raise RuntimeError("Install huggingface_hub to use Hugging Face Inference.") from err

    try:
        client = InferenceClient(model=model, token=token)
        out = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return str(out.choices[0].message.content or "")
    except Exception:
        return ""


def describe_impact(deltas: dict[str, float], event_text: str, causal_reasoning: str = "") -> str:
    non_zero = {key: value for key, value in deltas.items() if abs(float(value)) > 1e-9}
    if not non_zero:
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
        "liquidity_index": "Liquidity",
        "credit_spread": "Credit spread",
        "geopolitical_risk": "Geopolitical risk",
        "supply_chain_stability": "Supply-chain stability",
        "sector_energy": "Energy sector",
        "sector_agriculture": "Agriculture sector",
        "sector_finance": "Finance sector",
        "sector_manufacturing": "Manufacturing sector",
        "sector_technology": "Technology sector",
    }
    lines = [f"Economic impact of: '{event_text}'"]
    if causal_reasoning:
        lines.append(f"Cause: {causal_reasoning}")
    for field, delta in non_zero.items():
        if field not in labels:
            continue
        arrow = "up" if delta > 0 else "down"
        lines.append(f"- {labels[field]}: {arrow} {abs(delta):.3f}")
    return "\n".join(lines)


def _parse_json_payload(raw: str) -> dict[str, Any] | None:
    clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    decoder = json.JSONDecoder()
    obj = None
    for idx, char in enumerate(clean):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(clean[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            obj = candidate
            break
    if obj is None:
        return None
    if not isinstance(obj, dict):
        return None

    payload = _validate_payload(obj)
    if payload is None:
        return None
    return payload


def _validate_payload(obj: dict[str, Any]) -> dict[str, Any] | None:
    raw_deltas = obj.get("deltas")
    if not isinstance(raw_deltas, dict):
        # Backwards-compatible parser for older flat LLM outputs.
        raw_deltas = {key: obj.get(key, 0.0) for key in DELTA_FIELDS}

    deltas: dict[str, float] = {}
    for key in DELTA_FIELDS:
        try:
            value = float(raw_deltas.get(key, 0.0))
        except (TypeError, ValueError):
            return None
        deltas[key] = _clamp_delta(value)

    for sector_key in SECTOR_DELTA_FIELDS:
        if sector_key in raw_deltas or sector_key in obj:
            try:
                deltas[sector_key] = _clamp_delta(float(raw_deltas.get(sector_key, obj.get(sector_key, 0.0))))
            except (TypeError, ValueError):
                return None

    try:
        confidence = _clip(float(obj.get("confidence", 0.65)), 0.0, 1.0)
    except (TypeError, ValueError):
        return None

    reasoning = str(
        obj.get("reasoning")
        or obj.get("causal_reasoning")
        or "Structured economic interpretation applied."
    )[:500]
    event_type = _normalize_event_type(str(obj.get("event_type", "demand shock")))
    sectors = _normalize_sectors(obj.get("affected_sectors"), deltas)
    return _canonical_payload(
        deltas=deltas,
        confidence=confidence,
        reasoning=reasoning,
        affected_sectors=sectors,
        event_type=event_type,
    )


def _fallback_event_payload(event_text: str) -> dict[str, Any]:
    text = normalize_event_text(event_text)
    deltas: dict[str, float] = _zero_deltas()
    reasons: list[str] = []
    sectors: set[str] = set()
    event_types: list[str] = []
    magnitude = _infer_magnitude(text)

    if "oil" in text and any(word in text for word in ["crisis", "cut", "shortage", "opec", "middle east"]):
        _add_deltas(
            deltas,
            {
                "oil_price": max(0.2, magnitude),
                "energy_cost": max(0.15, magnitude * 0.75),
                "market_volatility": max(0.1, magnitude * 0.5),
                "inflation": 0.05,
                "geopolitical_risk": max(0.1, magnitude * 0.5),
                "cooperation_index": -0.04,
            },
        )
        sectors.update({"energy", "manufacturing"})
        event_types.extend(["supply shock", "geopolitical"])
        reasons.append("Oil supply stress raises energy costs and inflation while increasing geopolitical risk.")

    if any(word in text for word in ["peace", "agreement", "ceasefire", "cooperation", "climate pact"]):
        _add_deltas(
            deltas,
            {
                "cooperation_index": 0.2,
                "market_volatility": -0.1,
                "trade_tension": -0.15,
                "geopolitical_risk": -0.1,
                "supply_chain_stability": 0.08,
            },
        )
        sectors.update({"energy", "manufacturing", "finance"})
        event_types.append("cooperation / agreement")
        reasons.append("Diplomatic agreement improves coordination and reduces uncertainty.")

    if "rate hike" in text or "central bank raises" in text or ("central bank" in text and "raise" in text):
        _add_deltas(
            deltas,
            {
                "interest_rate": 0.02,
                "liquidity_index": -0.1,
                "gdp_growth": -0.05,
                "credit_spread": 0.06,
                "market_volatility": 0.05,
            },
        )
        sectors.update({"finance", "technology"})
        event_types.append("policy")
        reasons.append("Tighter monetary policy drains liquidity and slows growth expectations.")

    if any(phrase in text for phrase in ["supply chain", "drought", "shortage"]):
        _add_deltas(
            deltas,
            {
                "food_index": 0.15,
                "resource_scarcity": 0.2,
                "supply_chain_stability": -0.2,
                "market_volatility": 0.08,
            },
        )
        sectors.update({"agriculture", "manufacturing"})
        event_types.append("supply shock")
        reasons.append("Supply disruption raises scarcity and weakens logistics stability.")

    if any(word in text for word in ["crash", "recession", "bank failure", "panic", "systemic"]):
        _add_deltas(
            deltas,
            {
                "market_volatility": 0.3,
                "gdp_growth": -0.06,
                "gold_price": 0.15,
                "cooperation_index": -0.06,
                "liquidity_index": -0.2,
                "credit_spread": 0.2,
            },
        )
        sectors.update({"finance", "technology"})
        event_types.append("systemic crisis")
        reasons.append("Systemic stress increases volatility, tightens credit, and pushes agents defensive.")

    if any(word in text for word in ["tariff", "sanction", "embargo"]) or "trade war" in text:
        _add_deltas(
            deltas,
            {
                "trade_tension": 0.25,
                "market_volatility": 0.15,
                "cooperation_index": -0.18,
                "resource_scarcity": 0.1,
                "geopolitical_risk": 0.18,
                "supply_chain_stability": -0.15,
            },
        )
        sectors.update({"manufacturing", "finance"})
        event_types.append("geopolitical")
        reasons.append("Trade conflict reduces cooperation and raises supply friction.")

    if not any(abs(value) > 1e-9 for value in deltas.values()):
        deltas["market_volatility"] = 0.04
        sectors.add("finance")
        event_types.append("demand shock")
        reasons.append("Event has uncertain but limited market impact.")

    _apply_cross_variable_effects(deltas)
    return _canonical_payload(
        deltas=deltas,
        confidence=0.58,
        reasoning=" ".join(reasons) or "Rule-based interpretation applied.",
        affected_sectors=sorted(sectors),
        event_type=_choose_event_type(event_types),
    )


def _zero_deltas() -> dict[str, float]:
    return {key: 0.0 for key in DELTA_FIELDS}


def _empty_payload(reasoning: str, confidence: float = 0.0) -> dict[str, Any]:
    return _canonical_payload(
        deltas=_zero_deltas(),
        confidence=confidence,
        reasoning=reasoning,
        affected_sectors=[],
        event_type="demand shock",
    )


def _canonical_payload(
    *,
    deltas: dict[str, float],
    confidence: float,
    reasoning: str,
    affected_sectors: list[str],
    event_type: str,
) -> dict[str, Any]:
    canonical = _zero_deltas()
    for key, value in deltas.items():
        if key in DELTA_FIELDS or key in SECTOR_DELTA_FIELDS:
            canonical[key] = _clamp_delta(value)
    sectors = _normalize_sectors(affected_sectors, canonical)
    clean_reasoning = reasoning.strip() or "Rule-based interpretation applied."
    return {
        "deltas": canonical,
        "confidence": _clip(confidence, 0.0, 1.0),
        "reasoning": clean_reasoning[:500],
        "causal_reasoning": clean_reasoning[:500],
        "affected_sectors": sectors,
        "event_type": _normalize_event_type(event_type),
    }


def _add_deltas(deltas: dict[str, float], updates: dict[str, float]) -> None:
    for key, value in updates.items():
        if key in deltas:
            deltas[key] = _clamp_delta(deltas[key] + float(value))


def _apply_cross_variable_effects(deltas: dict[str, float]) -> None:
    oil = max(0.0, deltas.get("oil_price", 0.0))
    energy = max(0.0, deltas.get("energy_cost", 0.0))
    volatility = max(0.0, deltas.get("market_volatility", 0.0))
    scarcity = max(0.0, deltas.get("resource_scarcity", 0.0))
    if oil:
        deltas["energy_cost"] = _clamp_delta(deltas["energy_cost"] + 0.5 * oil)
    if energy:
        deltas["inflation"] = _clamp_delta(deltas["inflation"] + 0.12 * energy)
    if volatility:
        deltas["cooperation_index"] = _clamp_delta(deltas["cooperation_index"] - 0.15 * volatility)
        deltas["liquidity_index"] = _clamp_delta(deltas["liquidity_index"] - 0.08 * volatility)
    if scarcity:
        deltas["market_volatility"] = _clamp_delta(deltas["market_volatility"] + 0.1 * scarcity)


def _infer_magnitude(text: str) -> float:
    if any(word in text for word in ["severe", "catastrophic", "massive"]):
        return 0.35
    if any(word in text for word in ["major", "large", "significant"]):
        return 0.2
    if any(word in text for word in ["moderate", "medium"]):
        return 0.1
    if any(word in text for word in ["slight", "minor", "small"]):
        return 0.05
    return 0.2


def _normalize_sectors(raw_sectors: Any, deltas: dict[str, float]) -> list[str]:
    sectors: set[str] = set()
    if isinstance(raw_sectors, list):
        sectors.update(str(item).lower().strip() for item in raw_sectors if str(item).lower().strip() in VALID_SECTORS)
    if abs(deltas.get("oil_price", 0.0)) > 0 or abs(deltas.get("energy_cost", 0.0)) > 0:
        sectors.add("energy")
    if abs(deltas.get("food_index", 0.0)) > 0 or abs(deltas.get("resource_scarcity", 0.0)) > 0:
        sectors.add("agriculture")
    if abs(deltas.get("credit_spread", 0.0)) > 0 or abs(deltas.get("liquidity_index", 0.0)) > 0:
        sectors.add("finance")
    if abs(deltas.get("supply_chain_stability", 0.0)) > 0 or abs(deltas.get("trade_tension", 0.0)) > 0:
        sectors.add("manufacturing")
    return sorted(sectors)


def _normalize_event_type(event_type: str) -> str:
    clean = event_type.lower().strip()
    if clean in VALID_EVENT_TYPES:
        return clean
    if "cooperation" in clean or "agreement" in clean:
        return "cooperation / agreement"
    if "policy" in clean or "rate" in clean:
        return "policy"
    if "crisis" in clean:
        return "systemic crisis"
    if "geo" in clean:
        return "geopolitical"
    if "supply" in clean:
        return "supply shock"
    return "demand shock"


def _choose_event_type(event_types: list[str]) -> str:
    priority = ["systemic crisis", "policy", "cooperation / agreement", "geopolitical", "supply shock", "demand shock"]
    for item in priority:
        if item in event_types:
            return item
    return "demand shock"


def _clamp_delta(value: float) -> float:
    return _clip(float(value), -DELTA_LIMIT, DELTA_LIMIT)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
