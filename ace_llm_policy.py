"""Shared LLM -> JSON -> action policy utilities for EIE.

The demo and GRPO notebook both use this module so prompt format, generation,
JSON extraction, validation, and fallback behavior stay identical.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from ace_reward import ACTIONS
from ace_text_inject import (
    call_groq_chat_completion,
    call_huggingface_chat_completion,
    get_groq_api_key,
    get_hf_user_token,
)

ROUND_TYPES = {"competitive", "cooperative", "resource"}
VALID_ACTIONS = set(ACTIONS)


def extract_first_valid_json(text: str) -> dict[str, Any] | None:
    """Extract the first complete valid JSON object using brace balancing."""
    stack: list[str] = []
    start: int | None = None

    for i, ch in enumerate(text or ""):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        return obj
                    start = None
    return None


def build_action_prompt(env: Any, agent: Any) -> str:
    """Strict prompt template. Always ends with JSON:."""
    world_state = env.world.to_prompt_str()
    visible_alliances = sorted([list(pair) for pair in getattr(env, "alliances", set())])
    recent_history = json.dumps(getattr(env, "round_history", [])[-3:], indent=2)
    trust = json.dumps(getattr(agent, "trust_scores", {}), indent=2)
    memory = agent.memory_summary() if hasattr(agent, "memory_summary") else "No memory."

    return f"""You are a strategic economic agent.

Return ONLY valid JSON.
No explanation.
No markdown.

Valid actions:
challenge, propose_alliance, accept_alliance, betray, allocate_resources, execute_contract, submit_bid

Format:

{{
  "predicted_round": "competitive/cooperative/resource",
  "action": "...",
  "parameters": {{}},
  "reasoning": "short"
}}

World State:
{world_state}

Agent:
name={agent.name}
role={agent.company_type}
objective={agent.primary_objective}
trust={trust}
memory={memory}

Visible alliances:
{visible_alliances}

Recent history:
{recent_history}

JSON:"""


def normalize_action(parsed: dict[str, Any] | None, fallback: dict[str, Any]) -> dict[str, Any]:
    """Return a safe structured action for the simulator."""
    if not isinstance(parsed, dict):
        return fallback

    predicted = str(parsed.get("predicted_round", fallback["predicted_round"])).lower().strip()
    if predicted not in ROUND_TYPES:
        predicted = fallback["predicted_round"]

    action = str(parsed.get("action", fallback["action"])).lower().strip()
    if action not in VALID_ACTIONS:
        action = fallback["action"]

    parameters = parsed.get("parameters")
    if not isinstance(parameters, dict):
        parameters = fallback.get("parameters", {})

    if action in {"submit_bid", "allocate_resources"}:
        try:
            parameters["amount"] = max(0.0, min(100.0, float(parameters.get("amount", 50.0))))
        except (TypeError, ValueError):
            parameters["amount"] = 50.0

    return {
        "predicted_round": predicted,
        "action": action,
        "parameters": parameters,
        "beliefs": parsed.get("beliefs") if isinstance(parsed.get("beliefs"), dict) else fallback.get("beliefs", {}),
        "factors": parsed.get("factors") if isinstance(parsed.get("factors"), dict) else fallback.get("factors", {}),
        "reasoning": str(parsed.get("reasoning", fallback.get("reasoning", "Structured fallback action.")))[:240],
    }


def generate_action(prompt: str, model: Any = None, tokenizer: Any = None) -> str:
    """Generate with a local HF model if provided, otherwise Groq if configured.

    The local-model path matches the GRPO notebook requirement and removes the
    prompt echo before returning text.
    """
    if model is not None and tokenizer is not None:
        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            inputs = inputs.to("cuda")
        except Exception:
            pass
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.2,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt) :]
        return text.strip()

    prov = os.getenv("LLM_PROVIDER", "fallback").lower().strip()
    if prov == "groq" and get_groq_api_key():
        return call_groq_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.2,
            max_tokens=160,
        ).strip()
    if prov == "huggingface" and get_hf_user_token():
        m = os.getenv("HF_INFERENCE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        return call_huggingface_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=m,
            temperature=0.2,
            max_tokens=160,
        ).strip()

    return ""


def llm_policy(
    env: Any,
    agent: Any,
    *,
    fallback_fn: Callable[[], dict[str, Any]],
    generator: Callable[[str], str] | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Shared production-safe LLM policy.

    Always returns a valid simulator action. On empty output, malformed JSON, or
    exceptions, it falls back immediately.
    """
    fallback = fallback_fn()
    try:
        prompt = build_action_prompt(env, agent)
        raw = generator(prompt) if generator is not None else generate_action(prompt)
        if debug and raw:
            print(f"\n[LLM RAW:{agent.name}]")
            print(raw)
            print(f"[/LLM RAW:{agent.name}]\n")
        parsed = extract_first_valid_json(raw)
        return normalize_action(parsed, fallback)
    except Exception:
        return fallback
