"""Judge-facing EIE Economic Intelligence Engine demo.

Run:
    LLM_PROVIDER=groq GROQ_API_KEY=... python demo_gradio.py

The app also works without an API key by using deterministic adaptive fallback
agents, so the demo never crashes during judging.
"""

from __future__ import annotations

import copy
import json
import os
import random
import statistics
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ModuleNotFoundError:  # pragma: no cover
    gr = None


def load_local_env(path: str = ".env") -> None:
    """Load local env vars for development without overriding Space secrets."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()

from ace_agents import AgentProfile
from ace_llm_policy import build_action_prompt, extract_first_valid_json, llm_policy, normalize_action
from ace_reward import ACTIONS
from ace_text_inject import call_groq_chat_completion, describe_impact
from ace_world_env import ACEWorldEnv


ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
PROVIDERS = {"fallback", "groq", "anthropic"}
UI_PROVIDER_CHOICES = ["fallback", "groq"]
PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
PERFECT_DEMO_EVENT = "oil crisis hits Middle East"
PRESET_EVENTS = [
    PERFECT_DEMO_EVENT,
    "global cooperation agreement signed",
    "major food supply chain disruption",
]
TRAINING_SCENARIOS = {
    "oil_crisis": "Oil crisis disrupts shipping, raises energy costs, and increases geopolitical risk",
    "peace_scenario": "Peace agreement lowers trade tension, improves cooperation, and stabilizes supply chains",
}
COOP_ACTIONS = {"propose_alliance", "accept_alliance", "execute_contract"}
BETRAY_ACTIONS = {"betray"}
AGGRESSIVE_ACTIONS = {"challenge", "betray", "submit_bid"}
APP_CSS = """
:root {
  --ace-bg: #070b1a;
  --ace-card: rgba(255, 255, 255, 0.08);
  --ace-card-strong: rgba(255, 255, 255, 0.14);
  --ace-border: rgba(255, 255, 255, 0.16);
  --ace-text: #e5e7eb;
  --ace-muted: #94a3b8;
  --ace-cyan: #22d3ee;
  --ace-purple: #a78bfa;
  --ace-pink: #fb7185;
}

body, .gradio-container {
  background:
    radial-gradient(circle at 8% 8%, rgba(34, 211, 238, 0.24), transparent 28%),
    radial-gradient(circle at 88% 12%, rgba(167, 139, 250, 0.28), transparent 28%),
    radial-gradient(circle at 50% 95%, rgba(251, 113, 133, 0.16), transparent 30%),
    linear-gradient(180deg, #070b1a 0%, #0f172a 48%, #111827 100%) !important;
  color: var(--ace-text) !important;
}

.gradio-container {
  max-width: 1180px !important;
  margin: auto;
}

.block, .form, .panel, .gr-box, .gr-form, .gr-panel {
  border-color: var(--ace-border) !important;
}

.hero {
  position: relative;
  overflow: hidden;
  padding: 30px 34px;
  border-radius: 30px;
  color: white;
  background:
    linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(37, 99, 235, 0.86) 50%, rgba(124, 58, 237, 0.88) 100%);
  border: 1px solid rgba(255,255,255,0.22);
  box-shadow: 0 24px 80px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.22);
  backdrop-filter: blur(18px);
}

.hero::after {
  content: "";
  position: absolute;
  width: 320px;
  height: 320px;
  right: -110px;
  top: -130px;
  background: radial-gradient(circle, rgba(255,255,255,0.28), transparent 92%);
}

.hero p {
  margin: 0;
  max-width: 980px;
  color: rgba(255,255,255,0.88);
}

.hero h1 {
  font-size: 46px;
  font-weight: 800;
  margin: 0 0 8px 0;
  letter-spacing: -0.045em;
}

.mode-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 12px;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.55);
  border: 1px solid rgba(255,255,255,0.16);
  color: #bfdbfe;
  font-size: 13px;
  font-weight: 750;
}

.command-shell {
  margin: 20px 0 24px;
  padding: 18px;
  border-radius: 28px;
  background: linear-gradient(180deg, rgba(255,255,255,0.13), rgba(255,255,255,0.06));
  border: 1px solid rgba(255,255,255,0.18);
  box-shadow: 0 22px 70px rgba(2, 6, 23, 0.32);
  backdrop-filter: blur(18px);
}

.story-stage {
  padding: 22px;
  border-radius: 28px;
  background:
    radial-gradient(circle at 15% 0%, rgba(56, 189, 248, 0.16), transparent 32%),
    linear-gradient(180deg, rgba(30, 41, 59, 0.74), rgba(15, 23, 42, 0.72));
  border: 1px solid rgba(148, 163, 184, 0.22);
  box-shadow: 0 24px 75px rgba(2, 6, 23, 0.28);
}

.info-dot {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  margin-left: 6px;
  font-size: 11px;
  color: #0f172a;
  background: #67e8f9;
  cursor: help;
}

.demo-hint {
  margin: 16px 0 6px;
  padding: 15px 18px;
  border-radius: 20px !important;
  border: 1px solid rgba(34, 211, 238, 0.32);
  background: linear-gradient(135deg, rgba(14, 165, 233, 0.18) 0%, rgba(124, 58, 237, 0.18) 100%);
  color: #e0f2fe;
  font-weight: 700;
  box-shadow: 0 14px 35px rgba(2, 6, 23, 0.22);
}

.training-proof {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 12px 0 16px;
}

.metric-card {
  padding: 16px;
  border-radius: 20px;
  background: linear-gradient(180deg, rgba(34, 211, 238, 0.16) 0%, rgba(124, 58, 237, 0.12) 100%);
  border: 1px solid rgba(148, 163, 184, 0.24);
  box-shadow: 0 14px 35px rgba(2, 6, 23, 0.22);
}

.metric-card b {
  display:block;
  font-size: 13px;
  color: #cbd5e1;
  margin-bottom: 7px;
}

.metric-card span {
  font-size: 25px;
  font-weight: 850;
  color: #f8fafc;
}

.flow-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin: 0;
}

.flow-step {
  background: linear-gradient(180deg, rgba(255,255,255,0.11) 0%, rgba(255,255,255,0.055) 100%);
  border-radius: 22px;
  border: 1px solid var(--ace-border);
  padding: 16px 18px;
  box-shadow: 0 16px 38px rgba(2, 6, 23, 0.24);
  backdrop-filter: blur(14px);
  transition: all 0.25s ease;
}

.flow-step b {
  display: block;
  margin-bottom: 6px;
  color: #f8fafc;
}

.flow-step span {
  color: #cbd5e1;
  font-size: 13px;
}

.world-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-top: 18px;
}

.world-card {
  padding: 17px;
  border-radius: 22px;
  background: rgba(15, 23, 42, 0.58);
  border: 1px solid rgba(148, 163, 184, 0.22);
}

.world-card b {
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: #f8fafc;
  font-size: 14px;
}

.world-value {
  margin-top: 8px;
  color: #93c5fd;
  font-size: 24px;
  font-weight: 850;
}

.world-status {
  color: #cbd5e1;
  font-size: 12px;
  margin-left: 6px;
}

.mini-bar {
  height: 8px;
  border-radius: 999px;
  margin-top: 12px;
  background: rgba(148, 163, 184, 0.18);
  overflow: hidden;
}

.mini-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #38bdf8, #a78bfa);
  transition: width 0.45s ease;
}

.flow-step:hover {
  transform: translateY(-3px);
  box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}

.agent-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.075) 100%);
  padding: 18px;
  border-radius: 24px;
  border: 1px solid var(--ace-border);
  margin-bottom: 14px;
  box-shadow: 0 18px 42px rgba(2, 6, 23, 0.24);
  backdrop-filter: blur(14px);
  border-left: 4px solid #38bdf8;
  transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}

.agent-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 24px 70px rgba(2, 6, 23, 0.34);
}

.agent-card.aggressive { border-left-color: #fb7185; }
.agent-card.cooperative { border-left-color: #38bdf8; }
.agent-card.defensive { border-left-color: #f59e0b; }

.agent-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 14px;
}

.agent-action {
  margin-top: 14px;
  font-size: 22px;
  font-weight: 850;
  color: #f8fafc;
}

.agent-pill {
  display: inline-flex;
  padding: 5px 9px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
  background: rgba(56, 189, 248, 0.16);
  color: #bae6fd;
}

.agent-meta {
  display:flex;
  justify-content:space-between;
  gap: 10px;
  margin-top: 12px;
  color: #cbd5e1;
  font-size: 13px;
}

.agent-why {
  margin-top: 10px;
  color: #94a3b8;
  font-size: 12px;
  opacity: 0;
  max-height: 0;
  overflow: hidden;
  transition: all 0.22s ease;
}

.agent-card:hover .agent-why {
  opacity: 1;
  max-height: 80px;
}

.agent-card-header {
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap: 12px;
}

.agent-card h3 {
  margin:0;
  color: #f8fafc;
}

.agent-card small, .agent-muted {
  color: #cbd5e1;
}

.agent-resources {
  font-weight: 800;
  color:#86efac;
}

.belief-pre {
  margin: 6px 0 0;
  padding: 10px;
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.55);
  color: #e0f2fe;
  white-space: pre-wrap;
  border: 1px solid rgba(148, 163, 184, 0.22);
}

button {
  border-radius: 16px !important;
  font-weight: 600;
  transition: all 0.2s ease;
  box-shadow: 0 12px 28px rgba(2, 6, 23, 0.18);
}

button:hover {
  transform: scale(1.04);
}

textarea {
  font-family: monospace;
  font-size: 13px;
  background: rgba(15, 23, 42, 0.72) !important;
  color: #e5e7eb !important;
  border: 1px solid rgba(148, 163, 184, 0.28) !important;
}

.section-title {
  margin: 20px 0 10px;
  padding: 12px 16px;
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(34, 211, 238, 0.16) 0%, rgba(167, 139, 250, 0.16) 100%);
  color: #f8fafc;
  font-weight: 700;
  letter-spacing: 0.3px;
  border: 1px solid var(--ace-border);
  box-shadow: 0 12px 28px rgba(2, 6, 23, 0.18);
}

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

.flow-strip, .agent-card {
  animation: fadeInUp 0.4s ease;
}

.why-matters {
  margin: 18px 0 8px;
  padding: 16px 18px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(34, 211, 238, 0.16), rgba(251, 113, 133, 0.15));
  color: #e2e8f0;
  font-weight: 700;
  border: 1px solid var(--ace-border);
}

.small-note { color: #cbd5e1; font-size: 13px; margin: 8px 0 2px; }
textarea, input { border-radius: 14px !important; }

label, .wrap, .prose, .markdown, .gradio-container h1, .gradio-container h2, .gradio-container h3 {
  color: var(--ace-text) !important;
}

input, select {
  background: rgba(15, 23, 42, 0.72) !important;
  color: #e5e7eb !important;
  border-color: rgba(148, 163, 184, 0.28) !important;
}

/* Terminal-grade visual reset: black canvas, crisp borders, no glossy gradients. */
body, .gradio-container {
  background: #000 !important;
  color: #d1fae5 !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace !important;
}

.gradio-container {
  max-width: 1560px !important;
}

.hero, .command-shell, .story-stage, .flow-step, .world-card, .agent-card, .metric-card, .section-title, .why-matters {
  background: #050505 !important;
  border: 1px solid #1f2937 !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
}

.hero {
  border-color: #22c55e !important;
}

.hero h1, .section-title {
  color: #22c55e !important;
  text-shadow: 0 0 14px rgba(34, 197, 94, 0.24);
}

.hero p, .small-note, .agent-muted, .flow-step span, label {
  color: #94a3b8 !important;
}

.mode-pill, .agent-pill {
  background: #020617 !important;
  border: 1px solid #334155 !important;
  color: #67e8f9 !important;
}

textarea, input, select, .wrap, .block, .form {
  background: #020617 !important;
  color: #d1fae5 !important;
  border-color: #334155 !important;
}

input[type="checkbox"], input[type="radio"] {
  accent-color: #22c55e !important;
  outline: 1px solid #64748b !important;
}

label:has(input[type="radio"]:checked),
label:has(input[type="checkbox"]:checked) {
  color: #22c55e !important;
  border-color: #22c55e !important;
  background: #052e16 !important;
}

label:has(input[type="radio"]) {
  padding: 8px 10px !important;
  border: 1px solid #334155 !important;
  border-radius: 10px !important;
}

button {
  background: #020617 !important;
  color: #d1fae5 !important;
  border: 1px solid #22c55e !important;
  box-shadow: none !important;
}

button:hover {
  background: #052e16 !important;
  transform: translateY(-1px);
}

.info-dot {
  position: relative;
  background: #22c55e !important;
  color: #000 !important;
}

.info-dot[data-tip]:hover::after {
  content: attr(data-tip);
  position: absolute;
  z-index: 20;
  left: 50%;
  bottom: 145%;
  transform: translateX(-50%);
  width: 220px;
  padding: 9px 10px;
  border-radius: 8px;
  background: #020617;
  color: #d1fae5;
  border: 1px solid #22c55e;
  font-size: 11px;
  line-height: 1.35;
  box-shadow: 0 0 20px rgba(34, 197, 94, 0.18);
}

.mini-fill {
  background: #22c55e !important;
}

.agent-card.aggressive { border-left-color: #ef4444 !important; }
.agent-card.cooperative { border-left-color: #38bdf8 !important; }
.agent-card.defensive { border-left-color: #f59e0b !important; }

.agent-bars {
  display: grid;
  gap: 8px;
  margin-top: 12px;
}

.agent-bar-row {
  display: grid;
  grid-template-columns: 86px 1fr 48px;
  align-items: center;
  gap: 8px;
  color: #94a3b8;
  font-size: 12px;
}

.agent-bar-track {
  height: 7px;
  border-radius: 999px;
  background: #111827;
  overflow: hidden;
  border: 1px solid #1f2937;
}

.agent-bar-fill {
  height: 100%;
  background: #22c55e;
}

.terminal-panel {
  border: 1px solid #1f2937;
  background: #020617;
  padding: 16px;
  border-radius: 14px;
}
"""
try:
    import plotly.graph_objects as go
except ModuleNotFoundError:  # pragma: no cover
    go = None


def make_fresh_env() -> ACEWorldEnv:
    return ACEWorldEnv()


def normalize_provider(provider: str | None) -> str:
    value = (provider or os.getenv("LLM_PROVIDER", "fallback")).lower().strip()
    return value if value in PROVIDERS else "fallback"


def resolve_model(provider: str | None, model_choice: str | None = None) -> str:
    provider = normalize_provider(provider)
    choice = (model_choice or "").strip()
    if choice:
        return choice
    if provider == "anthropic":
        return ANTHROPIC_MODEL
    return GROQ_MODEL


def apply_model_choice(provider: str | None, model_choice: str | None = None) -> None:
    model = resolve_model(provider, model_choice)
    provider = normalize_provider(provider)
    if provider == "anthropic":
        os.environ["ANTHROPIC_MODEL"] = model
    elif provider == "groq":
        os.environ["GROQ_MODEL"] = model


def apply_llm_runtime_config(provider: str | None, model_choice: str | None = None) -> tuple[bool, str]:
    """Apply selected provider/model and verify required env secrets exist."""
    provider = normalize_provider(provider)
    apply_model_choice(provider, model_choice)

    key_env = PROVIDER_KEY_ENV.get(provider)

    if provider == "fallback":
        return False, "Choose Groq or Anthropic for LLM-Based RL. API keys should be set as environment variables or Hugging Face Space secrets."
    if key_env and os.getenv(key_env):
        return True, f"LLM ready: {provider} / {resolve_model(provider, model_choice)}"
    if key_env:
        return False, f"Missing {key_env}. Add it to your .env file locally or to Hugging Face Space secrets, then restart/rebuild."
    return False, "Unsupported provider selected."


def llm_or_fallback_decision(
    env: ACEWorldEnv,
    agent: AgentProfile,
    provider: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    available = [item.agent_id for item in env.agents]
    fallback = agent.choose_fallback_action(
        env.world.derive_round_probabilities(),
        env.round_number + 1,
        available,
    )
    provider = normalize_provider(provider)

    if provider == "fallback":
        return fallback

    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()

            def anthropic_generator(prompt: str) -> str:
                response = client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=260,
                    system="Return ONLY valid JSON. No markdown. No explanation.",
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text.strip()

            return llm_policy(
                env,
                agent,
                fallback_fn=lambda: fallback,
                generator=anthropic_generator,
                debug=debug,
            )

        except Exception:
            return fallback

    if provider == "groq" and os.getenv("GROQ_API_KEY"):
        try:
            def groq_generator(prompt: str) -> str:
                return call_groq_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=GROQ_MODEL,
                    temperature=0.2,
                    max_tokens=160,
                )

            return llm_policy(
                env,
                agent,
                fallback_fn=lambda: fallback,
                generator=groq_generator,
                debug=debug,
            )

        except Exception:
            return fallback

    return fallback


def repair_json_candidate(raw: str) -> str:
    """Notebook-style repair for truncated JSON candidates."""
    text = raw or ""
    missing = text.count("{") - text.count("}")
    if missing > 0:
        text += "}" * missing
    return text


def training_mode_generator(provider: str, model_choice: str | None = None, debug: bool = False):
    provider = normalize_provider(provider)
    model_name = resolve_model(provider, model_choice)
    if provider == "groq" and os.getenv("GROQ_API_KEY"):
        def groq_generator(prompt: str) -> str:
            return call_groq_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                temperature=0.7,
                max_tokens=220,
            )

        return groq_generator

    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()

            def anthropic_generator(prompt: str) -> str:
                response = client.messages.create(
                    model=model_name,
                    max_tokens=320,
                    temperature=0.7,
                    system="Return ONLY valid JSON. No markdown. No explanation.",
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text.strip()

            return anthropic_generator
        except Exception:
            if debug:
                print("[training_mode_generator] Anthropic setup failed; using sampled fallback.")
    return None


def evaluate_candidate_action(env: ACEWorldEnv, agent: AgentProfile, action: dict[str, Any], valid_json: bool) -> float:
    env_copy = copy.deepcopy(env)
    available = [item.agent_id for item in env_copy.agents]
    actions = []
    for item in env_copy.agents:
        if item.agent_id == agent.agent_id:
            actions.append(action)
        else:
            actions.append(
                item.choose_fallback_action(
                    env_copy.world.derive_round_probabilities(),
                    env_copy.round_number + 1,
                    available,
                )
            )
    result = env_copy.step(actions)
    reward = next(
        float(item["reward"]["total"])
        for item in result["results"]
        if item["agent"].agent_id == agent.agent_id
    )
    return reward if valid_json else reward - 1.0


def sample_training_action(
    env: ACEWorldEnv,
    agent: AgentProfile,
    provider: str,
    model_choice: str | None = None,
    debug: bool = False,
    k: int = 3,
) -> tuple[dict[str, Any], str]:
    """Notebook-style stochastic multi-sample policy for LLM-Based RL."""
    available = [item.agent_id for item in env.agents]
    fallback = agent.choose_fallback_action(
        env.world.derive_round_probabilities(),
        env.round_number + 1,
        available,
    )
    generator = training_mode_generator(provider, model_choice=model_choice, debug=debug)
    prompt = build_action_prompt(env, agent)
    rng = random.Random((env.round_number + 1) * 1000 + agent.agent_id)
    samples = []

    for sample_idx in range(k):
        if generator is None:
            candidate = random_training_action(agent, env, rng)
            raw = json.dumps(candidate)
        else:
            raw = generator(prompt)

        repaired = repair_json_candidate(raw)
        parsed = extract_first_valid_json(repaired)
        valid_json = parsed is not None
        normalized = normalize_action(parsed, fallback)

        if rng.random() < 0.3:
            normalized["action"] = rng.choice(["challenge", "propose_alliance", "accept_alliance", "betray", "allocate_resources", "execute_contract", "submit_bid"])
        if rng.random() < 0.3:
            normalized["predicted_round"] = rng.choice(["competitive", "cooperative", "resource"])

        reward = evaluate_candidate_action(env, agent, normalized, valid_json)
        samples.append({"idx": sample_idx + 1, "raw": raw, "action": normalized, "reward": reward, "valid_json": valid_json})

    baseline = statistics.mean(item["reward"] for item in samples)
    for item in samples:
        item["advantage"] = item["reward"] - baseline
    best = max(samples, key=lambda item: item["reward"])
    lines = [f"{agent.name} sampled {k} strategies:"]
    for item in samples:
        star = " BEST" if item is best else ""
        lines.append(
            f"  {item['idx']}. {item['action']['action']} -> reward {item['reward']:+.2f}, "
            f"adv {item['advantage']:+.2f}{star}"
        )
    lines.append(f"  Selected: {best['action']['action']} because it scored highest in copied-environment evaluation.")
    return best["action"], "\n".join(lines)


def training_mode_decisions(env: ACEWorldEnv, provider: str, model_choice: str | None = None, debug: bool = False) -> tuple[list[dict[str, Any]], str]:
    actions = []
    reports = []
    for agent in env.agents:
        action, report = sample_training_action(env, agent, provider, model_choice=model_choice, debug=debug, k=3)
        actions.append(action)
        reports.append(report)
    return actions, "\n\n".join(reports)


def render_world(env: ACEWorldEnv) -> tuple[Any, ...]:
    world = env.world
    probs = world.derive_round_probabilities()
    return (
        round(world.oil_price, 3),
        round(world.gold_price, 3),
        round(world.food_index, 3),
        round(world.energy_cost, 3),
        round(world.interest_rate, 4),
        round(world.inflation, 4),
        round(world.gdp_growth, 4),
        round(world.liquidity_index, 3),
        round(world.credit_spread, 3),
        round(world.trade_tension, 3),
        round(world.market_volatility, 3),
        round(world.cooperation_index, 3),
        round(world.resource_scarcity, 3),
        round(world.geopolitical_risk, 3),
        round(world.supply_chain_stability, 3),
        world.economic_regime(),
        world.sector_health,
        (
            f"Competitive: {probs['competitive']:.1%} | "
            f"Cooperative: {probs['cooperative']:.1%} | "
            f"Resource: {probs['resource']:.1%}"
        ),
        render_causal_log(env),
    )


def bar(label: str, value: float, width: int = 12, display: str | None = None) -> str:
    clipped = max(0.0, min(1.0, float(value)))
    filled = int(round(clipped * width))
    shown = display if display is not None else f"{value:.2f}"
    return f"{label:<18} {shown}\n{'█' * filled}{'░' * (width - filled)}"


def render_world_gauges(env: ACEWorldEnv) -> str:
    world = env.world
    market_stress = min(1.0, 0.45 * world.market_volatility + 0.35 * world.geopolitical_risk + 0.2 * world.credit_spread)
    cooperation = world.cooperation_index
    resource_pressure = min(
        1.0,
        0.4 * world.resource_scarcity
        + 0.25 * max(0.0, world.oil_price - 1.0)
        + 0.2 * max(0.0, world.food_index - 1.0)
        + 0.15 * (1.0 - world.supply_chain_stability),
    )
    liquidity = world.liquidity_index
    cards = [
        ("Market Stress", market_stress, "volatility + shocks", "Volatility, geopolitical risk, and credit stress"),
        ("Cooperation", cooperation, "agent alignment", "How favorable the world is for alliances and contracts"),
        ("Resource Pressure", resource_pressure, "scarcity signal", "Oil, food, energy, and supply-chain pressure"),
        ("Liquidity", liquidity, "capital flow", "How easy it is for agents to finance actions"),
    ]

    def status(value: float, inverse: bool = False) -> str:
        score = 1.0 - value if inverse else value
        if score >= 0.7:
            return "Critical" if not inverse else "Strong"
        if score >= 0.45:
            return "Rising" if not inverse else "Moderate"
        return "Stable" if not inverse else "Weak"

    html = ["<div class='world-grid'>"]
    for label, value, caption, title in cards:
        label_status = status(value, inverse=label in {"Cooperation", "Liquidity"})
        html.append(
            f"""
<div class="world-card" title="{escape(title)}">
  <b>{escape(label)} <span class="info-dot" data-tip="{escape(title)}">i</span></b>
  <div class="world-value">{value:.2f}<span class="world-status">{escape(label_status)}</span></div>
  <div class="mini-bar"><div class="mini-fill" style="width:{max(0.0, min(1.0, value)) * 100:.0f}%"></div></div>
  <div class="agent-muted" style="margin-top:8px;">{escape(caption)}</div>
</div>
"""
        )
    html.append("</div>")
    return "\n".join(html)


def render_probability_bars(env: ACEWorldEnv) -> str:
    probs = env.world.derive_round_probabilities()
    return "\n".join(
        [
            bar("Competitive", probs["competitive"], display=f"{probs['competitive']:.0%}"),
            bar("Cooperative", probs["cooperative"], display=f"{probs['cooperative']:.0%}"),
            bar("Resource", probs["resource"], display=f"{probs['resource']:.0%}"),
        ]
    )


def render_flow_strip(env: ACEWorldEnv) -> str:
    event = escape(env.world.event_log[-1] if env.world.event_log else "Waiting for event")
    probs = env.world.derive_round_probabilities()
    likely = max(probs, key=probs.get)
    rounds = env.round_number
    agent_state = "Agents ready" if rounds == 0 else f"Round {rounds} complete"
    return f"""
<div class="story-stage">
<div class="flow-strip">
  <div class="flow-step">
    <b>🌍 Event <span class="info-dot" data-tip="User-provided real-world scenario">i</span></b>
    <span>{event}</span>
  </div>
  <div class="flow-step">
    <b>📊 Shift <span class="info-dot" data-tip="Derived macroeconomic changes from the event">i</span></b>
    <span>{escape(env.world.economic_regime().title())} | stress {env.world.market_volatility:.2f}</span>
  </div>
  <div class="flow-step">
    <b>🤖 Agents <span class="info-dot" data-tip="Multi-agent strategic responses based on incentives">i</span></b>
    <span>{escape(likely.title())} ({probs[likely]:.0%})</span>
  </div>
  <div class="flow-step">
    <b>🎯 Outcome <span class="info-dot" data-tip="Aggregate system behavior after the latest round">i</span></b>
    <span>{escape(agent_state)}</span>
  </div>
</div>
</div>
"""


def render_economic_flow(env: ACEWorldEnv) -> str:
    if not env.world.causal_log:
        return "Event\n  ↓\nNo shock injected yet\n  ↓\nRound probabilities remain near baseline"
    item = env.world.causal_log[-1]
    deltas = item["deltas"]
    changed = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
    first_line = " + ".join(f"{key} {'↑' if value > 0 else '↓'}" for key, value in changed) or "limited direct impact"
    probs = env.world.derive_round_probabilities()
    likely = max(probs, key=probs.get)
    return "\n".join(
        [
            f"Event: {item['event']}",
            "  ↓",
            first_line,
            "  ↓",
            item.get("reasoning", "Economic pressure changes incentives."),
            "  ↓",
            f"{likely.upper()} rounds now most likely ({probs[likely]:.0%})",
        ]
    )


def render_causal_log(env: ACEWorldEnv) -> str:
    if not env.world.causal_log:
        return "No event injected yet."
    lines = []
    for item in env.world.causal_log[-3:]:
        deltas = ", ".join(
            f"{key}: {value:+.2f}"
            for key, value in item["deltas"].items()
            if abs(float(value)) > 1e-9
        )
        sectors = ", ".join(item.get("affected_sectors", [])) or "none"
        lines.append(
            f"Event: {item['event']}\n"
            f"Type: {item.get('event_type', 'unknown')}\n"
            f"Affected sectors: {sectors}\n"
            f"Deltas: {deltas or 'none'}\n"
            f"Reasoning: {item['reasoning']}\n"
            f"Confidence: {item['confidence']:.2f}"
        )
    return "\n\n".join(lines)


def render_agent_rows(env: ACEWorldEnv, round_result: dict[str, Any] | None = None) -> list[list[Any]]:
    result_by_id = {}
    if round_result:
        result_by_id = {item["agent"].agent_id: item for item in round_result["results"]}

    rows = []
    for agent in env.agents:
        item = result_by_id.get(agent.agent_id)
        if item:
            action = item["action"]
            reward = item["reward"]
            rows.append(
                [
                    agent.name,
                    agent.company_type,
                    round(agent.resources, 1),
                    action["predicted_round"],
                    action["action"],
                    json.dumps(action.get("parameters", {})),
                    _belief_text(item.get("beliefs", {})),
                    "correct" if item["correct"] else "wrong",
                    round(reward["total"], 3),
                    round(reward["inference"], 2),
                    round(reward["action"], 2),
                    round(reward.get("market_return", 0.0), 3),
                    action.get("reasoning", ""),
                ]
            )
        else:
            rows.append(
                [
                    agent.name,
                    agent.company_type,
                    round(agent.resources, 1),
                    "-",
                    "-",
                    "-",
                    _belief_text(agent.beliefs),
                    "-",
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    agent.memory_summary(),
                ]
            )
    return rows


def render_agent_cards(env: ACEWorldEnv, round_result: dict[str, Any] | None = None) -> str:
    result_by_id = {}
    if round_result:
        result_by_id = {item["agent"].agent_id: item for item in round_result["results"]}
    cards = []
    previous_resources = _previous_resources(env)
    for agent in env.agents:
        item = result_by_id.get(agent.agent_id)
        resources = round(agent.resources, 1)
        delta = resources - previous_resources.get(agent.name, resources)
        if item:
            action = item["action"]
            reward = item["reward"]
            action_name = action["action"]
            reasoning = action.get("reasoning", "")
            reward_line = f"{reward['total']:+.2f}"
        else:
            action_name = "waiting"
            reasoning = agent.primary_objective
            reward_line = "+0.00"
        stance = _behavior_label(action_name).lower()
        avg_trust = statistics.mean(agent.trust_scores.values()) if agent.trust_scores else 0.5
        belief_value = max(agent.beliefs.values()) if agent.beliefs else 0.0
        resource_value = max(0.0, min(1.0, resources / 160.0))
        reward_value = max(0.0, min(1.0, (float(reward_line) + 2.0) / 4.0))
        reward_class = "#86efac" if reward_line.startswith("+") and reward_line != "+0.00" else "#fda4af" if reward_line.startswith("-") else "#fde68a"
        cards.append(f"""
<div class="agent-card {escape(stance)}">
  <div class="agent-card-header">
    <div>
      <h3>{escape(agent.emoji)} {escape(agent.name)}</h3>
      <small>{escape(agent.company_type)}</small>
    </div>
    <span class="agent-pill">{escape(stance.title())}</span>
  </div>
  <div class="agent-action">{escape(action_name.replace("_", " "))}</div>
  <div class="agent-meta">
    <span>Reward <b style="color:{reward_class};">{escape(reward_line)}</b></span>
    <span>Trust {avg_trust:.2f} <span class="info-dot" data-tip="Mean trust toward other agents">i</span></span>
  </div>
  <div class="agent-meta">
    <span>Resources {resources:.1f}</span>
    <span>{delta:+.1f}</span>
  </div>
  <div class="agent-bars">
    <div class="agent-bar-row"><span>reward</span><div class="agent-bar-track"><div class="agent-bar-fill" style="width:{reward_value * 100:.0f}%"></div></div><span>{escape(reward_line)}</span></div>
    <div class="agent-bar-row"><span>trust</span><div class="agent-bar-track"><div class="agent-bar-fill" style="width:{avg_trust * 100:.0f}%"></div></div><span>{avg_trust:.2f}</span></div>
    <div class="agent-bar-row"><span>belief</span><div class="agent-bar-track"><div class="agent-bar-fill" style="width:{belief_value * 100:.0f}%"></div></div><span>{belief_value:.0%}</span></div>
    <div class="agent-bar-row"><span>capital</span><div class="agent-bar-track"><div class="agent-bar-fill" style="width:{resource_value * 100:.0f}%"></div></div><span>{resources:.0f}</span></div>
  </div>
  <div class="agent-why">Why: {escape(reasoning[:150] or agent.memory_summary()[:150])}</div>
</div>
""")
    return "<div class='agent-grid'>" + "\n".join(cards) + "</div>"


def _belief_text(beliefs: dict[str, float]) -> str:
    if not beliefs:
        return "-"
    return " | ".join(f"{key[:4]} {value:.0%}" for key, value in beliefs.items())


def _best_q_line(agent: AgentProfile) -> str:
    best: tuple[float, str, str] | None = None
    for action, regimes in agent.q_values.items():
        if not isinstance(regimes, dict):
            continue
        for regime, value in regimes.items():
            candidate = (float(value), action, regime)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None or abs(best[0]) < 1e-9:
        return "no learned preference yet"
    return f"best Q={best[1]} in {best[2]} ({best[0]:+.2f})"


def _opponent_model_line(agent: AgentProfile) -> str:
    if not agent.opponent_memory:
        return "no opponent observations yet"
    riskiest_id, model = max(
        agent.opponent_memory.items(),
        key=lambda item: item[1].get("betrayal_rate", 0.0) + item[1].get("aggression", 0.0),
    )
    return (
        f"Agent {riskiest_id}: aggression={model.get('aggression', 0.0):.2f}, "
        f"cooperation={model.get('cooperation', 0.0):.2f}, "
        f"betrayal={model.get('betrayal_rate', 0.0):.2f}"
    )


def _belief_bars(beliefs: dict[str, float]) -> str:
    return "\n".join(bar(key.title(), value, width=10, display=f"{value:.0%}") for key, value in beliefs.items())


def _previous_resources(env: ACEWorldEnv) -> dict[str, float]:
    if not env.round_history:
        return {agent.name: agent.resources for agent in env.agents}
    # Approximate previous resources from current minus latest displayed reward.
    previous = {agent.name: agent.resources for agent in env.agents}
    for item in env.round_history[-1].get("results", []):
        previous[item["name"]] = previous.get(item["name"], 100.0) - float(item.get("reward", 0.0)) * 8.0
    return previous


def render_history(env: ACEWorldEnv) -> str:
    if not env.round_history:
        return "No rounds played yet."
    lines = []
    for entry in reversed(env.round_history[-10:]):
        correct = [item["name"] for item in entry["results"] if item["correct"]]
        lines.append(
            f"Round {entry['round']} | Event: {entry['event']} | Actual: {entry['ground_truth'].upper()} | "
            f"Correct: {', '.join(correct) or 'none'}"
        )
    return "\n".join(lines)


def render_interactions(env: ACEWorldEnv) -> str:
    if not env.interaction_log:
        return "No direct interactions yet. Run a round to see alliances, challenges, and betrayals."
    return "\n".join(f"- {item}" for item in env.interaction_log[-10:])


def render_behavior_evolution(env: ACEWorldEnv) -> str:
    lines = []
    for agent in env.agents:
        actions = [item["action"] for item in agent.self_memory[-5:]]
        if not actions:
            trajectory = "No rounds yet"
        else:
            trajectory = " → ".join(_behavior_label(action) for action in actions)
        latest = agent.self_memory[-1]["reward"] if agent.self_memory else 0.0
        lines.append(f"{agent.name}: {trajectory} (latest reward {latest:+.2f})")
    return "\n".join(lines)


def _behavior_label(action: str) -> str:
    if action in {"challenge", "betray", "submit_bid"}:
        return "Aggressive"
    if action in {"propose_alliance", "accept_alliance", "execute_contract"}:
        return "Cooperative"
    if action == "allocate_resources":
        return "Defensive"
    return "Neutral"


def render_optimal_comparison(round_result: dict[str, Any] | None) -> str:
    if not round_result:
        return "Run a round to compare actions against the hidden optimal behavior."
    ground_truth = round_result["ground_truth"]
    optimal = {
        "competitive": {"challenge", "betray", "submit_bid"},
        "cooperative": {"propose_alliance", "accept_alliance", "execute_contract", "submit_bid"},
        "resource": {"allocate_resources", "execute_contract"},
    }[ground_truth]
    lines = [f"Actual round: {ground_truth.upper()}", f"Optimal actions: {', '.join(sorted(optimal))}"]
    for item in round_result["results"]:
        action = item["action"]["action"]
        mark = "OK" if action in optimal else "MISS"
        lines.append(f"- {item['agent'].name}: {action} {mark}")
    return "\n".join(lines)


def random_training_action(agent: AgentProfile, env: ACEWorldEnv, rng: random.Random) -> dict[str, Any]:
    action = rng.choice(list(ACTIONS))
    parameters: dict[str, Any] = {}
    other_ids = [item.agent_id for item in env.agents if item.agent_id != agent.agent_id]
    if other_ids and action in {"propose_alliance", "accept_alliance", "betray", "challenge"}:
        parameters["target"] = rng.choice(other_ids)
    if action in {"submit_bid", "allocate_resources"}:
        parameters["amount"] = rng.randint(10, 100)
    return {
        "predicted_round": rng.choice(["competitive", "cooperative", "resource"]),
        "action": action,
        "parameters": parameters,
        "reasoning": "random baseline for training comparison",
    }


def mean_training_trust(env: ACEWorldEnv) -> float:
    values = [
        value
        for agent in env.agents
        for value in agent.trust_scores.values()
    ]
    return statistics.mean(values) if values else 0.5


def flatten_training_round(result: dict[str, Any], env: ACEWorldEnv, episode: int, policy: str, scenario: str) -> list[dict[str, Any]]:
    rows = []
    for item in result["results"]:
        action = item["action"]["action"]
        rows.append(
            {
                "episode": episode,
                "policy": policy,
                "scenario": scenario,
                "agent": item["agent"].name,
                "action": action,
                "reward": float(item["reward"]["total"]),
                "correct": int(bool(item["correct"])),
                "cooperation": int(action in COOP_ACTIONS),
                "betrayal": int(action in BETRAY_ACTIONS),
                "aggression": int(action in AGGRESSIVE_ACTIONS),
                "avg_trust": mean_training_trust(env),
            }
        )
    return rows


def grouped_training_mean(rows: list[dict[str, Any]], keys: list[str], fields: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    for row in rows:
        key = tuple(row[item] for item in keys)
        buckets.setdefault(key, {field: [] for field in fields})
        for field in fields:
            buckets[key][field].append(float(row[field]))
    output = []
    for key, values in buckets.items():
        item = {keys[idx]: key[idx] for idx in range(len(keys))}
        for field, numbers in values.items():
            item[field] = statistics.mean(numbers) if numbers else 0.0
        output.append(item)
    return sorted(output, key=lambda item: tuple(item[key] for key in keys))


def train_agents_for_ui(event_text: str, seed: int, episodes: int = 40) -> ACEWorldEnv:
    env = ACEWorldEnv(rng_seed=seed)
    env.apply_event(event_text, provider="fallback")
    for _ in range(episodes):
        env.step()
    return env


def evaluate_training_policy(
    event_text: str,
    scenario: str,
    policy: str,
    seed: int,
    trained_agents: list[AgentProfile] | None = None,
    episodes: int = 10,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    env = ACEWorldEnv(rng_seed=seed)
    env.apply_event(event_text, provider="fallback")
    if trained_agents is not None:
        env.agents = copy.deepcopy(trained_agents)
        env.previous_market = env._market_snapshot()
    rows: list[dict[str, Any]] = []
    for episode in range(episodes):
        if policy == "random_baseline":
            actions = [random_training_action(agent, env, rng) for agent in env.agents]
            result = env.step(actions)
        else:
            result = env.step()
        rows.extend(flatten_training_round(result, env, episode, policy, scenario))
    return rows


def render_training_metric_cards(lifts: list[dict[str, Any]]) -> str:
    oil = next((item for item in lifts if item["scenario"] == "oil_crisis"), {})
    peace = next((item for item in lifts if item["scenario"] == "peace_scenario"), {})
    cards = [
        ("Oil Reward Lift", f"{oil.get('reward_lift_vs_random', 0.0):+.2f}"),
        ("Peace Reward Lift", f"{peace.get('reward_lift_vs_random', 0.0):+.2f}"),
        ("Peace Accuracy Lift", f"{peace.get('accuracy_lift_vs_random', 0.0):+.0%}"),
        ("Trust Delta", f"{peace.get('trust_delta_vs_untrained', 0.0):+.2f}"),
    ]
    return "<div class='training-proof'>" + "".join(
        f"<div class='metric-card'><b>{escape(label)}</b><span>{escape(value)}</span></div>"
        for label, value in cards
    ) + "</div>"


def training_comparison_plot(summary: list[dict[str, Any]]):
    if go is None:
        return None
    fig = go.Figure()
    policies = ["random_baseline", "untrained_fallback", "trained_agents"]
    for scenario in sorted({item["scenario"] for item in summary}):
        values = []
        for policy in policies:
            row = next((item for item in summary if item["scenario"] == scenario and item["policy"] == policy), {})
            values.append(row.get("reward", 0.0))
        fig.add_trace(go.Bar(name=scenario, x=policies, y=values))
    fig.update_layout(
        title="Phase 1 Training Proof: Reward by Policy",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        height=330,
        margin=dict(l=20, r=20, t=45, b=30),
    )
    return fig


def q_value_evidence_rows(env: ACEWorldEnv, scenario: str) -> list[list[Any]]:
    rows = []
    for agent in env.agents:
        for round_type in ["competitive", "cooperative", "resource"]:
            action_values = [
                (action, float(values.get(round_type, 0.0)))
                for action, values in agent.q_values.items()
            ]
            best_action, best_value = max(action_values, key=lambda item: item[1])
            signal = sum(abs(value) for _, value in action_values)
            rows.append([scenario, agent.name, round_type, best_action, round(best_value, 3), round(signal, 3)])
    return rows


def run_phase1_training_proof() -> tuple[str, list[list[Any]], list[list[Any]], list[list[Any]], list[list[Any]], Any, str]:
    trained_envs = {
        scenario: train_agents_for_ui(event_text, seed=900 + idx)
        for idx, (scenario, event_text) in enumerate(TRAINING_SCENARIOS.items())
    }
    eval_rows: list[dict[str, Any]] = []
    for idx, (scenario, event_text) in enumerate(TRAINING_SCENARIOS.items()):
        trained_agents = trained_envs[scenario].agents
        for policy in ["random_baseline", "untrained_fallback", "trained_agents"]:
            eval_rows.extend(
                evaluate_training_policy(
                    event_text,
                    scenario,
                    policy,
                    seed=1000 + idx * 10 + len(eval_rows),
                    trained_agents=trained_agents if policy == "trained_agents" else None,
                )
            )
    summary = grouped_training_mean(
        eval_rows,
        ["scenario", "policy"],
        ["reward", "cooperation", "betrayal", "aggression", "avg_trust", "correct"],
    )
    lifts = []
    for scenario in sorted(TRAINING_SCENARIOS):
        trained = next(item for item in summary if item["scenario"] == scenario and item["policy"] == "trained_agents")
        random_base = next(item for item in summary if item["scenario"] == scenario and item["policy"] == "random_baseline")
        untrained = next(item for item in summary if item["scenario"] == scenario and item["policy"] == "untrained_fallback")
        lifts.append(
            {
                "scenario": scenario,
                "reward_lift_vs_random": trained["reward"] - random_base["reward"],
                "reward_lift_vs_untrained": trained["reward"] - untrained["reward"],
                "accuracy_lift_vs_random": trained["correct"] - random_base["correct"],
                "cooperation_delta_vs_untrained": trained["cooperation"] - untrained["cooperation"],
                "betrayal_delta_vs_untrained": trained["betrayal"] - untrained["betrayal"],
                "aggression_delta_vs_untrained": trained["aggression"] - untrained["aggression"],
                "trust_delta_vs_untrained": trained["avg_trust"] - untrained["avg_trust"],
            }
        )
    summary_rows = [
        [
            item["scenario"],
            item["policy"],
            round(item["reward"], 3),
            round(item["correct"], 3),
            round(item["cooperation"], 3),
            round(item["betrayal"], 3),
            round(item["aggression"], 3),
            round(item["avg_trust"], 3),
        ]
        for item in summary
    ]
    q_rows = []
    for scenario, env in trained_envs.items():
        q_rows.extend(q_value_evidence_rows(env, scenario))
    lift_rows = [
        [
            item["scenario"],
            round(item["reward_lift_vs_random"], 3),
            round(item["reward_lift_vs_untrained"], 3),
            round(item["accuracy_lift_vs_random"], 3),
            round(item["cooperation_delta_vs_untrained"], 3),
            round(item["betrayal_delta_vs_untrained"], 3),
            round(item["aggression_delta_vs_untrained"], 3),
            round(item["trust_delta_vs_untrained"], 3),
        ]
        for item in lifts
    ]
    action_counter = Counter(
        (row["scenario"], row["policy"], row["action"])
        for row in eval_rows
    )
    action_rows = [
        [scenario, policy, action, count]
        for (scenario, policy, action), count in sorted(
            action_counter.items(),
            key=lambda item: (item[0][0], item[0][1], -item[1], item[0][2]),
        )
    ]
    lift_lines = [
        f"{item['scenario']}: reward lift vs random {item['reward_lift_vs_random']:+.3f}, "
        f"vs untrained {item['reward_lift_vs_untrained']:+.3f}, "
        f"accuracy lift {item['accuracy_lift_vs_random']:+.1%}, "
        f"trust delta {item['trust_delta_vs_untrained']:+.3f}"
        for item in lifts
    ]
    narrative = (
        "Phase 1 proof complete. Agents were trained for 40 rounds per scenario, then evaluated on fresh worlds "
        "against random and untrained baselines.\n"
        + "\n".join(lift_lines)
        + "\nThis mirrors the notebook: environment learning first, LLM/GRPO second."
    )
    return render_training_metric_cards(lifts), summary_rows, lift_rows, action_rows, q_rows, training_comparison_plot(summary), narrative


def resource_plot(env: ACEWorldEnv):
    if go is None:
        return None
    fig = go.Figure()
    for agent in env.agents:
        xs = [0]
        ys = [100.0]
        running = 100.0
        for memory in agent.self_memory:
            running += float(memory.get("reward", 0.0)) * 8.0
            xs.append(memory["round"])
            ys.append(running)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=agent.name))
    fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0"),
    height=300,
    margin=dict(l=20, r=20, t=30, b=20),
)
    return fig


def world_plot(env: ACEWorldEnv):
    if go is None:
        return None
    history = env.world_history or [
        {
            "round": 0,
            "oil_price": env.world.oil_price,
            "market_volatility": env.world.market_volatility,
            "cooperation_index": env.world.cooperation_index,
        }
    ]
    xs = [item["round"] for item in history]
    oil = [item["oil_price"] for item in history]
    volatility = [item["market_volatility"] for item in history]
    cooperation = [item["cooperation_index"] for item in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=oil, mode="lines+markers", name="Oil"))
    fig.add_trace(go.Scatter(x=xs, y=volatility, mode="lines+markers", name="Volatility"))
    fig.add_trace(go.Scatter(x=xs, y=cooperation, mode="lines+markers", name="Cooperation"))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), title="World Dynamics")
    return fig


def render_trust(env: ACEWorldEnv) -> dict[str, float]:
    trust = {}
    for agent in env.agents:
        for other_id, value in agent.trust_scores.items():
            trust[f"{agent.agent_id}->{other_id}"] = round(value, 2)
    return trust


def inject_event(event_text: str, provider: str, debug_llm: bool, env: ACEWorldEnv | None, model_choice: str | None = None):
    env = env or make_fresh_env()
    apply_llm_runtime_config(provider, model_choice)
    text = event_text.strip() or "No event provided."
    trace = env.apply_event(text, provider=normalize_provider(provider), debug=bool(debug_llm))
    impact = describe_impact(trace["deltas"], text, trace["reasoning"])
    return (
        env,
        impact,
        *render_world(env),
        render_flow_strip(env),
        render_world_gauges(env),
        render_economic_flow(env),
        render_probability_bars(env),
        render_agent_rows(env),
        render_agent_cards(env),
        render_trust(env),
        render_interactions(env),
        render_behavior_evolution(env),
        render_optimal_comparison(None),
        "Switch to LLM-Based RL and run a round to see sampled actions, rewards, advantages, and selected strategy.",
        resource_plot(env),
        world_plot(env),
        render_history(env),
        "Inject an event, then run a round.",
    )


def select_preset_event(preset_event: str) -> str:
    return preset_event or PERFECT_DEMO_EVENT


def select_provider_model(provider: str) -> str:
    return resolve_model(provider)


def run_simulation(event_text: str, provider: str, model_choice: str, debug_llm: bool, env: ACEWorldEnv | None, policy_mode: str):
    env = env or make_fresh_env()
    provider = normalize_provider(provider)
    llm_ready, llm_status = apply_llm_runtime_config(provider, model_choice)
    if policy_mode == "LLM-Based RL" and not llm_ready:
        return (
            env,
            llm_status,
            *render_world(env),
            render_flow_strip(env),
            render_world_gauges(env),
            render_economic_flow(env),
            render_probability_bars(env),
            render_agent_rows(env),
            render_agent_cards(env),
            render_trust(env),
            render_interactions(env),
            render_behavior_evolution(env),
            render_optimal_comparison(None),
            f"{llm_status}\n\nLLM-Based RL needs a live provider secret because it samples model-generated strategies before selecting the best action.",
            resource_plot(env),
            world_plot(env),
            render_history(env),
            "Waiting for LLM credentials.",
        )
    text = event_text.strip() or PERFECT_DEMO_EVENT
    latest_event = env.world.event_log[-1] if env.world.event_log else None
    if latest_event != text:
        trace = env.apply_event(text, provider=provider, debug=bool(debug_llm))
        impact = describe_impact(trace["deltas"], text, trace["reasoning"])
    else:
        impact = f"Continuing scenario: {text}"

    round_result = run_round(env, provider, debug_llm, policy_mode, model_choice)
    env = round_result[0]
    return (
        env,
        impact,
        *round_result[1:],
    )


def run_round(env: ACEWorldEnv | None, provider: str, debug_llm: bool, policy_mode: str = "Agent-Based RL", model_choice: str | None = None):
    env = env or make_fresh_env()
    provider = normalize_provider(provider)
    llm_ready, llm_status = apply_llm_runtime_config(provider, model_choice)
    training_report = "Agent-Based RL used: agents acted through learned fallback policy, Q-values, trust, memory, and opponent models."

    if policy_mode == "LLM-Based RL":
        if not llm_ready:
            return (
                env,
                *render_world(env),
                render_flow_strip(env),
                render_world_gauges(env),
                render_economic_flow(env),
                render_probability_bars(env),
                render_agent_rows(env),
                render_agent_cards(env),
                render_trust(env),
                render_interactions(env),
                render_behavior_evolution(env),
                render_optimal_comparison(None),
                f"{llm_status}\n\nSet the provider key in .env or Hugging Face Space secrets, confirm provider/model, then run again.",
                resource_plot(env),
                world_plot(env),
                render_history(env),
                "Waiting for LLM credentials.",
            )
        actions, training_report = training_mode_decisions(env, provider, model_choice=model_choice, debug=bool(debug_llm))
        result = env.step(actions)
    else:
        result = env.step()

    ground_truth = result["ground_truth"]
    correct = [item["agent"].name for item in result["results"] if item["correct"]]
    god_mode = (
        f"Actual hidden round: {ground_truth.upper()}\n"
        f"Correct agents: {', '.join(correct) or 'none'}\n"
        f"Alliances: {sorted([list(pair) for pair in env.alliances])}\n"
        f"Regime: {env.world.economic_regime()}"
    )
    return (
        env,
        *render_world(env),
        render_flow_strip(env),
        render_world_gauges(env),
        render_economic_flow(env),
        render_probability_bars(env),
        render_agent_rows(env, result),
        render_agent_cards(env, result),
        render_trust(env),
        render_interactions(env),
        render_behavior_evolution(env),
        render_optimal_comparison(result),
        training_report,
        resource_plot(env),
        world_plot(env),
        render_history(env),
        god_mode,
    )


def run_five_rounds(env: ACEWorldEnv | None, provider: str, debug_llm: bool, policy_mode: str = "Agent-Based RL", model_choice: str | None = None):
    env = env or make_fresh_env()
    provider = normalize_provider(provider)
    llm_ready, llm_status = apply_llm_runtime_config(provider, model_choice)
    last = None
    reports = []
    for _ in range(5):
        if policy_mode == "LLM-Based RL":
            if not llm_ready:
                break
            actions, report = training_mode_decisions(env, provider, model_choice=model_choice, debug=bool(debug_llm))
            reports.append(report)
            last = env.step(actions)
        else:
            last = env.step()
    ground_truth = last["ground_truth"] if last else "none"
    god_mode = f"Ran 5 rounds. Last hidden round: {ground_truth.upper()}."
    training_report = "\n\n--- ROUND SAMPLE TRACE ---\n\n".join(reports[-2:]) if reports else "Agent-Based RL used: repeated rounds updated agent memory, trust, opponent models, and Q-values."
    if policy_mode == "LLM-Based RL" and not llm_ready:
        training_report = f"{llm_status}\n\nSet the provider key in .env or Hugging Face Space secrets, confirm provider/model, then run again."
        god_mode = "Waiting for LLM credentials."
    return (
        env,
        *render_world(env),
        render_flow_strip(env),
        render_world_gauges(env),
        render_economic_flow(env),
        render_probability_bars(env),
        render_agent_rows(env, last),
        render_agent_cards(env, last),
        render_trust(env),
        render_interactions(env),
        render_behavior_evolution(env),
        render_optimal_comparison(last),
        training_report,
        resource_plot(env),
        world_plot(env),
        render_history(env),
        god_mode,
    )


def run_full_demo(provider: str, debug_llm: bool, env: ACEWorldEnv | None, policy_mode: str = "Agent-Based RL", model_choice: str | None = None):
    env = make_fresh_env()
    injected = inject_event(PERFECT_DEMO_EVENT, provider, debug_llm, env, model_choice)
    env = injected[0]
    impact = injected[1]
    round_result = run_round(env, provider, debug_llm, policy_mode, model_choice)
    env = round_result[0]
    final_result = run_five_rounds(env, provider, debug_llm, policy_mode, model_choice)
    return (
        final_result[0],
        impact,
        *final_result[1:],
    )


def reset_demo():
    env = make_fresh_env()
    return (
        env,
        "World reset.",
        *render_world(env),
        render_flow_strip(env),
        render_world_gauges(env),
        render_economic_flow(env),
        render_probability_bars(env),
        render_agent_rows(env),
        render_agent_cards(env),
        render_trust(env),
        render_interactions(env),
        render_behavior_evolution(env),
        render_optimal_comparison(None),
        "Switch to LLM-Based RL and run a round to see sampled actions, rewards, advantages, and selected strategy.",
        resource_plot(env),
        world_plot(env),
        render_history(env),
        "God Mode ready.",
    )


def build_ui():
    if gr is None:
        raise ModuleNotFoundError("Install demo dependencies with: pip install -r requirements_demo.txt")

    world_outputs = []
    with gr.Blocks(title="EIE Economic Intelligence Engine") as demo:
        env_state = gr.State(make_fresh_env())

        gr.HTML(
            """
<div class="hero">
  <h1>EIE</h1>
  <p>Economic Intelligence Engine: Learning Decisions in Dynamic Economic and Geopolitical Systems</p>
  <div class="mode-pill">Event Reported → LLM Interprets → World Change → Agents Act → The world and the agents change, yet again :)</div>
</div>
"""
        )

        with gr.Column(elem_classes=["command-shell"]):
            preset_events = gr.Dropdown(
                choices=PRESET_EVENTS,
                value=PERFECT_DEMO_EVENT,
                label="Sample events",
            )
            event_input = gr.Textbox(
                label=None,
                value=PERFECT_DEMO_EVENT,
                lines=1,
                placeholder="Type an event, e.g. oil crisis hits Middle East",
            )
            with gr.Row():
                policy_mode = gr.Radio(
                    choices=["Agent-Based RL", "LLM-Based RL"],
                    value="Agent-Based RL",
                    label="Mode",
                    scale=3,
                )
                run_btn = gr.Button("Run →", variant="primary", scale=1)
                reset_btn = gr.Button("Reset", variant="secondary", scale=1)
            with gr.Row():
                use_llm = gr.Dropdown(
                    choices=UI_PROVIDER_CHOICES,
                    value=os.getenv("LLM_PROVIDER", "fallback") if os.getenv("LLM_PROVIDER", "fallback") in UI_PROVIDER_CHOICES else "fallback",
                    label="Provider",
                    scale=1,
                )
                model_choice = gr.Textbox(
                    label="Model",
                    value=os.getenv("GROQ_MODEL", GROQ_MODEL),
                    placeholder="llama-3.3-70b-versatile",
                    scale=2,
                )
                debug_llm = gr.Checkbox(label="Debug", value=False, scale=1)
            gr.HTML('<div class="small-note">Agent-Based RL uses Q-values, trust, memory, and opponent models. LLM-Based RL uses provider/model here and reads keys from environment variables or Hugging Face Space secrets.</div>')

        flow_strip = gr.HTML(render_flow_strip(make_fresh_env()))

        gr.HTML('<div class="section-title">World State</div>')
        world_gauges = gr.HTML(render_world_gauges(make_fresh_env()))

        gr.HTML('<div class="section-title">Agents</div>')
        agent_cards = gr.HTML()

        with gr.Accordion("AI Reasoning", open=True, elem_classes=["terminal-panel"]):
            training_samples = gr.Textbox(
                label="LLM-Based RL sampled strategies",
                lines=10,
                interactive=False,
                value="Switch to LLM-Based RL and run a round to see sampled actions, rewards, advantages, and selected strategy.",
            )
            gr.HTML('<div class="small-note">Multiple strategies are evaluated via reward-based selection on copied environments.</div>')

        with gr.Accordion("System Console", open=True, elem_classes=["terminal-panel"]):
            gr.HTML('<div class="small-note">Provider/model are selected in the command bar. Agent-Based RL runs without model calls; LLM-Based RL uses the selected model plus env/Space secrets.</div>')
            impact_box = gr.Textbox(label="Immediate Economic Impact", lines=5, interactive=False)
            economic_flow = gr.Textbox(label="Event → Economy → Incentives", lines=6, interactive=False)
            round_prob_bars = gr.Textbox(label="Hidden Round Pressure", lines=6, interactive=False)
            with gr.Row():
                interactions = gr.Textbox(label="Interaction Story", lines=6, interactive=False)
                behavior = gr.Textbox(label="Behavior Evolution", lines=6, interactive=False)
                optimal = gr.Textbox(label="Actions vs Optimal", lines=6, interactive=False)
            with gr.Row():
                resource_chart = gr.Plot(label="Agent Resources")
                world_chart = gr.Plot(label="World Dynamics")
            causal_box = gr.Textbox(label="Causal Trace", lines=8, interactive=False)
            with gr.Row():
                trust_json = gr.JSON(label="Trust Matrix")
                god_mode = gr.Textbox(label="God Mode Reveal", lines=5, interactive=False)
                history = gr.Textbox(label="Round History", lines=10, interactive=False)
            with gr.Row():
                oil = gr.Number(label="Oil x", precision=3)
                gold = gr.Number(label="Gold x", precision=3)
                food = gr.Number(label="Food x", precision=3)
                energy = gr.Number(label="Energy x", precision=3)
            with gr.Row():
                interest = gr.Number(label="Interest", precision=4)
                inflation = gr.Number(label="Inflation", precision=4)
                gdp = gr.Number(label="GDP Growth", precision=4)
                regime = gr.Textbox(label="Regime", interactive=False)
            with gr.Row():
                liquidity = gr.Slider(0, 1, label="Liquidity", interactive=False)
                credit = gr.Slider(0, 1, label="Credit Spread", interactive=False)
                tension = gr.Slider(0, 1, label="Trade Tension", interactive=False)
            with gr.Row():
                volatility = gr.Slider(0, 1, label="Volatility", interactive=False)
                cooperation = gr.Slider(0, 1, label="Cooperation", interactive=False)
                scarcity = gr.Slider(0, 1, label="Scarcity", interactive=False)
            with gr.Row():
                geopolitics = gr.Slider(0, 1, label="Geopolitical Risk", interactive=False)
                supply_chain = gr.Slider(0, 1, label="Supply Chain Stability", interactive=False)
            sectors = gr.JSON(label="Sector Health")
            probabilities = gr.Textbox(label="Round Probabilities", interactive=False)
            agents_table = gr.Dataframe(
                headers=[
                    "Agent",
                    "Type",
                    "Resources",
                    "Predicted",
                    "Action",
                    "Params",
                    "Beliefs",
                    "Correct",
                    "Total Reward",
                    "Inference",
                    "Action Reward",
                    "Market Return",
                    "Reasoning / Memory",
                ],
                row_count=(7, "dynamic"),
                column_count=(13, "fixed"),
                label="Detailed reward table",
            )

        with gr.Accordion("Training Proof", open=False, elem_classes=["terminal-panel"]):
            gr.HTML('<div class="small-note">Notebook-aligned proof: train agents, compare random vs untrained vs trained, then inspect lift, action shift, and Q-values.</div>')
            phase1_train_btn = gr.Button("Run Phase 1 Training Proof", variant="secondary")
            training_narrative = gr.Textbox(label="Training Workflow Result", lines=5, interactive=False)
            training_metrics = gr.HTML("<div class='training-proof'><div class='metric-card'><b>Status</b><span>Not run</span></div></div>")
            training_chart = gr.Plot(label="Reward Comparison")
            training_table = gr.Dataframe(
                headers=["Scenario", "Policy", "Reward", "Accuracy", "Cooperation", "Betrayal", "Aggression", "Avg Trust"],
                row_count=(6, "dynamic"),
                column_count=(8, "fixed"),
                label="Evaluation summary",
            )
            lift_table = gr.Dataframe(
                headers=["Scenario", "Reward vs Random", "Reward vs Untrained", "Accuracy Lift", "Coop Δ", "Betrayal Δ", "Aggression Δ", "Trust Δ"],
                row_count=(2, "dynamic"),
                column_count=(8, "fixed"),
                label="Comparative lift table",
            )
            action_shift_table = gr.Dataframe(
                headers=["Scenario", "Policy", "Action", "Count"],
                row_count=(40, "dynamic"),
                column_count=(4, "fixed"),
                label="Action distribution shift",
            )
            q_table = gr.Dataframe(
                headers=["Scenario", "Agent", "Round Type", "Best Learned Action", "Best Q", "Q Signal"],
                row_count=(42, "dynamic"),
                column_count=(6, "fixed"),
                label="Learned Q-value evidence",
            )

        world_outputs = [
            oil,
            gold,
            food,
            energy,
            interest,
            inflation,
            gdp,
            liquidity,
            credit,
            tension,
            volatility,
            cooperation,
            scarcity,
            geopolitics,
            supply_chain,
            regime,
            sectors,
            probabilities,
            causal_box,
        ]
        story_outputs = [flow_strip, world_gauges, economic_flow, round_prob_bars]
        sim_outputs = [
            agents_table,
            agent_cards,
            trust_json,
            interactions,
            behavior,
            optimal,
            training_samples,
            resource_chart,
            world_chart,
            history,
            god_mode,
        ]

        preset_events.change(
            select_preset_event,
            inputs=[preset_events],
            outputs=[event_input],
        )
        use_llm.change(
            select_provider_model,
            inputs=[use_llm],
            outputs=[model_choice],
        )
        run_btn.click(
            run_simulation,
            inputs=[event_input, use_llm, model_choice, debug_llm, env_state, policy_mode],
            outputs=[env_state, impact_box, *world_outputs, *story_outputs, *sim_outputs],
        )
        phase1_train_btn.click(
            run_phase1_training_proof,
            outputs=[training_metrics, training_table, lift_table, action_shift_table, q_table, training_chart, training_narrative],
        )
        reset_btn.click(
            reset_demo,
            outputs=[env_state, impact_box, *world_outputs, *story_outputs, *sim_outputs],
        )
        demo.load(
            reset_demo,
            outputs=[env_state, impact_box, *world_outputs, *story_outputs, *sim_outputs],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(share=True, css=APP_CSS)
