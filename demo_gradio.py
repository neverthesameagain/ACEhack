"""Judge-facing ACE++ Option B demo.

Run:
    ANTHROPIC_API_KEY=... python demo_gradio.py

The app also works without an API key by using deterministic adaptive fallback
agents, so the demo never crashes during judging.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    import gradio as gr
except ModuleNotFoundError:  # pragma: no cover
    gr = None

from ace_agents import AgentProfile
from ace_reward import ACTIONS
from ace_text_inject import describe_impact
from ace_world_env import ACEWorldEnv, WorldState


INFERENCE_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")


def make_fresh_env() -> ACEWorldEnv:
    return ACEWorldEnv()


def parse_agent_json(raw: str) -> dict[str, Any] | None:
    clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    decoder = json.JSONDecoder()
    for idx, char in enumerate(clean):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(clean[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def llm_or_fallback_decision(env: ACEWorldEnv, agent: AgentProfile) -> dict[str, Any]:
    available = [item.agent_id for item in env.agents]
    fallback = agent.choose_fallback_action(
        env.world.derive_round_probabilities(),
        env.round_number + 1,
        available,
    )

    if not os.getenv("ANTHROPIC_API_KEY"):
        return fallback

    try:
        import anthropic

        client = anthropic.Anthropic()
        user_prompt = "\n".join(
            [
                f"Upcoming round: {env.round_number + 1}",
                f"Visible alliances: {sorted([list(pair) for pair in env.alliances])}",
                "Recent global round history:",
                json.dumps(env.round_history[-3:], indent=2),
                "Return ONLY the JSON object.",
            ]
        )
        response = client.messages.create(
            model=INFERENCE_MODEL,
            max_tokens=260,
            system=agent.system_prompt(env.world.to_prompt_str()),
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        parsed = parse_agent_json(raw)
        if not isinstance(parsed, dict):
            return fallback
        return {
            "predicted_round": parsed.get("predicted_round", fallback["predicted_round"]),
            "action": parsed.get("action", fallback["action"]),
            "parameters": parsed.get("parameters", fallback["parameters"]),
            "reasoning": parsed.get("reasoning", fallback["reasoning"]),
        }
    except Exception:
        return fallback


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
        round(world.trade_tension, 3),
        round(world.market_volatility, 3),
        round(world.cooperation_index, 3),
        round(world.resource_scarcity, 3),
        (
            f"Competitive: {probs['competitive']:.1%} | "
            f"Cooperative: {probs['cooperative']:.1%} | "
            f"Resource: {probs['resource']:.1%}"
        ),
        render_causal_log(env),
    )


def render_causal_log(env: ACEWorldEnv) -> str:
    if not env.world.causal_log:
        return "No event injected yet."
    lines = []
    for item in env.world.causal_log[-3:]:
        deltas = ", ".join(f"{key}: {value:+.2f}" for key, value in item["deltas"].items())
        lines.append(
            f"Event: {item['event']}\n"
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
                    "correct" if item["correct"] else "wrong",
                    round(reward["total"], 3),
                    round(reward["inference"], 2),
                    round(reward["action"], 2),
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
                    "-",
                    0.0,
                    0.0,
                    0.0,
                    agent.memory_summary(),
                ]
            )
    return rows


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


def render_trust(env: ACEWorldEnv) -> dict[str, float]:
    trust = {}
    for agent in env.agents:
        for other_id, value in agent.trust_scores.items():
            trust[f"{agent.agent_id}->{other_id}"] = round(value, 2)
    return trust


def inject_event(event_text: str, env: ACEWorldEnv | None):
    env = env or make_fresh_env()
    text = event_text.strip() or "No event provided."
    trace = env.apply_event(text)
    impact = describe_impact(trace["deltas"], text, trace["reasoning"])
    return (
        env,
        impact,
        *render_world(env),
        render_agent_rows(env),
        render_trust(env),
        render_history(env),
        "Inject an event, then run a round.",
    )


def run_round(env: ACEWorldEnv | None, use_llm: bool):
    env = env or make_fresh_env()
    if use_llm:
        actions = [llm_or_fallback_decision(env, agent) for agent in env.agents]
    else:
        probs = env.world.derive_round_probabilities()
        ids = [agent.agent_id for agent in env.agents]
        actions = [agent.choose_fallback_action(probs, env.round_number + 1, ids) for agent in env.agents]

    result = env.step(actions)
    ground_truth = result["ground_truth"]
    correct = [item["agent"].name for item in result["results"] if item["correct"]]
    god_mode = (
        f"Actual hidden round: {ground_truth.upper()}\n"
        f"Correct agents: {', '.join(correct) or 'none'}\n"
        f"Alliances: {sorted([list(pair) for pair in env.alliances])}"
    )
    return (
        env,
        *render_world(env),
        render_agent_rows(env, result),
        render_trust(env),
        render_history(env),
        god_mode,
    )


def run_five_rounds(env: ACEWorldEnv | None, use_llm: bool):
    env = env or make_fresh_env()
    last = None
    for _ in range(5):
        if use_llm:
            actions = [llm_or_fallback_decision(env, agent) for agent in env.agents]
        else:
            probs = env.world.derive_round_probabilities()
            ids = [agent.agent_id for agent in env.agents]
            actions = [agent.choose_fallback_action(probs, env.round_number + 1, ids) for agent in env.agents]
        last = env.step(actions)
    ground_truth = last["ground_truth"] if last else "none"
    god_mode = f"Ran 5 rounds. Last hidden round: {ground_truth.upper()}."
    return (
        env,
        *render_world(env),
        render_agent_rows(env, last),
        render_trust(env),
        render_history(env),
        god_mode,
    )


def reset_demo():
    env = make_fresh_env()
    return (
        env,
        "World reset.",
        *render_world(env),
        render_agent_rows(env),
        render_trust(env),
        render_history(env),
        "God Mode ready.",
    )


def build_ui():
    if gr is None:
        raise ModuleNotFoundError("Install demo dependencies with: pip install -r requirements_demo.txt")

    world_outputs = []
    with gr.Blocks(title="ACE++ Option B", theme=gr.themes.Soft()) as demo:
        env_state = gr.State(make_fresh_env())

        gr.Markdown("# ACE++ Option B: Adaptive Coalition Economy")
        gr.Markdown(
            "Type a real-world event, watch the economy change, then run multi-agent rounds. "
            "Agents remember outcomes, model opponents, and adapt habits over time."
        )

        with gr.Row():
            with gr.Column(scale=1):
                event_input = gr.Textbox(
                    label="World Event",
                    value="oil crisis hits Middle East",
                    lines=2,
                )
                use_llm = gr.Checkbox(
                    label="Use Anthropic API for agent decisions if configured",
                    value=bool(os.getenv("ANTHROPIC_API_KEY")),
                )
                with gr.Row():
                    inject_btn = gr.Button("Inject Event", variant="primary")
                    round_btn = gr.Button("Run Round", variant="secondary")
                    burst_btn = gr.Button("Run 5 Rounds")
                    reset_btn = gr.Button("Reset", variant="stop")
                impact_box = gr.Textbox(label="Economic Impact", lines=8, interactive=False)
                causal_box = gr.Textbox(label="Causal Trace", lines=10, interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("### World State")
                with gr.Row():
                    oil = gr.Number(label="Oil x", precision=3)
                    gold = gr.Number(label="Gold x", precision=3)
                with gr.Row():
                    food = gr.Number(label="Food x", precision=3)
                    energy = gr.Number(label="Energy x", precision=3)
                with gr.Row():
                    interest = gr.Number(label="Interest", precision=4)
                    inflation = gr.Number(label="Inflation", precision=4)
                    gdp = gr.Number(label="GDP Growth", precision=4)
                tension = gr.Slider(0, 1, label="Trade Tension", interactive=False)
                volatility = gr.Slider(0, 1, label="Volatility", interactive=False)
                cooperation = gr.Slider(0, 1, label="Cooperation", interactive=False)
                scarcity = gr.Slider(0, 1, label="Scarcity", interactive=False)
                probabilities = gr.Textbox(label="Round Probabilities", interactive=False)

        gr.Markdown("### Agent Decisions")
        agents_table = gr.Dataframe(
            headers=[
                "Agent",
                "Type",
                "Resources",
                "Predicted",
                "Action",
                "Params",
                "Correct",
                "Total Reward",
                "Inference",
                "Action Reward",
                "Reasoning / Memory",
            ],
            row_count=(4, "fixed"),
            col_count=(11, "fixed"),
            label="Live agent panel",
        )

        with gr.Row():
            trust_json = gr.JSON(label="Trust Matrix")
            god_mode = gr.Textbox(label="God Mode Reveal", lines=5, interactive=False)
            history = gr.Textbox(label="Round History", lines=10, interactive=False)

        world_outputs = [
            oil,
            gold,
            food,
            energy,
            interest,
            inflation,
            gdp,
            tension,
            volatility,
            cooperation,
            scarcity,
            probabilities,
            causal_box,
        ]

        inject_btn.click(
            inject_event,
            inputs=[event_input, env_state],
            outputs=[env_state, impact_box, *world_outputs, agents_table, trust_json, history, god_mode],
        )
        round_btn.click(
            run_round,
            inputs=[env_state, use_llm],
            outputs=[env_state, *world_outputs, agents_table, trust_json, history, god_mode],
        )
        burst_btn.click(
            run_five_rounds,
            inputs=[env_state, use_llm],
            outputs=[env_state, *world_outputs, agents_table, trust_json, history, god_mode],
        )
        reset_btn.click(
            reset_demo,
            outputs=[env_state, impact_box, *world_outputs, agents_table, trust_json, history, god_mode],
        )
        demo.load(
            reset_demo,
            outputs=[env_state, impact_box, *world_outputs, agents_table, trust_json, history, god_mode],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(share=True)
