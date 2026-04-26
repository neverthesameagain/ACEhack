# EIE Economic Intelligence Engine Design Document

Learning Decisions in Dynamic Economic and Geopolitical Systems

## 1. Executive Summary

EIE Economic Intelligence Engine is a demo-first multi-agent economic simulation. A user injects a natural-language event, the event changes a structured economic world, seven strategic agents observe the world with uncertainty, choose structured actions, interact socially, receive decomposed rewards, and update future behavior through memory, trust, opponent modeling, and Q-values.

The core story is:

```text
Text Event
-> Structured Economic Deltas
-> World State + Hidden Round Probabilities
-> Noisy Agent Observations
-> Agent Beliefs + Memory + Trust
-> Structured Actions
-> Social Resolution + Market Effects
-> Rewards
-> Updated Resources, Trust, Opponent Models, Q-values
-> Behavior Shift Over Time
```

The system is intentionally optimized for stability and interpretability. It is not trying to be perfect macroeconomic forecasting or full-scale RL. It is designed to show a clear, judge-friendly causal chain: events change incentives, incentives change behavior, repeated interaction changes strategy.

## 2. Product Goals

- Show natural-language events affecting a live economy.
- Make agent thinking visible through beliefs, actions, reasoning, rewards, trust, memory, and Q-values.
- Demonstrate agent-level learning before LLM training.
- Keep the demo reliable even when live LLM calls fail.
- Use one shared LLM-to-action policy path in both the demo and notebook.
- Support Hugging Face Spaces deployment and OpenEnv-style evaluation.

## 3. Non-Goals

- No claim of real financial prediction accuracy.
- No full-scale PPO or large distributed RL training.
- No dependency on LLM calls for the demo to work.
- No unbounded or unconstrained world updates from raw model output.

## 4. Runtime Entry Points

### Hugging Face Space

`app.py` is the Space entrypoint:

```text
app.py
-> demo_gradio.build_ui()
-> demo.launch(css=APP_CSS)
```

The Space metadata in `README.md` points to `app_file: app.py`.

### Interactive Demo

`demo_gradio.py` owns the judge-facing UI:

- event input and preset scenarios
- provider selector: `fallback`, `groq`, `anthropic`
- raw model debug checkbox
- event injection
- single-round execution
- five-round execution
- full scripted demo button
- world gauges and probability bars
- agent table and agent cards
- trust map
- interaction log
- behavior evolution panel
- action-vs-optimal comparison
- resource and world plots

### OpenEnv

`openenv.yaml` points to:

```text
entry: openenv_ace:ACEOpenMultiAgentEnv
app_entry: app.py
```

`openenv_ace.py` wraps the same `ACEWorldEnv` used by the demo. It exposes:

- `reset() -> state`
- `step(actions) -> (state, rewards, done, info)`
- `state() -> dict`

If `openenv` is not installed locally, the file defines a small fallback `Environment` base class so smoke tests still import.

## 5. Current Runtime Files

- `app.py`: Hugging Face Space launcher.
- `demo_gradio.py`: UI and demo orchestration.
- `ace_world_env.py`: world state, hidden round sampling, environment step loop, social effects.
- `ace_agents.py`: seven differentiated adaptive agents.
- `ace_reward.py`: decomposed reward functions.
- `ace_text_inject.py`: natural language event to structured economic delta parser.
- `ace_llm_policy.py`: shared LLM-to-JSON-to-action reliability layer.
- `openenv_ace.py`: OpenEnv adapter.
- `openenv.yaml`: OpenEnv metadata.
- `ACE_Experimental_Narrative.ipynb`: two-phase experimental narrative notebook.
- `requirements.txt`: demo/runtime dependencies.
- `legacy/`: older prototypes and unused files.

## 6. World Model

`ace_world_env.py` defines `WorldState`.

### Economic Variables

The world tracks:

- `oil_price`
- `gold_price`
- `food_index`
- `energy_cost`
- `interest_rate`
- `inflation`
- `gdp_growth`
- `trade_tension`
- `market_volatility`
- `cooperation_index`
- `resource_scarcity`
- `liquidity_index`
- `credit_spread`
- `geopolitical_risk`
- `supply_chain_stability`
- `sector_health`

Sector health covers:

- `energy`
- `agriculture`
- `finance`
- `manufacturing`
- `technology`

### Economic Regimes

`WorldState.economic_regime()` maps world variables to readable regimes:

- `crisis`
- `stagflation`
- `recession`
- `inflationary`
- `growth`
- `mixed`

These regimes are for explanation and visualization. The actual game round is sampled from hidden round probabilities.

### Hidden Round Types

The environment samples one hidden ground-truth round type each step:

- `competitive`
- `cooperative`
- `resource`

`WorldState.derive_round_probabilities()` computes the probabilities from world signals:

- competitive pressure rises with trade tension, volatility, oil spikes, geopolitical risk, and credit spread.
- cooperative pressure rises with cooperation index, lower volatility, positive growth, and liquidity.
- resource pressure rises with scarcity, food pressure, energy pressure, and supply-chain instability.

This makes the world a lightweight POMDP: agents do not directly know the sampled hidden round. They infer it from noisy observations.

## 7. Text-to-Economy Translation

`ace_text_inject.py` converts user text into economic deltas.

### Provider Path

`parse_event_payload()` supports:

- Groq when `LLM_PROVIDER=groq` and `GROQ_API_KEY` are set.
- Anthropic when `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` are set.
- deterministic fallback rules otherwise.

Default Groq model in the demo:

```text
llama-3.3-70b-versatile
```

Default Anthropic model:

```text
claude-sonnet-4-20250514
```

### Event Schema

The event parser asks the model for strict JSON containing:

- `event_type`
- `deltas`
- `confidence`
- `reasoning`
- `affected_sectors`

The deltas include all supported world variables and nested `sector_health`.

### Safety

Event parsing is bounded and fault tolerant:

- empty event returns an empty payload.
- malformed LLM output falls back to deterministic rules.
- repeated events can be cached.
- deltas are clamped.
- world variables are clamped by the environment.
- fallback rules cover major demo scenarios such as oil crisis, peace agreement, rate hike, supply-chain disruption, systemic crisis, and trade conflict.

Important distinction:

- `ace_text_inject.py` handles text-to-world parsing.
- `ace_llm_policy.py` handles agent action parsing with the stack-based JSON extractor.

## 8. Environment Step Loop

`ACEWorldEnv.step()` is the central simulation loop.

For each round:

1. Increment `round_number`.
2. Apply endogenous world dynamics.
3. Sample hidden `ground_truth` round type from world probabilities.
4. Generate noisy observations for each agent.
5. Use provided actions or generate fallback actions.
6. Sanitize each action through `_safe_action()`.
7. Resolve social side effects.
8. Compute reward components.
9. Add market return, social bonus, and central-bank stability bonus.
10. Update each agent's beliefs and memory.
11. Update agent resources, balance sheet, strategy counters, opponent models, and Q-values.
12. Append round history and world history.
13. Return `ground_truth`, per-agent `results`, and a `history_entry`.

If fewer actions are passed than there are agents, the missing actions are filled with fallback actions. This prevents partial LLM failure from breaking a round.

## 9. Agents

`ace_agents.py` defines seven agents:

| ID | Agent | Role | Strategic Bias |
| --- | --- | --- | --- |
| 0 | PetroCorp | Energy Company | Benefits from oil spikes, high risk tolerance, less cooperative |
| 1 | GlobalFoods Inc | Food Importer & Distributor | Wants stable supply chains and cooperation |
| 2 | Aurelius Capital | Hedge Fund | Likes volatility and aggressive opportunities |
| 3 | CentralBank of EIE | Central Bank / Regulator | Wants stability, low systemic risk, high cooperation |
| 4 | LogiChain Global | Logistics & Supply Chain Operator | Wants supply-chain stability and capacity preservation |
| 5 | ShieldRe Insurance | Insurance & Risk Underwriter | Avoids systemic loss and unreliable partners |
| 6 | NovaTech Systems | Technology Infrastructure Firm | Wants growth contracts and infrastructure resilience |

### Static Agent State

Each agent has:

- company type
- emoji
- primary objective
- oil exposure
- gold exposure
- food exposure
- cooperation preference
- risk tolerance

### Dynamic Agent State

Each agent tracks:

- resources
- balance sheet
- portfolio
- beliefs over round types
- self memory
- opponent memory
- trust scores
- strategy success counters
- Q-values by action and round type

`fresh_agent_profiles()` deep-copies the templates and initializes trust scores against every other agent.

## 10. Agent-Level Learning

Agent-level learning is the primary learning mechanism in the live environment.

### Belief Updates

Agents update beliefs with `update_beliefs()`. Beliefs combine:

- previous beliefs
- world-derived round probabilities
- noisy observed signals such as oil, volatility, trade tension, scarcity, food pressure, energy cost, cooperation, and liquidity

### Fallback Policy

`choose_fallback_action()` is not a dumb default. It is the stable local policy used when LLMs are disabled or fail.

It scores candidate actions using:

- expected profit
- expected risk
- trust alignment
- historical strategy success
- Q-values
- identity bias
- opponent adjustment

It also includes epsilon-greedy exploration.

### Q-Value Updates

After each round, `update_after_round()` updates Q-values roughly as:

```text
Q[action][round_type] <- Q[action][round_type] + alpha * (reward - Q[action][round_type])
```

This is lightweight tabular adaptation, not deep RL.

### Trust and Opponent Modeling

Agents track other agents' tendencies:

- aggression
- cooperation
- betrayal rate

Social actions change trust:

- alliances and accepted cooperation can raise trust.
- betrayal drops trust sharply.
- challenges can damage trust and produce winner/loser effects.

## 11. Action Space

The reward module defines these actions:

- `submit_bid`
- `propose_alliance`
- `accept_alliance`
- `reject_alliance`
- `betray`
- `challenge`
- `allocate_resources`
- `execute_contract`

The current strict LLM prompt exposes the demo-facing action set:

- `challenge`
- `propose_alliance`
- `accept_alliance`
- `betray`
- `allocate_resources`
- `execute_contract`
- `submit_bid`

`reject_alliance` remains supported by the simulator/reward layer but is not emphasized in the strict LLM action prompt.

## 12. Reward Design

`ace_reward.py` decomposes rewards to avoid reward hacking.

### Base Components

Weights:

- inference: `1.0`
- action: `0.5`
- format: `0.25`
- personality: `0.2`
- behavior: `0.2`

Components:

- `compute_inference_reward()`: rewards correct hidden-round prediction.
- `compute_action_reward()`: rewards action family suitability for the actual round.
- `compute_format_reward()`: rewards concise valid structure and penalizes invalid JSON.
- `compute_personality_reward()`: rewards actions aligned with agent identity.
- `compute_behavior_reward()`: rewards actions that have historically worked for that agent in that round type.

### Environment Additions

`ACEWorldEnv.step()` adds:

- market return based on world movement and exposures
- social bonus from interactions
- central bank stability bonus for `CentralBank of EIE`

The UI displays these components so judges can see why an agent was rewarded.

## 13. Shared LLM Policy Layer

`ace_llm_policy.py` is the shared LLM decision layer used by:

- `demo_gradio.py`
- `ACE_Experimental_Narrative.ipynb`

This prevents the notebook and production demo from drifting into different prompt/parser behavior.

The UI intentionally separates two reinforcement-learning modes:

- `Agent-Based RL`: the default live simulation mode. Agents act through the environment policy and update Q-values, trust, memory, strategy success, and opponent models over repeated rounds.
- `LLM-Based RL`: the notebook-style policy optimization mode. It samples three candidate LLM strategies, repairs/extracts JSON, evaluates each candidate on copied environments, computes relative advantage, and selects the highest-reward strategy.

This is separation, not blending. Agent-Based RL proves the environment supports adaptive agents; LLM-Based RL shows how the LLM policy can be optimized to internalize those strategies.

### Prompt Contract

`build_action_prompt(env, agent)` includes:

- strict role instruction
- valid actions
- required JSON format
- world state
- agent role and objective
- trust scores
- memory summary
- visible alliances
- recent history

It always ends with:

```text
JSON:
```

### Generation

`generate_action()` supports:

- local Hugging Face model + tokenizer path for notebook experiments
- Groq path when `LLM_PROVIDER=groq`
- empty output when no model/provider is configured

Local generation uses:

- `max_new_tokens=120`
- `temperature=0.2`
- `do_sample=True`
- prompt echo removal

LLM-Based RL uses a separate sampling wrapper in `demo_gradio.py`:

- provider temperature `0.7`
- `k=3` sampled candidate actions
- notebook-style truncated JSON brace repair
- optional action/round exploration
- copied-environment reward evaluation
- best-action selection by reward / advantage

### Stack-Based JSON Extraction

`extract_first_valid_json()` scans the raw text, balances braces, and returns the first valid JSON object.

This handles:

- prompt echo
- extra text
- markdown fences
- multiple objects
- trailing junk
- malformed partial output

If no valid JSON is found, it returns `None`.

### Normalization and Fallback

`normalize_action()` validates:

- `predicted_round`
- `action`
- `parameters`
- numeric `amount`
- optional `beliefs`
- optional `factors`
- `reasoning`

`llm_policy()` always returns a valid action:

```text
fallback = fallback_fn()
prompt = build_action_prompt(env, agent)
raw = generate_action(prompt)
parsed = extract_first_valid_json(raw)
return normalize_action(parsed, fallback)
```

Any exception returns fallback.

## 14. Demo UI Flow

### Preset Scenarios

Current presets:

- `oil crisis hits Middle East`
- `global cooperation agreement signed`
- `major food supply chain disruption`

### User Flow

The visible UI is now a terminal-like storytelling interface rather than a research dashboard. The background is pitch black, the typography is monospace, borders are crisp, and gradients are intentionally avoided.

1. Type a real-world event into the floating command bar.
2. Choose `Agent-Based RL` or `LLM-Based RL`.
3. Choose provider/model directly in the command bar when using LLM-Based RL.
4. Click the single primary action: `Run ->`.
5. The app injects a new event if needed, runs the next round, and updates the story.
6. Inspect the story strip, compact world state, and agent cards.
7. `AI Reasoning` is visible by default so judges can immediately see sampled actions when `LLM-Based RL` is active.
8. `System Console` is visible in the main frame for advanced state and diagnostics without presenting it as a separate page.
9. `Training Proof` sits below the System Console and contains notebook-style lift, action-shift, and Q-value evidence.

### Rendered Panels

Visible by default:

- terminal-style premium header
- floating event command bar
- mode toggle: `Agent-Based RL | LLM-Based RL`
- provider/model picker
- one primary `Run ->` button and one secondary `Reset` button
- story strip: event -> shift -> agents -> outcome
- compact 2x2 world state cards
- agent cards with compact reward/trust/belief/capital bars
- AI reasoning console

Secondary diagnostics:

- `AI Reasoning`: visible by default; sampled LLM actions, rewards, advantages, selected strategy.
- `System Console`: raw state, trust matrix, logs, plots, reward table, god mode.
- `Training Proof`: random vs untrained vs trained comparison, lift table, action distribution shift, and Q-value evidence.

The main screen avoids dense grids while still exposing diagnostics in a terminal-style main-frame console.

### Phase 1 Training Proof Workflow

The Gradio app now mirrors the notebook's strongest Phase 1 evidence. The `Run Phase 1 Training Proof` button:

1. trains fresh agents for `40` rounds in the oil-crisis scenario.
2. trains fresh agents for `40` rounds in the peace scenario.
3. evaluates `random_baseline`, `untrained_fallback`, and `trained_agents` on fresh comparable worlds.
4. reports reward lift, accuracy lift, cooperation/betrayal/aggression deltas, and trust delta.
5. displays action-distribution shift across policies.
6. displays Q-value evidence showing the best learned action per agent and hidden round type.

This workflow is intentionally independent from the live event simulation state. It gives judges a quick proof that the learning claims in the notebook are not only offline analysis; the deployed UI can reproduce the same training comparison.

## 15. Notebook Design

`ACE_Experimental_Narrative.ipynb` is structured in two phases.

### Phase 1: Agent-Level Training

This phase comes first and is the primary evidence of learning.

Goal:

```text
Agents exhibit adaptive, strategic behavior driven by rewards and interactions.
```

It uses the existing environment mechanisms:

- Q-value updates
- trust dynamics
- memory updates
- opponent modeling
- strategy success counters

It explicitly frames this as tabular-RL-like policy adaptation, not deep PPO.

Mandatory quantitative outputs:

- reward vs episodes
- cooperation rate vs episodes
- betrayal rate vs episodes
- trust evolution

Mandatory qualitative cases:

- Oil Crisis: before learning random behavior, after learning more aggressive/defensive crisis behavior.
- Peace Scenario: before unnecessary competition, after cooperation emerges.
- Repeated Interaction: alliance, betrayal, trust drop, then adaptation.

Phase 1 also includes an explicit training/evaluation section:

- train scenario-specific agents for repeated rounds.
- preserve learned agent state: Q-values, trust, memory, opponent models, and strategy counters.
- evaluate random, untrained fallback, and trained agents on fresh worlds.
- compute reward lift, accuracy lift, cooperation/betrayal/aggression deltas, trust delta, action distribution shift, and Q-value evidence.

The same comparison is exposed in the Gradio app through the `Run Phase 1 Training Proof` workflow.

### Phase 2: LLM Policy Training with GRPO

This comes second as an advanced extension.

The notebook explicitly states:

```text
We use lightweight GRPO (Group Relative Policy Optimization),
not full-scale PPO training, due to compute constraints.
```

Dataset flow:

```text
create base_env
build prompt
sample multiple completions
for each completion:
    deepcopy base_env
    evaluate completion independently
compute group baseline
advantages = rewards - baseline
store datapoint
```

Required datapoint format:

```python
{
    "prompt": str,
    "completions": [str, str, str],
    "advantages": [float, float, float],
}
```

The key correctness invariant is that each completion is evaluated on the same environment state using a deep copy. No sample mutates shared state.

The GRPO trainer cell is defensive. If optional dependencies or accelerator support are missing, the notebook records the error and still runs the rest of the analysis.

## 16. OpenEnv Adapter

`ACEOpenMultiAgentEnv` is intentionally thin.

Constructor:

```python
ACEOpenMultiAgentEnv(seed=7, max_rounds=10, event_text=None)
```

Behavior:

- creates `ACEWorldEnv(rng_seed=seed)`
- optionally applies `event_text` through fallback event parsing
- `reset()` recreates the seeded environment
- `step(actions)` parses JSON string/list/dict actions
- invalid actions become `None`/empty dicts, letting the environment fallback path handle them
- returns list of per-agent rewards
- `done` becomes true when `round_number >= max_rounds`

This makes OpenEnv use the same simulator as the demo rather than a separate environment.

## 17. Deployment and Configuration

### Dependencies

`requirements.txt` currently includes:

- `gradio>=6.0.0`
- `anthropic>=0.20.0`
- `groq`
- `plotly>=5.0.0`
- `pandas>=2.0.0`
- `matplotlib>=3.8.0`

### Secrets

Fallback mode requires no secrets.

Live LLM event parsing and decisions can use:

```text
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
```

or:

```text
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

The deployed app should not require OpenAI credentials.

## 18. Reliability Guarantees

The system is designed to keep running under common failure modes.

### Event Injection Failure

If the event LLM fails:

- fallback event rules run
- deltas are bounded
- causal trace is still produced

### Agent LLM Failure

If the action LLM fails:

- `extract_first_valid_json()` returns `None`
- `llm_policy()` returns fallback action
- the environment continues

### Bad Action Payload

If an action is malformed:

- `_safe_action()` coerces missing/invalid fields
- unknown round types become `resource`
- unknown actions become `allocate_resources`
- numeric amounts default to `50.0`

### Missing Actions

If fewer than seven actions are supplied:

- `ACEWorldEnv.step()` fills the missing actions with fallback policy decisions.

### Missing Plotting Libraries in Notebook

The notebook plot cells are defensive:

- use `matplotlib` when installed
- print computed series if plotting is unavailable

## 19. Main Design Tradeoffs

- Agent-level learning is prioritized over expensive LLM training because it is faster, visible, and reliable in a live demo.
- The fallback policy is intentionally strong so the demo remains coherent without model calls.
- GRPO is included as a compact training narrative and policy-improvement extension, not as the core runtime dependency.
- Dense decomposed rewards are preferred over a single opaque score because judges can inspect why behavior changed.
- Economic dynamics are stylized and bounded to favor interpretability over realism.
- The OpenEnv adapter is thin to avoid maintaining two separate environment implementations.

## 20. Success Criteria

The system is successful if:

- free-text events visibly change world variables and round probabilities
- agents behave differently based on identity and exposure
- repeated rounds change trust, memory, Q-values, and action choices
- social interactions include cooperation, betrayal, and trust shifts
- LLM output parsing never crashes the simulation
- the demo works without API keys
- the notebook starts with agent-level learning and only then introduces GRPO
- the GRPO dataset has nonzero size and uses identical-state evaluation
- Hugging Face launches through `app.py`
- OpenEnv can import `openenv_ace:ACEOpenMultiAgentEnv`

# EIE Economic Intelligence Engine Design Document

## Purpose

EIE Economic Intelligence Engine is a demo-first multi-agent economic simulation. A user enters a real-world event, the system translates that event into economic state changes, seven strategic agents act under noisy observations, and the UI shows how incentives, trust, rewards, resources, and behavior evolve over repeated rounds.

The product goal is not perfect reinforcement learning. The goal is a robust, interpretable, and demo-safe learning system where the causal chain is visible:

```text
Text event -> World state deltas -> Hidden regime probabilities -> Agent beliefs -> Structured actions -> Rewards -> Memory/trust/Q-values -> Next behavior
```

## System Goals

- Make agent reasoning and adaptation visible to judges.
- Keep the demo running even when LLM providers fail or return malformed text.
- Use one shared LLM-to-action policy in both the Gradio demo and the GRPO notebook.
- Preserve reproducibility through deterministic fallback policies and seeded environments.
- Keep all state changes bounded, inspectable, and explained.

## Runtime Entry Points

- `app.py`: Hugging Face Space entrypoint. Builds and launches the Gradio app.
- `demo_gradio.py`: Main interactive UI and demo orchestration.
- `openenv.yaml`: OpenEnv metadata. Points to `openenv_ace:ACEOpenMultiAgentEnv`.
- `openenv_ace.py`: OpenEnv-compatible wrapper around the same simulation environment.
- `ACE_Experimental_Narrative.ipynb`: Training and analysis notebook for GRPO, JSON safety, and behavior evolution.

## Core Modules

### `ace_world_env.py`

Owns the economic world and multi-agent environment.

- `WorldState` stores macro signals, market variables, game signals, sector health, liquidity, credit spread, geopolitical risk, and supply chain stability.
- `WorldState.apply_event()` applies structured deltas from the event parser, then runs endogenous dynamics.
- `WorldState.derive_round_probabilities()` converts world signals into probabilities for hidden round types: `competitive`, `cooperative`, and `resource`.
- `WorldState.noisy_observation()` gives each agent a partial, noisy view of the world.
- `ACEWorldEnv.step()` advances one round, samples the hidden ground truth, evaluates agent actions, applies social side effects, computes rewards, updates agent memory, and records history.

### `ace_text_inject.py`

Translates natural language events into validated economic deltas.

- Uses a strict event JSON schema for LLM providers.
- Supports Groq and Anthropic when keys are configured.
- Falls back to deterministic rule-based parsing when LLM output is missing or invalid.
- Clamps all deltas and world values to keep the simulation stable.
- Returns a causal trace containing deltas, confidence, affected sectors, event type, and reasoning.

### `ace_agents.py`

Defines seven differentiated company agents.

Agents have static identity:

- company type
- objective
- exposure to oil, food, gold, and cooperation
- risk tolerance

Agents also have dynamic state:

- resources
- balance sheet
- beliefs over hidden round types
- trust scores
- self memory
- opponent memory
- strategy success counters
- Q-values

`choose_fallback_action()` is the deterministic safety policy. It scores candidate actions using expected profit, risk, trust alignment, historical success, Q-values, identity bias, and opponent modeling.

`update_after_round()` updates resources, balance sheet, memory, trust/opponent summaries, strategy counters, and Q-values after each round.

### `ace_reward.py`

Computes dense, decomposed rewards.

Reward components are intentionally separated:

- inference reward: did the agent predict the hidden round type?
- action reward: was the chosen action suitable for the actual round?
- format reward: did the output obey the expected structured format?
- personality reward: did the action align with the agent identity?
- behavior reward: did the action reflect learned historical success?
- market/social/stability additions are applied inside `ACEWorldEnv`.

This avoids reward hacking where a model can increase total reward by exploiting one shortcut, such as always using one bid amount.

### `ace_llm_policy.py`

Shared production policy for LLM decisions. This is used by the demo and notebook.

The policy contract is:

```text
build_action_prompt(env, agent)
-> generate_action(prompt)
-> extract_first_valid_json(raw_text)
-> normalize_action(parsed, fallback)
-> always return a valid action
```

Key reliability behavior:

- The prompt always ends with `JSON:`.
- The model is instructed to return only JSON.
- Generation uses low temperature.
- Prompt echo is removed for local model generation.
- JSON is extracted with stack-based brace balancing.
- Invalid or missing JSON immediately falls back to `choose_fallback_action()`.
- Action names, round labels, parameters, and amounts are normalized before reaching the environment.

## Demo Flow

### 1. User selects or types an event

In `demo_gradio.py`, the user can choose a preset event or enter free text, such as:

```text
Oil crisis disrupts shipping and raises energy costs
```

### 2. Event becomes world deltas

`inject_event()` calls:

```text
ACEWorldEnv.apply_event()
-> WorldState.apply_event()
-> parse_event_payload()
```

The parser returns structured deltas such as higher oil price, higher energy cost, higher inflation, lower supply chain stability, or higher geopolitical risk. The world clamps values and runs feedback dynamics.

### 3. UI explains why the world changed

The Gradio app renders:

- world gauges
- causal impact text
- probability bars for hidden round types
- economic flow visualization
- affected sector information

This makes the event-to-economy translation visible rather than a black box.

### 4. Agents choose actions

For each agent:

- If provider is `fallback`, the agent uses `choose_fallback_action()`.
- If provider is `groq` or `anthropic`, `llm_or_fallback_decision()` calls the shared `llm_policy()`.
- If the provider errors, returns invalid JSON, or omits required fields, the fallback action is used.

The structured action schema is:

```json
{
  "predicted_round": "competitive/cooperative/resource",
  "action": "challenge/propose_alliance/accept_alliance/betray/allocate_resources/execute_contract/submit_bid",
  "parameters": {},
  "reasoning": "short"
}
```

### 5. Environment resolves the round

`ACEWorldEnv.step()`:

- advances endogenous dynamics
- samples the hidden ground-truth round type
- gives each agent a noisy observation
- applies social side effects such as alliances, betrayal, trust shifts, and challenges
- computes decomposed rewards
- updates agent memory and Q-values
- records round history and world history

### 6. UI shows adaptation

The demo renders:

- agent cards with role, action, reasoning, beliefs, trust, Q-values, and resources
- interaction log for alliances, challenges, and betrayal
- behavior evolution panel
- action-vs-optimal comparison
- resource and world-history plots

## LLM Reliability Design

LLM output is never trusted directly.

Failure modes handled:

- prompt echo
- multiple JSON objects
- markdown fences
- natural language before or after JSON
- truncated JSON
- invalid actions
- missing parameters
- provider exceptions
- missing API keys

The critical action parser is stack-based:

```python
def extract_first_valid_json(text):
    stack = []
    start = None

    for i, ch in enumerate(text or ""):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        continue

    return None
```

If this returns `None`, the simulation does not fail. It uses the agent fallback action.

## GRPO Notebook Flow

`ACE_Experimental_Narrative.ipynb` demonstrates the training story.

### Dataset collection

For each step:

```text
create base_env
build prompt from base_env and agent
sample k completions
for each completion:
    deep-copy base_env
    evaluate completion independently
compute baseline mean reward
advantages = reward - baseline
store datapoint
```

Each datapoint has the required format:

```python
{
    "prompt": str,
    "completions": [str, str, str],
    "advantages": [float, float, float],
}
```

The important invariant is that every completion is evaluated on the same environment state. No completion mutates the shared base environment.

### Training

The notebook configures `GRPOTrainer` for:

- one epoch
- small dataset
- short completions
- low-cost model by default
- JSON-aware reward function
- defensive dependency handling

If optional ML dependencies are unavailable, the notebook still runs the parser, dataset, rollout, and evaluation sections so the demo narrative remains usable.

### Evaluation

The notebook reports:

- reward before vs. after preference selection
- action consistency
- rewards over episodes
- cooperation over time
- betrayal over time
- action distribution
- trust over time
- qualitative case studies for oil crisis, peace agreement, and repeated interaction

## Data and State Safety

The system is designed around bounded state.

- World variables are clamped to hard ranges.
- Event deltas are capped.
- Agent resource updates are bounded at zero.
- Trust scores are clamped between zero and one.
- LLM actions are normalized before environment execution.
- The demo can run entirely without API keys.

## Deployment Model

For Hugging Face Spaces:

```text
Space starts app.py
app.py imports demo_gradio.build_ui()
demo_gradio creates ACEWorldEnv state
user drives event injection and rounds
optional Groq/Anthropic secrets enable live LLM decisions
fallback logic keeps the demo alive without secrets
```

For OpenEnv:

```text
openenv.yaml
-> openenv_ace:ACEOpenMultiAgentEnv
-> ACEWorldEnv
```

The OpenEnv adapter exposes reset, step, and state while reusing the same underlying simulator.

## Design Tradeoffs

- Lightweight adaptation is preferred over expensive full RL in the live demo.
- Fallback policy is intentionally strong so the demo remains coherent without LLM calls.
- GRPO is shown as a compact training narrative, not as a large-scale training pipeline.
- Dense rewards are decomposed for interpretability, even if that is less elegant than one learned reward model.
- The UI prioritizes visible causality and judge comprehension over minimalism.

## Success Criteria

The system is successful when:

- free-text events visibly change the economy
- agent actions differ by identity and incentives
- repeated rounds change memory, trust, Q-values, and behavior
- JSON parsing never crashes the run
- the notebook builds a non-empty GRPO dataset
- the demo works with or without live LLM providers

