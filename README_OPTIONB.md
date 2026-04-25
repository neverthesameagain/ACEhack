# ACE++ Option B: Text-Injection Adaptive Coalition Economy

ACE++ Option B is a demo-first multi-agent economic simulation. A judge types a real-world event, the system converts that event into structured economic deltas, and four company agents react according to their incentives, trust, memory, and learned habits.

## Core Pipeline

```text
Text event
  -> structured world-state deltas
  -> clamped economic state update
  -> changed hidden round probabilities
  -> agent-specific decisions
  -> decoupled reward + memory updates
  -> visible adaptation over rounds
```

## What Makes It Different

- `WorldState` tracks commodities, macro indicators, volatility, cooperation, scarcity, event history, and causal traces.
- `ace_text_inject.py` turns events like "oil crisis" into validated deltas with fallback rules when no API key is present.
- `ace_agents.py` defines four distinct company archetypes with different exposures and risk profiles.
- Agents maintain memory, trust, opponent models, and strategy success counters.
- `ace_reward.py` separates inference reward from action reward, preventing bid-size reward hacking.
- `demo_gradio.py` is the judge-facing interactive demo.

## Agent Archetypes

- PetroCorp: energy company, benefits from oil spikes, aggressive and competitive.
- GlobalFoods Inc: food importer, hurt by oil/food inflation, prefers cooperation.
- Aurelius Capital: hedge fund, profits from volatility, opportunistic and high risk.
- CentralBank of ACE: regulator, stabilizes markets and prefers cooperation.

## Run The Demo

```bash
pip install -r requirements_demo.txt
ANTHROPIC_API_KEY=your_key python demo_gradio.py
```

The demo also works without `ANTHROPIC_API_KEY`; it uses deterministic adaptive fallback agents so judging never blocks on API availability.

## Suggested Demo Script

1. Type: `oil crisis hits Middle East`
2. Click `Inject Event`
3. Show oil, energy, volatility, and trade tension increasing.
4. Click `Run Round`
5. Show PetroCorp becoming aggressive, GlobalFoods becoming defensive, Aurelius exploiting volatility, and CentralBank trying to stabilize.
6. Reveal God Mode: actual hidden round, correct agents, rewards, and alliances.
7. Click `Run 5 Rounds`
8. Show trust, resources, and memories changing.

## Example Events

- `OPEC cuts production by 20%`
- `G7 signs major climate cooperation pact`
- `Central bank raises rates 75 basis points`
- `Russia-Ukraine peace deal announced`
- `Tech sector crash wipes out 30% of equity markets`
- `Global food supply disruption from drought`
- `Trade war escalates with new tariffs`

## Training

Use `training_v2.ipynb` as the Option B training scaffold. It imports:

- `WorldState`
- `AGENT_PROFILES`
- `compute_total_reward`
- existing OpenEnv-compatible wrappers

The key training principle is decoupled reward:

```text
total = inference + action + format + personality + behavior
```

Inference is logged separately from action reward so reward hacking is visible immediately.

## Test Checklist

- `WorldState.derive_round_probabilities()` changes under extreme values.
- `parse_event_payload("oil crisis")` returns oil/energy/volatility deltas.
- `compute_total_reward()` rewards correct inference independently of bid amount.
- `demo_gradio.py` runs without an API key.
- Repeated rounds change resources, trust, and agent memories.
