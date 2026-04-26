---
title: EIE Economic Intelligence Engine
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.13.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# EIE Economic Intelligence Engine

Learning Decisions in Dynamic Economic and Geopolitical Systems

EIE Economic Intelligence Engine is a live multi-agent economic simulation where natural language events reshape the world, agents reason under uncertainty, and social dynamics drive adaptation.

## Demo Flow

Press **🚀 Run Full Demo** or follow the guided path:

1. Inject `oil crisis hits Middle East`
2. Run one round
3. Run five rounds
4. Watch beliefs, trust, actions, rewards, and resources evolve

## What The Demo Shows

- Text-to-economy translation using the new `ace_text_inject.py` pipeline
- Bounded world updates with causal trace and confidence
- Seven differentiated agents with beliefs, memory, trust, Q-values, and opponent models
- Alliances, challenges, betrayals, retaliation, and cooperation
- Lightweight Q-style adaptation without heavy training loops

## Runtime Files

- `app.py`: Hugging Face Space entrypoint
- `openenv.yaml`: OpenEnv metadata pointing to `openenv_ace:ACEOpenMultiAgentEnv`
- `openenv_ace.py`: OpenEnv-compatible adapter around the current simulator
- `demo_gradio.py`: judge-facing UI
- `ace_text_inject.py`: robust text-to-economy engine
- `ace_world_env.py`: world model and multi-agent environment
- `ace_agents.py`: adaptive company agents
- `ace_reward.py`: decomposed reward function

Older prototypes, notebooks, logs, and training artifacts are preserved in `legacy/`.

## Secrets

The app runs without API keys using deterministic fallback logic. For live LLM decisions, set one of these Space secrets:

- `LLM_PROVIDER=groq` and `GROQ_API_KEY`
- `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`

For the current deployment, the recommended setup is:

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile
```

The model can be LLaMA, but the active client is Groq. The current runtime does **not** use `OPENAI_API_KEY` or `OPENAI_BASE_URL`; those only appear in legacy files preserved under `legacy/`.
