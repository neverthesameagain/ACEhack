# EIE: Teaching AI Agents to Read the World Before They Act

Most economic simulations start with numbers.

Oil price: 1.2. Inflation: 0.08. Volatility: 0.5.

Clean. Structured. Comforting. Easy to put on a chart.

It is also how people pretend the world is orderly, right up until it is not.

A few sliders move, a plot updates, and somewhere deep in the system a policy decides what to do next, confidently, as if the rest of the world agreed to be predictable.

**The real world is less polite. It usually starts with chaos, then slowly turns that chaos into numbers you can read later.**

The real world starts with a headline.

An oil crisis hits the Middle East. A food supply chain breaks. A cooperation agreement gets signed. A market shock ripples across sectors before anyone has clean data. Companies, regulators, insurers, funds, and logistics operators all see the same event differently, and they act with partial information, incentives, memory, fear, trust, and habit.

That is the world we wanted to build with **EIE: Economic Intelligence Engine**.

EIE is a live multi-agent economic simulation where a natural-language event becomes a changing economy, and a group of strategic agents must decide how to respond. It is not trying to forecast the real economy. It is trying to make something more useful for a demo, a benchmark, and a learning environment:

**a clear story of how events change incentives, how incentives change behavior, and how repeated behavior becomes strategy.**

EIE is not a crystal ball. It is closer to a pressure chamber. If you want a one-line pitch, this is the honest one:

**EIE does not try to predict the economy. It tries to simulate how intelligent agents survive inside it.**

## If you are skimming, read this

- **Type a headline, not a vector.** A messy real-world event becomes a structured economy.
- **Seven agents, seven incentives, one messy system.** This is not “one model speaks for the market.”
- **Two learning modes, on purpose:** environment adaptation (Agent-Based RL) is not the same thing as model-strategy search (LLM-Based RL).
- **Trust and Q-values are not decoration.** The demo shows the chain from uncertainty to action to update.
- **The demo is deployable, not a local fantasy.** OpenEnv metadata plus Hugging Face Spaces, with fallback that keeps the show running.

## The Problem

Modern AI systems are very good at talking about the world, but decision-making is harder than explanation.

If you ask a model, "What happens if oil supply is disrupted?", it can write a solid paragraph. But what should a shipping company do? What should a central bank do? What should a hedge fund do? What if those agents have different incentives? What if they remember past betrayals? What if their trust changes after every round?

That is the gap EIE explores.

We wanted a system where:

- A user can type a messy real-world event.
- The event changes a structured economic world.
- Agents only see partial, noisy observations.
- Each agent acts according to its role, incentives, beliefs, memory, trust, and learned Q-values.
- The UI explains the causal chain instead of hiding it.
- The system still works even if live LLM calls fail.

So the project became less like a static dashboard and more like a small playable world.

## The Core Flow

The demo is built around one simple narrative:

```text
Event Reported
-> LLM Interprets
-> World Change
-> Agents Act
-> Outcome
```

In the deployed UI, this appears as a story strip:

```text
Event -> Shift -> Agents -> Outcome
```

For example, if the user runs:

```text
oil crisis hits Middle East
```

EIE translates that into economic deltas:

- oil prices rise
- energy costs rise
- inflation rises
- geopolitical risk rises
- market volatility rises
- cooperation falls
- supply-chain stability weakens

Then the world shifts into something like a stagflationary stress regime. Agents observe that world, form beliefs about whether the hidden round is competitive, cooperative, or resource-driven, and choose actions.

The important part is that the UI does not just say "round complete." It shows the actual chain:

```text
oil crisis hits Middle East
-> oil_price up + geopolitical_risk up + energy_cost up
-> market stress rises
-> agents respond with alliances, bids, contracts, or challenges
-> rewards, trust, and Q-values update
```

That is the whole product philosophy: make the reasoning visible.

A lot of demos will impress you *once*. EIE tries to be interesting *twice*: first when the world shifts, and again when you realize you can follow the full causal chain like a storyboard.

## The Seven Agents

EIE has seven agents, each with a different role in the economy:

- **PetroCorp**: an energy company exposed to oil and energy shocks.
- **GlobalFoods Inc**: a food importer and distributor vulnerable to supply-chain and food-price pressure.
- **Aurelius Capital**: a hedge fund that is more aggressive and opportunity-seeking.
- **CentralBank of EIE**: a regulator trying to maintain stability and reduce systemic risk.
- **LogiChain Global**: a logistics operator watching transport and supply-chain stress.
- **ShieldRe Insurance**: a risk underwriter that cares about volatility and downside exposure.
- **NovaTech Systems**: a technology infrastructure firm balancing capital, stability, and cooperation.

Each agent tracks more than a single action score. It has:

- resources
- beliefs about the hidden round type
- trust scores toward other agents
- short-term memory
- strategy success history
- opponent models
- Q-values for learned action preferences

This makes the agents feel less like independent buttons and more like characters in a strategic system. PetroCorp may benefit from an oil shock, while GlobalFoods may suffer. Aurelius Capital may bid aggressively when volatility rises. CentralBank of EIE may prefer alliances and stability even when the round pressure is competitive.

Sometimes agents are right. Sometimes they are wrong. That is intentional. A good simulation should show uncertainty, not pretend everyone sees the world perfectly.

## Two Modes, Two Stories

One of the biggest design decisions was separating **Agent-Based RL** from **LLM-Based RL**.

They sound similar, but they prove different things.

### Agent-Based RL

This is the default mode. It does not require an API key.

Agents act using the simulator's internal policy: incentives, Q-values, trust, memory, opponent modeling, and role-specific preferences. After each round, they update based on rewards and interaction outcomes.

This mode proves that the environment itself supports learning and adaptation. The agents are not just asking a model what sounds good. They are updating internal state over repeated interaction.

### LLM-Based RL

This mode uses Groq and the selected model, usually:

```text
llama-3.3-70b-versatile
```

Instead of taking one model output directly, EIE samples multiple candidate strategies for each agent. It repairs and extracts JSON, normalizes the action, evaluates each candidate on copied environments, computes relative advantage, and selects the highest-reward strategy.

That makes the LLM policy feel closer to a training loop:

```text
sample actions
-> score actions
-> compare rewards
-> choose the best strategy
```

This is why the UI has an **AI Reasoning** panel. In Agent-Based RL, it explains that the internal learned policy was used. In LLM-Based RL, it can show sampled actions, rewards, advantages, and the selected best action.

We deliberately did not blend these modes into one vague "AI mode." The separation makes the story cleaner:

- Agent-Based RL shows environment-level adaptation.
- LLM-Based RL shows model-policy optimization.

**No mushy "AI do everything" button.** The demo names what is learning, and which learning is which.

## Why Natural Language Matters

The text event system is one of the most important parts of EIE.

Most toy economic simulations make the user set sliders. That is fine for debugging, but not very natural for a demo. People think in events. Judges think in scenarios. Markets react to headlines.

So EIE lets the user type something like:

```text
major food supply chain disruption
```

or:

```text
global cooperation agreement signed
```

The event parser turns that into bounded structured deltas. It uses a strict JSON schema and clamps values so the world cannot explode because of one weird model response. If Groq is configured, the LLM can parse the event. If not, deterministic fallback rules keep the demo alive.

Reliability matters here. A live demo should never collapse because a model returned markdown instead of JSON.

## The UI Is a Story, Not a Spreadsheet

The first version of a system like this naturally wants to become a giant table.

But a judge does not want to hunt through fifty numbers to understand what happened.

So the Gradio UI was designed around a few high-signal pieces:

- a compact event command bar
- a mode toggle for Agent-Based RL vs LLM-Based RL
- fallback/Groq provider selection
- a model textbox
- a story strip
- 2x2 world-state cards
- horizontal agent cards
- AI reasoning
- a system console for deeper diagnostics
- training proof below the main console

The top of the UI answers the big questions immediately:

- What event happened?
- How did the world shift?
- What are the agents doing?
- What was the outcome?

The detailed console is still there, but it is not the first thing the user has to understand. If someone wants the raw trust matrix, causal trace, world variables, reward decomposition, or action-vs-optimal comparison, it is available.

The UI is meant to feel like watching an economic incident unfold, not filling out a form.

## Training Proof

The notebook and UI both include a training story because "agents learn" should not just be a claim.

The Phase 1 proof trains agents across scenarios, then compares:

- random baseline
- untrained fallback policy
- trained agents

It reports reward lift, accuracy lift, cooperation changes, aggression changes, trust changes, action distribution shifts, and learned Q-value evidence.

This is useful because it gives the project a measurable backbone. The demo can be fun and visual, but the training proof shows that repeated interaction changes behavior.

One of the most interesting parts is that the results are not always perfectly flattering. For example, a trained policy can improve reward in an oil crisis while not improving every peace-scenario metric. That is a good thing. It makes the system feel honest. Learning is contextual, and not every objective improves at the same time.

## OpenEnv and Deployment

EIE is not only a Gradio demo. It also has an OpenEnv-compatible entrypoint:

```text
openenv.yaml
-> openenv_ace:ACEOpenMultiAgentEnv
```

The adapter wraps the same `ACEWorldEnv` used by the UI. That means evaluation and demo behavior come from the same simulator rather than two separate implementations.

For Hugging Face Spaces, the entrypoint is simple:

```text
app.py
-> demo_gradio.build_ui()
-> demo.launch(css=APP_CSS)
```

The app can run in fallback mode with no secrets. For live LLM-Based RL, the deployment uses Groq environment variables:

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Those go into `.env` locally or Hugging Face Space repository secrets in production. There is no API-key textbox in the UI, which keeps the demo cleaner and avoids awkward secret handling during judging.

## What Makes EIE Different

EIE is not just "LLM generates agent actions."

The interesting part is the whole loop:

```text
headline
-> structured world update
-> partial observations
-> agent beliefs
-> strategic actions
-> social effects
-> rewards
-> trust updates
-> Q-value updates
-> behavior shift
```

It combines a few ideas that are usually shown separately:

- language-driven event injection
- multi-agent economic simulation
- partially observable state
- trust and opponent modeling
- decomposed rewards
- lightweight reinforcement learning
- LLM policy sampling
- notebook-aligned training evidence
- deployable Gradio storytelling UI
- OpenEnv-compatible wrapper

The goal is not to build a perfect economy. The goal is to build a convincing sandbox where economic events become strategic pressure, and agents visibly adapt.

If you are wondering why a hackathon should care, here is the blunt answer: **EIE makes intelligence legible in a system where no one gets perfect information, which is the most realistic part of the whole project.**

## The Demo Moment

The best way to understand EIE is to run one event.

Type:

```text
oil crisis hits Middle East
```

Then press Run.

The world shifts. Market stress rises. Cooperation weakens. Agents choose alliances, bids, contracts, or challenges. The system reveals the hidden round type. Some agents get rewarded. Some miss the optimal action. Trust changes. Resources move. The next round starts from a different world than the first.

That is the core magic.

Not a chatbot answering a question.

Not a static dashboard.

**Not a model confidently explaining the world to an empty room.**

A small economy that reads the news, changes its state, and watches its agents learn what kind of world they are living in.

A system where the economy reacts, agents adapt, and intelligence is something you can point at, not a vibe.

That is EIE.
