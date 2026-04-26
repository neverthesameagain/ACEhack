# EIE: Economic INtelligence Engine

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

![Event Input UI](https://raw.githubusercontent.com/neverthesameagain/ACEhack/dcf5423/Screenshot%202026-04-26%20at%203.19.43%E2%80%AFPM.png)

*You start with a simple headline. The system handles everything else — from interpretation to simulation.*

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

![Causal Trace](https://raw.githubusercontent.com/neverthesameagain/ACEhack/dcf5423/Screenshot%202026-04-26%20at%203.17.54%E2%80%AFPM.png)

*Each event is translated into structured economic signals — with explicit reasoning, affected sectors, and confidence.*

A lot of demos will impress you *once*. EIE tries to be interesting *twice*: first when the world shifts, and again when you realize you can follow the full causal chain like a storyboard.

## "Under the Hood: How the System Actually Works"

If the headline is the opening scene, the engine is what makes the plot tighten.

EIE does not “randomize the economy” for drama. It **turns messy language into bounded, structured pressure**—then lets that pressure move through agents who are stubborn, forgetful, trusting, and hungry for payoff. The intelligence is not one oracle. It is the loop: **what the world becomes, what each actor thinks it is, and what they dare to do next.**

### World modeling

The visible economy is a small set of macro-style levers you can read at a glance: **GDP-like growth, inflation, volatility, sector stress**, and the bruises that show up when supply chains, energy, or cooperation start to crack. Under that surface sit quieter forces—**competition, cooperation, and resource limits**—that decide whether the round feels like a knife fight, a treaty table, or a scramble for scarce bandwidth.

Nothing updates as pure noise. Event parsing and world rules **clamp and shape** deltas so the simulation stays interpretable: a wild sentence might still move the needle, but it cannot silently teleport the system into nonsense. The causal spine is intentionally simple to follow:

```text
Event → Signals → Economic shifts → World state
```

A headline becomes **signals** (what kind of shock this is), those signals become **shifts** (who gets squeezed, who gets opportunity), and the **world state** is the shared stage everyone has to play on—whether they like the script or not.

### Agent design

Each agent is a small mind with four stubborn habits:

- **Beliefs** — a *partial* picture of what kind of round this is, built from what they can observe. Nobody gets the whole board.
- **Memory** — what happened last time, and the time before that. History becomes bias, and bias becomes behavior.
- **Trust** — a living score toward other actors. Trust is not flavor text; it is a channel that opens or closes cooperation.
- **Incentives** — the role’s real objective function: stability, alpha, survival, reputation, downside protection—pick your poison.

Those four pieces are why the cast feels less like seven sliders and more like seven agendas colliding.

### Action space

Agents do not “vote on a mood.” They take **social-economic moves** the environment understands: **cooperate, compete, form alliances, challenge rivals, and allocate resources** (who gets funded, protected, starved, or prioritized). The action set is how private incentives become public consequences.

### Interactions

Actions are not private daydreams. **What you do lands on someone else’s ledger.** A bid can feel like a flex; an alliance can feel like a life raft; a challenge can feel like a door kicked open. Trust **creeps up or curdles** based on what actually happened—not what anyone claimed would happen.

That is where the theater comes from: **retaliation and cooperation are not scripted outcomes; they are emergent habits** once incentives, memory, and trust start feeding each other.

### Learning

No equations—just the story math everyone already understands from games and markets:

- Agents try actions.
- The world hands back **rewards** (and social side effects).
- Strategies **update**: what looked clever stops working; what looked cowardly starts paying.

Over time, the cast **gets sharper at surviving the regime it keeps creating**—which is different from “being right about everything.”



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

![Agent Behavior](https://raw.githubusercontent.com/neverthesameagain/ACEhack/dcf5423/Screenshot%202026-04-26%20at%203.13.04%E2%80%AFPM.png)

*Agents don't just act — they respond to incentives, form alliances, and adapt based on trust and past outcomes.*

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

### Agent-Based RL vs LLM-Based RL (two kinds of clever)

Same world. Same agents. Two different ways the system *chooses* the next move.

**Agent-Based RL** is **slow muscle**. Agents learn from **past rewards**, carry **Q-values** and habits forward, and balance **exploration vs exploitation** the classic way: sometimes try a weird move, usually lean on what has paid before. Adaptation is **gradual**—round after round, the policy thickens.

**LLM-Based RL** is **fast imagination**. The model **generates multiple candidate strategies**, the simulator **scores them with the same reward machinery**, and the system **picks the best action now**. It is less “I have slowly learned to love this tactic” and more “here are five futures; this one wins the scoreboard today.”

| | **Agent-Based RL** | **LLM-Based RL** |
| --- | --- | --- |
| **What learns** | The agent’s policy over time | The selection process over candidates |
| **Adaptation** | Gradual, history-shaped | Immediate, compare-and-choose |
| **Exploration** | Built into the RL policy | Implicit in sampling many strategies |
| **Best mental image** | Reputation earned in repeated rounds | A bracket of plans, scored like trials |

**The reveal, in one line:** EIE is not pretending the world is perfectly predictable—it is showing you **how structured pressure becomes state, and how state becomes strategy**, whether that strategy is earned slowly or selected sharply.

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


## OpenEnv and Deployment

EIE is not only a Gradio demo. It has an OpenEnv-compatible entrypoint:

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

![Round Summary](https://raw.githubusercontent.com/neverthesameagain/ACEhack/dcf5423/Screenshot%202026-04-26%20at%203.15.21%E2%80%AFPM.png)

*After each round, the system summarizes what actually happened — including hidden dynamics and whether agents acted optimally.*

That is the core magic.

Not a chatbot answering a question.

Not a static dashboard.

**Not a model confidently explaining the world to an empty room.**

A small economy that reads the news, changes its state, and watches its agents learn what kind of world they are living in.

A system where the economy reacts, agents adapt, and intelligence is something you can point at, not a vibe.

That is EIE.
