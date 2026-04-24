"""
ACE++ — TRL/GRPO Training Script
==================================
This connects the ACEEnv to a real LLM via HuggingFace TRL + Unsloth.

Run in Colab with:
    !pip install unsloth trl transformers accelerate

Model: Qwen2.5-3B-Instruct (swap for any instruct model)
"""

# ---------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------
import json
import re
import random
from ace_env_fixed import ACEEnv   # your fixed env

# ---------------------------------------------------------------
# 1. Prompt builder — this is what the LLM sees each turn
# ---------------------------------------------------------------

SYSTEM_PROMPT = """You are an economic AI agent competing in a hidden-market environment.

Each round, you observe a market state (JSON). Your job:
1. INFER the hidden round type from the signals.
2. OUTPUT a JSON action.

Round types:
- "competitive": high demand, high volatility, high competition → bid aggressively (amount > 60)
- "cooperative": low demand, low volatility, low competition → bid conservatively (amount < 40)
- "resource": medium signals → allocate resources (amount 35–65)

You MUST respond with ONLY valid JSON. No extra text. Format:
{
  "predicted_round": "<cooperative|competitive|resource>",
  "action": "<bid|allocate|solo>",
  "amount": <float>
}"""


def build_prompt(observation: dict) -> str:
    market = observation["market_state"]
    history = observation.get("history", [])

    history_str = ""
    if history:
        recent = history[-2:]  # last 2 rounds only (keep context short)
        history_str = "\nRecent history:\n"
        for h in recent:
            correct_str = "✓" if h.get("correct") else "✗"
            history_str += (
                f"  Round {h['round']}: predicted={h['predicted_round']} "
                f"actual={h['actual_round_type']} {correct_str} "
                f"reward={h['r_total']:.1f}\n"
            )

    prompt = f"""Current market state:
{json.dumps(market, indent=2)}
{history_str}
Round {observation['round']} of the episode. What is your action?"""
    return prompt


# ---------------------------------------------------------------
# 2. Reward function — this is what GRPO optimizes
# ---------------------------------------------------------------

def ace_reward_function(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    Called by GRPOTrainer for a batch of rollouts.
    Each completion is one LLM response to one prompt.
    We re-run the env step to score it.

    NOTE: We can't pass env state through GRPO directly, so we
    re-parse the market state from the prompt and score locally.
    This is the standard pattern for RLVR with verifiable rewards.
    """
    rewards = []

    for completion, prompt in zip(completions, prompts):
        reward = _score_single_completion(completion, prompt)
        rewards.append(reward)

    return rewards


def _score_single_completion(completion: str, prompt: str) -> float:
    """Parse LLM output and compute ACE++ reward without env state."""
    # Extract JSON from completion (LLMs sometimes wrap in ```json)
    json_match = re.search(r'\{.*?\}', completion, re.DOTALL)
    if not json_match:
        return -1.5  # severe penalty for unparseable output

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return -1.5

    # Validate required fields
    if "predicted_round" not in parsed or "action" not in parsed:
        return -1.0

    # Extract ground truth from prompt context
    # (we embed it as a hidden comment for scoring — see build_prompt_with_truth)
    round_type = kwargs_from_prompt(prompt).get("actual_round_type")
    if round_type is None:
        return 0.0  # can't score without ground truth

    pred = parsed["predicted_round"]
    action = parsed.get("action", "solo")
    amount = float(parsed.get("amount", 50))

    # Inference reward
    if pred == round_type:
        r_inference = 1.0
    else:
        r_inference = -0.5

    # Task reward
    r_task = _task_reward(action, amount, round_type)

    return r_task + 1.2 * r_inference


def kwargs_from_prompt(prompt: str) -> dict:
    """Extract hidden ground truth embedded in prompt."""
    match = re.search(r'GROUND_TRUTH:(\w+)', prompt)
    if match:
        return {"actual_round_type": match.group(1)}
    return {}


def _task_reward(action, amount, round_type):
    if action == "bid":
        if round_type == "competitive":
            return 2.0 if amount > 60 else -1.0
        elif round_type == "cooperative":
            return 2.0 if amount < 40 else -1.0
        else:
            return 2.0 if 35 <= amount <= 65 else -1.0
    elif action == "allocate":
        return 1.0 if round_type == "resource" else 0.0
    return 0.5


# ---------------------------------------------------------------
# 3. Dataset builder — generate (prompt, ground_truth) pairs
# ---------------------------------------------------------------

from datasets import Dataset

def generate_ace_dataset(n_samples: int = 500) -> Dataset:
    """
    Generate training examples by rolling out the env with a random policy.
    Each example = one step = (prompt, actual_round_type).
    The model will learn to predict better via GRPO.
    """
    env = ACEEnv(num_rounds=5)
    records = []

    while len(records) < n_samples:
        obs = env.reset()

        for _ in range(env.num_rounds):
            actual_round_type = env.current_round_type  # ground truth

            # Build prompt — embed ground truth as hidden comment for scorer
            prompt = build_prompt(obs) + f"\n<!-- GROUND_TRUTH:{actual_round_type} -->"

            records.append({
                "prompt": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                          f"<|im_start|>user\n{prompt}<|im_end|>\n"
                          f"<|im_start|>assistant\n",
                "actual_round_type": actual_round_type,
            })

            # Random action to advance env
            random_action = json.dumps({
                "predicted_round": random.choice(["competitive", "cooperative", "resource"]),
                "action": "bid",
                "amount": random.uniform(20, 80),
            })
            obs, _, done, _ = env.step(random_action)
            if done:
                break

    return Dataset.from_list(records[:n_samples])


# ---------------------------------------------------------------
# 4. GRPO Training — paste this into Colab
# ---------------------------------------------------------------

TRAINING_SCRIPT = '''
# ---- Install ----
# !pip install unsloth trl transformers accelerate datasets -q

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from ace_training import generate_ace_dataset, ace_reward_function, SYSTEM_PROMPT

# ---- Load model ----
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=1024,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ---- Dataset ----
dataset = generate_ace_dataset(n_samples=500)

# ---- GRPO Config ----
training_args = GRPOConfig(
    output_dir="ace_grpo_out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    num_generations=4,       # GRPO rollouts per prompt
    max_new_tokens=128,
    temperature=0.8,
    report_to="none",        # swap to "wandb" if you want W&B
)

# ---- Trainer ----
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[ace_reward_function],
    tokenizer=tokenizer,
)

trainer.train()

# ---- Save ----
model.save_pretrained("ace_model_final")
tokenizer.save_pretrained("ace_model_final")
print("Training done. Model saved.")
'''


# ---------------------------------------------------------------
# 5. Evaluation — before vs after comparison
# ---------------------------------------------------------------

def evaluate_model(model, tokenizer, n_episodes=20):
    """
    Run the trained model through n_episodes and report metrics.
    Compare against random baseline.
    """
    from transformers import pipeline
    import numpy as np

    env = ACEEnv(num_rounds=5)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    max_new_tokens=128, temperature=0.1)

    inference_accuracies = []
    total_rewards = []

    for _ in range(n_episodes):
        obs = env.reset()

        for _ in range(env.num_rounds):
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{build_prompt(obs)}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            output = pipe(prompt)[0]["generated_text"]
            # Strip prompt from output
            completion = output[len(prompt):]

            obs, _, done, info = env.step(completion)
            if done:
                break

        state = env.state()
        inference_accuracies.append(state["inference_accuracy"])
        total_rewards.append(state["total_reward"])

    print("=== Evaluation Results ===")
    print(f"Inference accuracy : {np.mean(inference_accuracies):.1%} ± {np.std(inference_accuracies):.1%}")
    print(f"Total reward/ep    : {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Random baseline acc: ~33%  (3-class random)")
    return inference_accuracies, total_rewards


if __name__ == "__main__":
    # Verify dataset generation works
    print("Generating 20-sample dataset...")
    ds = generate_ace_dataset(n_samples=20)
    print(f"Dataset size: {len(ds)}")
    print("Sample prompt (truncated):")
    print(ds[0]["prompt"][:400])
    print("\nDataset looks good. Run the GRPO training block in Colab.")