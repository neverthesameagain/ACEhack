"""
ACE++ Environment — Fixed & OpenEnv-ready
==========================================
Key fixes vs v1:
  1. Round type is sampled in reset() and at the END of step() (for next round).
     The agent ALWAYS predicts the round whose market_state they already observed.
  2. market_state returned in observation is for the NEXT round (correct POMDP).
  3. current_round_type is stored on self — never re-sampled mid-step.
  4. validate_action returns structured error JSON (matches spec).
  5. Step returns flat scalar reward (not list) for single-agent TRL compatibility.
     Multi-agent wrapper is a separate class below.
"""

import random
import json


ROUND_TYPES = ["cooperative", "competitive", "resource"]


class ACEEnv:
    """
    Single-agent ACE++ environment.

    Episode flow:
        obs = env.reset()           # obs contains market_state for round 0
        obs, reward, done, info = env.step(action_json)
            # action_json must predict round 0's type (what was shown in reset obs)
            # obs now contains market_state for round 1
        ...

    Action format (JSON string):
        {
          "predicted_round": "cooperative" | "competitive" | "resource",
          "action": "bid" | "allocate" | "solo",
          "amount": float          # required if action == "bid"
        }
    """

    def __init__(self, num_rounds=5, inference_weight=1.2):
        self.num_rounds = num_rounds
        self.inference_weight = inference_weight  # w in R_total formula

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self):
        self.current_round = 0
        self.total_reward = 0.0
        self.correct_inferences = 0
        self.history = []

        # Sample round 0's type NOW — agent will see its market signal
        self.current_round_type = self._sample_round_type()
        market_state = self._query_market_state(self.current_round_type)

        return {
            "round": self.current_round,
            "market_state": market_state,   # signal for CURRENT round
            "history": [],
        }

    def step(self, action: str):
        """
        action: JSON string from the agent.
        Returns: (observation, reward, done, info)

        Scores the agent's prediction against self.current_round_type
        (the round whose market_state was shown in the previous observation).
        Then advances to the next round.
        """
        # ---- Validate ----
        parsed, error = self._validate_action(action)
        if error:
            # Advance round even on error so episodes don't stall
            self.current_round += 1
            done = self.current_round >= self.num_rounds
            next_round_type = self._advance_round()
            obs = self._make_observation(next_round_type)
            return obs, -1.0, done, {"error": error, "debug_round_type": self.current_round_type}

        # ---- Score against the CURRENT round (already shown to agent) ----
        scored_round_type = self.current_round_type   # what agent saw signal for
        pred = parsed["predicted_round"]
        act  = parsed.get("action", "solo")
        amount = float(parsed.get("amount", 50))

        # Inference reward
        if pred == scored_round_type:
            r_inference = 1.0
            self.correct_inferences += 1
        else:
            r_inference = -0.5

        # Task reward
        r_task = self._task_reward(act, amount, scored_round_type)

        total_reward = r_task + self.inference_weight * r_inference
        self.total_reward += total_reward

        # ---- Log ----
        step_log = {
            "round": self.current_round,
            "actual_round_type": scored_round_type,
            "predicted_round": pred,
            "correct": pred == scored_round_type,
            "action": act,
            "amount": amount,
            "r_task": r_task,
            "r_inference": r_inference,
            "r_total": total_reward,
        }
        self.history.append(step_log)

        # ---- Advance to next round ----
        self.current_round += 1
        done = self.current_round >= self.num_rounds

        # Sample next round's type, generate its market signal
        next_round_type = self._advance_round()
        obs = self._make_observation(next_round_type)

        info = {
            "debug_round_type": scored_round_type,          # what was just played
            "next_round_type": next_round_type,             # what signal encodes
            "correct_inference": pred == scored_round_type,
            "inference_accuracy": self.correct_inferences / self.current_round,
            "step_log": step_log,
        }

        return obs, total_reward, done, info

    def state(self):
        """Full internal state — for OpenEnv / God Mode panel."""
        return {
            "current_round": self.current_round,
            "current_round_type": self.current_round_type,
            "total_reward": self.total_reward,
            "inference_accuracy": (
                self.correct_inferences / self.current_round
                if self.current_round > 0 else 0.0
            ),
            "history": self.history,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_round(self):
        """Sample and store the type for the round that just started."""
        if self.current_round < self.num_rounds:
            self.current_round_type = self._sample_round_type()
        return self.current_round_type

    def _make_observation(self, round_type):
        return {
            "round": self.current_round,
            "market_state": self._query_market_state(round_type),
            "history": self.history[-3:],
        }

    def _sample_round_type(self):
        return random.choice(ROUND_TYPES)

    def _query_market_state(self, round_type):
        """
        Mock Market Query API.
        Returns JSON that CORRELATES with round_type but doesn't reveal it directly.
        Agents must learn the mapping.
        """
        if round_type == "competitive":
            return {
                "demand_index":       round(random.uniform(0.80, 1.00), 2),
                "volatility":         round(random.uniform(0.70, 1.00), 2),
                "competition_signal": "high",
                "cooperation_signal": "low",
            }
        elif round_type == "cooperative":
            return {
                "demand_index":       round(random.uniform(0.20, 0.45), 2),
                "volatility":         round(random.uniform(0.10, 0.35), 2),
                "competition_signal": "low",
                "cooperation_signal": "high",
            }
        else:  # resource
            return {
                "demand_index":       round(random.uniform(0.45, 0.60), 2),
                "volatility":         round(random.uniform(0.40, 0.60), 2),
                "competition_signal": "medium",
                "cooperation_signal": "medium",
            }

    def _task_reward(self, action, amount, round_type):
        """
        Task reward based on action × round_type match.
        Optimal strategy differs per round — agent must infer round to act well.
        """
        if action == "bid":
            if round_type == "competitive":
                return 2.0 if amount > 60 else -1.0
            elif round_type == "cooperative":
                return 2.0 if amount < 40 else -1.0
            else:  # resource
                return 2.0 if 35 <= amount <= 65 else -1.0
        elif action == "allocate":
            return 1.0 if round_type == "resource" else 0.0
        elif action == "solo":
            return 0.5  # safe but low reward
        return 0.0

    def _validate_action(self, action: str):
        """
        Returns (parsed_dict, None) on success.
        Returns (None, error_dict) on failure.
        """
        try:
            parsed = json.loads(action)
        except json.JSONDecodeError as e:
            return None, {
                "status": "error",
                "error_type": "JSON_PARSE_ERROR",
                "message": str(e),
                "expected_format": {
                    "predicted_round": "cooperative | competitive | resource",
                    "action": "bid | allocate | solo",
                    "amount": "float (required for bid)",
                },
            }

        missing = [k for k in ("predicted_round", "action") if k not in parsed]
        if missing:
            return None, {
                "status": "error",
                "error_type": "MISSING_KEYS",
                "message": f"Missing required keys: {missing}",
            }

        if parsed["predicted_round"] not in ROUND_TYPES:
            return None, {
                "status": "error",
                "error_type": "INVALID_ROUND_TYPE",
                "message": f"predicted_round must be one of {ROUND_TYPES}",
            }

        return parsed, None


# ------------------------------------------------------------------
# Quick smoke test — run this file directly to verify
# ------------------------------------------------------------------
if __name__ == "__main__":
    env = ACEEnv(num_rounds=5)
    obs = env.reset()
    print("=== RESET ===")
    print(json.dumps(obs, indent=2))

    for step_num in range(5):
        market = obs["market_state"]

        # Scripted policy: read the signal, predict, bid accordingly
        sig = market["competition_signal"]
        if sig == "high":
            pred, amount = "competitive", 75.0
        elif sig == "low":
            pred, amount = "cooperative", 30.0
        else:
            pred, amount = "resource", 50.0

        action = json.dumps({
            "predicted_round": pred,
            "action": "bid",
            "amount": amount,
        })

        obs, reward, done, info = env.step(action)

        print(f"\n=== STEP {step_num + 1} ===")
        print(f"  Actual round : {info['debug_round_type']}")
        print(f"  Predicted    : {pred}  ({'✓' if info['correct_inference'] else '✗'})")
        print(f"  Reward       : {reward:.2f}")
        print(f"  Acc so far   : {info['inference_accuracy']:.0%}")

        if done:
            print("\n=== EPISODE DONE ===")
            print(f"  Total reward : {env.state()['total_reward']:.2f}")
            print(f"  Final acc    : {env.state()['inference_accuracy']:.0%}")
            break