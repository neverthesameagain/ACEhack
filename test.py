from env import ACEEnv
import json

env = ACEEnv()
obs = env.reset()

for _ in range(5):
    actions = []

    # ✅ Use CURRENT observation (correct now)
    market = obs["market_state"]

    for _ in range(2):
        if market["competition_signal"] == "high":
            pred = "competitive"
        elif market["competition_signal"] == "low":
            pred = "cooperative"
        else:
            pred = "resource"

        action = json.dumps({
            "predicted_round": pred,
            "action": "bid",
            "amount": 60
        })

        actions.append(action)

    # Step AFTER building actions
    obs, rewards, done, info = env.step(actions)

    print("Rewards:", rewards)
    print("Actual round:", info["debug_round_type"])
    print("Observation:", obs)
    print("Inference Accuracy:", info["inference_accuracy"])
    print("------")