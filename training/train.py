import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from collections import defaultdict

from environment.delivery_env import DeliveryEnv
from agents.q_agent import QLearningAgent
from agents.baselines import RandomBaseline, RoundRobinBaseline, GreedyNearestBaseline


# -----------------------------------------------------------------------
# Single episode runner
# -----------------------------------------------------------------------

def run_episode_agent(env: DeliveryEnv, agent: QLearningAgent, train=True):
    """Run one episode with the Q-learning agent. Returns metrics dict."""
    state = env.reset()
    state_key = env.encode_state()
    total_reward = 0.0
    transitions = []
    info = {}

    while not env.done:
        if env.pending_orders:
            action = agent.act(state_key, env)
            next_state, reward, done, info = env.step(env.decode_action(action))
            next_state_key = env.encode_state()

            if train:
                transitions.append((
                    state_key, action, reward, next_state_key,
                    done, env.valid_actions()
                ))

            state_key = next_state_key
            total_reward = info.get("final_reward", total_reward)
        else:
            env._run_until_decision()

    # Batch update Q-table
    if train and transitions:
        for (s, a, r, ns, d, va) in transitions:
            agent.update(s, a, r, ns, d, va)
        agent.decay_epsilon()

    return _episode_metrics(env, info, total_reward)


def run_episode_baseline(env: DeliveryEnv, baseline):
    if hasattr(baseline, 'reset'):
        baseline.reset()

    state = env.reset()
    info = {}
    total_reward = 0.0

    while not env.done:
        if env.pending_orders:
            action_idx = baseline.act(state, env)
            state, reward, done, info = env.step(env.decode_action(action_idx))
            total_reward = info.get("final_reward", total_reward)
        else:
            env._run_until_decision()

    return _episode_metrics(env, info, total_reward)


def _episode_metrics(env, info, total_reward):
    delivered = [o for o in env.orders.values() if o.delivered]
    late = [o for o in delivered if o.is_late]
    on_time_pct = (
        100 * (len(delivered) - len(late)) / len(delivered)
        if delivered else 0.0
    )
    avg_wait = (
        sum(o.wait_time for o in delivered) / len(delivered)
        if delivered else 0.0
    )
    load_counts = [d.completed_orders for d in env.drivers]
    load_std = _std(load_counts)

    return {
        "reward": round(total_reward, 1),
        "completed": len(delivered),
        "missed_deadlines": len(late),
        "on_time_pct": round(on_time_pct, 1),
        "avg_wait": round(avg_wait, 2),
        "driver_loads": load_counts,
        "load_balance_std": round(load_std, 2),
    }


def _std(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


# -----------------------------------------------------------------------
# Main training run
# -----------------------------------------------------------------------

def train(
    n_episodes=1000,
    traffic_pattern="rush_hour",
    verbose_every=100,
    save_path="training/results.json",
):
    print(f"=== Training RL agent for {n_episodes} episodes ===")
    print(f"Traffic pattern: {traffic_pattern}\n")

    env = DeliveryEnv(traffic_pattern=traffic_pattern)
    agent = QLearningAgent()

    rl_history = []
    for ep in range(1, n_episodes + 1):
        metrics = run_episode_agent(env, agent, train=True)
        rl_history.append(metrics)

        if ep % verbose_every == 0:
            recent = rl_history[-verbose_every:]
            avg_reward = sum(m["reward"] for m in recent) / len(recent)
            avg_ontime = sum(m["on_time_pct"] for m in recent) / len(recent)
            print(
                f"Ep {ep:4d} | "
                f"avg reward: {avg_reward:7.1f} | "
                f"on-time: {avg_ontime:5.1f}% | "
                f"epsilon: {agent.epsilon:.3f} | "
                f"states: {len(agent.q_table)}"
            )

    # -----------------------------------------------------------------------
    # Run baselines (100 episodes each)
    # -----------------------------------------------------------------------
    print("\n=== Running baselines (100 episodes each) ===")
    baselines = {
        "random": RandomBaseline(),
        "round_robin": RoundRobinBaseline(),
        "greedy_nearest": GreedyNearestBaseline(),
    }
    baseline_history = {}
    for name, baseline in baselines.items():
        history = []
        for _ in range(100):
            metrics = run_episode_baseline(env, baseline)
            history.append(metrics)
        avg_r = sum(m["reward"] for m in history) / len(history)
        avg_ot = sum(m["on_time_pct"] for m in history) / len(history)
        print(f"  {name:20s} | avg reward: {avg_r:7.1f} | on-time: {avg_ot:5.1f}%")
        baseline_history[name] = history

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = {
        "rl_history": rl_history,
        "baseline_history": baseline_history,
        "config": {
            "n_episodes": n_episodes,
            "traffic_pattern": traffic_pattern,
        },
    }
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    return results


if __name__ == "__main__":
    train(n_episodes=2000, verbose_every=200)