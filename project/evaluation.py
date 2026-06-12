"""
Experiment runner, statistical analysis, and plots.

Funcs:
    run_experiments(agents, env, n_episodes)
        Run each agent for n_episodes and collect per-episode mets.

    statistical_analysis(results)
        Compute mean, std, and 95% confidence interval for each metric.

    generate_plots(results, stats, save_dir)
        Produce all required plots:
          1. Comparison bar chart  — all agents side by side on key metrics
          2. Learning curve        — episode reward over time (per agent)
          3. On-time % over time   — rolling average per agent
          4. Avg wait over time    — rolling average per agent
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from baselines import RandomAgent, GreedyAgent, run_baseline_episode
from rl_agent import QLearningAgent
from evaluation import run_experiments, statistical_analysis, generate_plots
from delivery_env import DeliveryEnvironment



env = DeliveryEnvironment()
agents = {
    "Random":       RandomAgent(),
    "Greedy":       GreedyAgent(),
    "RL":           QLearningAgent(),
}
results = run_experiments(agents, env, n_episodes=30)
stats = statistical_analysis(results)
generate_plots(results, stats, save_dir="results/plots/")

try:
    from rl_agent import QLearningAgent   # type: ignore
    RLAgent = QLearningAgent
    print("rl_agent.py loaded, using QLearningAgent")
except ImportError:
    print("rl_agent.py not found")


def run_experiments(
    agents: Dict,
    env: DeliveryEnvironment,
    n_episodes: int = 30,
) -> Dict[str, List[dict]]:
    """
    Run each agent for n_episodes and collect per-episode mets.

    Args:
        agents      : dict mapping agent name -> agent instance
        env         : DeliveryEnvironment (reset() is called each ep)
        n_episodes  : number of eps to run per agent

    Returns:
        results dict  agent_name -> list of metric dicts
        Each metric dict has keys:
            completed, on_time, late, on_time_pct,
            avg_wait, total_reward, episode_length
    """
    results = {name: [] for name in agents}

    for name, agent in agents.items():
        print(f"\nRunning {name} for {n_episodes} episodes...")
        for ep in range(n_episodes):
            metrics = run_baseline_episode(agent, env)
            results[name].append(metrics)
            if (ep + 1) % 5 == 0:
                print(
                    f"  Episode {ep+1:3d}/{n_episodes} — "
                    f"completed={metrics['completed']:2d}  "
                    f"on_time={metrics['on_time_pct']:5.1f}%  "
                    f"reward={metrics['total_reward']:8.1f}"
                )

    return results


def statistical_analysis(results: Dict[str, List[dict]]) -> Dict[str, dict]:
    """
    For each agent and each metric, compute mean, std, and 95% CI.

    Returns:
        stats dict  agent_name -> metric_name -> {mean, std, ci95}
    """
    metrics_of_interest = [
        "completed", "on_time_pct", "avg_wait", "total_reward", "late"
    ]
    stats = {}

    for name, episodes in results.items():
        stats[name] = {}
        for metric in metrics_of_interest:
            values = [ep[metric] for ep in episodes]
            arr    = np.array(values, dtype=float)
            mean   = float(np.mean(arr))
            std    = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            # 95% CI using t-distribution approximation (z=1.96 for large n)
            ci95   = 1.96 * std / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
            stats[name][metric] = {
                "mean": round(mean, 3),
                "std":  round(std,  3),
                "ci95": round(ci95, 3),
            }

    return stats


COLORS = {
    "Random": "#e53935",   # red
    "Greedy": "#fb8c00",   # orange
    "RL":     "#1e88e5",   # blue
}

def _agent_color(name: str) -> str:
    for key, color in COLORS.items():
        if key.lower() in name.lower():
            return color
    return "#757575"

def _rolling_avg(values: List[float], window: int = 5) -> List[float]:
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start:i+1])))
    return out


# plot 1

def _plot_comparison(stats: Dict, save_path: str):
    """Bar chart comparing all agents on 4 key metrics."""
    metrics     = ["on_time_pct", "avg_wait", "completed", "total_reward"]
    labels      = ["On-time %", "Avg Wait (min)", "Orders Completed", "Total Reward"]
    agent_names = list(stats.keys())

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Agent Comparison — Mean ± 95% CI", fontsize=14, fontweight="bold")

    for ax, metric, label in zip(axes, metrics, labels):
        means  = [stats[n][metric]["mean"] for n in agent_names]
        ci95s  = [stats[n][metric]["ci95"] for n in agent_names]
        colors = [_agent_color(n) for n in agent_names]

        bars = ax.bar(agent_names, means, yerr=ci95s, capsize=5,
                      color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate bar tops
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(ci95s) * 0.1,
                    f"{mean:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# plot 2

def _plot_learning_curve(results: Dict, save_path: str):
    """Smoothed episode reward over time for each agent."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Episode Reward over Time", fontsize=13, fontweight="bold")

    for name, episodes in results.items():
        rewards  = [ep["total_reward"] for ep in episodes]
        smoothed = _rolling_avg(rewards, window=5)
        color    = _agent_color(name)
        xs       = list(range(1, len(episodes) + 1))
        ax.plot(xs, rewards,  color=color, alpha=0.25, linewidth=1)
        ax.plot(xs, smoothed, color=color, alpha=1.0,  linewidth=2, label=name)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# plot 3

def _plot_ontime_curve(results: Dict, save_path: str):
    """Smoothed on-time delivery percentage per episode."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("On-Time Delivery % over Episodes", fontsize=13, fontweight="bold")

    for name, episodes in results.items():
        values   = [ep["on_time_pct"] for ep in episodes]
        smoothed = _rolling_avg(values, window=5)
        color    = _agent_color(name)
        xs       = list(range(1, len(episodes) + 1))
        ax.plot(xs, values,   color=color, alpha=0.25, linewidth=1)
        ax.plot(xs, smoothed, color=color, alpha=1.0,  linewidth=2, label=name)

    ax.set_xlabel("Episode")
    ax.set_ylabel("On-Time %")
    ax.set_ylim(0, 105)
    ax.axhline(80, color="gray", linestyle="--", linewidth=1, label="80% target")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# plot 4

def _plot_wait_curve(results: Dict, save_path: str):
    """Smoothed average wait time per episode."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Average Wait Time over Episodes (lower = better)",
                 fontsize=13, fontweight="bold")

    for name, episodes in results.items():
        values   = [ep["avg_wait"] for ep in episodes]
        smoothed = _rolling_avg(values, window=5)
        color    = _agent_color(name)
        xs       = list(range(1, len(episodes) + 1))
        ax.plot(xs, values,   color=color, alpha=0.25, linewidth=1)
        ax.plot(xs, smoothed, color=color, alpha=1.0,  linewidth=2, label=name)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Wait (min)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def generate_plots(
    results: Dict[str, List[dict]],
    stats:   Dict[str, dict],
    save_dir: str = "results/",
):
    """
    Generate and save all 4 plots to save_dir.

    Args:
        results  : output of run_experiments()
        stats    : output of statistical_analysis()
        save_dir : directory to write .png files into (created if missing)
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nGenerating plots → {save_dir}")

    _plot_comparison(   stats,   os.path.join(save_dir, "comparison.png"))
    _plot_learning_curve(results, os.path.join(save_dir, "learning_curve.png"))
    _plot_ontime_curve(  results, os.path.join(save_dir, "ontime_curve.png"))
    _plot_wait_curve(    results, os.path.join(save_dir, "wait_curve.png"))

    print("All plots saved.")


def print_summary(stats: Dict):
    """Print a formatted summary table to stdout."""
    metrics = ["completed", "on_time_pct", "avg_wait", "total_reward", "late"]
    labels  = ["Completed", "On-time %", "Avg Wait", "Reward", "Late"]

    col_w = 22
    header = f"{'Metric':<14}" + "".join(f"{n:>{col_w}}" for n in stats)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for metric, label in zip(metrics, labels):
        row = f"{label:<14}"
        for name in stats:
            s = stats[name][metric]
            row += f"  {s['mean']:>7.2f} ± {s['ci95']:.2f}".rjust(col_w)
        print(row)

    print("=" * len(header))
