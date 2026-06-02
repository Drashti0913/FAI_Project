import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


COLORS = {
    "rl":          "#378ADD",
    "random":      "#B4B2A9",
    "round_robin": "#EF9F27",
    "greedy":      "#1D9E75",
}
NAMES = {
    "rl":          "RL Agent",
    "random":      "Random",
    "round_robin": "Round Robin",
    "greedy":      "Greedy Nearest",
}


def smooth(values, window=30):
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window): i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def load_results(path="/Users/drashti/Desktop/FAI Project/training/results.json"):
    with open(path) as f:
        return json.load(f)


def plot_learning_curves(results, save_dir="evaluation/plots"):
    os.makedirs(save_dir, exist_ok=True)
    rl = results["rl_history"]
    baselines = results["baseline_history"]

    episodes = list(range(1, len(rl) + 1))
    metrics_to_plot = [
        ("reward",        "Total reward",       True),
        ("on_time_pct",   "On-time % ",         False),
        ("avg_wait",      "Avg wait time (min)", True),
        ("missed_deadlines", "Missed deadlines", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("RL Agent vs Baselines — Learning Curves", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for ax, (key, label, lower_better) in zip(axes, metrics_to_plot):
        # RL curve (smoothed + raw)
        raw_rl = [m[key] for m in rl]
        sm_rl = smooth(raw_rl, window=40)
        ax.plot(episodes, raw_rl, color=COLORS["rl"], alpha=0.15, linewidth=0.6)
        ax.plot(episodes, sm_rl, color=COLORS["rl"], linewidth=2.0, label=NAMES["rl"])

        # Baseline flat lines (mean ± std band)
        for bname, bhistory in baselines.items():
            vals = [m[key] for m in bhistory]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            color = COLORS.get(bname, "#999")
            ax.axhline(mean_val, color=color, linewidth=1.5,
                       linestyle="--", label=NAMES.get(bname, bname))
            ax.axhspan(mean_val - std_val, mean_val + std_val,
                       color=color, alpha=0.07)

        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, "learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_final_comparison(results, save_dir="evaluation/plots"):
    os.makedirs(save_dir, exist_ok=True)
    rl = results["rl_history"]
    baselines = results["baseline_history"]

    # Use last 100 episodes of RL for fair comparison
    rl_last = rl[-100:]

    metrics = ["reward", "on_time_pct", "avg_wait", "missed_deadlines"]
    labels  = ["Total Reward", "On-time %", "Avg Wait (min)", "Missed Deadlines"]

    all_agents = {"rl": rl_last, **baselines}
    agent_keys = list(all_agents.keys())
    x = np.arange(len(metrics))
    width = 0.18
    offsets = np.linspace(-(len(agent_keys)-1)/2, (len(agent_keys)-1)/2, len(agent_keys)) * width

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, key in enumerate(agent_keys):
        history = all_agents[key]
        means = [np.mean([m[met] for m in history]) for met in metrics]
        stds  = [np.std( [m[met] for m in history]) for met in metrics]
        color = COLORS.get(key, "#888")
        bars = ax.bar(x + offsets[i], means, width, label=NAMES.get(key, key),
                      color=color, alpha=0.85, yerr=stds, capsize=3, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title("Final Performance Comparison (last 100 episodes)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, "final_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_traffic_heatmap(env, save_dir="evaluation/plots"):
    """Visualize what the agent learned about traffic."""
    os.makedirs(save_dir, exist_ok=True)
    city = env.city
    edges = list(city.edges.keys())

    # Build matrix: edges × hours
    matrix = np.zeros((len(edges), 24))
    for i, edge in enumerate(edges):
        for h in range(24):
            learned = city.learned_traffic[edge][h]
            if learned is not None:
                matrix[i, h] = learned
            else:
                matrix[i, h] = city.edges[edge]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_xlabel("Hour of day", fontsize=11)
    ax.set_ylabel("Edge index", fontsize=11)
    ax.set_title("Learned traffic: average edge travel time by hour", fontsize=12, fontweight="bold")
    ax.set_xticks(range(24))
    plt.colorbar(im, ax=ax, label="Travel time (min)")
    ax.spines[["top", "right"]].set_visible(False)

    # Mark rush hours
    for h in [8, 9, 10, 17, 18, 19]:
        ax.axvline(h, color="#378ADD", linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, "traffic_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_all(results_path="training/results.json"):
    results = load_results(results_path)
    plot_learning_curves(results)
    plot_final_comparison(results)
    print("All plots saved to evaluation/plots/")


if __name__ == "__main__":
    plot_all()