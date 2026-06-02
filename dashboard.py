"""
dashboard.py — Interactive Visualization Dashboard
Dynamic Multi-Order Delivery Optimization
CS4100 Final Project — Khoury College, Northeastern University

Run from project root:
    python dashboard.py

Reads from:
    training/results.json       (training history + baselines)
    environment/delivery_env.py (DeliveryEnv for live replay)
    agents/q_agent.py           (QLearningAgent)
    agents/baselines.py         (baselines)

Outputs:
    evaluation/plots/           (all saved figures)
"""

import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

# ── project imports ──────────────────────────────────────────────────────────
from environment.delivery_env import DeliveryEnv
from agents.q_agent import QLearningAgent
from agents.baselines import RandomBaseline, RoundRobinBaseline, GreedyNearestBaseline
from training.train import run_episode_agent, run_episode_baseline

PLOT_DIR = "evaluation/plots"
RESULTS_PATH = "training/results.json"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────
C = {
    "rl":     "#378ADD",
    "random": "#B4B2A9",
    "rr":     "#1D9E75",
    "greedy": "#EF9F27",
    "bg":     "#FAFAF8",
    "grid":   "#EEECEA",
    "text":   "#2C2C2A",
    "muted":  "#888780",
    "red":    "#E24B4A",
    "rush":   "#E24B4A",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(C["bg"])
    ax.grid(True, color=C["grid"], linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C["grid"])
    ax.tick_params(colors=C["muted"], labelsize=9)
    ax.xaxis.label.set_color(C["muted"])
    ax.yaxis.label.set_color(C["muted"])
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", color=C["text"], pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)

def smooth(values, w=40):
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - w): i + 1]
        out.append(sum(chunk) / len(chunk))
    return out

def load_results(path=RESULTS_PATH):
    with open(path) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — Learning Curves (4-panel)
# ════════════════════════════════════════════════════════════════════════════
def plot_learning_curves(results, save=True):
    """4-panel learning curve: reward, on-time %, avg wait, missed deadlines."""
    rl  = results["rl_history"]
    bsl = results["baseline_history"]
    eps = list(range(1, len(rl) + 1))

    metrics = [
        ("reward",           "Total reward",        "higher → better"),
        ("on_time_pct",      "On-time %",           "higher → better"),
        ("avg_wait",         "Avg wait time (min)", "lower → better"),
        ("missed_deadlines", "Missed deadlines",    "lower → better"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("RL Agent vs Baselines — Learning Curves",
                 fontsize=14, fontweight="bold", color=C["text"], y=0.98)
    axes = axes.flatten()

    bsl_style = {
        "random": dict(color=C["random"], ls="--", lw=1.4, label="Random"),
        "round_robin": dict(color=C["rr"], ls="--", lw=1.4, label="Round robin"),
        "greedy_nearest": dict(color=C["greedy"], ls="--", lw=1.4, label="Greedy nearest"),
    }

    for ax, (key, label, hint) in zip(axes, metrics):
        raw = [m[key] for m in rl]
        sm  = smooth(raw, w=40)

        # RL raw + smoothed
        ax.plot(eps, raw, color=C["rl"], alpha=0.12, linewidth=0.5)
        ax.plot(eps, sm,  color=C["rl"], linewidth=2.2, label="RL agent", zorder=5)

        # Baseline flat lines with ±1σ band
        for bname, bdata in bsl.items():
            vals = [m[key] for m in bdata]
            mu, sd = np.mean(vals), np.std(vals)
            style = bsl_style.get(bname, dict(color="#ccc", ls="--", lw=1))
            ax.axhline(mu, **{k: v for k, v in style.items() if k != "label"},
                       label=style["label"], zorder=4)
            ax.axhspan(mu - sd, mu + sd, color=style["color"], alpha=0.06)

        _style_ax(ax, title=f"{label}  ({hint})", xlabel="Episode")
        ax.legend(fontsize=8, framealpha=0.85, loc="best")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "01_learning_curves.png")
    if save: fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Final Performance Bar Comparison
# ════════════════════════════════════════════════════════════════════════════
def plot_final_comparison(results, save=True):
    """Grouped bar chart comparing last-100-episode performance."""
    rl  = results["rl_history"][-100:]
    bsl = results["baseline_history"]

    agents = {"RL Agent": rl, **{k.replace("_", " ").title(): v
                                  for k, v in bsl.items()}}
    colors = [C["rl"], C["random"], C["rr"], C["greedy"]]

    metrics = [
        ("reward",           "Total reward"),
        ("on_time_pct",      "On-time %"),
        ("avg_wait",         "Avg wait (min)"),
        ("missed_deadlines", "Missed deadlines"),
    ]

    n_metrics = len(metrics)
    n_agents  = len(agents)
    width = 0.18
    offsets = np.linspace(-(n_agents - 1) / 2, (n_agents - 1) / 2, n_agents) * width
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(C["bg"])

    for i, (name, hist) in enumerate(agents.items()):
        means = [np.mean([m[k] for m in hist]) for k, _ in metrics]
        stds  = [np.std( [m[k] for m in hist]) for k, _ in metrics]
        bars  = ax.bar(x + offsets[i], means, width,
                       label=name, color=colors[i], alpha=0.85,
                       yerr=stds, capsize=3,
                       error_kw=dict(linewidth=0.8, color=C["muted"]))

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=11)
    ax.axhline(0, color=C["text"], linewidth=0.5)
    _style_ax(ax, title="Final Performance Comparison (last 100 episodes)")
    ax.legend(fontsize=9, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "02_final_comparison.png")
    if save: fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — City Grid Visualization
# ════════════════════════════════════════════════════════════════════════════
def plot_city_grid(env, episode_num=0, save=True):
    """Draw the 5×5 city with edge weights, rush zone, and driver positions."""
    city    = env.city
    drivers = env.drivers

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("City Grid — Edge Weights & Driver State",
                 fontsize=13, fontweight="bold", color=C["text"])

    RUSH_NODES = {6, 7, 11, 12, 13, 17, 18}
    GRID = 5
    node_pos = {n: (n % GRID, GRID - 1 - n // GRID) for n in range(25)}

    for panel, hour in enumerate([0, 9]):  # normal vs rush hour
        ax = axes[panel]
        ax.set_facecolor(C["bg"])
        ax.set_aspect("equal")
        ax.set_title(f"{'Rush hour (9am)' if hour == 9 else 'Off-peak (midnight)'}",
                     fontsize=11, color=C["text"])
        ax.axis("off")

        # Rush zone background
        if hour == 9:
            from matplotlib.patches import FancyBboxPatch
            ax.add_patch(FancyBboxPatch(
                (0.7, 0.7), 2.6, 2.6,
                boxstyle="round,pad=0.1",
                facecolor=C["rush"], alpha=0.08,
                edgecolor=C["rush"], linewidth=1, linestyle="--"
            ))
            ax.text(2.0, 3.45, "rush zone", ha="center", fontsize=8,
                    color=C["rush"], alpha=0.7)

        # Edges
        for (a, b), base_w in city.edges.items():
            xa, ya = node_pos[a]
            xb, yb = node_pos[b]
            actual = city.get_travel_time(a, b, hour)
            ratio  = actual / base_w  # 1.0 or 2.0
            color  = C["rush"] if ratio > 1.0 else C["muted"]
            lw     = 2.5 if ratio > 1.0 else 1.2
            alpha  = 0.7 if ratio > 1.0 else 0.35
            ax.plot([xa, xb], [ya, yb], color=color, lw=lw, alpha=alpha, zorder=1)
            # Edge weight label
            mx, my = (xa + xb) / 2, (ya + yb) / 2
            ax.text(mx, my, str(base_w), fontsize=6, ha="center", va="center",
                    color=C["muted"], zorder=3,
                    bbox=dict(boxstyle="round,pad=0.1", fc=C["bg"], ec="none", alpha=0.7))

        # Node dots
        for n, (x, y) in node_pos.items():
            in_rush = n in RUSH_NODES and hour == 9
            col = C["rush"] if in_rush else "#D3D1C7"
            ax.scatter(x, y, s=120, color=col, zorder=4, edgecolors=C["bg"], linewidth=1.5)
            ax.text(x, y - 0.28, str(n), fontsize=7, ha="center",
                    va="center", color=C["muted"], zorder=5)

        # Driver positions (only first panel)
        if panel == 0 and drivers:
            driver_colors = ["#378ADD", "#1D9E75", "#D85A30"]
            for d in drivers:
                x, y = node_pos[d.current_node]
                ax.scatter(x, y, s=350, color=driver_colors[d.id],
                           zorder=6, edgecolors="white", linewidth=2)
                ax.text(x, y, f"D{d.id}", fontsize=8, ha="center", va="center",
                        color="white", fontweight="bold", zorder=7)

        # Pending order destinations
        if panel == 0:
            for oid in env.pending_orders[:5]:
                o = env.orders[oid]
                x, y = node_pos[o.destination]
                ax.scatter(x, y, marker="s", s=200, color=C["greedy"],
                           zorder=5, alpha=0.8, edgecolors="white", linewidth=1.5)
                ax.text(x, y, f"#{oid}", fontsize=6.5, ha="center", va="center",
                        color="white", fontweight="bold", zorder=7)

    # Legend
    legend_elements = [
        mpatches.Patch(color=C["rush"],   alpha=0.7, label="Rush hour edge (2×)"),
        mpatches.Patch(color=C["muted"],  alpha=0.5, label="Normal edge (1×)"),
        mpatches.Patch(color="#378ADD",   label="Driver 0"),
        mpatches.Patch(color="#1D9E75",   label="Driver 1"),
        mpatches.Patch(color="#D85A30",   label="Driver 2"),
        mpatches.Patch(color=C["greedy"], label="Pending order"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "03_city_grid.png")
    if save: fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════
#  PLOT 4 — Traffic Learning Heatmap
# ════════════════════════════════════════════════════════════════════════════
def plot_traffic_heatmap(env, save=True):
    """Heatmap of learned average travel times per edge per hour."""
    city  = env.city
    edges = sorted(city.edges.keys())

    matrix = np.zeros((len(edges), 24))
    for i, edge in enumerate(edges):
        for h in range(24):
            learned = city.learned_traffic[edge][h]
            matrix[i, h] = learned if learned is not None else city.edges[edge]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Traffic Learning — What the RL Agent Discovered",
                 fontsize=13, fontweight="bold", color=C["text"])

    # ── left: heatmap ─────────────────────────────────────────────────────
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list(
        "traffic", ["#1D9E75", "#EF9F27", "#E24B4A"]
    )
    im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                   interpolation="nearest", vmin=5, vmax=30)
    ax.set_xlabel("Hour of day", fontsize=9, color=C["muted"])
    ax.set_ylabel("Edge index", fontsize=9, color=C["muted"])
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 3)],
                       fontsize=8, color=C["muted"], rotation=30)
    ax.tick_params(axis="y", labelsize=7, colors=C["muted"])

    # Rush hour markers
    for h in [8, 9, 10, 17, 18, 19]:
        ax.axvline(h, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(9, -1.5, "morning\nrush", ha="center", fontsize=7,
            color="white", va="top")
    ax.text(18, -1.5, "evening\nrush", ha="center", fontsize=7,
            color="white", va="top")

    plt.colorbar(im, ax=ax, label="Avg travel time (min)", shrink=0.8)
    _style_ax(ax, title="Edge × Hour heatmap (agent learned from scratch)")

    # ── right: avg by hour ─────────────────────────────────────────────────
    ax2 = axes[1]
    mean_by_hour = matrix.mean(axis=0)
    hours = np.arange(24)
    colors_h = [C["rush"] if (8 <= h <= 10 or 17 <= h <= 19) else C["rl"]
                for h in hours]
    ax2.barh(hours, mean_by_hour, color=colors_h, alpha=0.85, height=0.7)
    ax2.set_xlabel("Avg time (min)", fontsize=9, color=C["muted"])
    ax2.set_yticks(range(0, 24, 3))
    ax2.set_yticklabels([f"{h}h" for h in range(0, 24, 3)], fontsize=8)
    ax2.invert_yaxis()
    _style_ax(ax2, title="Avg by hour")

    legend_elements = [
        mpatches.Patch(color=C["rush"], label="Rush hour"),
        mpatches.Patch(color=C["rl"],   label="Off-peak"),
    ]
    ax2.legend(handles=legend_elements, fontsize=8, loc="lower right")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "04_traffic_heatmap.png")
    if save: fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════
#  PLOT 5 — Episode Replay: Single Episode Step-by-Step
# ════════════════════════════════════════════════════════════════════════════
def plot_episode_replay(env, agent, n_snapshots=6, save=True):
    """Run one episode and capture grid snapshots at regular intervals."""
    GRID = 5
    node_pos = {n: (n % GRID, GRID - 1 - n // GRID) for n in range(25)}
    driver_colors = ["#378ADD", "#1D9E75", "#D85A30"]
    RUSH_NODES = {6, 7, 11, 12, 13, 17, 18}

    snapshots = []
    state = env.reset()
    state_key = env.encode_state()
    step = 0

    while not env.done:
        if env.pending_orders:
            action = agent.act(state_key, env)
            _, _, done, info = env.step(env.decode_action(action))
            state_key = env.encode_state()
        else:
            env._run_until_decision()

        step += 1
        # capture evenly spaced snapshots
        if step % max(1, 480 // n_snapshots) == 0 or env.done:
            snapshots.append({
                "time": env.time,
                "drivers": [(d.current_node, d.queue_length(), d.status) for d in env.drivers],
                "pending": list(env.pending_orders[:4]),
                "orders": {oid: env.orders[oid].destination
                           for oid in env.pending_orders[:4]},
                "completed": env.completed_count,
            })
        if len(snapshots) >= n_snapshots:
            break

    cols = 3
    rows = (len(snapshots) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Episode Replay — RL Agent Dispatching Orders",
                 fontsize=13, fontweight="bold", color=C["text"])
    axes = np.array(axes).flatten()

    for idx, snap in enumerate(snapshots):
        ax = axes[idx]
        ax.set_facecolor(C["bg"])
        ax.set_aspect("equal")
        ax.set_title(f"t = {snap['time']} min  |  {snap['completed']} delivered",
                     fontsize=9, color=C["text"])
        ax.axis("off")

        # Edges
        for (a, b) in env.city.edges:
            xa, ya = node_pos[a]; xb, yb = node_pos[b]
            ax.plot([xa, xb], [ya, yb], color="#D3D1C7", lw=1.2, alpha=0.5, zorder=1)

        # Rush zone
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch(
            (0.7, 0.7), 2.6, 2.6, boxstyle="round,pad=0.1",
            facecolor=C["rush"], alpha=0.05,
            edgecolor=C["rush"], linewidth=0.8, linestyle="--"
        ))

        # Nodes
        for n, (x, y) in node_pos.items():
            ax.scatter(x, y, s=60, color="#D3D1C7", zorder=3,
                       edgecolors=C["bg"], linewidth=1)
            ax.text(x, y - 0.3, str(n), fontsize=6, ha="center",
                    color=C["muted"], zorder=4)

        # Pending orders
        for oid, dest in snap["orders"].items():
            x, y = node_pos[dest]
            ax.scatter(x, y, marker="s", s=180, color=C["greedy"],
                       zorder=5, alpha=0.85, edgecolors="white", linewidth=1.5)

        # Drivers
        for did, (node, qlen, status) in enumerate(snap["drivers"]):
            x, y = node_pos[node]
            ax.scatter(x, y, s=320, color=driver_colors[did],
                       zorder=6, edgecolors="white", linewidth=2)
            ax.text(x, y, f"D{did}\n({qlen})",
                    fontsize=6.5, ha="center", va="center",
                    color="white", fontweight="bold", zorder=7)

    # Hide unused subplots
    for idx in range(len(snapshots), len(axes)):
        axes[idx].set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(color=c, label=f"Driver {i}")
        for i, c in enumerate(driver_colors)
    ] + [mpatches.Patch(color=C["greedy"], label="Pending order")]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "05_episode_replay.png")
    if save: fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════
#  PLOT 6 — Driver Load Balance
# ════════════════════════════════════════════════════════════════════════════
def plot_load_balance(results, save=True):
    """Box plots + violin of per-driver load distribution across episodes."""
    rl  = results["rl_history"]
    bsl = results["baseline_history"]

    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=False)
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Driver Load Balance — Orders Completed per Driver",
                 fontsize=13, fontweight="bold", color=C["text"])

    all_agents = {"RL Agent": rl, **{
        k.replace("_", " ").title(): v for k, v in bsl.items()
    }}
    colors = [C["rl"], C["random"], C["rr"], C["greedy"]]

    for ax, (name, hist), col in zip(axes, all_agents.items(), colors):
        loads = [ep["driver_loads"] for ep in hist]
        data  = [[ep["driver_loads"][d] for ep in hist] for d in range(3)]

        vp = ax.violinplot(data, positions=[0, 1, 2], showmedians=True)
        for body in vp["bodies"]:
            body.set_facecolor(col)
            body.set_alpha(0.4)
        vp["cmedians"].set_color(col)
        vp["cbars"].set_color(C["muted"])
        vp["cmins"].set_color(C["muted"])
        vp["cmaxes"].set_color(C["muted"])

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Driver 0", "Driver 1", "Driver 2"], fontsize=9)
        _style_ax(ax, title=name, ylabel="Orders completed")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "06_load_balance.png")
    if save: fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════
#  PLOT 7 — Reward Distribution (box + scatter)
# ════════════════════════════════════════════════════════════════════════════
def plot_reward_distribution(results, save=True):
    """Box plots of reward distribution for all agents."""
    rl  = results["rl_history"]
    bsl = results["baseline_history"]

    all_agents = {
        "RL Agent":     [m["reward"] for m in rl],
        "Random":       [m["reward"] for m in bsl["random"]],
        "Round Robin":  [m["reward"] for m in bsl["round_robin"]],
        "Greedy Near.": [m["reward"] for m in bsl["greedy_nearest"]],
    }
    colors = [C["rl"], C["random"], C["rr"], C["greedy"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Reward Distribution — All Agents",
                 fontsize=13, fontweight="bold", color=C["text"])

    # ── left: box plot ─────────────────────────────────────────────────────
    ax = axes[0]
    bp = ax.boxplot(
        list(all_agents.values()),
        tick_labels=list(all_agents.keys()),
        patch_artist=True,
        notch=False,
        widths=0.5,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=C["muted"]),
        capprops=dict(color=C["muted"]),
        flierprops=dict(marker="o", markerfacecolor=C["muted"],
                        markersize=3, alpha=0.4, linestyle="none"),
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax.axhline(0, color=C["text"], linewidth=0.5, linestyle="--")
    _style_ax(ax, title="Box plot", ylabel="Total reward")

    # ── right: rolling mean comparison ────────────────────────────────────
    ax2 = axes[1]
    rl_rewards = [m["reward"] for m in rl]
    sm = smooth(rl_rewards, w=40)
    eps = list(range(1, len(rl) + 1))
    ax2.plot(eps, rl_rewards, color=C["rl"], alpha=0.1, linewidth=0.5)
    ax2.plot(eps, sm, color=C["rl"], linewidth=2, label="RL (smoothed)")

    for (name, vals), col, ls in zip(
        list(all_agents.items())[1:],
        [C["random"], C["rr"], C["greedy"]],
        ["--", "--", "--"]
    ):
        ax2.axhline(np.mean(vals), color=col, linewidth=1.5, linestyle=ls, label=name)

    _style_ax(ax2, title="Rolling mean vs baselines",
              xlabel="Episode", ylabel="Reward")
    ax2.legend(fontsize=9, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "07_reward_distribution.png")
    if save: fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════
#  PLOT 8 — Q-Table Coverage (states explored over episodes)
# ════════════════════════════════════════════════════════════════════════════
def plot_epsilon_and_states(env, n_episodes=200, save=True):
    """
    Re-runs a short training sequence and records epsilon + state count.
    Shows exploration-exploitation trade-off live.
    """
    agent = QLearningAgent()
    epsilons, states = [], []

    print("  Re-running mini training (200 eps) for epsilon curve …")
    for ep in range(n_episodes):
        state = env.reset()
        sk = env.encode_state()
        while not env.done:
            if env.pending_orders:
                action = agent.act(sk, env)
                _, r, done, info = env.step(env.decode_action(action))
                nsk = env.encode_state()
                agent.update(sk, action, r, nsk, done, env.valid_actions())
                sk = nsk
            else:
                env._run_until_decision()
        agent.decay_epsilon()
        epsilons.append(agent.epsilon)
        states.append(len(agent.q_table))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Exploration-Exploitation Trade-off",
                 fontsize=13, fontweight="bold", color=C["text"])

    eps_x = list(range(1, n_episodes + 1))

    ax = axes[0]
    ax.plot(eps_x, epsilons, color=C["rl"], linewidth=2)
    ax.fill_between(eps_x, epsilons, alpha=0.15, color=C["rl"])
    ax.axhline(0.1, color=C["muted"], linewidth=1, linestyle="--",
               label="ε_min = 0.10")
    _style_ax(ax, title="Epsilon decay", xlabel="Episode", ylabel="ε (exploration rate)")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.plot(eps_x, states, color=C["greedy"], linewidth=2)
    ax2.fill_between(eps_x, states, alpha=0.15, color=C["greedy"])
    _style_ax(ax2, title="States discovered", xlabel="Episode",
              ylabel="Unique states in Q-table")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "08_epsilon_states.png")
    if save: fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════
#  SUMMARY CARD — key numbers for the report
# ════════════════════════════════════════════════════════════════════════════
def print_summary(results):
    rl  = results["rl_history"][-100:]
    bsl = results["baseline_history"]

    def avg(hist, key):
        return round(np.mean([m[key] for m in hist]), 2)

    print("\n" + "═" * 58)
    print("  PROJECT RESULTS SUMMARY  (last 100 episodes)")
    print("═" * 58)
    header = f"{'Metric':<22} {'RL':>8} {'Random':>8} {'RndRobin':>10} {'Greedy':>8}"
    print(header)
    print("─" * 58)
    for key, label in [
        ("reward",           "Total reward"),
        ("on_time_pct",      "On-time %"),
        ("avg_wait",         "Avg wait (min)"),
        ("missed_deadlines", "Missed deadlines"),
        ("completed",        "Orders completed"),
    ]:
        vals = [avg(rl, key)] + [avg(bsl[b], key)
                for b in ["random", "round_robin", "greedy_nearest"]]
        row = f"  {label:<20} " + "  ".join(f"{v:>8.2f}" for v in vals)
        print(row)
    print("═" * 58)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Delivery Optimization Dashboard")
    parser.add_argument("--skip-replay",   action="store_true",
                        help="Skip the episode replay plot (faster)")
    parser.add_argument("--skip-epsilon",  action="store_true",
                        help="Skip the epsilon/states plot (needs re-training)")
    parser.add_argument("--results",       default=RESULTS_PATH,
                        help="Path to results.json")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Delivery Optimization — Dashboard Generator    ║")
    print("║  CS4100 · Khoury College · Northeastern Univ.   ║")
    print("╚══════════════════════════════════════════════════╝\n")

    # ── load results ──────────────────────────────────────────────────────
    if not os.path.exists(args.results):
        print(f"ERROR: {args.results} not found. Run training/train.py first.")
        sys.exit(1)

    print("Loading results …")
    results = load_results(args.results)
    print(f"  {len(results['rl_history'])} RL episodes loaded.")

    # ── set up env + trained agent ────────────────────────────────────────
    print("\nInitialising environment …")
    env = DeliveryEnv(traffic_pattern="rush_hour")
    env.reset()

    agent = QLearningAgent()

    # ── generate all plots ────────────────────────────────────────────────
    print("\nGenerating plots …")

    print("\n[1/7] Learning curves")
    plot_learning_curves(results)

    print("[2/7] Final comparison")
    plot_final_comparison(results)

    print("[3/7] City grid")
    plot_city_grid(env)

    print("[4/7] Traffic heatmap")
    plot_traffic_heatmap(env)

    if not args.skip_replay:
        print("[5/7] Episode replay")
        plot_episode_replay(env, agent)
    else:
        print("[5/7] Episode replay  → skipped")

    print("[6/7] Load balance")
    plot_load_balance(results)

    print("[7/7] Reward distribution")
    plot_reward_distribution(results)

    if not args.skip_epsilon:
        print("[+] Epsilon/states curve  (takes ~30s) …")
        plot_epsilon_and_states(DeliveryEnv(traffic_pattern="rush_hour"))

    # ── print summary ─────────────────────────────────────────────────────
    print_summary(results)

    print(f"\n✓ All plots saved to  {PLOT_DIR}/")
    print("  01_learning_curves.png")
    print("  02_final_comparison.png")
    print("  03_city_grid.png")
    print("  04_traffic_heatmap.png")
    print("  05_episode_replay.png")
    print("  06_load_balance.png")
    print("  07_reward_distribution.png")
    print("  08_epsilon_states.png")
    print("\nDone. Use these plots in your project summary + GitHub README.\n")


if __name__ == "__main__":
    main()