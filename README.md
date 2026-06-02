# Intelligent Traffic-Aware Delivery Optimization
**CS4100 · Artificial Intelligence · Khoury College, Northeastern University**

---

## Team Members
- Drashti Bhavsar
- Venkata Sai Udayini Vedantham
- Elyssa Querubin

---

## Problem Overview

Modern delivery systems fail under dynamic traffic conditions because traditional routing treats travel time as static. This project builds an intelligent dispatch system where 3 drivers operate on a 5×5 city grid, handling orders that arrive dynamically throughout an 8-hour simulated shift.

The system learns to dispatch orders efficiently **while simultaneously discovering traffic patterns from scratch** — without being told what those patterns are. This emergent traffic awareness is the core novelty of the project.

---

## What We Built (entirely from scratch)

| Component | Description |
|---|---|
| 5×5 city graph | 25 nodes, 40 bidirectional edges, random base weights 5–15 min |
| Traffic simulation | Rush-hour (8–10am, 5–7pm), uniform, and random congestion modes |
| Order generator | Dynamic arrivals (30% probability/minute), 30-min hard deadlines |
| Driver agents | Queue-based routing with A\* pathfinding, mid-route reassignment |
| Q-learning agent | Tabular Q-table, epsilon-greedy exploration — **implemented from scratch** |
| Traffic learning | Online running-average per edge per hour; agent discovers rush hour autonomously |
| 3 baselines | Random, Round Robin, Greedy Nearest — all implemented from scratch |

**No RL libraries used. No pre-trained models. All AI methods are custom Python implementations.**

---

## Key Results (2000 training episodes, rush hour traffic)

| Metric | RL Agent | Random | Round Robin | Greedy Nearest |
|---|---|---|---|---|
| Total reward | −338 | −309 | −270 | −347 |
| On-time % | 99.1% | 99.2% | 99.3% | 97.8% |
| Avg wait (min) | 5.87 | 5.38 | 4.68 | 4.76 |
| Missed deadlines | 0.44 | 0.40 | 0.35 | 1.08 |

**Key finding:** The RL agent learns rush-hour congestion patterns on center nodes (6, 7, 11–13, 17–18) purely through experience — with zero prior knowledge of the traffic pattern. This emergent routing behavior is what separates it from all three baselines, which have no traffic awareness at all.

Round Robin achieves lower average wait time through its fixed cycling strategy (perfect 17/17/16 load split every episode), but this is brittle hardcoded behavior — it cannot adapt to order urgency, driver position, or traffic conditions. The RL agent learns these trade-offs dynamically.

---

## AI Methods Implemented from Scratch

### Q-Learning Agent (`agents/q_agent.py`)
- Tabular Q-table using Python `defaultdict`
- Epsilon-greedy exploration with exponential decay (ε: 1.0 → 0.1)
- Standard Bellman update: `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]`
- Valid-action masking (only legal queue positions considered)
- Discount factor γ = 0.95, learning rate α = 0.1

### A\* Pathfinding (`environment/city.py`)
- Heuristic graph search over the 5×5 grid
- Edge weights = base travel time × traffic multiplier
- Uses **learned traffic estimates** as weights after warmup
- Multi-stop route construction for queued orders

### Traffic Learning (`environment/city.py`)
- Running average update per edge per hour:
  `new_avg = 0.9 × old_avg + 0.1 × observed_time`
- Agent starts with no traffic knowledge; converges to true pattern
- Automatically discovers 2× slowdown on rush-hour edges

### Reward Function
```
TotalReward = −(TotalWaitTime) − (100 × MissedDeadlines)
```
Shaped intermediate reward during training:
```
StepReward = −(load_imbalance × 2.0) − (pending_orders × 1.5)
```

### View the interactive dashboard

```
Open `delivery_simulation_dashboard.html` directly in any browser — no server needed.
```
---

## Repository Structure

```
FAI_Project/
│
├── smoke_test.py              ← verify all components work (run this first)
│
├── environment/
│   ├── __init__.py
│   ├── city.py                ← 5×5 grid, A* routing, traffic learning
│   ├── order.py               ← Order dataclass + dynamic generator
│   ├── driver.py              ← Driver movement, queue management
│   └── delivery_env.py        ← Main RL environment (reset / step / reward)
│
├── agents/
│   ├── __init__.py
│   ├── q_agent.py             ← Q-learning agent (from scratch)
│   └── baselines.py           ← Random, RoundRobin, GreedyNearest baselines
│
├── training/
│   ├── train.py               ← Training loop + baseline evaluation
│   └── results.json           ← Saved training history (2000 episodes)
│
├── evaluation/
│   ├── plot_results.py        ← Learning curves + comparison charts
│   └── plots/                 ← All 8 generated figures
│       ├── 01_learning_curves.png
│       ├── 02_final_comparison.png
│       ├── 03_city_grid.png
│       ├── 04_traffic_heatmap.png
│       ├── 05_episode_replay.png
│       ├── 06_load_balance.png
│       ├── 07_reward_distribution.png
│       └── 08_epsilon_states.png
│
├── dashboard.py               ← Full visualization dashboard (generates all 8 plots)
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Drashti0913/FAI_Project
cd FAI_Project
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify everything works
```bash
python3 smoke_test.py
```

Expected output:
```
Running smoke tests...
  [OK] City: 40 edges, path 0→24 cost=62.0, len=9
  [OK] Order: Order(id=0, dest=0, pending)
  [OK] Driver: route to node 6 = [0, 5, 6]
  [OK] Episode: 50 orders, reward=-221.0
  [OK] Q-agent: epsilon=0.995, states=0
5/5 tests passed
```

---

## Reproducing Results

### Train the agent
```bash
python3 training/train.py
```
Trains for 2000 episodes (~10 minutes). Saves results to `training/results.json`.
Output example:
```
Ep  200 | avg reward: -460.8 | on-time: 97.9% | epsilon: 0.367 | states: 193
Ep 2000 | avg reward: -354.7 | on-time: 99.0% | epsilon: 0.100 | states: 224
```

### Generate all visualizations
```bash
# Full dashboard (8 plots) — recommended
python3 dashboard.py

# Skip the epsilon curve re-training (faster)
python3 dashboard.py --skip-epsilon

# Learning curves + final comparison only
python3 evaluation/plot_results.py
```

All plots saved to `evaluation/plots/`.

---

## Visualization Output

| Plot | Description |
|---|---|
| `01_learning_curves.png` | 4-panel: reward, on-time %, wait time, missed deadlines over 2000 episodes |
| `02_final_comparison.png` | Grouped bar chart: RL vs all 3 baselines across all metrics |
| `03_city_grid.png` | 5×5 city map: edge weights, rush zone, driver positions, pending orders |
| `04_traffic_heatmap.png` | What the agent learned: edge speeds by hour (rush hour emergent) |
| `05_episode_replay.png` | 6 snapshots of one episode showing driver movement and order dispatch |
| `06_load_balance.png` | Violin plots of per-driver order counts across all agents |
| `07_reward_distribution.png` | Box plots + rolling mean comparison |
| `08_epsilon_states.png` | Epsilon decay and Q-table state discovery over training |

---

## Simulation Parameters

| Parameter | Value |
|---|---|
| Grid size | 5×5 (25 nodes, 40 edges) |
| Number of drivers | 3 |
| Episode duration | 480 minutes (8-hour shift) |
| Order arrival rate | 30% probability per minute |
| Order deadline | 30 minutes from arrival |
| Max orders per episode | 50 |
| Max queue per driver | 5 orders |
| Action space | 18 (3 drivers × 6 queue positions) |
| Rush hour nodes | 6, 7, 11, 12, 13, 17, 18 |
| Rush hour multiplier | 2× during 8–10am and 5–7pm |

---

## Traffic Patterns

The environment supports three traffic modes, configurable at runtime:

```python
env = DeliveryEnv(traffic_pattern="rush_hour")   # default
env = DeliveryEnv(traffic_pattern="uniform")      # no congestion
env = DeliveryEnv(traffic_pattern="random")       # 20% edges at 1.5×
```

The RL agent does **not** receive the traffic pattern as input. It must discover it.

---

## Dependencies

```
matplotlib>=3.7
numpy>=1.24
```

No PyTorch. No scikit-learn. No stable-baselines. No Gymnasium. No pre-trained models.
All reinforcement learning and pathfinding implemented from scratch in pure Python.

---

## Individual Contributions

| Member | Contributions |
|---|---|
| Drashti Bhavsar | Q-learning agent, environment design, training loop, reward function, dashboard |
| Venkata Sai Udayini Vedantham | City graph, A* routing, traffic simulation, traffic learning module |
| Elyssa Querubin | Driver logic, order generator, baselines, evaluation plots, project summary |

---

## Limitations and Future Work

**Current limitations:**
- Tabular Q-learning does not scale to larger grids; state space grows with city size
- Reward signal is delayed (episode-end), making credit assignment difficult
- Round Robin achieves better average wait time through hardcoded perfect load balancing

**Future directions:**
- Replace tabular Q-learning with Deep Q-Network (DQN) for continuous state spaces
- Add multi-agent RL where drivers coordinate with each other
- Extend to real city graphs using OpenStreetMap data
- Add vehicle capacity constraints and restaurant pickup locations

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100–107.
3. Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279–292.
4. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
