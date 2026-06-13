# Dynamic Multi-Order Delivery Optimization with Traffic-Aware Routing

**CS 5100 - Artificial Intelligence | Khoury College, Northeastern University**

**Team:** Drashti Bhavsar, Venkata Sai Udayini Vedantham, Elyssa Querubin

**GitHub Repository:** https://github.com/Drashti0913/FAI_Project

---

## 1. Problem Overview

### Motivation

Modern delivery systems operate under dynamic conditions where orders arrive continuously, drivers are distributed across a city, and traffic varies by time of day. Traditional dispatch approaches treat travel time as static and assign orders using simple heuristics, ignoring queue congestion, deadline urgency, and real-time conditions. This leads to suboptimal routes, missed deadlines, and imbalanced workloads.

### Problem Statement

**How should a delivery system manage a continuously growing set of orders for drivers already in motion, deciding both which drivers receive new orders and in what sequence to serve all pending deliveries, while minimizing customer wait time and learning from experience?**

### Our Solution

We developed an intelligent dispatch system that learns optimal order assignment and traffic-aware routing purely from experience, with no prior knowledge of traffic patterns. The system:

- **Environment:** 5×5 city grid with three drivers
- **Dynamic Orders:** Continuous arrivals with 30-minute deadlines
- **Traffic Patterns:** Realistic rush-hour congestion on center edges
- **Learning:** Discovers that certain edges slow down at specific hours solely by observing travel times

Unlike classical Vehicle Routing Problem solvers that require full information upfront, our approach learns continuously and adapts to changing conditions.

---

## 2. Environment Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Drashti0913/FAI_Project.git
cd FAI_Project/project
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install numpy matplotlib pygame flask tqdm torch
```

### Verify Installation

```bash
python delivery_env.py
```

Should complete without errors, showing "Testing environment with random actions..."

---

## 3. Reproducing Results

### Single-Driver Q-Learning Training

**Train the RL agent (recommended first run):**

```bash
python train_delivery.py train
```

**Configuration (in `train_delivery.py` lines 11-12):**
```python
NUM_EPISODES = 100000      # Number of training episodes
EPSILON_DECAY = 0.999977   # Exploration decay rate
```

**Expected results:**
- **1,000 episodes:** ~-10,000 reward, ~20% on-time, ~5 orders completed
- **10,000 episodes:** ~-20,000 reward, ~55% on-time, ~45 orders completed  
- **100,000 episodes:** ~0 reward, ~75% on-time, 50 orders completed (cap)

**Training time:** ~15-20 minutes on standard laptop

**Outputs:**
- `Q_table_final.pkl` - Trained Q-table
- `training_100000.png` - Learning curves showing improvement

### Multi-Driver Training (3 Drivers)

**For 3-driver system:**

```bash
# Modify delivery_env.py line ~120:
# num_drivers=3

python train_delivery.py train
```

**Note:** Multi-driver training requires significantly more memory and time.

**Our results with 3 drivers:**
- **1,000 episodes:** -9,348 reward, 28.7% on-time, 408,959 Q-states
- **10,000 episodes:** -3,900 reward, 51.5% on-time, 4,051,347 Q-states
- **50,000+ episodes:** Training exceeded hardware memory capacity (21M+ states)

**Hardware limitation:** Tabular Q-learning with 3 drivers exceeded RAM at ~50k episodes. DQN implementation (`dqn_agent.py`, `train_dqn.py`) provided as alternative for large state spaces.

### Baseline Comparison

**Ran all three agents (Random, Greedy, RL) to generate comparison plots:**

**What this does:**
- Loads trained RL agent
- Runs 30 episodes each for Random, Greedy, and RL agents
- Computes statistical metrics (mean, std, 95% CI)
- Generates 4 plots in `results/` folder

**Our evaluation results (1 driver):**

| Agent | Avg Reward | On-Time % | Orders Completed | Avg Wait (min) |
|-------|-----------|-----------|------------------|----------------|
| Random | -19,723 | 2.5% | 10.0 | 230.0 |
| Greedy (BFS) | +6,791 | 91.5% | 50.0 | 12.9 |
| RL (Q-Learning) | -10,622 | 70.7% | 33.7 | 66.6 |

**Key findings:**
- RL substantially outperforms Random (68 percentage point improvement in on-time delivery)
- RL approaches Greedy performance despite learning from scratch
- Greedy uses hardcoded BFS shortest path; RL discovers paths through experience

### Live Web UI Demo

**Launch interactive visualization:**

```bash
python server.py
```

Open browser: `http://localhost:5000`

**Features:**
- Real-time 5×5 grid with 3 drivers
- Traffic visualization (rush hour edges highlighted in red)
- Live statistics (completed orders, on-time %, average wait)
- Interactive controls (play/pause, speed adjustment, time jump)
- Event log showing deliveries and assignments

---

## 4. Code Organization

### Repository Structure

```
FAI_Project/project/
│
├── delivery_env.py                # Simulation environment (200 lines)
├── assignment.py                  # Assignment heuristic (150 lines)
├── rl_agent.py                    # Q-learning agent (120 lines)
└── dqn_agent.py                   # DQN agent for large state spaces (150 lines)
│
├── train_delivery.py              # Q-learning training (180 lines)
├── train_dqn.py                   # DQN training (200 lines)
│
├── baselines.py                   # Random and Greedy baselines (140 lines)
│-── evaluation.py                 # Statistical analysis and plots (220 lines)
│
├── server.py                      # Flask web UI (350 lines)
│
└── Results
    ├── RL_with_1_driver/          # Single-driver training results
    ├── RL_with_3_drivers/         # Multi-driver training results
    └── Training_plots_1_driver/   # Learning curve visualizations
```

### File Descriptions

**`delivery_env.py`**
- `DeliveryEnvironment`: Main simulation orchestrator
- `CityGraph`: 5×5 grid with bidirectional edges and traffic multipliers
- `Driver`: State management (position, queue, status)
- `Order`: Order data (destination, deadlines, assignment)
- Action-based step function for RL control
- Traffic data collection for learning

**`assignment.py`**
- `assign_order_to_driver()`: Simulation-based driver selection
- `evaluate_queue()`: Multi-objective scoring (deadlines, distance, fairness)
- `locally_optimize_queue()`: Hill-climbing queue optimization
- `_bfs_path()`: BFS pathfinding for travel time estimation
- `_estimate_travel_time()`: Uses learned traffic data when available

**`rl_agent.py`**
- `QLearningAgent`: Tabular Q-learning implementation
- `discretize_state()`: Encodes active drivers only (state space optimization)
- `select_action()`: Epsilon-greedy exploration strategy
- `update()`: Q-learning Bellman update rule
- Sparse Q-table storage using Python dictionary

**`dqn_agent.py`** [For large state spaces]
- `DQNAgent`: Deep Q-Network with experience replay
- `DQNNetwork`: Neural network (10 → 128 → 128 → 125)
- Handles millions of states with fixed memory footprint

**`baselines.py`**
- `RandomAgent`: Random movement baseline
- `GreedyAgent`: BFS shortest-path navigation (ignores traffic)
- `run_baseline_episode()`: Standardized evaluation framework

**`evaluation.py`**
- Loads trained models
- Runs comparison experiments (30 episodes per agent)
- Statistical analysis (mean, std, 95% confidence intervals)
- Generates comparison plots

**`train_delivery.py`**
- Q-learning training loop with tqdm progress tracking
- Checkpoint saving every 50k episodes
- Learning curve generation
- Configurable episodes and epsilon decay

**`server.py`**
- Flask web server for live demo
- Real-time grid visualization
- Traffic pattern highlighting (rush hours)
- Interactive simulation controls

---

## Approach Details

### Two-Component Architecture

**1. Assignment Heuristic (Classical AI)**

When a new order arrives:
1. For each driver, try inserting order at each queue position
2. Evaluate each arrangement via simulation (compute expected arrival times)
3. Score based on: deadline compliance (+3/min early, -15/min late), route distance (-2 per unit), fairness (older orders penalized less)
4. Select driver and position with highest score
5. Apply local hill-climbing (swap adjacent orders if improves score)

**Algorithms:** Greedy optimization, BFS pathfinding, hill-climbing, multi-objective scoring

**2. Navigation Agent (Reinforcement Learning)**

Q-learning agent controlling driver movement:
- **State:** Active driver positions, next destinations, queue lengths, hour of day
- **Action:** Composite movement for all drivers (5³ = 125 actions)
- **Reward:** -1 per active driver (time penalty) + 100 per delivery + on-time bonuses/late penalties + traffic penalties
- **Learning:** Bellman updates with α=0.1, γ=0.95, ε: 1.0 → 0.1

**Algorithms:** Q-learning, epsilon-greedy exploration, temporal difference learning

### Traffic Learning Mechanism

As drivers traverse edges, the system records actual travel times:
```
learned_traffic[edge][hour].append(actual_time)
```

Assignment heuristic uses these learned averages for travel time estimation. Over episodes, both components improve synergistically - RL discovers traffic through movement penalties, assignment uses discovered data for better planning.

**Emergent behavior:** Agent learns rush hour patterns (hours 8-10, 17-19) without being explicitly told, exhibiting different routing at different times of day.

---

## Key Results Summary

### Single-Driver Validation

**Training progression (100k episodes):**
- Episode 10k: -20,000 reward, 55% on-time
- Episode 100k: ~0 reward, 75% on-time
- **Q-table size:** 379,223 unique states
- **Training time:** 14 minutes

### Multi-Driver System

**Scaling results:**
- 1k episodes: 28.7% on-time, 408k states
- 10k episodes: 51.5% on-time, 4M states
- 50k episodes: Training halted at 21M states (memory exhausted)

**Conclusion:** Tabular Q-learning viable for single-driver, requires DQN for multi-driver at scale.

### Baseline Comparison (30 episodes, 1 driver)

| Metric | Random | Greedy | RL (Ours) |
|--------|--------|--------|-----------|
| Avg Reward | -19,723 | +6,791 | -10,622 |
| On-Time % | 2.5% | 91.5% | 70.7% |
| Orders Completed | 10.0 | 50.0 | 33.7 |
| Avg Wait (min) | 230.0 | 12.9 | 66.6 |

**RL agent beats Random by large margin, approaches Greedy despite learning from scratch.**

---

## Algorithms Implemented From Scratch

**Classical AI:**
1. BFS (Breadth-First Search) pathfinding
2. Greedy simulation-based optimization
3. Multi-objective scoring function
4. Hill-climbing (local search)

**Reinforcement Learning:**
1. Q-learning (Bellman equation)
2. Epsilon-greedy exploration
3. Temporal difference learning
4. DQN with experience replay (optional variant)

**Libraries used for building blocks only:**
- NumPy (array operations)
- Matplotlib (plotting)
- PyTorch (neural network layers for DQN only - RL algorithm logic implemented from scratch)

**Not used:** scikit-learn, stable-baselines, or any pre-trained models

---

## Challenges & Lessons Learned

### Hardware Limitations

Single-driver training scaled to 100k+ episodes successfully. Three-driver training exhausted memory around 50k episodes as Q-table exceeded 21 million states. DQN implementation provided as solution for large state spaces via function approximation.

### Environment Design

Multiple iterations required to finalize state representation. Key decision: encoding only active drivers (those with orders) reduced effective state space from billions to millions while preserving learning capability.

### Reward Shaping

Initial hypothesis that progress shaping would accelerate learning. Empirical results showed minimal benefit - rewards improved faster without it. Disabled partway through training. Lesson: validate reward shaping empirically rather than assuming it helps.

---

## Team Contributions

| Member | Primary Contributions | Approx. % |
|--------|----------------------|-----------|
| **Drashti Bhavsar** | `assignment.py` (simulation-based heuristic, BFS pathfinding, queue optimization), Web UI (`server.py`), Environment integration, Project summary, Debugging and testing | 33% |
| **Venkata Sai Udayini Vedantham** | Environment design and specification, `rl_agent.py` (Q-learning implementation, state discretization, training loop), Debugging and testing, Experimental runs | 34% |
| **Elyssa Querubin** | `evaluation.py` (statistical analysis, comparison framework), `baselines.py` (Random and Greedy agents), Project summary, Debugging and testing all agents | 33% |

**All team members contributed to:** Environment design decisions, integration and debugging, demo preparation, documentation and project writeup.

**Individual git commit history demonstrates extent of code contributions per member.**

---

## Results Visualization

All training plots and evaluation results are available in the `results/` directory:

- `RL_with_1_driver/` - Single-driver training results and plots
- `RL_with_3_drivers/` - Multi-driver training results
- `Training_plots_1_driver/` - Learning curves showing improvement over episodes

**Key plots:**
- `comparison.png` - Bar chart comparing Random vs Greedy vs RL
- `learning_curve.png` - Episode reward over time
- `ontime_curve.png` - On-time percentage progression
- `wait_curve.png` - Average wait time progression

---

## Usage Examples

### Training a New Agent

```bash
# Single driver (recommended)
python train_delivery.py train

# Monitor progress - updates every 1000 episodes
# Saves checkpoints every 50k episodes
# Final model: Q_table_final.pkl
```

### Running Live Demo

```bash
# Start web server
python server.py

# Open browser to http://localhost:5000
# Use interactive controls to see system in action
```

### Quick Test Demo

```python
from delivery_env import DeliveryEnvironment
from rl_agent import QLearningAgent

env = DeliveryEnvironment(num_drivers=1)
agent = QLearningAgent()
agent.load('Q_table_final.pkl')

state = env.reset()
for step in range(100):
    state_discrete = agent.discretize_state(env)
    action_int = agent.select_action(state_discrete, epsilon=0)
    actions = agent.decode_action(action_int)
    state, reward, done = env.step(actions)
    if done:
        break

print(f"Completed {len(env.completed_orders)} orders")
```

---

## References

1. Sutton, R.S., & Barto, A.G. (2018). *Reinforcement Learning: An Introduction (2nd ed.)*. MIT Press. Core reference for Q-learning algorithm and epsilon-greedy exploration.

2. Watkins, C.J.C.H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279-292. Original formulation of the Q-learning update rule.

3. Hart, P.E., Nilsson, N.J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107. Foundation for pathfinding algorithms.

---

## AI Disclosure

AI tools were used to assist with code debugging, conceptual understanding, brainstorming, and improving summary document clarity. Web UI implementation utilized AI-assisted development. All core AI algorithms (Q-learning, BFS, greedy search, hill-climbing) were implemented from scratch by team members.

---

## License

This project is submitted as coursework for CS 4100, Northeastern University, Summer 2026.

---

## Acknowledgments

Special thanks to Professor Rajagopalvenkat and TA Andrei for guidance, feedback, and suggestions throughout project development, particularly regarding state space optimization and DQN recommendation for multi-driver scaling.
