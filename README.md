# Dynamic Multi-Order Delivery Optimization

**CS 5100 - Artificial Intelligence | Khoury College, Northeastern University**

**Team:** Venkata Sai Udayini Vedantham, Drashti Bhavsar, Elyssa Querubin

---

## Overview

### Problem Statement

Modern delivery services face a fundamental challenge: orders arrive continuously while drivers are already in motion. Existing systems treat routing as a static problem, computing paths once at dispatch time and executing them linearly. This fails to account for the dynamic nature of real delivery operations where new orders keep arriving and must be integrated into active routes without disrupting existing deliveries.

This project addresses: **How should a delivery system manage a continuously growing set of orders for drivers already in motion, deciding both which drivers receive new orders and in what sequence to serve all pending deliveries, while minimizing customer wait time and learning from experience?**

### Our Approach

We developed a two-component hybrid AI system:

1. **Assignment Heuristic (Classical AI):** Simulation-based greedy optimization that selects which driver receives each new order and determines optimal queue positioning through exhaustive evaluation and local hill-climbing search.

2. **Navigation Agent (Reinforcement Learning):** Q-learning agent that learns optimal routing through experience, discovering efficient paths and traffic patterns without prior knowledge.

### Key Results

**Single-Driver Validation (100,000 episodes, 14 minutes):**
- Initial performance: -21,106 reward, 4.5% on-time delivery
- Final performance: +3,325 reward, 85.6% on-time delivery
- **Improvement: 81 percentage points**
- Q-table coverage: 99.77% (379,223 unique states learned)

**Baseline Comparison (3 drivers):**
- Random baseline: 12.5% on-time, -47k reward
- Greedy baseline (BFS): 100% on-time, +8k reward
- RL agent: Demonstrates learning and traffic awareness

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd delivery-optimization
```

2. **Install required packages:**
```bash
pip install numpy matplotlib pygame flask tqdm torch
```

**Package purposes:**
- `numpy`: Array operations and numerical computations
- `matplotlib`: Plotting and visualization
- `pygame`: Real-time simulation visualization
- `flask`: Web UI demo server
- `tqdm`: Progress bars during training
- `torch`: Neural network building blocks (for DQN variant, optional)

### Verify Installation

```bash
python delivery_env.py
```

Should output: "Testing environment with random actions..." and complete successfully.

---

## Reproducing Results

### 1. Single-Driver Q-Learning Training (Recommended First)

Train the RL agent on single-driver validation:

```bash
python train_delivery.py train
```

**Configuration:** Edit lines 11-12 in `train_delivery.py`:
```python
NUM_EPISODES = 100000
EPSILON_DECAY = 0.999977
```

**Expected output:**
- Training progress with tqdm bar
- Checkpoints every 10k episodes
- Final Q-table saved to `Q_table_final.pkl`
- Learning curves saved to `training_100000.png`
- Training time: ~15-20 minutes on standard laptop

**Results to expect:**
- Episode 10k: ~-21k reward, ~5% on-time
- Episode 100k: ~0 reward, ~77% on-time
- Final evaluation: ~85% on-time delivery

### 2. Evaluation and Baseline Comparison

Compare trained RL agent against baselines:

```bash
python evaluation.py
```

**Prerequisites:** `Q_table_final.pkl` must exist (from step 1)

**What this does:**
- Loads trained Q-table
- Runs 30 episodes each: Random, Greedy, RL agents
- Computes statistical metrics (mean, std, 95% CI)
- Generates 4 comparison plots in `results/` folder

**Expected results:**
- Random: ~12% on-time
- Greedy: ~100% on-time (BFS shortest path)
- RL: ~85% on-time (learned policy with traffic awareness)

### 3. Multi-Driver Training (Optional - Requires More Compute)

For 3-driver system:

**Option A: Tabular Q-Learning (High Memory)**
```bash
# Requires 8GB+ RAM
# In delivery_env.py, set num_drivers=3
python train_delivery.py train
```

**Option B: DQN (Lower Memory, Recommended for 3 drivers)**
```bash
# Uses neural network approximation
python train_dqn.py train
```

**Note:** Multi-driver training requires significantly more episodes (500k+) and compute time (several hours).

### 4. Live Web UI Demo

Launch interactive visualization:

```bash
python server.py
```

Then open in browser: `http://localhost:5000`

**Features:**
- Real-time 5×5 city grid visualization
- 3 drivers with smart assignment
- Traffic patterns (rush hour highlighting)
- Live statistics (completed orders, on-time %, avg wait)
- Jump to different hours to see traffic effects

---

## Code Organization

### Core Components

**`delivery_env.py`** (200 lines)
- `DeliveryEnvironment`: Main simulation class
- `CityGraph`: 5×5 grid graph with traffic patterns
- `Driver`: Driver state and queue management
- `Order`: Order representation with deadlines
- Implements action-based step function for RL control
- Tracks traffic data for learning

**`assignment.py`** (150 lines)
- `assign_order_to_driver()`: Simulation-based driver selection
- `evaluate_queue()`: Multi-objective scoring function
- `locally_optimize_queue()`: Hill-climbing queue optimization
- `_bfs_path()`: Breadth-first search for travel time estimation
- Implements classical AI optimization heuristics

**`rl_agent.py`** (120 lines)
- `QLearningAgent`: Tabular Q-learning implementation
- `discretize_state()`: State encoding (active drivers only)
- `select_action()`: Epsilon-greedy action selection
- `update()`: Q-learning Bellman update rule
- Sparse Q-table storage (dictionary-based)

**`dqn_agent.py`** (150 lines) [Optional - for large state spaces]
- `DQNAgent`: Deep Q-Network implementation
- `DQNNetwork`: Neural network architecture (PyTorch)
- Experience replay buffer
- Handles millions of states with fixed memory

### Training and Evaluation

**`train_delivery.py`** (180 lines)
- Training loop for tabular Q-learning
- Progress tracking with tqdm
- Metric logging (reward, on-time %, completions)
- Learning curve generation
- Checkpoint saving

**`train_dqn.py`** (200 lines) [Optional]
- DQN training with experience replay
- Neural network optimization
- For multi-driver large state spaces

**`baselines.py`** (140 lines)
- `RandomAgent`: Random movement baseline
- `GreedyAgent`: BFS shortest-path baseline
- `run_baseline_episode()`: Baseline evaluation framework

**`evaluation.py`** (220 lines)
- Loads trained models
- Runs comparison experiments
- Statistical analysis (mean, std, 95% CI)
- Generates 4 comparison plots

### Visualization

**`server.py`** (350 lines)
- Flask web server for live demo
- Real-time grid visualization
- Traffic pattern highlighting
- Event logging and statistics
- Interactive controls (play/pause, speed, time jump)

**`visualize_pygame.py`** (250 lines) [Optional]
- Pygame-based standalone visualization
- Real-time driver movement
- Order tracking
- Stats panel

### Documentation

**`environment_spec.pdf`**
- Complete environment specification
- State/action space details
- Reward function breakdown

**`methods_reference.pdf`**
- All algorithms explained
- From-scratch vs library usage justification
- Team contributions

---

## Project Structure

```
delivery-optimization/
├── README.md                      # This file
│
├── Core Implementation
│   ├── delivery_env.py            # Simulation environment
│   ├── assignment.py              # Classical AI assignment heuristic
│   ├── rl_agent.py                # Tabular Q-learning agent
│   └── dqn_agent.py               # Deep Q-Network agent (optional)
│
├── Training
│   ├── train_delivery.py         # Q-learning training script
│   └── train_dqn.py               # DQN training script (optional)
│
├── Evaluation
│   ├── baselines.py               # Baseline agent implementations
│   └── evaluation.py              # Comparison experiments and plots
│
├── Visualization
│   ├── server.py                  # Web UI demo
│   └── visualize_pygame.py        # Pygame visualization (optional)
│
├── Documentation
│   ├── environment_spec.pdf       # Environment details
│   ├── methods_reference.pdf      # Algorithms explained
│   └── proposal.pdf               # Original project proposal
│
└── Results (generated after training)
    ├── Q_table_final.pkl          # Trained Q-table
    ├── training_100000.png        # Learning curves
    └── results/                   # Comparison plots
        ├── comparison.png
        ├── learning_curve.png
        ├── ontime_curve.png
        └── wait_curve.png
```

---

## Implementation Highlights

### Algorithms Implemented From Scratch

**Classical AI:**
1. BFS (Breadth-First Search) pathfinding
2. Simulation-based greedy optimization
3. Multi-objective scoring function
4. Hill-climbing (local search)

**Reinforcement Learning:**
1. Q-learning (Bellman equation)
2. Epsilon-greedy exploration
3. Temporal difference learning
4. Experience replay (DQN variant)

**No machine learning libraries used** (no sklearn, no stable-baselines). PyTorch used only for neural network building blocks in DQN variant, per course guidelines.

### Key Design Decisions

**State Space Optimization:** Only encode active drivers (those with orders). Idle driver positions don't contribute to navigation learning. Reduces effective state space from billions to millions.

**Two-Layer Architecture:** Assignment handles strategic decisions (which driver, what order), RL handles tactical execution (how to navigate). Clear separation of concerns.

**Reward Shaping:** Progressive training - strong guidance early (epsilon > 0.3), pure objective later (epsilon < 0.3). Accelerates learning without biasing final policy.

**Traffic Learning:** Agent discovers traffic patterns through experience. No hardcoded rush hours - learns that certain edges are slow at certain times through reward penalties.

---

## Training Hyperparameters

**Q-Learning (Tabular):**
- Learning rate (alpha): 0.1
- Discount factor (gamma): 0.95
- Epsilon: 1.0 → 0.1 (decay: 0.999977 for 100k episodes)
- Episodes: 100k (single-driver), 500k+ (multi-driver)

**DQN (Neural Network):**
- Network: 10 → 64 → 64 → 125
- Learning rate: 0.001
- Batch size: 32
- Memory buffer: 10,000 experiences
- Episodes: 50k recommended

---

## Troubleshooting

**Issue: Training runs out of memory**
- Solution: Use single-driver validation first, or switch to DQN for multi-driver

**Issue: Reward not improving**
- Check: Is epsilon decaying correctly?
- Check: Are deliveries happening? (should see +100 rewards)
- Solution: Increase progress shaping strength (±10 instead of ±0.5)

**Issue: Web UI not displaying**
- Check: Is server running? (`python server.py`)
- Check: Browser console for errors (F12)
- Solution: Verify all dependencies installed

---

## Team Contributions

**Venkata Sai Udayini Vedantham:**
- RL navigation agent implementation (Q-learning and DQN)
- State space design and discretization
- Training loop and evaluation framework
- Environment modifications for RL control

**Drashti Bhavsar:**
- Assignment heuristic implementation
- BFS pathfinding for travel time estimation
- Queue evaluation and optimization
- Simulation-based driver selection

**Elyssa Querubin:**
- Baseline agent implementations (Random, Greedy)
- Statistical analysis and comparison plots
- Web UI development and integration
- Visualization and demo preparation

---

## References

- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach* - Chapters on Search and MDPs
- Watkins, C.J.C.H. & Dayan, P. Q-learning. *Machine Learning*, 8, 279-292 (1992)
- Mnih, V. et al. Human-level control through deep reinforcement learning. *Nature* 518, 529-533 (2015) [DQN]

---

## License

This project is submitted as coursework for CS 5100, Northeastern University.

---

## Acknowledgments

Special thanks to Professor Rajagopalvenkat and TA Andrei for guidance and feedback throughout the project development.
