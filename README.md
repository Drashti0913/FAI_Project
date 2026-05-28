# Intelligent Traffic-Aware Autonomous Delivery Optimization System

## Team Members

* Drashti Bhavsar
* Venkata Sai Udayini Vedantham
* Elyssa Querubin

---

# Project Overview

Modern delivery systems face increasing challenges due to traffic congestion, inefficient routing, dynamic road conditions, and growing delivery demands. Traditional navigation systems often prioritize the shortest distance, but the shortest route may not always minimize delivery time under real-world traffic conditions.

This project aims to develop an intelligent traffic-aware autonomous delivery optimization system that combines classical AI search algorithms with Reinforcement Learning (RL). The system simulates a city road network where an autonomous delivery vehicle computes efficient delivery routes while dynamically adapting its driving behavior based on surrounding traffic conditions.

The project integrates:

* Graph-based pathfinding algorithms
* Dynamic traffic simulation
* Reinforcement Learning for autonomous driving decisions
* Multi-destination delivery optimization

---

# Objectives

The primary objectives of this project are:

* Implement classical AI search algorithms for delivery route optimization
* Simulate dynamic traffic-aware city environments
* Develop an RL-based autonomous driving agent
* Compare algorithm performance under varying traffic conditions
* Minimize delivery time, congestion delay, and travel cost

---

# Proposed Features

## Route Optimization

The system will compute optimal delivery routes using:

* A* Search
* Dijkstra’s Algorithm
* Greedy Best-First Search

A* Search will serve as the primary route planner due to its heuristic-based efficiency.

## Reinforcement Learning-Based Driving Decisions

A Reinforcement Learning agent will control real-time vehicle behavior, including:

* Lane changing
* Overtaking slower vehicles
* Speed adjustment
* Collision avoidance

The RL agent will learn optimal driving strategies through interaction with a simulated traffic environment.

## Dynamic Traffic Simulation

Traffic conditions and congestion levels will change dynamically during execution, requiring adaptive route replanning and intelligent driving behavior.

---

# Technologies and Tools

* Python
* NetworkX
* NumPy
* Matplotlib
* Gymnasium / OpenAI Gym
* Stable-Baselines3

Potential simulation tools:

* CARLA Simulator
* SUMO Traffic Simulator

---

# Repository Structure

```text
traffic-aware-delivery-optimization/
│
├── README.md
├── requirements.txt
├── main.py
│
├── src/
│   ├── algorithms/
│   ├── rl/
│   ├── simulation/
│   ├── evaluation/
│   └── visualization/
│
├── data/
├── results/
└── docs/
```

---

# Current Progress

### Completed

* Project proposal finalized
* Initial system architecture designed
* Repository structure planned

### In Progress

* Graph representation of city road network
* A* Search implementation
* Traffic simulation module

### Planned

* Reinforcement Learning environment
* RL driving agent training
* Performance evaluation and comparison

---

# Installation

Clone the repository:

```bash
git clone <repository-link>
cd traffic-aware-delivery-optimization
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Mac/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

The main execution file will be:

```bash
python main.py
```

Additional execution instructions will be added as development progresses.

---

# Evaluation Metrics

The system will be evaluated using:

* Total delivery time
* Route efficiency
* Traffic delay
* Collision count
* Computational performance
* RL reward convergence

---

# References

1. A* Search Algorithm
   https://www.geeksforgeeks.org/a-search-algorithm/

2. Dijkstra’s Algorithm
   https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/

3. Red Blob Games Pathfinding Guide
   https://www.redblobgames.com/pathfinding/a-star/introduction.html

4. CARLA Simulator
   https://carla.org/

5. Stable-Baselines3 Documentation
   https://stable-baselines3.readthedocs.io/en/master/
