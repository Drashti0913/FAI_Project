"""
contains baseline agents for comparison with the RL agent.

Two baselines:
  RandomAgent             - picks a random neighbor at each step/ movement,
                            orders are assigned by heuristic
  GreedyAgent             - moves each driver one step closer to its next
                            destination using Manhattan distance (greedy local search)
"""
import heapq
import random
from collections import deque
from typing import List, Dict
from delivery_env import DeliveryEnvironment


# movement helper

def _bfs_next_step(start: int, goal: int, graph) -> int:
    """
    Returns the index (within graph.get_neighbors(start)) of the first step
    on the BFS shortest path from start toward goal.
    Returns -1 (stay) if start == goal or no path exists.
    """
    if start == goal:
        return -1

    visited = {start: None}
    queue = deque([start])

    while queue:
        node = queue.popleft()
        for nb in graph.get_neighbors(node):
            if nb not in visited:
                visited[nb] = node
                if nb == goal:
                    # Reconstruct path back to the first step
                    path_node = nb
                    while visited[path_node] != start:
                        path_node = visited[path_node]
                    neighbors = graph.get_neighbors(start)
                    if path_node in neighbors:
                        return neighbors.index(path_node)
                    return -1
                queue.append(nb)

    return -1  # no path found


class RandomAgent:
    """
    picks a completely random neighbor each step.
    Order assignment is still handled by the heuristic in _generate_order().
    """

    def select_actions(self, env: DeliveryEnvironment) -> List[int]:
        """
        Returns one action per driver.
        Action = index into graph.get_neighbors(driver.current_node),
        or len of neighbors (stay) if the driver is idle or has no neighbors.
        """
        actions = []
        for driver in env.drivers:
            if not driver.order_queue:
                actions.append(len(env.graph.get_neighbors(driver.current_node)))  # idle
                continue
            neighbors = env.graph.get_neighbors(driver.current_node)
            if neighbors:
                actions.append(random.randint(0, len(neighbors) - 1))
            else:
                actions.append(len(env.graph.get_neighbors(driver.current_node)))
        return actions

    def __repr__(self):
        return "RandomAgent"

class GreedyAgent:
    """
    At each step, move each active driver one step closer
    to its next queued destination via BFS (shortest path, ignoring traffic).
    """

    def select_actions(self, env: DeliveryEnvironment) -> List[int]:
        """
        Returns one action per driver.
        Follows BFS shortest path to first order's destination.
        Falls back to random if no path exists.
        """
        actions = []
        for driver in env.drivers:
            if not driver.order_queue:
                actions.append(len(env.graph.get_neighbors(driver.current_node)))  # idle
                continue

            dest = env.orders[driver.order_queue[0]].destination
            action = _bfs_next_step(driver.current_node, dest, env.graph)

            if action == -1:
                # Already at destination or no path, stay (delivery auto-triggers)
                actions.append(len(env.graph.get_neighbors(driver.current_node))) # never a valid num
            else:
                actions.append(action)

        return actions

    def __repr__(self):
        return "GreedyAgent"




def run_baseline_episode(agent, env: DeliveryEnvironment) -> Dict:
    """
    Run one full episode with the given agent and return a metrics dict.

    The environment's internal _generate_order() uses the assignment heuristic
    automatically, baseline only controls driver movement.

    Returns:
        {
            "completed":        int,   total orders delivered
            "on_time":          int,   orders delivered before deadline
            "late":             int,   orders delivered after deadline
            "on_time_pct":      float, percentage delivered on time
            "avg_wait":         float, mean (delivery_time - arrival_time)
            "total_reward":     float, cumulative step rewards
            "episode_length":   int,   timesteps elapsed
        }
    """
    env.reset()
    total_reward = 0.0
    done = False

    while not done:
        actions = agent.select_actions(env)
        _, reward, done = env.step(actions)
        total_reward += reward

    # Compute delivery metrics from completed orders
    delivered = [
        env.orders[oid]
        for oid in env.completed_orders
        if env.orders[oid].delivered
    ]

    on_time = sum(
        1 for o in delivered
        if o.delivery_time is not None and o.delivery_time <= o.deadline
    )
    late = len(delivered) - on_time
    avg_wait = (
        sum(o.delivery_time - o.arrival_time for o in delivered) / len(delivered)
        if delivered else 0.0
    )

    return {
        "completed":      len(delivered),
        "on_time":        on_time,
        "late":           late,
        "on_time_pct":    round(on_time / len(delivered) * 100, 1) if delivered else 0.0,
        "avg_wait":       round(avg_wait, 2),
        "total_reward":   round(total_reward, 2),
        "episode_length": env.current_time,
    }


if __name__ == "__main__":
    env = DeliveryEnvironment(num_drivers=4, grid_size=10,
                              traffic_pattern='rush_hour',
                              order_arrival_rate=0.3)

    for AgentClass in [RandomAgent, GreedyAgent]:
        agent = AgentClass()
        results = run_baseline_episode(agent, env)
        print(f"\n{agent}")
        print(f"  Completed:    {results['completed']}")
        print(f"  On-time:      {results['on_time']}  ({results['on_time_pct']}%)")
        print(f"  Late:         {results['late']}")
        print(f"  Avg wait:     {results['avg_wait']} min")
        print(f"  Total reward: {results['total_reward']}")
        print(f"  Episode len:  {results['episode_length']} steps")