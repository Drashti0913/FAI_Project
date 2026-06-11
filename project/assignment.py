"""
assignment.py — Simulation-Based Order Assignment Heuristic (Person 2)

Works with the ORIGINAL DeliveryEnvironment from delivery_env.py.

When a new order arrives, this module decides:
  1. Which driver should receive it
  2. Where in that driver's queue to insert it

Algorithm:
  - For each driver, try inserting the order at every possible queue position
  - Score each arrangement using deadline feasibility, distance, and fairness
  - Pick the driver + position with the highest score
  - Apply local swap optimization (hill climbing) to refine the queue

Key Design:
  - Uses learned_traffic data from env for better time estimates
  - Gets smarter over episodes as RL agent gathers traffic data
  - O(n2) complexity fast even for 10 orders per driver
"""

from typing import List, Tuple
from collections import deque
import numpy as np


# ------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------

def assign_order_to_driver(order, env) -> Tuple[int, int]:
    """
    Decide which driver gets the new order and at which queue position.

    Args:
        order : Order object (has .destination, .deadline, .arrival_time, .id)
        env   : DeliveryEnvironment (has .drivers, .orders, .current_time,
                .graph, .learned_traffic)

    Returns:
        (driver_id, queue_position) best assignment found
    """
    best_score    = float("-inf")
    best_driver   = 0
    best_position = 0

    for driver in env.drivers:
        # Try inserting the new order at every possible position
        max_pos = len(driver.order_queue)
        for pos in range(max_pos + 1):
            score = simulate_driver_with_order(driver, order, pos, env)
            if score > best_score:
                best_score    = score
                best_driver   = driver.id
                best_position = pos

    return best_driver, best_position


# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------

def simulate_driver_with_order(driver, new_order, insert_pos: int, env) -> float:
    """
    Simulate inserting new_order at insert_pos in driver's queue.
    Returns a score higher is better.
    """
    hypothetical_queue = list(driver.order_queue)
    hypothetical_queue.insert(insert_pos, new_order.id)

    # Temporarily include new order so evaluate_queue can find it
    orders = dict(env.orders)
    orders[new_order.id] = new_order

    return evaluate_queue(
        queue      = hypothetical_queue,
        start_node = driver.current_node,
        start_time = env.current_time,
        orders     = orders,
        env        = env,
    )


# ------------------------------------------------------------------
# Queue Evaluation
# ------------------------------------------------------------------

def evaluate_queue(queue: list, start_node: int, start_time: int,
                   orders: dict, env) -> float:
    """
    Score a complete queue arrangement for a driver.

    Components per order:
      Deadline:  +margin x 3    if on time
                 -lateness x 15 if late
      Distance:  -distance x 2
      Fairness:  -age x position x 0.5
    """
    if not queue:
        return 0.0

    score        = 0.0
    current_node = start_node
    current_time = float(start_time)
    hour         = (int(current_time) // 60) % 24

    for position, order_id in enumerate(queue):
        if order_id not in orders:
            continue

        order        = orders[order_id]
        dest         = order.destination
        travel_time  = _estimate_travel_time(current_node, dest, hour, env)
        arrival_time = current_time + travel_time

        # Deadline component
        if arrival_time <= order.deadline:
            score += (order.deadline - arrival_time) * 3.0
        else:
            score -= (arrival_time - order.deadline) * 15.0

        # Distance component
        score -= _manhattan_distance(current_node, dest) * 2.0

        # Fairness component
        age    = current_time - order.arrival_time
        score -= age * position * 0.5

        current_node  = dest
        current_time  = arrival_time
        hour          = (int(current_time) // 60) % 24

    return score


# ------------------------------------------------------------------
# Local Swap Optimization (Hill Climbing)
# ------------------------------------------------------------------

def locally_optimize_queue(driver, env, max_iterations: int = 10):
    """
    Improve driver queue ordering using adjacent-swap hill climbing.
    Modifies driver.order_queue in place.
    """
    if len(driver.order_queue) <= 1:
        return

    for _ in range(max_iterations):
        improved      = False
        current_score = evaluate_queue(
            queue      = driver.order_queue,
            start_node = driver.current_node,
            start_time = env.current_time,
            orders     = env.orders,
            env        = env,
        )

        for i in range(len(driver.order_queue) - 1):
            # Try swap
            driver.order_queue[i], driver.order_queue[i + 1] = (
                driver.order_queue[i + 1], driver.order_queue[i]
            )
            new_score = evaluate_queue(
                queue      = driver.order_queue,
                start_node = driver.current_node,
                start_time = env.current_time,
                orders     = env.orders,
                env        = env,
            )

            if new_score > current_score:
                current_score = new_score
                improved      = True
                break
            else:
                # Revert
                driver.order_queue[i], driver.order_queue[i + 1] = (
                    driver.order_queue[i + 1], driver.order_queue[i]
                )

        if not improved:
            break


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _estimate_travel_time(from_node: int, to_node: int,
                           hour: int, env) -> float:
    """
    Estimate travel time using learned traffic if available.
    Falls back to BFS path + base edge weights.
    """
    if from_node == to_node:
        return 0.0

    path = _bfs_path(from_node, to_node, env.graph)
    if not path:
        return _manhattan_distance(from_node, to_node) * 10.0

    total = 0.0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]

        # Check learned traffic in both edge directions
        learned = env.learned_traffic.get((a, b)) or env.learned_traffic.get((b, a))
        if learned and hour in learned and learned[hour]:
            total += float(np.mean(learned[hour]))
        else:
            total += env.graph.get_edge_cost(a, b, hour)

    return total


def _bfs_path(start: int, goal: int, graph) -> list:
    """BFS shortest path from start to goal."""
    if start == goal:
        return [start]

    visited = {start}
    queue   = deque([[start]])

    while queue:
        path = queue.popleft()
        node = path[-1]
        for nb in graph.get_neighbors(node):
            if nb == goal:
                return path + [nb]
            if nb not in visited:
                visited.add(nb)
                queue.append(path + [nb])
    return []


def _manhattan_distance(node_a: int, node_b: int, grid_size: int = 5) -> int:
    """Manhattan distance between two grid nodes."""
    row_a, col_a = divmod(node_a, grid_size)
    row_b, col_b = divmod(node_b, grid_size)
    return abs(row_a - row_b) + abs(col_a - col_b)


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from delivery_env import DeliveryEnvironment, Order

    env   = DeliveryEnvironment()
    state = env.reset()

    test_order = Order(id=999, destination=18, arrival_time=0, deadline=30)

    driver_id, pos = assign_order_to_driver(test_order, env)
    print(f"Best assignment: driver={driver_id}, position={pos}")

    env.orders[999] = test_order
    env.drivers[driver_id].order_queue.insert(pos, 999)

    score = evaluate_queue(
        queue      = env.drivers[driver_id].order_queue,
        start_node = env.drivers[driver_id].current_node,
        start_time = env.current_time,
        orders     = env.orders,
        env        = env,
    )
    print(f"Queue score: {score:.2f}")

    locally_optimize_queue(env.drivers[driver_id], env)
    print(f"Queue after optimization: {env.drivers[driver_id].order_queue}")
    print("assignment.py working correctly!")