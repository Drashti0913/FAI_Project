import random
import math


class RandomBaseline:
    """Assign each order to a random driver, append to end of queue."""

    def act(self, state, env):
        driver_id = random.randint(0, 2)
        queue_pos = env.drivers[driver_id].queue_length()
        return env.encode_action(driver_id, queue_pos)

    def __repr__(self):
        return "RandomBaseline"


class RoundRobinBaseline:
    """Cycle through drivers 0→1→2→0→..., always append to end."""

    def __init__(self):
        self._next = 0

    def act(self, state, env):
        driver_id = self._next % 3
        self._next += 1
        queue_pos = env.drivers[driver_id].queue_length()
        return env.encode_action(driver_id, queue_pos)

    def reset(self):
        self._next = 0

    def __repr__(self):
        return "RoundRobinBaseline"


class GreedyNearestBaseline:
    """
    Assign to the driver whose current position is closest to
    the new order's destination (Manhattan distance on the grid).
    Always appends to end of queue.
    """

    def act(self, state, env):
        if not env.pending_orders:
            return env.encode_action(0, 0)

        order = env.orders[env.pending_orders[0]]
        dest = order.destination

        best_driver = 0
        best_dist = float("inf")

        for driver in env.drivers:
            dr, dc = divmod(driver.current_node, 5)
            or_, oc = divmod(dest, 5)
            dist = abs(dr - or_) + abs(dc - oc)
            if dist < best_dist:
                best_dist = dist
                best_driver = driver.id

        queue_pos = env.drivers[best_driver].queue_length()
        return env.encode_action(best_driver, queue_pos)

    def __repr__(self):
        return "GreedyNearestBaseline"