import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Driver:
    """
    One delivery driver on the 5x5 grid.

    - Moves one node per timestep along its computed route.
    - Queue holds order IDs in delivery priority order.
    - Route is recomputed whenever the queue changes.
    - Actual travel time per edge is recorded for traffic learning.
    """

    id: int
    current_node: int
    order_queue: List[int] = field(default_factory=list)   # list of order IDs
    route: List[int] = field(default_factory=list)         # nodes to traverse
    status: str = "idle"   # "idle" | "delivering"

    # Tracking
    completed_orders: int = 0
    total_distance: int = 0   # edges traversed

    # Internal: node entered at which minute (for traffic learning)
    _edge_enter_time: Optional[int] = None
    _prev_node: Optional[int] = None

    def assign_order(self, order_id: int, position: int):
        """Insert order_id at queue[position]. Clamps position to valid range."""
        position = max(0, min(position, len(self.order_queue)))
        self.order_queue.insert(position, order_id)
        self.status = "delivering"

    def recompute_route(self, orders_by_id: dict, city, hour: int):
        """
        Rebuild self.route to visit all queued orders' destinations in order,
        starting from current_node.
        """
        if not self.order_queue:
            self.route = []
            self.status = "idle"
            return

        destinations = [orders_by_id[oid].destination for oid in self.order_queue]
        self.route = city.multi_stop_route(
            self.current_node, destinations, hour, use_learned=True
        )
        # Drop the first element — it's the current node (already here)
        if self.route and self.route[0] == self.current_node:
            self.route = self.route[1:]

        self.status = "delivering"

    def step(self, current_time: int, orders_by_id: dict, city) -> List[int]:
        """
        Advance driver one step along its route.

        Returns list of order IDs delivered this step.
        Also records edge traversal for traffic learning.
        """
        if not self.route or self.status == "idle":
            return []

        next_node = self.route[0]
        hour = (current_time // 60) % 24

        # Record actual travel time for the edge just traversed
        if self._prev_node is not None:
            actual_time = current_time - self._edge_enter_time
            city.update_learned_traffic(
                self._prev_node, self.current_node, hour, actual_time
            )

        # Move
        self._prev_node = self.current_node
        self._edge_enter_time = current_time
        self.current_node = next_node
        self.route.pop(0)
        self.total_distance += 1

        # Check for deliveries at new position
        delivered_ids = []
        for order_id in list(self.order_queue):
            order = orders_by_id[order_id]
            if order.destination == self.current_node:
                order.delivered = True
                order.delivery_time = current_time
                self.order_queue.remove(order_id)
                self.completed_orders += 1
                delivered_ids.append(order_id)

        # If queue now empty, go idle
        if not self.order_queue:
            self.route = []
            self.status = "idle"

        return delivered_ids

    def queue_length(self):
        return len(self.order_queue)

    def to_state_dict(self):
        """Compact state representation for the RL agent."""
        return {
            "id": self.id,
            "position": self.current_node,
            "queue_length": self.queue_length(),
            "status": self.status,
        }

    def __repr__(self):
        return (
            f"Driver(id={self.id}, node={self.current_node}, "
            f"queue={self.order_queue}, status={self.status})"
        )


def make_drivers(num_drivers=3, grid_size=5, seed=None) -> List[Driver]:
    """Create drivers at random starting positions."""
    if seed is not None:
        random.seed(seed)
    num_nodes = grid_size * grid_size
    return [
        Driver(id=i, current_node=random.randint(0, num_nodes - 1))
        for i in range(num_drivers)
    ]