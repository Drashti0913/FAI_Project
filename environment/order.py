import random
from dataclasses import dataclass, field
from typing import Optional


DEADLINE_MINUTES = 30


@dataclass
class Order:
    id: int
    destination: int          # target node (0-24)
    arrival_time: int         # simulation minute when order was placed
    deadline: int             # arrival_time + 30
    assigned_driver: Optional[int] = None
    delivered: bool = False
    delivery_time: Optional[int] = None

    # Lifecycle helpers
    @property
    def is_pending(self):
        return self.assigned_driver is None and not self.delivered

    @property
    def is_late(self):
        return self.delivered and self.delivery_time > self.deadline

    @property
    def wait_time(self):
        if self.delivered:
            return self.delivery_time - self.arrival_time
        return None

    @property
    def time_waiting(self):
        """Minutes since arrival (for state observation, call at current time)."""
        return None  # caller must subtract arrival_time from current_time

    def __repr__(self):
        status = "delivered" if self.delivered else (
            f"assigned→{self.assigned_driver}" if self.assigned_driver is not None
            else "pending"
        )
        return f"Order(id={self.id}, dest={self.destination}, {status})"


class OrderGenerator:
    """
    Generates orders dynamically each timestep with 30% probability.
    """

    def __init__(self, arrival_rate=0.3, grid_size=5, seed=None):
        self.arrival_rate = arrival_rate
        self.num_nodes = grid_size * grid_size
        self._next_id = 0
        if seed is not None:
            random.seed(seed)

    def step(self, current_time) -> Optional[Order]:
        """
        Call once per timestep. Returns a new Order or None.
        """
        if random.random() < self.arrival_rate:
            order = Order(
                id=self._next_id,
                destination=random.randint(0, self.num_nodes - 1),
                arrival_time=current_time,
                deadline=current_time + DEADLINE_MINUTES,
            )
            self._next_id += 1
            return order
        return None

    def reset(self):
        self._next_id = 0