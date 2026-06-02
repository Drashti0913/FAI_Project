import random
from typing import Optional, Tuple, Dict, Any, List

from environment.city import City
from environment.order import Order, OrderGenerator
from environment.driver import Driver, make_drivers


# Fixed parameters (from spec)
NUM_DRIVERS = 3
GRID_SIZE = 5
EPISODE_DURATION = 480       # minutes
MAX_ORDERS = 50              # episode ends early if reached
MAX_QUEUE_LENGTH = 5         # cap per driver → 3 × 6 = 18 actions
MISSED_DEADLINE_PENALTY = 100


class DeliveryEnv:
    """
    Gym-style delivery environment.

    Observation is a dict; action is (driver_id, queue_position).
    Agent acts only when a new order arrives.
    """

    def __init__(
        self,
        traffic_pattern: str = "rush_hour",
        order_arrival_rate: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.traffic_pattern = traffic_pattern
        self.order_arrival_rate = order_arrival_rate
        self.seed = seed

        # These are rebuilt on reset()
        self.city: Optional[City] = None
        self.drivers: List[Driver] = []
        self.orders: Dict[int, Order] = {}       # id -> Order
        self.pending_orders: List[int] = []      # unassigned order IDs
        self.order_gen: Optional[OrderGenerator] = None

        self.time: int = 0
        self.completed_count: int = 0
        self.done: bool = False

        # City persists across episodes so traffic learning accumulates
        self._city_initialized = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Start a new episode. City (and learned traffic) persists."""
        if not self._city_initialized:
            self.city = City(
                traffic_pattern=self.traffic_pattern,
                seed=self.seed,
            )
            self._city_initialized = True

        self.drivers = make_drivers(NUM_DRIVERS, GRID_SIZE, seed=self.seed)
        self.orders = {}
        self.pending_orders = []
        self.order_gen = OrderGenerator(
            arrival_rate=self.order_arrival_rate,
            grid_size=GRID_SIZE,
            seed=self.seed,
        )
        self.time = 0
        self.completed_count = 0
        self.done = False

        return self._get_state()

    def step(self, action: Tuple[int, int]) -> Tuple[dict, float, bool, dict]:
        """
        Apply action (driver_id, queue_position) to the latest pending order,
        then advance simulation until the next order arrives or episode ends.

        Returns: (next_state, reward, done, info)
        reward here is 0 (episode reward computed at end).
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        driver_id, queue_pos = action
        driver_id = max(0, min(driver_id, NUM_DRIVERS - 1))
        queue_pos = max(0, min(queue_pos, MAX_QUEUE_LENGTH))

        # Assign the most recently arrived pending order
        if self.pending_orders:
            order_id = self.pending_orders.pop(0)
            order = self.orders[order_id]
            order.assigned_driver = driver_id

            driver = self.drivers[driver_id]
            driver.assign_order(order_id, queue_pos)
            hour = (self.time // 60) % 24
            driver.recompute_route(self.orders, self.city, hour)

        # Advance until next decision point (new order arrives) or end
        self._run_until_decision()

        state = self._get_state()
        # Immediate shaped reward: penalize unbalanced queues and pending backlog
        queue_loads = [d.queue_length() for d in self.drivers]
        reward = -sum(queue_loads) * 0.5 - len(self.pending_orders) * 2.0
        info = self._get_info()

        if self.done:
            reward = self._compute_final_reward()
            info["final_reward"] = reward
            info["completed"] = self.completed_count
            info["missed_deadlines"] = sum(
                1 for o in self.orders.values() if o.is_late
            )

        return state, reward, self.done, info

    # ------------------------------------------------------------------
    # Internal simulation loop
    # ------------------------------------------------------------------

    def _run_until_decision(self):
        """
        Tick the clock until a new order arrives (pending_orders non-empty)
        or the episode terminates.
        """
        while not self.done:
            self.time += 1
            hour = (self.time // 60) % 24

            # Move all drivers one step
            for driver in self.drivers:
                delivered = driver.step(self.time, self.orders, self.city)
                self.completed_count += len(delivered)

            # Check episode termination
            if self.time >= EPISODE_DURATION or self.completed_count >= MAX_ORDERS:
                self.done = True
                return

            # Try to generate a new order
            new_order = self.order_gen.step(self.time)
            if new_order is not None:
                self.orders[new_order.id] = new_order
                self.pending_orders.append(new_order.id)
                return  # hand control back to agent

    # ------------------------------------------------------------------
    # State + reward
    # ------------------------------------------------------------------

    def _get_state(self) -> dict:
        hour = (self.time // 60) % 24
        pending_info = []
        for oid in self.pending_orders:
            o = self.orders[oid]
            pending_info.append({
                "id": oid,
                "destination": o.destination,
                "waiting": self.time - o.arrival_time,
                "time_to_deadline": o.deadline - self.time,
            })

        return {
            "current_time": self.time,
            "hour": hour,
            "drivers": [d.to_state_dict() for d in self.drivers],
            "pending_orders": len(self.pending_orders),
            "pending_info": pending_info,
            "completed_orders": self.completed_count,
            "learned_traffic": self.city.learned_traffic,  # full traffic memory
        }

    def _compute_final_reward(self) -> float:
        total_wait = 0
        missed = 0
        for order in self.orders.values():
            if order.delivered:
                total_wait += order.wait_time
                if order.is_late:
                    missed += 1
        return -float(total_wait) - MISSED_DEADLINE_PENALTY * missed

    def _get_info(self) -> dict:
        delivered = [o for o in self.orders.values() if o.delivered]
        late = [o for o in delivered if o.is_late]
        avg_wait = (
            sum(o.wait_time for o in delivered) / len(delivered)
            if delivered else 0.0
        )
        return {
            "time": self.time,
            "completed": self.completed_count,
            "pending": len(self.pending_orders),
            "avg_wait": round(avg_wait, 2),
            "missed_deadlines": len(late),
            "driver_loads": [d.completed_orders for d in self.drivers],
        }

    # ------------------------------------------------------------------
    # Action space helpers
    # ------------------------------------------------------------------

    def action_space_size(self) -> int:
        return NUM_DRIVERS * (MAX_QUEUE_LENGTH + 1)

    def decode_action(self, action_idx: int) -> Tuple[int, int]:
        """Convert flat action index → (driver_id, queue_position)."""
        driver_id = action_idx // (MAX_QUEUE_LENGTH + 1)
        queue_pos = action_idx % (MAX_QUEUE_LENGTH + 1)
        return driver_id, queue_pos

    def encode_action(self, driver_id: int, queue_pos: int) -> int:
        return driver_id * (MAX_QUEUE_LENGTH + 1) + queue_pos

    def valid_actions(self) -> List[int]:
        """Return action indices where queue_pos <= driver's current queue length."""
        valid = []
        for d in self.drivers:
            max_pos = min(d.queue_length(), MAX_QUEUE_LENGTH)
            for pos in range(max_pos + 1):
                valid.append(self.encode_action(d.id, pos))
        return valid

    # ------------------------------------------------------------------
    # State encoding for Q-table
    # ------------------------------------------------------------------

    def encode_state(self) -> tuple:
        hour = (self.time // 60) % 24
        hour_bucket = 0
        if 8 <= hour <= 10:
            hour_bucket = 1
        elif 17 <= hour <= 19:
            hour_bucket = 2

        # Which driver has the shortest queue (0, 1, or 2)
        min_load_driver = min(range(3), key=lambda i: self.drivers[i].queue_length())

        # Bucket queue lengths: 0, 1, 2, 3+
        queue_buckets = tuple(min(d.queue_length(), 3) for d in self.drivers)

        # Nearest pending order destination bucketed into grid quadrant (0-3)
        dest_quadrant = -1
        if self.pending_orders:
            dest = self.orders[self.pending_orders[0]].destination
            row, col = divmod(dest, 5)
            dest_quadrant = (row // 3) * 2 + (col // 3)  # 4 quadrants

        return (hour_bucket, queue_buckets, dest_quadrant)

    def __repr__(self):
        return (
            f"DeliveryEnv(t={self.time}, completed={self.completed_count}, "
            f"pending={len(self.pending_orders)}, done={self.done})"
        ) 