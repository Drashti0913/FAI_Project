import numpy as np
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class Order:
    """
    It represents an Order.
    """
    id: int
    destination: int
    arrival_time: int
    deadline: int
    assigned_driver: Optional[int] = None
    delivered: bool = False
    delivery_time: Optional[int] = None

    def is_late(self, current_time: int) -> bool:
        return current_time > self.deadline
    
@dataclass
class Driver:
    """
    It represents a Driver.
    """
    id: int
    current_node: int
    order_queue: List[int]
    route: List[int]
    status: str = 'idle' # 'idle' or 'delivering'

    def add_order(self, order_id: int, position: int):
        """
        It adds an order to the driver's queue at a specific position."""
        self.order_queue.insert(position, order_id)
        
    def complete_delivery(self) -> int:
        """
        It removes and return the first order from the queue."""
        if self.order_queue:
            return self.order_queue.pop(0)
        return None
    
class CityGraph:
    """
    It represnts the city as a graph.
    """

    def __init__(self, grid_size: int =5):
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size
        self.edges = {}
        self.traffic_multipliers = defaultdict(lambda: defaultdict(lambda: 1.0))  # Default traffic multiplier is 1.0 (no traffic)
        self._build_grid_graph()

    def _build_grid_graph(self):
        """
        Creates a grid graph with horizontal and vertical edges.
        """
        #horizontal edges
        for row in range(self.grid_size):
            for col in range(self.grid_size - 1):
                node1 = row * self.grid_size + col
                node2 = row * self.grid_size + col + 1
                # random base travel time between 5-15 minutes
                travel_time = random.randint(5, 15)
                self.edges[(node1, node2)] = travel_time
                self.edges[(node2, node1)] = travel_time  

        #vertical edges
        for row in range(self.grid_size - 1):
            for col in range(self.grid_size):
                node1 = row * self.grid_size + col
                node2 = (row + 1) * self.grid_size + col
                travel_time = random.randint(5, 15)
                self.edges[(node1, node2)] = travel_time
                self.edges[(node2, node1)] = travel_time

    def set_traffic_pattern(self, pattern: str = 'rush_hour'):
        """
        Sets the traffic multipliers based on pattern.
        """  
        center_nodes = self._get_center_nodes()

        if pattern == 'rush_hour':
            #Morning and evening rush (hours 8-10 and 17-19)
            for edge in self.edges.keys():
                if self._edge_near_center(edge, center_nodes):
                    for hour in [8,9,10,17,18,19]:
                        self.traffic_multipliers[edge][hour] = 2.0
                    
        elif pattern == 'uniform':
            #no difference in traffic 
            pass

        elif pattern == 'random':
            #random congestion on 20% of edges
            for edge in random.sample(list(self.edges.keys()), len(self.edges) // 5):
                for hour in range(24):
                    self.traffic_multipliers[edge][hour] = 1.5

    def _get_center_nodes(self) -> set:
        """
        Gets the nodes in center of the grid.
        """
        center = self.grid_size // 2
        return {
            center *self.grid_size + center,
            center * self.grid_size + (center - 1),
            (center - 1) * self.grid_size + center,
            (center - 1) * self.grid_size + (center - 1)
        }

    def _edge_near_center(self, edge: Tuple[int, int], center_nodes: set) -> bool:
        """
        checks if edges connects to center nodes.
        """
        return edge[0] in center_nodes or edge[1] in center_nodes
    
    def get_edge_cost(self, node1: int, node2: int, hour: int) -> float:
        """
        Gets the travel time for edge at a specific hour.
        """
        edge = (node1, node2)
        if edge not in self.edges:
            return float('inf')
        
        base_time = self.edges[edge]
        multiplier = self.traffic_multipliers[edge][hour]
        return base_time * multiplier
    
    def get_neighbors(self, node: int) -> List[int]:
        """
        Get the neighboring nodes.
        Only checks once since the edges are bidirectional.
        """
        neighbors = []
        for (n1, n2) in self.edges.keys():
            if n1 == node:
                neighbors.append(n2)
            
        return neighbors
    
class DeliveryEnvironment:
    """
    The main environment simulation.
    """

    def __init__(self, 
                 num_drivers: int = 3,
                 grid_size: int = 5,
                 traffic_pattern: str = 'rush_hour',
                 order_arrival_rate: float = 0.3):
        self.graph = CityGraph(grid_size)
        self.graph.set_traffic_pattern(traffic_pattern)

        self.num_drivers = num_drivers
        self.drivers = self._initialize_drivers()

        self.orders = {}
        self.pending_orders = []
        self.completed_orders = []

        self.current_time = 0
        self.order_counter = 0
        self.order_arrival_rate = order_arrival_rate

        #for the traffic learning
        self.learned_traffic = defaultdict(lambda: defaultdict(list))

    def _initialize_drivers(self) -> List[Driver]:
        """
        Initializes drivers at random locations.
        """
        drivers = []
        for i in range(self.num_drivers):
            start_node = random.randint(0, self.graph.num_nodes - 1)
            drivers.append(Driver(
                id=i,
                current_node=start_node,
                order_queue=[],
                route=[],
                status='idle'))
        return drivers
        
    def reset(self) -> dict:
        """
        Resets the environment for new episode.
        """
        self.drivers = self._initialize_drivers()
        self.orders = {}
        self.pending_orders = []
        self.completed_orders = []
        self.current_time = 0
        self.order_counter = 0

        return self.get_state()
        
    def get_state(self) -> dict:
        """
        Gets the current state of environment.
        """

        return {
            'current_time': self.current_time,
            'hour': (self.current_time // 60) % 24,
            'drivers': [
                {
                    'id': d.id,
                    'position': d.current_node,
                    'queue_length': len(d.order_queue),
                    'status': d.status
                } for d in self.drivers
            ],
            'pending_orders': len(self.pending_orders),
            'completed_orders': len(self.completed_orders)
        }
    
    def step(self) -> Tuple[dict, float, bool]:
        """
        Advances the simulation by one time step.
        Returns: (state, reward, done)
        """
        #generate new orders
        if random.random() < self.order_arrival_rate:
            self._generate_order()

        for driver in self.drivers:
            if driver.route and driver.status == 'delivering':
                next_node = driver.route.pop(0)

                #recording the travel time for learning
                hour = (self.current_time // 60) % 24
                edge = (driver.current_node, next_node)
                actual_time = self.graph.get_edge_cost(driver.current_node, next_node, hour)
                self.learned_traffic[edge][hour].append(actual_time)
                driver.current_node = next_node

                #check if delivery destination is reached
                if driver.order_queue and driver.current_node == self.orders[driver.order_queue[0]].destination:
                    self._complete_delivery(driver)

        #updating the time
        self.current_time += 1

        #reward
        reward = self._calculate_reward()
        done = self._is_episode_done()

        return self.get_state(), reward, done
    
    def _generate_order(self):
        """
        Generate a new random order.
        """
        order = Order(
            id = self.order_counter,
            destination=random.randint(0, self.graph.num_nodes - 1),
            arrival_time=self.current_time,
            deadline=self.current_time + 30 #deadline of 30 min
        )
        self.orders[order.id] = order
        self.pending_orders.append(order.id)
        self.order_counter += 1

    def assign_order(self, order_id: int, driver_id: int, position: int):
        """
        Assign order to driver at specific position in queue.
        """
        if order_id not in self.pending_orders:
            return False
        
        order = self.orders[order_id]
        order.assigned_driver = driver_id
        self.drivers[driver_id].add_order(order_id, position)
        self.pending_orders.remove(order_id)

        #simple route for the driver
        self._compute_simple_route(self.drivers[driver_id])

        return True
    
    def _compute_simple_route(self, driver: Driver):
        """
        Compute simple direct route to all the queued orders.
        """
        if not driver.order_queue:
            driver.route = []
            driver.status = 'idle'
            return
        
        route = []
        current = driver.current_node

        for order_id in driver.order_queue:
            destination = self.orders[order_id].destination
            
            if current != destination:
                route.append(destination)
            current = destination

        driver.route = route
        if route:
            driver.status = 'delivering'
        else:
            driver.status = 'idle'

    def _complete_delivery(self, driver: Driver):
        """
        Complete the current delivery for a driver.
        """
        order_id = driver.complete_delivery()
        if order_id is not None:
            order = self.orders[order_id]
            order.delivered = True
            order.delivery_time = self.current_time
            self.completed_orders.append(order_id)
            driver.status = 'idle' if not driver.order_queue else 'delivering'

    def _calculate_reward(self) -> float:
        """
        Calculate the reward for current state.
        """

        total_wait = 0
        missed_deadlines = 0

        for order_id in self.completed_orders:
            order = self.orders[order_id]
            wait_time = order.delivery_time - order.arrival_time
            total_wait += wait_time

            if order.is_late(order.delivery_time):
                missed_deadlines += 1

        reward = -total_wait - (missed_deadlines * 100)  # Penalize wait time and missed deadlines
        return reward
    
    def _is_episode_done(self) -> bool:
        """
        checks if episode is done.
        """

        #episode ends after 8 simulated hours (480 min) or till some 50 order limit

        return self.current_time >= 480 or len(self.completed_orders) >= 50
    
    def get_learned_traffic_estimate(self, edge: Tuple[int, int], hour: int) -> float:
        """
        Gets the learned average traffic for edge at hour.
        """
        if edge in self.learned_traffic and hour in self.learned_traffic[edge]:
            times = self.learned_traffic[edge][hour]
            if times:
                return np.mean(times) if times else self.graph.get_edge_cost(edge[0], edge[1], hour)
        return self.graph.get_edge_cost(edge[0], edge[1], hour)
        
if __name__ == "__main__":
    env = DeliveryEnvironment(
        num_drivers=3,
        grid_size=5,
        traffic_pattern='rush_hour',
        order_arrival_rate=0.3
    )

    #running a simple simulation
    state = env.reset()
    print("Initial State:", state)
    print(f"City: {env.graph.num_nodes} nodes, {len(env.graph.edges)} edges")
    print()

    for step in range(100):
        state, reward, done = env.step()

        if env.pending_orders:
            order_id = env.pending_orders[0]
            driver_id = random.randint(0, env.num_drivers - 1)
            position = len(env.drivers[driver_id].order_queue)
            env.assign_order(order_id, driver_id, position)
        
        if step % 20 == 0:
            print(f"Step {step}: Time {state['current_time']}, "
                  f"Completed: {len(env.completed_orders)}, "
                  f"Pending: {len(env.pending_orders)}")
        
        if done:
            print(f"\nEpisode finished at step {step}")
            break

    print(f"\nFinal reward: {reward}")
    print(f"Completed orders: {len(env.completed_orders)}")
   
    #stats
    if env.completed_orders:
        late_orders = sum(1 for oid in env.completed_orders 
                         if env.orders[oid].delivery_time > env.orders[oid].deadline)
        avg_time = sum(env.orders[oid].delivery_time - env.orders[oid].arrival_time 
                      for oid in env.completed_orders) / len(env.completed_orders)
        
        print(f"Late orders: {late_orders}")
        print(f"On-time rate: {100*(len(env.completed_orders)-late_orders)/len(env.completed_orders):.1f}%")
        print(f"Average delivery time: {avg_time:.1f} minutes")
 

    



                    