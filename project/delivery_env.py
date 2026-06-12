import numpy as np
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Order:
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
    id: int
    current_node: int
    order_queue: List[int]
    route: List[int]
    status: str = 'idle'
    
    def add_order(self, order_id: int):
        self.order_queue.append(order_id)
    
    def complete_delivery(self) -> int:
        if self.order_queue:
            return self.order_queue.pop(0)
        return None

class CityGraph:
    def __init__(self, grid_size: int = 5):
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size
        self.edges = {}
        self.traffic_multipliers = defaultdict(lambda: defaultdict(lambda: 1.0))
        self._build_grid_graph()
        
    def _build_grid_graph(self):
        for row in range(self.grid_size):
            for col in range(self.grid_size - 1):
                node1 = row * self.grid_size + col
                node2 = row * self.grid_size + col + 1
                travel_time = random.randint(5, 15)
                self.edges[(node1, node2)] = travel_time
                self.edges[(node2, node1)] = travel_time
        
        for row in range(self.grid_size - 1):
            for col in range(self.grid_size):
                node1 = row * self.grid_size + col
                node2 = (row + 1) * self.grid_size + col
                travel_time = random.randint(5, 15)
                self.edges[(node1, node2)] = travel_time
                self.edges[(node2, node1)] = travel_time
    
    def set_traffic_pattern(self, pattern: str = 'rush_hour'):
        center_nodes = self._get_center_nodes()
        
        if pattern == 'rush_hour':
            for edge in self.edges.keys():
                if self._edge_near_center(edge, center_nodes):
                    for hour in [8, 9, 10, 17, 18, 19]:
                        self.traffic_multipliers[edge][hour] = 2.0
        elif pattern == 'random':
            edges_list = list(self.edges.keys())
            for edge in random.sample(edges_list, len(edges_list) // 5):
                for hour in range(24):
                    self.traffic_multipliers[edge][hour] = 1.5
    
    def _get_center_nodes(self):
        center = self.grid_size // 2
        return {
            center * self.grid_size + center,
            center * self.grid_size + (center - 1),
            (center - 1) * self.grid_size + center,
            (center - 1) * self.grid_size + (center - 1)
        }
    
    def _edge_near_center(self, edge, center_nodes):
        return edge[0] in center_nodes or edge[1] in center_nodes
    
    def get_edge_cost(self, node1: int, node2: int, hour: int) -> float:
        edge = (node1, node2)
        if edge not in self.edges:
            return float('inf')
        return self.edges[edge] * self.traffic_multipliers[edge][hour]
    
    def get_neighbors(self, node: int) -> List[int]:
        neighbors = []
        for (n1, n2) in self.edges.keys():
            if n1 == node:
                neighbors.append(n2)
        return neighbors


class DeliveryEnvironment:
    def __init__(self, num_drivers=1, grid_size=5, traffic_pattern='rush_hour', 
                 order_arrival_rate=0.3, use_progress_shaping=True):
        self.graph = CityGraph(grid_size)
        self.graph.set_traffic_pattern(traffic_pattern)
        self.num_drivers = num_drivers
        self.drivers = self._initialize_drivers()
        self.orders = {}
        self.completed_orders = []
        self.orders_delivered_this_step = []
        self.current_time = 0
        self.order_counter = 0
        self.order_arrival_rate = order_arrival_rate
        self.learned_traffic = defaultdict(lambda: defaultdict(list))
        self.use_progress_shaping = use_progress_shaping
    
    def _initialize_drivers(self):
        drivers = []
        for i in range(self.num_drivers):
            drivers.append(Driver(i, random.randint(0, self.graph.num_nodes - 1), [], [], 'idle'))
        return drivers
    
    def reset(self):
        self.drivers = self._initialize_drivers()
        self.orders = {}
        self.completed_orders = []
        self.orders_delivered_this_step = []
        self.current_time = 0
        self.order_counter = 0
        return self.get_state()
    
    def get_state(self):
        return {
            'current_time': self.current_time,
            'hour': (self.current_time // 60) % 24,
            'drivers': [{'id': d.id, 'position': d.current_node, 'queue_length': len(d.order_queue)} for d in self.drivers],
            'completed_orders': len(self.completed_orders)
        }
    
    def step(self, actions):
        prev_nodes = [d.current_node for d in self.drivers]
        self.orders_delivered_this_step = []
        
        if random.random() < self.order_arrival_rate:
            self._generate_order()
        
        for i, driver in enumerate(self.drivers):
            if driver.order_queue:
                action = actions[i]
                neighbors = self.graph.get_neighbors(driver.current_node)
                
                if action < len(neighbors):
                    next_node = neighbors[action]
                    hour = (self.current_time // 60) % 24
                    edge = (driver.current_node, next_node)
                    actual_time = self.graph.get_edge_cost(driver.current_node, next_node, hour)
                    self.learned_traffic[edge][hour].append(actual_time)
                    driver.current_node = next_node
                
                if driver.order_queue:
                    first_order = self.orders[driver.order_queue[0]]
                    if driver.current_node == first_order.destination:
                        self._complete_delivery(driver)
        
        self.current_time += 1
        reward = self._calculate_step_reward(prev_nodes)
        done = self._is_episode_done()
        
        return self.get_state(), reward, done
    
    def _generate_order(self):
        order = Order(self.order_counter, random.randint(0, self.graph.num_nodes - 1), 
                     self.current_time, self.current_time + 30)
        self.orders[order.id] = order
        self.order_counter += 1
        
        def manhattan_distance(n1, n2):
            r1, c1 = n1 // 5, n1 % 5
            r2, c2 = n2 // 5, n2 % 5
            return abs(r2-r1) + abs(c2-c1)
        
        best_driver = min(self.drivers, key=lambda d: manhattan_distance(d.current_node, order.destination) + len(d.order_queue)*10)
        best_driver.add_order(order.id)
        order.assigned_driver = best_driver.id
    
    def _complete_delivery(self, driver):
        order_id = driver.complete_delivery()
        if order_id:
            order = self.orders[order_id]
            order.delivered = True
            order.delivery_time = self.current_time
            self.completed_orders.append(order_id)
            self.orders_delivered_this_step.append(order_id)
    
    def _calculate_step_reward(self, prev_nodes):
        reward = 0
        
        num_active = sum(1 for d in self.drivers if d.order_queue)
        reward -= num_active
        
        hour = (self.current_time // 60) % 24
        for i, driver in enumerate(self.drivers):
            if prev_nodes[i] != driver.current_node:
                edge = (prev_nodes[i], driver.current_node)
                if edge in self.graph.traffic_multipliers:
                    mult = self.graph.traffic_multipliers[edge][hour]
                    if mult > 1.0:
                        reward -= (mult - 1.0) * 3
        
        for order_id in self.orders_delivered_this_step:
            order = self.orders[order_id]
            reward += 100
            wait = order.delivery_time - order.arrival_time
            deadline_window = order.deadline - order.arrival_time
            if wait <= deadline_window:
                reward += (deadline_window - wait) * 2
            else:
                reward -= (wait - deadline_window) * 10
        
        if self.use_progress_shaping:
            for i, driver in enumerate(self.drivers):
                if driver.order_queue:
                    next_dest = self.orders[driver.order_queue[0]].destination
                    r1, c1 = prev_nodes[i] // 5, prev_nodes[i] % 5
                    r2, c2 = driver.current_node // 5, driver.current_node % 5
                    r3, c3 = next_dest // 5, next_dest % 5
                    prev_dist = abs(r3-r1) + abs(c3-c1)
                    curr_dist = abs(r3-r2) + abs(c3-c2)
                    if curr_dist < prev_dist:
                        reward += 5
                    elif curr_dist > prev_dist:
                        reward -= 5
        
        return reward
    
    def _is_episode_done(self):
        return self.current_time >= 480 or len(self.completed_orders) >= 50
    

if __name__ == "__main__":
    print("Testing environment...")
    env = DeliveryEnvironment()
    state = env.reset()
    print(f"Created: {env.graph.num_nodes} nodes")
    
    for i in range(50):
        actions = [random.randint(0, 4) for _ in range(1)]
        state, reward, done = env.step(actions)
        if i % 10 == 0:
            print(f"Step {i}: Reward={reward:.1f}")
        if done:
            break
    
    print("Test OK!")