import random
import heapq
from collections import defaultdict


class City:
    """
    5x5 grid graph representing the city.
    Nodes 0-24, edges connect adjacent nodes horizontally/vertically.
    Each edge has a base travel time (5-15 min) and a traffic multiplier.
    """

    GRID_SIZE = 5

    # Center nodes that get rush-hour congestion
    RUSH_NODES = {6, 7, 11, 12, 13, 17, 18}

    def __init__(self, traffic_pattern="rush_hour", seed=None):
        if seed is not None:
            random.seed(seed)

        self.traffic_pattern = traffic_pattern
        self.edges = {}          # (a, b) -> base_time (a < b always)
        self.adjacency = defaultdict(list)  # node -> [(neighbor, base_time)]

        self._build_grid()
        self._assign_random_weights()
        self._setup_traffic()

        # Traffic learning: learned_traffic[(a,b)][hour] = running avg
        self.learned_traffic = defaultdict(lambda: defaultdict(lambda: None))

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_grid(self):
        G = self.GRID_SIZE
        for r in range(G):
            for c in range(G):
                n = r * G + c
                if c < G - 1:   # horizontal neighbor
                    self._add_edge(n, n + 1)
                if r < G - 1:   # vertical neighbor
                    self._add_edge(n, n + G)

    def _add_edge(self, a, b):
        key = (min(a, b), max(a, b))
        self.edges[key] = None  # weight assigned later

    def _assign_random_weights(self):
        for key in self.edges:
            self.edges[key] = random.randint(5, 15)
        # Build adjacency list
        for (a, b), w in self.edges.items():
            self.adjacency[a].append((b, w))
            self.adjacency[b].append((a, w))

    def _setup_traffic(self):
        """Pre-compute which edges get congestion under the chosen pattern."""
        self.congested_edges = set()  # edges affected by traffic

        if self.traffic_pattern == "random":
            all_edges = list(self.edges.keys())
            k = max(1, int(0.2 * len(all_edges)))
            self.congested_edges = set(random.sample(all_edges, k))

        elif self.traffic_pattern == "rush_hour":
            for (a, b) in self.edges:
                if a in self.RUSH_NODES or b in self.RUSH_NODES:
                    self.congested_edges.add((a, b))

        # uniform: no congested edges

    # ------------------------------------------------------------------
    # Travel time (what actually happens in the simulation)
    # ------------------------------------------------------------------

    def get_travel_time(self, a, b, hour):
        """Actual travel time between adjacent nodes at a given hour."""
        key = (min(a, b), max(a, b))
        base = self.edges[key]
        multiplier = self._get_multiplier(key, hour)
        return base * multiplier

    def _get_multiplier(self, edge_key, hour):
        if self.traffic_pattern == "uniform":
            return 1.0

        if self.traffic_pattern == "random":
            return 1.5 if edge_key in self.congested_edges else 1.0

        if self.traffic_pattern == "rush_hour":
            if edge_key not in self.congested_edges:
                return 1.0
            # Morning rush: 8-10, evening rush: 17-19
            if 8 <= hour <= 10 or 17 <= hour <= 19:
                return 2.0
            return 1.0

        return 1.0

    # ------------------------------------------------------------------
    # Traffic learning (called after each edge traversal)
    # ------------------------------------------------------------------

    def update_learned_traffic(self, a, b, hour, observed_time):
        """Update running average for edge (a,b) at this hour."""
        key = (min(a, b), max(a, b))
        old = self.learned_traffic[key][hour]
        if old is None:
            self.learned_traffic[key][hour] = float(observed_time)
        else:
            self.learned_traffic[key][hour] = 0.9 * old + 0.1 * observed_time

    def get_learned_time(self, a, b, hour):
        """
        Best estimate of travel time using learned data.
        Falls back to base weight if no data yet.
        """
        key = (min(a, b), max(a, b))
        learned = self.learned_traffic[key][hour]
        if learned is not None:
            return learned
        return float(self.edges[key])  # uninformed: use base weight

    # ------------------------------------------------------------------
    # Routing: A* / Dijkstra using learned (or true) weights
    # ------------------------------------------------------------------

    def shortest_path(self, start, goal, hour, use_learned=True):
        """
        Returns (path, estimated_cost) from start to goal.
        path is a list of nodes including start and goal.
        Uses learned traffic weights if use_learned=True.
        """
        if start == goal:
            return [start], 0.0

        dist = {start: 0.0}
        prev = {start: None}
        heap = [(0.0, start)]

        while heap:
            cost, u = heapq.heappop(heap)
            if u == goal:
                break
            if cost > dist.get(u, float("inf")):
                continue
            for v, base_w in self.adjacency[u]:
                if use_learned:
                    w = self.get_learned_time(u, v, hour)
                else:
                    w = float(base_w)
                new_cost = cost + w
                if new_cost < dist.get(v, float("inf")):
                    dist[v] = new_cost
                    prev[v] = u
                    heapq.heappush(heap, (new_cost, v))

        # Reconstruct path
        if goal not in prev and goal != start:
            return [], float("inf")  # no path found

        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path, dist.get(goal, float("inf"))

    def multi_stop_route(self, start, stops, hour, use_learned=True):
        """
        Build a route visiting all stops in order from start.
        Returns flat list of nodes (no duplicates at joins).
        """
        if not stops:
            return [start]

        full_route = [start]
        current = start
        for stop in stops:
            path, _ = self.shortest_path(current, stop, hour, use_learned)
            if path:
                full_route.extend(path[1:])  # skip first node (already in route)
            current = stop
        return full_route

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def node_position(self, n):
        """Return (row, col) of node n."""
        return divmod(n, self.GRID_SIZE)

    def neighbors(self, n):
        return [v for v, _ in self.adjacency[n]]

    def is_rush_hour(self, hour):
        return (8 <= hour <= 10) or (17 <= hour <= 19)

    def __repr__(self):
        return (f"City(pattern={self.traffic_pattern}, "
                f"nodes=25, edges={len(self.edges)})")