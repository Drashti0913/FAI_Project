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
    

