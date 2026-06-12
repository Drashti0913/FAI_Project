"""
DQN Agent for Delivery Optimization
Uses neural network to approximate Q-function (handles large state spaces)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQNNetwork(nn.Module):
    """Neural network to approximate Q-function"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """DQN agent for multi-driver delivery navigation"""
    
    def __init__(self, state_size=10, action_size=125, 
                 learning_rate=0.001, gamma=0.95, 
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.999977,
                 batch_size=64, memory_size=10000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        print(f"DQN Agent initialized on {self.device}")
        print(f"Network: {state_size} -> 128 -> 128 -> {action_size}")
    
    def discretize_state(self, env):
        """
        Convert environment to fixed-size numerical vector (NOT tuple!)
        DQN needs numerical vector input for neural network
        """
        state_vector = []
        
        # Encode each driver (3 drivers × 3 features = 9 values)
        for driver in env.drivers:
            if driver.order_queue:
                # Active driver
                next_dest = env.orders[driver.order_queue[0]].destination
                queue_len = len(driver.order_queue)
                
                # Normalize to 0-1 range
                state_vector.extend([
                    driver.current_node / 24.0,  # Normalize node (0-24 → 0-1)
                    next_dest / 24.0,            # Normalize destination
                    min(queue_len, 10) / 10.0    # Normalize queue (cap at 10)
                ])
            else:
                # Idle driver
                state_vector.extend([0.0, 0.0, 0.0])
        
        # Add hour (normalized)
        hour = (env.current_time // 60) % 24
        state_vector.append(hour / 23.0)  # Normalize hour (0-23 → 0-1)
        
        return np.array(state_vector, dtype=np.float32)
    
    def encode_action(self, a0, a1, a2):
        """Encode composite action to single integer"""
        return a0 * 25 + a1 * 5 + a2
    
    def decode_action(self, action_int):
        """Decode integer to individual actions"""
        a0 = action_int // 25
        a1 = (action_int % 25) // 5
        a2 = action_int % 5
        return [a0, a1, a2]
    
    def select_action(self, state, epsilon=None):
        """Epsilon-greedy action selection"""
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            # Explore
            return random.randint(0, self.action_size - 1)
        else:
            # Exploit
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return int(torch.argmax(q_values).item())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """
        Train network on batch of experiences from memory
        This is the DQN learning step
        """
        if len(self.memory) < self.batch_size:
            return  # Not enough samples yet
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save DQN model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Saved DQN model to {filepath}")
    
    def load(self, filepath):
        """Load DQN model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"Loaded DQN model from {filepath}")
    
    def select_actions(self, env):
        """Baseline-compatible interface"""
        state = self.discretize_state(env)
        action_int = self.select_action(state, epsilon=0.0)
        return self.decode_action(action_int)


if __name__ == "__main__":
    # Test DQN
    from delivery_env import DeliveryEnvironment
    
    env = DeliveryEnvironment(num_drivers=3)
    agent = DQNAgent(state_size=10, action_size=125)
    
    state_dict = env.reset()
    state = agent.discretize_state(env)
    
    print(f"State vector: {state}")
    print(f"State shape: {state.shape}")
    
    action_int = agent.select_action(state)
    actions = agent.decode_action(action_int)
    
    print(f"Selected action: {action_int}")
    print(f"Decoded actions: {actions}")
    
    next_state_dict, reward, done = env.step(actions)
    next_state = agent.discretize_state(env)
    
    # Store experience
    agent.remember(state, action_int, reward, next_state, done)
    
    # Try training (needs more samples)
    if len(agent.memory) >= agent.batch_size:
        loss = agent.replay()
        print(f"Training loss: {loss}")
    
    print("DQN test OK!")