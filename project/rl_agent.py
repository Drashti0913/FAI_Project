import numpy as np
import random
import pickle

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.9995):
        self.Q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
    def discretize_state(self, env):
        state_components = []
        for driver in env.drivers:
            if driver.order_queue:
                next_dest = env.orders[driver.order_queue[0]].destination
                queue_len = len(driver.order_queue)
                state_components.extend([driver.current_node, next_dest, queue_len])
            else:
                state_components.extend([-1, -1, 0])
        hour = (env.current_time // 60) % 24
        state_components.append(hour)
        return tuple(state_components)
    
    def encode_action(self, a0, a1, a2):
        return a0 * 25 + a1 * 5 + a2
    
    def decode_action(self, action_int):
        a0 = action_int // 25
        a1 = (action_int % 25) // 5
        a2 = action_int % 5
        return [a0, a1, a2]
    
    def select_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if random.random() < epsilon:
            return random.randint(0, 124)
        if state not in self.Q:
            return random.randint(0, 124)
        return int(np.argmax(self.Q[state]))
    
    def update(self, state, action, reward, next_state, done):
        if state not in self.Q:
            self.Q[state] = np.zeros(125)
        if next_state not in self.Q:
            self.Q[next_state] = np.zeros(125)
        
        old_q = self.Q[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] = old_q + self.alpha * (target - old_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'Q': self.Q, 'epsilon': self.epsilon}, f)
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.Q = data['Q']
            self.epsilon = data['epsilon']


if __name__ == "__main__":
    print("Testing RL agent...")
    from delivery_env import DeliveryEnvironment
    
    env = DeliveryEnvironment()
    agent = QLearningAgent()
    
    state_dict = env.reset()
    state = agent.discretize_state(env)
    print(f"State: {state}")
    
    action_int = agent.select_action(state)
    actions = agent.decode_action(action_int)
    print(f"Actions: {actions}")
    
    next_state_dict, reward, done = env.step(actions)
    next_state = agent.discretize_state(env)
    
    agent.update(state, action_int, reward, next_state, done)
    
    print(f"Reward: {reward:.1f}")
    print(f"Q-table size: {len(agent.Q)}")
    print("\nAgent test OK!")