import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from delivery_env import DeliveryEnvironment
from rl_agent import QLearningAgent

# ============================================================================
# CHANGE THESE TWO LINES ONLY:
# ============================================================================
NUM_EPISODES = 1000  # How many episodes to train
EPSILON_DECAY = 0.9977   # Decay rate (use table below)

# Decay rate reference:
# 1,000 episodes    → 0.9977
# 10,000 episodes   → 0.99977
# 100,000 episodes  → 0.999977     ← Default
# 500,000 episodes  → 0.9999954
# 1,000,000 episodes → 0.9999977
# 5,000,000 episodes → 0.99999954
# ============================================================================

train_flag = 'train' in sys.argv

def train_qlearning():
    """Train Q-learning agent"""
    env = DeliveryEnvironment(grid_size=5, order_arrival_rate=0.3, use_progress_shaping=True)
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=EPSILON_DECAY)
    
    episode_rewards = []
    episode_on_time = []
    episode_completions = []
    episode_lengths = []
    
    progress_shaping_disabled = False
    progress_shaping_threshold = 0.3
    
    print("="*70)
    print(f"Q-Learning Training: {NUM_EPISODES} episodes")
    print(f"Gamma=0.95, Epsilon=1.0->0.1, Decay={EPSILON_DECAY}")
    print("="*70)
    
    start_time = time.time()
    
    for episode in tqdm(range(NUM_EPISODES), desc="Training", unit="ep"):
        state_dict = env.reset()
        state = agent.discretize_state(env)
        episode_reward = 0
        steps = 0
        
        for step in range(500):
            action_int = agent.select_action(state, agent.epsilon)
            actions = agent.decode_action(action_int)
            next_state_dict, reward, done = env.step(actions)
            next_state = agent.discretize_state(env)
            agent.update(state, action_int, reward, next_state, done)
            episode_reward += reward
            state = next_state
            steps += 1
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        episode_completions.append(len(env.completed_orders))
        
        if env.completed_orders:
            on_time = sum(1 for oid in env.completed_orders if env.orders[oid].delivery_time <= env.orders[oid].deadline)
            episode_on_time.append((on_time / len(env.completed_orders)) * 100)
        else:
            episode_on_time.append(0)
        
        agent.decay_epsilon()
        
        if not progress_shaping_disabled and agent.epsilon < progress_shaping_threshold:
            env.use_progress_shaping = False
            progress_shaping_disabled = True
            tqdm.write(f"\n>>> Episode {episode}: ε={agent.epsilon:.4f} < {progress_shaping_threshold}")
            tqdm.write(f">>> Progress shaping DISABLED\n")
        
        if episode % 10000 == 0 and episode > 0:
            elapsed = time.time() - start_time
            window = min(10000, episode)
            tqdm.write(f"\n{'='*70}")
            tqdm.write(f"Checkpoint @ Episode {episode}/{NUM_EPISODES}")
            tqdm.write(f"Avg reward (last {window}):   {np.mean(episode_rewards[-window:]):.1f}")
            tqdm.write(f"Avg on-time (last {window}):  {np.mean(episode_on_time[-window:]):.1f}%")
            tqdm.write(f"Q-states: {len(agent.Q)}, ε: {agent.epsilon:.6f}, Time: {elapsed/60:.1f}m")
            tqdm.write(f"{'='*70}\n")
        
        if episode > 0 and episode % 50000 == 0:
            agent.save(f'Q_table_{episode}.pkl')
            tqdm.write(f">>> Saved: Q_table_{episode}.pkl\n")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"Training Complete!")
    print("="*70)
    print(f"Episodes:                {NUM_EPISODES}")
    print(f"Time:                    {total_time/3600:.2f}h ({total_time/60:.1f}m)")
    print(f"Speed:                   {NUM_EPISODES/total_time:.2f} eps/s")
    print(f"Unique states:           {len(agent.Q)}")
    print(f"Final epsilon:           {agent.epsilon:.6f}")
    print(f"\nFinal 1000 episodes:")
    print(f"  Average reward:        {np.mean(episode_rewards[-1000:]):.1f}")
    print(f"  Average on-time:       {np.mean(episode_on_time[-1000:]):.1f}%")
    print(f"  Average completed:     {np.mean(episode_completions[-1000:]):.1f}")
    print(f"  Average steps:         {np.mean(episode_lengths[-1000:]):.1f}")
    print("="*70)
    
    agent.save('Q_table_final.pkl')
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0,0].plot(episode_rewards, alpha=0.3)
    window = 1000
    if len(episode_rewards) > window:
        ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0,0].plot(range(window-1, len(episode_rewards)), ma, 'r-', linewidth=2, label='MA(1000)')
        axes[0,0].legend()
    axes[0,0].set_title('Episode Reward')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(episode_on_time, alpha=0.3, color='green')
    if len(episode_on_time) > window:
        ma = np.convolve(episode_on_time, np.ones(window)/window, mode='valid')
        axes[0,1].plot(range(window-1, len(episode_on_time)), ma, 'r-', linewidth=2, label='MA(1000)')
        axes[0,1].legend()
    axes[0,1].set_title('On-Time %')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(episode_completions, alpha=0.3, color='purple')
    axes[1,0].set_title('Orders Completed')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(episode_lengths, alpha=0.3, color='orange')
    axes[1,1].set_title('Episode Length')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Results ({NUM_EPISODES} episodes)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'training_{NUM_EPISODES}.png', dpi=150)
    print(f"\nSaved: training_{NUM_EPISODES}.png")

def evaluate_qlearning():
    """Evaluate trained agent"""
    env = DeliveryEnvironment(use_progress_shaping=False)
    agent = QLearningAgent()
    agent.load('Q_table_final.pkl')
    
    num_eval = 10000
    episode_rewards = []
    episode_on_time = []
    episode_lengths = []
    
    states_encountered = set()
    unseen = 0
    qtable_acts = 0
    random_acts = 0
    
    print(f"\nEvaluating (10000 episodes, NO exploration)...\n")
    start_time = time.time()
    
    for episode in tqdm(range(num_eval), desc="Evaluating", unit="ep"):
        state_dict = env.reset()
        state = agent.discretize_state(env)
        ep_reward = 0
        steps = 0
        
        for step in range(500):
            states_encountered.add(state)
            
            if state in agent.Q:
                action_int = int(np.argmax(agent.Q[state]))
                qtable_acts += 1
            else:
                action_int = np.random.randint(0, 125)
                random_acts += 1
                unseen += 1
            
            actions = agent.decode_action(action_int)
            next_state_dict, reward, done = env.step(actions)
            next_state = agent.discretize_state(env)
            
            ep_reward += reward
            state = next_state
            steps += 1
            if done:
                break
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        
        if env.completed_orders:
            on_time = sum(1 for oid in env.completed_orders if env.orders[oid].delivery_time <= env.orders[oid].deadline)
            episode_on_time.append((on_time / len(env.completed_orders)) * 100)
        else:
            episode_on_time.append(0)
    
    eval_time = time.time() - start_time
    total_acts = qtable_acts + random_acts
    
    print("\n" + "="*70)
    print(f"  Evaluation Results  (episodes=10000)")
    print("="*70)
    print(f"  Average reward over 10000 episodes   : {np.mean(episode_rewards):.2f}")
    print(f"  Average on-time percentage           : {np.mean(episode_on_time):.2f}%")
    print(f"  Average episode length (steps)       : {np.mean(episode_lengths):.2f}")
    print(f"  Total user time for 10000 episodes   : {eval_time:.2f}s")
    print(f"  Unique states in Q-table (training)  : {len(agent.Q)}")
    print(f"  Unique states during evaluation      : {len(states_encountered)}")
    print(f"  Unique unseen states during eval     : {unseen}")
    print(f"  % actions using Q-table              : {100*qtable_acts/total_acts:.2f}%")
    print(f"  % random actions (unseen states)     : {100*random_acts/total_acts:.2f}%")
    print("="*70)

if train_flag:
    print("\n>>> TRAINING MODE\n")
    train_qlearning()
else:
    print("\n>>> EVALUATION MODE\n")
    evaluate_qlearning()