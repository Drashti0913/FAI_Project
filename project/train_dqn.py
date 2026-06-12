"""
DQN Training Script - Much faster than tabular Q-learning!
"""
import sys
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from delivery_env import DeliveryEnvironment
from dqn_agent import DQNAgent

# ============================================================================
# CHANGE THESE:
# ============================================================================
NUM_EPISODES = 1000       # DQN needs fewer episodes than tabular!
EPSILON_DECAY = 0.9977   # For 50k episodes
# ============================================================================

train_flag = 'train' in sys.argv

def train_dqn():
    """Train DQN agent - faster than tabular Q-learning"""
    env = DeliveryEnvironment(num_drivers=3, grid_size=5, order_arrival_rate=0.3)
    agent = DQNAgent(state_size=10, action_size=125, epsilon_decay=EPSILON_DECAY)
    
    episode_rewards = []
    episode_on_time = []
    episode_completions = []
    losses = []
    
    print("="*70)
    print(f"DQN Training: {NUM_EPISODES} episodes (3 drivers)")
    print(f"Device: {agent.device}")
    print("="*70)
    
    start_time = time.time()
    
    for episode in tqdm(range(NUM_EPISODES), desc="DQN Training", unit="ep"):
        state_dict = env.reset()
        state = agent.discretize_state(env)
        episode_reward = 0
        episode_loss = []
        
        for step in range(500):
            # Select action
            action_int = agent.select_action(state, agent.epsilon)
            actions = agent.decode_action(action_int)
            
            # Execute
            next_state_dict, reward, done = env.step(actions)
            next_state = agent.discretize_state(env)
            
            # Store in replay buffer
            agent.remember(state, action_int, reward, next_state, done)
            
            # Train network (if enough samples)
            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_completions.append(len(env.completed_orders))
        
        if env.completed_orders:
            on_time = sum(1 for oid in env.completed_orders 
                         if env.orders[oid].delivery_time <= env.orders[oid].deadline)
            episode_on_time.append((on_time / len(env.completed_orders)) * 100)
        else:
            episode_on_time.append(0)
        
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        agent.decay_epsilon()
        
        # Progress updates
        if episode % 5000 == 0 and episode > 0:
            elapsed = time.time() - start_time
            window = min(5000, episode)
            
            tqdm.write(f"\n{'='*70}")
            tqdm.write(f"Checkpoint @ Episode {episode}/{NUM_EPISODES}")
            tqdm.write(f"Avg reward (last {window}):   {np.mean(episode_rewards[-window:]):.1f}")
            tqdm.write(f"Avg on-time (last {window}):  {np.mean(episode_on_time[-window:]):.1f}%")
            tqdm.write(f"Avg loss (last 100):           {np.mean(losses[-100:]) if losses else 0:.4f}")
            tqdm.write(f"Memory size: {len(agent.memory)}, ε: {agent.epsilon:.4f}, Time: {elapsed/60:.1f}m")
            tqdm.write(f"{'='*70}\n")
        
        if episode > 0 and episode % 25000 == 0:
            agent.save(f'dqn_model_{episode}.pth')
            tqdm.write(f">>> Saved: dqn_model_{episode}.pth\n")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("DQN Training Complete!")
    print("="*70)
    print(f"Episodes:             {NUM_EPISODES}")
    print(f"Time:                 {total_time/3600:.2f}h ({total_time/60:.1f}m)")
    print(f"Speed:                {NUM_EPISODES/total_time:.2f} eps/s")
    print(f"Memory buffer:        {len(agent.memory)} experiences")
    print(f"Final epsilon:        {agent.epsilon:.6f}")
    print(f"\nFinal 1000 episodes:")
    print(f"  Average reward:     {np.mean(episode_rewards[-1000:]):.1f}")
    print(f"  Average on-time:    {np.mean(episode_on_time[-1000:]):.1f}%")
    print(f"  Average completed:  {np.mean(episode_completions[-1000:]):.1f}")
    if losses:
        print(f"  Final loss:         {np.mean(losses[-100:]):.4f}")
    print("="*70)
    
    agent.save('dqn_model_final.pth')
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Reward
    axes[0,0].plot(episode_rewards, alpha=0.3)
    window = 1000
    if len(episode_rewards) > window:
        ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0,0].plot(range(window-1, len(episode_rewards)), ma, 'r-', linewidth=2, label='MA(1000)')
        axes[0,0].legend()
    axes[0,0].set_title('Episode Reward (DQN)')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].grid(True, alpha=0.3)
    
    # On-time %
    axes[0,1].plot(episode_on_time, alpha=0.3, color='green')
    if len(episode_on_time) > window:
        ma = np.convolve(episode_on_time, np.ones(window)/window, mode='valid')
        axes[0,1].plot(range(window-1, len(episode_on_time)), ma, 'r-', linewidth=2, label='MA(1000)')
        axes[0,1].legend()
    axes[0,1].set_title('On-Time %')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].grid(True, alpha=0.3)
    
    # Completions
    axes[1,0].plot(episode_completions, alpha=0.3, color='purple')
    axes[1,0].set_title('Orders Completed')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].grid(True, alpha=0.3)
    
    # Loss
    if losses:
        axes[1,1].plot(losses, alpha=0.6, color='orange')
        axes[1,1].set_title('Training Loss (lower = better)')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_yscale('log')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle(f'DQN Training Results ({NUM_EPISODES} episodes)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'dqn_training_{NUM_EPISODES}.png', dpi=150)
    print(f"\nSaved: dqn_training_{NUM_EPISODES}.png")

def evaluate_dqn():
    """Evaluate trained DQN agent"""
    env = DeliveryEnvironment(num_drivers=3)
    agent = DQNAgent()
    agent.load('dqn_model_final.pth')
    
    num_eval = 1000
    episode_rewards = []
    episode_on_time = []
    
    print(f"\nEvaluating DQN (1000 episodes, NO exploration)...\n")
    start_time = time.time()
    
    for episode in tqdm(range(num_eval), desc="Evaluating", unit="ep"):
        state_dict = env.reset()
        state = agent.discretize_state(env)
        ep_reward = 0
        
        for step in range(500):
            action_int = agent.select_action(state, epsilon=0.0)
            actions = agent.decode_action(action_int)
            
            next_state_dict, reward, done = env.step(actions)
            next_state = agent.discretize_state(env)
            
            ep_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(ep_reward)
        
        if env.completed_orders:
            on_time = sum(1 for oid in env.completed_orders 
                         if env.orders[oid].delivery_time <= env.orders[oid].deadline)
            episode_on_time.append((on_time / len(env.completed_orders)) * 100)
        else:
            episode_on_time.append(0)
    
    eval_time = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"  DQN Evaluation Results  (1000 episodes)")
    print("="*70)
    print(f"  Average reward:              {np.mean(episode_rewards):.2f}")
    print(f"  Average on-time:             {np.mean(episode_on_time):.2f}%")
    print(f"  Std deviation (reward):      {np.std(episode_rewards):.2f}")
    print(f"  Evaluation time:             {eval_time:.2f}s")
    print("="*70)

if train_flag:
    print("\n>>> DQN TRAINING MODE\n")
    train_dqn()
else:
    print("\n>>> DQN EVALUATION MODE\n")
    evaluate_dqn()