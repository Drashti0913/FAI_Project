import random
from collections import defaultdict


class QLearningAgent:
    """
    Tabular Q-learning agent for order dispatch.

    State: env.encode_state() → hashable tuple
    Action: flat index 0..17  (3 drivers × 6 positions)
    """

    def __init__(
        self,
        action_space_size: int = 18,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = action_space_size
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: state → array of Q-values per action
        self.q_table = defaultdict(lambda: [0.0] * action_space_size)
        self._episodes = 0

    # ------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------

    def act(self, state_key, env) -> int:
        """Epsilon-greedy action selection over valid actions only."""
        valid = env.valid_actions()
        if not valid:
            return 0

        if random.random() < self.epsilon:
            return random.choice(valid)

        q_vals = self.q_table[state_key]
        return max(valid, key=lambda a: q_vals[a])

    # ------------------------------------------------------------------
    # Learn
    # ------------------------------------------------------------------

    def update(
        self,
        state_key,
        action: int,
        reward: float,
        next_state_key,
        done: bool,
        valid_next_actions,
    ):
        """Standard Q-update."""
        q_vals = self.q_table[state_key]
        old_q = q_vals[action]

        if done or not valid_next_actions:
            target = reward
        else:
            next_q = self.q_table[next_state_key]
            best_next = max(next_q[a] for a in valid_next_actions)
            target = reward + self.gamma * best_next

        q_vals[action] = old_q + self.lr * (target - old_q)

    # ------------------------------------------------------------------
    # Epsilon decay (call once per episode)
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._episodes += 1

    @property
    def episodes_trained(self):
        return self._episodes

    def __repr__(self):
        return (
            f"QLearningAgent(episodes={self._episodes}, "
            f"epsilon={self.epsilon:.3f}, "
            f"states_seen={len(self.q_table)})"
        )