import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt

"""
Creating the frozen lake environment using gymnasium

there are 2 different maps, one for the 4x4 and one for the 8x8 grid. You can also create a custom map by changing the description parameter to a list of strings.

S = starting point, safe
F = floor, safe
H = spill pits, terminal, no reward
G = target shelf goal, terminal, reward +1

"""
Creating the frozen lake environment using gymnasium

there are 2 different maps, one for the 4x4 and one for the 8x8 grid. You can also create a custom map by changing the description parameter to a list of strings.

S = starting point, safe
F = floor, safe
H = spill pits, terminal, no reward
G = target shelf goal, terminal, reward +1

4x4 map:
S F F F
F H F H
F F F H
H F F G

8x8 map:
S F F F F F F F
F F F F F F F F
F F F H F F F F
F F F F F H F F
F F F H F F F F
F H H F F F H F
F H F F H F H F
F F F H F F F G
"""
class FrozenLakeAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ):
        """
        Creatung a Q-learning agent for the Frozen Lake environment.

        Args: 
            env: The training enviroment 
            learning_rate: How quickly to update Q-values (0 - 1)
            initial_epsilon: Initial exploration rate (0 - 1)

    # Cell 2: Q-Learning Agent and Helper Functions
            epsilon_decay: How much to reduce epsiolon each episode (0 - 1)
            final_epsilon: Minimum exploration rate (0 - 1)
            discount_factor: How much to value future rewards (0 - 1)
        """

        self.env = env 

        # Q-table: maps (state, action) to expected reward
        # Default dict automatically creates entries with zeroes for new states 
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor # Here we can balance how much our agent values immediate vs future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []
    
    def get_action(self, obs: int) -> int:
        """
        Choose an action using the epsiolon greedy strategy

        Returns:
            action: 0 (Left), 1 (Down), 2 (Right), 3 (Up)
        """

        if np.random.random() < self.epsilon:
            # Explore: choose random action
            return self.env.action_space.sample()
        else:
            # Exploit: choose best action from Q-table
            return np.argmax(self.q_table[obs])

    # Update Q-value based on experience from environment    
    def update(
        self, 
        obs: int,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: int,
    ):
        """
        Update the Q-value based on experience

        Args:
            obs: Current state
            action: Action taken
            reward: Reward received
            terminated: Whether the episode has ended
            next_obs: Next state after action
        """

        # Current Q-value
        current_q = self.q_table[obs][action]

        # Max Q-value for next state
        max_future_q = np.max(self.q_table[next_obs]) if not terminated else 0

        # Q-learning formula
        new_q = (1 - self.lr) * current_q + self.lr * (reward + self.discount_factor * max_future_q)

        # Update Q-table
        self.q_table[obs][action] = new_q

        # Track training error (absolute change in Q-value)
        self.training_error.append(abs(new_q - current_q))

    def decay_epsilon(self):
        # Decay epsilon after each episode
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


# Training parameters (This definetly needs to be tuned to get good performance)
EPISODES = 100000           # How many games to play
LEARNING_RATE = 0.05        # How fast to learn (α)
DISCOUNT_FACTOR = 0.99      # Value future rewards (γ) 
INITIAL_EPSILON = 1.0       # Start with 100% exploration for now
FINAL_EPSILON = 0.1         # Ends with 1% exploration for now
EPSILON_DECAY = 0.9999      # How fast to reduce exploration

agent = FrozenLakeAgent(
    env=env,
    learning_rate=LEARNING_RATE,
    initial_epsilon=INITIAL_EPSILON,
    epsilon_decay=EPSILON_DECAY,
    final_epsilon=FINAL_EPSILON,
    discount_factor=DISCOUNT_FACTOR
)

# Training loop for the Q-learning agent
if __name__ == "__main__":
    # Track training progress
    episode_rewards = []
    episode_lengths = []
    
    print("Starting training...")
    
    for episode in range(EPISODES):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Agent chooses action
            action = agent.get_action(obs)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Agent learns from experience
            agent.update(obs, action, reward, terminated, next_obs)
            
            # Update tracking
            episode_reward += reward
            episode_length += 1
            obs = next_obs
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Track episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Print progress
        if (episode + 1) % 1000 == 0:
            recent_success = sum(episode_rewards[-100:]) / 100
            print(f"Episode {episode + 1}: Success rate = {recent_success:.3f}")
    
    print("Training completed!")

# Evaluate the trained agent to get results and see if it is actually learning something
def evaluate_trained_agent(agent, env, num_episodes=1000):
    """Evaluate the trained Q-learning agent without exploration."""
    successes = 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # Use greedy policy (no exploration)  
            action = np.argmax(agent.q_table[obs])
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if done and reward > 0:
                successes += 1
                
    success_rate = successes / num_episodes
    print(f"Q-learning Agent Success Rate: {success_rate:.3f} ({success_rate*100:.1f}%)")
    return success_rate

# We need to compare our q-learning agent against some baselines such as a random policy to see if it is actually learning anything useful
def evaluate_random_baseline(env, num_episodes=1000):
    """Evaluate random policy baseline."""
    print("Testing Random Policy Baseline...")
    successes = 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:  # Prevent infinite loops
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            if done and reward > 0:
                successes += 1
                
    success_rate = successes / num_episodes
    print(f"Random Policy Success Rate: {success_rate:.3f} ({success_rate*100:.1f}%)")
    return success_rate

# We need to compare our q-learning agent against some baselines such as a heuristic policy to see if it is actually learning anything useful
def evaluate_heuristic_baseline(env, num_episodes=1000):
    """Evaluate simple heuristic baseline (attempt shortest deterministic route)."""
    print("Testing Simple Heuristic Baseline...")
    print("Strategy: Always try to move down or right toward goal")
    successes = 0
    

    for episode in range(num_episodes):
        # Reset environment for new episode, we need this to get the agent back to the starting positions S after each episode
        obs, _ = env.reset()
        done = False
        steps = 0
        
        # Limit steps to avoid infinite loops
        while not done and steps < 200:
            # Simple heuristic: alternate between down (1) and right (2)
            # This fails on slippery surface because it doesn't account for holes
            if steps % 2 == 0:
                action = 1  # Down
            else:
                action = 2  # Right

            # Take action in environment     
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            if done and reward > 0:
                successes += 1
                
    success_rate = successes / num_episodes
    print(f"Heuristic Policy Success Rate: {success_rate:.3f} ({success_rate*100:.1f}%)")
    return success_rate


"""

Visualizing Training Progress using matplotlib

"""

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500-episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    episode_rewards,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per episode)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    episode_lengths,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")


plt.tight_layout()
plt.show()

# Evaluate final agent performance and compare with baselines
print("\n FINAL EVALUATION & BASELINE COMPARISON")
print("="*60)
final_success_rate = evaluate_trained_agent(agent, env, num_episodes=1000)

print("\n BASELINE COMPARISONS")
print("="*50)
random_success = evaluate_random_baseline(env, num_episodes=1000)
heuristic_success = evaluate_heuristic_baseline(env, num_episodes=1000)

print("\n PERFORMANCE SUMMARY") 
print("="*50)
print(f"Q-learning Agent: {final_success_rate:.3f} ({final_success_rate*100:.1f}%)")
print(f"Random Baseline: {random_success:.3f} ({random_success*100:.1f}%)")  
print(f"Heuristic Baseline: {heuristic_success:.3f} ({heuristic_success*100:.1f}%)")
if random_success > 0:
    print(f"Improvement over Random: {final_success_rate/random_success:.1f}x better")
if heuristic_success > 0:
    print(f"Improvement over Heuristic: {final_success_rate/heuristic_success:.1f}x better")

# Debugging the q table and parameters had some issues with it
print("\n Q-TABLE ANALYSIS")
print("="*30)
print(agent.q_table)