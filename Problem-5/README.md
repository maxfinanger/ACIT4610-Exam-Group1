# Problem 5 — Warehouse Robot on a Slippery Floor (Q-learning)

## Overview

This exercise implements tabular Q-learning to solve a navigation task inspired by a
warehouse robot operating on a slippery floor. The environment used is OpenAI Gym's
FrozenLake-v1 which models a grid world containing safe tiles, holes (hazards), a start
location (S), and a goal (G). With `slippery=True` the agent's actions become stochastic,
so the robot may slip to unintended directions. The goal is to learn a robust policy that
maximises success rate and minimises steps to reach the goal.
Key points:
- Environment: `FrozenLake-v1` (grid map, slippery transitions)
- Observation: discrete state index (tile)
- Actions: 0=Left, 1=Down, 2=Right, 3=Up
- Reward: +1 for reaching goal, 0 otherwise (sparse)

## Structure of this README

- Problem description
- Setup (requirements and installation)
- Implementation notes (Q-learning agent)
- Training (hyperparameters and logging)
- Evaluation and baselines
- Experiments and optimisation
- Deliverables & report guidance

## Problem description

The agent must navigate from the fixed start tile (`S`) to the goal tile (`G`) while
avoiding holes (`H`). On a slippery floor (`slippery=True`) the intended action may
result in an adjacent move, modelling real-world uncertainty (e.g., wheel slip).
## Setup

Requirements (recommended):
- Python 3.8+
- gym or gymnasium (the README uses Gym's `FrozenLake-v1`)
- numpy
- matplotlib

Minimal installation (example using pip):

```powershell
python -m pip install --upgrade pip;
pip install gym numpy matplotlib
```

Notes:
- If you use `gymnasium`, API calls are similar; adapt imports accordingly.
- For reproducible experiments, pin package versions in a `requirements.txt`.

## Implementation notes

Agent: tabular Q-learning

Contract (brief):
- Inputs: environment (`FrozenLake-v1`), hyperparameters (α, γ, ε schedule, episodes)
- Outputs: learned Q-table (n_states × n_actions), training logs (rewards, success rate)
- Error modes: unstable learning with too-large α; poor exploration if ε decays too fast

Core algorithm details:
- Q-table shape: `[n_states, n_actions]`, initialized to zeros
- Policy: ε-greedy (with decay: linear or exponential)
- Update rule: Q(s,a) ← Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))

Suggested hyperparameters (starting point):
- α (learning rate): 0.1
- γ (discount factor): 0.99
- ε start: 1.0, ε end: 0.01, decay over 10_000 episodes

Edge cases to consider:
- Episodes that end immediately in a hole
- Long episodes when agent loops; consider max steps per episode
- Very sparse reward in larger grids (8×8) — may need more training episodes

## Training

Training recommendations:
- Train for a large number of episodes (e.g., 10k–50k) to overcome sparse rewards
- Track per-episode return and success (reached goal) as booleans
- Log moving averages (e.g., window=100) for smoother learning curves

What to log:
- Episode reward
- Steps taken
- Success (0/1)
- ε value

Quick example (pseudo):

```python
# ... create env = gym.make("FrozenLake-v1", map_name="6x6", is_slippery=True)
# initialize Q = np.zeros((env.observation_space.n, env.action_space.n))
# for episode in range(N):
#     reset, done = env.reset(), False
#     while not done:
#         choose action with eps-greedy
#         step, reward, done, _ = env.step(action)
#         update Q-table
#     log reward and success
```

## Evaluation

Metrics to report:
- Success rate: fraction of episodes that reach the goal
- Average steps to success
- Cumulative reward per episode and moving averages

Baselines to compare against:
- Random policy baseline (choose actions uniformly)
- Simple deterministic heuristic (e.g., always move toward goal); can fail under slip

Visualization:
- Plot episode reward and a moving average (window=100)
- Plot success rate over time

## Experiments & optimisation

Ideas to explore:
- ε schedules: linear vs exponential decay
- γ values: smaller γ (myopic) vs larger γ (far-sighted)
- α values: stability (small α) vs fast learning (large α)
- Algorithm variants: SARSA (on-policy), Double Q-learning
- Reward shaping: small step penalty (e.g., -0.01) to penalize long paths (discuss trade-offs)

Suggested experimental procedure:
1. Fix all hyperparameters except one (one-variable-at-a-time)
2. Run multiple seeds and average results for robustness
3. Produce learning curves and tabulate final success rates

## Deliverables

- Python script or notebook containing the implementation, training loop, and plots
- README with setup and run instructions (this file)
- Short report with method, hyperparameters, plots, comparison to baselines, and conclusions

## References

- OpenAI Gym: https://www.gymlibrary.dev/
- Sutton & Barto — Reinforcement Learning: An Introduction

## Notes / Next steps


Problem 5: Solving a Real-World Problem Using Reinforcement Learning
Overview
This lab exercise applies reinforcement learning—specifically Q-learning—to address a realworld-inspired control task. Students will use a publicly available environment to train an RL
agent, assess its performance, and optimise it for robust behaviour.
Problem Statement: Warehouse Robot on a Slippery Floor
Students will develop a warehouse floor robot that must navigate from a loading bay to a target
shelf without falling into hazards (holes) and while dealing with a slippery surface. Because
the robot’s movement can slip unpredictably, it needs to learn a safe, efficient route instead of
relying on fixed plans.
This exercise uses the FrozenLake-v1 environment (Gym), which models a slippery warehouse
floor as a grid with safe tiles, holes (hazards), a start point, and a goal. The agent must learn a
policy that maximizes success rate and minimizes steps under stochastic transitions.
Dataset / Environment:
• Environment: FrozenLake-v1 (4×4 or 8×8 map; “slippery=True” to induce stochastic
motion).
• Observation space (state): a discrete tile index (grid cell).
• Action space: {Left, Down, Right, Up}.
• Rewards: Reaching the goal yields +1; falling into a hole yields 0 (episode terminates).
Stepping on safe tiles yields 0 (sparse reward).
• Real-world analogy: Autonomous robot navigating a slick warehouse aisle with spill
zones (holes) to reach a pick location (goal).
Environment Description:
• Grid size: 6×6 (default) or 8×8 for a harder variant.
• Start/Goal: Fixed start at “S” and target shelf at “G”.
• Hazards: “H” tiles represent spill pits; entering them ends the episode.
• Stochasticity: On a slippery floor, intended actions may slip to adjacent directions,
modeling real-world uncertainty (wheels slipping, micro-surface variation).
Tasks:
1. Understanding the Environment:
• Instantiate FrozenLake-v1 and print state and action spaces.
• Visualize the grid and annotate S (start), G (goal), and H (holes).
• Explain the reward structure and the effect of slippery=True.
2. Setting Up the RL Agent:
• Implement tabular Q-learning with a Q-table of size [n_states × n_actions].
• Use ε-greedy exploration (ε decay), learning rate α, discount factor γ.
• Initialize Q(s, a) = 0 for all feasible state–action pairs.
Page 12 of 13
3. Training the RL Agent:
• Train over many episodes (e.g., 10k+) to handle sparse rewards and slippage.
• Tune α, γ, ε schedule; track episode returns and success rate.
• Consider separate runs for 6×6 vs 8×8 maps to compare difficulty.
4. Evaluation:
• Report success rate (fraction of episodes reaching the goal).
• Plot cumulative reward per episode and a moving average.
• Compare against:
o Random policy baseline.
o Simple heuristic (e.g., always attempt a shortest deterministic route) to
show why planning fails on slippery floors.
5. Optimization:
• Experiment with:
o ε schedules (linear vs exponential decay),
o γ (myopic vs far-sighted),
o α (stability vs speed).
• Optional algorithmic variants: SARSA (on-policy), Double Q-learning
(reduces overestimation).
• Optional shaping: small step penalty (−0.01) to encourage shorter paths
(discuss pros/cons).
6. Reporting:
• Document your approach, design decisions (hyperparameters, ε schedule), and
training curves.
• Discuss challenges: sparse rewards and stochastic transitions.
• Note potential improvements: larger Q-tables (8×8), eligibility traces, or
moving to function approximation (DQN) if you later convert observations to
rich features (optional).
Deliverables:
• Code: Python notebook/script with Q-learning implementation, training loop,
evaluation, and plots. Include a README with setup steps and run instructions.
• Report: Brief write-up describing the problem, method, results (plots +
metrics), and conclusions/next steps.
Tools and Libraries:
• Python (3.x)
• Gym/Gymnasium (for FrozenLake-v1)
• NumPy (Q-table, numeric ops)
• Matplotlib (for plotting results)
• TensorFlow/PyTorch (optional, for more advanced RL algorithms like DQN)
Page 13 of 13
Evaluation Criteria
• Correctness and Completeness of Implementation
• Comparison and Analysis of Algorithm Variations: learning curves; compare against
random/heuristic baselines.
• Depth and Insight of Analysis
• Quality and Clarity of the Report
• Code Quality, Documentation & Clarity of the Report: clean structure, comments,
clear README; reproducible results.