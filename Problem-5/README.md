# Problem 5 — Warehouse Robot on a Slippery Floor (Q-learning)

## Overview

This exercise implements Q-learning to solve a navigation task inspired by a
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

## Problem description

The agent must navigate from the fixed start tile (`S`) to the goal tile (`G`) while
avoiding holes (`H`). On a slippery floor (`slippery=True`) the intended action may
result in an adjacent move, modelling real-world uncertainty (e.g., wheel slip).

## Setup
Requirements:
- gymnasium 
- numpy
- matplotlib

## Implementation notes

Agent: Q-learning

Info:
- Inputs: environment (`FrozenLake-v1`), hyperparameters (α, γ, ε schedule, episodes)
- Outputs: learned Q-table (n_states × n_actions), training logs (rewards, success rate)
- Error modes: unstable learning with too-large α; poor exploration if ε decays too fast

Core algorithm details:
- Q-table shape: `[n_states, n_actions]`, initialized to zeros
- Policy: ε-greedy (with decay: linear or exponential)
- Update rule: Q(s,a) ← Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))

hyperparameters (starting point):
- α (learning rate): 0.1
- γ (discount factor): 0.99
- ε start: 1.0, ε end: 0.01, decay over 10_000 episodes

Edge cases to consider:
- Episodes that end immediately in a hole
- Long episodes when agent loops; consider max steps per episode
- Very sparse reward in larger grids (8×8) — may need more training episodes

## Evaluation

Metrics we track:
- Success rate: fraction of episodes that reach the goal
- Average steps to success
- Cumulative reward per episode and moving averages

Baselines to compare against:
- Random policy baseline
- Simple deterministic heuristic (e.g., always move toward goal); can fail under slip

Visualization:
- Plot episode reward and a moving average (window=100)
- Plot success rate over time

## References

- Gymansium frozen lake scenario: https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- Gymnasium documentation: https://gymnasium.farama.org/introduction/basic_usage/

## Notes / Next steps

Optimization:
Experiment with:
- ε schedules (linear vs exponential decay),
- γ (myopic vs far-sighted),
- α (stability vs speed).
- Optional algorithmic variants: SARSA (on-policy), Double Q-learning
(reduces overestimation).
- Optional shaping: small step penalty (−0.01) to encourage shorter paths
(discuss pros/cons).