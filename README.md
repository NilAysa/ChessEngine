# Chess Engine – MCTS Version

This branch replaces the traditional Alpha-Beta search with **Monte Carlo Tree Search (MCTS)** as the main decision-making algorithm.

## Key changes
- Alpha-Beta pruning is **not used** in this version
- **MCTS** is introduced as the primary search algorithm
- An **Experience Book** is added to store and reuse information from previous simulations

This version is intended for experimentation and further extensions, including parallelization and improved rollout/evaluation strategies.
