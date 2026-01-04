## MCTS Engine – v2 Improvements

This branch (`mcts-engine-v2`) builds on the initial MCTS structure and introduces
several refinements to improve stability, move quality, and search behavior.

### Key improvements
- Removed remaining Alpha–Beta legacy assumptions (depth semantics, move ordering bias)
- Clarified `go depth N` as an iteration budget, not ply depth
- Introduced PUCT-based selection with explicit priors
- Added progressive widening to control early tree expansion
- Integrated quiescence-light evaluation (capture-only) at leaf nodes
- Improved positional move priors (development, center control, early queen/rook penalties)
- Stabilized score semantics (root-perspective average value over simulations)
- Experience Book is now used consistently as a prior bias, not a decision override

Overall, this version produces more stable openings, fewer random-looking moves,
and clearer score behavior compared to the initial MCTS implementation.
