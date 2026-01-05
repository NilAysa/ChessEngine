# ChessCpp_mcts — What’s New (vs. ChessEngine-mcts-engine-v2)

This version introduces an NNUE evaluation pipeline and a batch-evaluation layer (CPU now, GPU-ready later), while keeping the existing MCTS + ExperienceBook architecture.

## Highlights

### 1) NNUE Integration (new evaluation path)
- Added a lightweight NNUE implementation and an embedded weights header:
  - `engine/nnue.cpp`
  - `engine/nnue.hpp`
  - `engine/nnue_weights_generated.hpp`
- Engine initialization now includes NNUE setup:
  - `initNNUE()` is called from `engine/uci.cpp`
- Added a runtime switch:
  - `USE_NNUE` (defined in `engine/evaluation.cpp`)
- MCTS leaf evaluation now uses a dedicated evaluator:
  - `evaluateLeaf(...)` (instead of classic-only evaluation)

### 2) NNUE + Classic Evaluation Blending (safe rollout)
- Leaf evaluation is blended to control runtime and stability:
  - `classic = evaluateClassic(...)`
  - `nn = nnueEvaluateWhitePerspective(board)`
  - final leaf = blended output (default: **25% NNUE**, `a=250` permille)
- Automatic fallback:
  - If NNUE is disabled or weights are unavailable, evaluation falls back to classic.

### 3) Batch Evaluation Layer (CPU now, GPU-ready)
- Added:
  - `engine/batch_eval.cpp`
  - `engine/batch_eval.hpp`
- MCTS one-ply minimax now:
  1. Builds an array of child boards
  2. Evaluates them via a single batch call:
     - `batchEvaluateRootPerspective(...)`
  3. Chooses best move via minimax on returned scores
- Current implementation is CPU-loop based, but the API is designed to be replaced by GPU batching later.

### 4) PST / Piece Values Refactor
- Moved classic piece values and PST tables into:
  - `engine/pst_tables.hpp`

### 5) Search Iteration Scaling Change
- Updated depth→iterations mapping in `engine/search.cpp`:
  - base changed from **8000** to **3000**
  - still uses `iterations = base * depth^2` (with depth clamp)
- Result: faster default search for the same `go depth N`.

## NNUE Weights Generation (Training Notebook)
- The NNUE weights header is produced by:
  - `ChessTrain.ipynb`
- Feature encoding (`INPUT_DIM = 781`) matches the engine:
  - 12×64 one-hot piece-square planes (768)
  - +1 side-to-move
  - +4 castling rights
  - +8 en-passant file one-hot
- Targets are labeled by Stockfish:
  - evaluated in centipawns (depth=10)
  - clipped to ±1000 cp
  - normalized using `SCALE = 400.0` (`y = cp / 400`)
- Model architecture:
  - MLP `781 -> 128 -> 1` with ReLU
- Export:
  - `export_cpp_header(model)` writes `nnue_weights_generated.hpp`
  - Engine consumes the same header directly.

## File Summary (New Files)
- `engine/nnue.cpp`, `engine/nnue.hpp` — NNUE forward evaluation and helpers
- `engine/nnue_weights_generated.hpp` — embedded NNUE weights (auto-generated)
- `engine/batch_eval.cpp`, `engine/batch_eval.hpp` — batch evaluation interface
- `engine/pst_tables.hpp` — classic piece values + PST tables extracted for reuse

## Notes
- ExperienceBook is unchanged conceptually and still biases move ordering as before.
- NNUE is intentionally used primarily in the leaf evaluation path to control runtime cost.
