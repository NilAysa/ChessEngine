# ChessEngine — GPU Version 2 (gpu-v2)

This branch represents the second iteration of GPU integration in the engine.

The primary objective of Version 2 was to fix correctness inconsistencies introduced in `gpu-v1` and ensure that GPU-based evaluation produces results consistent with the CPU evaluation pipeline.

---

## Purpose of Version 2

In `gpu-v1`, NNUE evaluation was offloaded to the GPU, but the final leaf evaluation logic did not fully match the CPU implementation. This could result in different evaluation scores and therefore different best moves when comparing CPU-only and GPU-enabled runs.

Version 2 resolves this issue.

---

## Key Changes Compared to Version 1

### 1. Unified Leaf Evaluation Logic

The GPU path was modified to follow the exact same evaluation logic as the CPU leaf evaluation:

- Classic evaluation is computed on the CPU.
- NNUE evaluation is computed on the GPU.
- Final score is blended identically to the CPU version (classic + NNUE combination).
- If NNUE weights are not available, the system falls back to classic evaluation.

This guarantees consistent scoring between CPU and GPU modes.

---

### 2. NNUE Feature Indexing Fix

The CUDA NNUE implementation was adjusted to correct square indexing/orientation differences between CPU and GPU feature extraction.

This ensures that both execution paths feed identical feature representations to the neural network.
