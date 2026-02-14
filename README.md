# ChessEngine

This repository contains the full development lifecycle of a custom chess engine, evolving from a sequential CPU implementation to a GPU-accelerated Monte Carlo Tree Search (MCTS) engine with batched NNUE evaluation.

The `main` branch serves as documentation. All implementations are organized across dedicated branches representing different architectural stages.

---

## Project Purpose

The purpose of this project was to:

- Implement a functional chess engine from scratch.
- Establish a correct sequential baseline.
- Introduce CPU parallelism.
- Transition from classical search to Monte Carlo Tree Search (MCTS).
- Integrate NNUE neural evaluation.
- Offload evaluation to GPU using CUDA.
- Design and benchmark a fully asynchronous batched GPU evaluation pipeline.

The repository reflects incremental architectural refinement and performance engineering.

---

## Branch Overview

Each branch represents a major development milestone.

### 1. Sequential Engine (`chess_sequential`)

- Single-threaded implementation.
- CPU-only evaluation.
- Baseline for correctness and performance comparison.

This version establishes the core engine logic and functional stability.

---

### 2. CPU Parallel Engine (`chess_parallel`)

- Multi-threaded search.
- Shared tree structure.
- Synchronization and simulation reservation mechanisms.
- Improved throughput over sequential baseline.

This stage focuses strictly on CPU-level parallelism.

---

### 3. MCTS-Based Engine

Branches:
- `mcts-engine`
- `mcts-engine-v2`
- `mcts-engine-v3`

Major changes:
- Transition from classical search to Monte Carlo Tree Search.
- PUCT selection formula.
- Progressive widening.
- In-flight simulation tracking.
- Visit count and value backpropagation.
- Reservation-based selection to reduce contention.

This stage establishes a scalable search framework suitable for batching.

---

### 4. NNUE Integration

Neural evaluation (NNUE) was integrated to improve position scoring.

Features:
- Root-perspective evaluation.
- Batched evaluation interface.
- CPU fallback logic.
- Weight loading verification.
- Efficient board feature extraction.

At this stage, neural evaluation becomes the primary performance bottleneck.

---

### 5. CUDA GPU Acceleration

GPU evolution branches:

- `gpu-v1`
- `gpu-v1.1`
- `gpu-v2`
- `gpu-v2.1`
- `gpu-v2.2`
- `gpu-v3`
- `gpu-v3.1`

Each version incrementally improves:

- Batch scheduling logic
- Host-to-device transfer efficiency
- Asynchronous execution
- Stream concurrency
- Double buffering
- Round-robin buffer selection
- CPU/GPU threshold switching

The most advanced implementation (`gpu-v3.1`) includes:

- Two CUDA streams
- Double device buffers
- Asynchronous memory copies
- Overlapping CPU tree search with GPU evaluation
- In-flight batch tracking
- CPU fallback when GPU overhead is not justified

This architecture maximizes overlap between CPU and GPU workloads.

---

## Final Engine Pipeline (gpu-v3.1)

The final architecture operates as follows:

1. CPU performs Selection and Expansion.
2. Leaf boards are collected into a batch.
3. The batch is dispatched asynchronously to the GPU.
4. GPU performs parallel NNUE evaluation.
5. CPU continues reserving additional simulations.
6. Completed GPU batches are collected.
7. Backpropagation updates visits and value sums.

Key design elements:

- Batched evaluation
- PUCT-based selection
- Progressive widening
- In-flight simulation counters
- Double-buffered CUDA streams
- Asynchronous GPU execution
- CPU fallback threshold

The design minimizes GPU idle time and reduces launch overhead.

---

## Performance Engineering Focus

The project investigates:

- CPU vs GPU evaluation differences
- Batch size thresholds
- CUDA initialization overhead
- Asynchronous stream scheduling
- Determinism in parallel search
- Accurate time measurement (time vs NPS)
- Colab GPU vs local execution behavior

Multiple configurations were benchmarked:

- Pure CPU mode
- GPU-enabled mode
- CUDA disabled on GPU machine
- Colab execution vs local execution

---

## Build Instructions

Build (Release):

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

For CUDA builds:

cmake -S . -B build -DUSE_CUDA_NNUE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

CUDA Toolkit must be installed and compatible with the selected architecture.

---

## UCI Usage Example

uci
isready
ucinewgame
position startpos
go depth 10


Engine outputs:

- bestmove
- info depth
- time
- nps

---

## Current Recommended Branch

The most advanced and optimized implementation is located in:

`gpu-v3.1`

Other branches remain available for comparison and experimentation.

---

## Technical Scope

This repository demonstrates practical implementation of:

- Monte Carlo Tree Search
- Neural network integration in C++
- CPU multi-threading
- CUDA programming
- Asynchronous stream pipelines
- Double buffering strategies
- Performance benchmarking methodology

---

## Author

Developed by Nil Aysa.
