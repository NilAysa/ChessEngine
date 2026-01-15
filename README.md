## What is Parallelized

The following component is parallelized on the GPU:

### NNUE Batch Evaluation (Leaf Evaluation)

During MCTS search, leaf positions must be evaluated many times. Instead of evaluating each position sequentially on the CPU, the engine groups multiple positions into a batch and evaluates them in parallel on the GPU.

Each batch contains multiple independent chess positions. All positions in the batch are evaluated in a single CUDA kernel launch. The CPU continues to handle MCTS selection, expansion, backpropagation, and move generation.

This separation keeps the search logic simple and stable while accelerating the most expensive numeric computation.

## Parallelization Strategy

The GPU parallelization follows a two-level model.

At the grid level, one CUDA block corresponds to one chess position. The block index identifies the position inside the batch.

At the block level, one CUDA thread corresponds to one NNUE hidden neuron. Each block contains 128 threads, matching the NNUE hidden layer size.

Execution flow inside a block is as follows. Thread 0 builds a sparse list of active NNUE feature indices for the position. All threads compute their hidden neuron values in parallel using the sparse feature list. Thread 0 then computes the output neuron and writes the final centipawn score.

This design maps naturally to the NNUE structure and follows the SIMT execution model.

## CUDA Techniques Used

SIMT execution model is used, where the same kernel code is executed by many threads in parallel. Each thread computes one neuron, and each block processes one position.

Batching is used to evaluate multiple independent positions in a single kernel launch. This reduces kernel launch overhead and improves GPU utilization.

Pinned (page-locked) host memory is used via cudaMallocHost. This enables true asynchronous memory transfers and is required for efficient cudaMemcpyAsync operations.

CUDA streams are used with a persistent stream. The execution order is asynchronous host-to-device copy, kernel execution, and asynchronous device-to-host copy. Synchronization is done with cudaStreamSynchronize instead of cudaDeviceSynchronize, avoiding unnecessary global device stalls.

Persistent device buffers are used for boards and output scores. GPU memory is reused across calls and reallocated only if the batch size exceeds capacity. This avoids costly cudaMalloc and cudaFree calls inside the search loop.

CPU fallback logic is implemented. If CUDA initialization or kernel execution fails, the engine automatically falls back to CPU evaluation. This ensures robustness on systems without a compatible GPU.

## Runtime Confirmation

When GPU evaluation is active, the engine prints:

info string CUDA NNUE batch eval ACTIVE (async+pinned)

If GPU evaluation is unavailable, the engine automatically falls back to CPU evaluation.

## Summary

NNUE evaluation is parallelized on the GPU using CUDA. One block processes one position, and one thread computes one hidden neuron. The implementation uses batching, CUDA streams, pinned memory, and persistent buffers.
