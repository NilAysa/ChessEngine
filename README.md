## GPU Batch Evaluation: Pipelining and Packed Transfers

These changes improve the CUDA execution path in the engine, making the GPU noticeably faster than the CPU in the MCTS batched evaluation mode.

---

### Summary of Changes

#### 1) Pipelined (Asynchronous) CUDA Batch Evaluation

An asynchronous API for batch evaluation has been introduced:

- `nnueCudaSubmitBatchBlendedWhite(...)` *(submit phase)*
- `nnueCudaCollectBatch(...)` *(collect phase)*

The implementation uses **double buffering** with **two CUDA streams** and `cudaEvent` synchronization.

This removes the per-batch blocking `cudaStreamSynchronize` call and enables workload overlap:

- The CPU performs *selection/expand* for the next batch  
- While the GPU evaluates the previous batch  

This significantly reduces idle time and improves overall throughput.

---

#### 2) Reduced Host-to-Device Payload (PackedBoard)

Instead of transferring the entire `Board` structure to the GPU, a compact `PackedBoard` structure is now used. It contains only the data required for:

- Classic PST-based evaluation  
- NNUE feature extraction (bitboards and essential metadata)

This reduces H2D/D2H transfer overhead and improves total evaluation performance.

---

#### 3) Increased MCTS Batch Size

The `batchSize` parameter was increased from **64** to **256**.

A larger batch size:

- Reduces the number of GPU kernel launches  
- Reduces synchronization and collect cycles  
- Improves GPU utilization and amortizes launch overhead  

---

### Performance Rationale

**Before:**

- GPU was used only for leaf evaluation, but each batch incurred a hard synchronization.
- The CPU frequently stalled waiting for the GPU.
- Transferring the full `Board` structure increased memory transfer overhead.
- A small batch size (64) resulted in many kernel launches and synchronization cycles.

**After:**

- CPU and GPU workloads are overlapped through pipelining.
- Memory transfers are smaller due to `PackedBoard`.
- Larger batches reduce launch overhead and improve device occupancy.
- Overall GPU throughput is significantly improved relative to CPU-only execution.

---

### Fallback Behavior

- If CUDA initialization fails or NNUE is unavailable, the engine automatically falls back to CPU batch evaluation.
- The synchronous API (`batchEvaluateRootPerspective`) remains available and internally wraps the asynchronous submit/collect mechanism.

---

### Key Modified Files

- `engine/mcts.cpp` – Pipelined `runBatched()` implementation (CPU/GPU overlap)
- `engine/batch_eval.cpp/.hpp` – Asynchronous batch evaluation API with CPU fallback
- `engine/nnue_cuda.cu/.hpp` – Double-buffered CUDA evaluation and `PackedBoard` transfer
- `engine/mcts.hpp` – `batchSize` updated from 64 → 256

---

### Tuning Recommendations

For further performance optimization, the following parameters can be adjusted:

- `batchSize` (e.g., 128 / 256 / 512 depending on GPU capability)
- GPU activation threshold (e.g., `GPU_THRESHOLD = 64`)

These parameters should be tuned according to the target hardware.
