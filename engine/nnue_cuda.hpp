#pragma once
#include "typedefs.hpp"

#ifdef USE_CUDA_NNUE
#include <cuda_runtime.h>
#endif

// init/caching weights on device
bool nnueCudaInit();     // returns true if CUDA available + init OK
void nnueCudaShutdown(); // optional cleanup

// Evaluate NNUE for a batch of boards, returns WHITE-perspective cp
bool nnueCudaEvaluateBatchWhite(const Board* boards, int n, int* outCp);

// Evaluate blended (classic + NNUE) on GPU, returns WHITE-perspective cp
bool nnueCudaEvaluateBatchBlendedWhite(const Board* boards, int n, int blendPermille, int* outCp);

// ------------------------------
// Async (pipelined) API
// ------------------------------
// Pattern:
//   CudaBatchHandle h;
//   nnueCudaSubmitBatchBlendedWhite(..., h);
//   // CPU radi nešto drugo...
//   nnueCudaCollectBatch(h, outCp);

struct CudaBatchHandle {
#ifdef USE_CUDA_NNUE
    cudaEvent_t done = nullptr;
#endif
    int n = 0;
    int bufferIndex = 0; // 0/1 (double-buffer)
};

bool nnueCudaSubmitBatchBlendedWhite(const Board* boards, int n, int blendPermille, CudaBatchHandle& outHandle);
bool nnueCudaCollectBatch(CudaBatchHandle& handle, int* outCp);
