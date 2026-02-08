#pragma once
#include "typedefs.hpp"

#ifdef USE_CUDA_NNUE
#include "nnue_cuda.hpp"
#endif

// Sync API (ostaje isto)
void batchEvaluateRootPerspective(const Board* boards, int n, int rootTurn, double* outScores);

// ------------------------------
// Async (pipelined) batch eval
// ------------------------------
// Omogućava da CPU radi selection/expand za sljedeći batch dok GPU evaluira prethodni.
// Ako CUDA nije dostupna, ove funkcije rade CPU fallback.

struct BatchEvalHandle {
    int n = 0;
    int rootTurn = WHITE;
    bool usingGPU = false;

#ifdef USE_CUDA_NNUE
    // WHITE-perspective cp (int) dolazi iz CUDA, pa se u collect pretvara u root perspektivu (double)
    CudaBatchHandle cuda;
#endif

    // CPU fallback storage
    const Board* cpuBoards = nullptr;
};

bool batchEvaluateRootPerspectiveAsync(const Board* boards, int n, int rootTurn, BatchEvalHandle& outHandle);
bool batchEvaluateRootPerspectiveCollect(BatchEvalHandle& handle, double* outScores);
