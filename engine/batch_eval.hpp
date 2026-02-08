#pragma once
#include "typedefs.hpp"
#include <vector>

#ifdef USE_CUDA_NNUE
#include "nnue_cuda.hpp"
#endif

void batchEvaluateRootPerspective(const Board* boards, int n, int rootTurn, double* outScores);

struct BatchEvalHandle {
    int n = 0;
    int rootTurn = WHITE;
    bool usingGPU = false;

#ifdef USE_CUDA_NNUE
    CudaBatchHandle cuda;
#endif

    const Board* cpuBoards = nullptr;
    std::vector<Board> cpuBoardsOwned;
};

bool batchEvaluateRootPerspectiveAsync(const Board* boards, int n, int rootTurn, BatchEvalHandle& outHandle);
bool batchEvaluateRootPerspectiveCollect(BatchEvalHandle& handle, double* outScores);
