#include "batch_eval.hpp"
#include "evaluation.hpp"
#include "typedefs.hpp"
#include "nnue.hpp"   // nnueHasWeights()

#include <vector>
#include <iostream>

#ifdef USE_CUDA_NNUE
#include "nnue_cuda.hpp"
#endif

// ----------------------------------------
// Async GPU/CPU batch eval (promjena #4/#5)
// ----------------------------------------
bool batchEvaluateRootPerspectiveAsync(const Board* boards, int n, int rootTurn, BatchEvalHandle& outHandle) {
    outHandle.n = n;
    outHandle.rootTurn = rootTurn;
    outHandle.usingGPU = false;
    outHandle.cpuBoards = boards;

#ifdef USE_CUDA_NNUE
    static bool tried = false;
    static bool cudaOK = false;

    if (!tried) {
        tried = true;
        cudaOK = nnueCudaInit();
        std::cout << "info string CUDA NNUE init = " << (cudaOK ? "OK" : "FAILED") << "\n";
    }

    // Prag: ispod ovoga često je CPU brži (GPU overhead).
    // (Ako hoćeš jači GPU benefit: stavi 64 ili 128.)
    constexpr int GPU_THRESHOLD = 1;

    // 25% NNUE, 75% classic (identično evaluateLeaf u evaluation.cpp)
    constexpr int a = 250; // permille

    const bool useNN = (USE_NNUE && nnueHasWeights());
    if (cudaOK && useNN && n >= GPU_THRESHOLD) {
        static bool printedGPU = false;
        if (!printedGPU) {
            printedGPU = true;
            std::cout << "info string CUDA batch eval ACTIVE (classic+nnue BLENDED ON GPU, pipelined)\n";
        }

        if (!nnueCudaSubmitBatchBlendedWhite(boards, n, a, outHandle.cuda)) {
            cudaOK = false;
        }
        else {
            outHandle.usingGPU = true;
            return true;
        }
    }
#endif

    // CPU fallback (immediate)
    static bool printedCPU = false;
    if (!printedCPU) {
        printedCPU = true;
        std::cout << "info string CPU batch eval ACTIVE\n";
    }

    outHandle.cpuBoardsOwned.assign(boards, boards + n);
    outHandle.cpuBoards = outHandle.cpuBoardsOwned.data();
    outHandle.usingGPU = false;
    return true;

}

bool batchEvaluateRootPerspectiveCollect(BatchEvalHandle& handle, double* outScores) {
    const int n = handle.n;
    if (n <= 0) return true;

#ifdef USE_CUDA_NNUE
    if (handle.usingGPU) {
        std::vector<int> blended((size_t)n);
        if (!nnueCudaCollectBatch(handle.cuda, blended.data())) return false;

        for (int i = 0; i < n; ++i) {
            int evWhite = blended[(size_t)i];
            outScores[i] = (handle.rootTurn == WHITE) ? (double)evWhite : (double)-evWhite;
        }
        return true;
    }
#endif

    // CPU fallback
    for (int i = 0; i < n; ++i) {
        int evWhite = evaluateLeaf(handle.cpuBoards[i], UN_DETERMINED);
        outScores[i] = (handle.rootTurn == WHITE) ? (double)evWhite : (double)-evWhite;
    }
    return true;
}

// Sync wrapper: koristi async submit + collect
void batchEvaluateRootPerspective(const Board* boards, int n, int rootTurn, double* outScores) {
    BatchEvalHandle h{};
    batchEvaluateRootPerspectiveAsync(boards, n, rootTurn, h);
    batchEvaluateRootPerspectiveCollect(h, outScores);
}
