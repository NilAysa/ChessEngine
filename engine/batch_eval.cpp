#include "batch_eval.hpp"
#include "evaluation.hpp"
#include "typedefs.hpp"

#include <vector>
#include <iostream>

#ifdef USE_CUDA_NNUE
#include "nnue_cuda.hpp"
#endif

void batchEvaluateRootPerspective(const Board* boards, int n, int rootTurn, double* outScores) {

#ifdef USE_CUDA_NNUE
    static bool tried = false;
    static bool cudaOK = false;

    if (!tried) {
        tried = true;
        cudaOK = nnueCudaInit();
        std::cout << "info string CUDA NNUE init = " << (cudaOK ? "OK" : "FAILED") << "\n";
    }

    // Prag: ispod ovoga često je CPU brži (GPU overhead)
    constexpr int GPU_THRESHOLD = 1;

    if (cudaOK && n >= GPU_THRESHOLD) {
        static bool printedGPU = false;
        if (!printedGPU) {
            printedGPU = true;
            std::cout << "info string CUDA NNUE batch eval ACTIVE (async+pinned)\n";
        }

        std::vector<int> cp((size_t)n);
        if (nnueCudaEvaluateBatchWhite(boards, n, cp.data())) {
            for (int i = 0; i < n; ++i) {
                int evWhite = cp[i];
                outScores[i] = (rootTurn == WHITE) ? (double)evWhite : (double)-evWhite;
            }
            return;
        }
        // ako CUDA faila u runtime-u, padni na CPU
        cudaOK = false;
    }
#endif

    // CPU fallback (always available)
    static bool printedCPU = false;
    if (!printedCPU) {
        printedCPU = true;
        std::cout << "info string CPU batch eval ACTIVE\n";
    }

    for (int i = 0; i < n; ++i) {
        int evWhite = evaluateLeaf(boards[i], UN_DETERMINED);
        outScores[i] = (rootTurn == WHITE) ? (double)evWhite : (double)-evWhite;
    }
}

