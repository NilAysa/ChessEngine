#include "batch_eval.hpp"
#include "evaluation.hpp"
#include "typedefs.hpp"
#include "nnue.hpp"   // nnueHasWeights()

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
    constexpr int GPU_THRESHOLD = 64;

    // 25% NNUE, 75% classic (identično evaluateLeaf u evaluation.cpp)
    constexpr int a = 250; // permille

    static bool printedValues = false;

    const bool useNN = (USE_NNUE && nnueHasWeights());

    if (cudaOK && useNN && n >= GPU_THRESHOLD) {
        static bool printedGPU = false;
        if (!printedGPU) {
            printedGPU = true;
            std::cout << "info string CUDA batch eval ACTIVE (classic+nnue BLENDED ON GPU, async+pinned)\n";
        }

        // GPU radi i classic i NNUE i blending -> vraća WHITE-perspective cp
        std::vector<int> blended((size_t)n);
        if (nnueCudaEvaluateBatchBlendedWhite(boards, n, a, blended.data())) {

            // DEBUG ISPIS (samo jednom)
            if (!printedValues && n > 0) {
                std::cout << "info string EVAL DEBUG | blended[0] = " << blended[0] << "\n";
                printedValues = true;
            }

            for (int i = 0; i < n; ++i) {
                int evWhite = blended[i];
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
