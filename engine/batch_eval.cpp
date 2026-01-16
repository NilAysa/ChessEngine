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
    constexpr int GPU_THRESHOLD = 1;

    // Isti faktor kao u evaluateLeaf (evaluation.cpp)
    // 25% NNUE, 75% classic
    constexpr int a = 250; // permille

    if (cudaOK && n >= GPU_THRESHOLD) {
        static bool printedGPU = false;
        if (!printedGPU) {
            printedGPU = true;
            std::cout << "info string CUDA NNUE batch eval ACTIVE (classic+nnue blend, async+pinned)\n";
        }

        // 1) CPU classic evaluacija (jeftino)
        std::vector<int> classic((size_t)n);
        for (int i = 0; i < n; ++i) {
            classic[i] = evaluate(boards[i], UN_DETERMINED); // evaluate() je uvijek classic
        }

        // 2) GPU NNUE evaluacija (skupo -> GPU)
        std::vector<int> nn((size_t)n);
        if (nnueCudaEvaluateBatchWhite(boards, n, nn.data())) {

            const bool useNN = (USE_NNUE && nnueHasWeights());

            for (int i = 0; i < n; ++i) {
                int evWhite;
                if (useNN) {
                    // 3) identično evaluateLeaf(): classic + 25% NNUE
                    evWhite = (classic[i] * (1000 - a) + nn[i] * a) / 1000;
                }
                else {
                    evWhite = classic[i];
                }

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
