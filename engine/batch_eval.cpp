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
        if (cudaOK) {
            std::cout << "info string CUDA path compiled: USE_CUDA_NNUE=ON\n";
        }
        else {
            std::cout << "info string CUDA init failed -> CPU fallback\n";
        }
    }

    // CUDA path: use GPU NNUE only for now (simple correctness test)
    if (cudaOK) {
        std::vector<int> nnCp(n);
        if (nnueCudaEvaluateBatchWhite(boards, n, nnCp.data())) {
            // (Optional) one-time print so you KNOW GPU is used
            static bool printed = false;
            if (!printed) {
                printed = true;
                std::cout << "info string CUDA batch NNUE ACTIVE\n";
            }

            for (int i = 0; i < n; ++i) {
                int whiteScore = nnCp[i]; // white-perspective
                outScores[i] = (rootTurn == WHITE) ? (double)whiteScore : (double)-whiteScore;
            }
            return;
        }
        else {
            std::cout << "info string CUDA batch failed -> CPU fallback\n";
        }
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
