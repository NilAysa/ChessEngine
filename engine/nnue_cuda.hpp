#pragma once
#include "typedefs.hpp"

// init/caching weights on device
bool nnueCudaInit();     // returns true if CUDA available + init OK
void nnueCudaShutdown(); // optional cleanup

// Evaluate NNUE for a batch of boards, returns WHITE-perspective cp
// outCp size = n
bool nnueCudaEvaluateBatchWhite(const Board* boards, int n, int* outCp);

// Evaluate blended (classic + NNUE) on GPU, returns WHITE-perspective cp
// blendedCp = (classic*(1000-a) + nnue*a) / 1000, where a is permille (0..1000)
bool nnueCudaEvaluateBatchBlendedWhite(const Board* boards, int n, int blendPermille, int* outCp);
