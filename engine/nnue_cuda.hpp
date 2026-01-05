#pragma once
#include "typedefs.hpp"

// init/caching weights on device
bool nnueCudaInit();     // returns true if CUDA available + init OK
void nnueCudaShutdown(); // optional cleanup

// Evaluate NNUE for a batch of boards, returns WHITE-perspective cp
// outCp size = n
bool nnueCudaEvaluateBatchWhite(const Board* boards, int n, int* outCp);
