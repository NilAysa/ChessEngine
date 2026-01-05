#pragma once
#include "typedefs.hpp"

// Batch evaluate boards and return scores in ROOT perspective.
// Uses your existing evaluate() which is white-perspective (positive = better for WHITE).
void batchEvaluateRootPerspective(const Board* boards, int n, int rootTurn, double* outScores);
