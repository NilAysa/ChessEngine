#pragma once
#include "typedefs.hpp"

extern int MAX_EVAL;
extern int MIN_EVAL;

void initEvaluation();

// global switch (default false)
extern bool USE_NNUE;

int evaluate(const Board& board, int result);

// Leaf evaluacija (NNUE samo ovdje)
int evaluateLeaf(const Board& board, int result);
