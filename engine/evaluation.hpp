#pragma once

#include "typedefs.hpp"


extern int MAX_EVAL;
extern int MIN_EVAL;


void initEvaluation();


int evaluate(const Board& board, int result);
