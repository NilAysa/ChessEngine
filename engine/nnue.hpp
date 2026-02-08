#pragma once
#include "typedefs.hpp"

// Init (za kasnije učitavanje težina)
void initNNUE();

// Da evaluation.cpp može sigurno odlučiti hoće li koristiti NNUE ili classic
bool nnueHasWeights();

// NNUE eval (WHITE-perspective: pozitivan = bolje za bijelog)
int nnueEvaluateWhitePerspective(const Board& b);
