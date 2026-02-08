#pragma once

#include "typedefs.hpp"

struct TTEntry {
    Bitboard zobrist; //hash pozicije
    int eval; //evaluacija pozicije, koliko je dobra pozciija
    int nodeType; //tip cvora: UPPER, LOWER, EXACT za alpha/beta pruning
    int depth; //dubina pretrage
    Move move; //najbolji potez iz te pozicije
};

enum NODE_TYPE { EXACT, LOWER, UPPER };

extern TTEntry TT_TABLE[];
extern const int TT_SIZE;

TTEntry getTTEntry(Bitboard zobrist);
void addTTEntry(const Board& board, int eval, Move move, int depth, int beta, int alpha);
