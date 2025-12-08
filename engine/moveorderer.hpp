#pragma once
#include "typedefs.hpp"
#include "tt.hpp"

// mali helper da ga vidi i search.cpp
inline bool moves_are_equal(const Move& a, const Move& b) {
    return a.fromSquare == b.fromSquare &&
        a.toSquare == b.toSquare &&
        a.promotion == b.promotion;
}

void score_moves(const Board& board,
    const TTEntry& entry,
    const Move& killer1,
    const Move& killer2,
    Move* moves, int cmoves);

// Vrati indeks narednog poteza najvećeg score-a (koji nije exhausted), ili -1.
int select_move(Move* moves, int cmoves);
