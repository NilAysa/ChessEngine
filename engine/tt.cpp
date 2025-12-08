#include "tt.hpp"
#include <cstdint>

// npr. 2^20 = 1'048'576 unosa (~64 MB ako je TTEntry ~64 B)
constexpr int TT_BITS = 20;
const int TT_SIZE = 1 << TT_BITS;

TTEntry TT_TABLE[TT_SIZE];

static inline int ttIndex(Bitboard z) {
    // ako ti je Bitboard = uint64_t, možeš maskirati donje bitove
    return static_cast<int>(z) & (TT_SIZE - 1);
}

TTEntry getTTEntry(Bitboard zobrist) {
    return TT_TABLE[ttIndex(zobrist)];
}

//funkcjia koja upisuje u niz obrađenu poziciju
void addTTEntry(const Board& board, int eval, Move move, int depth, int beta, int alpha) {
    TTEntry entry;
    //odredjivanje tipa cvora
    if (eval <= alpha)
        entry.nodeType = UPPER;
    else if (eval >= beta)
        entry.nodeType = LOWER;
    else 
        entry.nodeType = EXACT;

    entry.eval = eval;
    entry.move = move;
    entry.depth = depth;
    entry.zobrist = board.hash;  

    TT_TABLE[ttIndex(entry.zobrist)] = entry;
}
