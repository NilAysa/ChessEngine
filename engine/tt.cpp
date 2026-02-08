#include "tt.hpp"


//transposition table, koristi se za memorisanje vec obradjenih pozicija da bi se ubrzao alfa-beta pruning
//omogucava da se preskoce velike grane stabla ako je pozicija vec obradjena
const int TT_SIZE = 100000;
TTEntry TT_TABLE[TT_SIZE]; 

//vraca entry na mjestu zobrist%TT_SIZE
TTEntry getTTEntry(Bitboard zobrist) {
    return TT_TABLE[zobrist % TT_SIZE];
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

    TT_TABLE[entry.zobrist % TT_SIZE] = entry;
}
