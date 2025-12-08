#include "evaluation.hpp"
#include "board.hpp"
#include "utils.hpp"


// vrijednosti pojedinacnih tipova figura u igri opcenito
const int PIECE_VALUES[] = {
    100, 320, 330, 500, 900, 2000,
   -100,-320,-330,-500,-900,-2000 //za crne tj protivnicke figure pa su zato negativne
};

// ocjena koliko je pogodno svako polje na ploci za odredjenu figuru
const int PAWN_W_PST[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0
};

const int KNIGHT_W_PST[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
};

const int BISHOP_W_PST[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};

const int ROOK_W_PST[64] = {
     0,  0,  5, 10, 10,  5,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};

const int QUEEN_W_PST[64] = {
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -10,  5,  5,  5,  5,  5,  0,-10,
     0,  0,  5,  5,  5,  5,  0, -5,
    -5,  0,  5,  5,  5,  5,  0, -5,
   -10,  0,  5,  5,  5,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
};

const int KING_W_PST[64] = {
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30
};

// za crne se uzima sve isto samo preslikano po horizontalnoj osi
int PAWN_B_PST[64];
int KNIGHT_B_PST[64];
int BISHOP_B_PST[64];
int ROOK_B_PST[64];
int QUEEN_B_PST[64];
int KING_B_PST[64];

// limiti vrijednosti
int MIN_EVAL = -100000;
int MAX_EVAL = 100000;

// inicijalizacija za crne figure
void initEvaluation() {
    for (int i = 0; i < 64; ++i) {
        PAWN_B_PST[i] = PAWN_W_PST[63 - i];
        KNIGHT_B_PST[i] = KNIGHT_W_PST[63 - i];
        BISHOP_B_PST[i] = BISHOP_W_PST[63 - i];
        ROOK_B_PST[i] = ROOK_W_PST[63 - i];
        QUEEN_B_PST[i] = QUEEN_W_PST[63 - i];
        KING_B_PST[i] = KING_W_PST[63 - i];
    }
}
//racuna numericku vrijednost pozicije za neku figuru
int evaluate(const Board& board, int result) {
    //ako je prethodnim potezom vec zavrsena igra samo vraca rezultat
    if (result == DRAW) return 0;
    if (result == WHITE_WIN) return MAX_EVAL;
    if (result == BLACK_WIN) return MIN_EVAL;

    int eval = 0;
    //lambda funkcija, za figuru racuna evaluation vrijednost u odnosu na tip figure, lokaciju, boju
    auto addPieces = [&](uint64_t bitboard, int pieceType, const int* pst, bool isWhite) {
        uint64_t b = bitboard;
        while (b) {
            int sq = countTrailingZeros(b);
            eval += PIECE_VALUES[pieceType];
            eval += isWhite ? pst[sq] : -pst[sq];
            b &= b - 1;
        }
    };

    //poziva funkciju za sve figure, true i false za bijelu i crnu boju
    addPieces(board.pawn_W,   PAWN_W,   PAWN_W_PST, true);
    addPieces(board.knight_W, KNIGHT_W, KNIGHT_W_PST, true);
    addPieces(board.bishop_W, BISHOP_W, BISHOP_W_PST, true);
    addPieces(board.rook_W,   ROOK_W,   ROOK_W_PST, true);
    addPieces(board.queen_W,  QUEEN_W,  QUEEN_W_PST, true);

    addPieces(board.pawn_B,   PAWN_B,   PAWN_B_PST, false);
    addPieces(board.knight_B, KNIGHT_B, KNIGHT_B_PST, false);
    addPieces(board.bishop_B, BISHOP_B, BISHOP_B_PST, false);
    addPieces(board.rook_B,   ROOK_B,   ROOK_B_PST, false);
    addPieces(board.queen_B,  QUEEN_B,  QUEEN_B_PST, false);

    // za kralja se posebno racuna, gleda se samo pozicija
    eval += KING_W_PST[board.whiteKingSq];
    eval -= KING_B_PST[board.blackKingSq];

    return eval;
}
