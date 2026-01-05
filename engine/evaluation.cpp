#include "evaluation.hpp"
#include "board.hpp"
#include "utils.hpp"
#include "nnue.hpp"
#include "pst_tables.hpp"


bool USE_NNUE = true;

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
static int evaluateClassic(const Board& board, int result) {
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

int evaluate(const Board& board, int result) {
    // NAMJERNO: uvijek classic (stabilno za rollout / sve osim leaf-a)
    return evaluateClassic(board, result);
}

int evaluateLeaf(const Board& board, int result) {
    int classic = evaluateClassic(board, result);

    if (!(USE_NNUE && nnueHasWeights())) {
        return classic;
    }

    int nn = nnueEvaluateWhitePerspective(board);

    // 25% NNUE na startu
    const int a = 250; // permille: 0..1000
    return (classic * (1000 - a) + nn * a) / 1000;
}


