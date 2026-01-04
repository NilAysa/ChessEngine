#include "moveorderer.hpp"
#include "bitboards.hpp"  // SQUARE_BITBOARDS

// MVV-LVA tabele (jednostavne, možeš kasnije doraditi po želji)
static const int VICTIMS[] = {
    100 * 16,   // pawn
    320 * 16,   // knight
    330 * 16,   // bishop
    500 * 16,   // rook
    900 * 16,   // queen
};

// ATTACKERS indeksira po Move.pieceType (0..11)
static const int ATTACKERS[] = {
    100, 200, 300, 400, 500, 600,   // bijele
    100, 200, 300, 400, 500, 600    // crne
};

static const int MVV_LVA_PAWN_EXCHANGE = (100 * 16) - 100;

void score_moves(const Board& board, Move* moves, int cmoves) {
    Bitboard enemyOcc = board.turn ? board.occupancyBlack : board.occupancyWhite;

    for (int mi = 0; mi < cmoves; ++mi) {
        Move& m = moves[mi];
        m.score = 0;

        // 1) En passant kao specijalni capture (ciljno polje prazno, ali vrijedi kao uzimanje pješaka).
        if (board.epSquare == m.toSquare) {
            m.score = MVV_LVA_PAWN_EXCHANGE;
            continue;
        }

        // 2) Običan capture? (cilj zauzet protivničkom figurom)
        const bool isCapture = (enemyOcc & SQUARE_BITBOARDS[m.toSquare]) != 0;
        if (isCapture) {
            int capturedPiece = -1;
            Bitboard toBB = SQUARE_BITBOARDS[m.toSquare];

            // Nađi koji je tip figure uzet (samo materijalni tip: pawn..queen)
            // Za protivnika pogledaj 5 prvih bitboarda odgovarajuće boje.
            const Bitboard* bb = board.turn ? &board.pawn_B : &board.pawn_W;
            for (int pt = 0; pt < 5; ++pt) {     // pawn..queen
                if (toBB & bb[pt]) { capturedPiece = pt; break; }
            }

            if (capturedPiece >= 0 && capturedPiece < 5 &&
                m.pieceType >= 0 && m.pieceType < 12) {
                m.score = VICTIMS[capturedPiece] - ATTACKERS[m.pieceType];
            }
            else {
                m.score = 1; // fallback
            }
            continue;
        }

        // 3) Tihi potez (score ostaje 0) — kasnije možeš dodati killers/history.
    }
}

int select_move(Move* moves, int cmoves) {
    int bestScore = 0;
    int bestIdx = -1;

    for (int i = 0; i < cmoves; ++i) {
        if (!moves[i].exhausted && moves[i].score >= bestScore) {
            bestScore = moves[i].score;
            bestIdx = i;
        }
    }
    if (bestIdx != -1) moves[bestIdx].exhausted = true;
    return bestIdx;
}
