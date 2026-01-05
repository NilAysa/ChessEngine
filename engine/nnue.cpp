#include <vector>
#include <cmath>
#include <cstdint>

#include "nnue.hpp"
#include "utils.hpp"
#include "typedefs.hpp"

// GENERATED WEIGHTS (iz Colab-a)
#include "nnue_weights_generated.hpp"

// Feature layout (mora odgovarati notebook-u)
static constexpr int FEAT_PIECES = 12 * 64;   // 768
static constexpr int FEAT_STM = 768;       // 1
static constexpr int FEAT_CASTLE = 769;       // 4 (769..772)
static constexpr int FEAT_EP = 773;       // 8 (773..780)

// Engine Square enum: H1=0 ... A1=7; python-chess: A1=0 ... H1=7
static inline int sqEngineToPy(int sq_engine) {
    return sq_engine ^ 7; // flip file unutar ranka
}

void initNNUE() {
    // compile-time weights: nema runtime učitavanja
}

bool nnueHasWeights() {
    return true;
}

static inline void pushActive(std::vector<int>& act, int idx) {
    if (idx >= 0 && idx < NNUE_INPUT_DIM) act.push_back(idx);
}

static inline void addPiecesPlane(std::vector<int>& act, Bitboard bb, int planeBase) {
    while (bb) {
        int sq_engine = countTrailingZeros((uint64_t)bb);
        int sq_py = sqEngineToPy(sq_engine);
        pushActive(act, planeBase + sq_py);
        bb &= bb - 1;
    }
}

int nnueEvaluateWhitePerspective(const Board& b) {
    std::vector<int> act;
    act.reserve(48);

    // 12 planes: WP,WN,WB,WR,WQ,WK,BP,BN,BB,BR,BQ,BK
    addPiecesPlane(act, b.pawn_W, 0 * 64);
    addPiecesPlane(act, b.knight_W, 1 * 64);
    addPiecesPlane(act, b.bishop_W, 2 * 64);
    addPiecesPlane(act, b.rook_W, 3 * 64);
    addPiecesPlane(act, b.queen_W, 4 * 64);
    addPiecesPlane(act, b.king_W, 5 * 64);

    addPiecesPlane(act, b.pawn_B, 6 * 64);
    addPiecesPlane(act, b.knight_B, 7 * 64);
    addPiecesPlane(act, b.bishop_B, 8 * 64);
    addPiecesPlane(act, b.rook_B, 9 * 64);
    addPiecesPlane(act, b.queen_B, 10 * 64);
    addPiecesPlane(act, b.king_B, 11 * 64);

    // side-to-move (1 ako bijeli na potezu)
    if (b.turn == WHITE) pushActive(act, FEAT_STM);

    // castling bits u tvom engine-u su 4-bit: (bit0 WK, bit1 WQ, bit2 BK, bit3 BQ)
    if (b.castling & 0b0001) pushActive(act, FEAT_CASTLE + 0); // WK
    if (b.castling & 0b0010) pushActive(act, FEAT_CASTLE + 1); // WQ
    if (b.castling & 0b0100) pushActive(act, FEAT_CASTLE + 2); // BK
    if (b.castling & 0b1000) pushActive(act, FEAT_CASTLE + 3); // BQ

    // en-passant file one-hot (a..h)
    if (b.epSquare >= 0 && b.epSquare < 64) {
        int file_engine = (b.epSquare & 7); // engine: 0=h ... 7=a
        int file_py = 7 - file_engine;      // python: 0=a ... 7=h
        pushActive(act, FEAT_EP + file_py);
    }

    // Forward: 781 -> 128 (ReLU) -> 1
    float hidden[NNUE_HIDDEN_DIM];

    for (int i = 0; i < NNUE_HIDDEN_DIM; i++) {
        float sum = NNUE_B1[i];
        for (int idx : act) {
            sum += NNUE_W1[i][idx];
        }
        hidden[i] = (sum > 0.0f) ? sum : 0.0f;
    }

    float out = NNUE_B2;
    for (int i = 0; i < NNUE_HIDDEN_DIM; i++) {
        out += NNUE_W2[i] * hidden[i];
    }

    // U notebook-u si trenirala y_norm = y/400, pa vraćamo centipawns:
    float cp = out * 400.0f;

    if (cp > 100000.0f) cp = 100000.0f;
    if (cp < -100000.0f) cp = -100000.0f;

    return (int)std::lround((double)cp);
    //return 1234; - test

}
