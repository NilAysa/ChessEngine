#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include "movegen.hpp"
#include "board.hpp"
#include "utils.hpp"
#include <fen.hpp>

#define getMove(from, to, promo, castling, pType) Move{from, to, promo, castling, pType}
#define addMove moves[length] = move; length++;

// maske koje se koriste za provjere 
Bitboard PAWN_START_WHITE = 0xFF00ULL;
Bitboard PAWN_START_BLACK = 0x00FF000000000000ULL;
Bitboard WHITE_CASTLE_K_PATH = 0b110;
Bitboard WHITE_CASTLE_Q_PATH = 0b1110000;
Bitboard BLACK_CASTLE_K_PATH = 0b11ULL << G8;
Bitboard BLACK_CASTLE_Q_PATH = 0b111ULL << D8;
int WHITE_PROMOTIONS[4] = { QUEEN_W, BISHOP_W, KNIGHT_W, ROOK_W };
int BLACK_PROMOTIONS[4] = { QUEEN_B, BISHOP_B, KNIGHT_B, ROOK_B };
int NO_PROMOTION = -1;
int NOT_CASTLE = 0;

// fiksne attack matrice za kralja, konja, pješaka i dodatne matrice za topa i lovca koje koriste magics za popunjavanje
Bitboard PAWN_W_ATTACKS_EAST[64];
Bitboard PAWN_W_ATTACKS_WEST[64];
Bitboard PAWN_B_ATTACKS_EAST[64];
Bitboard PAWN_B_ATTACKS_WEST[64];
Bitboard KNIGHT_MOVEMENT[64];
Bitboard BISHOP_MOVEMENT[64];
Bitboard ROOK_MOVEMENT[64];
Bitboard KING_MOVEMENT[64];
Bitboard BISHOP_ATTACKS[64][512];
Bitboard ROOK_ATTACKS[64][4096];


// koliko bita se gleda za magics indeksiranje
int ROOK_RELEVANT_BITS[64] = {
    12, 11, 11, 11, 11, 11, 11, 12,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    12, 11, 11, 11, 11, 11, 11, 12
};
int BISHOP_RELEVANT_BITS[64] = {
    6, 5, 5, 5, 5, 5, 5, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 5, 5, 5, 5, 5, 5, 6
};

// za svako polje stvara bitboard kuda se kralj iz tog polja smije kretati
void initKingMovementTable() {
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard moves = 0ULL;
        int file = sq % 8;

        if (sq <= A7) moves |= 1ULL << (sq + 8);  // UP
        if (sq >= H2) moves |= 1ULL << (sq - 8);  // DOWN
        if (file < A) moves |= 1ULL << (sq + 1);  // LEFT
        if (file > H) moves |= 1ULL << (sq - 1);  // RIGHT
        if (file < A && sq <= A7) moves |= 1ULL << (sq + 9);  // UP LEFT
        if (file > H && sq <= A7) moves |= 1ULL << (sq + 7);  // UP RIGHT
        if (file < A && sq >= H2) moves |= 1ULL << (sq - 7);  // DOWN LEFT
        if (file > H && sq >= H2) moves |= 1ULL << (sq - 9);  // DOWN RIGHT

        KING_MOVEMENT[sq] = moves;
    }
}

// za svako polje na ploci testira moguci skok konja i upisuje u bitboard-ove
void initKnightMovementTable() {
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard bb = 0ULL;
        int file = sq % 8;

        if (file != A && sq <= A6) bb |= 1ULL << (sq + 17);
        if (file != H && sq <= A6) bb |= 1ULL << (sq + 15);
        if (file != A && sq >= A2) bb |= 1ULL << (sq - 15);
        if (file != H && sq > H3)  bb |= 1ULL << (sq - 17);
        if (file < B && sq <= A7)  bb |= 1ULL << (sq + 10);
        if (file < B && sq >= H2)  bb |= 1ULL << (sq - 6);
        if (file > G && sq >= H2)  bb |= 1ULL << (sq - 10);
        if (file > G && sq <= A7)  bb |= 1ULL << (sq + 6);

        KNIGHT_MOVEMENT[sq] = bb;
    }
}

// popunjava cetiri tabele koje unaprijed racunaju dijagonalne napade sa svake pozicije
void initPawnAttackTables() {
    for (int sq = 0; sq < 64; ++sq) {
        int file = sq % 8;

        Bitboard attack = 0ULL;
        if (file != A && sq <= H8) attack |= 1ULL << (sq + 9);
        PAWN_W_ATTACKS_WEST[sq] = attack;

        attack = 0ULL;
        if (file != H && sq <= H8) attack |= 1ULL << (sq + 7);
        PAWN_W_ATTACKS_EAST[sq] = attack;

        attack = 0ULL;
        if (file != H && sq >= H2) attack |= 1ULL << (sq - 9);
        PAWN_B_ATTACKS_WEST[sq] = attack;

        attack = 0ULL;
        if (file != A && sq >= H2) attack |= 1ULL << (sq - 7);
        PAWN_B_ATTACKS_EAST[sq] = attack;
    }
}


// funkcija vraca bitboard koji simulira jednu mogucu konfiguraciju prepreka na poljima
// kuda bi se inace mogao kretati lovac ili top
Bitboard occupancyMask(int index, int bits, Bitboard attackMask) {
    Bitboard occupancy = 0ULL;
    for (int i = 0; i < bits; ++i) {
        int square = countTrailingZeros(attackMask);
        attackMask = toggleBit(attackMask, square);
        if (index & (1 << i)) occupancy |= 1ULL << square;
    }
    return occupancy;
}

// generiše kretanje kroz plocu za lovca i topa bez lookup tabele tj bez magic vrijednosti
// krece se od pocetnog polja u legalnom pravcu kretanja dok ne dodje do prepreke
Bitboard bishopAttacksOnTheFly(int square, Bitboard block) {
    Bitboard attacks = 0ULL;
    int tr = square / 8, tf = square % 8;

    for (int r = tr + 1, f = tf + 1; r <= 7 && f <= 7; ++r, ++f) {
        attacks |= 1ULL << (r * 8 + f);
        if (block & (1ULL << (r * 8 + f))) break;
    }
    for (int r = tr + 1, f = tf - 1; r <= 7 && f >= 0; ++r, --f) {
        attacks |= 1ULL << (r * 8 + f);
        if (block & (1ULL << (r * 8 + f))) break;
    }
    for (int r = tr - 1, f = tf + 1; r >= 0 && f <= 7; --r, ++f) {
        attacks |= 1ULL << (r * 8 + f);
        if (block & (1ULL << (r * 8 + f))) break;
    }
    for (int r = tr - 1, f = tf - 1; r >= 0 && f >= 0; --r, --f) {
        attacks |= 1ULL << (r * 8 + f);
        if (block & (1ULL << (r * 8 + f))) break;
    }
    return attacks;
}

Bitboard rookAttacksOnTheFly(int square, Bitboard block) {
    Bitboard attacks = 0ULL;
    int tr = square / 8, tf = square % 8;

    for (int r = tr + 1; r <= 7; ++r) {
        attacks |= 1ULL << (r * 8 + tf);
        if (block & (1ULL << (r * 8 + tf))) break;
    }
    for (int r = tr - 1; r >= 0; --r) {
        attacks |= 1ULL << (r * 8 + tf);
        if (block & (1ULL << (r * 8 + tf))) break;
    }
    for (int f = tf + 1; f <= 7; ++f) {
        attacks |= 1ULL << (tr * 8 + f);
        if (block & (1ULL << (tr * 8 + f))) break;
    }
    for (int f = tf - 1; f >= 0; --f) {
        attacks |= 1ULL << (tr * 8 + f);
        if (block & (1ULL << (tr * 8 + f))) break;
    }
    return attacks;
}

// vraca sva moguca kretanja topa i lovca, kao da je ploca prazna
Bitboard bishopMovement(int square) {
    int tf = square % 8, tr = square / 8;
    Bitboard movement = 0ULL;
    // Up-right
    for (int r = tr + 1, f = tf + 1; r <= 7 && f <= 7; ++r, ++f) movement |= (1ULL << (r * 8 + f));
    // Up-left
    for (int r = tr + 1, f = tf - 1; r <= 7 && f >= 0; ++r, --f) movement |= (1ULL << (r * 8 + f));
    // Down-right
    for (int r = tr - 1, f = tf + 1; r >= 0 && f <= 7; --r, ++f) movement |= (1ULL << (r * 8 + f));
    // Down-left
    for (int r = tr - 1, f = tf - 1; r >= 0 && f >= 0; --r, --f) movement |= (1ULL << (r * 8 + f));
    return movement;
}

Bitboard rookMovement(int square) {
    int tf = square % 8, tr = square / 8;
    Bitboard movement = 0ULL;
    // Up
    for (int r = tr + 1; r <= 7; ++r) movement |= (1ULL << (r * 8 + tf));
    // Down
    for (int r = tr - 1; r >= 0; --r) movement |= (1ULL << (r * 8 + tf));
    // Right
    for (int f = tf + 1; f <= 7; ++f) movement |= (1ULL << (tr * 8 + f));
    // Left
    for (int f = tf - 1; f >= 0; --f) movement |= (1ULL << (tr * 8 + f));
    return movement;
}

// funkcija koja generise lookup tabele za napade lovca i topa koristeći bitboard sa svim mogućim
// kretanjima i bitboard sa svim kombinacijama prepreka
void initBishopRookAttackTables() {
    for (int square = 0; square < 64; square++) {
        // kretanje za praznu plocu
        BISHOP_MOVEMENT[square] = bishopMovement(square);
        ROOK_MOVEMENT[square] = rookMovement(square);

        Bitboard bishopMask = BISHOP_MOVEMENT[square];
        Bitboard rookMask = ROOK_MOVEMENT[square];

        int bishopRelevantBits = BISHOP_RELEVANT_BITS[square];
        int rookRelevantBits = ROOK_RELEVANT_BITS[square];

        int bishopVariations = 1 << bishopRelevantBits;
        int rookVariations = 1 << rookRelevantBits;

        for (int i = 0; i < bishopVariations; i++) {
            Bitboard occupancy = occupancyMask(i, bishopRelevantBits, bishopMask);
            Bitboard magic_index = occupancy * BISHOP_MAGICS[square] >> (64 - bishopRelevantBits);
            BISHOP_ATTACKS[square][magic_index] = bishopAttacksOnTheFly(square, occupancy);
        }

        for (int i = 0; i < rookVariations; i++) {
            Bitboard occupancy = occupancyMask(i, rookRelevantBits, rookMask);
            Bitboard magic_index = occupancy * ROOK_MAGICS[square] >> (64 - rookRelevantBits);
            ROOK_ATTACKS[square][magic_index] = rookAttacksOnTheFly(square, occupancy);
        }
    }
}

// POTPUNO IZBACI magic lookup za lovca i topa
Bitboard getBishopAttacks(int square, Bitboard occupancy) {
    // occupancy je board->occupancy (sve figure), baš kako i sada prosljeđuješ
    return bishopAttacksOnTheFly(square, occupancy);
}

Bitboard getRookAttacks(int square, Bitboard occupancy) {
    return rookAttacksOnTheFly(square, occupancy);
}


Bitboard getKingMask(const Board& board) {
    int kingSquare = board.turn ? board.whiteKingSq : board.blackKingSq;
    Bitboard opponentOccupancy = board.turn ? board.occupancyWhite : board.occupancyBlack;
    return KING_MOVEMENT[kingSquare] & ~opponentOccupancy;
}

static bool isSlidingPiece(int pieceType) {
    return pieceType == BISHOP_W || pieceType == BISHOP_B ||
        pieceType == ROOK_W || pieceType == ROOK_B ||
        pieceType == QUEEN_W || pieceType == QUEEN_B;
}

static bool isPathClear(const Board& board, int from, int to) {
    int fromFile = from % 8;
    int fromRank = from / 8;
    int toFile = to % 8;
    int toRank = to / 8;

    int df = (toFile > fromFile) - (toFile < fromFile);   // -1, 0 ili 1
    int dr = (toRank > fromRank) - (toRank < fromRank);   // -1, 0 ili 1

    // nije ni prava ni dijagonala → nije sliding potez
    if (df == 0 && dr == 0) return false;
    if (!((df == 0) || (dr == 0) || (abs(df) == abs(dr)))) return false;

    int sq = from + dr * 8 + df;
    while (sq != to) {
        if (getBit(board.occupancy, sq))
            return false; // neka figura blokira
        sq += dr * 8 + df;
    }
    return true;
}


void validateMove(const Board& board, Move* move) {
    // Za lovca, topa i kraljicu dodatno provjeri da ne preskaču figure
    if (isSlidingPiece(move->pieceType)) {
        if (!isPathClear(board, move->fromSquare, move->toSquare)) {
            move->validation = ILLEGAL;
            return;
        }
    }

    if (move->castle) {
        int sq = 0;
        if (move->castle == K) sq = F1;
        if (move->castle == Q) sq = D1;
        if (move->castle == k) sq = F8;
        if (move->castle == q) sq = D8;

        if (isSquareAttacked(board, sq)) {
            move->validation = ILLEGAL;
            return;
        }

        bool isInCheck = isSquareAttacked(board, board.turn ? board.whiteKingSq : board.blackKingSq);
        if (isInCheck) {
            move->validation = ILLEGAL;
            return;
        }
    }

    Board cpy = board;
    pushMove(&cpy, *move);

    cpy.turn = board.turn;

    if (inCheck(cpy))
        move->validation = ILLEGAL;
    else
        move->validation = LEGAL;
}

void pawnSingleAndDblPushes(const Board& board, Bitboard* single, Bitboard* dbl) {
    *single = board.turn ? (board.pawn_W << 8) : (board.pawn_B >> 8);
    *single &= ~board.occupancy;

    *dbl = board.turn
        ? ((board.pawn_W & PAWN_START_WHITE) << 16)
        : ((board.pawn_B & PAWN_START_BLACK) >> 16);
    *dbl &= ~board.occupancy;

    *dbl = board.turn ? (*dbl >> 8) : (*dbl << 8);
    *dbl &= *single;
    *dbl = board.turn ? (*dbl << 8) : (*dbl >> 8);
}

void addPawnAdvanceWithPossiblePromos(const Board& board,
    bool isPromoting,
    int turn, int from, int to,
    Move moves[], int* indx) {
    if (isPromoting) {
        for (int i = 0; i < 4; i++) {
            Move move = getMove(from, to,
                turn ? WHITE_PROMOTIONS[i] : BLACK_PROMOTIONS[i],
                NOT_CASTLE,
                board.turn ? PAWN_W : PAWN_B);
            validateMove(board, &move);
            if (move.validation == ILLEGAL) continue;

            moves[*indx] = move;
            (*indx)++;
        }
        return;
    }

    Move move = getMove(from, to, NO_PROMOTION, NOT_CASTLE,
        board.turn ? PAWN_W : PAWN_B);
    validateMove(board, &move);
    if (move.validation == ILLEGAL) return;

    moves[*indx] = move;
    (*indx)++;
}

void initMoveGeneration() {
    initKnightMovementTable();
    initKingMovementTable();
    initPawnAttackTables();
    initBishopRookAttackTables();
}

bool isSquareAttacked(const Board& board, int square) {
    const bool blackToMove = board.turn;

    Bitboard sqBb = SQUARE_BITBOARDS[square];
    Bitboard pawn = blackToMove ? board.pawn_B : board.pawn_W;
    Bitboard king = blackToMove ? board.king_B : board.king_W;
    Bitboard knight = blackToMove ? board.knight_B : board.knight_W;
    Bitboard bishop = blackToMove ? board.bishop_B : board.bishop_W;
    Bitboard rook = blackToMove ? board.rook_B : board.rook_W;
    Bitboard queen = blackToMove ? board.queen_B : board.queen_W;

    while (queen) {
        int sq = countTrailingZeros(queen);
        Bitboard attacks = getBishopAttacks(sq, board.occupancy)
            | getRookAttacks(sq, board.occupancy);
        if (attacks & sqBb) return true;
        queen &= queen - 1;
    }
    while (bishop) {
        int sq = countTrailingZeros(bishop);
        if (getBishopAttacks(sq, board.occupancy) & sqBb) return true;
        bishop &= bishop - 1;
    }
    while (rook) {
        int sq = countTrailingZeros(rook);
        if (getRookAttacks(sq, board.occupancy) & sqBb) return true;
        rook &= rook - 1;
    }
    while (knight) {
        int sq = countTrailingZeros(knight);
        if (KNIGHT_MOVEMENT[sq] & sqBb) return true;
        knight &= knight - 1;
    }
    while (pawn) {
        int sq = countTrailingZeros(pawn);
        if (blackToMove) {
            if ((PAWN_B_ATTACKS_EAST[sq] | PAWN_B_ATTACKS_WEST[sq]) & sqBb) return true;
        }
        else {
            if ((PAWN_W_ATTACKS_EAST[sq] | PAWN_W_ATTACKS_WEST[sq]) & sqBb) return true;
        }
        pawn &= pawn - 1;
    }
    while (king) {
        int sq = countTrailingZeros(king);
        if (KING_MOVEMENT[sq] & sqBb) return true;
        king &= king - 1;
    }
    return false;
}

int legalMoves(Board* board, Move moves[]) {
    int length = 0;

    int kingSquare = board->turn ? board->whiteKingSq : board->blackKingSq;
    Bitboard friendlyOccupancy = board->turn ? board->occupancyWhite : board->occupancyBlack;
    Bitboard bishopBitboard = board->turn ? board->bishop_W : board->bishop_B;
    Bitboard rookBitboard = board->turn ? board->rook_W : board->rook_B;
    Bitboard queenBitboard = board->turn ? board->queen_W : board->queen_B;
    Bitboard knightBitboard = board->turn ? board->knight_W : board->knight_B;
    Bitboard pawnMask = board->turn ? board->pawn_W : board->pawn_B;

    Bitboard attackMask = 0;

    // Generate pawn pushes
    Bitboard singlePush, doublePush;
    pawnSingleAndDblPushes(*board, &singlePush, &doublePush);

    // En-passant polje kao bitboard
    Bitboard epSquare = (board->epSquare == -1)
        ? 0ULL
        : SQUARE_BITBOARDS[board->epSquare];

    // Protivničke figure + potencijalni en-passant
    Bitboard occForCaptures =
        epSquare | (board->turn ? board->occupancyBlack : board->occupancyWhite);

    // ======================
    //   PAWN CAPTURES
    // ======================
    while (pawnMask) {
        int sq = countTrailingZeros(pawnMask);

        // Da li ovaj pijun može da promoviše
        bool isPromoting = board->turn
            ? (SQUARE_BITBOARDS[sq] & RANK_7)
            : (SQUARE_BITBOARDS[sq] & RANK_1);

        if (board->turn) {
            // bijeli: napadi prema gore (iz njegove perspektive)
            Bitboard eastAttacks = PAWN_W_ATTACKS_EAST[sq] & occForCaptures;
            attackMask |= PAWN_W_ATTACKS_EAST[sq];
            attackMask |= PAWN_W_ATTACKS_WEST[sq];

            if (eastAttacks) {
                int toSquare = sq + 7;  // isto kao u tvojoj originalnoj logici
                addPawnAdvanceWithPossiblePromos(*board, isPromoting, board->turn,
                    sq, toSquare, moves, &length);
            }

            Bitboard westAttacks = PAWN_W_ATTACKS_WEST[sq] & occForCaptures;
            if (westAttacks) {
                int toSquare = sq + 9;
                addPawnAdvanceWithPossiblePromos(*board, isPromoting, board->turn,
                    sq, toSquare, moves, &length);
            }
        }
        else {
            // crni
            Bitboard eastAttacks = PAWN_B_ATTACKS_EAST[sq] & occForCaptures;
            attackMask |= PAWN_B_ATTACKS_EAST[sq];
            attackMask |= PAWN_B_ATTACKS_WEST[sq];

            if (eastAttacks) {
                int toSquare = sq - 7;
                addPawnAdvanceWithPossiblePromos(*board, isPromoting, board->turn,
                    sq, toSquare, moves, &length);
            }

            Bitboard westAttacks = PAWN_B_ATTACKS_WEST[sq] & occForCaptures;
            if (westAttacks) {
                int toSquare = sq - 9;
                addPawnAdvanceWithPossiblePromos(*board, isPromoting, board->turn,
                    sq, toSquare, moves, &length);
            }
        }

        pawnMask &= pawnMask - 1;
    }

    // King napadna maska (svi potencijalni king moves, ne samo legalni)
    Bitboard kingMovesMask = getKingMask(*board);
    attackMask |= KING_MOVEMENT[kingSquare];

    // ======================
    //   PAWN SINGLE PUSHES
    // ======================
    while (singlePush) {
        int sq = countTrailingZeros(singlePush);
        int fromSquare = board->turn ? (sq - 8) : (sq + 8);

        bool isPromoting = board->turn
            ? (fromSquare >= H7 && fromSquare <= A7)
            : (fromSquare >= H2 && fromSquare <= A2);

        addPawnAdvanceWithPossiblePromos(*board, isPromoting, board->turn,
            fromSquare, sq, moves, &length);

        singlePush &= singlePush - 1;
    }

    // ======================
    //   PAWN DOUBLE PUSHES
    // ======================
    int pawnPieceType = board->turn ? PAWN_W : PAWN_B;
    while (doublePush) {
        int sq = countTrailingZeros(doublePush);
        int fromSquare = board->turn ? (sq - 16) : (sq + 16);

        Move move = getMove(fromSquare, sq, NO_PROMOTION, NOT_CASTLE, pawnPieceType);
        validateMove(*board, &move);
        if (move.validation == LEGAL) {
            addMove
        }
        doublePush &= doublePush - 1;
    }

    // ======================
    //   KING MOVES
    // ======================
    while (kingMovesMask) {
        int sq = countTrailingZeros(kingMovesMask);
        Move move = getMove(kingSquare, sq, NO_PROMOTION, NOT_CASTLE,
            board->turn ? KING_W : KING_B);
        validateMove(*board, &move);
        if (move.validation == LEGAL) {
            addMove
        }
        kingMovesMask &= kingMovesMask - 1;
    }

    // ======================
    //   BISHOP MOVES
    // ======================
    while (bishopBitboard) {
        int sq = countTrailingZeros(bishopBitboard);
        Bitboard attacks = getBishopAttacks(sq, board->occupancy);
        attackMask |= attacks;
        attacks &= ~friendlyOccupancy;

        while (attacks) {
            int to = countTrailingZeros(attacks);
            Move move = getMove(sq, to, NO_PROMOTION, NOT_CASTLE,
                board->turn ? BISHOP_W : BISHOP_B);
            validateMove(*board, &move);
            if (move.validation == LEGAL) {
                addMove
            }
            attacks &= attacks - 1;
        }
        bishopBitboard &= bishopBitboard - 1;
    }

    // ======================
    //   ROOK MOVES
    // ======================
    while (rookBitboard) {
        int sq = countTrailingZeros(rookBitboard);
        Bitboard attacks = getRookAttacks(sq, board->occupancy);
        attackMask |= attacks;
        attacks &= ~friendlyOccupancy;

        while (attacks) {
            int to = countTrailingZeros(attacks);
            Move move = getMove(sq, to, NO_PROMOTION, NOT_CASTLE,
                board->turn ? ROOK_W : ROOK_B);
            validateMove(*board, &move);
            if (move.validation == LEGAL) {
                addMove
            }
            attacks &= attacks - 1;
        }
        rookBitboard &= rookBitboard - 1;
    }

    // ======================
    //   QUEEN MOVES
    // ======================
    while (queenBitboard) {
        int sq = countTrailingZeros(queenBitboard);
        Bitboard rookAtt = getRookAttacks(sq, board->occupancy);
        Bitboard bishAtt = getBishopAttacks(sq, board->occupancy);
        Bitboard attacks = rookAtt | bishAtt;

        attackMask |= attacks;
        attacks &= ~friendlyOccupancy;

        while (attacks) {
            int to = countTrailingZeros(attacks);
            Move move = getMove(sq, to, NO_PROMOTION, NOT_CASTLE,
                board->turn ? QUEEN_W : QUEEN_B);
            validateMove(*board, &move);
            if (move.validation == LEGAL) {
                addMove
            }
            attacks &= attacks - 1;
        }
        queenBitboard &= queenBitboard - 1;
    }

    // ======================
    //   KNIGHT MOVES
    // ======================
    while (knightBitboard) {
        int from = countTrailingZeros(knightBitboard);

        // sve mete konja (za masku napada)
        attackMask |= KNIGHT_MOVEMENT[from];

        // legalne mete (bez naših figura)
        Bitboard attacks = KNIGHT_MOVEMENT[from] & ~friendlyOccupancy;

        while (attacks) {
            int to = countTrailingZeros(attacks);
            Move move = getMove(from, to, NO_PROMOTION, NOT_CASTLE,
                board->turn ? KNIGHT_W : KNIGHT_B);
            validateMove(*board, &move);
            if (move.validation == LEGAL) {
                addMove
            }
            attacks &= attacks - 1;
        }
        knightBitboard &= knightBitboard - 1;
    }

    // ======================
    //   CASTLING
    // ======================
    if (board->turn) {
        // bijeli
        if (board->castling & K && !(board->occupancy & WHITE_CASTLE_K_PATH)) {
            Move move = getMove(E1, G1, NO_PROMOTION, K, -1);
            validateMove(*board, &move);
            if (move.validation == LEGAL) {
                addMove
            }
        }
        if (board->castling & Q && !(board->occupancy & WHITE_CASTLE_Q_PATH)) {
            Move move = getMove(E1, C1, NO_PROMOTION, Q, -1);
            validateMove(*board, &move);
            if (move.validation == LEGAL) {
                addMove
            }
        }
    }
    else {
        // crni
        if (board->castling & k && !(board->occupancy & BLACK_CASTLE_K_PATH)) {
            Move move = getMove(E8, G8, NO_PROMOTION, k, -1);
            validateMove(*board, &move);
            if (move.validation == LEGAL) {
                addMove
            }
        }
        if (board->castling & q && !(board->occupancy & BLACK_CASTLE_Q_PATH)) {
            Move move = getMove(E8, C8, NO_PROMOTION, q, -1);
            validateMove(*board, &move);
            if (move.validation == LEGAL) {
                addMove
            }
        }
    }

    // Nakon sto si generisao sve legalne poteze, ako ti treba napadna maska protivnika,
    // izracunaj je eksplicitno:
    board->attacks = computeAttacks(*board);
    return length;

}
bool inCheck(const Board& board) {
    // napadi protivnika na trenutnog igraca
    Bitboard att = computeAttacks(board);
    int kingSq = board.turn ? board.whiteKingSq : board.blackKingSq;
    return (att & SQUARE_BITBOARDS[kingSq]) != 0;
}


