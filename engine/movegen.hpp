#pragma once

#include <cstdint>
#include "typedefs.hpp"
#include "magics.hpp"

extern Bitboard PAWN_START_WHITE;
extern Bitboard PAWN_START_BLACK;

extern Bitboard PAWN_W_ATTACKS_EAST[64];
extern Bitboard PAWN_W_ATTACKS_WEST[64];
extern Bitboard PAWN_B_ATTACKS_EAST[64];
extern Bitboard PAWN_B_ATTACKS_WEST[64];

extern Bitboard KNIGHT_MOVEMENT[64];
extern Bitboard BISHOP_MOVEMENT[64];
extern Bitboard ROOK_MOVEMENT[64];
extern Bitboard KING_MOVEMENT[64];

void initMoveGeneration();

int legalMoves(Board* board, Move moves[]);

// sada prima const Board& umjesto kopije
bool isSquareAttacked(const Board& board, int square);

Bitboard getRookAttacks(int square, Bitboard occupancy);
Bitboard getBishopAttacks(int square, Bitboard occupancy);

// ako ga i dalje koristiš negdje, ostavi deklaraciju;
// ako ne postoji definicija, ili ukloni ovu liniju ili je prebaci u utils.hpp
int bitScanForward(Bitboard bb);
bool inCheck(const Board& board);
