#pragma once
#include "typedefs.hpp"

// Zobrist tabele
extern Bitboard PIECES[12][64];
extern Bitboard EN_PASSANT[64];
extern Bitboard CASTLING[16];
extern Bitboard WHITE_TO_MOVE;

// Inicijalizacija svih Zobrist ključeva
void initZobrist();

// Izračun Zobrist heša za dato stanje table
Bitboard hash(const Board& board);  