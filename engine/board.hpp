#pragma once

#include "typedefs.hpp"
#include "bitboards.hpp"
#include <cstdint>
#include <iostream>

//vraca da li je bit 1 ili 0 na lokaciji odredjenoj indeksom square na ploci
#define getBit(bitboard, square) ((bitboard) & SQUARE_BITBOARDS[(square)])
//vraca novi Bitboard gdje je bit na lokaciji indeksa square na ploci postavljen na 0
#define popBit(bitboard, square) ((bitboard) & ~SQUARE_BITBOARDS[(square)])
//vraca novi Bitboard gdje je bit na lokaciji indeksa square na ploci postavljen na 1
#define setBit(bitboard, square) ((bitboard) | SQUARE_BITBOARDS[(square)])
//vraca Bitboard sa suprotnom vrijendosti bita na lokaciji indeksa square na ploci
#define toggleBit(bitboard, square) ((bitboard) ^ SQUARE_BITBOARDS[(square)])

int result(Board board, Move pseudoLegal[], int length);
void computeOccupancyMasks(Board* board);
void printBoard(const Board& board);
void pushMove(Board* board, const Move& move);

extern char SQUARE_NAMES[64][3];
