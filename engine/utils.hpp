#pragma once

#include "typedefs.hpp"
#include <string>
#include <cstdint>
#include <bit>

void printMoves(Move moves[], int length);
void printBitboard(Bitboard bb);
void printBits(Bitboard bb);
int countTrailingZeros(uint64_t x);

inline int countTrailingZeros(uint64_t x) {
    return x == 0 ? 64 : std::countr_zero(x);
}