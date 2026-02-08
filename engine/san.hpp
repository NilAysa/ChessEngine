#pragma once
#include "typedefs.hpp"

// UCI helperi 

void pushSan(Board* board, const char* san);
void sanToMove(const Board& board, Move* move, const char* san);
void moveToSan(const Move& move, char san[]);
