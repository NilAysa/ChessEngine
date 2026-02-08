#pragma once
#include "typedefs.hpp"

// Izračuna score za svaki potez u nizu (MVV-LVA, en passant).
void score_moves(const Board& board, Move* moves, int cmoves);

// Vrati indeks narednog poteza najvećeg score-a (koji nije exhausted), ili -1.
int select_move(Move* moves, int cmoves);
