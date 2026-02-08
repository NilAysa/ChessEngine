#pragma once

#include "typedefs.hpp"

extern int SEARCH_NODES_SEARCHED;
extern Move SEARCH_BEST_MOVE;

int search(Board board, int depth);
