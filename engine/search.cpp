#include "search.hpp"
#include "mcts.hpp"
#include "experience.hpp"
#include <iostream>

int SEARCH_NODES_SEARCHED = 0;
Move SEARCH_BEST_MOVE{};

static int depthToIterations(int depth) {
    if (depth < 1) depth = 1;
    if (depth > 25) depth = 25;        // sigurnosni clamp
    const int base = 8000;             // 6000–12000 po brzini
    return base * depth * depth;
}

int search(Board board, int depth) {
    int iterations = depthToIterations(depth);
    int rootTurn = board.turn;

    MctsSearch mcts(board, rootTurn, iterations);
    mcts.run();

    SEARCH_BEST_MOVE = mcts.bestMove();

    // Najsigurnije: prijavi koliko si iteracija dao MCTS-u
    SEARCH_NODES_SEARCHED = iterations;

    // BOOK update
    std::vector<BookChildStat> stats = mcts.rootChildStats();

    std::cout << "info string BOOKDBG hash=" << board.hash
        << " rootChildrenStats=" << (int)stats.size()
        << " storedBefore=" << ExperienceBook::instance().positionsStored()
        << "\n";

    ExperienceBook::instance().updateFromRootStats(board.hash, stats);

    std::cout << "info string BOOKDBG storedAfter="
        << ExperienceBook::instance().positionsStored()
        << "\n";

    return mcts.bestChildScoreCp();
}
