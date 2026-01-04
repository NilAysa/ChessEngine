#include "search.hpp"
#include "mcts.hpp"
#include "experience.hpp"
#include <iostream>

int SEARCH_NODES_SEARCHED = 0;
Move SEARCH_BEST_MOVE{};

static int depthToIterations(int depth) {
    if (depth < 1) depth = 1;
    return 16000 * depth;
}

int search(Board board, int depth) {
    int iterations = depthToIterations(depth);
    int rootTurn = board.turn;

    MctsSearch mcts(board, rootTurn, iterations);
    mcts.run();

    SEARCH_BEST_MOVE = mcts.bestMove();
    SEARCH_NODES_SEARCHED = mcts.iterationsUsed();

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
