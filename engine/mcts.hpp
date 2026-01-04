#pragma once

#include <vector>
#include <memory>

#include "typedefs.hpp"
#include "experience.hpp"

struct MctsNode {
    Board state{};
    Move  moveFromParent{};
    bool  hasMoveFromParent = false;

    int visits = 0;
    double valueSum = 0.0; // cp, root perspektiva

    bool initialized = false;
    bool terminal = false;
    int  terminalResult = UN_DETERMINED;

    std::vector<Move> unexpanded;
    std::vector<std::unique_ptr<MctsNode>> children;

    MctsNode() = default;
    explicit MctsNode(const Board& b) : state(b) {}
};

class MctsSearch {
public:
    MctsSearch(const Board& root, int rootTurn, int iterations);

    void run();

    Move bestMove() const;
    int  bestChildScoreCp() const;

    // NEW: stats root djece (za ExperienceBook update)
    std::vector<BookChildStat> rootChildStats() const;

    int iterationsUsed() const { return iterations; }

private:
    int rootTurn;
    int iterations;

    double C = 1.2;
    int mateScore = 100000;

    std::unique_ptr<MctsNode> rootNode;

private:
    void ensureInitialized(MctsNode* node);

    double terminalValue(int gameResult) const;
    double staticEvalRootPerspective(const Board& b) const;
    double evaluateLeaf_OnePlyMinimax(const Board& b);

    MctsNode* selectChildUCB(MctsNode* node) const;

    void simulateOnce();
};
