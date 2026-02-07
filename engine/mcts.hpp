#pragma once

#include <vector>
#include <memory>

#include "typedefs.hpp"
#include "experience.hpp"

// MCTS Node (CPU tree). GPU se koristi za batch evaluacije leaf-ova.
//
// Parallel MCTS (batched leaf-parallel) koristi "inFlight" brojač (O iz literature)
// da bi tree-policy odmah "vidio" da je neka grana već rezervisana u tekućem batch-u.
// Time se smanjuje stampede (više simulacija u isti child).
struct MctsNode {
    Board state{};
    Move  moveFromParent{};
    bool  hasMoveFromParent = false;

    int visits = 0;          // completed simulations
    int inFlight = 0;        // O: broj nedovršenih simulacija kroz ovaj node (rezervacije)
    double valueSum = 0.0;   // suma vrijednosti (cp), uvijek u root perspektivi

    bool initialized = false;
    bool terminal = false;
    int  terminalResult = UN_DETERMINED;

    double prior = 0.0;          // P za PUCT (0..1)
    int initialUnexpanded = 0;   // koliko je poteza bilo na početku (za rank prior)
    double priorNorm = 1.0;      // normalizacija (harmonic sum)

    std::vector<Move> unexpanded;
    std::vector<std::unique_ptr<MctsNode>> children;

    MctsNode() = default;
    explicit MctsNode(const Board& b) : state(b) {}
};

class MctsSearch {
public:
    MctsSearch(const Board& root, int rootTurn, int iterations);

    // Pokreni MCTS. Interno koristi batched leaf-parallel pristup (CPU tree + GPU eval).
    void run();

    Move bestMove() const;
    int  bestChildScoreCp() const;

    // stats root djece (za ExperienceBook update)
    std::vector<BookChildStat> rootChildStats() const;

    int iterationsUsed() const { return iterations; }

private:
    int rootTurn;
    int iterations;

    // Koliko leaf-ova skupljamo prije GPU batch evaluacije.
    // 64 je dobar default za NNUE kernel (dovoljno da saturira GPU, a da ne ubije latenciju).
    int batchSize = 256;

    double C = 1.2;
    int mateScore = 100000;

    std::unique_ptr<MctsNode> rootNode;

private:
    void ensureInitialized(MctsNode* node);

    double terminalValue(int gameResult) const;
    double staticEvalRootPerspective(const Board& b) const;
    double evaluateLeaf_OnePlyMinimax(const Board& b);
    double qsearchCaptures(const Board& b, int depthLeft);

    // PUCT selection (koristi visits+inFlight da izbjegne stampede u batch-u)
    MctsNode* selectChildUCB(MctsNode* node) const;

    // Stari single-sim API (ostavljen radi debug-a / fallback-a)
    void simulateOnce();

    // NEW: batched leaf-parallel runner
    void runBatched();

    // NEW: selection+optional expansion do leaf-a, bez evaluacije/backprop-a.
    // Popuni path (uključuje leaf node), vrati leaf board za evaluaciju ili terminal info.
    void reserveOneSimulation(std::vector<MctsNode*>& outPath,
        Board& outLeafBoard,
        bool& outIsTerminal,
        int& outTerminalResult);

    // NEW: završi simulaciju (decrement inFlight, increment visits, add valueSum)
    void completeSimulation(const std::vector<MctsNode*>& path, double value);
};
