#include "mcts.hpp"
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>

#include "movegen.hpp"
#include "board.hpp"
#include "evaluation.hpp"
#include "moveorderer.hpp"
#include "tt.hpp"

MctsSearch::MctsSearch(const Board& root, int rootTurn_, int iterations_)
    : rootTurn(rootTurn_), iterations(iterations_) {
    rootNode = std::make_unique<MctsNode>(root);
}

void MctsSearch::run() {
    for (int i = 0; i < iterations; ++i) {
        simulateOnce();
    }
}

Move MctsSearch::bestMove() const {
    if (!rootNode) return Move{};

    if (rootNode->children.empty()) {
        Board b = rootNode->state;
        Move moves[256];
        int n = legalMoves(&b, moves);
        return (n > 0) ? moves[0] : Move{};
    }

    const MctsNode* best = nullptr;
    double bestQ = -std::numeric_limits<double>::infinity();
    int bestVisits = -1;

    for (const auto& ch : rootNode->children) {
        if (ch->visits <= 0) continue;
        double q = ch->valueSum / (double)ch->visits;
        if (q > bestQ || (q == bestQ && ch->visits > bestVisits)) {
            bestQ = q;
            bestVisits = ch->visits;
            best = ch.get();
        }
    }

    if (!best) {
        for (const auto& ch : rootNode->children) {
            if (ch->visits > bestVisits) {
                bestVisits = ch->visits;
                best = ch.get();
            }
        }
    }

    return (best && best->hasMoveFromParent) ? best->moveFromParent : Move{};
}

int MctsSearch::bestChildScoreCp() const {
    if (!rootNode || rootNode->children.empty()) return 0;

    const MctsNode* best = nullptr;
    double bestQ = -std::numeric_limits<double>::infinity();
    int bestVisits = -1;

    for (const auto& ch : rootNode->children) {
        if (ch->visits <= 0) continue;
        double q = ch->valueSum / (double)ch->visits;
        if (q > bestQ || (q == bestQ && ch->visits > bestVisits)) {
            bestQ = q;
            bestVisits = ch->visits;
            best = ch.get();
        }
    }

    if (!best) return 0;

    double mean = best->valueSum / (double)best->visits;
    if (mean > 32000) mean = 32000;
    if (mean < -32000) mean = -32000;
    return (int)std::lround(mean);
}

std::vector<BookChildStat> MctsSearch::rootChildStats() const {
    std::vector<BookChildStat> out;
    if (!rootNode) return out;

    out.reserve(rootNode->children.size());
    for (const auto& ch : rootNode->children) {
        if (!ch->hasMoveFromParent) continue;
        if (ch->visits <= 0) continue;

        BookChildStat st;
        st.m = ch->moveFromParent;
        st.visits = ch->visits;
        st.meanQ = ch->valueSum / (double)ch->visits; // cp
        out.push_back(st);
    }
    return out;
}

double MctsSearch::terminalValue(int gameResult) const {
    if (gameResult == DRAW) return 0.0;
    int winnerTurn = (gameResult == WHITE_WIN) ? WHITE : BLACK;
    return (winnerTurn == rootTurn) ? (double)mateScore : (double)-mateScore;
}

double MctsSearch::staticEvalRootPerspective(const Board& b) const {
    int evWhite = evaluate(b, UN_DETERMINED);
    return (rootTurn == WHITE) ? (double)evWhite : (double)-evWhite;
}

double MctsSearch::evaluateLeaf_OnePlyMinimax(const Board& b) {
    Board tmp = b;

    Move moves[256];
    int n = legalMoves(&tmp, moves);
    int res = result(tmp, moves, n);

    if (res != UN_DETERMINED)
        return terminalValue(res);

    const bool rootToMove = (tmp.turn == rootTurn);

    double best = rootToMove
        ? -std::numeric_limits<double>::infinity()
        : std::numeric_limits<double>::infinity();

    for (int i = 0; i < n; ++i) {
        Board child = tmp;
        pushMove(&child, moves[i]);

        double v = staticEvalRootPerspective(child);

        if (rootToMove) {
            if (v > best) best = v;
        }
        else {
            if (v < best) best = v;
        }
    }

    if (!std::isfinite(best))
        best = staticEvalRootPerspective(tmp);

    return best;
}

void MctsSearch::ensureInitialized(MctsNode* node) {
    if (!node || node->initialized) return;

    Board b = node->state;

    Move moves[256];
    int n = legalMoves(&b, moves);

    int res = result(b, moves, n);
    node->terminalResult = res;

    if (res != UN_DETERMINED) {
        node->terminal = true;
        node->initialized = true;
        return;
    }

    node->terminal = false;

    // 1) bazni ordering iz tvog moveorderer-a
    TTEntry entry = getTTEntry(b.hash);
    score_moves(b, entry, moves, n);

    // DEBUG: koliko book zna poteza za ovu poziciju
    static bool PRINT_BOOK_DEBUG = true;
    int known = ExperienceBook::instance().debugKnownMovesCount(b.hash);

    int hits = 0;
    int sumBonus = 0;
    int maxBonus = 0;

    int sumBase = 0;
    int sumAfter = 0;

    for (int i = 0; i < n; ++i) {
        // base score (bez book-a)
        sumBase += moves[i].score;

        int bonus = ExperienceBook::instance().bonusForMove(b.hash, moves[i]);
        if (bonus != 0) {
            hits++;
            sumBonus += bonus;
            if (bonus > maxBonus) maxBonus = bonus;
        }

        // apply
        moves[i].score += bonus;

        // after score (sa book-om)
        sumAfter += moves[i].score;
    }

    if (PRINT_BOOK_DEBUG && node && !node->hasMoveFromParent) {
        std::cout << "info string BOOK knownMoves=" << known
            << " hitsThisPos=" << hits
            << " sumBonus=" << sumBonus
            << " maxBonus=" << maxBonus
            << "\n";

        std::cout << "info string BOOK sumBaseScore=" << sumBase
            << " sumAfterScore=" << sumAfter
            << " delta=" << (sumAfter - sumBase)
            << "\n";
    }

    // 3) sortiraj ascending, pa pop_back daje najveći score prvo
    std::vector<Move> v(moves, moves + n);
    std::sort(v.begin(), v.end(), [](const Move& a, const Move& b) {
        return a.score < b.score;
        });

    node->unexpanded = std::move(v);
    node->initialized = true;
}

MctsNode* MctsSearch::selectChildUCB(MctsNode* node) const {
    if (!node || node->children.empty()) return nullptr;

    const bool rootToMove = (node->state.turn == rootTurn);

    double bestScore = -std::numeric_limits<double>::infinity();
    MctsNode* bestChild = nullptr;

    double logParent = std::log((double)node->visits + 1.0);

    for (auto& ch : node->children) {
        MctsNode* c = ch.get();

        double q = 0.0;
        if (c->visits > 0)
            q = (c->valueSum / (double)c->visits) / 1000.0; // normalize

        double u = C * std::sqrt(logParent / ((double)c->visits + 1.0));

        // root max, opponent min
        double score = (rootToMove ? q : -q) + u;

        if (score > bestScore) {
            bestScore = score;
            bestChild = c;
        }
    }

    return bestChild;
}

void MctsSearch::simulateOnce() {
    std::vector<MctsNode*> path;
    path.reserve(128);

    MctsNode* node = rootNode.get();
    path.push_back(node);

    while (true) {
        ensureInitialized(node);

        if (node->terminal) {
            double val = terminalValue(node->terminalResult);
            for (MctsNode* p : path) {
                p->visits += 1;
                p->valueSum += val;
            }
            return;
        }

        if (!node->unexpanded.empty()) {
            Move m = node->unexpanded.back();
            node->unexpanded.pop_back();

            Board childState = node->state;
            pushMove(&childState, m);

            auto child = std::make_unique<MctsNode>(childState);
            child->moveFromParent = m;
            child->hasMoveFromParent = true;

            MctsNode* childPtr = child.get();
            node->children.push_back(std::move(child));

            node = childPtr;
            path.push_back(node);

            double val = evaluateLeaf_OnePlyMinimax(node->state);

            for (MctsNode* p : path) {
                p->visits += 1;
                p->valueSum += val;
            }
            return;
        }

        MctsNode* next = selectChildUCB(node);
        if (!next) {
            double val = evaluateLeaf_OnePlyMinimax(node->state);
            for (MctsNode* p : path) {
                p->visits += 1;
                p->valueSum += val;
            }
            return;
        }

        node = next;
        path.push_back(node);
    }
}
