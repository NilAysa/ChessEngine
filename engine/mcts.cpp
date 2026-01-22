#include "mcts.hpp"
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>

#include "movegen.hpp"
#include "board.hpp"
#include "evaluation.hpp"
#include "moveorderer.hpp"
#include "bitboards.hpp" // SQUARE_BITBOARDS
#include "batch_eval.hpp"


static constexpr double PW_A = 1.5; // jačina widening-a (1.0–2.5 je normalno)


MctsSearch::MctsSearch(const Board& root, int rootTurn_, int iterations_)
    : rootTurn(rootTurn_), iterations(iterations_) {
    rootNode = std::make_unique<MctsNode>(root);
}

void MctsSearch::run() {
    // GPU-friendly batching: collect multiple leaf expansions and evaluate all children in one big batch.
    // This keeps tree logic on CPU, but maximizes GPU utilization in batchEvaluateRootPerspective(...).
    constexpr int LEAF_BATCH = 1; // tune: 32..256 depending on GPU/CPU balance

    std::vector<std::vector<MctsNode*>> paths;
    std::vector<Board> leafStates;
    paths.reserve(LEAF_BATCH);
    leafStates.reserve(LEAF_BATCH);

    int done = 0;
    while (done < iterations) {
        paths.clear();
        leafStates.clear();

        // 1) Collect a batch of leaves (selection+expansion only)
        while ((int)leafStates.size() < LEAF_BATCH && done < iterations) {
            std::vector<MctsNode*> path;
            path.reserve(128);
            Board leaf{};
            bool isTerminal = false;
            double termVal = 0.0;

            simulateOnceCollect(path, leaf, isTerminal, termVal);
            ++done;

            if (isTerminal) {
                // terminal already evaluated -> immediate backprop
                for (MctsNode* p : path) {
                    p->visits += 1;
                    p->valueSum += termVal;
                }
            }
            else {
                paths.push_back(std::move(path));
                leafStates.push_back(leaf);
            }
        }

        if (leafStates.empty()) continue;

        // 2) Evaluate all collected leaves using one-ply minimax with ONE big batch-eval over ALL children.
        std::vector<double> leafVals(leafStates.size(), 0.0);
        evaluateLeaves_OnePlyMinimaxBatch(leafStates, leafVals);

        // 3) Backprop values
        for (size_t i = 0; i < leafVals.size(); ++i) {
            double val = leafVals[i];
            for (MctsNode* p : paths[i]) {
                p->visits += 1;
                p->valueSum += val;
            }
        }
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
    int evWhite = evaluateLeaf(b, UN_DETERMINED);
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

    // 1) Napravi sve child boardove
    std::vector<Board> childBoards;
    childBoards.reserve((size_t)n);

    for (int i = 0; i < n; ++i) {
        Board child = tmp;
        pushMove(&child, moves[i]);
        childBoards.push_back(child);
    }

    // 2) Batch evaluacija (CPU sada, GPU kasnije)
    std::vector<double> scores((size_t)n, 0.0);
    batchEvaluateRootPerspective(childBoards.data(), n, rootTurn, scores.data());

    // 3) Minimax (root perspective)
    for (int i = 0; i < n; ++i) {
        double v = scores[(size_t)i];
        if (rootToMove) {
            if (v > best) best = v;
        }
        else {
            if (v < best) best = v;
        }
    }

    if (!std::isfinite(best))
        best = staticEvalRootPerspective(tmp);

    // quiescence-light (4 ply capture-only)
    double q = qsearchCaptures(tmp, 4);

    // blend: quiescence ima prednost jer je taktički stabilniji
    double out = 0.25 * best + 0.75 * q;
    return out;
}

static double clamp01(double x) {
    if (x > 1.0) return 1.0;
    if (x < -1.0) return -1.0;
    return x;
}

// Quiescence-light: samo capture potezi, ograničeno dubinom
double MctsSearch::qsearchCaptures(const Board& b, int depthLeft) {
    // "stand pat"
    double stand = staticEvalRootPerspective(b);
    double best = stand;

    if (depthLeft <= 0) return best;

    Board tmp = b;
    Move moves[256];
    int n = legalMoves(&tmp, moves);

    // filtriraj samo capture (uklj. en-passant)
    Bitboard enemyOcc = tmp.turn ? tmp.occupancyBlack : tmp.occupancyWhite;

    bool anyCapture = false;
    for (int i = 0; i < n; ++i) {
        bool cap = (enemyOcc & SQUARE_BITBOARDS[moves[i].toSquare]) != 0;
        if (tmp.epSquare == moves[i].toSquare) cap = true;
        if (!cap) continue;

        anyCapture = true;

        Board child = tmp;
        pushMove(&child, moves[i]);

        // minimax po tome ko je na potezu (root perspektiva)
        double v = qsearchCaptures(child, depthLeft - 1);

        if (tmp.turn == rootTurn) {
            if (v > best) best = v;
        }
        else {
            if (v < best) best = v;
        }
    }

    if (!anyCapture) return stand;
    return best;
}

static inline int file_of(int sq) { return sq & 7; }      // 0..7
static inline int rank_of(int sq) { return sq >> 3; }     // 0..7

static inline bool is_start_square_minor(int pieceType, int fromSq) {
    // pieceType kod tebe: 0..5 bijele, 6..11 crne (pawn..king)
    const int pt = pieceType % 6; // 1=knight,2=bishop,3=rook,4=queen,5=king
    if (pt != 1 && pt != 2) return false;

    // bijele minor figure start na r=0: b1 g1 c1 f1
    // crne minor figure start na r=7: b8 g8 c8 f8
    const int r = rank_of(fromSq);
    if (pieceType < 6) { // white
        if (r != 0) return false;
        return (fromSq == 1 || fromSq == 6 || fromSq == 2 || fromSq == 5);
    }
    else { // black
        if (r != 7) return false;
        return (fromSq == 57 || fromSq == 62 || fromSq == 58 || fromSq == 61);
    }
}

static inline int positional_prior_bonus(const Board& b, const Move& m) {
    int bonus = 0;

    // capture check (da ne kažnjavamo taktičke poteze)
    Bitboard enemyOcc = b.turn ? b.occupancyBlack : b.occupancyWhite;
    bool isCapture = (enemyOcc & SQUARE_BITBOARDS[m.toSquare]) != 0;
    if (b.epSquare == m.toSquare) isCapture = true;

    const int from = m.fromSquare;
    const int to = m.toSquare;
    const int pt = m.pieceType % 6; // 0 pawn, 1 N, 2 B, 3 R, 4 Q, 5 K
    const int rf = rank_of(from), rt = rank_of(to);
    const int ff = file_of(from), ft = file_of(to);

    // 1) Rokada (ako tvoj Move ima flag)
    if (m.castle != 0) {
        bonus += 30;
    }

    // 2) Razvoj: konj/lovac sa početnih polja na "normalan" kvadrat
    if (is_start_square_minor(m.pieceType, from)) {
        // ne ostaj na back rank
        if (rt != rf) bonus += 18;
        else bonus += 8; // čak i pomjeranje po ranku je nešto, ali manje
    }

    // 3) Bonus za centralna polja (d4,e4,d5,e5 približno)
    const bool inCenter = (ft >= 2 && ft <= 5 && rt >= 2 && rt <= 5);
    if (inCenter && (pt == 1 || pt == 2 || pt == 0)) bonus += 10;

    // 4) Kazna za rano pomjeranje dame (ako nije capture)
    if (pt == 4 && !isCapture) {
        // ako je dama otišla sa početnog kvadrata u ranoj fazi, minus
        // (heuristika lagana, jer MCTS svakako evaluira dalje)
        bonus -= 8;
    }

    // 5) Rook na open/semi-open file (grubo)
    if (pt == 3) {
        // ako je na file-u bez sopstvenih pješaka, mali bonus
        // (ovdje vrlo grubo, ali jeftino)
        Bitboard ownPawns = (b.turn == WHITE) ? b.pawn_W : b.pawn_B;
        int rookFile = ft;
        Bitboard fileMask = 0;
        for (int r = 0; r < 8; ++r) fileMask |= SQUARE_BITBOARDS[r * 8 + rookFile];
        if ((ownPawns & fileMask) == 0) bonus += 6;
    }

    return bonus;
}

static inline int progressive_limit(int parentVisits) {
    // Progressive widening: k = A * sqrt(visits)
    double k = PW_A * std::sqrt((double)std::max(1, parentVisits));
    int ki = (int)std::floor(k);
    if (ki < 1) ki = 1;
    return ki;
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

    // 1) bazni ordering iz moveorderer-a (bez TT)
    score_moves(b, moves, n);

    // 2) ubaci heuristike + experience book bonus u moves[i].score
    for (int i = 0; i < n; ++i) {
        moves[i].score += positional_prior_bonus(b, moves[i]);

        int bonus = ExperienceBook::instance().bonusForMove(b.hash, moves[i]);
        moves[i].score += bonus;
    }

    // 3) sortiraj ascending, pa pop_back daje najveći score prvo
    std::vector<Move> v(moves, moves + n);
    std::sort(v.begin(), v.end(), [](const Move& a, const Move& b) {
        return a.score < b.score;
        });

    node->unexpanded = std::move(v);
    node->initialUnexpanded = (int)node->unexpanded.size();

    // harmonic sum H_n = 1 + 1/2 + ... + 1/n  (rank prior normalizacija)
    double H = 0.0;
    for (int i = 1; i <= node->initialUnexpanded; ++i) H += 1.0 / (double)i;
    node->priorNorm = (H > 0.0) ? H : 1.0;

    node->initialized = true;
}


MctsNode* MctsSearch::selectChildUCB(MctsNode* node) const {
    if (!node) return nullptr;
    if (node->children.empty()) return nullptr;

    const double parentVisits = (double)std::max(1, node->visits);

    MctsNode* best = nullptr;
    double bestScore = -std::numeric_limits<double>::infinity();

    for (auto& chPtr : node->children) {
        MctsNode* ch = chPtr.get();
        if (!ch) continue;

        double q = (ch->visits > 0) ? (ch->valueSum / (double)ch->visits) : 0.0;
        double u = C * ch->prior * std::sqrt(parentVisits) / (1.0 + (double)ch->visits);

        double score = q + u;
        if (score > bestScore) {
            bestScore = score;
            best = ch;
        }
    }
    return best;
}

void MctsSearch::simulateOnceCollect(std::vector<MctsNode*>& path,
    Board& outLeafState,
    bool& outIsTerminal,
    double& outTerminalVal) {
    path.clear();
    MctsNode* node = rootNode.get();
    path.push_back(node);

    while (true) {
        ensureInitialized(node);

        // Terminal
        if (node->terminal) {
            outIsTerminal = true;
            outTerminalVal = terminalValue(node->terminalResult);
            return;
        }

        int k = progressive_limit(node->visits);

        // EXPAND only if children < k
        if (!node->unexpanded.empty() && (int)node->children.size() < k) {
            int remaining_before_pop = (int)node->unexpanded.size();
            int rank = node->initialUnexpanded - remaining_before_pop; // 0,1,2...

            Move m = node->unexpanded.back();
            node->unexpanded.pop_back();

            Board childState = node->state;
            pushMove(&childState, m);

            auto child = std::make_unique<MctsNode>(childState);
            child->moveFromParent = m;
            child->hasMoveFromParent = true;

            child->prior = (1.0 / (double)(rank + 1)) / node->priorNorm;

            MctsNode* childPtr = child.get();
            node->children.push_back(std::move(child));

            node = childPtr;
            path.push_back(node);

            outIsTerminal = false;
            outLeafState = node->state;
            return;
        }

        // SELECT
        MctsNode* next = selectChildUCB(node);
        if (!next) {
            outIsTerminal = false;
            outLeafState = node->state;
            return;
        }

        node = next;
        path.push_back(node);
    }
}

// GPU-friendly: evaluate many leaves by flattening ALL their child boards into ONE big batch.
// This maximizes GPU utilization (one kernel launch / one memcpy burst) instead of many small batches.
void MctsSearch::evaluateLeaves_OnePlyMinimaxBatch(const std::vector<Board>& leaves,
    std::vector<double>& outVals) {
    const int L = (int)leaves.size();
    outVals.assign((size_t)L, 0.0);

    struct LeafInfo {
        int childStart = 0;
        int childCount = 0;
        bool rootToMove = false;
        bool terminal = false;
        double terminalVal = 0.0;
    };

    std::vector<LeafInfo> info((size_t)L);
    std::vector<Board> allChildren;
    allChildren.reserve((size_t)L * 40); // rough average

    // 1) Build flattened children list
    for (int li = 0; li < L; ++li) {
        Board tmp = leaves[(size_t)li];
        Move moves[256];
        int n = legalMoves(&tmp, moves);
        int res = result(tmp, moves, n);
        if (res != UN_DETERMINED) {
            info[(size_t)li].terminal = true;
            info[(size_t)li].terminalVal = terminalValue(res);
            continue;
        }

        const bool rootToMove = (tmp.turn == rootTurn);
        info[(size_t)li].rootToMove = rootToMove;
        info[(size_t)li].childStart = (int)allChildren.size();
        info[(size_t)li].childCount = n;

        const int start = info[(size_t)li].childStart;
        allChildren.resize((size_t)start + (size_t)n);

        // parallelize per-leaf move application on CPU (feeds GPU faster)
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n; ++i) {
            allChildren[(size_t)start + (size_t)i] = tmp;
            pushMove(&allChildren[(size_t)start + (size_t)i], moves[i]);
        }
    }

    // 2) Batch evaluate all children at once (GPU path inside batchEvaluateRootPerspective)
    std::vector<double> childScores(allChildren.size(), 0.0);
    if (!allChildren.empty()) {
        batchEvaluateRootPerspective(allChildren.data(), (int)allChildren.size(), rootTurn, childScores.data());
    }

    // 3) Reduce childScores -> leaf value (minimax) + qsearchCaptures, then blend
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int li = 0; li < L; ++li) {
        if (info[(size_t)li].terminal) {
            outVals[(size_t)li] = info[(size_t)li].terminalVal;
            continue;
        }

        const int start = info[(size_t)li].childStart;
        const int n = info[(size_t)li].childCount;
        const bool rootToMove = info[(size_t)li].rootToMove;

        double best = rootToMove
            ? -std::numeric_limits<double>::infinity()
            : std::numeric_limits<double>::infinity();

        for (int i = 0; i < n; ++i) {
            double v = childScores[(size_t)start + (size_t)i];
            if (rootToMove) {
                if (v > best) best = v;
            }
            else {
                if (v < best) best = v;
            }
        }

        if (!std::isfinite(best)) {
            best = staticEvalRootPerspective(leaves[(size_t)li]);
        }

        // quiescence-light (4 ply capture-only)
        double q = qsearchCaptures(leaves[(size_t)li], 4);
        double out = 0.25 * best + 0.75 * q;
        outVals[(size_t)li] = out;
    }
}
