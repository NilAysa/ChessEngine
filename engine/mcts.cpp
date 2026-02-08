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
    // Batched leaf-parallel MCTS (CPU tree + GPU batch evaluacija leaf-ova).
    // Ako neko želi debug single-step, može postaviti batchSize=1 u kodu.
    if (batchSize <= 1) {
        for (int i = 0; i < iterations; ++i) simulateOnce();
        return;
    }

    runBatched();
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

    // 3) Bonus za centralna polja
    // strogi centar: d4 e4 d5 e5  (sq: 27,28,35,36 ako je a1=0)
    if (to == 27 || to == 28 || to == 35 || to == 36) bonus += 14;

    // širi centar: c3..f6 okvir
    if (ft >= 2 && ft <= 5 && rt >= 2 && rt <= 5) bonus += 6;

    // 4) Penalizuj rani rook/queen 'šetanje' (osim ako je capture)
    if (!isCapture && (pt == 3 || pt == 4)) { // rook ili queen
        // ako se pomjera u prvih par rankova svoje strane, malo kazni
        if (m.pieceType < 6) { // white
            if (rf <= 1 && rt <= 2) bonus -= 8;
        }
        else { // black
            if (rf >= 6 && rt >= 5) bonus -= 8;
        }
    }

    // 5) Dvostruki pawn push malo nagradi u openingu
    if (pt == 0 && !isCapture) { // pawn
        int dr = std::abs(rt - rf);
        if (dr == 2) bonus += 6;
    }

    // clamp (da bonus ne divlja)
    if (bonus > 60) bonus = 60;
    if (bonus < -40) bonus = -40;

    return bonus;
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

    // 1) bazni ordering iz moveorderer-a
    score_moves(b, moves, n);

    // 2) dodaj prior bonus + ExperienceBook bonus (oboje ide u moves[i].score)
    for (int i = 0; i < n; ++i) {
        moves[i].score += positional_prior_bonus(b, moves[i]);

        int eb = ExperienceBook::instance().bonusForMove(b.hash, moves[i]);
        moves[i].score += eb;
    }

    // 3) sortiraj ascending, pa pop_back daje najveći score prvo
    std::vector<Move> v(moves, moves + n);
    std::sort(v.begin(), v.end(), [](const Move& a, const Move& b) {
        return a.score < b.score;
        });

    node->unexpanded = std::move(v);
    node->initialUnexpanded = (int)node->unexpanded.size();

    // harmonic sum H_n = 1 + 1/2 + ... + 1/n
    double H = 0.0;
    for (int i = 1; i <= node->initialUnexpanded; ++i) H += 1.0 / (double)i;
    node->priorNorm = (H > 0.0) ? H : 1.0;

    node->initialized = true;
}


MctsNode* MctsSearch::selectChildUCB(MctsNode* node) const {
    if (!node || node->children.empty()) return nullptr;

    const bool rootToMove = (node->state.turn == rootTurn);

    double bestScore = -std::numeric_limits<double>::infinity();
    MctsNode* bestChild = nullptr;

    // Effective visit counts include inFlight (O) to reduce stampede in parallel/batched setting.
    const int parentN = node->visits + node->inFlight;

    for (auto& ch : node->children) {
        MctsNode* c = ch.get();

        double q = 0.0;
        if (c->visits > 0)
            q = (c->valueSum / (double)c->visits) / 600.0;   // 600 cp = ~1.0
        if (q > 1.0) q = 1.0;
        if (q < -1.0) q = -1.0;

        double p = c->prior;
        if (p < 1e-6) p = 1e-6;

        const int childN = c->visits + c->inFlight;

        // PUCT exploration term (use effective visits)
        double u = C * p * std::sqrt((double)parentN + 1.0) / ((double)childN + 1.0);

        // root max, opponent min
        double score = (rootToMove ? q : -q) + u;

        if (score > bestScore) {
            bestScore = score;
            bestChild = c;
        }
    }

    return bestChild;
}


static inline int progressive_limit(int visits) {
    // k = 1 + floor(a * sqrt(visits))
    // visits=0 -> k=1, visits=1 -> k=2, visits=4 -> k=4 (za a=1.5)
    int k = 1 + (int)std::floor(PW_A * std::sqrt((double)std::max(0, visits)));
    return k;
}


void MctsSearch::simulateOnce() {
    std::vector<MctsNode*> path;
    path.reserve(128);

    MctsNode* node = rootNode.get();
    path.push_back(node);

    while (true) {
        ensureInitialized(node);

        // Terminal
        if (node->terminal) {
            double val = terminalValue(node->terminalResult);
            for (MctsNode* p : path) {
                p->visits += 1;
                p->valueSum += val;
            }
            return;
        }

        int k = progressive_limit(node->visits);

        // EXPAND samo ako je children < k
        if (!node->unexpanded.empty() && (int)node->children.size() < k) {
            // rank prior: prvi expand (najbolji) ima rank=0
            int remaining_before_pop = (int)node->unexpanded.size();
            int rank = node->initialUnexpanded - remaining_before_pop; // 0,1,2...

            Move m = node->unexpanded.back();
            node->unexpanded.pop_back();

            Board childState = node->state;
            pushMove(&childState, m);

            auto child = std::make_unique<MctsNode>(childState);
            child->moveFromParent = m;
            child->hasMoveFromParent = true;

            // PUCT prior (0..1), normalizovan
            child->prior = (1.0 / (double)(rank + 1)) / node->priorNorm;

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

        // SELECT
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


void MctsSearch::completeSimulation(const std::vector<MctsNode*>& path, double value) {
    // path uključuje root..leaf
    for (MctsNode* p : path) {
        p->inFlight -= 1;
        p->visits += 1;
        p->valueSum += value;
    }
}

void MctsSearch::reserveOneSimulation(std::vector<MctsNode*>& outPath,
    Board& outLeafBoard,
    bool& outIsTerminal,
    int& outTerminalResult) {
    outPath.clear();
    outPath.reserve(128);

    MctsNode* node = rootNode.get();
    outPath.push_back(node);

    while (true) {
        ensureInitialized(node);

        // Terminal
        if (node->terminal) {
            outIsTerminal = true;
            outTerminalResult = node->terminalResult;

            // rezerviši path (inFlight++)
            for (MctsNode* p : outPath) p->inFlight += 1;

            outLeafBoard = node->state;
            return;
        }

        // Progressive widening treba gledati effective visits (visits+inFlight)
        int k = progressive_limit(node->visits + node->inFlight);

        // EXPAND samo ako je children < k
        if (!node->unexpanded.empty() && (int)node->children.size() < k) {
            // rank prior: prvi expand (najbolji) ima rank=0
            int remaining_before_pop = (int)node->unexpanded.size();
            int rank = node->initialUnexpanded - remaining_before_pop; // 0,1,2...

            Move m = node->unexpanded.back();
            node->unexpanded.pop_back();

            Board childState = node->state;
            pushMove(&childState, m);

            auto child = std::make_unique<MctsNode>(childState);
            child->moveFromParent = m;
            child->hasMoveFromParent = true;

            // PUCT prior (0..1), normalizovan
            child->prior = (1.0 / (double)(rank + 1)) / node->priorNorm;

            MctsNode* childPtr = child.get();
            node->children.push_back(std::move(child));

            node = childPtr;
            outPath.push_back(node);

            // rezerviši path (inFlight++)
            for (MctsNode* p : outPath) p->inFlight += 1;

            outIsTerminal = false;
            outTerminalResult = UN_DETERMINED;
            outLeafBoard = node->state;
            return;
        }

        // SELECT
        MctsNode* next = selectChildUCB(node);
        if (!next) {
            // nema djece (npr. stalemate / bug / fallback) -> evaluiraj ovaj node
            // rezerviši path (inFlight++)
            for (MctsNode* p : outPath) p->inFlight += 1;

            outIsTerminal = false;
            outTerminalResult = UN_DETERMINED;
            outLeafBoard = node->state;
            return;
        }

        node = next;
        outPath.push_back(node);
    }
}

void MctsSearch::runBatched() {
    int done = 0;

    // Double-buffer: dok GPU evaluira batch "prev", CPU može da gradi batch "cur".
    // Ovo smanjuje idle vrijeme i pravi vidljiviju razliku GPU vs CPU (promjena #4).

    // prev buffers (in-flight evaluation)
    std::vector<std::vector<MctsNode*>> prevPaths;
    std::vector<int> prevPendingMap;
    std::vector<double> prevScores;
    BatchEvalHandle prevHandle{};
    bool hasPrev = false;

    // cur buffers (to be submitted)
    std::vector<std::vector<MctsNode*>> paths;
    std::vector<Board> leafBoards;
    std::vector<int> pendingMap;

    paths.reserve((size_t)batchSize);
    leafBoards.reserve((size_t)batchSize);
    pendingMap.reserve((size_t)batchSize);

    prevPaths.reserve((size_t)batchSize);
    prevPendingMap.reserve((size_t)batchSize);

    while (done < iterations) {
        const int cur = std::min(batchSize, iterations - done);

        paths.clear();
        leafBoards.clear();
        pendingMap.clear();
        paths.resize((size_t)cur);

        // 1) CPU: rezerviši cur simulacija (selection + opcionalni expand), bez evaluacije
        for (int i = 0; i < cur; ++i) {
            Board leaf{};
            bool isTerm = false;
            int termRes = UN_DETERMINED;

            reserveOneSimulation(paths[(size_t)i], leaf, isTerm, termRes);

            if (isTerm) {
                double v = terminalValue(termRes);
                completeSimulation(paths[(size_t)i], v);
            }
            else {
                pendingMap.push_back(i);
                leafBoards.push_back(leaf);
            }
        }

        // 2) Dok GPU radi prethodni batch, tek sada čekamo i backprop-ujemo njegov rezultat.
        if (hasPrev) {
            prevScores.assign(prevPendingMap.size(), 0.0);
            batchEvaluateRootPerspectiveCollect(prevHandle, prevScores.data());

            for (size_t j = 0; j < prevPendingMap.size(); ++j) {
                const int pathIdx = prevPendingMap[j];
                double v = prevScores[j];
                completeSimulation(prevPaths[(size_t)pathIdx], v);
            }

            hasPrev = false;
        }

        // 3) Submit evaluaciju za trenutni batch (async). Rezultat dolazi u sljedećoj iteraciji.
        if (!leafBoards.empty()) {
            prevHandle = BatchEvalHandle{};
            batchEvaluateRootPerspectiveAsync(leafBoards.data(), (int)leafBoards.size(), rootTurn, prevHandle);

            prevPaths.swap(paths);
            prevPendingMap.swap(pendingMap);
            hasPrev = true;
        }

        done += cur;
    }

    // Flush zadnji in-flight batch
    if (hasPrev) {
        prevScores.assign(prevPendingMap.size(), 0.0);
        batchEvaluateRootPerspectiveCollect(prevHandle, prevScores.data());

        for (size_t j = 0; j < prevPendingMap.size(); ++j) {
            const int pathIdx = prevPendingMap[j];
            double v = prevScores[j];
            completeSimulation(prevPaths[(size_t)pathIdx], v);
        }
    }
}


/*
void MctsSearch::runBatched() {
    int done = 0;

    std::vector<std::vector<MctsNode*>> paths;
    std::vector<Board> leafBoards;
    std::vector<int> pendingMap;
    std::vector<double> scores;

    paths.reserve((size_t)batchSize);
    leafBoards.reserve((size_t)batchSize);
    pendingMap.reserve((size_t)batchSize);
    scores.reserve((size_t)batchSize);

    while (done < iterations) {
        const int cur = std::min(batchSize, iterations - done);

        paths.clear();
        leafBoards.clear();
        pendingMap.clear();
        paths.resize((size_t)cur);

        // 1) reserve
        for (int i = 0; i < cur; ++i) {
            Board leaf{};
            bool isTerm = false;
            int termRes = UN_DETERMINED;

            reserveOneSimulation(paths[(size_t)i], leaf, isTerm, termRes);

            if (isTerm) {
                double v = terminalValue(termRes);
                completeSimulation(paths[(size_t)i], v);
            }
            else {
                pendingMap.push_back(i);
                leafBoards.push_back(leaf);
            }
        }

        // 2) eval (sync wrapper koristi GPU async ispod haube, ali čeka)
        scores.assign(leafBoards.size(), 0.0);
        if (!leafBoards.empty()) {
            batchEvaluateRootPerspective(leafBoards.data(), (int)leafBoards.size(), rootTurn, scores.data());
        }

        // 3) complete
        for (size_t j = 0; j < pendingMap.size(); ++j) {
            const int pathIdx = pendingMap[j];
            double v = scores[j];
            completeSimulation(paths[(size_t)pathIdx], v);
        }

        done += cur;
    }
}
*/
