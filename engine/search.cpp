#include <cstring>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "search.hpp"
#include "evaluation.hpp"
#include "board.hpp"
#include "movegen.hpp"
#include "tt.hpp"
#include "zobrist.hpp"
#include "moveorderer.hpp"
#include "bitboards.hpp"   // za SQUARE_BITBOARDS u killer heuristici

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

constexpr int MAX_PLY = 64;   // dovoljno za tvoje dubine

static Move KILLER1[MAX_PLY];
static Move KILLER2[MAX_PLY];

int alphabeta(Board board, int depth, int alpha, int beta);

// Search variables
int SEARCH_NODES_SEARCHED = 0;
Move SEARCH_BEST_MOVE{};
int _DEPTH;

// koliko je niti stvarno korišteno u zadnjoj pretrazi
int SEARCH_THREADS_USED = 1;

// ---------------------------------------------------------
// Sekvencijalna verzija – koristi se kad nema smisla paliti niti
// ---------------------------------------------------------
static int search_sequential(Board board, int depth) {
    SEARCH_BEST_MOVE = Move{};
    SEARCH_NODES_SEARCHED = 0;
    _DEPTH = depth;

    for (int i = 0; i < MAX_PLY; ++i) {
        KILLER1[i] = Move{};
        KILLER2[i] = Move{};
    }

    return alphabeta(board, depth, MIN_EVAL, MAX_EVAL);
}

// ---------------------------------------------------------
// Glavna search funkcija koju zove UCI
// ---------------------------------------------------------
int search(Board board, int depth) {
#ifndef _OPENMP
    // Ako nema OpenMP podrške, ostajemo potpuno sekvencijalni
    return search_sequential(board, depth);
#else
    // Saznamo koliko nam je niti OpenMP realno dao
    int numThreads = 1;
#pragma omp parallel
    {
#pragma omp single
        numThreads = omp_get_num_threads();
    }

    // Ako je dubina mala ili imamo samo jednu nit, ne koristi se paralelizacija
    if (depth <= 3 || numThreads <= 1) {
        SEARCH_THREADS_USED = 1;
        return search_sequential(board, depth);
    }

    // Inače ćemo stvarno koristiti sve niti na rootu
    SEARCH_THREADS_USED = numThreads;

    // Root-paralelizacija inspirisana PV-split/YBWC:
    //  - prvi najbolji potez pretražujemo sekvencijalno (da dobijemo dobar alpha)
    //  - ostale poteze puštamo paralelno

    SEARCH_BEST_MOVE = Move{};
    SEARCH_NODES_SEARCHED = 0;
    _DEPTH = depth;

    for (int i = 0; i < MAX_PLY; ++i) {
        KILLER1[i] = Move{};
        KILLER2[i] = Move{};
    }

    int alpha = MIN_EVAL;
    int beta = MAX_EVAL;
    int origAlpha = alpha;

    // TT lookup na korijenu (isto kao u alphabeta)
    TTEntry entry = getTTEntry(board.hash);
    if (board.hash == entry.zobrist && entry.depth >= depth) {

        if (entry.nodeType == EXACT) {
            SEARCH_BEST_MOVE = entry.move;
            return entry.eval;
        }
        else if (entry.nodeType == LOWER) {
            alpha = max(alpha, entry.eval);
        }
        else if (entry.nodeType == UPPER) {
            beta = min(beta, entry.eval);
        }

        if (alpha >= beta) {
            SEARCH_BEST_MOVE = entry.move;
            return entry.eval;
        }
    }

    Move moves[256]{};
    int cmoves = legalMoves(&board, moves);

    int res = result(board, moves, cmoves);
    if (res) {
        int eval = evaluate(board, res);

        if (res != DRAW)
            eval += (1 * _DEPTH - depth) * (board.turn ? 1 : -1);

        eval *= (board.turn ? 1 : -1);
        addTTEntry(board, eval, Move{}, depth, beta, origAlpha);
        return eval;
    }
    else if (depth == 0) {
        int eval = evaluate(board, res) * (board.turn ? 1 : -1);
        addTTEntry(board, eval, Move{}, depth, beta, origAlpha);
        return eval;
    }

    int eval = MIN_EVAL;
    Move best_move{};

    // *** VAŽNO: i na rootu sada koristimo novu verziju score_moves
    int rootPly = 0;
    Move k1 = KILLER1[rootPly];
    Move k2 = KILLER2[rootPly];
    score_moves(board, entry, k1, k2, moves, cmoves);

    // 1) Prvi (najbolji po heuristici) potez – sekvencijalno
    int firstIdx = select_move(moves, cmoves);
    if (firstIdx == -1) {
        // Nema validnih poteza – sigurnosna mreža (realno već pokriveno gore)
        int e = evaluate(board, 0) * (board.turn ? 1 : -1);
        addTTEntry(board, e, Move{}, depth, beta, origAlpha);
        return e;
    }

    if (moves[firstIdx].validation == LEGAL) {
        Board child = board;
        pushMove(&child, moves[firstIdx]);
        eval = -alphabeta(child, depth - 1, -beta, -alpha);
        best_move = moves[firstIdx];

        alpha = max(alpha, eval);
        SEARCH_BEST_MOVE = best_move;
    }

    // 2) Ostali potezi – paralelno, sa dijeljenjem alpha
#pragma omp parallel
    {
        int localBestEval = eval;
        Move localBestMove = best_move;

#pragma omp for schedule(dynamic)
        for (int i = 0; i < cmoves; ++i) {
            if (i == firstIdx)
                continue;
            if (moves[i].validation != LEGAL)
                continue;

            Board child = board;
            pushMove(&child, moves[i]);

            // sigurno čitanje alpha kroz kritičnu sekciju (MSVC nema atomic read)
            int curAlpha;
#pragma omp critical(alpha_read)
            {
                curAlpha = alpha;
            }

            int childEval = -alphabeta(child, depth - 1, -beta, -curAlpha);

            if (childEval > localBestEval) {
                localBestEval = childEval;
                localBestMove = moves[i];

                // siguran update alpha
#pragma omp critical(alpha_update)
                {
                    if (childEval > alpha)
                        alpha = childEval;
                }
            }
        }

        // Merge rezultata niti
#pragma omp critical(best_update)
        {
            if (localBestEval > eval) {
                eval = localBestEval;
                best_move = localBestMove;
            }
        }
    }

    addTTEntry(board, eval, best_move, depth, beta, origAlpha);
    SEARCH_BEST_MOVE = best_move;
    return eval;
#endif
}

// ---------------------------------------------------------
// Rekurzivni alphabeta sa killer + futility pruning
// ---------------------------------------------------------
int alphabeta(Board board, int depth, int alpha, int beta) {
#ifdef _OPENMP
#pragma omp atomic
#endif
    SEARCH_NODES_SEARCHED++;

    int origAlpha = alpha;

    // TT table lookup
    TTEntry entry = getTTEntry(board.hash);
    if (board.hash == entry.zobrist && entry.depth >= depth) {

        if (entry.nodeType == EXACT) {

            if (depth == _DEPTH)
                SEARCH_BEST_MOVE = entry.move;

            return entry.eval;
        }
        else if (entry.nodeType == LOWER) {
            alpha = max(alpha, entry.eval);
        }
        else if (entry.nodeType == UPPER) {
            beta = min(beta, entry.eval);
        }

        if (alpha >= beta) {
            if (depth == _DEPTH)
                SEARCH_BEST_MOVE = entry.move;

            return entry.eval;
        }
    }

    // Jednostavan futility pruning na depth == 1
    if (depth == 1) {
        if (!inCheck(board)) {
            int staticEval = evaluate(board, UN_DETERMINED) * (board.turn ? 1 : -1);
            const int FUT_MARGIN = 200; // ~2 "desetine pješaka"

            // Ako ni uz optimističnu marginu ne možemo probiti alpha -> reži
            if (staticEval + FUT_MARGIN <= alpha)
                return staticEval;
        }
    }

    Move moves[256]{};
    int cmoves = legalMoves(&board, moves);

    int res = result(board, moves, cmoves);
    if (res) {
        int eval = evaluate(board, res);

        if (res != DRAW)
            eval += (1 * _DEPTH - depth) * (board.turn ? 1 : -1);

        return eval * (board.turn ? 1 : -1);
    }
    else if (depth == 0) {
        return evaluate(board, res) * (board.turn ? 1 : -1);
    }

    int nextMove;
    int eval2 = MIN_EVAL;
    Move best_move2{};

    int ply = _DEPTH - depth;
    if (ply < 0) ply = 0;
    if (ply >= MAX_PLY) ply = MAX_PLY - 1;

    Move k1 = KILLER1[ply];
    Move k2 = KILLER2[ply];

    score_moves(board, entry, k1, k2, moves, cmoves);

    while ((nextMove = select_move(moves, cmoves)) != -1) {
        if (moves[nextMove].validation == LEGAL) {
            Board child = board;
            pushMove(&child, moves[nextMove]);
            int childEval = -alphabeta(child, depth - 1, -beta, -alpha);

            if (childEval > eval2) {
                eval2 = childEval;
                best_move2 = moves[nextMove];

                if (depth == _DEPTH)
                    SEARCH_BEST_MOVE = best_move2;
            }

            alpha = max(alpha, childEval);

            if (alpha >= beta) {
                // beta-cutoff: ako je miran potez, postavi killer
                int cutPly = _DEPTH - depth;
                if (cutPly >= 0 && cutPly < MAX_PLY) {
                    // otprilike: nije capture, nema rokade, nema promocije
                    Bitboard enemyOcc = board.turn ? board.occupancyBlack
                        : board.occupancyWhite;
                    bool isCapture =
                        (enemyOcc & SQUARE_BITBOARDS[moves[nextMove].toSquare]) != 0;

                    if (!isCapture && moves[nextMove].promotion == -1
                        && moves[nextMove].castle == 0) {
                        if (!moves_are_equal(moves[nextMove], KILLER1[cutPly])) {
                            KILLER2[cutPly] = KILLER1[cutPly];
                            KILLER1[cutPly] = moves[nextMove];
                        }
                    }
                }
                break;
            }
        }
    }

    addTTEntry(board, eval2, best_move2, depth, beta, origAlpha);

    return eval2;
}
