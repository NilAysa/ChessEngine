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

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

int alphabeta(Board board, int depth, int alpha, int beta);

// ----- Globalne varijable -----
int SEARCH_NODES_SEARCHED = 0;
Move SEARCH_BEST_MOVE{};
int _DEPTH = 0;
int SEARCH_THREADS_USED = 1;

// Padding za false sharing
struct PaddedInt {
    int value;
    char pad[64 - sizeof(int)];
};

// Sekvencijalna pretraga
static int search_sequential(Board board, int depth) {
    SEARCH_BEST_MOVE = Move{};
    SEARCH_NODES_SEARCHED = 0;
    _DEPTH = depth;
    SEARCH_THREADS_USED = 1;

    return alphabeta(board, depth, MIN_EVAL, MAX_EVAL);
}

// Glavna funkcija pretrage
int search(Board board, int depth) {
#ifndef _OPENMP
    return search_sequential(board, depth);
#else
    int numThreads = 1;
#pragma omp parallel
    {
#pragma omp single
        numThreads = omp_get_num_threads();
    }

    if (depth <= 3 || numThreads <= 1) {
        SEARCH_THREADS_USED = 1;
        return search_sequential(board, depth);
    }

    SEARCH_THREADS_USED = numThreads;
    SEARCH_BEST_MOVE = Move{};
    SEARCH_NODES_SEARCHED = 0;
    _DEPTH = depth;

    int alpha = MIN_EVAL;
    int beta = MAX_EVAL;
    int origAlpha = alpha;

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
            eval += (_DEPTH - depth) * (board.turn ? 1 : -1);
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
    score_moves(board, entry, moves, cmoves);

    int firstIdx = select_move(moves, cmoves);
    if (firstIdx == -1) {
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

#pragma omp parallel
    {
        PaddedInt localAlpha; localAlpha.value = alpha;
        int localBestEval = eval;
        Move localBestMove = best_move;

#pragma omp for schedule(dynamic)
        for (int i = 0; i < cmoves; ++i) {
            if (i == firstIdx || moves[i].validation != LEGAL)
                continue;

            Board child = board;
            pushMove(&child, moves[i]);
            int childEval = -alphabeta(child, depth - 1, -beta, -localAlpha.value);

            if (childEval > localBestEval) {
                localBestEval = childEval;
                localBestMove = moves[i];

#pragma omp critical(alpha_update)
                {
                    if (childEval > alpha)
                        alpha = childEval;
                }
            }
        }

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

int alphabeta(Board board, int depth, int alpha, int beta) {
#ifdef _OPENMP
#pragma omp atomic
#endif
    SEARCH_NODES_SEARCHED++;

    int origAlpha = alpha;

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

    Move moves[256]{};
    int cmoves = legalMoves(&board, moves);

    int res = result(board, moves, cmoves);
    if (res) {
        int eval = evaluate(board, res);
        if (res != DRAW)
            eval += (_DEPTH - depth) * (board.turn ? 1 : -1);
        return eval * (board.turn ? 1 : -1);
    }
    else if (depth == 0) {
        return evaluate(board, res) * (board.turn ? 1 : -1);
    }

    int nextMove;
    int eval2 = MIN_EVAL;
    Move best_move2{};
    score_moves(board, entry, moves, cmoves);

    while ((nextMove = select_move(moves, cmoves)) != -1) {
        if (moves[nextMove].validation != LEGAL)
            continue;

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
        if (alpha >= beta)
            break;
    }

    addTTEntry(board, eval2, best_move2, depth, beta, origAlpha);
    return eval2;
}
