#include "batch_eval.hpp"
#include "evaluation.hpp"   // <- provides evaluate(...)
#include "typedefs.hpp"     // <- provides WHITE/BLACK constants

void batchEvaluateRootPerspective(const Board* boards, int n, int rootTurn, double* outScores) {
    for (int i = 0; i < n; ++i) {
        // evaluate(board, UN_DETERMINED) returns score from WHITE perspective.
        int evWhite = evaluateLeaf(boards[i], UN_DETERMINED);

        // Convert to ROOT perspective:
        // - if rootTurn == WHITE => keep evWhite
        // - if rootTurn == BLACK => invert
        outScores[i] = (rootTurn == WHITE) ? (double)evWhite : (double)-evWhite;
    }
}
