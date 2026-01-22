// engine/nnue_cuda.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>

#include "typedefs.hpp"
#include "nnue_cuda.hpp"

// reuse your generated weights
#include "nnue_weights_generated.hpp"

// reuse classic PST tables (host constants) to upload into __constant__
#include "pst_tables.hpp"

static inline bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("[CUDA] %s failed: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// --- device buffers for NNUE weights ---
static float* d_W1 = nullptr;  // [128 * 781]
static float* d_B1 = nullptr;  // [128]
static float* d_W2 = nullptr;  // [128]
static float* d_B2 = nullptr;  // [1]

// --- device constant buffers for classic eval ---
__device__ __constant__ int c_PAWN_PST[64];
__device__ __constant__ int c_KNIGHT_PST[64];
__device__ __constant__ int c_BISHOP_PST[64];
__device__ __constant__ int c_ROOK_PST[64];
__device__ __constant__ int c_QUEEN_PST[64];
__device__ __constant__ int c_KING_PST[64];

// --- device buffers for input/output (resizable) ---
static Board* d_boards = nullptr;
static int*   d_outCp  = nullptr;
static int    cap      = 0;

// --- host pinned staging buffers (for true async memcpy) ---
static Board* h_boardsPinned = nullptr;
static int*   h_outPinned    = nullptr;
static int    h_cap          = 0;

// --- one persistent stream ---
static cudaStream_t g_stream = nullptr;

static bool g_inited = false;

static __device__ __forceinline__ float relu(float x) { return x > 0.0f ? x : 0.0f; }
static __device__ __forceinline__ int mirrorSqDev(int sq) { return 63 - sq; }

// --- classic eval (device) ---
// NOTE: must match evaluation.cpp (material + PST)
static __device__ __forceinline__ int evalClassicWhitePerspectiveDev(const Board& b) {
    int eval = 0;

    auto addWhite = [&](Bitboard bb, int valAbs, const int* pst) {
        while (bb) {
            int sq = __ffsll((unsigned long long)bb) - 1;
            bb &= (bb - 1);
            eval += valAbs;
            eval += pst[sq];
        }
    };

    auto addBlack = [&](Bitboard bb, int valAbs, const int* pst) {
        while (bb) {
            int sq = __ffsll((unsigned long long)bb) - 1;
            bb &= (bb - 1);
            eval -= valAbs;
            eval -= pst[mirrorSqDev(sq)];
        }
    };

    // abs values: pawn=100, knight=320, bishop=330, rook=500, queen=900
    addWhite(b.pawn_W,   100, c_PAWN_PST);
    addWhite(b.knight_W, 320, c_KNIGHT_PST);
    addWhite(b.bishop_W, 330, c_BISHOP_PST);
    addWhite(b.rook_W,   500, c_ROOK_PST);
    addWhite(b.queen_W,  900, c_QUEEN_PST);

    addBlack(b.pawn_B,   100, c_PAWN_PST);
    addBlack(b.knight_B, 320, c_KNIGHT_PST);
    addBlack(b.bishop_B, 330, c_BISHOP_PST);
    addBlack(b.rook_B,   500, c_ROOK_PST);
    addBlack(b.queen_B,  900, c_QUEEN_PST);

    // king PST special-case (same as evaluation.cpp)
    eval += c_KING_PST[b.whiteKingSq];
    eval -= c_KING_PST[mirrorSqDev(b.blackKingSq)];

    // clamp like evaluation.cpp bounds
    if (eval >  100000) eval =  100000;
    if (eval < -100000) eval = -100000;
    return eval;
}

// Build sparse active feature indices (same logic as your existing code)
__device__ __forceinline__ int buildActive(const Board& b, int* act) {
    int cnt = 0;

    auto addPiecesPlane = [&](Bitboard bb, int base) {
        while (bb) {
            int sq = __ffsll((unsigned long long)bb) - 1;
            bb &= (bb - 1);
            act[cnt++] = base + (sq ^ 7);
        }
    };

    // 12 planes * 64 = 768
    addPiecesPlane(b.pawn_W,   0 * 64);
    addPiecesPlane(b.knight_W, 1 * 64);
    addPiecesPlane(b.bishop_W, 2 * 64);
    addPiecesPlane(b.rook_W,   3 * 64);
    addPiecesPlane(b.queen_W,  4 * 64);
    addPiecesPlane(b.king_W,   5 * 64);

    addPiecesPlane(b.pawn_B,   6 * 64);
    addPiecesPlane(b.knight_B, 7 * 64);
    addPiecesPlane(b.bishop_B, 8 * 64);
    addPiecesPlane(b.rook_B,   9 * 64);
    addPiecesPlane(b.queen_B, 10 * 64);
    addPiecesPlane(b.king_B,  11 * 64);

    // STM
    if (b.turn == WHITE) act[cnt++] = 768;

    // castling bits (K,Q,k,q) => 769..772
    if (b.castling & 0b0001) act[cnt++] = 769;
    if (b.castling & 0b0010) act[cnt++] = 770;
    if (b.castling & 0b0100) act[cnt++] = 771;
    if (b.castling & 0b1000) act[cnt++] = 772;

    // en-passant file (if exists) => 773..780 (8 files)
    if (b.epSquare >= 0 && b.epSquare < 64) {
        int file = b.epSquare & 7;
        act[cnt++] = 773 + file;
    }

    return cnt;
}

__global__ void nnueBatchBlendedKernel(const Board* boards, int n,
                                       const float* W1, const float* B1,
                                       const float* W2, const float* B2,
                                       int blendPermille,
                                       int* outCp)
{
    int pos = (int)blockIdx.x;
    if (pos >= n) return;

    int i = (int)threadIdx.x; // 0..127

    __shared__ int   act[64];     // max active features
    __shared__ int   actCount;
    __shared__ float hidden[NNUE_HIDDEN_DIM];
    __shared__ int   classicEval;

    if (i == 0) {
        const Board& b = boards[pos];
        classicEval = evalClassicWhitePerspectiveDev(b);
        actCount = buildActive(b, act);
    }
    __syncthreads();

    // Hidden neuron i: dot over sparse act
    float sum = B1[i];
    for (int k = 0; k < actCount; ++k) {
        int idx = act[k];
        sum += W1[i * NNUE_INPUT_DIM + idx];
    }
    hidden[i] = relu(sum);
    __syncthreads();

    if (i == 0) {
        float out = B2[0];
        for (int j = 0; j < NNUE_HIDDEN_DIM; ++j) out += W2[j] * hidden[j];

        // same scaling/clamp as nnue.cpp
        float nnCpF = out * 400.0f;
        if (nnCpF >  100000.0f) nnCpF =  100000.0f;
        if (nnCpF < -100000.0f) nnCpF = -100000.0f;
        int nnCp = (int)llroundf(nnCpF);

        // blend like evaluateLeaf() and batch_eval.cpp
        int a = blendPermille;
        if (a < 0) a = 0;
        if (a > 1000) a = 1000;

        int blended = (classicEval * (1000 - a) + nnCp * a) / 1000;
        if (blended >  100000) blended =  100000;
        if (blended < -100000) blended = -100000;

        outCp[pos] = blended;
    }
}

bool nnueCudaInit() {
    if (g_inited) return true;

    int devCount = 0;
    if (!ck(cudaGetDeviceCount(&devCount), "cudaGetDeviceCount")) return false;
    if (devCount == 0) return false;

    // stream
    if (!ck(cudaStreamCreate(&g_stream), "cudaStreamCreate")) return false;

    // allocate NNUE weights on device
    if (!ck(cudaMalloc(&d_W1, sizeof(float) * NNUE_HIDDEN_DIM * NNUE_INPUT_DIM), "malloc d_W1")) return false;
    if (!ck(cudaMalloc(&d_B1, sizeof(float) * NNUE_HIDDEN_DIM), "malloc d_B1")) return false;
    if (!ck(cudaMalloc(&d_W2, sizeof(float) * NNUE_HIDDEN_DIM), "malloc d_W2")) return false;
    if (!ck(cudaMalloc(&d_B2, sizeof(float) * 1), "malloc d_B2")) return false;

    if (!ck(cudaMemcpy(d_W1, NNUE_W1, sizeof(float) * NNUE_HIDDEN_DIM * NNUE_INPUT_DIM, cudaMemcpyHostToDevice), "cpy W1")) return false;
    if (!ck(cudaMemcpy(d_B1, NNUE_B1, sizeof(float) * NNUE_HIDDEN_DIM, cudaMemcpyHostToDevice), "cpy B1")) return false;
    if (!ck(cudaMemcpy(d_W2, NNUE_W2, sizeof(float) * NNUE_HIDDEN_DIM, cudaMemcpyHostToDevice), "cpy W2")) return false;
    if (!ck(cudaMemcpy(d_B2, &NNUE_B2, sizeof(float), cudaMemcpyHostToDevice), "cpy B2")) return false;

    // upload PSTs to constant memory (classic eval on GPU)
    if (!ck(cudaMemcpyToSymbol(c_PAWN_PST,   PAWN_W_PST,   sizeof(int) * 64), "cpy PST pawn")) return false;
    if (!ck(cudaMemcpyToSymbol(c_KNIGHT_PST, KNIGHT_W_PST, sizeof(int) * 64), "cpy PST knight")) return false;
    if (!ck(cudaMemcpyToSymbol(c_BISHOP_PST, BISHOP_W_PST, sizeof(int) * 64), "cpy PST bishop")) return false;
    if (!ck(cudaMemcpyToSymbol(c_ROOK_PST,   ROOK_W_PST,   sizeof(int) * 64), "cpy PST rook")) return false;
    if (!ck(cudaMemcpyToSymbol(c_QUEEN_PST,  QUEEN_W_PST,  sizeof(int) * 64), "cpy PST queen")) return false;
    if (!ck(cudaMemcpyToSymbol(c_KING_PST,   KING_W_PST,   sizeof(int) * 64), "cpy PST king")) return false;

    g_inited = true;
    std::printf("[CUDA] NNUE CUDA init OK (stream + persistent buffers + PST const)\n");
    return true;
}

void nnueCudaShutdown() {
    if (d_W1) cudaFree(d_W1), d_W1 = nullptr;
    if (d_B1) cudaFree(d_B1), d_B1 = nullptr;
    if (d_W2) cudaFree(d_W2), d_W2 = nullptr;
    if (d_B2) cudaFree(d_B2), d_B2 = nullptr;

    if (d_boards) cudaFree(d_boards), d_boards = nullptr;
    if (d_outCp)  cudaFree(d_outCp),  d_outCp  = nullptr;

    if (h_boardsPinned) cudaFreeHost(h_boardsPinned), h_boardsPinned = nullptr;
    if (h_outPinned)    cudaFreeHost(h_outPinned),    h_outPinned    = nullptr;

    cap = 0;
    h_cap = 0;

    if (g_stream) cudaStreamDestroy(g_stream), g_stream = nullptr;

    g_inited = false;
}

// Evaluate NNUE for a batch of boards, returns WHITE-perspective cp
bool nnueCudaEvaluateBatchWhite(const Board* boards, int n, int* outCp) {
    if (!g_inited) return false;
    if (n <= 0) return true;

    // Resize device buffers
    if (n > cap) {
        if (d_boards) cudaFree(d_boards);
        if (d_outCp)  cudaFree(d_outCp);

        if (!ck(cudaMalloc(&d_boards, sizeof(Board) * n), "malloc d_boards")) return false;
        if (!ck(cudaMalloc(&d_outCp,  sizeof(int)   * n), "malloc d_outCp"))  return false;

        cap = n;
    }

    // Resize pinned host staging (for true async H2D/D2H)
    if (n > h_cap) {
        if (h_boardsPinned) cudaFreeHost(h_boardsPinned);
        if (h_outPinned)    cudaFreeHost(h_outPinned);

        if (!ck(cudaMallocHost(&h_boardsPinned, sizeof(Board) * n), "cudaMallocHost boards")) return false;
        if (!ck(cudaMallocHost(&h_outPinned,    sizeof(int)   * n), "cudaMallocHost out"))    return false;
        h_cap = n;
    }

    // Copy to pinned staging (CPU memcpy)
    std::memcpy(h_boardsPinned, boards, sizeof(Board) * n);

    // Async H2D
    if (!ck(cudaMemcpyAsync(d_boards, h_boardsPinned, sizeof(Board) * n, cudaMemcpyHostToDevice, g_stream),
            "cudaMemcpyAsync H2D boards")) return false;

    // Kernel (NNUE-only): reuse blended kernel with blendPermille=1000
    dim3 block(NNUE_HIDDEN_DIM, 1, 1);
    dim3 grid(n, 1, 1);
    nnueBatchBlendedKernel<<<grid, block, 0, g_stream>>>(d_boards, n, d_W1, d_B1, d_W2, d_B2, 1000, d_outCp);
    if (!ck(cudaGetLastError(), "kernel launch")) return false;

    // Async D2H
    if (!ck(cudaMemcpyAsync(h_outPinned, d_outCp, sizeof(int) * n, cudaMemcpyDeviceToHost, g_stream),
            "cudaMemcpyAsync D2H out")) return false;

    // Wait only for this stream (no global device sync)
    if (!ck(cudaStreamSynchronize(g_stream), "cudaStreamSynchronize")) return false;

    std::memcpy(outCp, h_outPinned, sizeof(int) * n);
    return true;
}

// Evaluate blended classic+NNUE on GPU, returns WHITE-perspective cp
bool nnueCudaEvaluateBatchBlendedWhite(const Board* boards, int n, int blendPermille, int* outCp) {
    if (!g_inited) return false;
    if (n <= 0) return true;

    // Resize device buffers
    if (n > cap) {
        if (d_boards) cudaFree(d_boards);
        if (d_outCp)  cudaFree(d_outCp);

        if (!ck(cudaMalloc(&d_boards, sizeof(Board) * n), "malloc d_boards")) return false;
        if (!ck(cudaMalloc(&d_outCp,  sizeof(int)   * n), "malloc d_outCp"))  return false;

        cap = n;
    }

    // Resize pinned host staging
    if (n > h_cap) {
        if (h_boardsPinned) cudaFreeHost(h_boardsPinned);
        if (h_outPinned)    cudaFreeHost(h_outPinned);

        if (!ck(cudaMallocHost(&h_boardsPinned, sizeof(Board) * n), "cudaMallocHost boards")) return false;
        if (!ck(cudaMallocHost(&h_outPinned,    sizeof(int)   * n), "cudaMallocHost out"))    return false;
        h_cap = n;
    }

    std::memcpy(h_boardsPinned, boards, sizeof(Board) * n);

    if (!ck(cudaMemcpyAsync(d_boards, h_boardsPinned, sizeof(Board) * n, cudaMemcpyHostToDevice, g_stream),
            "cudaMemcpyAsync H2D boards")) return false;

    dim3 block(NNUE_HIDDEN_DIM, 1, 1);
    dim3 grid(n, 1, 1);
    nnueBatchBlendedKernel<<<grid, block, 0, g_stream>>>(d_boards, n, d_W1, d_B1, d_W2, d_B2, blendPermille, d_outCp);
    if (!ck(cudaGetLastError(), "kernel launch")) return false;

    if (!ck(cudaMemcpyAsync(h_outPinned, d_outCp, sizeof(int) * n, cudaMemcpyDeviceToHost, g_stream),
            "cudaMemcpyAsync D2H out")) return false;

    if (!ck(cudaStreamSynchronize(g_stream), "cudaStreamSynchronize")) return false;

    std::memcpy(outCp, h_outPinned, sizeof(int) * n);
    return true;
}
