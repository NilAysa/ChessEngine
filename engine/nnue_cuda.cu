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

// ------------------------------
// PackedBoard: manji payload za H2D (promjena #5)
// ------------------------------
struct PackedBoard {
    Bitboard pawn_W, knight_W, bishop_W, rook_W, queen_W, king_W;
    Bitboard pawn_B, knight_B, bishop_B, rook_B, queen_B, king_B;
    int turn;
    int castling;
    int epSquare;
    int whiteKingSq;
    int blackKingSq;
};

static inline PackedBoard packBoardHost(const Board& b) {
    PackedBoard p{};
    p.pawn_W = b.pawn_W;   p.knight_W = b.knight_W; p.bishop_W = b.bishop_W; p.rook_W = b.rook_W; p.queen_W = b.queen_W; p.king_W = b.king_W;
    p.pawn_B = b.pawn_B;   p.knight_B = b.knight_B; p.bishop_B = b.bishop_B; p.rook_B = b.rook_B; p.queen_B = b.queen_B; p.king_B = b.king_B;
    p.turn = b.turn;
    p.castling = b.castling;
    p.epSquare = b.epSquare;
    p.whiteKingSq = b.whiteKingSq;
    p.blackKingSq = b.blackKingSq;
    return p;
}

// ------------------------------
// Double-buffered device + pinned host buffers (promjena #4)
// ------------------------------
static PackedBoard* d_boards[2] = { nullptr, nullptr };
static int*         d_outCp[2]  = { nullptr, nullptr };
static int          cap[2]      = { 0, 0 };

static PackedBoard* h_boardsPinned[2] = { nullptr, nullptr };
static int*         h_outPinned[2]    = { nullptr, nullptr };
static int          h_cap[2]          = { 0, 0 };

static cudaStream_t g_stream[2] = { nullptr, nullptr };
static int g_rr = 0; // round-robin buffer index

static bool g_inited = false;

static __device__ __forceinline__ float relu(float x) { return x > 0.0f ? x : 0.0f; }
static __device__ __forceinline__ int mirrorSqDev(int sq) { return 63 - sq; }

// --- classic eval (device) ---
static __device__ __forceinline__ int evalClassicWhitePerspectiveDev(const PackedBoard& b) {
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

    // king PST special-case
    eval += c_KING_PST[b.whiteKingSq];
    eval -= c_KING_PST[mirrorSqDev(b.blackKingSq)];

    if (eval >  100000) eval =  100000;
    if (eval < -100000) eval = -100000;
    return eval;
}

// Build sparse active feature indices (same logic)
__device__ __forceinline__ int buildActive(const PackedBoard& b, int* act) {
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

    // en-passant file => 773..780
    if (b.epSquare >= 0 && b.epSquare < 64) {
        int file_engine = b.epSquare & 7;   // 0=h ... 7=a (kao na CPU)
        int file_py = 7 - file_engine;      // 0=a ... 7=h
        act[cnt++] = 773 + file_py;
    }


    return cnt;
}

__global__ void nnueBatchBlendedKernel(const PackedBoard* boards, int n,
                                       const float* W1, const float* B1,
                                       const float* W2, const float* B2,
                                       int blendPermille,
                                       int* outCp)
{
    int pos = (int)blockIdx.x;
    if (pos >= n) return;

    int i = (int)threadIdx.x; // 0..127

    __shared__ int   act[64];
    __shared__ int   actCount;
    __shared__ float hidden[NNUE_HIDDEN_DIM];
    __shared__ int   classicEval;

    if (i == 0) {
        const PackedBoard& b = boards[pos];
        classicEval = evalClassicWhitePerspectiveDev(b);
        actCount = buildActive(b, act);
    }
    __syncthreads();

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

        float nnCpF = out * 400.0f;
        if (nnCpF >  100000.0f) nnCpF =  100000.0f;
        if (nnCpF < -100000.0f) nnCpF = -100000.0f;
        int nnCp = (int)llroundf(nnCpF);

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

    // streams (double-buffer)
    if (!ck(cudaStreamCreate(&g_stream[0]), "cudaStreamCreate stream0")) return false;
    if (!ck(cudaStreamCreate(&g_stream[1]), "cudaStreamCreate stream1")) return false;

    // allocate NNUE weights on device
    if (!ck(cudaMalloc(&d_W1, sizeof(float) * NNUE_HIDDEN_DIM * NNUE_INPUT_DIM), "malloc d_W1")) return false;
    if (!ck(cudaMalloc(&d_B1, sizeof(float) * NNUE_HIDDEN_DIM), "malloc d_B1")) return false;
    if (!ck(cudaMalloc(&d_W2, sizeof(float) * NNUE_HIDDEN_DIM), "malloc d_W2")) return false;
    if (!ck(cudaMalloc(&d_B2, sizeof(float) * 1), "malloc d_B2")) return false;

    if (!ck(cudaMemcpy(d_W1, NNUE_W1, sizeof(float) * NNUE_HIDDEN_DIM * NNUE_INPUT_DIM, cudaMemcpyHostToDevice), "cpy W1")) return false;
    if (!ck(cudaMemcpy(d_B1, NNUE_B1, sizeof(float) * NNUE_HIDDEN_DIM, cudaMemcpyHostToDevice), "cpy B1")) return false;
    if (!ck(cudaMemcpy(d_W2, NNUE_W2, sizeof(float) * NNUE_HIDDEN_DIM, cudaMemcpyHostToDevice), "cpy W2")) return false;
    if (!ck(cudaMemcpy(d_B2, &NNUE_B2, sizeof(float), cudaMemcpyHostToDevice), "cpy B2")) return false;

    // upload PSTs to constant memory
    if (!ck(cudaMemcpyToSymbol(c_PAWN_PST,   PAWN_W_PST,   sizeof(int) * 64), "cpy PST pawn")) return false;
    if (!ck(cudaMemcpyToSymbol(c_KNIGHT_PST, KNIGHT_W_PST, sizeof(int) * 64), "cpy PST knight")) return false;
    if (!ck(cudaMemcpyToSymbol(c_BISHOP_PST, BISHOP_W_PST, sizeof(int) * 64), "cpy PST bishop")) return false;
    if (!ck(cudaMemcpyToSymbol(c_ROOK_PST,   ROOK_W_PST,   sizeof(int) * 64), "cpy PST rook")) return false;
    if (!ck(cudaMemcpyToSymbol(c_QUEEN_PST,  QUEEN_W_PST,  sizeof(int) * 64), "cpy PST queen")) return false;
    if (!ck(cudaMemcpyToSymbol(c_KING_PST,   KING_W_PST,   sizeof(int) * 64), "cpy PST king")) return false;

    g_inited = true;
    std::printf("[CUDA] NNUE CUDA init OK (2 streams + double buffers + PST const)\n");
    return true;
}

void nnueCudaShutdown() {
    if (d_W1) cudaFree(d_W1), d_W1 = nullptr;
    if (d_B1) cudaFree(d_B1), d_B1 = nullptr;
    if (d_W2) cudaFree(d_W2), d_W2 = nullptr;
    if (d_B2) cudaFree(d_B2), d_B2 = nullptr;

    for (int i = 0; i < 2; ++i) {
        if (d_boards[i]) cudaFree(d_boards[i]), d_boards[i] = nullptr;
        if (d_outCp[i])  cudaFree(d_outCp[i]),  d_outCp[i]  = nullptr;

        if (h_boardsPinned[i]) cudaFreeHost(h_boardsPinned[i]), h_boardsPinned[i] = nullptr;
        if (h_outPinned[i])    cudaFreeHost(h_outPinned[i]),    h_outPinned[i]    = nullptr;

        cap[i] = 0;
        h_cap[i] = 0;

        if (g_stream[i]) cudaStreamDestroy(g_stream[i]), g_stream[i] = nullptr;
    }

    g_inited = false;
}

// ------------------------------
// Helpers
// ------------------------------
static bool ensureCapacity(int idx, int n) {
    if (n > cap[idx]) {
        if (d_boards[idx]) cudaFree(d_boards[idx]), d_boards[idx] = nullptr;
        if (d_outCp[idx])  cudaFree(d_outCp[idx]),  d_outCp[idx]  = nullptr;

        if (!ck(cudaMalloc(&d_boards[idx], sizeof(PackedBoard) * n), "malloc d_boards")) return false;
        if (!ck(cudaMalloc(&d_outCp[idx],  sizeof(int)        * n), "malloc d_outCp"))  return false;
        cap[idx] = n;
    }

    if (n > h_cap[idx]) {
        if (h_boardsPinned[idx]) cudaFreeHost(h_boardsPinned[idx]), h_boardsPinned[idx] = nullptr;
        if (h_outPinned[idx])    cudaFreeHost(h_outPinned[idx]),    h_outPinned[idx]    = nullptr;

        if (!ck(cudaMallocHost(&h_boardsPinned[idx], sizeof(PackedBoard) * n), "cudaMallocHost boards")) return false;
        if (!ck(cudaMallocHost(&h_outPinned[idx],    sizeof(int)        * n), "cudaMallocHost out"))    return false;
        h_cap[idx] = n;
    }

    return true;
}

// ------------------------------
// Async API (submit/collect)
// ------------------------------
bool nnueCudaSubmitBatchBlendedWhite(const Board* boards, int n, int blendPermille, CudaBatchHandle& outHandle) {
    if (!g_inited) return false;
    if (n <= 0) {
        outHandle.n = 0;
        outHandle.bufferIndex = 0;
        outHandle.done = nullptr;
        return true;
    }

    const int idx = (g_rr++ & 1);
    if (!ensureCapacity(idx, n)) return false;

    // Pack -> pinned (manji payload)
    for (int i = 0; i < n; ++i) {
        h_boardsPinned[idx][i] = packBoardHost(boards[i]);
    }

    cudaStream_t s = g_stream[idx];

    if (!ck(cudaMemcpyAsync(d_boards[idx], h_boardsPinned[idx], sizeof(PackedBoard) * n, cudaMemcpyHostToDevice, s),
            "cudaMemcpyAsync H2D boards")) return false;

    dim3 block(NNUE_HIDDEN_DIM, 1, 1);
    dim3 grid(n, 1, 1);
    nnueBatchBlendedKernel<<<grid, block, 0, s>>>(d_boards[idx], n, d_W1, d_B1, d_W2, d_B2, blendPermille, d_outCp[idx]);
    if (!ck(cudaGetLastError(), "kernel launch")) return false;

    if (!ck(cudaMemcpyAsync(h_outPinned[idx], d_outCp[idx], sizeof(int) * n, cudaMemcpyDeviceToHost, s),
            "cudaMemcpyAsync D2H out")) return false;

    cudaEvent_t ev;
    if (!ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate")) return false;
    if (!ck(cudaEventRecord(ev, s), "cudaEventRecord")) return false;

    outHandle.done = ev;
    outHandle.n = n;
    outHandle.bufferIndex = idx;
    return true;
}

bool nnueCudaCollectBatch(CudaBatchHandle& handle, int* outCp) {
    if (!g_inited) return false;
    if (handle.n <= 0) return true;

    const int idx = handle.bufferIndex & 1;
    if (!handle.done) return false;

    if (!ck(cudaEventSynchronize(handle.done), "cudaEventSynchronize")) return false;
    cudaEventDestroy(handle.done);
    handle.done = nullptr;

    std::memcpy(outCp, h_outPinned[idx], sizeof(int) * handle.n);
    return true;
}

// ------------------------------
// Backwards compatible synchronous wrappers
// ------------------------------
bool nnueCudaEvaluateBatchBlendedWhite(const Board* boards, int n, int blendPermille, int* outCp) {
    CudaBatchHandle h{};
    if (!nnueCudaSubmitBatchBlendedWhite(boards, n, blendPermille, h)) return false;
    return nnueCudaCollectBatch(h, outCp);
}

bool nnueCudaEvaluateBatchWhite(const Board* boards, int n, int* outCp) {
    return nnueCudaEvaluateBatchBlendedWhite(boards, n, 1000, outCp);
}
