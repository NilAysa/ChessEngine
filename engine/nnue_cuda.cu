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

static inline bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("[CUDA] %s failed: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// --- device buffers for weights ---
static float* d_W1 = nullptr;  // [128 * 781]
static float* d_B1 = nullptr;  // [128]
static float* d_W2 = nullptr;  // [128]
static float* d_B2 = nullptr;  // [1]

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

__global__ void nnueBatchKernel(const Board* boards, int n,
                                const float* W1, const float* B1,
                                const float* W2, const float* B2,
                                int* outCp)
{
    int pos = (int)blockIdx.x;
    if (pos >= n) return;

    int i = (int)threadIdx.x; // 0..127

    __shared__ int   act[64];     // dovoljno za max aktivnih (tipično < 40)
    __shared__ int   actCount;
    __shared__ float hidden[NNUE_HIDDEN_DIM];

    if (i == 0) {
        actCount = buildActive(boards[pos], act);
    }
    __syncthreads();

    // Hidden neuron i: dot over sparse act
    float sum = B1[i];
    // NOTE: sparse -> divergent count, ali actCount je zajednički po bloku
    for (int k = 0; k < actCount; ++k) {
        int idx = act[k];
        sum += W1[i * NNUE_INPUT_DIM + idx];
    }
    hidden[i] = relu(sum);
    __syncthreads();

    if (i == 0) {
        float out = B2[0];
        // W2[j] * hidden[j]
        for (int j = 0; j < NNUE_HIDDEN_DIM; ++j) out += W2[j] * hidden[j];

        float cp = out * 400.0f;
        if (cp >  100000.0f) cp =  100000.0f;
        if (cp < -100000.0f) cp = -100000.0f;
        outCp[pos] = (int)llroundf(cp);
    }
}

bool nnueCudaInit() {
    if (g_inited) return true;

    int devCount = 0;
    if (!ck(cudaGetDeviceCount(&devCount), "cudaGetDeviceCount")) return false;
    if (devCount == 0) return false;

    // stream
    if (!ck(cudaStreamCreate(&g_stream), "cudaStreamCreate")) return false;

    // allocate weights on device
    if (!ck(cudaMalloc(&d_W1, sizeof(float) * NNUE_HIDDEN_DIM * NNUE_INPUT_DIM), "malloc d_W1")) return false;
    if (!ck(cudaMalloc(&d_B1, sizeof(float) * NNUE_HIDDEN_DIM), "malloc d_B1")) return false;
    if (!ck(cudaMalloc(&d_W2, sizeof(float) * NNUE_HIDDEN_DIM), "malloc d_W2")) return false;
    if (!ck(cudaMalloc(&d_B2, sizeof(float) * 1), "malloc d_B2")) return false;

    if (!ck(cudaMemcpy(d_W1, NNUE_W1, sizeof(float) * NNUE_HIDDEN_DIM * NNUE_INPUT_DIM, cudaMemcpyHostToDevice), "cpy W1")) return false;
    if (!ck(cudaMemcpy(d_B1, NNUE_B1, sizeof(float) * NNUE_HIDDEN_DIM, cudaMemcpyHostToDevice), "cpy B1")) return false;
    if (!ck(cudaMemcpy(d_W2, NNUE_W2, sizeof(float) * NNUE_HIDDEN_DIM, cudaMemcpyHostToDevice), "cpy W2")) return false;
    if (!ck(cudaMemcpy(d_B2, &NNUE_B2, sizeof(float), cudaMemcpyHostToDevice), "cpy B2")) return false;

    g_inited = true;
    std::printf("[CUDA] NNUE CUDA init OK (stream + persistent buffers)\n");
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

    // Kernel
    dim3 block(NNUE_HIDDEN_DIM, 1, 1);
    dim3 grid(n, 1, 1);
    nnueBatchKernel<<<grid, block, 0, g_stream>>>(d_boards, n, d_W1, d_B1, d_W2, d_B2, d_outCp);
    if (!ck(cudaGetLastError(), "kernel launch")) return false;

    // Async D2H
    if (!ck(cudaMemcpyAsync(h_outPinned, d_outCp, sizeof(int) * n, cudaMemcpyDeviceToHost, g_stream),
            "cudaMemcpyAsync D2H out")) return false;

    // Wait only for this stream (no global device sync)
    if (!ck(cudaStreamSynchronize(g_stream), "cudaStreamSynchronize")) return false;

    std::memcpy(outCp, h_outPinned, sizeof(int) * n);
    return true;
}
