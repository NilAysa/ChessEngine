// engine/nnue_cuda.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

#include "typedefs.hpp"
#include "nnue_cuda.hpp"
#include "nnue_weights_generated.hpp"

// ------------------------------------------------------------
// Device buffers (weights)
// W1T: [INPUT_DIM][HIDDEN_DIM]  (transponovano radi koalesiranja)
static float* d_W1T = nullptr; // 781 * 128
static float* d_B1  = nullptr; // 128
static float* d_W2  = nullptr; // 128
static float* d_B2  = nullptr; // 1

// device buffers (boards + outputs)
static Board* d_boards = nullptr;
static int*   d_outCp  = nullptr;
static int    capBoards = 0;

static inline bool ck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        printf("CUDA error: %s -> %s\n", msg, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// --- helpers ---
__device__ __forceinline__ int sqEngineToPy_dev(int sq_engine) { return sq_engine ^ 7; }
__device__ __forceinline__ int ctz64_dev(unsigned long long x) { return __ffsll((long long)x) - 1; }

// Collect active indices into local array (max 64)
__device__ int collectActive(const Board& b, int* act) {
    int cnt = 0;

    auto addPiecesPlane = [&](unsigned long long bb, int planeBase) {
        while (bb) {
            int sq_engine = ctz64_dev(bb);
            int sq_py = sqEngineToPy_dev(sq_engine);
            act[cnt++] = planeBase + sq_py;
            bb &= (bb - 1);
            if (cnt >= 64) return; // sigurnosno
        }
    };

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

    // side-to-move feature: tvoj CPU kod koristi 768..; ovdje zadržimo isto
    // (pretpostavka: WHITE=1, BLACK=0)
    act[cnt++] = 768 + (b.turn == WHITE ? 1 : 0);

    // castling (4 bita)
    // mapiraj na 769..772
    if (b.castling & 1) act[cnt++] = 769; // K
    if (b.castling & 2) act[cnt++] = 770; // Q
    if (b.castling & 4) act[cnt++] = 771; // k
    if (b.castling & 8) act[cnt++] = 772; // q

    // ep file: (tvoj CPU kod je imao 773..780)
    if (b.epSquare >= 0) {
        int file = (b.epSquare & 7);
        act[cnt++] = 773 + file;
    }

    return cnt;
}

// warp reduce (sum)
__device__ __forceinline__ float warpReduceSum(float v) {
    // standardni butterfly; warp je 32 niti :contentReference[oaicite:10]{index=10}
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// block reduce using warpReduce + shared for warp sums
__device__ float blockReduceSum(float v) {
    __shared__ float warpSums[4]; // 128 threads -> 4 warpa

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    v = warpReduceSum(v);
    if (lane == 0) warpSums[warp] = v;
    __syncthreads();

    // warp 0 final reduce
    float out = 0.0f;
    if (warp == 0) {
        out = (lane < 4) ? warpSums[lane] : 0.0f;
        out = warpReduceSum(out);
    }
    return out; // validno u svim nitima (ali smisleno u warp0)
}

__global__ void nnueBatchKernelT(const Board* __restrict__ boards,
                                 int n,
                                 const float* __restrict__ W1T,
                                 const float* __restrict__ B1,
                                 const float* __restrict__ W2,
                                 const float* __restrict__ B2,
                                 int* __restrict__ outCp) {
    int pos = blockIdx.x;
    if (pos >= n) return;

    __shared__ int   act[64];
    __shared__ int   actCnt;
    __shared__ float hidden[NNUE_HIDDEN_DIM];

    // 1) thread0 skuplja active feature-e
    if (threadIdx.x == 0) {
        actCnt = collectActive(boards[pos], act);
        if (actCnt > 64) actCnt = 64;
    }
    __syncthreads();

    // 2) svaki thread = jedan neuron j
    int j = threadIdx.x; // 0..127
    float sum = B1[j];

    // KOALESIRANO: za fiksni feature, niti (j) čitaju susjedne adrese W1T[feat*128 + j] :contentReference[oaicite:11]{index=11}
    for (int k = 0; k < actCnt; ++k) {
        int feat = act[k];
        sum += W1T[feat * NNUE_HIDDEN_DIM + j];
    }

    // ReLU
    sum = (sum > 0.0f) ? sum : 0.0f;
    hidden[j] = sum;
    __syncthreads();

    // 3) dot(W2, hidden) kao redukcija
    float partial = W2[j] * hidden[j];
    float dot = blockReduceSum(partial);

    // 4) thread0 upisuje output
    if (threadIdx.x == 0) {
        float out = B2[0] + dot;
        float cp = out * 400.0f;
        cp = fminf(100000.0f, fmaxf(-100000.0f, cp));
        outCp[pos] = (int)llroundf(cp);
    }
}

bool nnueCudaInit() {
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) return false;

    // --- Host transpose W1 -> W1T ---
    std::vector<float> h_W1T(NNUE_INPUT_DIM * NNUE_HIDDEN_DIM);
    for (int h = 0; h < NNUE_HIDDEN_DIM; ++h) {
        for (int in = 0; in < NNUE_INPUT_DIM; ++in) {
            h_W1T[in * NNUE_HIDDEN_DIM + h] = NNUE_W1[h][in];
        }
    }

    // allocate weights on device
    if (!ck(cudaMalloc(&d_W1T, sizeof(float) * NNUE_INPUT_DIM * NNUE_HIDDEN_DIM), "malloc d_W1T")) return false;
    if (!ck(cudaMalloc(&d_B1,  sizeof(float) * NNUE_HIDDEN_DIM), "malloc d_B1")) return false;
    if (!ck(cudaMalloc(&d_W2,  sizeof(float) * NNUE_HIDDEN_DIM), "malloc d_W2")) return false;
    if (!ck(cudaMalloc(&d_B2,  sizeof(float)), "malloc d_B2")) return false;

    if (!ck(cudaMemcpy(d_W1T, h_W1T.data(),
                       sizeof(float) * NNUE_INPUT_DIM * NNUE_HIDDEN_DIM,
                       cudaMemcpyHostToDevice), "cpy W1T")) return false;

    if (!ck(cudaMemcpy(d_B1, NNUE_B1, sizeof(float) * NNUE_HIDDEN_DIM, cudaMemcpyHostToDevice), "cpy B1")) return false;
    if (!ck(cudaMemcpy(d_W2, NNUE_W2, sizeof(float) * NNUE_HIDDEN_DIM, cudaMemcpyHostToDevice), "cpy W2")) return false;
    if (!ck(cudaMemcpy(d_B2, NNUE_B2, sizeof(float), cudaMemcpyHostToDevice), "cpy B2")) return false;

    printf("[CUDA] NNUE CUDA init OK (W1 transposed for coalescing)\n");
    return true;
}

void nnueCudaShutdown() {
    if (d_W1T) cudaFree(d_W1T);
    if (d_B1)  cudaFree(d_B1);
    if (d_W2)  cudaFree(d_W2);
    if (d_B2)  cudaFree(d_B2);
    d_W1T = d_B1 = d_W2 = d_B2 = nullptr;

    if (d_boards) cudaFree(d_boards);
    if (d_outCp)  cudaFree(d_outCp);
    d_boards = nullptr; d_outCp = nullptr;
    capBoards = 0;
}

bool nnueCudaEvaluateBatchWhite(const Board* boards, int n, int* outCp) {
    if (!boards || !outCp || n <= 0) return false;
    if (!d_W1T || !d_B1 || !d_W2 || !d_B2) return false;

    // resize device buffers if needed
    if (n > capBoards) {
        if (d_boards) cudaFree(d_boards);
        if (d_outCp)  cudaFree(d_outCp);
        d_boards = nullptr; d_outCp = nullptr;

        if (!ck(cudaMalloc(&d_boards, sizeof(Board) * n), "malloc d_boards")) return false;
        if (!ck(cudaMalloc(&d_outCp,  sizeof(int)   * n), "malloc d_outCp"))  return false;
        capBoards = n;
    }

    if (!ck(cudaMemcpy(d_boards, boards, sizeof(Board) * n, cudaMemcpyHostToDevice), "cpy boards H2D")) return false;

    dim3 block(NNUE_HIDDEN_DIM, 1, 1); // 128 threads
    dim3 grid(n, 1, 1);                // 1 block per board
    nnueBatchKernelT<<<grid, block>>>(d_boards, n, d_W1T, d_B1, d_W2, d_B2, d_outCp);

    if (cudaDeviceSynchronize() != cudaSuccess) return false;
    if (!ck(cudaMemcpy(outCp, d_outCp, sizeof(int) * n, cudaMemcpyDeviceToHost), "cpy out D2H")) return false;

    return true;
}
