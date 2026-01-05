#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#include "typedefs.hpp"
#include "nnue_cuda.hpp"

// reuse your generated weights
#include "nnue_weights_generated.hpp"

// --- device buffers for weights ---
static float* d_W1 = nullptr;  // [128 * 781]
static float* d_B1 = nullptr;  // [128]
static float* d_W2 = nullptr;  // [128]
static float* d_B2 = nullptr;  // [1]

// device buffers for boards + outputs
static Board* d_boards = nullptr;
static int*   d_outCp  = nullptr;

// helper macro
static inline void ck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        printf("CUDA error: %s -> %s\n", msg, cudaGetErrorString(e));
    }
}

__device__ __forceinline__ int sqEngineToPy_dev(int sq_engine) {
    return sq_engine ^ 7;
}

__device__ __forceinline__ int ctz64_dev(unsigned long long x) {
    return __ffsll((long long)x) - 1; // ffs is 1-based
}

// Collect active indices into local array (max 64 is safe here)
__device__ int collectActive(const Board& b, int* act) {
    int cnt = 0;

    auto addPiecesPlane = [&](unsigned long long bb, int planeBase) {
        while (bb) {
            int sq_engine = ctz64_dev(bb);
            int sq_py = sqEngineToPy_dev(sq_engine);
            int idx = planeBase + sq_py;
            act[cnt++] = idx;
            bb &= (bb - 1);
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

    // STM
    if (b.turn == WHITE) act[cnt++] = 768;

    // castling bits (K,Q,k,q) => 769..772
    if (b.castling & 0b0001) act[cnt++] = 769;
    if (b.castling & 0b0010) act[cnt++] = 770;
    if (b.castling & 0b0100) act[cnt++] = 771;
    if (b.castling & 0b1000) act[cnt++] = 772;

    // ep file one-hot 773..780
    if (b.epSquare >= 0 && b.epSquare < 64) {
        int file_engine = (b.epSquare & 7); // 0=h..7=a
        int file_py = 7 - file_engine;      // 0=a..7=h
        act[cnt++] = 773 + file_py;
    }

    return cnt;
}

__global__ void nnueBatchKernel(const Board* boards, int n, const float* W1, const float* B1,
                                const float* W2, const float* B2, int* outCp) {
    int pos = blockIdx.x;
    if (pos >= n) return;

    // one block = one position, 128 threads = hidden neurons
    int i = threadIdx.x;
    __shared__ float hidden[NNUE_HIDDEN_DIM];

    // collect active indices (thread0)
    __shared__ int act[64];
    __shared__ int actN;
    if (i == 0) {
        actN = collectActive(boards[pos], act);
    }
    __syncthreads();

    // compute hidden neuron i
    if (i < NNUE_HIDDEN_DIM) {
        float sum = B1[i];
        // W1 is flattened: W1[i*IN + idx]
        int base = i * NNUE_INPUT_DIM;
        for (int k = 0; k < actN; ++k) {
            sum += W1[base + act[k]];
        }
        hidden[i] = (sum > 0.0f) ? sum : 0.0f;
    }
    __syncthreads();

    // output by thread0
    if (i == 0) {
        float out = B2[0];
        for (int j = 0; j < NNUE_HIDDEN_DIM; ++j) {
            out += W2[j] * hidden[j];
        }
        float cp = out * 400.0f;
        if (cp > 100000.0f) cp = 100000.0f;
        if (cp < -100000.0f) cp = -100000.0f;
        outCp[pos] = (int)llroundf(cp);
    }
}

bool nnueCudaInit() {
    printf("[CUDA] NNUE CUDA init OK\n");

    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) return false;

    // allocate weights
    ck(cudaMalloc(&d_W1, sizeof(float) * NNUE_HIDDEN_DIM * NNUE_INPUT_DIM), "malloc d_W1");
    ck(cudaMalloc(&d_B1, sizeof(float) * NNUE_HIDDEN_DIM), "malloc d_B1");
    ck(cudaMalloc(&d_W2, sizeof(float) * NNUE_HIDDEN_DIM), "malloc d_W2");
    ck(cudaMalloc(&d_B2, sizeof(float) * 1), "malloc d_B2");

    // copy weights from host static arrays
    ck(cudaMemcpy(d_W1, NNUE_W1, sizeof(float) * NNUE_HIDDEN_DIM * NNUE_INPUT_DIM, cudaMemcpyHostToDevice), "cpy W1");
    ck(cudaMemcpy(d_B1, NNUE_B1, sizeof(float) * NNUE_HIDDEN_DIM, cudaMemcpyHostToDevice), "cpy B1");
    ck(cudaMemcpy(d_W2, NNUE_W2, sizeof(float) * NNUE_HIDDEN_DIM, cudaMemcpyHostToDevice), "cpy W2");
    ck(cudaMemcpy(d_B2, &NNUE_B2, sizeof(float) * 1, cudaMemcpyHostToDevice), "cpy B2");

    return true;
}

void nnueCudaShutdown() {
    if (d_W1) cudaFree(d_W1);
    if (d_B1) cudaFree(d_B1);
    if (d_W2) cudaFree(d_W2);
    if (d_B2) cudaFree(d_B2);
    d_W1 = d_B1 = d_W2 = d_B2 = nullptr;

    if (d_boards) cudaFree(d_boards);
    if (d_outCp) cudaFree(d_outCp);
    d_boards = nullptr;
    d_outCp = nullptr;
}

bool nnueCudaEvaluateBatchWhite(const Board* boards, int n, int* outCp) {
    if (!d_W1) return false;

    // (re)alloc batch buffers if needed
    static int cap = 0;
    if (n > cap) {
        if (d_boards) cudaFree(d_boards);
        if (d_outCp) cudaFree(d_outCp);
        ck(cudaMalloc(&d_boards, sizeof(Board) * n), "malloc d_boards");
        ck(cudaMalloc(&d_outCp, sizeof(int) * n), "malloc d_outCp");
        cap = n;
    }

    ck(cudaMemcpy(d_boards, boards, sizeof(Board) * n, cudaMemcpyHostToDevice), "cpy boards H2D");

    dim3 block(128, 1, 1);
    dim3 grid(n, 1, 1);
    nnueBatchKernel<<<grid, block>>>(d_boards, n, d_W1, d_B1, d_W2, d_B2, d_outCp);
    if (cudaDeviceSynchronize() != cudaSuccess) return false;

    ck(cudaMemcpy(outCp, d_outCp, sizeof(int) * n, cudaMemcpyDeviceToHost), "cpy out D2H");
    return true;
}
